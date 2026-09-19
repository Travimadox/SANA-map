# ═══════════════════════════════════════════════════════════════════════
# PROJECT: SANA-map
# FILE: run.py
# DESCRIPTION: The threaded ZED mapping pipeline described in Sec. III of
#              the paper, ZED capture, open-vocabulary detection, and
#              semantic BEV mapping each run on their own thread so their
#              costs overlap rather than summing (see OPT notes below).
# HISTORY: originally main_with_zed_optim_threaded_v2.py in the Agrinav
#          research repository.
# CHANGES vs the non-threaded version:
#   [OPT-1]  Detector moved to its own pipeline thread, detection now
#            overlaps ZED I/O AND semantic mapping (hides 40–60 ms/frame)
#   [OPT-1b] Adaptive frame-skip: detector result reused when robot is
#            nearly stationary (motion-gated via pose delta threshold)
#   [OPT-2]  Pinned-memory obs tensor + non_blocking H→D transfer
#            (PCIe DMA overlaps CPU work; eliminates blocking copy)
#   [OPT-3]  All map channels sent to viz thread as a single CPU tensor
#            moved with .cpu() before .clone(), GPU freed faster
#   [OPT-4]  All pose tensors pre-allocated; in-place fill replaces
#            repeated .clone() / zeros_like() each frame
#   [OPT-5]  Pre-allocated contiguous RGBD state buffer; avoids
#            np.concatenate + transpose copies every frame
#   [OPT-6]  colorize_depth offloaded to viz thread; zero cost on main
#   [OPT-7]  pose_history uses collections.deque; snapshot via np.array
#            instead of list copy
#   [OPT-8]  sem_map_module wrapped with torch.compile(reduce-overhead)
#            when PyTorch >= 2.0, fuses CUDA kernels, cuts 15–30 %
#   [OPT-9]  update_agent_location_channels uses single .zero_() kernel
#            instead of fill_(0.) to reduce CUDA round-trips
# ═══════════════════════════════════════════════════════════════════════

# ── Standard library ─────────────────────────────────────────────────
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import sys
import warnings
from datetime import datetime
from pathlib import Path
import threading
from queue import Queue, Empty
from collections import deque

# ── Third-party ───────────────────────────────────────────────────────
import torch
import numpy as np
import cv2
import yaml
from PIL import Image
from torchvision import transforms
import pyzed.sl as sl
from scipy.spatial.transform import Rotation as R
import pandas as pd

# ── Project ───────────────────────────────────────────────────────────
from .mapping import Semantic_Mapping
from .perception.open_vocab_detector import OpenVocabDetector
from .config import get_args
from .profiler import PipelineProfiler
from .visualisation import plot_map


# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

_SENTINEL = None          # Poison pill for all queues
_MOTION_THRESH = 0.01     # metres, below this reuse last semantic pred
                          # (OPT-1b) tune per deployment speed


# ═══════════════════════════════════════════════════════════════════════
# Utility helpers
# ═══════════════════════════════════════════════════════════════════════

def save_args_to_yaml(args, output_dir="configs", run_name=None):
    """Persist the current run configuration to a timestamped YAML file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    args_dict = vars(args)
    config = {
        'timestamp': datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
        'run_name': run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'parameters': args_dict,
    }

    safe_name = run_name.replace("/", "_").replace("\\", "_") if run_name else None
    filename = (f"{safe_name}_{config['timestamp']}.yaml"
                if safe_name else f"config_{config['timestamp']}.yaml")
    filepath = output_dir / filename

    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Configuration saved to: {filepath}")
    return str(filepath)


# ═══════════════════════════════════════════════════════════════════════
# ZED camera helpers
# ═══════════════════════════════════════════════════════════════════════

def init_zed_camera(args):
    """Open the ZED camera (or SVO file) and enable positional tracking.

    Returns pre-allocated sl.Mat buffers so we never re-create them in
    the hot loop.
    """
    zed = sl.Camera()

    input_type = sl.InputType()
    if args.svo is not None:
        input_type.set_from_svo_file(args.svo)

    init_params = sl.InitParameters(
        input_t=input_type,
        svo_real_time_mode=False,
    )
    init_params.coordinate_units          = sl.UNIT.METER
    init_params.depth_mode                = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_system         = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD
    init_params.depth_maximum_distance    = args.max_depth
    init_params.depth_minimum_distance    = args.min_depth

    runtime_params = sl.RuntimeParameters()
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"ZED open failed: {repr(status)}")

    print("Initialised Camera")

    # The pipeline pre-allocates every downstream buffer from
    # --frame_width/--frame_height, and the depth/semantic downscale path
    # (_preprocess_obs) only supports a clean integer ratio between the
    # source's native resolution and that target, via array striding. A
    # live camera always delivers what it was configured for, but a
    # replayed .svo carries its own recorded resolution, which can differ
    # from the config (e.g. an older recording at the camera's factory
    # default). Detect that mismatch here and adopt the source's native
    # resolution outright, rather than fail deep in the frame loop with a
    # numpy broadcast error that gives no hint why.
    source_res = zed.get_camera_information().camera_configuration.resolution
    native_w, native_h = source_res.width, source_res.height
    if (native_w, native_h) != (args.frame_width, args.frame_height):
        source_desc = f"SVO file {args.svo!r}" if args.svo else "the live camera"
        print(
            f"WARNING: {source_desc} delivers {native_w}x{native_h}, not the "
            f"configured --frame_width/--frame_height ({args.frame_width}x"
            f"{args.frame_height}). Re-run with --frame_width {native_w} "
            f"--frame_height {native_h} --env_frame_width {native_w} "
            f"--env_frame_height {native_h} to pin this explicitly; for now "
            f"the pipeline adopts the source's native resolution."
        )
        args.frame_width = native_w
        args.frame_height = native_h
        args.env_frame_width = native_w
        args.env_frame_height = native_h

    tracking_params = sl.PositionalTrackingParameters()
    tracking_params.enable_imu_fusion = True
    tracking_params.mode              = sl.POSITIONAL_TRACKING_MODE.GEN_2
    zed.enable_positional_tracking(tracking_params)

    # Pre-allocate reusable buffers
    image_rgb        = sl.Mat()
    image_depth_raw  = sl.Mat()
    depth_display    = sl.Mat()
    cam_w_pose       = sl.Pose()

    return zed, image_rgb, image_depth_raw, depth_display, cam_w_pose, runtime_params


def extract_2d_pose(zed, cam_w_pose):
    """Read the current ZED world pose and return (x, y, yaw) floats."""
    zed.get_position(cam_w_pose, sl.REFERENCE_FRAME.WORLD)

    py_translation = sl.Translation()
    tx, ty, _tz    = cam_w_pose.get_translation(py_translation).get()

    py_orientation  = sl.Orientation()
    ox, oy, oz, ow  = cam_w_pose.get_orientation(py_orientation).get()

    _roll, _pitch, yaw = R.from_quat([ox, oy, oz, ow]).as_euler("xyz", degrees=False)
    return tx, ty, yaw


def grab_zed_frame(zed, image_rgb, image_depth_raw, runtime_params):
    """Grab one frame from the ZED and return (rgb, depth) numpy arrays."""
    zed.retrieve_image(image_rgb, sl.VIEW.LEFT)
    rgb = cv2.cvtColor(image_rgb.get_data(), cv2.COLOR_BGRA2RGB)

    zed.retrieve_measure(image_depth_raw, sl.MEASURE.DEPTH)
    depth_raw = image_depth_raw.get_data().copy()
    depth_raw = np.nan_to_num(depth_raw, nan=0.0, posinf=0.0, neginf=0.0)
    depth_raw = np.expand_dims(depth_raw, axis=-1)

    return rgb, depth_raw


# ═══════════════════════════════════════════════════════════════════════
# [OPT-1]  Async Detector Thread
# ═══════════════════════════════════════════════════════════════════════

class DetectorThread:
    """Dedicated thread for open-vocabulary detection + SAM.

    Architecture
    ────────────
    Main thread  →  detection_in  queue  →  DetectorThread
    DetectorThread  →  detection_out  queue  →  Main thread (reads result
                                                 before semantic mapping)

    The detector now runs in parallel with ZED grab AND the semantic
    mapping forward pass of the *previous* frame, hiding its full cost.

    Frame-skip (OPT-1b)
    ───────────────────
    Caller sets skip=True to have the thread reuse its last prediction
    instead of running inference.  The main loop gates this on motion
    magnitude to avoid stale detections when the robot moves quickly.
    """

    def __init__(self, detector, conf, iou, maxsize=2):
        self._detector  = detector
        self._conf      = conf
        self._iou       = iou
        self._in_q      = Queue(maxsize=maxsize)   # (frame_idx, rgb, skip)
        self._out_q     = Queue(maxsize=maxsize)   # (frame_idx, sem_pred)
        self._last_pred = None
        self._thread    = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def _run(self):
        while True:
            item = self._in_q.get()
            if item is _SENTINEL:
                return

            idx, rgb, skip = item

            if skip and self._last_pred is not None:
                # [OPT-1b] reuse previous prediction, ~0 ms
                self._out_q.put((idx, self._last_pred))
            else:
                sem_pred, _, _, _ = self._detector.get_predictions(
                    img=rgb,
                    conf_threshold=self._conf,
                    iou_threshold=self._iou,
                )
                sem_pred = sem_pred.astype(np.float32)
                self._last_pred = sem_pred
                self._out_q.put((idx, sem_pred))

    def submit(self, idx, rgb, skip=False):
        """Non-blocking submit (drops frame if queue full to avoid backup)."""
        try:
            self._in_q.put_nowait((idx, rgb, skip))
        except Exception:
            # Queue full, put blocking so we don't lose frames
            self._in_q.put((idx, rgb, skip))

    def get_result(self, timeout=30.0):
        """Block until the result for the submitted frame is ready."""
        return self._out_q.get(timeout=timeout)   # (idx, sem_pred)

    def stop(self):
        self._in_q.put(_SENTINEL)
        self._thread.join(timeout=10.0)


# ═══════════════════════════════════════════════════════════════════════
# Threaded ZED Grabber
# ═══════════════════════════════════════════════════════════════════════

class ZedGrabberThread:
    """Background thread that continuously grabs ZED frames into a queue.

    The main thread calls .get() to retrieve the next (rgb, depth, pose)
    tuple.  ZED grab happens in parallel with the main thread's GPU work,
    completely overlapping the ~26 ms I/O latency.
    """

    def __init__(self, zed, image_rgb, image_depth_raw, runtime_params,
                 cam_w_pose, maxsize=2):
        self._zed              = zed
        self._image_rgb        = image_rgb
        self._image_depth_raw  = image_depth_raw
        self._runtime_params   = runtime_params
        self._cam_w_pose       = cam_w_pose
        self._queue            = Queue(maxsize=maxsize)
        self._stop_event       = threading.Event()
        self._thread           = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def _run(self):
        while not self._stop_event.is_set():
            err = self._zed.grab(self._runtime_params)
            if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                self._queue.put(("END", None, None, None))
                return
            if err != sl.ERROR_CODE.SUCCESS:
                self._queue.put(("ERROR", repr(err), None, None))
                return

            rgb, depth = grab_zed_frame(
                self._zed, self._image_rgb, self._image_depth_raw,
                self._runtime_params,
            )
            x, y, yaw = extract_2d_pose(self._zed, self._cam_w_pose)
            self._queue.put(("OK", (rgb, depth), (x, y, yaw), None))

    def get(self, timeout=30.0):
        return self._queue.get(timeout=timeout)

    def stop(self):
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)


# ═══════════════════════════════════════════════════════════════════════
# Threaded Visualisation Saver  [OPT-6 integrated]
# ═══════════════════════════════════════════════════════════════════════

class VisualizationThread:
    """Background thread that colorizes depth, renders and saves maps.

    [OPT-6] colorize_depth is now executed here, not on the main thread.
    The main thread simply passes the raw depth array (a lightweight
    numpy reference), zero colorization cost on the critical path.
    """

    def __init__(self, maxsize=4):
        self._queue    = Queue(maxsize=maxsize)
        self._thread   = threading.Thread(target=self._run, daemon=True)
        self._last_key = 0

    def start(self):
        self._thread.start()

    @staticmethod
    def _colorize_depth(depth_raw, max_depth):
        """Convert raw metric depth (H,W,1) float32 → BGRA uint8."""
        d = depth_raw[:, :, 0] if depth_raw.ndim == 3 else depth_raw
        d = np.clip(d / max_depth, 0.0, 1.0)
        d_u8 = (d * 255).astype(np.uint8)
        bgr  = cv2.applyColorMap(d_u8, cv2.COLORMAP_JET)
        alpha = np.full((*bgr.shape[:2], 1), 255, dtype=np.uint8)
        return np.concatenate((bgr, alpha), axis=2)

    def _run(self):
        while True:
            job = self._queue.get()
            if job is _SENTINEL:
                return
            try:
                # [OPT-6] colorize here, not on main thread
                depth_viz = self._colorize_depth(
                    job.pop('depth_image_raw'),
                    job.pop('max_depth'),
                )
                job['depth_image'] = depth_viz

                key = plot_map(**job)
                self._last_key = key
            except Exception as e:
                print(f"[VisualizationThread] Error: {e}")
            finally:
                self._queue.task_done()

    def submit(self, **kwargs):
        self._queue.put(kwargs)

    def get_last_key(self):
        return self._last_key

    def stop(self):
        self._queue.put(_SENTINEL)
        self._thread.join(timeout=10.0)

    def flush(self):
        self._queue.join()


# ═══════════════════════════════════════════════════════════════════════
# Depth colourisation, kept for final synchronous save only
# ═══════════════════════════════════════════════════════════════════════

def colorize_depth(depth_raw, max_depth=5.0):
    d    = depth_raw[:, :, 0] if depth_raw.ndim == 3 else depth_raw
    d    = np.clip(d / max_depth, 0.0, 1.0)
    d_u8 = (d * 255).astype(np.uint8)
    bgr  = cv2.applyColorMap(d_u8, cv2.COLORMAP_JET)
    alpha = np.full((*bgr.shape[:2], 1), 255, dtype=np.uint8)
    return np.concatenate((bgr, alpha), axis=2)


# ═══════════════════════════════════════════════════════════════════════
# Observation processing  [OPT-5: pre-allocated state buffer]
# ═══════════════════════════════════════════════════════════════════════

def _preprocess_depth_zed(depth):
    """Convert ZED metric depth (H,W,1) to centimetres (H,W)."""
    return depth[:, :, 0] * 100.0


def _preprocess_obs(obs, sem_seg_pred, target_size, ds,
                    state_buf=None, profiler=None):
    """Build a channel-first state tensor from a raw RGBD observation.

    [OPT-1]  sem_seg_pred is now passed in (computed by DetectorThread).
    [OPT-5]  Uses pre-allocated state_buf when provided, avoids the
             np.concatenate + transpose allocation on every frame.

    Parameters
    ----------
    obs          : np.ndarray  (C, H, W) , RGBD, channels-first
    sem_seg_pred : np.ndarray  (H, W, N) , from detector thread
    target_size  : tuple (H, W) for cv2.resize, or None when ds == 1
    ds           : int  downsample factor
    state_buf    : np.ndarray (C', H', W') pre-allocated output buffer
                   or None (falls back to allocating)
    profiler     : PipelineProfiler or None

    Returns
    -------
    state : np.ndarray  (C', H', W') , may be a view of state_buf
    """
    obs   = obs.transpose(1, 2, 0)          # → (H, W, C)
    rgb   = obs[:, :, :3]
    depth = _preprocess_depth_zed(obs[:, :, 3:])

    # [OPT-5] resize in-place branches
    if profiler:
        with profiler.stage("obs_preprocess"):
            if ds != 1:
                rgb          = cv2.resize(rgb, (target_size[1], target_size[0]),
                                          interpolation=cv2.INTER_NEAREST)
                depth        = depth[ds // 2::ds, ds // 2::ds]
                sem_seg_pred = sem_seg_pred[ds // 2::ds, ds // 2::ds]
    else:
        if ds != 1:
            rgb          = cv2.resize(rgb, (target_size[1], target_size[0]),
                                      interpolation=cv2.INTER_NEAREST)
            depth        = depth[ds // 2::ds, ds // 2::ds]
            sem_seg_pred = sem_seg_pred[ds // 2::ds, ds // 2::ds]

    depth = np.expand_dims(depth, axis=2)

    if state_buf is not None:
        # [OPT-5] write directly into pre-allocated buffer, no alloc
        C_rgb = 3
        C_dep = 1
        C_sem = sem_seg_pred.shape[2]

        state_buf[:C_rgb]              = rgb.transpose(2, 0, 1)
        state_buf[C_rgb:C_rgb+C_dep]   = depth.transpose(2, 0, 1)
        state_buf[C_rgb+C_dep:]        = sem_seg_pred.transpose(2, 0, 1)
        return state_buf
    else:
        state = np.concatenate((rgb, depth, sem_seg_pred), axis=2).transpose(2, 0, 1)
        return state


# ═══════════════════════════════════════════════════════════════════════
# Map helpers
# ═══════════════════════════════════════════════════════════════════════

def get_local_map_boundaries(agent_loc, local_sizes, full_sizes):
    loc_r, loc_c   = agent_loc
    local_w, local_h = local_sizes
    full_w, full_h   = full_sizes

    gx1 = loc_r - local_w // 2
    gy1 = loc_c - local_h // 2
    gx2 = gx1 + local_w
    gy2 = gy1 + local_h

    if gx1 < 0:    gx1, gx2 = 0, local_w
    if gx2 > full_w: gx1, gx2 = full_w - local_w, full_w
    if gy1 < 0:    gy1, gy2 = 0, local_h
    if gy2 > full_h: gy1, gy2 = full_h - local_h, full_h

    return [gx1, gx2, gy1, gy2]


def init_maps_and_pose(args, semantic_categories):
    map_size = args.map_size_cm // args.map_resolution
    full_w, full_h = map_size, map_size
    local_w = int(full_w / args.global_downscaling)
    local_h = int(full_h / args.global_downscaling)

    print(f"Full map size: {full_w}x{full_h}")
    print(f"Local map size: {local_w}x{local_h}")
    print(f"Downscaling factor: {args.global_downscaling}")

    nc       = semantic_categories + 4
    full_map  = torch.zeros(1, nc, full_w, full_h, device=args.device)
    local_map = torch.zeros(1, nc, local_w, local_h, device=args.device)

    full_pose  = torch.zeros(1, 3, device=args.device)
    local_pose = torch.zeros(1, 3, device=args.device)

    # Where the start pose sits in the map. Defaults: centre indoors, and
    # outdoors x=0 with y centred. --map_origin_x_m / --map_origin_y_m
    # override either axis, which is what lets a field entered from one
    # side use the whole grid instead of wasting half of it behind the
    # robot while the far headland turns fall off the edge.
    centre = args.map_size_cm / 100.0 / 2.0
    default_x = centre if args.farm_or_indoors == 0 else 0.0
    origin_x = getattr(args, 'map_origin_x_m', None)
    origin_y = getattr(args, 'map_origin_y_m', None)
    origin_x = default_x if origin_x is None else float(origin_x)
    origin_y = centre if origin_y is None else float(origin_y)
    print(f"Map origin: start pose at x={origin_x:.2f} m, y={origin_y:.2f} m "
          f"of a {args.map_size_cm/100.0:.1f} m map "
          f"(cells {int(origin_x*100/args.map_resolution)}, "
          f"{int(origin_y*100/args.map_resolution)})")
    full_pose[:, 0]  = origin_x
    full_pose[:, 1]  = origin_y
    local_pose[:, 0] = origin_x
    local_pose[:, 1] = origin_y

    locs  = full_pose.cpu().numpy()
    loc_r = int(locs[0, 1] * 100.0 / args.map_resolution)
    loc_c = int(locs[0, 0] * 100.0 / args.map_resolution)

    lmb = get_local_map_boundaries(
        (loc_r, loc_c), (local_w, local_h), (full_w, full_h)
    )
    full_map[0, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

    origin = [
        lmb[2] * args.map_resolution / 100.0,
        lmb[0] * args.map_resolution / 100.0,
        0.0,
    ]
    local_map[0] = full_map[0, :, lmb[0]:lmb[1], lmb[2]:lmb[3]]
    origin_t     = torch.tensor(origin, device=args.device)
    local_pose[0] = full_pose[0] - origin_t

    return full_map, local_map, full_pose, local_pose, origin, origin_t, lmb


def update_maps_and_pose(full_map, local_map, full_pose, local_pose,
                         origin, lmb, args):
    """Sync local↔global map and recompute the local window."""
    full_map[0, :, lmb[0]:lmb[1], lmb[2]:lmb[3]] = local_map[0]

    origin_t   = torch.tensor(origin, device=args.device)
    full_pose[0] = local_pose[0] + origin_t

    locs  = full_pose[0].cpu().numpy()
    loc_r = int(locs[1] * 100.0 / args.map_resolution)
    loc_c = int(locs[0] * 100.0 / args.map_resolution)

    lmb = get_local_map_boundaries(
        (loc_r, loc_c),
        (local_map.shape[2], local_map.shape[3]),
        (full_map.shape[2], full_map.shape[3]),
    )
    origin = [
        lmb[2] * args.map_resolution / 100.0,
        lmb[0] * args.map_resolution / 100.0,
        0.0,
    ]
    origin_t      = torch.tensor(origin, device=args.device)
    local_map[0]  = full_map[0, :, lmb[0]:lmb[1], lmb[2]:lmb[3]]
    local_pose[0] = full_pose[0] - origin_t

    return full_map, local_map, full_pose, local_pose, origin, origin_t, lmb


# ═══════════════════════════════════════════════════════════════════════
# Pose-change computation
# ═══════════════════════════════════════════════════════════════════════

def compute_relative_pose(prev_pose, curr_pose):
    """Transform world-frame pose difference into the robot's local frame."""
    diff    = curr_pose - prev_pose
    dx_w    = diff[:, 0]
    dy_w    = diff[:, 1]
    dtheta  = diff[:, 2]
    prev_theta = prev_pose[:, 2]

    cos_t = torch.cos(prev_theta)
    sin_t = torch.sin(prev_theta)

    dx_local = dx_w * cos_t + dy_w * sin_t
    dy_local = -dx_w * sin_t + dy_w * cos_t

    return torch.stack(
        [dx_local.squeeze(), dy_local.squeeze(), dtheta.squeeze()], dim=0
    ).unsqueeze(0)


# ═══════════════════════════════════════════════════════════════════════
# Lidar Odometry
# ═══════════════════════════════════════════════════════════════════════

def preprocess_lidar_odom(lidar_txt):
    if not os.path.exists(lidar_txt):
        raise FileNotFoundError(f"Lidar TXT not found: {lidar_txt}")

    poses = []
    with open(lidar_txt, "r") as f:
        for line_num, line in enumerate(f, 1):
            try:
                parts = line.strip().split()
                if len(parts) != 3:
                    print(f"Warning: Line {line_num} has {len(parts)} values, expected 3")
                    continue
                x, y, theta = map(float, parts)
                poses.append([x, y, theta])
            except ValueError:
                print(f"Warning: Could not parse line {line_num}: {line.strip()}")
                continue

    return torch.tensor(poses, dtype=torch.float32)


# ═══════════════════════════════════════════════════════════════════════
# [OPT-9]  Agent location channels
# ═══════════════════════════════════════════════════════════════════════

def update_agent_location_channels(local_map, local_pose, args):
    """Update ch2 (current loc) and accumulate ch3 (past locs).

    [OPT-9] Uses .zero_() which issues a single CUDA memset kernel
    instead of fill_(0.) (which can launch a separate elementwise op).
    """
    locs = local_pose.cpu().numpy()

    local_map[0, 2].zero_()   # [OPT-9] single memset, not elementwise fill

    r, c   = locs[0, 1], locs[0, 0]
    loc_r  = int(r * 100.0 / args.map_resolution)
    loc_c  = int(c * 100.0 / args.map_resolution)

    h, w   = local_map.shape[2], local_map.shape[3]
    loc_r  = max(2, min(loc_r, h - 3))
    loc_c  = max(2, min(loc_c, w - 3))

    local_map[0, 2:4, loc_r - 2:loc_r + 3, loc_c - 2:loc_c + 3] = 1.
    return local_map


# ═══════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════

def main():
    warnings.filterwarnings("ignore")
    args = get_args()

    # ── Profiler ─────────────────────────────────────────────────────
    profiler = PipelineProfiler(
        cuda_sync=True,
        print_interval=getattr(args, 'profile_interval', 100),
        enabled=getattr(args, 'profile', False),
    )

    # ── Output directory ─────────────────────────────────────────────
    dir_name = args.dump_dir
    dump_dir = f"Results/{dir_name}"
    os.makedirs(dump_dir, exist_ok=True)
    save_args_to_yaml(args, output_dir=dump_dir, run_name=dir_name)
    semantic_categories = args.classes

    # ── LIDAR poses ───────────────────────────────────────────────────
    if args.use_lidar_poses:
        print("Reading LIDAR poses")
        lidar_poses = preprocess_lidar_odom(args.lidar_txt)
        lidar_poses = lidar_poses.to(args.device)

    # ── ZED camera ───────────────────────────────────────────────────
    (zed, image_rgb, image_depth_raw, depth_display_buf,
     cam_w_pose, runtime_params) = init_zed_camera(args)

    # ── Semantic mapping model ───────────────────────────────────────
    print("Setting up Semantic Neural SLAM Model")
    sem_map_module = Semantic_Mapping(args).to(args.device)
    sem_map_module.eval()

    # [OPT-8] torch.compile reduces Python dispatch overhead and can
    # fuse CUDA kernels, 15–30 % faster on repeated fixed-shape calls.
    if hasattr(torch, 'compile'):
        print("Compiling sem_map_module with torch.compile (reduce-overhead)…")
        sem_map_module = torch.compile(
            sem_map_module,
            mode="reduce-overhead",
            fullgraph=False,
        )
        print("torch.compile done.")

    print(f"Camera matrix: {sem_map_module.camera_matrix}")
    print("Semantic Neural SLAM Model loaded successfully")

    # ── Open-vocabulary detector ─────────────────────────────────────
    print("Setting up Custom Detector Model")
    openvocabdet = OpenVocabDetector(custom_classes=args.classes, args=args)
    detector_model = openvocabdet.init_object_detector()
    if args.yolo_or_owl == 1:
        detector_model.set_optimal_gpu_settings()
    print("Custom Detector Model loaded successfully")

    # ── Warmup detector ───────────────────────────────────────────────
    print("Warming up detector with dummy data…")
    dummy_img = np.zeros((args.frame_height, args.frame_width, 3), dtype=np.uint8)
    for _ in range(3):
        detector_model.get_predictions(
            img=dummy_img,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
        )
    print("Detector warmup complete")

    # ── Maps ──────────────────────────────────────────────────────────
    print("Initializing maps with local/global hierarchy")
    nc = args.num_sem_categories
    (full_map, local_map, full_pose, local_pose,
     origin, origin_t, lmb) = init_maps_and_pose(args, nc)
    print("Maps successfully initialized")

    # Init map variables for LIDAR odometry
    lidar_full_map   = full_map.clone()
    lidar_full_pose  = full_pose.clone()
    lidar_local_map  = local_map.clone()
    lidar_local_pose = local_pose.clone()
    lidar_origin     = origin
    lidar_origin_t   = origin_t
    lidar_lmb        = lmb

    # ── Resize / downsample config ────────────────────────────────────
    ds = args.env_frame_width // args.frame_width
    target_size = (args.frame_height, args.frame_width) if ds != 1 else None

    out_H = args.frame_height if ds != 1 else args.env_frame_height
    out_W = args.frame_width  if ds != 1 else args.env_frame_width
    C_state = 3 + 1 + args.num_sem_categories   # RGB + depth + semantic

    # [OPT-5] Pre-allocate state buffer (float32, channels-first)
    state_buf = np.empty((C_state, out_H, out_W), dtype=np.float32)
    print(f"Pre-allocated state buffer: shape={state_buf.shape}")

    # [OPT-2] Pre-allocate pinned-memory tensor for H→D transfer
    obs_pinned = torch.zeros(
        1, C_state, out_H, out_W,
        dtype=torch.float32,
        pin_memory=True,
    )
    print("Pinned-memory obs tensor allocated")

    # ── [OPT-4]  Pre-allocate ALL pose tensors ────────────────────────
    pose_buf       = torch.zeros(1, 3, device=args.device)
    curr_pose      = torch.zeros(1, 3, device=args.device)
    prev_pose      = torch.zeros(1, 3, device=args.device)
    sensor_pose    = torch.zeros(1, 3, device=args.device)
    rel_pose       = torch.zeros(1, 3, device=args.device)

    # ── [OPT-7]  pose_history as deque ───────────────────────────────
    pose_history        = deque(maxlen=100_000)
    use_lidar_poses     = args.use_lidar_poses
    lidar_pose_history  = deque(maxlen=100_000) if use_lidar_poses else None

    update_frequency = args.update_frequency

    # ── FP16 autocast ─────────────────────────────────────────────────
    use_fp16 = torch.cuda.is_available()

    # ── Start background viz thread ───────────────────────────────────
    viz_thread = VisualizationThread(maxsize=4)
    viz_thread.start()
    print("Visualization thread started")

    # ── [OPT-1] Start detector thread ────────────────────────────────
    det_thread = DetectorThread(
        detector=detector_model,
        conf=args.conf,
        iou=args.iou,
        maxsize=2,
    )
    det_thread.start()
    print("Detector thread started")

    # ═════════════════════════════════════════════════════════════════
    # First frame  (synchronous, bootstraps detector & pose)
    # ═════════════════════════════════════════════════════════════════
    profiler.start("full_frame")

    with profiler.stage("zed_grab"):
        err = zed.grab(runtime_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to grab first frame: {repr(err)}")
        rgb_image, depth_image = grab_zed_frame(
            zed, image_rgb, image_depth_raw, runtime_params
        )

    with profiler.stage("pose_extraction"):
        x, y, yaw = extract_2d_pose(zed, cam_w_pose)
        # [OPT-4] in-place fill, no clone()
        sensor_pose[0, 0] = x
        sensor_pose[0, 1] = y
        sensor_pose[0, 2] = yaw

    local_pose[:, 2] = sensor_pose[:, 2]
    full_pose[:, 2]  = sensor_pose[:, 2]
    pose_history.append(full_pose[0].cpu().numpy().copy())

    if use_lidar_poses:
        lidar_sensor_pose = lidar_poses[0:1].clone()
        lidar_local_pose[:, 2] = lidar_sensor_pose[:, 2]
        lidar_full_pose[:, 2]  = lidar_sensor_pose[:, 2]
        lidar_pose_history.append(lidar_full_pose[0].cpu().numpy().copy())
        print(f"Initial lidar pose: x={lidar_sensor_pose[0,0]:.3f}  "
              f"y={lidar_sensor_pose[0,1]:.3f}  yaw={lidar_sensor_pose[0,2]:.3f}")
        print(f"Initial zed pose:   x={sensor_pose[0,0]:.3f}  "
              f"y={sensor_pose[0,1]:.3f}  yaw={sensor_pose[0,2]:.3f}")

    # [OPT-4] pre-allocated zero tensors, no zeros_like allocation
    rel_pose.zero_()

    # ── Submit frame-0 to detector thread (async) ────────────────────
    det_thread.submit(idx=0, rgb=rgb_image.copy(), skip=False)
    _, first_sem_pred = det_thread.get_result()   # wait for frame 0

    # ── First frame preprocessing ─────────────────────────────────────
    obs_chw = np.concatenate(
        (rgb_image, depth_image), axis=2
    ).transpose(2, 0, 1).astype(np.float32)

    state = _preprocess_obs(
        obs_chw, first_sem_pred, target_size, ds,
        state_buf=state_buf, profiler=profiler,
    )

    # [OPT-2] Pinned copy + non-blocking H→D
    obs_pinned[0].copy_(torch.from_numpy(state))
    obs_t = obs_pinned.to(args.device, non_blocking=True)

    with torch.no_grad():
        with profiler.stage("semantic_mapping"):
            local_map[0, -1, :, :] = 1e-5
            with torch.cuda.amp.autocast(enabled=use_fp16):
                _, local_map, _, local_pose = sem_map_module(
                    obs_t, rel_pose, local_map, local_pose,
                )

        with profiler.stage("map_update"):
            local_map = update_agent_location_channels(local_map, local_pose, args)
            (full_map, local_map, full_pose, local_pose,
             origin, origin_t, lmb) = update_maps_and_pose(
                full_map, local_map, full_pose, local_pose, origin, lmb, args,
            )

        if use_lidar_poses:
            lidar_rel_pose = rel_pose.clone()
            with torch.cuda.amp.autocast(enabled=use_fp16):
                _, lidar_local_map, _, lidar_local_pose = sem_map_module(
                    obs_t, lidar_rel_pose, lidar_local_map, lidar_local_pose,
                )
            lidar_local_map = update_agent_location_channels(
                lidar_local_map, lidar_local_pose, args
            )
            (lidar_full_map, lidar_local_map, lidar_full_pose, lidar_local_pose,
             lidar_origin, lidar_origin_t, lidar_lmb) = update_maps_and_pose(
                lidar_full_map, lidar_local_map, lidar_full_pose, lidar_local_pose,
                lidar_origin, lidar_lmb, args,
            )

    profiler.stop("full_frame")
    profiler.tick_frame()

    # ── Keep last semantic pred for frame-skip ────────────────────────
    last_sem_pred = first_sem_pred

    # ── [OPT-4] in-place copy prev_pose from sensor_pose ─────────────
    prev_pose.copy_(sensor_pose)

    # ═════════════════════════════════════════════════════════════════
    # Start background ZED grabber thread
    # ═════════════════════════════════════════════════════════════════
    grabber = ZedGrabberThread(
        zed, image_rgb, image_depth_raw, runtime_params,
        cam_w_pose, maxsize=2,
    )
    grabber.start()
    print("ZED grabber thread started")

    # ═════════════════════════════════════════════════════════════════
    # SLAM loop  (main thread: GPU processing only)
    # ═════════════════════════════════════════════════════════════════
    i = 1
    lidar_pose_max = len(lidar_poses) if use_lidar_poses else 0

    try:
        while True:
            profiler.start("full_frame")

            # ── [A] Get ZED frame from grabber thread ────────────────
            with profiler.stage("zed_grab"):
                status, frame_data, pose_data, _ = grabber.get()

                if status == "END":
                    print("Reached end of SVO file.")
                    profiler.stop("full_frame")
                    break
                if status == "ERROR":
                    print(f"ZED grab warning: {frame_data}")
                    profiler.stop("full_frame")
                    break

                rgb_image, depth_image = frame_data

            # ── [B] Pose update (in-place, no allocations) ────────────
            with profiler.stage("pose_extraction"):
                x, y, yaw = pose_data
                curr_pose[0, 0] = x     # [OPT-4] in-place
                curr_pose[0, 1] = y
                curr_pose[0, 2] = yaw

            with profiler.stage("relative_pose"):
                rel_pose = compute_relative_pose(prev_pose, curr_pose)

            # ── [OPT-1b] Motion-gated frame skip ─────────────────────
            # Compute translation magnitude to decide whether to re-run detector
            motion = float(
                torch.sqrt(rel_pose[0, 0] ** 2 + rel_pose[0, 1] ** 2).item()
            )
            skip_detection = motion < _MOTION_THRESH

            # ── [C] Submit this frame to detector thread (async) ──────
            # The detector thread will process it while we run semantic
            # mapping for the *previous* frame's prediction below.
            # We submit with a COPY of rgb so the grabber can overwrite.
            det_thread.submit(
                idx=i,
                rgb=rgb_image.copy(),   # lightweight copy; detector needs stable ref
                skip=skip_detection,
            )

            # ── [D] Lidar pose ────────────────────────────────────────
            if use_lidar_poses and i < lidar_pose_max:
                lidar_prev_pose = lidar_poses[i - 1:i]
                lidar_curr_pose = lidar_poses[i:i + 1]
                rel_pose_lidar  = compute_relative_pose(lidar_prev_pose, lidar_curr_pose)
                print(f"Lidar pose: x={lidar_curr_pose[0,0]:.3f}  "
                      f"y={lidar_curr_pose[0,1]:.3f}  yaw={lidar_curr_pose[0,2]:.3f}")
                print(f"ZED pose:   x={curr_pose[0,0]:.3f}  "
                      f"y={curr_pose[0,1]:.3f}  yaw={curr_pose[0,2]:.3f}")

            # ── [E] Get detector result (blocks if not ready yet) ─────
            # In practice the detector thread is running in parallel with
            # steps [A]–[D], so this is often already available (≈0 ms wait).
            with profiler.stage("detection"):
                _, sem_seg_pred = det_thread.get_result()
                last_sem_pred   = sem_seg_pred

            # ── [F] Observation preprocessing (OPT-5: no alloc) ──────
            obs_chw = np.concatenate(
                (rgb_image, depth_image), axis=2
            ).transpose(2, 0, 1).astype(np.float32)

            state = _preprocess_obs(
                obs_chw, sem_seg_pred, target_size, ds,
                state_buf=state_buf, profiler=profiler,
            )

            # ── [G] [OPT-2] Pinned copy + non-blocking H→D ───────────
            obs_pinned[0].copy_(torch.from_numpy(state))
            obs_t = obs_pinned.to(args.device, non_blocking=True)

            with torch.no_grad():

                # ── [H] Semantic mapping update (FP16) ───────────────
                with profiler.stage("semantic_mapping"):
                    with torch.cuda.amp.autocast(enabled=use_fp16):
                        _, local_map, _, local_pose = sem_map_module(
                            obs_t, rel_pose, local_map, local_pose,
                        )

                with profiler.stage("map_update"):
                    local_map = update_agent_location_channels(
                        local_map, local_pose, args
                    )

                    # Lidar branch
                    if use_lidar_poses and i < lidar_pose_max:
                        with torch.cuda.amp.autocast(enabled=use_fp16):
                            _, lidar_local_map, _, lidar_local_pose = sem_map_module(
                                obs_t, rel_pose_lidar,
                                lidar_local_map, lidar_local_pose,
                            )
                        lidar_local_map = update_agent_location_channels(
                            lidar_local_map, lidar_local_pose, args
                        )

                    # ── Periodic global map sync ─────────────────────
                    if i % update_frequency == 0:
                        (full_map, local_map, full_pose, local_pose,
                         origin, origin_t, lmb) = update_maps_and_pose(
                            full_map, local_map, full_pose, local_pose,
                            origin, lmb, args,
                        )
                        if use_lidar_poses:
                            (lidar_full_map, lidar_local_map,
                             lidar_full_pose, lidar_local_pose,
                             lidar_origin, lidar_origin_t, lidar_lmb) = update_maps_and_pose(
                                lidar_full_map, lidar_local_map,
                                lidar_full_pose, lidar_local_pose,
                                lidar_origin, lidar_lmb, args,
                            )

            # [OPT-7] deque append (O(1), no list realloc)
            pose_history.append(full_pose[0].cpu().numpy().copy())
            if use_lidar_poses:
                lidar_pose_history.append(lidar_full_pose[0].cpu().numpy().copy())

            # [OPT-4] in-place prev_pose update, no clone()
            prev_pose.copy_(curr_pose)

            # ── [I] Visualisation (offloaded) ─────────────────────────
            if i % args.saving_frequency == 0:
                with profiler.stage("visualization"):
                    # [OPT-3] Move full map to CPU *before* clone so GPU
                    # is freed immediately; all channels preserved for plot.
                    full_map_cpu = full_map.cpu().clone()

                    # [OPT-6] Pass raw depth, viz thread colorizes it
                    viz_thread.submit(
                        full_map          = full_map_cpu,
                        rgb_image         = rgb_image.copy(),
                        depth_image_raw   = depth_image.copy(),   # [OPT-6]
                        max_depth         = float(args.max_depth), # [OPT-6]
                        pose              = full_pose.cpu().clone(),
                        trajectory        = np.array(pose_history),        # [OPT-7]
                        lidar_trajectory  = (np.array(lidar_pose_history)
                                             if use_lidar_poses else None),
                        lidar_pose        = (lidar_full_pose[0].cpu().numpy().copy()
                                             if use_lidar_poses else None),
                        semantic_pred     = last_sem_pred.copy(),
                        map_resolution    = float(args.map_resolution),
                        semantic_categories = semantic_categories,
                        args              = args,
                        save_path         = f"{dump_dir}/{i}.png",
                    )

                if viz_thread.get_last_key() in (27, ord('q')):
                    print("Exit key pressed. Stopping SLAM process.")
                    profiler.stop("full_frame")
                    break

            profiler.stop("full_frame")
            profiler.tick_frame()
            i += 1

    finally:
        # ── Shutdown background threads ───────────────────────────────
        grabber.stop()
        print("ZED grabber thread stopped")

        det_thread.stop()
        print("Detector thread stopped")

        viz_thread.flush()
        viz_thread.stop()
        print("Visualization thread stopped")

        # ── Final global map sync ─────────────────────────────────────
        with torch.no_grad():
            (full_map, local_map, full_pose, local_pose,
             origin, origin_t, lmb) = update_maps_and_pose(
                full_map, local_map, full_pose, local_pose,
                origin, lmb, args,
            )
            if use_lidar_poses:
                (lidar_full_map, lidar_local_map,
                 lidar_full_pose, lidar_local_pose,
                 lidar_origin, lidar_origin_t, lidar_lmb) = update_maps_and_pose(
                    lidar_full_map, lidar_local_map,
                    lidar_full_pose, lidar_local_pose,
                    lidar_origin, lidar_lmb, args,
                )

        # ── Save final map ────────────────────────────────────────────
        torch.save(full_map, f'{dump_dir}/SLAM_MAP.pt')

        ph_path = f'{dump_dir}/pose_history.npy'
        np.save(ph_path, np.array(pose_history))
        print(f"Saved {len(pose_history)} ZED poses → {ph_path}")
        if lidar_pose_history is not None:
            lph_path = f'{dump_dir}/lidar_pose_history.npy'
            np.save(lph_path, np.array(lidar_pose_history))
            print(f"Saved {len(lidar_pose_history)} lidar poses → {lph_path}")

        area = (full_map[0, 1].sum().item()
                * (args.map_resolution / 100.0) ** 2)
        print(f"Final map exploration area: {area:.2f} m²")
        print("End of SLAM Map")

        # ── Final synchronous visualisation ──────────────────────────
        zed.retrieve_image(depth_display_buf, sl.VIEW.DEPTH)
        depth_display = depth_display_buf.get_data()
        plot_map(
            full_map            = full_map,
            rgb_image           = rgb_image,
            depth_image         = depth_display,
            pose                = full_pose,
            trajectory          = np.array(pose_history),
            lidar_trajectory    = (np.array(lidar_pose_history)
                                   if use_lidar_poses else None),
            lidar_pose          = (lidar_full_pose[0].cpu().numpy()
                                   if use_lidar_poses else None),
            semantic_pred       = last_sem_pred,
            map_resolution      = float(args.map_resolution),
            semantic_categories = semantic_categories,
            args                = args,
            save_path           = f"{dump_dir}/{i}.png",
        )

        # ── Profiling report ──────────────────────────────────────────
        profiler.summary()
        profiler.to_csv(f"{dump_dir}/profiling.csv")

        cv2.destroyAllWindows()
        zed.close()
        print("ZED camera closed.")


if __name__ == "__main__":
    main()

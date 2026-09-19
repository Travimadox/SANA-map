# ═══════════════════════════════════════════════════════════════════════
# PROJECT: SANA-map
# FILE: config.py
# DESCRIPTION: Command-line configuration for the mapping pipeline.
#
#   Supports --config <yaml>: parameters in the file become argparse
#   defaults, and any flag still passed on the command line overrides
#   them. The two example configs under configs/ (indoor.yaml,
#   berryfarm.yaml) reproduce the deployed settings from the paper.
# ═══════════════════════════════════════════════════════════════════════

import argparse

import torch
import yaml


def get_args():
    parser = argparse.ArgumentParser(
        description="SANA-map: real-time open-vocabulary semantic mapping."
    )

    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to a YAML config file. Parameters in the file become "
             "defaults; any CLI argument still overrides them."
    )

    # ── I/O ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--dump_dir", "-d", required=True, type=str,
        help="Directory to write the map, pose history, profiling CSV, "
             "and visualisation frames to."
    )
    parser.add_argument(
        "--svo", type=str, default=None,
        help="Optional ZED .svo recording to replay instead of the live camera."
    )
    parser.add_argument(
        "--saving_frequency", type=int, default=25,
        help="Save a visualisation frame every N frames."
    )
    parser.add_argument(
        "--lidar_txt", type=str, default=None,
        help="Optional path to an offline LIDAR pose trajectory (TUM format), "
             "recorded for evaluation only; does not affect the mapping pipeline."
    )
    parser.add_argument(
        "--use_lidar_poses", type=int, default=0,
        help="1: also warp a second map using the LIDAR trajectory in "
             "--lidar_txt, for offline pose-drift comparison."
    )

    # ── Open-vocabulary perception ─────────────────────────────────────
    parser.add_argument(
        "--classes", "-c", type=str, default="potted plant",
        help='Comma-separated list of prompt categories to map, e.g. '
             '"potted plant,flower pot".'
    )
    parser.add_argument(
        "--yolo_or_owl", type=int, default=2,
        help="Detector backend: 0=YOLO-World, 1=OWLv2, 2=YOLOE, "
             "3=Grounding DINO."
    )
    parser.add_argument(
        "--conf", "-t", type=float, default=0.1,
        help="Detection confidence threshold."
    )
    parser.add_argument(
        "--iou", "-i", type=float, default=0.5,
        help="Detection non-maximum-suppression IoU threshold."
    )
    parser.add_argument(
        "--fast_sam", type=int, default=0,
        help="1: use FastSAM instead of SAM for mask refinement."
    )
    parser.add_argument(
        "--use_reduced_masks", type=int, default=1,
        help="1: shrink instance masks toward their centroid before "
             "projection, reducing boundary spill (Sec. III, retention "
             "factor alpha)."
    )
    parser.add_argument(
        "--mask_reduction_factor", type=float, default=0.4,
        help="Retention factor alpha for --use_reduced_masks (0-1). "
             "alpha=0.4 was used for all reported system-level experiments."
    )
    parser.add_argument(
        "--mask_shape", type=str, default="circular", choices=["circular", "rectangular"],
        help="Shape of the reduced projection mask."
    )
    parser.add_argument(
        "--export", type=str, default="torch", choices=["torch", "onnx", "engine"],
        help="Detector weight format: 'torch' for standard inference, "
             "'onnx'/'engine' to export an accelerated fixed-shape model first."
    )

    # ── Camera / capture ─────────────────────────────────────────────
    parser.add_argument("--frame_width", "-fw", type=int, default=1280,
                        help="Capture width in pixels (ZED 2i: 1280).")
    parser.add_argument("--frame_height", "-fh", type=int, default=720,
                        help="Capture height in pixels (ZED 2i: 720).")
    parser.add_argument("--env_frame_width", "-efw", type=int, default=1280)
    parser.add_argument("--env_frame_height", "-efh", type=int, default=720)
    parser.add_argument("--hfov", type=float, default=100.39,
                        help="Horizontal field of view in degrees, used to "
                             "derive the pinhole intrinsics at the capture "
                             "resolution above (Sec. IV, Experimental Setup).")
    parser.add_argument("--camera_height", type=float, default=0.38,
                        help="Camera height above ground, metres.")
    parser.add_argument("--min_depth", type=float, default=0.5,
                        help="Minimum valid depth, metres.")
    parser.add_argument("--max_depth", type=float, default=5.0,
                        help="Maximum valid depth, metres.")

    # ── Map geometry ──────────────────────────────────────────────────
    parser.add_argument("--map_resolution", type=int, default=5,
                        help="Map cell size, cm per cell.")
    parser.add_argument("--map_size_cm", type=int, default=2500,
                        help="Map extent, cm. 2500 = 25x25 m (indoor); "
                             "6000 = 60x60 m (outdoor) in the paper.")
    parser.add_argument("--global_downscaling", type=int, default=1,
                        help="Divides --map_size_cm to get the local map "
                             "size the network operates over per frame.")
    parser.add_argument("--vision_range", type=int, default=100,
                        help="Egocentric voxel-grid extent, cells "
                             "(100 cells x 5 cm = 5x5 m).")
    parser.add_argument("--du_scale", type=int, default=1,
                        help="Depth-image downscaling factor before "
                             "back-projection.")
    parser.add_argument("--num_processes", type=int, default=1,
                        help="Batch size; 1 for a single live/SVO stream.")
    parser.add_argument("--num_sem_categories", type=int, default=1,
                        help="Number of semantic map channels. Set "
                             "automatically from --classes if left default.")
    parser.add_argument("--farm_or_indoors", type=int, default=0,
                        help="0: indoor map origin at map centre. "
                             "1: outdoor map origin at the map edge along x, "
                             "shifted toward the row entrance (Sec. III-A).")
    parser.add_argument("--map_origin_x_m", type=float, default=None,
                        help="Override the map's x origin, metres. "
                             "Default depends on --farm_or_indoors.")
    parser.add_argument("--map_origin_y_m", type=float, default=None,
                        help="Override the map's y origin, metres.")
    parser.add_argument("--update_frequency", type=int, default=25,
                        help="Fuse the egocentric observation into the "
                             "persistent global map every N frames, and "
                             "export a map snapshot at the same cadence "
                             "(Sec. III-C / Sec. IV-B).")

    # ── Height bands (voxel column -> BEV channels) ────────────────────
    parser.add_argument("--max_height", type=float, default=250.0,
                        help="Top of the voxel grid above ground, cm.")
    parser.add_argument("--min_z", type=float, default=40.0,
                        help="Bottom of the semantic-evidence height band, cm. "
                             "40-60 cm was used for the reported crop-"
                             "geometry experiments.")
    parser.add_argument("--max_z", type=float, default=60.0,
                        help="Top of the semantic-evidence height band, cm.")
    parser.add_argument("--cat_pred_threshold", type=float, default=5.0,
                        help="Normalises accumulated semantic evidence "
                             "before clamping to [0, 1].")
    parser.add_argument("--map_pred_threshold", type=float, default=1.0,
                        help="Normalises accumulated obstacle evidence.")
    parser.add_argument("--exp_pred_threshold", type=float, default=1.0,
                        help="Normalises accumulated explored-space evidence.")

    # ── Two-pass parse: load YAML defaults first, then CLI overrides ──
    pre_args, _ = parser.parse_known_args()
    if pre_args.config is not None:
        with open(pre_args.config, "r") as f:
            cfg = yaml.safe_load(f)
        params = cfg.get("parameters", cfg)

        if "classes" in params and isinstance(params["classes"], list):
            params["classes"] = ",".join(params["classes"])

        known_keys = {a.dest for a in parser._actions}
        params = {k: v for k, v in params.items() if k in known_keys}
        parser.set_defaults(**params)

    args = parser.parse_args()

    args.classes = [c.strip() for c in args.classes.split(",")] if args.classes else []
    if args.num_sem_categories == 1 and len(args.classes) > 1:
        args.num_sem_categories = len(args.classes)

    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.fast_sam = bool(args.fast_sam)
    args.use_lidar_poses = bool(args.use_lidar_poses)

    return args

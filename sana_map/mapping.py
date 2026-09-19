# ═══════════════════════════════════════════════════════════════════════
# PROJECT: SANA-map
# FILE: mapping.py
# DESCRIPTION: Semantic_Mapping, projects prompt-conditioned instance masks
#              and metric depth into an egocentric multichannel BEV
#              observation, then fuses it into the persistent global map.
#              This is the mapping formulation described in the paper's
#              Method section (SANA-map, Sec. III), adapted from Active
#              Neural SLAM / SemExp with fixed semantic channels replaced
#              by user-defined open-vocabulary categories.
# ═══════════════════════════════════════════════════════════════════════

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from .utils.nn_utils import get_grid, ChannelPool
from .utils import depth_utils as du


class Semantic_Mapping(nn.Module):
    """Projects one RGB-D + semantic-mask observation into the persistent
    multichannel BEV map (Sec. III of the paper).

    Channel layout: [0]=obstacle [1]=explored [2]=agent-now
    [3]=agent-visited [4:]=semantic categories.
    """

    def __init__(self, args):
        super(Semantic_Mapping, self).__init__()

        self.device = args.device
        self.screen_h = args.frame_height
        self.screen_w = args.frame_width
        self.resolution = args.map_resolution
        self.z_resolution = args.map_resolution
        self.map_size_cm = args.map_size_cm // args.global_downscaling
        self.n_channels = 3
        self.vision_range = args.vision_range
        self.dropout = 0.5
        self.fov = args.hfov
        self.du_scale = args.du_scale
        self.cat_pred_threshold = args.cat_pred_threshold
        self.exp_pred_threshold = args.exp_pred_threshold
        self.map_pred_threshold = args.map_pred_threshold
        self.num_sem_categories = args.num_sem_categories
        self.args = args

        # ── Temporal map persistence ───────────────────────────────────
        # forward() fuses the warped current observation into the running
        # map with an element-wise max (see fuse()). That max IS the
        # temporal persistence mechanism. Channels listed in
        # self.nopersist_ch bypass it and are overwritten with the current
        # frame's observation instead, so they carry no history.
        #
        # 'all' (default) leaves the fusion untouched.
        self.map_persistence = getattr(args, 'map_persistence', 'all')
        _sem = list(range(4, 4 + self.num_sem_categories))
        if self.map_persistence == 'all':
            self.nopersist_ch = None
        elif self.map_persistence == 'semantic':
            self.nopersist_ch = _sem
        elif self.map_persistence == 'none':
            self.nopersist_ch = [0, 1] + _sem
        else:
            raise ValueError(
                f"map_persistence must be one of 'all'/'semantic'/'none', "
                f"got {self.map_persistence!r}")

        self.max_height = int(args.max_height / self.z_resolution)
        self.min_height = int(0 / self.z_resolution)

        self.agent_height = args.camera_height * 100.
        self.shift_loc = [self.vision_range *
                          self.resolution // 2, 0, np.pi / 2.0]

        # Pinhole intrinsics derived from the deployed capture resolution
        # and the camera's horizontal field of view (Sec. IV, Experimental
        # Setup: 1280x720 at 100.39 deg HFOV for the ZED 2i).
        self.camera_matrix = du.get_camera_matrix(
            self.screen_w, self.screen_h, self.fov)

        self.pool = ChannelPool(1)

        vr = self.vision_range

        self.init_grid = torch.zeros(
            args.num_processes, 1 + self.num_sem_categories, vr, vr,
            self.max_height - self.min_height
        ).float().to(self.device)
        self.feat = torch.ones(
            args.num_processes, 1 + self.num_sem_categories,
            self.screen_h // self.du_scale * self.screen_w // self.du_scale
        ).float().to(self.device)

    def forward(self, obs, pose_obs, maps_last, poses_last):
        bs, c, h, w = obs.size()
        depth = obs[:, 3, :, :]

        point_cloud_t = du.get_point_cloud_from_z_t(
            depth, self.camera_matrix, self.device, scale=self.du_scale)

        agent_view_t = du.transform_camera_view_t(
            point_cloud_t, self.agent_height, 0, self.device)

        agent_view_centered_t = du.transform_pose_t(
            agent_view_t, self.shift_loc, self.device)

        max_h = self.max_height
        min_h = self.min_height
        xy_resolution = self.resolution
        z_resolution = self.z_resolution
        vision_range = self.vision_range
        XYZ_cm_std = agent_view_centered_t.float()
        XYZ_cm_std[..., :2] = (XYZ_cm_std[..., :2] / xy_resolution)
        XYZ_cm_std[..., :2] = (XYZ_cm_std[..., :2] -
                               vision_range // 2.) / vision_range * 2.
        XYZ_cm_std[..., 2] = XYZ_cm_std[..., 2] / z_resolution
        XYZ_cm_std[..., 2] = (XYZ_cm_std[..., 2] -
                              (max_h + min_h) // 2.) / (max_h - min_h) * 2.
        self.feat[:, 1:, :] = nn.AvgPool2d(self.du_scale)(
            obs[:, 4:, :, :]
        ).view(bs, c - 4, h // self.du_scale * w // self.du_scale)

        XYZ_cm_std = XYZ_cm_std.permute(0, 3, 1, 2)
        XYZ_cm_std = XYZ_cm_std.view(XYZ_cm_std.shape[0],
                                     XYZ_cm_std.shape[1],
                                     XYZ_cm_std.shape[2] * XYZ_cm_std.shape[3])

        voxels = du.splat_feat_nd(
            self.init_grid * 0., self.feat, XYZ_cm_std).transpose(2, 3)

        min_z = int(self.args.min_z / z_resolution - min_h)
        max_z = int(self.args.max_z / z_resolution - min_h)

        agent_height_proj = voxels[..., min_z:max_z].sum(4)
        all_height_proj = voxels.sum(4)

        fp_map_pred = agent_height_proj[:, 0:1, :, :]
        fp_exp_pred = all_height_proj[:, 0:1, :, :]
        fp_map_pred = fp_map_pred / self.map_pred_threshold
        fp_exp_pred = fp_exp_pred / self.exp_pred_threshold
        fp_map_pred = torch.clamp(fp_map_pred, min=0.0, max=1.0)
        fp_exp_pred = torch.clamp(fp_exp_pred, min=0.0, max=1.0)

        pose_pred = poses_last

        agent_view = torch.zeros(bs, c,
                                 self.map_size_cm // self.resolution,
                                 self.map_size_cm // self.resolution
                                 ).to(self.device)

        x1 = self.map_size_cm // (self.resolution * 2) - self.vision_range // 2
        x2 = x1 + self.vision_range
        y1 = self.map_size_cm // (self.resolution * 2)
        y2 = y1 + self.vision_range
        agent_view[:, 0:1, y1:y2, x1:x2] = fp_map_pred
        agent_view[:, 1:2, y1:y2, x1:x2] = fp_exp_pred
        agent_view[:, 4:, y1:y2, x1:x2] = torch.clamp(
            agent_height_proj[:, 1:, :, :] / self.cat_pred_threshold,
            min=0.0, max=1.0)

        corrected_pose = pose_obs

        def get_new_pose_batch(pose, rel_pose_change):

            pose[:, 1] += rel_pose_change[:, 0] * \
                torch.sin(pose[:, 2] / 57.29577951308232) \
                + rel_pose_change[:, 1] * \
                torch.cos(pose[:, 2] / 57.29577951308232)
            pose[:, 0] += rel_pose_change[:, 0] * \
                torch.cos(pose[:, 2] / 57.29577951308232) \
                - rel_pose_change[:, 1] * \
                torch.sin(pose[:, 2] / 57.29577951308232)
            pose[:, 2] += rel_pose_change[:, 2] * 57.29577951308232

            pose[:, 2] = torch.fmod(pose[:, 2] - 180.0, 360.0) + 180.0
            pose[:, 2] = torch.fmod(pose[:, 2] + 180.0, 360.0) - 180.0

            return pose

        current_poses = get_new_pose_batch(poses_last, corrected_pose)
        st_pose = current_poses.clone().detach()

        st_pose[:, :2] = - (st_pose[:, :2]
                            * 100.0 / self.resolution
                            - self.map_size_cm // (self.resolution * 2)) /\
            (self.map_size_cm // (self.resolution * 2))
        st_pose[:, 2] = 90. - (st_pose[:, 2])

        rot_mat, trans_mat = get_grid(st_pose, agent_view.size(),
                                      self.device)

        rotated = F.grid_sample(agent_view, rot_mat, align_corners=True)
        translated = F.grid_sample(rotated, trans_mat, align_corners=True)

        map_pred = self.fuse(maps_last, translated)

        return fp_map_pred, map_pred, pose_pred, current_poses

    def fuse(self, maps_last, translated):
        """Combine the running map with the warped current observation.

        The element-wise max IS this system's temporal persistence: a cell
        once seen as occupied stays occupied. Channels in self.nopersist_ch
        bypass it and take the current observation alone.
        """
        maps2 = torch.cat((maps_last.unsqueeze(1), translated.unsqueeze(1)), 1)
        map_pred, _ = torch.max(maps2, 1)

        # torch.max returns a fresh tensor, so this in-place write is safe.
        if self.nopersist_ch is not None:
            map_pred[:, self.nopersist_ch] = translated[:, self.nopersist_ch]

        return map_pred

# ═══════════════════════════════════════════════════════════════════════
# PROJECT: SANA-map
# FILE: visualisation.py
# DESCRIPTION: Contains code for visualising the semantic map uisng Open CV
# AUTHOR: Travimadox Webb
# LAST MODIFIED: 1/11/2025
# TODO
# Add Legend to map
# No Goal plotting
# ═══════════════════════════════════════════════════════════════════════

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion

# Color Palette for the different categories in the map
COLOR_PALETTE = [
    (0, 0, 0),         # 0: Black 
    (128, 128, 128),   # 1: Obstacles (Gray)
    (230, 230, 230),   # 2: Explored (Light Gray)
    (0, 0, 200),       # 3: Visited Path (Red)
    (0, 255, 0),       # 4: Goal (Green - not used here, but a placeholder)
    (255, 100, 0),     # 5: Potted Plants (Blue) 
    (255, 255, 0),     # 6: No Category (Cyan) 
    (0, 165, 255),     # 7: Semantic Cat 3 (Orange)
]

# Add more colors to accomodate for extra semnatic categories if needed
for i in range(20): COLOR_PALETTE.append(tuple(np.random.randint(0, 255, 3)))

def create_pil_palette():
    """Creates a flattened palette for PIL."""
    palette = []
    for color in COLOR_PALETTE: palette.extend(color)
    while len(palette) < 768: palette.extend((0,0,0))
    return palette

def plot_map(
        full_map,
        rgb_image,
        depth_image,
        pose,
        trajectory,
        lidar_trajectory=None,
        lidar_pose=None,
        semantic_pred=None,
        map_resolution=5,
        semantic_categories=None,
        args=None,
        save_path=None):

    # 1. --- Prepare Data ---
    map_data = full_map[0].cpu().numpy()
    agent_pose = pose[0].cpu().numpy()
    map_h, map_w = map_data.shape[1], map_data.shape[2]
    
    obstacle_map = np.rint(map_data[0]) == 1
    explored_map = np.rint(map_data[1]) == 1
    sem_channels = map_data[4:]

    # 2. --- Compose Single Integer Map ---
    int_map = np.zeros((map_h, map_w), dtype=np.uint8)
    int_map[explored_map] = 2
    int_map[obstacle_map] = 1

    """
    if semantic_categories:
        if sem_channels.shape[0] > 0: int_map[np.rint(sem_channels[0]) == 1] = 5
        if sem_channels.shape[0] > 1: int_map[np.rint(sem_channels[1]) == 1] = 6
    """
    
    label_offset = 5

    if semantic_categories:
        for i in range(sem_channels.shape[0]):
            current_label = label_offset + i
            int_map[np.rint(sem_channels[i]) == 1] = current_label

    

    """
    if semantic_categories:
        # --- EROSION LOGIC START ---
        # Channel 0: Plants (We want to erode this to separate blobs)
        if sem_channels.shape[0] > 0: 
            # 1. Get the raw mask
            plant_mask = np.rint(sem_channels[0]) == 1
            
            # 2. Apply Erosion
            # structure: defines the shape (3x3 square)
            # iterations: how much to shrink (2 is usually a good start)
            eroded_plant_mask = binary_erosion(
                plant_mask, 
                structure=np.ones((3,3)), 
                iterations=1 
            )
            
            # 3. Add to map
            int_map[eroded_plant_mask] = 5

        # Channel 1: Other Category (Apply standard logic or different erosion)
        if sem_channels.shape[0] > 1: 
            int_map[np.rint(sem_channels[1]) == 1] = 6
        # --- EROSION LOGIC END ---
    """

    
    

    # 3. --- Draw Trajectory ---
    if len(trajectory) > 1:
        traj_pixels = []
        for p in trajectory:
            x_px = np.clip(int(p[0] * 100.0 / map_resolution), 0, map_w - 1)
            y_px = np.clip(int(p[1] * 100.0 / map_resolution), 0, map_h - 1)
            traj_pixels.append((x_px, y_px))
        
        # Draw trajectory on a temporary mask to avoid flipping issues
        visited_vis = np.zeros((map_h, map_w), dtype=np.uint8)
        for i in range(len(traj_pixels) - 1):
            cv2.line(visited_vis, traj_pixels[i], traj_pixels[i+1], color=255, thickness=2)
        int_map[visited_vis == 255] = 3

    # 3b. --- Draw Lidar Trajectory (Green) ---
    if lidar_trajectory is not None and len(lidar_trajectory) > 1:
        lidar_traj_pixels = []
        for lp in lidar_trajectory:
            lx_px = np.clip(int(lp[0] * 100.0 / map_resolution), 0, map_w - 1)
            ly_px = np.clip(int(lp[1] * 100.0 / map_resolution), 0, map_h - 1)
            lidar_traj_pixels.append((lx_px, ly_px))
        
        # Draw lidar trajectory on a separate mask to avoid overwriting
        lidar_vis = np.zeros((map_h, map_w), dtype=np.uint8)
        for li in range(len(lidar_traj_pixels) - 1):
            cv2.line(lidar_vis, lidar_traj_pixels[li], lidar_traj_pixels[li+1], color=255, thickness=2)
        int_map[lidar_vis == 255] = 4  # Index 4 = Green in COLOR_PALETTE

    # 4. --- Colorize, Flip, and Draw Agent ---
    pil_palette = create_pil_palette()
    sem_map_vis = Image.new("P", (map_w, map_h))
    sem_map_vis.putpalette(pil_palette)
    sem_map_vis.putdata(int_map.flatten().astype(np.uint8))
    sem_map_vis = sem_map_vis.convert("RGB")
    sem_map_vis = np.array(sem_map_vis)
    sem_map_vis[int_map == 0] = [255, 255, 255]

    sem_map_vis = np.flipud(sem_map_vis)
    sem_map_vis = np.ascontiguousarray(sem_map_vis, dtype=np.uint8)
    
    # Draw ZED agent arrow (Red)
    start_x, start_y, start_o = agent_pose
    zed_x_px = int(start_x * 100. / map_resolution)
    zed_y_px = int(map_h - (start_y * 100. / map_resolution))
    zed_angle_rad = np.deg2rad(-start_o)
    zed_dx = int(8 * 2.5 * np.cos(zed_angle_rad))
    zed_dy = int(8 * 2.5 * np.sin(zed_angle_rad))
    cv2.arrowedLine(sem_map_vis, (zed_x_px, zed_y_px), (zed_x_px + zed_dx, zed_y_px + zed_dy), (0, 0, 200), 2, tipLength=0.5)

    # Draw Lidar agent arrow (Green)
    if lidar_pose is not None:
        lidar_start_x, lidar_start_y, lidar_start_o = lidar_pose
        lidar_x_px = int(lidar_start_x * 100. / map_resolution)
        lidar_y_px = int(map_h - (lidar_start_y * 100. / map_resolution))
        lidar_angle_rad = np.deg2rad(-lidar_start_o)
        lidar_dx = int(8 * 2.5 * np.cos(lidar_angle_rad))
        lidar_dy = int(8 * 2.5 * np.sin(lidar_angle_rad))
        cv2.arrowedLine(sem_map_vis, (lidar_x_px, lidar_y_px), (lidar_x_px + lidar_dx, lidar_y_px + lidar_dy), (0, 200, 0), 2, tipLength=0.5)

    # 5. --- Create Composite 2x2 Image ---
    subplot_size = (480, 640)
    map_subplot_size = (480, 480)
    
    rgb_display = cv2.cvtColor(rgb_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
    rgb_display = cv2.resize(rgb_display, (subplot_size[1], subplot_size[0]))
    
    depth_display = depth_image
    depth_display = cv2.cvtColor(depth_display, cv2.COLOR_BGRA2BGR)
    #depth_display = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
    #depth_display = cv2.applyColorMap((depth_image * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    depth_display = cv2.applyColorMap(depth_display, cv2.COLORMAP_VIRIDIS)
    depth_display = cv2.resize(depth_display, (subplot_size[1], subplot_size[0]))

    detections_display = rgb_display.copy()
    overlay = np.zeros_like(detections_display, dtype=np.uint8)

    """
    if semantic_categories and semantic_pred is not None:
        colors = [(255, 100, 100)] 
        for i in range(semantic_pred.shape[2]):
            original_mask = semantic_pred[:, :, i] > 0
            target_size = (detections_display.shape[1], detections_display.shape[0])
            mask = cv2.resize(original_mask.astype(np.uint8), target_size, interpolation=cv2.INTER_NEAREST).astype(bool)
            if i < len(colors): overlay[mask] = colors[i]
    """
    if semantic_categories and semantic_pred is not None:
        for i in range(semantic_pred.shape[2]):
            original_mask = semantic_pred[:, :, i] > 0
            target_size = (detections_display.shape[1], detections_display.shape[0])
            mask = cv2.resize(original_mask.astype(np.uint8), target_size, interpolation=cv2.INTER_NEAREST).astype(bool)
            
            # Use COLOR_PALETTE instead of hardcoded colors
            # Offset by 5 to use the semantic category colors (indices 5+)
            color_idx = i + 5  # Start from index 5 (Potted Plants)
            if color_idx < len(COLOR_PALETTE):
                overlay[mask] = COLOR_PALETTE[color_idx]
    detections_display = cv2.addWeighted(overlay, 0.5, detections_display, 0.5, 0)
    
    slam_map_display = cv2.resize(sem_map_vis, map_subplot_size, interpolation=cv2.INTER_NEAREST)
    
    total_h, total_w = subplot_size[0] * 2 + 80, subplot_size[1] * 2 + 60
    vis_image = np.ones((total_h, total_w, 3), dtype=np.uint8) * 240

    vis_image[40:520, 20:660] = rgb_display
    vis_image[40:520, 680:1320] = detections_display
    vis_image[560:1040, 20:660] = depth_display
    
    slam_x_start = 680 + ((subplot_size[1] - map_subplot_size[1]) // 2)
    slam_y_start = 560 + ((subplot_size[0] - map_subplot_size[0]) // 2)
    vis_image[slam_y_start:slam_y_start+map_subplot_size[0], slam_x_start:slam_x_start+map_subplot_size[1]] = slam_map_display
    cv2.rectangle(vis_image, (slam_x_start, slam_y_start), (slam_x_start + map_subplot_size[1], slam_y_start + map_subplot_size[0]), (0, 0, 0), 1)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(vis_image, 'RGB Image', (20, 30), font, 1, (0,0,0), 2)
    cv2.putText(vis_image, 'Custom Detections', (680, 30), font, 1, (0,0,0), 2)
    cv2.putText(vis_image, 'Depth Image', (20, 550), font, 1, (0,0,0), 2)
    cv2.putText(vis_image, 'Semantic SLAM Map', (680, 550), font, 1, (0,0,0), 2)

    # 6. --- Output ---
    key = -1
    if save_path:
        cv2.imwrite(save_path, vis_image)

    if args and hasattr(args, 'live_plotting') and args.live_plotting:
        cv2.imshow("Semantic SLAM Visualization", vis_image)
        key = cv2.waitKey(1) & 0xFF
    
    return key


def plot_map_only(
        full_map,
        semantic_categories=None,
        args=None,
        save_path="Results/Maptest.png"):

    # 1. --- Prepare Data ---
    map_data = full_map[0].cpu().numpy()
    map_h, map_w = map_data.shape[1], map_data.shape[2]
    
    obstacle_map = np.rint(map_data[0]) == 1
    explored_map = np.rint(map_data[1]) == 1
    sem_channels = map_data[4:]

    # 2. --- Compose Single Integer Map ---
    int_map = np.zeros((map_h, map_w), dtype=np.uint8)
    int_map[explored_map] = 2
    int_map[obstacle_map] = 1

    if semantic_categories:
        if sem_channels.shape[0] > 0: int_map[np.rint(sem_channels[0]) == 1] = 5
        if sem_channels.shape[0] > 1: int_map[np.rint(sem_channels[1]) == 1] = 6

    # 3. --- Colorize and Flip ---
    pil_palette = create_pil_palette()
    sem_map_vis = Image.new("P", (map_w, map_h))
    sem_map_vis.putpalette(pil_palette)
    sem_map_vis.putdata(int_map.flatten().astype(np.uint8))
    sem_map_vis = sem_map_vis.convert("RGB")
    sem_map_vis = np.array(sem_map_vis)
    sem_map_vis[int_map == 0] = [255, 255, 255]

    sem_map_vis = np.flipud(sem_map_vis)
    vis_image = np.ascontiguousarray(sem_map_vis, dtype=np.uint8)

    # 4. --- Output ---
    key = -1
    if save_path:
        cv2.imwrite(save_path, vis_image)

    if args and hasattr(args, 'live_plotting') and args.live_plotting:
        cv2.imshow("Semantic SLAM Map", vis_image)
        key = cv2.waitKey(1) & 0xFF
    
    return key
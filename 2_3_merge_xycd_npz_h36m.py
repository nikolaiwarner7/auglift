import numpy as np
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import ipdb
import os
import pickle
from glob import glob
import torch
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
import argparse

# Set multiprocessing start method for safety with NumPy/zipfile
if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set

# ============================================================================
# COORDINATE SYSTEM NOTES:
# ============================================================================
# - H36M images: 1000x1000 (square)
# - RTMPose processes CROPPED patches (from person detector bbox), not full images!
#   * RTMPose feature maps (rtm_feature_coarse): ~12x9 spatial dims (from CROPPED region)
#   * Saved bbox: [x1, y1, x2, y2] in full image coordinates
#   * Keypoint coords in full image space must be transformed to bbox-relative space
# - DepthAnythingV2 processes full images, outputs at full resolution
# - DAV2 feature maps (dav2_feature_256): spatial dims match full image aspect ratio
# - All 2D keypoints are in original FULL image coordinates (0 to W-1, 0 to H-1)
# - Bilinear sampling maps coordinates appropriately:
#   * RTMPose: full image coords → bbox-relative coords → feature grid coords
#   * DAV2: full image coords → feature grid coords
# ============================================================================
# FEATURE CACHING FLAGS:
# ============================================================================
# CACHE_RTM_FEATURES:  Cache per-joint RTMPose backbone features (17×1024-D per frame)
#                      Use for learning appearance/pose-aware representations
#                      File size impact: ~35KB per frame (1024 channels × 17 joints × 2 bytes)
# CACHE_DAV2_FEATURES: Cache per-joint DepthAnythingV2 features (17×256-D per frame)
#                      Use for learning depth-aware representations
#                      File size impact: ~9KB per frame (256 channels × 17 joints × 2 bytes)
# Can enable/disable independently to control NPZ file size
# ============================================================================
# FEATURE SAMPLING STRATEGY:
# ============================================================================
# Controls how features are sampled from coarse feature maps at keypoint locations.
# Options:
#   'nearest_1': Use only the nearest grid cell (most robust, least precise)
#   'nearest_2': Use 2 nearest cells - horizontal or vertical pair (middle ground)
#   'nearest_4': Use 4 nearest cells with uniform average (robust but localized)
#   'bilinear': Standard bilinear interpolation (most precise, least robust)
# 
# For coarse RTM features (12×9 grid), nearest_4 offers good robustness without
# spreading too wide, since each cell already covers ~80-100 pixels.
# ============================================================================


# ============================================================================
# ORIGINAL BILINEAR INTERPOLATION (COMMENTED OUT - REPLACED WITH CONFIGURABLE SAMPLING)
# ============================================================================
# def bilinear_sample_numpy(feat_map, u, v):
#     """Bilinear sampling for a single point in numpy.
#     
#     Args:
#         feat_map: (C, H, W) numpy array
#         u, v: float coordinates in [0, W-1] and [0, H-1]
#     Returns:
#         (C,) sampled features
#     """
#     C, H, W = feat_map.shape
#     
#     # Get integer coordinates
#     u0, v0 = int(np.floor(u)), int(np.floor(v))
#     u1, v1 = u0 + 1, v0 + 1
#     
#     # Clamp to valid range
#     u0 = np.clip(u0, 0, W - 1)
#     u1 = np.clip(u1, 0, W - 1)
#     v0 = np.clip(v0, 0, H - 1)
#     v1 = np.clip(v1, 0, H - 1)
#     
#     # Get fractional parts
#     du = u - u0
#     dv = v - v0
#     
#     # Bilinear interpolation
#     w00 = (1 - du) * (1 - dv)
#     w01 = (1 - du) * dv
#     w10 = du * (1 - dv)
#     w11 = du * dv
#     
#     result = (w00 * feat_map[:, v0, u0] + 
#               w01 * feat_map[:, v1, u0] + 
#               w10 * feat_map[:, v0, u1] + 
#               w11 * feat_map[:, v1, u1])
#     
#     return result


def sample_features_numpy(feat_map, u, v, mode='nearest_4'):
    """Sample features at a fractional grid location with configurable strategy.
    
    Args:
        feat_map: (C, H, W) numpy array
        u, v: float coordinates in [0, W-1] and [0, H-1]
        mode: Sampling strategy - 'nearest_1', 'nearest_2', 'nearest_4', or 'bilinear'
    Returns:
        (C,) sampled features
    """
    C, H, W = feat_map.shape
    
    # Get integer coordinates
    u0, v0 = int(np.floor(u)), int(np.floor(v))
    u1, v1 = u0 + 1, v0 + 1
    
    # Clamp to valid range
    u0 = np.clip(u0, 0, W - 1)
    u1 = np.clip(u1, 0, W - 1)
    v0 = np.clip(v0, 0, H - 1)
    v1 = np.clip(v1, 0, H - 1)
    
    if mode == 'nearest_1':
        # Use only the nearest grid cell (most robust to detector noise)
        u_nearest = u0 if (u - u0) < 0.5 else u1
        v_nearest = v0 if (v - v0) < 0.5 else v1
        return feat_map[:, v_nearest, u_nearest]
    
    elif mode == 'nearest_2':
        # Use 2 nearest cells - pick horizontal or vertical pair based on which border is closer
        du = u - u0
        dv = v - v0
        
        # Determine if we're closer to horizontal or vertical boundary
        dist_to_horiz_border = min(du, 1 - du)  # Distance to left or right cell boundary
        dist_to_vert_border = min(dv, 1 - dv)   # Distance to top or bottom cell boundary
        
        if dist_to_horiz_border < dist_to_vert_border:
            # Closer to horizontal border - use left and right cells
            result = (feat_map[:, v0, u0] + feat_map[:, v0, u1]) / 2.0
        else:
            # Closer to vertical border - use top and bottom cells
            result = (feat_map[:, v0, u0] + feat_map[:, v1, u0]) / 2.0
        
        return result
    
    elif mode == 'nearest_4':
        # Use 4 nearest cells with uniform average (no distance weighting)
        # More robust than bilinear, still localized to 2×2 neighborhood
        result = (feat_map[:, v0, u0] + 
                  feat_map[:, v1, u0] + 
                  feat_map[:, v0, u1] + 
                  feat_map[:, v1, u1]) / 4.0
        return result
    
    elif mode == 'bilinear':
        # Standard bilinear interpolation (distance-weighted)
        du = u - u0
        dv = v - v0
        
        w00 = (1 - du) * (1 - dv)
        w01 = (1 - du) * dv
        w10 = du * (1 - dv)
        w11 = du * dv
        
        result = (w00 * feat_map[:, v0, u0] + 
                  w01 * feat_map[:, v1, u0] + 
                  w10 * feat_map[:, v0, u1] + 
                  w11 * feat_map[:, v1, u1])
        return result
    
    else:
        raise ValueError(f"Unknown sampling mode: {mode}. Use 'nearest_1', 'nearest_2', 'nearest_4', or 'bilinear'.")


def sample_feat_for_joint(feat_map, kp, Hf, Wf, Himg, Wimg, bbox=None):
    """Sample features at a keypoint location.
    
    Args:
        feat_map: (C, H, W) feature map
        kp: (2,) keypoint [x, y] in FULL image coordinates
        Hf, Wf: feature map spatial dimensions (e.g., 9, 12 for RTM)
        Himg, Wimg: reference image dimensions (full image)
        bbox: (4,) [x1, y1, x2, y2] bounding box in full image coords (optional)
              If provided, keypoint coords will be transformed to bbox-relative coords
              using the RTMPose preprocessing logic (isotropic resize + pad).
    Returns:
        (C,) sampled features
    """
    if bbox is not None:
        # RTMPose preprocessing: isotropic resize and center-padding.
        # This logic mirrors the `pad_resize` function used in data generation.
        # Model input size for RTMPose-L is 288x384 (HxW).
        target_h, target_w = 288, 384
        
        x1, y1, x2, y2 = bbox
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        
        # 1. Calculate isotropic scale factor
        scale = min(target_w / bbox_w, target_h / bbox_h)
        
        # 2. Calculate padding
        new_w, new_h = int(bbox_w * scale), int(bbox_h * scale)
        pad_x = (target_w - new_w) / 2
        pad_y = (target_h - new_h) / 2
        
        # 3. Map keypoint from full image to model input space
        #    - Get keypoint relative to bbox top-left corner
        #    - Scale it
        #    - Add padding offset
        kp_x_model_input = (kp[0] - x1) * scale + pad_x
        kp_y_model_input = (kp[1] - y1) * scale + pad_y
        
        # 4. Map from model input space (384x288) to feature map space (Wf x Hf)
        u = (kp_x_model_input / target_w) * (Wf - 1)
        v = (kp_y_model_input / target_h) * (Hf - 1)
    else:
        # Standard mapping for full image features (e.g., DAV2)
        u = (kp[0] / Wimg) * (Wf - 1)
        v = (kp[1] / Himg) * (Hf - 1)
    
    return sample_features_numpy(feat_map, u, v, mode=FEATURE_SAMPLING_MODE)


def visualize_joint_patches(depth_data, keypoints_2d, Himg, Wimg, img_base_name, img_dir):
    """Visualize hand/foot patch extraction for debugging.
    
    Args:
        depth_data: NPZ data containing feature maps
        keypoints_2d: (17, 2) keypoint coordinates
        Himg, Wimg: Image dimensions
        img_base_name: Base name for the image file
        img_dir: Directory containing H36M images
    """
    import matplotlib.pyplot as plt
    import cv2
    from matplotlib.patches import Rectangle
    
    # Choose joints of interest: hands, feet, head (H36M joint indices)
    # Indices consistent with your tested script and common H36M layout:
    # 0=root, 3=R-ankle/foot, 6=L-ankle/foot, 10=head, 13=L-wrist, 16=R-wrist
    JOINTS_OF_INTEREST = [13, 16, 6, 3, 10]            # L-wrist, R-wrist, L-ankle, R-ankle, Head
    JOINT_NAMES        = ["L-wrist", "R-wrist", "L-ankle", "R-ankle", "Head"]
    colors             = ["red", "green", "blue", "cyan", "magenta"]

    
    # Try to load the actual image
    # H36M image name format: s_01_act_02_subact_01_ca_01_000001 -> need to find actual image
    img_path = None
    if img_dir and os.path.exists(img_dir):
        # Our folders are named exactly like the prefix (e.g., s_01_act_02_subact_01_ca_01)
        # and files are "<prefix>_<frame>.jpg"
        parts = img_base_name.split('_')
        seq   = '_'.join(parts[:-1])     # e.g., s_01_act_02_subact_01_ca_01
        frame = parts[-1]                # e.g., 000009
        candidate = os.path.join(img_dir, seq, f"{seq}_{frame}.jpg")
        if os.path.exists(candidate):
            img_path = candidate

    
    if img_path and os.path.exists(img_path):
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    else:
        # Create blank image with checkerboard pattern for visualization
        img = np.ones((Himg, Wimg, 3), dtype=np.uint8) * 200
        tile_size = 50
        for i in range(0, Himg, tile_size):
            for j in range(0, Wimg, tile_size):
                if (i // tile_size + j // tile_size) % 2 == 0:
                    img[i:i+tile_size, j:j+tile_size] = 220
    
    # Get feature map dimensions
    has_rtm = 'rtm_feature_coarse' in depth_data and depth_data['rtm_feature_coarse'] is not None
    has_dav2 = 'dav2_feature_256' in depth_data and depth_data['dav2_feature_256'] is not None
    
    if not has_rtm and not has_dav2:
        print(f"  No feature maps found for {img_base_name}")
        return
    
    # Create visualization figure
    num_cols = 1 + int(has_rtm) + int(has_dav2)
    fig, axes = plt.subplots(1, num_cols, figsize=(6*num_cols, 6))
    if num_cols == 1:
        axes = [axes]
    
    ax_idx = 0
    
    # --- Left panel: Image + 2D keypoints
    axes[ax_idx].imshow(img)
    for j, name, c in zip(JOINTS_OF_INTEREST, JOINT_NAMES, colors):
        x, y = keypoints_2d[j]
        axes[ax_idx].scatter(x, y, c=c, s=100, marker='o', edgecolors='white', linewidths=2, label=name)
        axes[ax_idx].text(x+20, y-20, name, color=c, fontsize=9, weight='bold',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    axes[ax_idx].set_title(f"Image + Keypoints\n{img_base_name}", fontsize=10)
    axes[ax_idx].legend(loc='upper right', fontsize=8)
    axes[ax_idx].set_xlim(0, Wimg)
    axes[ax_idx].set_ylim(Himg, 0)
    ax_idx += 1
    
    # --- RTM Feature Map Visualization
    if has_rtm:
        rtm_feat = depth_data['rtm_feature_coarse']
        if rtm_feat.ndim == 4:
            rtm_feat = rtm_feat[0]
        _, Hrtm, Wrtm = rtm_feat.shape
        
        # Load bbox if available (required for correct visualization)
        rtm_bbox = depth_data.get('rtm_bbox', None)
        
        # Compute L2-norm energy across channels (shows where features are strong)
        rtm_energy = np.linalg.norm(rtm_feat, axis=0)  # (Hrtm, Wrtm)
        
        # Validate and normalize energy map
        if np.isnan(rtm_energy).any() or np.isinf(rtm_energy).any():
            rtm_energy = np.nan_to_num(rtm_energy, nan=0.0, posinf=1.0, neginf=0.0)
        
        energy_range = rtm_energy.ptp()
        if energy_range > 1e-6:
            rtm_energy = (rtm_energy - rtm_energy.min()) / energy_range
        else:
            rtm_energy = np.zeros_like(rtm_energy)
        
        # Show image
        axes[ax_idx].imshow(img, alpha=1.0)
        
        # Draw bbox if available (RTM features come from this cropped region!)
        if rtm_bbox is not None:
            x1, y1, x2, y2 = rtm_bbox
            bbox_rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                                 linewidth=3, edgecolor='yellow', facecolor='none', 
                                 linestyle='--', label='RTM bbox')
            axes[ax_idx].add_patch(bbox_rect)
            
            # Overlay energy map ONLY within bbox region
            # Resize energy map to match bbox dimensions (ensure valid size)
            bbox_w, bbox_h = max(1, int(x2 - x1)), max(1, int(y2 - y1))
            
            # Only resize if bbox is large enough (> 10 pixels in both dims) and rtm_energy is valid
            if bbox_w > 10 and bbox_h > 10 and rtm_energy.size > 0:
                try:
                    # cv2.resize requires float32 or float64, convert from float16 if needed
                    energy_for_resize = rtm_energy.astype(np.float32) if rtm_energy.dtype == np.float16 else rtm_energy
                    energy_resized = cv2.resize(energy_for_resize, (bbox_w, bbox_h), interpolation=cv2.INTER_LINEAR)
                    
                    # Create RGBA overlay with transparency
                    energy_rgba = plt.cm.viridis(energy_resized)
                    energy_rgba[:, :, 3] = 0.6  # Set alpha channel
                    
                    # Display energy map at bbox location
                    axes[ax_idx].imshow(energy_rgba, extent=[x1, x2, y2, y1], interpolation='bilinear')
                except cv2.error as e:
                    # If resize fails, just skip the heatmap overlay (bbox will still be visible)
                    print(f"Warning: Could not resize RTM energy map for {img_base_name}: {e}")
                    pass

            # --- Verification: Draw original and reprojected keypoints ---
            # This helps verify that the coordinate transformation is correct.
            # If correct, the cyan 'x' should be directly on top of the white circle.

            # 1. Re-calculate scale and padding (same logic as in `sample_feat_for_joint`)
            target_h, target_w = 288, 384
            scale = min(target_w / bbox_w, target_h / bbox_h)
            new_w, new_h = int(bbox_w * scale), int(bbox_h * scale)
            pad_x = (target_w - new_w) / 2
            pad_y = (target_h - new_h) / 2

            # 2. Draw the grid cells corresponding to the joints of interest
            for j, name, c in zip(JOINTS_OF_INTEREST, JOINT_NAMES, colors):
                x, y = keypoints_2d[j]
                
                # Forward transform to find which grid cell (u,v) the keypoint falls into
                kp_x_model_input = (x - x1) * scale + pad_x
                kp_y_model_input = (y - y1) * scale + pad_y
                u = (kp_x_model_input / target_w) * (Wrtm - 1)
                v = (kp_y_model_input / target_h) * (Hrtm - 1)
                
                # Get the integer grid cell coordinates
                grid_u, grid_v = int(u), int(v)

                # Define the corners of this grid cell in the MODEL INPUT space (288x384)
                # Handle division by zero if Wrtm or Hrtm is 1
                cell_top_left_model_x = (grid_u / (Wrtm - 1)) * target_w if Wrtm > 1 else 0
                cell_top_left_model_y = (grid_v / (Hrtm - 1)) * target_h if Hrtm > 1 else 0
                cell_bot_right_model_x = ((grid_u + 1) / (Wrtm - 1)) * target_w if Wrtm > 1 else target_w
                cell_bot_right_model_y = ((grid_v + 1) / (Hrtm - 1)) * target_h if Hrtm > 1 else target_h
                
                # Backward transform the cell corners to the ORIGINAL IMAGE space
                cell_x1_img = x1 + (cell_top_left_model_x - pad_x) / scale
                cell_y1_img = y1 + (cell_top_left_model_y - pad_y) / scale
                cell_x2_img = x1 + (cell_bot_right_model_x - pad_x) / scale
                cell_y2_img = y1 + (cell_bot_right_model_y - pad_y) / scale
                
                cell_w_img = cell_x2_img - cell_x1_img
                cell_h_img = cell_y2_img - cell_y1_img

                # Draw the rectangle for the grid cell
                rect = Rectangle((cell_x1_img, cell_y1_img), cell_w_img, cell_h_img,
                                 linewidth=2, edgecolor=c, facecolor='none', alpha=0.8)
                axes[ax_idx].add_patch(rect)

            # 3. Draw original keypoints (white circles)
            for j, (x, y) in enumerate(keypoints_2d):
                if j in JOINTS_OF_INTEREST:
                    axes[ax_idx].scatter(x, y, s=80, c='white', edgecolor='black', linewidths=1.5, label='Original' if j == 13 else None, zorder=4)

            # 4. Calculate reprojected keypoints and draw them (cyan 'x')
            reprojected_kpts = []
            for j, (x, y) in enumerate(keypoints_2d):
                # Forward transform: image -> model input -> feature grid
                kp_x_model_input = (x - x1) * scale + pad_x
                kp_y_model_input = (y - y1) * scale + pad_y
                u = (kp_x_model_input / target_w) * (Wrtm - 1)
                v = (kp_y_model_input / target_h) * (Hrtm - 1)

                # Backward transform: feature grid -> model input -> image
                x_reprojected = x1 + ((u / (Wrtm - 1)) * target_w - pad_x) / scale
                y_reprojected = y1 + ((v / (Hrtm - 1)) * target_h - pad_y) / scale
                reprojected_kpts.append((x_reprojected, y_reprojected))

                if j in JOINTS_OF_INTEREST:
                    axes[ax_idx].scatter(x_reprojected, y_reprojected, s=90, marker='x', c='cyan', linewidths=2.0, label='Reprojected' if j == 13 else None, zorder=5)
            
            # 5. Draw feature grid cells for joints of interest (dashed colored boxes)
            for j, name, c in zip(JOINTS_OF_INTEREST, JOINT_NAMES, colors):
                x, y = keypoints_2d[j]
                
                # Forward transform to get grid coordinates
                kp_x_model_input = (x - x1) * scale + pad_x
                kp_y_model_input = (y - y1) * scale + pad_y
                u = (kp_x_model_input / target_w) * (Wrtm - 1)
                v = (kp_y_model_input / target_h) * (Hrtm - 1)
                
                # Get integer grid cell indices
                grid_u = int(np.floor(u))
                grid_v = int(np.floor(v))
                
                # Clamp to valid range
                grid_u = np.clip(grid_u, 0, Wrtm - 1)
                grid_v = np.clip(grid_v, 0, Hrtm - 1)
                
                # Map grid cell corners back to image space
                # Top-left corner of grid cell
                u0_norm = grid_u / (Wrtm - 1) if Wrtm > 1 else 0
                v0_norm = grid_v / (Hrtm - 1) if Hrtm > 1 else 0
                x0_model = u0_norm * target_w
                y0_model = v0_norm * target_h
                x0_img = x1 + (x0_model - pad_x) / scale
                y0_img = y1 + (y0_model - pad_y) / scale
                
                # Bottom-right corner of grid cell
                u1_norm = (grid_u + 1) / (Wrtm - 1) if grid_u < Wrtm - 1 else u0_norm
                v1_norm = (grid_v + 1) / (Hrtm - 1) if grid_v < Hrtm - 1 else v0_norm
                x1_model = u1_norm * target_w
                y1_model = v1_norm * target_h
                x1_img_corner = x1 + (x1_model - pad_x) / scale
                y1_img_corner = y1 + (y1_model - pad_y) / scale
                
                # Draw grid cell rectangle
                cell_w = x1_img_corner - x0_img
                cell_h = y1_img_corner - y0_img
                rect = Rectangle((x0_img, y0_img), cell_w, cell_h,
                               linewidth=2, edgecolor=c, facecolor='none', linestyle='--', alpha=0.7, zorder=3)
                axes[ax_idx].add_patch(rect)

        else: # if rtm_bbox is None
            # Fallback for when there's no bbox: just draw keypoints
            for j, name, c in zip(JOINTS_OF_INTEREST, JOINT_NAMES, colors):
                x, y = keypoints_2d[j]
                axes[ax_idx].scatter(x, y, c=c, s=80, marker='x', linewidths=3)

        title = f"RTM Features ({Hrtm}x{Wrtm})\nL2-norm energy"
        if rtm_bbox is None:
            title += "\nNo bbox (using full-image fallback)"
        axes[ax_idx].set_title(title, fontsize=10)
        axes[ax_idx].set_xlim(0, Wimg)
        axes[ax_idx].set_ylim(Himg, 0)
        if rtm_bbox is not None:
            # Manually create legend to avoid duplicate labels
            handles, labels = axes[ax_idx].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            axes[ax_idx].legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=8)
        ax_idx += 1
    
    # --- DAV2 Feature Map Visualization
    if has_dav2:
        dav2_feat = depth_data['dav2_feature_256']
        if dav2_feat.ndim == 4:
            dav2_feat = dav2_feat[0]
        _, Hdav, Wdav = dav2_feat.shape
        
        # Compute L2-norm energy across channels (shows where features are strong)
        dav2_energy = np.linalg.norm(dav2_feat, axis=0)  # (Hdav, Wdav)
        dav2_energy = (dav2_energy - dav2_energy.min()) / (dav2_energy.ptp() + 1e-6)
        
        # Visualize energy map
        axes[ax_idx].imshow(dav2_energy, cmap='plasma', extent=[0, Wimg, Himg, 0], alpha=0.6)
        axes[ax_idx].imshow(img, alpha=0.3)
        
        for j, name, c in zip(JOINTS_OF_INTEREST, JOINT_NAMES, colors):
            x, y = keypoints_2d[j]
            # Calculate grid coordinates
            u = (x / Wimg) * (Wdav - 1)
            v = (y / Himg) * (Hdav - 1)
            
            # Draw grid cell boundaries
            cell_w = Wimg / Wdav
            cell_h = Himg / Hdav
            grid_x = int(u) * cell_w
            grid_y = int(v) * cell_h
            
            rect = Rectangle((grid_x, grid_y), cell_w, cell_h, 
                           linewidth=2, edgecolor=c, facecolor='none')
            axes[ax_idx].add_patch(rect)
            axes[ax_idx].scatter(x, y, c=c, s=80, marker='x', linewidths=3)
        
        axes[ax_idx].set_title(f"DAV2 Features ({Hdav}×{Wdav})\nL2-norm energy", fontsize=10)
        axes[ax_idx].set_xlim(0, Wimg)
        axes[ax_idx].set_ylim(Himg, 0)
        ax_idx += 1
    
    plt.tight_layout()
    save_path = os.path.join(DEBUG_SAVE_DIR, f"{img_base_name}_patch_vis.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved debug visualization: {save_path}")


SAVE_FULL_CASP = True  # Set to True to save full casp_descriptors, False for summary only
CACHE_RTM_FEATURES = True  # Set to True to cache RTMPose detection feature maps (per-joint 192-D)
CACHE_DAV2_FEATURES = True  # Set to True to cache DepthAnythingV2 feature maps (per-joint 256-D)

# Debug visualization settings
DEBUG_PATCH_VIS = True  # Set to True to visualize patch extraction for debugging
DEBUG_SAVE_DIR = "./debug_patch_vis_h36m/"
DEBUG_NUM_SAMPLES = 100  # Number of random samples to visualize (sampled before main loop)

# ============================================================================
# FEATURE SAMPLING COORDINATE SOURCE
# ============================================================================
# Controls whether to sample RTM/DAV2 features at GT or detected 2D keypoint locations
# False (recommended): Sample at DETECTED keypoints → train/test consistency
# True (debugging only): Sample at GT keypoints → may cause train/test mismatch
USE_GT_KEYPOINTS_FOR_SAMPLING = False

# H36M images live in subfolders named like: s_01_act_02_subact_01_ca_01/...
IMG_DIR = "/srv/essa-lab/flash3/nwarner30/pose_estimation/h36m_data/images/images"

# ============================================================================
# COMMAND LINE FLAGS
# ============================================================================
parser = argparse.ArgumentParser(description='Merge H36M annotations with depth predictions')
parser.add_argument('--split', type=str, default='train', choices=['train', 'val'], 
                    help='Dataset split to process (train or val)')
parser.add_argument('--debug-mode', action='store_true', 
                    help='Run in single-process mode for debugging (disables multiprocessing)')
parser.add_argument('--debug-mini', action='store_true',
                    help='Run mini test with only first 50 samples')
parser.add_argument('--sampling-mode', type=str, default='nearest_4', 
                    choices=['nearest_1', 'nearest_2', 'nearest_4', 'bilinear'],
                    help='Feature sampling strategy')
args = parser.parse_args()

DEBUG_MINI = args.debug_mini
DEBUG_MODE = args.debug_mode
SPLIT = args.split
FEATURE_SAMPLING_MODE = args.sampling_mode  # Set from command line argument

# ============================================================================
# SPLIT-SPECIFIC PATHS
# ============================================================================
if SPLIT == 'train':
    #h36m train aug
    original_npz_path = '/srv/essa-lab/flash3/nwarner30/pose_estimation/h36m_data/annotations/h36m_train_vanilla_h36m_25hz.npz'
    # output_npz_path = '/srv/essa-lab/flash3/nwarner30/pose_estimation/h36m_data/annotations/merged_data_h36m_train_v8_cached_feats.npz'
    output_npz_path = '/srv/essa-lab/flash3/nwarner30/pose_estimation/h36m_data/annotations/merged_data_h36m_train_v10_cached_feats.npz'
    depth_root = '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_outputs_metric_h36m_v10_cache_maps/'
    cache_path = "/srv/essa-lab/flash3/nwarner30/pose_estimation/depth_file_map_h36m_v10_cached_feats_train.pkl"
elif SPLIT == 'val':
    # Validation set
    original_npz_path = '/srv/essa-lab/flash3/nwarner30/pose_estimation/h36m_data/annotations/h36m_test_vanilla_h36m_25hz.npz'
    # output_npz_path = '/srv/essa-lab/flash3/nwarner30/pose_estimation/h36m_data/annotations/merged_data_h36m_val_v8_cached_feats.npz'
    output_npz_path = '/srv/essa-lab/flash3/nwarner30/pose_estimation/h36m_data/annotations/merged_data_h36m_val_v10_cached_feats.npz'
    depth_root = '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_outputs_metric_h36m_v10_cache_maps/'
    cache_path = "/srv/essa-lab/flash3/nwarner30/pose_estimation/depth_file_map_h36m_v10_cached_feats_train.pkl"

print(f"[*] Processing {SPLIT} split")
print(f"   Input NPZ: {original_npz_path}")
print(f"   Output NPZ: {output_npz_path}")
print(f"   Depth root: {depth_root}")
print(f"   Cache path: {cache_path}")
if DEBUG_MINI:
    print(f"   [DEBUG] DEBUG_MINI mode: Processing only first 50 samples")
if DEBUG_MODE:
    print(f"   [DEBUG] DEBUG_MODE: Single-process execution (no multiprocessing)")
print()


# Legacy paths (commented out)
# output_npz_path = '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_h36m_train_v4_correct_sampling.npz'
# depth_root = '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_outputs_metric_h36m_v3'
# output_npz_path = '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_h36m_train_v5_gt_dav.npz'
# output_npz_path = '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_h36m_train_v4_highres.npz'
# output_npz_path = '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_h36m_train_v6_casp.npz'
# depth_root = '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_outputs_metric_h36m_v5_gtdav/'
# depth_root = '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_outputs_metric_h36m_v4_correct_highres/'
# depth_root = '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_outputs_metric_h36m_v6_casp/'
# cache_path = "/srv/essa-lab/flash3/nwarner30/pose_estimation/depth_file_map_h36m_v5.pkl"
# cache_path = "/srv/essa-lab/flash3/nwarner30/pose_estimation/depth_file_map_h36m_v3.pkl"
# cache_path = "/srv/essa-lab/flash3/nwarner30/pose_estimation/depth_file_map_h36m_v4_highres.pkl"
# cache_path = "/srv/essa-lab/flash3/nwarner30/pose_estimation/depth_file_map_h36m_v6_casp_val.pkl"

# SPLIT = 'val' # Uncomment to process validation set
# original_npz_path = '/srv/essa-lab/flash3/nwarner30/pose_estimation/h36m_data/annotations/h36m_test_vanilla_h36m_25hz.npz'
# output_npz_path = '/srv/essa-lab/flash3/nwarner30/pose_estimation/h36m_data/annotations/merged_data_h36m_val_v7_cached_feats.npz'
# depth_root = '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/h36m_val_v7_cached_feats/'
# cache_path = "/srv/essa-lab/flash3/nwarner30/pose_estimation/depth_file_map_h36m_v7_cached_feats_val.pkl"
# Legacy val paths:
# output_npz_path = '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_h36m_val_v4_correct_sampling.npz'
# depth_root = '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_outputs_metric_h36m_v3'
# output_npz_path = '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_h36m_val_v5_gt_dav.npz'
# depth_root = '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_outputs_metric_h36m_v6_casp/'
# depth_root = '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-OfficialV2/Depth-Anything-V2/metric_depth/test_outputs_metric_h36m_v6_casp/'


# Load and MATERIALIZE arrays, then close the npz (avoid lazy members in child procs)
_npz = np.load(original_npz_path, allow_pickle=True)

# Materialize all arrays into memory for process-safe access
ORIGINAL_DATA_DICT = {key: np.array(_npz[key]).copy() for key in _npz.files}
IMGNAMES = ORIGINAL_DATA_DICT['imgname']  # (N,)
PART_2D = ORIGINAL_DATA_DICT['part'][..., :2].copy()  # (N, 17, 2) - GT keypoints
VIS_1 = ORIGINAL_DATA_DICT['part'][..., 2:].copy()    # (N, 17, 1) - visibility

_npz.close()  # IMPORTANT: close the lazy npz handle

# Keep original_data reference for compatibility with later code
original_data = ORIGINAL_DATA_DICT

# Initialize an empty dictionary for new_data_map
new_data_map = {}

# ipdb> print(list(original_data.keys()))
# ['imgname', 'center', 'scale', 'part', 'S', 'ordinal_depths', 'bone_lengths', 'camera_params_array']
# ipdb> 


num_samples = len(original_data['S'])

# # Loop through all entries to calculate overall accuracy and bucket counts
# for i in range(num_samples):
#     ipdb.set_trace()
#     individual_keypoints_2d = original_data['part'][i]
#     individual_keypoints_3d = original_data['S'][i]
#     individual_imgname = original_data['imgname'][i]

#     # from our caches, h36m: s_01_act_02_subact_01_ca_01_000002_depth.npz
#     # from here: '1_2_1_1_img_000001.jpg'

#     # Compute ground truth ordinal depths
#     # gt_ordinal_depths = individual_keypoints_3d...

#     # Extract the stored prediction
#     pred_ordinal_depths = # find equivalency in our npz data.
#     # Equivalency in naming format.
#     # Maybe extract them to a central directory, then load each npz into this one.
    

#     # Later 2DO; compute accuracy of estimates (OD at least %, against the GT.)



# ============================================================================
# LOAD DEPTH FILE MAP - Must be at module level for multiprocessing
# ============================================================================
# Check if cached depth file map exists
if os.path.exists(cache_path):
    print("Loading cached depth file map...")
    with open(cache_path, "rb") as f:
        depth_file_map = pickle.load(f)
else:
    print("Building depth file map...")
    depth_file_map = {}
    for job_dir in tqdm(os.listdir(depth_root)):
        job_path = os.path.join(depth_root, job_dir)
        if os.path.isdir(job_path):
            for depth_file in glob(os.path.join(job_path, "*.npz")):
                base_name = os.path.basename(depth_file).replace("_depth.npz", "")
                depth_file_map[base_name] = depth_file
    
    # Save to cache
    with open(cache_path, "wb") as f:
        pickle.dump(depth_file_map, f)
    print("Depth file map cached successfully.")


# Function to process a single sample
def process_sample(args, debug_vis_indices=None):
    """Process a single sample.
    
    Args:
        args: Tuple of (i, individual_imgname)
        debug_vis_indices: Set of indices to generate debug visualizations for (optional)
    """
    i, individual_imgname = args  # Unpack passed data

    # Convert '1_2_1_1_img_000001.jpg' → 's_01_act_02_subact_01_ca_01_000001'
    img_parts = individual_imgname.replace(".jpg", "").split("_")
    # img_base_name = f"s_{int(img_parts[0]):02d}_act_{int(img_parts[1]):02d}_subact_{int(img_parts[2]):02d}_ca_{int(img_parts[3]):02d}_{img_parts[5]}"
    # corrected sampling factors here
    img_base_name = f"s_{int(img_parts[0]):02d}_act_{int(img_parts[1]):02d}_subact_{int(img_parts[2]):02d}_ca_{int(img_parts[3]):02d}_{int(img_parts[5])*2 - 1:06d}"

    # ipdb.set_trace()
    # Match image name to depth file
    if img_base_name in depth_file_map:
            try:
                depth_data = np.load(depth_file_map[img_base_name], allow_pickle=True)
            except Exception as e:
                print(f"Skipping corrupted depth file: {depth_file_map[img_base_name]} ({e})")
                return None, None, None, None, None, None, None, None
            predicted_da_depth = depth_data['keypoints_depth']
            predicted_keypoints_score = depth_data['keypoints_score']
            predicted_keypoints = depth_data['keypoints']
            
            # Check for CASP fields (backwards compatible)
            casp_descriptors = depth_data['casp_descriptors'] if 'casp_descriptors' in depth_data else None
            summary_casp_descriptor_10d = depth_data['summary_casp_descriptor_10d'] if 'summary_casp_descriptor_10d' in depth_data else None
            
            # Extract per-joint features if caching is enabled
            rtm_joint_feats = None
            dav2_joint_feats = None
            
            # Check if any caching is enabled and required fields exist
            needs_processing = (CACHE_RTM_FEATURES and 'rtm_feature_coarse' in depth_data) or \
                              (CACHE_DAV2_FEATURES and 'dav2_feature_256' in depth_data)
            
            if needs_processing:
                # ============================================================================
                # Choose keypoint source for feature sampling based on flag
                # ============================================================================
                gt_keypoints_2d = PART_2D[i]  # (17, 2) - GT x,y coordinates (for violation tracking)
                
                if USE_GT_KEYPOINTS_FOR_SAMPLING:
                    # Use GT keypoints for feature sampling (debugging only - causes train/test mismatch!)
                    if i == 0:  # Only print once at the start
                        print("WARNING: Using GT keypoints for feature sampling (USE_GT_KEYPOINTS_FOR_SAMPLING=True)")
                        print("   This creates train/test mismatch and should only be used for debugging!")
                    keypoints_2d = gt_keypoints_2d
                else:
                    # Use DETECTED keypoints for feature sampling (recommended for train/test consistency)
                    keypoints_2d = predicted_keypoints  # Already loaded from depth_data['keypoints']
                
                # Load and validate RTM features if caching is enabled
                rtm_coarse = None
                rtm_bbox = None
                Hrtm, Wrtm = None, None
                if CACHE_RTM_FEATURES and 'rtm_feature_coarse' in depth_data:
                    rtm_coarse = depth_data['rtm_feature_coarse']    # (1, 1024, 12, 9) or (1024, 12, 9)
                    # Remove batch dimension if present
                    if rtm_coarse.ndim == 4:
                        rtm_coarse = rtm_coarse[0]
                    # Validate channel count (RTMPose backbone has 1024 channels, not 192)
                    if rtm_coarse.shape[0] != 1024:
                        print(f"WARNING: Unexpected RTM channels {rtm_coarse.shape[0]}, expected 1024")
                        rtm_coarse = None
                    else:
                        _, Hrtm, Wrtm = rtm_coarse.shape
                    
                    # Load bbox coordinates (required for RTMPose features!)
                    if 'rtm_bbox' in depth_data:
                        rtm_bbox = depth_data['rtm_bbox']  # [x1, y1, x2, y2]
                    else:
                        print(f"WARNING: rtm_bbox not found for {img_base_name}, RTM features will be incorrect!")
                        rtm_coarse = None  # Can't use features without bbox
                
                # Load and validate DAV2 features if caching is enabled
                dav2_coarse = None
                Hdav, Wdav = None, None
                if CACHE_DAV2_FEATURES and 'dav2_feature_256' in depth_data:
                    dav2_coarse = depth_data['dav2_feature_256']     # (1, 256, 37, 66) or (256, 37, 66)
                    # Remove batch dimension if present
                    if dav2_coarse.ndim == 4:
                        dav2_coarse = dav2_coarse[0]
                    # Validate channel count
                    if dav2_coarse.shape[0] != 256:
                        print(f"WARNING: Unexpected DAV2 channels {dav2_coarse.shape[0]}, expected 256")
                        dav2_coarse = None
                    else:
                        _, Hdav, Wdav = dav2_coarse.shape
                
                # Get actual image size - try multiple sources in priority order
                Himg, Wimg = None, None
                
                # 1. Try to get from original_img_shape if stored in depth_data
                if 'original_img_shape' in depth_data:
                    img_shape = depth_data['original_img_shape']
                    if len(img_shape) >= 2:
                        Himg, Wimg = img_shape[0], img_shape[1]
                
                # 2. Try to load actual image from disk to get true dimensions (for first 500 samples)
                if (Himg is None or Wimg is None) and i < 500:
                    try:
                        import cv2
                        img_root = '/srv/essa-lab/flash3/nwarner30/pose_estimation/h36m_data/images/images'
                        
                        # Example: img_base_name = 's_01_act_02_subact_01_ca_01_000001'
                        # Directory structure: .../images/images/s_01_act_02_subact_01_ca_01/
                        parts = img_base_name.split('_')
                        seq = '_'.join(parts[:-1])   # 's_01_act_02_subact_01_ca_01'
                        frame = parts[-1]             # '000001'
                        
                        img_path = os.path.join(img_root, seq, f"{seq}_{frame}.jpg")
                        if os.path.exists(img_path):
                            img = cv2.imread(img_path)
                            if img is not None:
                                Himg, Wimg = img.shape[:2]
                    except:
                        pass  # Silently continue to fallbacks
                
                # 3. Try to get from depth map dimensions
                if Himg is None or Wimg is None:
                    if 'depth_map' in depth_data:
                        depth_map = depth_data['depth_map']
                        if depth_map.ndim == 3:  # (1, H, W)
                            depth_map = depth_map[0]
                        Himg, Wimg = depth_map.shape
                
                # 4. Fallback: H36M standard image size (1000x1000)
                if Himg is None or Wimg is None:
                    Himg, Wimg = 1000, 1000
                    if i < 10:  # Only warn for first 10 samples to avoid spam
                        print(f"WARNING: Using default image size (1000x1000) for {img_base_name}")
                
                # Track keypoint bound violations (don't print each one, just record)
                x_exceeds = keypoints_2d[:, 0] - Wimg
                y_exceeds = keypoints_2d[:, 1] - Himg
                max_x_exceed = np.max(x_exceeds[x_exceeds >= 0]) if np.any(x_exceeds >= 0) else 0
                max_y_exceed = np.max(y_exceeds[y_exceeds >= 0]) if np.any(y_exceeds >= 0) else 0
                
                violation = None
                if max_x_exceed > 0 or max_y_exceed > 0:
                    violation = (img_base_name, float(max_x_exceed), float(max_y_exceed), Wimg, Himg)
                
                # Sample RTM features for each joint (if enabled)
                # CRITICAL: RTM features are from CROPPED bbox regions, must pass bbox for coordinate transform!
                if rtm_coarse is not None:
                    rtm_feats_list = []
                    for kp in keypoints_2d:
                        rtm_feat = sample_feat_for_joint(rtm_coarse, kp, Hrtm, Wrtm, Himg, Wimg, bbox=rtm_bbox)
                        rtm_feats_list.append(rtm_feat)
                    # Stack to (17, C) array and convert to float16 for compression
                    rtm_joint_feats = np.stack(rtm_feats_list, axis=0).astype(np.float16)   # (17, 1024)
                
                # Sample DAV2 features for each joint (if enabled)
                if dav2_coarse is not None:
                    dav2_feats_list = []
                    for kp in keypoints_2d:
                        dav2_feat = sample_feat_for_joint(dav2_coarse, kp, Hdav, Wdav, Himg, Wimg)
                        dav2_feats_list.append(dav2_feat)
                    # Stack to (17, C) array and convert to float16 for compression
                    dav2_joint_feats = np.stack(dav2_feats_list, axis=0).astype(np.float16) # (17, 256)
                
                # Debug visualization for random samples
                if DEBUG_PATCH_VIS and debug_vis_indices is not None and i in debug_vis_indices:
                    # Try to infer H36M image directory (adjust path as needed)
                    # img_dir = None  # Set to actual H36M image directory if available
                    # visualize_joint_patches(depth_data, keypoints_2d, Himg, Wimg, img_base_name, img_dir)
                    visualize_joint_patches(depth_data, keypoints_2d, Himg, Wimg, img_base_name, IMG_DIR)
        
    else:
        predicted_da_depth = None
        predicted_keypoints_score = None
        predicted_keypoints = None
        casp_descriptors = None
        summary_casp_descriptor_10d = None
        rtm_joint_feats = None
        dav2_joint_feats = None
        violation = None

    return predicted_da_depth, predicted_keypoints_score, predicted_keypoints, casp_descriptors, summary_casp_descriptor_10d, rtm_joint_feats, dav2_joint_feats, violation


# ============================================================================
# MAIN EXECUTION - Must be protected for spawn multiprocessing
# ============================================================================
if __name__ == '__main__':
    # Apply DEBUG_MINI mode if enabled
    num_to_process = min(50, num_samples) if DEBUG_MINI else num_samples
    
    # Create debug directory if visualization is enabled
    if DEBUG_PATCH_VIS:
        os.makedirs(DEBUG_SAVE_DIR, exist_ok=True)
        print(f" Debug visualization enabled - saving to {DEBUG_SAVE_DIR}")
    
    # Select random indices for debug visualization
    debug_vis_indices = None
    if DEBUG_PATCH_VIS and DEBUG_NUM_SAMPLES > 0:
        np.random.seed(42)  # For reproducibility
        debug_vis_indices = set(np.random.choice(num_to_process, min(DEBUG_NUM_SAMPLES, num_to_process), replace=False))
        print(f" Selected {len(debug_vis_indices)} random samples for debug visualization")
    
    # Prepare list of (index, imgname) tuples with a progress bar
    task_list = [(i, IMGNAMES[i]) for i in tqdm(range(num_to_process), desc="Preparing tasks")]

    # # Prepare list of (index, imgname) tuples before multiprocessing
    # task_list = [(i, original_data['imgname'][i]) for i in tqdm(range(num_samples))]

    # Use multiprocessing to parallelize or single-process for debugging
    if DEBUG_MODE:
        print("Matching in single process mode for debugging...")
        results = []
        for task in tqdm(task_list, desc="Processing"):
            results.append(process_sample(task, debug_vis_indices=debug_vis_indices))  # Pass debug_vis_indices
    else:
        print("Matching with multiprocessing...")
        # Import here to avoid issues with spawn
        from tqdm.contrib.concurrent import process_map
        # Create partial function with debug_vis_indices
        from functools import partial
        process_fn = partial(process_sample, debug_vis_indices=debug_vis_indices)
        results = process_map(process_fn, task_list, max_workers=8, chunksize=1)


    # Initialize original_data_dict with all data from original_data except 'imgname' (if you need to copy other data)
    # original_data_dict = {key: original_data[key] for key in original_data.files}
    # IMPORTANT: Slice all arrays to match num_to_process when using DEBUG_MINI
    original_data_dict = {}
    for key, val in original_data.items():
        if isinstance(val, np.ndarray) and len(val) == num_samples:
            # Slice arrays that match the original dataset size
            original_data_dict[key] = val[:num_to_process]
        else:
            # Keep non-array data or arrays with different sizes as-is
            original_data_dict[key] = val


    # Unpack results - now expecting 8 return values (with optional violation)
    unpacked = list(zip(*results))
    predicted_da_depths, predicted_keypoints_scores, predicted_keypoints_list, casp_descriptors_list, summary_casp_descriptor_10d_list, rtm_joint_feats_list, dav2_joint_feats_list, violations = unpacked
    # Filter out None violations
    bound_violations = [v for v in violations if v is not None]

    # Print bound violation summary
    if bound_violations:
        print(f"\n  Keypoint Bound Violations Summary:")
        print(f"   Total samples with violations: {len(bound_violations)}")
        max_x_violation = max(v[1] for v in bound_violations)
        max_y_violation = max(v[2] for v in bound_violations)
        print(f"   Max X exceed: {max_x_violation:.2f} pixels")
        print(f"   Max Y exceed: {max_y_violation:.2f} pixels")
        if len(bound_violations) <= 5:
            for img_name, x_ex, y_ex, w, h in bound_violations:
                print(f"     - {img_name}: x_exceed={x_ex:.2f}, y_exceed={y_ex:.2f} (image: {w}x{h})")
        else:
            print(f"   (Showing first 5 violations)")
            for img_name, x_ex, y_ex, w, h in bound_violations[:5]:
                print(f"     - {img_name}: x_exceed={x_ex:.2f}, y_exceed={y_ex:.2f} (image: {w}x{h})")
        print()
    # ipdb.set_trace()

    none_indices = {
        "depth": [i for i, depth in enumerate(predicted_da_depths) if depth is None],
        "keypoints_score": [i for i, score in enumerate(predicted_keypoints_scores) if score is None],
        "keypoints": [i for i, keypoints in enumerate(predicted_keypoints_list) if keypoints is None],
    }

    print(f"None values found: { {k: len(v) for k, v in none_indices.items()} }")


    # Define default values based on expected shapes
    depth_shape = (17,)  # Replace with the actual expected shape
    keypoints_score_shape = (17,)  # Replace with actual expected shape
    keypoints_shape = (17, 2)  # Replace with actual expected shape

    default_depth = np.zeros(depth_shape)
    default_keypoints_score = np.zeros(keypoints_score_shape)
    default_keypoints = np.zeros(keypoints_shape)    # Replace None values with default values
    predicted_da_depths = np.array([depth if depth is not None else default_depth for depth in predicted_da_depths])
    # predicted_keypoints_scores = np.array([score if score is not None else default_keypoints_score for score in predicted_keypoints_scores])
    # predicted_keypoints_scores = np.array([
    #     np.zeros(17) if (x is None or (isinstance(x, np.ndarray) and x.item() is None)) else x.item()
    #     for x in predicted_keypoints_scores
    # ])
    # fixed code
    default_keypoints = np.zeros(keypoints_shape)

    # ipdb.set_trace()
    # Replace None values with default values
    predicted_da_depths = np.array([depth if depth is not None else default_depth for depth in predicted_da_depths])
    # predicted_keypoints_scores = np.array([score if score is not None else default_keypoints_score for score in predicted_keypoints_scores])
    # predicted_keypoints_scores = np.array([
    #     np.zeros(17) if (x is None or (isinstance(x, np.ndarray) and x.item() is None)) else x.item()
    #     for x in predicted_keypoints_scores
    # ])
    # fixed code
    predicted_keypoints_scores = np.array([
        default_keypoints_score if x is None else x
        for x in predicted_keypoints_scores
    ])


    predicted_keypoints_list = np.array([keypoints if keypoints is not None else default_keypoints for keypoints in predicted_keypoints_list])

    # ipdb.set_trace()

    original_data_dict['predicted_da_depth'] = predicted_da_depths
    original_data_dict['predicted_keypoints_score'] = predicted_keypoints_scores
    original_data_dict['predicted_keypoints'] = predicted_keypoints_list

    # Check if CASP data exists and add to output if present
    has_casp_data = any(x is not None for x in casp_descriptors_list)
    if has_casp_data:
        print(" CASP data detected - adding casp_descriptors and summary_casp_descriptor_10d fields")
        
        # Convert to numpy array, handling None values
        # For casp_descriptors: each element is a list of 17 dicts (one per joint)
        # Result should be (N,) object array where each element is a list of 17 dicts
        default_casp_list = [None] * 17  # Default for missing data
        casp_descriptors_array = np.array([
            c if c is not None else default_casp_list 
            for c in casp_descriptors_list
        ], dtype=object)
        
        # For summary_casp_descriptor_10d: create (N, 17, 10) array in float16 to save space
        default_summary = np.zeros((17, 10), dtype=np.float16)
        summary_casp_descriptor_10d_array = np.array([
            s if s is not None else default_summary 
            for s in summary_casp_descriptor_10d_list
        ], dtype=np.float16)
        
        # # SKIP casp_descriptors (huge object array) - we only need the compact summary
        # # original_data_dict['casp_descriptors'] = casp_descriptors_array
        # original_data_dict['summary_casp_descriptor_10d'] = summary_casp_descriptor_10d_array
        # print(f"  - summary_casp_descriptor_10d shape: {summary_casp_descriptor_10d_array.shape} (float16)")
        # print(f"    Skipping 'casp_descriptors' (full dicts) to save space - using compact summary only")

        if SAVE_FULL_CASP:
            # Save [depth, x, y] for each percentile, plus person_scale as flat float16 array
            casp_percentiles = ['Q10', 'Q25', 'Q50', 'Q75', 'Q90', 'central']
            def extract_casp_features(joint_dict):
                features = []
                for k in casp_percentiles:
                    depth = joint_dict[k] if joint_dict and k in joint_dict else 0.0
                    xy = joint_dict[k + '_xy'] if joint_dict and (k + '_xy') in joint_dict else [0.0, 0.0]
                    features.extend([depth, xy[0], xy[1]])
                person_scale = joint_dict['person_scale'] if joint_dict and 'person_scale' in joint_dict else 0.0
                features.append(person_scale)
                return np.array(features, dtype=np.float16)

            num_features = len(casp_percentiles) * 3 + 1  # 6*3 + 1 = 19

            casp_descriptors_array = np.array([
                np.array([
                    extract_casp_features(d)
                    for d in (c if c is not None else [None]*17)
                ], dtype=np.float16)
                for c in casp_descriptors_list
            ], dtype=np.float16)  # shape: (N, 17, 19)

            original_data_dict['casp_descriptors'] = casp_descriptors_array
            print(f"  - casp_descriptors shape: {casp_descriptors_array.shape} (float16)")
            print(f"  Saving full casp_descriptors as float16 array")
        else:
            original_data_dict['summary_casp_descriptor_10d'] = summary_casp_descriptor_10d_array
            print(f"  - summary_casp_descriptor_10d shape: {summary_casp_descriptor_10d_array.shape} (float16)")
            print(f"    Skipping 'casp_descriptors' (full dicts) to save space - using compact summary only")
            
        # Verify format of first non-None CASP descriptor
        first_valid = next((c for c in casp_descriptors_list if c is not None), None)
        if first_valid is not None:
            print(f"  - Each casp_descriptor is a list of {len(first_valid)} dicts")
            if len(first_valid) > 0 and first_valid[0] is not None:
                print(f"  - Dict keys: {list(first_valid[0].keys())}")
    else:
        print("  No CASP data found - using legacy depth format only")

    # Check if cached RTM features exist and add to output if enabled
    has_cached_rtm = any(x is not None for x in rtm_joint_feats_list)
    if has_cached_rtm and CACHE_RTM_FEATURES:
        print("Cached RTM feature maps detected - adding rtm_joint_feats field")
        
        # Convert to numpy arrays, handling None values
        default_rtm_feats = np.zeros((17, 1024), dtype=np.float16)  # RTMPose backbone: 1024 channels
        rtm_joint_feats_array = np.array([
            r if r is not None else default_rtm_feats
            for r in rtm_joint_feats_list
        ], dtype=np.float16)  # (N, 17, 1024)
        
        original_data_dict['rtm_joint_feats'] = rtm_joint_feats_array
        original_data_dict['rtm_grid_hw'] = np.array([12, 9], dtype=np.int16)  # [H, W] for RTM features (typical for RTMPose-L)
        
        print(f"  - rtm_joint_feats shape: {rtm_joint_feats_array.shape} (float16)")
        print(f"  - rtm_grid_hw: {original_data_dict['rtm_grid_hw']} (feature map spatial size)")
    else:
        print(" No cached RTM feature maps found (or caching disabled)")

    # Check if cached DAV2 features exist and add to output if enabled
    has_cached_dav2 = any(x is not None for x in dav2_joint_feats_list)
    if has_cached_dav2 and CACHE_DAV2_FEATURES:
        print("Cached DAV2 feature maps detected - adding dav2_joint_feats field")
        
        # Convert to numpy arrays, handling None values
        default_dav2_feats = np.zeros((17, 256), dtype=np.float16)
        dav2_joint_feats_array = np.array([
            d if d is not None else default_dav2_feats
            for d in dav2_joint_feats_list
        ], dtype=np.float16)  # (N, 17, 256)
        
        original_data_dict['dav2_joint_feats'] = dav2_joint_feats_array
        original_data_dict['dav2_grid_hw'] = np.array([37, 66], dtype=np.int16)  # [H, W] for DAV2 features
        
        print(f"  - dav2_joint_feats shape: {dav2_joint_feats_array.shape} (float16)")
        print(f"  - dav2_grid_hw: {original_data_dict['dav2_grid_hw']} (feature map spatial size)")
    else:
        print("No cached DAV2 feature maps found (or caching disabled)")

    # Add suffixes based on what's actually saved
    suffixes = []

    # Add CASP suffix if enabled and data exists
    if has_casp_data and SAVE_FULL_CASP:
        suffixes.append('fullcasp')

    # Add feature cache suffixes based on what was actually cached
    if has_cached_rtm and CACHE_RTM_FEATURES:
        suffixes.append('rtm')
    if has_cached_dav2 and CACHE_DAV2_FEATURES:
        suffixes.append('dav2')
    
    # Add suffix if using GT keypoints for feature sampling (debugging mode)
    if USE_GT_KEYPOINTS_FOR_SAMPLING and (CACHE_RTM_FEATURES or CACHE_DAV2_FEATURES):
        suffixes.append('feat_map_gt_sample')

    # Add sampling mode suffix to distinguish between different strategies
    if CACHE_RTM_FEATURES or CACHE_DAV2_FEATURES:
        suffixes.append(FEATURE_SAMPLING_MODE)

    # Apply suffixes to output path
    if suffixes:
        suffix_str = '_' + '_'.join(suffixes)
        output_npz_path = output_npz_path.replace('.npz', f'{suffix_str}.npz')

    np.savez(output_npz_path, **original_data_dict)
    print(f"Updated data successfully saved to {output_npz_path}")

    # ============================================================================
    # STEP 2: Create a second version with DETECTED keypoints replacing GT in 'part'
    # ============================================================================
    print("\n" + "="*80)
    print("Creating detection-based version (replacing GT 2D with detected 2D)...")
    print("="*80)

    # Make a copy of the data dict
    det_data_dict = original_data_dict.copy()

    # Get GT keypoints to extract visibility (must match the processed subset)
    gt_keypoints = original_data['part'][:num_to_process]  # (num_to_process, 17, 3) where last channel = visibility

    # Copy visibility channel from GT
    visibility = gt_keypoints[:, :, 2:]  # (num_to_process, 17, 1)

    # Concatenate predicted keypoints with GT visibility
    pred_keypoints_with_visibility = np.concatenate([predicted_keypoints_list, visibility], axis=-1)  # (num_to_process, 17, 3)

    # Replace 'part' with detected keypoints
    det_data_dict['part'] = pred_keypoints_with_visibility.copy()

    # Save the detection-based version
    det_output_path = output_npz_path.replace('.npz', '_dets.npz')
    np.savez(det_output_path, **det_data_dict)
    print(f"Detection-based annotations saved to: {det_output_path}")
    print(f"   'part' now contains: detected (x,y) + GT visibility")
    print("="*80 + "\n")

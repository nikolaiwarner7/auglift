import numpy as np
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import ipdb
import os
import pickle
from glob import glob

# Set multiprocessing start method for safety with NumPy/zipfile
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

# ============================================================================
# COORDINATE SYSTEM NOTES:
# ============================================================================
# - 3DPW images: Variable resolution (often 1920x1080, 1080x1920, etc - NON-SQUARE!)
# - RTMPose processes images at their native resolution, outputs 2D keypoints in image coordinates
# - RTMPose feature maps (det_feature_coarse): typically ~20x20 (varies with input size)
# - DepthAnythingV2 processes images, outputs depth maps at input resolution
# - DAV2 feature maps (dav2_feature_256): spatial dims depend on input size
# - All 2D keypoints are in original image coordinates (0 to W-1, 0 to H-1)
# - Bilinear sampling maps image coordinates to feature grid coordinates
# ============================================================================
# FEATURE CACHING FLAGS:
# ============================================================================
# CACHE_RTM_FEATURES:  Cache per-joint RTMPose detection features (17×192-D per frame)
#                      Use for learning appearance/detection-aware representations
#                      File size impact: ~13KB per frame
# CACHE_DAV2_FEATURES: Cache per-joint DepthAnythingV2 features (17×256-D per frame)
#                      Use for learning depth-aware representations
#                      File size impact: ~17KB per frame
# Can enable/disable independently to control NPZ file size
# ============================================================================

SAVE_FULL_CASP = False  # Set to True to save full casp_descriptors, False for summary only
CACHE_RTM_FEATURES = True  # Set to True to cache RTMPose detection feature maps (per-joint 192-D)
CACHE_DAV2_FEATURES = True  # Set to True to cache DepthAnythingV2 feature maps (per-joint 256-D)

# Debug visualization settings
DEBUG_PATCH_VIS = False  # Set to True to visualize patch extraction for debugging (disable after first run)
DEBUG_SAVE_DIR = "./debug_patch_vis_3dpw/"
DEBUG_NUM_SAMPLES = 100  # Number of random samples to visualize (sampled before main loop)

# ============================================================================
# FEATURE SAMPLING COORDINATE SOURCE
# ============================================================================
# Controls whether to sample RTM/DAV2 features at GT or detected 2D keypoint locations
# False (recommended): Sample at DETECTED keypoints → train/test consistency
# True (debugging only): Sample at GT keypoints → may cause train/test mismatch
USE_GT_KEYPOINTS_FOR_SAMPLING = False


def bilinear_sample_numpy(feat_map, u, v):
    """Bilinear sampling for a single point in numpy.
    
    Args:
        feat_map: (C, H, W) numpy array
        u, v: float coordinates in [0, W-1] and [0, H-1]
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
    
    # Get fractional parts
    du = u - u0
    dv = v - v0
    
    # Bilinear interpolation
    w00 = (1 - du) * (1 - dv)
    w01 = (1 - du) * dv
    w10 = du * (1 - dv)
    w11 = du * dv
    
    result = (w00 * feat_map[:, v0, u0] + 
              w01 * feat_map[:, v1, u0] + 
              w10 * feat_map[:, v0, u1] + 
              w11 * feat_map[:, v1, u1])
    
    return result


def sample_feat_for_joint(feat_map, kp, Hf, Wf, Himg, Wimg):
    """Sample features at a keypoint location.
    
    Args:
        feat_map: (C, H, W) feature map
        kp: (2,) keypoint [x, y] in image coordinates
        Hf, Wf: feature map spatial dimensions
        Himg, Wimg: original image dimensions
    Returns:
        (C,) sampled features
    """
    # Map 2D pixel → coarse grid coordinate
    u = (kp[0] / Wimg) * (Wf - 1)
    v = (kp[1] / Himg) * (Hf - 1)
    
    return bilinear_sample_numpy(feat_map, u, v)


def visualize_joint_patches(depth_data, keypoints_2d, Himg, Wimg, img_base_name, img_dir):
    """Visualize hand/foot patch extraction for debugging.
    
    Args:
        depth_data: NPZ data containing feature maps
        keypoints_2d: (17, 2) keypoint coordinates
        Himg, Wimg: Image dimensions
        img_base_name: Base name for the image file
        img_dir: Directory containing 3DPW images
    """
    import matplotlib.pyplot as plt
    import cv2
    from matplotlib.patches import Rectangle
    
    # Choose joints of interest: hands, feet, head (H36M skeleton indices)
    JOINTS_OF_INTEREST = [13, 16, 6, 3, 10]  # L-wrist, R-wrist, L-foot, R-foot, head
    JOINT_NAMES = ["L-wrist", "R-wrist", "L-foot", "R-foot", "Head"]
    colors = ["red", "green", "blue", "cyan", "magenta"]
    
    # Try to load actual image
    img_path = None
    img_root = '/srv/essa-lab/flash3/nwarner30/pose_estimation/3dpw_data/imageFiles'
    if os.path.exists(img_root):
        # 3DPW has two naming conventions:
        # Annotation format: courtyard_relaxOnBench_00_participant0_frame00000.jpg
        # Actual files: courtyard_relaxOnBench_00_image_00000.jpg (without imageSequence subdir!)
        
        # Extract scene name and frame number
        parts = img_base_name.split('_')
        if 'participant' in img_base_name and 'frame' in img_base_name:
            # Format: scene_participant#_frame#####
            participant_idx = next((i for i, p in enumerate(parts) if 'participant' in p), None)
            if participant_idx is not None:
                scene_name = '_'.join(parts[:participant_idx])
                # Extract frame number from last part
                frame_num = parts[-1].replace('frame', '').lstrip('0') or '0'
                frame_num_padded = frame_num.zfill(5)
                # Try format: scene/scene_image_#####.jpg (no imageSequence subfolder)
                candidate = os.path.join(img_root, scene_name, f'{scene_name}_image_{frame_num_padded}.jpg')
                if os.path.exists(candidate):
                    img_path = candidate
    
    if img_path and os.path.exists(img_path):
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        # IMPORTANT: Update Himg, Wimg to actual loaded image dimensions
        Himg, Wimg = img.shape[:2]
    else:
        # Create blank checkerboard using provided dimensions
        img = np.ones((Himg, Wimg, 3), dtype=np.uint8) * 200
        tile_size = 100
        for i in range(0, Himg, tile_size):
            for j in range(0, Wimg, tile_size):
                if (i // tile_size + j // tile_size) % 2 == 0:
                    img[i:i+tile_size, j:j+tile_size] = 220
    
    # Get feature map dimensions
    has_rtm = 'det_feature_coarse' in depth_data and depth_data['det_feature_coarse'] is not None
    has_dav2 = 'dav2_feature_256' in depth_data and depth_data['dav2_feature_256'] is not None
    
    if not has_rtm and not has_dav2:
        print(f"⚠️  No feature maps found for {img_base_name}")
        ipdb.set_trace()
        return
    
    # Create visualization
    num_cols = 1 + int(has_rtm) + int(has_dav2)
    fig, axes = plt.subplots(1, num_cols, figsize=(6*num_cols, 6))
    if num_cols == 1:
        axes = [axes]
    
    ax_idx = 0
    
    # Image + keypoints
    axes[ax_idx].imshow(img)
    for j, name, c in zip(JOINTS_OF_INTEREST, JOINT_NAMES, colors):
        x, y = keypoints_2d[j]
        axes[ax_idx].scatter(x, y, c=c, s=100, marker='o', edgecolors='white', linewidths=2, label=name)
        axes[ax_idx].text(x+40, y-40, name, color=c, fontsize=9, weight='bold',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    axes[ax_idx].set_title(f"Image + Keypoints\n{img_base_name}", fontsize=10)
    axes[ax_idx].legend(loc='upper right', fontsize=8)
    axes[ax_idx].set_xlim(0, Wimg)
    axes[ax_idx].set_ylim(Himg, 0)
    ax_idx += 1
    
    # RTM features
    if has_rtm:
        rtm_feat = depth_data['det_feature_coarse']
        if rtm_feat.ndim == 4:
            rtm_feat = rtm_feat[0]
        _, Hrtm, Wrtm = rtm_feat.shape
        
        rtm_energy = np.linalg.norm(rtm_feat, axis=0)
        rtm_energy = (rtm_energy - rtm_energy.min()) / (rtm_energy.ptp() + 1e-6)
        
        axes[ax_idx].imshow(rtm_energy, cmap='viridis', extent=[0, Wimg, Himg, 0], alpha=0.6)
        axes[ax_idx].imshow(img, alpha=0.3)
        
        for j, name, c in zip(JOINTS_OF_INTEREST, JOINT_NAMES, colors):
            x, y = keypoints_2d[j]
            u = (x / Wimg) * (Wrtm - 1)
            v = (y / Himg) * (Hrtm - 1)
            
            cell_w = Wimg / Wrtm
            cell_h = Himg / Hrtm
            grid_x = int(u) * cell_w
            grid_y = int(v) * cell_h
            
            rect = Rectangle((grid_x, grid_y), cell_w, cell_h, 
                           linewidth=2, edgecolor=c, facecolor='none')
            axes[ax_idx].add_patch(rect)
            axes[ax_idx].scatter(x, y, c=c, s=80, marker='x', linewidths=3)
        
        axes[ax_idx].set_title(f"RTM Features ({Hrtm}×{Wrtm})\nL2-norm energy", fontsize=10)
        axes[ax_idx].set_xlim(0, Wimg)
        axes[ax_idx].set_ylim(Himg, 0)
        ax_idx += 1
    
    # DAV2 features
    if has_dav2:
        dav2_feat = depth_data['dav2_feature_256']
        if dav2_feat.ndim == 4:
            dav2_feat = dav2_feat[0]
        _, Hdav, Wdav = dav2_feat.shape
        
        dav2_energy = np.linalg.norm(dav2_feat, axis=0)
        dav2_energy = (dav2_energy - dav2_energy.min()) / (dav2_energy.ptp() + 1e-6)
        
        axes[ax_idx].imshow(dav2_energy, cmap='plasma', extent=[0, Wimg, Himg, 0], alpha=0.6)
        axes[ax_idx].imshow(img, alpha=0.3)
        
        for j, name, c in zip(JOINTS_OF_INTEREST, JOINT_NAMES, colors):
            x, y = keypoints_2d[j]
            u = (x / Wimg) * (Wdav - 1)
            v = (y / Himg) * (Hdav - 1)
            
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
    print(f"📊 Saved debug visualization: {save_path}")



# Original base annotations (GT only)
# original_npz_path = '/srv/essa-lab/flash3/nwarner30/pose_estimation/data/annotations/3dpw_all.npz'
# original_npz_path = '/srv/essa-lab/flash3/nwarner30/pose_estimation/3dpw_data/processed_mmpose_shards/processed_3dpw_all_v2.npz'
original_npz_path = '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_3dpw_all_v3.npz'

# Output paths
# output_npz_path = '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_3dpw_all_v3.npz'
# depth_root = '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_outputs_metric_3dpw_v3/'
# cache_path = "/srv/essa-lab/flash3/nwarner30/pose_estimation/depth_file_map_3dpw_v3.pkl"

# Active: v6_casp version
# output_npz_path = '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_3dpw_all_v6_casp.npz'
output_npz_path = '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_3dpw_v8_cached_feats.npz'
# depth_root = '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_outputs_metric_3dpw_v6_casp/'
# depth_root = '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_outputs_metric_3dpw_train_v6_casp/'
depth_root = '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_outputs_metric_3dpw_train_v7_debug_oct'
cache_path = "/srv/essa-lab/flash3/nwarner30/pose_estimation/depth_file_map_3dpw_v8_img_features.pkl"

#######################
### v3 run again ###### v3 corrected for rgb.
#######################


# output_npz_path = '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_3dpw_all_v5_det_dav_hr_v2_fix3d.npz'
# depth_root = "/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_outputs_metric_3dpw_train_v5_det_dav_hr/" #fix high res may25


#v6 try outdoor
# output_npz_path = '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_3dpw_all_v6_gtdav.npz'
# depth_root = "/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_outputs_metric_3dpw_train_v6_gtdav/" #fix rgb

# output_npz_path = '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_3dpw_all_v6_det_dav.npz'
# depth_root = "/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_outputs_metric_3dpw_train_v6_det_dav/" #fix rgb


#v5
# output_npz_path = '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_3dpw_all_v5_gtdav.npz'
# depth_root = "/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_outputs_metric_3dpw_train_v5_gtdav/" #fix rgb

# output_npz_path = '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_3dpw_all_v5_det_dav.npz'
# depth_root = "/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_outputs_metric_3dpw_train_v5_det_dav/" #fix rgb


# output_npz_path = '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_3dpw_all_v3.npz'
# depth_root = "/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_outputs_metric_3dpw_v3" #fix rgb
# cache_path = "/srv/essa-lab/flash3/nwarner30/pose_estimation/depth_file_map_3dhp_test.pkl"

# ============================================================================
# LOAD NPZ DATA - MATERIALIZE FOR MULTIPROCESSING SAFETY
# ============================================================================
# Load and MATERIALIZE arrays, then close the npz (avoid lazy members in child procs)
_npz = np.load(original_npz_path, allow_pickle=True)

# Materialize all arrays into memory for process-safe access
ORIGINAL_DATA_DICT = {key: np.array(_npz[key]).copy() for key in _npz.files}
IMGNAMES = ORIGINAL_DATA_DICT['imgname']  # (N,)
GT_KEYPOINTS_ALL = ORIGINAL_DATA_DICT['part'][:, :, :2].copy()  # (N, 17, 2) - GT keypoints for passing to workers

_npz.close()  # IMPORTANT: close the lazy npz handle

# Keep original_data reference for compatibility with later code
original_data = ORIGINAL_DATA_DICT

# Initialize an empty dictionary for new_data_map
new_data_map = {}

# ipdb> print(list(original_data.keys()))
# ['imgname', 'center', 'scale', 'part', 'S', 'ordinal_depths', 'bone_lengths', 'camera_params_array']
# ipdb> 


num_samples = len(original_data['S'])

# Create debug directory if visualization is enabled
if DEBUG_PATCH_VIS:
    os.makedirs(DEBUG_SAVE_DIR, exist_ok=True)
    print(f"🐛 Debug visualization enabled - saving to {DEBUG_SAVE_DIR}")

# ============================================================================
# LOAD DEPTH FILE MAP - Must be at module level for multiprocessing
# ============================================================================
# Check if cached depth file map exists
if os.path.exists(cache_path):
    print(f"Loading cached depth file map from {cache_path}...")
    with open(cache_path, 'rb') as f:
        DEPTH_FILE_MAP = pickle.load(f)
    print(f"Loaded {len(DEPTH_FILE_MAP)} entries from cache")
else:
    print("Building depth file map...")
    DEPTH_FILE_MAP = {}
    for job_dir in tqdm(os.listdir(depth_root)):
        job_path = os.path.join(depth_root, job_dir)
        if os.path.isdir(job_path):
            for depth_file in glob(os.path.join(job_path, "*.npz")):
                base_name = os.path.basename(depth_file).replace("_depth.npz", "")
                DEPTH_FILE_MAP[base_name] = depth_file
    
    # Save cache for future runs
    print(f"Saving depth file map cache to {cache_path}...")
    with open(cache_path, 'wb') as f:
        pickle.dump(DEPTH_FILE_MAP, f)
    print(f"Cached {len(DEPTH_FILE_MAP)} entries")

def parse_filename(filename):
    parts = filename.split('_')
    setting = parts[0]
    action = parts[1]
    vidnum = parts[2]
    participant = parts[3]
    frame = parts[-1].replace('frame', '').replace('.jpg', '')
    return setting, action, vidnum, participant, frame

# Function to process a single sample
def process_sample(args, debug_vis_indices=None):
    """Process a single sample.
    
    Args:
        args: Tuple of (i, individual_imgname, gt_keypoints_2d)
        debug_vis_indices: Set of indices to generate debug visualizations for (optional)
    """
    i, individual_imgname, gt_keypoints_2d = args  # Now receives GT keypoints as argument
    
    img_base_name = individual_imgname.replace(".jpg", "")
    
    # Match image name to depth file
    if img_base_name in DEPTH_FILE_MAP:
        depth_data = np.load(DEPTH_FILE_MAP[img_base_name], allow_pickle=True)
        
        # 🔍 DEBUG: Check first sample only
        if i == 0:
            print(f"\n🔍 DEBUG: First depth file analysis")
            print(f"   File: {DEPTH_FILE_MAP[img_base_name]}")
            print(f"   Keys in file: {list(depth_data.keys())}")
            print(f"   Has 'casp_descriptors': {'casp_descriptors' in depth_data}")
            print(f"   Has 'summary_casp_descriptor_10d': {'summary_casp_descriptor_10d' in depth_data}")
            print(f"   Has 'det_feature_coarse': {'det_feature_coarse' in depth_data}")
            print(f"   Has 'dav2_feature_256': {'dav2_feature_256' in depth_data}")
            if 'casp_descriptors' in depth_data:
                casp_check = depth_data['casp_descriptors']
                print(f"   casp_descriptors type: {type(casp_check)}")
                if casp_check is not None:
                    print(f"   casp_descriptors length: {len(casp_check) if hasattr(casp_check, '__len__') else 'N/A'}")
            if 'summary_casp_descriptor_10d' in depth_data:
                summary_check = depth_data['summary_casp_descriptor_10d']
                print(f"   summary_casp_descriptor_10d type: {type(summary_check)}")
                if summary_check is not None:
                    print(f"   summary_casp_descriptor_10d shape: {summary_check.shape if hasattr(summary_check, 'shape') else 'N/A'}")
            print()
        
        predicted_da_depth = depth_data['keypoints_depth']
        predicted_keypoints_score = depth_data['keypoints_score']
        predicted_keypoints = depth_data['keypoints']
        
        # Check for CASP fields (backwards compatible)
        casp_descriptors = depth_data['casp_descriptors'] if 'casp_descriptors' in depth_data else None
        summary_casp_descriptor_10d = depth_data['summary_casp_descriptor_10d'] if 'summary_casp_descriptor_10d' in depth_data else None
        
        # ============================================================================
        # Debug visualization (independent of feature caching)
        # ============================================================================
        if DEBUG_PATCH_VIS and debug_vis_indices is not None and i in debug_vis_indices:
            # Get image dimensions for visualization
            Himg_vis, Wimg_vis = None, None
            if 'original_img_shape' in depth_data:
                img_shape = depth_data['original_img_shape']
                if len(img_shape) >= 2:
                    Himg_vis, Wimg_vis = img_shape[0], img_shape[1]
            
            if (Himg_vis is None or Wimg_vis is None) and 'depth_map' in depth_data:
                depth_map = depth_data['depth_map']
                if depth_map.ndim == 3:
                    depth_map = depth_map[0]
                Himg_vis, Wimg_vis = depth_map.shape
            
            if Himg_vis is None or Wimg_vis is None:
                Himg_vis, Wimg_vis = 1080, 1920  # Default
            
            visualize_joint_patches(
                depth_data, gt_keypoints_2d, Himg_vis, Wimg_vis, img_base_name,
                '/srv/essa-lab/flash3/nwarner30/pose_estimation/3dpw_data/imageFiles'
            )
        
        # ============================================================================
        # Extract per-joint features from RTM and DAV2 feature maps
        # ============================================================================
        rtm_joint_feats = None
        dav2_joint_feats = None
        
        # Check if any caching is enabled and required fields exist
        needs_processing = (CACHE_RTM_FEATURES and 'det_feature_coarse' in depth_data) or \
                          (CACHE_DAV2_FEATURES and 'dav2_feature_256' in depth_data)
        
        if needs_processing:
            # ============================================================================
            # Choose keypoint source for feature sampling based on flag
            # ============================================================================
            gt_keypoints_2d = original_data['part'][i, :, :2]  # (17, 2) - GT x,y coordinates (for violation tracking)
            
            if USE_GT_KEYPOINTS_FOR_SAMPLING:
                # Use GT keypoints for feature sampling (debugging only - causes train/test mismatch!)
                if i == 0:  # Only print once at the start
                    print("⚠️  WARNING: Using GT keypoints for feature sampling (USE_GT_KEYPOINTS_FOR_SAMPLING=True)")
                    print("   This creates train/test mismatch and should only be used for debugging!")
                keypoints_2d = gt_keypoints_2d
            else:
                # ✅ Use DETECTED keypoints for feature sampling (recommended for train/test consistency)
                keypoints_2d = predicted_keypoints  # Already loaded from depth_data['keypoints']
            
            # Get image dimensions - try multiple sources in priority order
            Himg, Wimg = None, None
            
            # 1. Try to get from original_img_shape if stored in depth_data
            if 'original_img_shape' in depth_data:
                img_shape = depth_data['original_img_shape']
                if len(img_shape) >= 2:
                    Himg, Wimg = img_shape[0], img_shape[1]
            
            # 2. Try to get from depth map dimensions
            if (Himg is None or Wimg is None) and 'depth_map' in depth_data:
                depth_map = depth_data['depth_map']
                if depth_map.ndim == 3:
                    depth_map = depth_map[0]  # Remove batch dim
                Himg, Wimg = depth_map.shape
            
            # 3. Fallback: load actual image to get dimensions
            if Himg is None or Wimg is None:
                try:
                    import cv2
                    # 3DPW path structure: outdoor_climbing_00/imageSequence/outdoor_climbing_00_frame_000006.jpg
                    img_root = '/srv/essa-lab/flash3/nwarner30/pose_estimation/3dpw_data/imageFiles'
                    parts = img_base_name.split('_')
                    if len(parts) >= 5:
                        scene_name = '_'.join(parts[:-2])  # e.g., outdoor_climbing_00
                        candidate = os.path.join(img_root, scene_name, 'imageSequence', f'{img_base_name}.jpg')
                        if os.path.exists(candidate):
                            img = cv2.imread(candidate)
                            if img is not None:
                                Himg, Wimg = img.shape[:2]
                except:
                    pass  # Silently continue
            
            # 4. Last resort: assume common 3DPW resolution (but warn)
            if Himg is None or Wimg is None:
                Himg, Wimg = 1080, 1920  # Common 3DPW resolution
                if i < 10:  # Only warn for first 10 samples
                    print(f"⚠️  Warning: Using default image size ({Himg}x{Wimg}) for {img_base_name}")
            
            # Track keypoint bound violations (for diagnostics) - ALWAYS check DETECTED keypoints
            # Don't use fallback dimensions - only check if we have reliable dimensions
            violation = None
            if 'original_img_shape' in depth_data:
                img_shape = depth_data['original_img_shape']
                if len(img_shape) >= 2:
                    Himg_actual, Wimg_actual = int(img_shape[0]), int(img_shape[1])
                    
                    # Check DETECTED keypoints (not the ones we sample with)
                    x_exceeds = keypoints_2d[:, 0] - Wimg_actual
                    y_exceeds = keypoints_2d[:, 1] - Himg_actual
                    max_x_exceed = np.max(x_exceeds[x_exceeds >= 0]) if np.any(x_exceeds >= 0) else 0
                    max_y_exceed = np.max(y_exceeds[y_exceeds >= 0]) if np.any(y_exceeds >= 0) else 0
                    
                    if max_x_exceed > 0 or max_y_exceed > 0:
                        violation = (img_base_name, float(max_x_exceed), float(max_y_exceed), Wimg_actual, Himg_actual)
            
            # Extract RTM features if requested and available
            if CACHE_RTM_FEATURES and 'det_feature_coarse' in depth_data and Himg is not None:
                rtm_feat = depth_data['det_feature_coarse']
                if rtm_feat.ndim == 4:
                    rtm_feat = rtm_feat[0]  # Remove batch dim: (C, H, W)
                
                # Convert to float32 to prevent overflow during interpolation
                rtm_feat = rtm_feat.astype(np.float32)
                
                # Validate channel count
                if rtm_feat.shape[0] != 192:
                    if i < 5:
                        print(f"⚠️  Warning: Unexpected RTM channels {rtm_feat.shape[0]}, expected 192")
                else:
                    C_rtm, Hrtm, Wrtm = rtm_feat.shape
                    rtm_feats_list = []
                    
                    for kp in keypoints_2d:
                        rtm_feat_sample = sample_feat_for_joint(rtm_feat, kp, Hrtm, Wrtm, Himg, Wimg)
                        rtm_feats_list.append(rtm_feat_sample)
                    
                    # Stack to (17, C) array and convert to float16
                    rtm_joint_feats = np.stack(rtm_feats_list, axis=0).astype(np.float16)
            
            # Extract DAV2 features if requested and available
            if CACHE_DAV2_FEATURES and 'dav2_feature_256' in depth_data and Himg is not None:
                dav2_feat = depth_data['dav2_feature_256']
                if dav2_feat.ndim == 4:
                    dav2_feat = dav2_feat[0]  # Remove batch dim: (C, H, W)
                
                # Convert to float32 to prevent overflow during interpolation
                dav2_feat = dav2_feat.astype(np.float32)
                
                # Validate channel count
                if dav2_feat.shape[0] != 256:
                    if i < 5:
                        print(f"⚠️  Warning: Unexpected DAV2 channels {dav2_feat.shape[0]}, expected 256")
                else:
                    C_dav, Hdav, Wdav = dav2_feat.shape
                    dav2_feats_list = []
                    
                    for kp in keypoints_2d:
                        dav2_feat_sample = sample_feat_for_joint(dav2_feat, kp, Hdav, Wdav, Himg, Wimg)
                        dav2_feats_list.append(dav2_feat_sample)
                    
                    # Stack to (17, C) array and convert to float16
                    dav2_joint_feats = np.stack(dav2_feats_list, axis=0).astype(np.float16)
        
        # Return with optional violation
        return predicted_da_depth, predicted_keypoints_score, predicted_keypoints, casp_descriptors, summary_casp_descriptor_10d, rtm_joint_feats, dav2_joint_feats, violation
    
    else:
        # No depth file found - return None for all fields
        predicted_da_depth = None
        predicted_keypoints_score = None
        predicted_keypoints = None
        casp_descriptors = None
        summary_casp_descriptor_10d = None
        rtm_joint_feats = None
        dav2_joint_feats = None
        violation = None
        
        return predicted_da_depth, predicted_keypoints_score, predicted_keypoints, casp_descriptors, summary_casp_descriptor_10d, rtm_joint_feats, dav2_joint_feats, violation

# np.savez(output_npz_path, **original_data_dict)
# print(f"Updated data successfully saved to {output_npz_path}")

# import multiprocessing as mp
from tqdm.contrib.concurrent import process_map  # For better multiprocessing tqdm integration


# ============================================================================
# MAIN EXECUTION - Must be wrapped for multiprocessing safety
# ============================================================================
if __name__ == "__main__":
    # Extract image names into a list to avoid repeated disk access
    imgnames = original_data['imgname']
    gt_keypoints_all = original_data['part'][:, :, :2]  # (N, 17, 2) - only x,y coords
    num_samples = len(imgnames)
    
    # Select random indices for debug visualization
    debug_vis_indices = None
    if DEBUG_PATCH_VIS and DEBUG_NUM_SAMPLES > 0:
        np.random.seed(42)  # For reproducibility
        debug_vis_indices = set(np.random.choice(num_samples, min(DEBUG_NUM_SAMPLES, num_samples), replace=False))
        print(f"🎲 Selected {len(debug_vis_indices)} random samples for debug visualization")
    
    # Prepare list of (index, imgname, gt_keypoints_2d) tuples
    print("Preparing tasks...")
    task_list = [(i, imgnames[i], gt_keypoints_all[i]) for i in range(num_samples)]
    
    # Use multiprocessing to parallelize
    USE_MULTIPROCESSING = merged_data_3dhp_test_v8_cached_feats_rtm_dav2_dets_fmap_pre_normalized  # Set to False for debugging
    NUM_WORKERS = 8  # Adjust based on your CPU cores
    
    if USE_MULTIPROCESSING:
        print(f"Matching with multiprocessing ({NUM_WORKERS} workers)...")
        # Create partial function with debug_vis_indices
        from functools import partial
        process_fn = partial(process_sample, debug_vis_indices=debug_vis_indices)
        results = process_map(process_fn, task_list, max_workers=NUM_WORKERS, chunksize=100)
    else:
        print("Matching in single process mode for debugging...")
        results = []
        for task in tqdm(task_list, desc="Processing"):
            results.append(process_sample(task, debug_vis_indices=debug_vis_indices))

    # Unpack results (now includes violations as 8th return value)
    predicted_da_depths, predicted_keypoints_scores, predicted_keypoints_list, casp_descriptors_list, summary_casp_descriptor_10d_list, rtm_joint_feats_list, dav2_joint_feats_list, violations_list = zip(*results)

    # Filter out None violations
    bound_violations = [v for v in violations_list if v is not None]
    if bound_violations:
        print(f"\n⚠️  Found {len(bound_violations)} samples with keypoints exceeding image bounds")
        # Print first few violations as examples
        for violation in bound_violations[:5]:
            img_base, x_exceed, y_exceed, W, H = violation
            print(f"   {img_base}: X exceeds by {x_exceed:.1f}px, Y exceeds by {y_exceed:.1f}px (image: {W}×{H})")

    # Initialize original_data_dict with all data from original_data (original_data is already a dict)
    original_data_dict = {key: original_data[key] for key in original_data.keys()}

    # # Unpack results and convert to arrays
    # predicted_da_depths, predicted_keypoints_scores, predicted_keypoints_list = zip(*results)
    # ipdb.set_trace()

    # ipdb.set_trace()

    none_indices = {
        "depth": [i for i, depth in enumerate(predicted_da_depths) if depth is None],
        "keypoints_score": [i for i, score in enumerate(predicted_keypoints_scores) if score is None],
        "keypoints": [i for i, keypoints in enumerate(predicted_keypoints_list) if keypoints is None],
    }

    print(f"None values found: { {k: len(v) for k, v in none_indices.items()} }")

    # ipdb.set_trace()
    # Define default values based on expected shapes
    depth_shape = (17,)  # Replace with the actual expected shape
    keypoints_score_shape = (17,)  # Replace with actual expected shape
    keypoints_shape = (17, 2)  # Replace with actual expected shape

    default_depth = np.zeros(depth_shape)
    default_keypoints_score = np.zeros(keypoints_score_shape)
    default_keypoints = np.zeros(keypoints_shape)

    # # Replace None values with default values
    # predicted_da_depths = np.array([depth if depth is not None else default_depth for depth in predicted_da_depths])
    # predicted_keypoints_scores = np.array([score if score is not None else default_keypoints_score for score in predicted_keypoints_scores])
    # predicted_keypoints_list = np.array([keypoints if keypoints is not None else default_keypoints for keypoints in predicted_keypoints_list])

    # # ipdb.set_trace()

    # Convert lists to arrays, handling None values
    # Convert lists to arrays, handling None values
    default_depth = np.zeros((17,))
    default_keypoints_score = np.zeros((17,))
    default_keypoints = np.zeros((17, 2))

    # NEW: Add default for casp_descriptors (list of 17 dicts)
    default_descriptor_dict = {
        'central': 0.0, 'central_xy': (0, 0),
        'Q10': 0.0, 'Q10_xy': (0, 0),
        'Q25': 0.0, 'Q25_xy': (0, 0),
        'Q50': 0.0, 'Q50_xy': (0, 0),
        'Q75': 0.0, 'Q75_xy': (0, 0),
        'Q90': 0.0, 'Q90_xy': (0, 0),
        'mad': 0.0,
        'occlusion_asymmetry': 0.0,
        'radius_used': 0.0,
        'person_scale': 0.0
    }
    default_casp_descriptor = [default_descriptor_dict.copy() for _ in range(17)]

    ipdb.set_trace()
    predicted_da_depths = np.array([d if d is not None else default_depth for d in predicted_da_depths])
    predicted_keypoints_scores = np.array([s if s not None else default_keypoints_score for s in predicted_keypoints_scores])
    predicted_keypoints_list = np.array([k if k is not None else default_keypoints for k in predicted_keypoints_list])


    # NEW: Handle None values in casp_descriptors_list
    casp_descriptors_list = [c if c is not None else default_casp_descriptor for c in casp_descriptors_list]

    original_data_dict['predicted_da_depth'] = predicted_da_depths
    original_data_dict['predicted_keypoints_score'] = predicted_keypoints_scores
    original_data_dict['predicted_keypoints'] = predicted_keypoints_list

    # ============================================================================
    # Check if CASP data exists and add to output if present
    # ============================================================================
    has_casp_descriptors = any(x is not None for x in casp_descriptors_list)
    has_casp_summary = any(x is not None for x in summary_casp_descriptor_10d_list)

    print(f"🔍 CASP data check:")
    print(f"  - casp_descriptors: {sum(1 for x in casp_descriptors_list if x is not None)}/{len(casp_descriptors_list)} non-None")
    print(f"  - summary_casp_descriptor_10d: {sum(1 for x in summary_casp_descriptor_10d_list if x is not None)}/{len(summary_casp_descriptor_10d_list)} non-None")

    if has_casp_summary:
        print("✅ CASP summary data detected - adding summary_casp_descriptor_10d")
        
        # UPDATE: Use float16 to save space
        default_summary = np.zeros((17, 10), dtype=np.float16)
        summary_casp_descriptor_10d_array = np.array([
            s if s is not None else default_summary
            for s in summary_casp_descriptor_10d_list
        ], dtype=np.float16)

        # Check how many are actually non-zero (real data vs. defaults)
        non_zero_count = np.count_nonzero(summary_casp_descriptor_10d_array.sum(axis=(1,2)))

        if SAVE_FULL_CASP and has_casp_descriptors:
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
            print(f"  ✅ Saving full casp_descriptors as float16 array")
            output_npz_path = output_npz_path.replace('.npz', '_fullcasp.npz')
        else:
            original_data_dict['summary_casp_descriptor_10d'] = summary_casp_descriptor_10d_array
            print(f"  - summary_casp_descriptor_10d shape: {summary_casp_descriptor_10d_array.shape} (float16)")
            print(f"  - Non-zero summaries: {non_zero_count}/{len(summary_casp_descriptor_10d_array)}")
            print(f"  ⚠️  Skipping 'casp_descriptors' (full dicts) to save space—compact only")
        
        if non_zero_count == 0:
            print("❌ WARNING: All CASP summaries are zero! Depth processing may have failed.")
            
        first_valid = next((c for c in casp_descriptors_list if c is not None), None)
        if first_valid is not None:
            print(f"  - Each casp_descriptor is a list of {len(first_valid)} dicts")
            if len(first_valid) > 0 and first_valid[0] is not None:
                print(f"  - Dict keys: {list(first_valid[0].keys())}")
    else:
        print("❌ No CASP summary data found - CASP field will be missing from output")
        print("   This will cause the '⚠️ CASP summary not found' warning during training")

    # ============================================================================
    # Add RTM and DAV2 per-joint features if requested
    # ============================================================================
    if CACHE_RTM_FEATURES or CACHE_DAV2_FEATURES:
        print(f"\n🔍 Feature map data check:")
        
        if CACHE_RTM_FEATURES:
            has_rtm = any(x is not None for x in rtm_joint_feats_list)
            rtm_count = sum(1 for x in rtm_joint_feats_list if x is not None)
            print(f"  - rtm_joint_feats: {rtm_count}/{len(rtm_joint_feats_list)} non-None")
            
            if has_rtm:
                # Default: (17, 192) array of zeros
                default_rtm = np.zeros((17, 192), dtype=np.float16)
                rtm_joint_feats_array = np.array([
                    f if f is not None else default_rtm
                    for f in rtm_joint_feats_list
                ], dtype=np.float16)
                
                original_data_dict['rtm_joint_feats'] = rtm_joint_feats_array
                print(f"  ✅ rtm_joint_feats shape: {rtm_joint_feats_array.shape} (float16)")
                
                # Check how many are non-zero
                non_zero_rtm = np.count_nonzero(rtm_joint_feats_array.sum(axis=(1,2)))
                print(f"  - Non-zero RTM features: {non_zero_rtm}/{len(rtm_joint_feats_array)}")
            else:
                print(f"  ⚠️  No RTM features found despite CACHE_RTM_FEATURES=True")
        
        if CACHE_DAV2_FEATURES:
            has_dav2 = any(x is not None for x in dav2_joint_feats_list)
            dav2_count = sum(1 for x in dav2_joint_feats_list if x is not None)
            print(f"  - dav2_joint_feats: {dav2_count}/{len(dav2_joint_feats_list)} non-None")
            
            if has_dav2:
                # Default: (17, 256) array of zeros
                default_dav2 = np.zeros((17, 256), dtype=np.float16)
                dav2_joint_feats_array = np.array([
                    f if f is not None else default_dav2
                    for f in dav2_joint_feats_list
                ], dtype=np.float16)
                
                original_data_dict['dav2_joint_feats'] = dav2_joint_feats_array
                print(f"  ✅ dav2_joint_feats shape: {dav2_joint_feats_array.shape} (float16)")
                
                # Check how many are non-zero
                non_zero_dav2 = np.count_nonzero(dav2_joint_feats_array.sum(axis=(1,2)))
                print(f"  - Non-zero DAV2 features: {non_zero_dav2}/{len(dav2_joint_feats_array)}")
            else:
                print(f"  ⚠️  No DAV2 features found despite CACHE_DAV2_FEATURES=True")

    # ============================================================================
    # Add suffix to output filename if using GT keypoints for feature sampling
    # ============================================================================
    if USE_GT_KEYPOINTS_FOR_SAMPLING and (CACHE_RTM_FEATURES or CACHE_DAV2_FEATURES):
        output_npz_path = output_npz_path.replace('.npz', '_feat_map_gt_sample.npz')
        print(f"⚠️  Using GT keypoints for feature sampling - adding suffix to filename")

    np.savez(output_npz_path, **original_data_dict)
    print(f"Updated data successfully saved to {output_npz_path}")

    # ============================================================================
    # STEP 2: Create detection-based version (replacing GT 2D with detected 2D)
    # ============================================================================
    print("\n" + "="*80)
    print("Creating detection-based version (replacing GT 2D with detected 2D)...")
    print("="*80)

    # Make a copy
    det_data_dict = original_data_dict.copy()

    # Get GT keypoints to extract visibility
    gt_keypoints = original_data['part']  # (N, 17, 3)
    visibility = gt_keypoints[:, :, 2:]    # (N, 17, 1)

    # Concatenate predicted keypoints with GT visibility
    pred_keypoints_with_visibility = np.concatenate([predicted_keypoints_list, visibility], axis=-1)

    # Replace 'part' with detected keypoints
    det_data_dict['part'] = pred_keypoints_with_visibility.copy()

    # Save detection-based version
    det_output_path = output_npz_path.replace('.npz', '_dets.npz')
    np.savez(det_output_path, **det_data_dict)
    print(f"✅ Detection-based annotations saved to: {det_output_path}")
    print(f"   'part' now contains: detected (x,y) + GT visibility")
    print("="*80 + "\n")

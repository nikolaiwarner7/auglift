"""
3D Pose Estimation with CASP (Confidence-Adaptive Sampling Procedure)

ALGORITHM OVERVIEW:
==================
This script estimates 3D keypoints by combining 2D pose detection with depth estimation
using a confidence-adaptive sampling strategy.

PIPELINE:
---------
1. DETECT PERSON: RTMDet → bounding boxes
2. ESTIMATE 2D POSE: RTMPose → 17 COCO keypoints + confidence scores
3. ESTIMATE DEPTH MAP: Depth-Anything-V2 → metric depth D(x,y)
4. CASP DEPTH SAMPLING: For each keypoint j:
   
   a) Compute adaptive radius:
      r_j = r_min + (r_max - r_min) × (1 - c_j)^γ
      
   b) Sample ~100 depth values from disk N_j of radius r_j
   
   c) Compute distributional descriptor:
      - central: median(samples)
      - quantiles: Q10, Q25, Q50, Q75, Q90
      - spread: MAD (Median Absolute Deviation)
      - occlusion_asymmetry: Q10 - central
   
   d) Assign depth_j = central (median)

KEY INSIGHT:
------------
• High confidence (c ≈ 1) → small radius (r ≈ 3px) → precise depth from tight neighborhood
• Low confidence (c ≈ 0) → large radius (r ≈ 15px) → robust depth from wide neighborhood
• Median-based depth is robust to occlusions and depth discontinuities

COMPARISON TO BASELINE:
-----------------------
OLD: depth_j = D[y_j, x_j]  // single pixel lookup
NEW: depth_j = median(sample N_j from disk of radius r_j(c_j))  // robust statistics

OUTPUTS:
--------
For each image, saves compressed .npz file with:
- keypoints: (17, 2) H36M 2D coordinates
- keypoints_score: (17,) confidence scores
- keypoints_depth: (17,) CASP median depth estimates
- keypoints_depth_exact: (17,) single-pixel depth lookups
- casp_descriptors: list of dicts with full distributional statistics
- [optional] depth_map: full downsampled depth map

Author: nwarner30
Date: 2025
"""

import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import ipdb

import sys
import importlib
import shutil
# Remove existing imports if they were cached
if "depth_anything_v2" in sys.modules:
    del sys.modules["depth_anything_v2"]

# Insert the correct path to metric_depth
sys.path.insert(0, "/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-OfficialV2/Depth-Anything-V2/metric_depth")
sys.path.append('/srv/essa-lab/flash3/nwarner30/image_editing/baselines_quantitative_dec24/PIDM/')
from jan25_utilities import process_frame_np

# Force reload after setting the correct path
import depth_anything_v2.dpt
importlib.reload(depth_anything_v2.dpt)

print("Loaded from:", depth_anything_v2.dpt.__file__)  # Verify correct path

# Import the model
from depth_anything_v2.dpt import DepthAnythingV2

from mmpose.apis import init_model, inference_topdown
from mmdet.apis import inference_detector, init_detector
from mmpose.utils import adapt_mmdet_pipeline
from mmpose.evaluation.functional import nms
from mmpose.structures import merge_data_samples, split_instances

from pytorch_fid.fid_score import get_activations_from_images, calculate_frechet_distance
from pytorch_fid.inception import InceptionV3


# Directory containing images
img_root = "/srv/essa-lab/flash3/nwarner30/pose_estimation/data/test_images_raw"

JOB_CHUNK_NUMBER = int(os.getenv("JOB_CHUNK_NUMBER", 0))  # Default to 0 if not set
NUM_JOBS = 6

# Configuration: Set to True to cache full depth maps (warning: large file sizes ~8MB per image)
CACHE_FULL_DEPTH_MAPS = False
# Depth map optimization options (when CACHE_FULL_DEPTH_MAPS is True)
DOWNSAMPLE_DEPTH = True  # Downsample to 1024x1024 (75% size reduction)
USE_FLOAT16 = True       # Use float16 instead of float32 (50% size reduction)
                        # Combined: ~85% size reduction (13MB → ~2MB per file)

# output_dir = f"/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_outputs_metric_3dhp_v4_highres/job_{JOB_CHUNK_NUMBER}/"
output_dir = f"/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_outputs_metric_3dhp_v6_patch_stats_test/job_{JOB_CHUNK_NUMBER}/"

plots_output_dir = f"_{JOB_CHUNK_NUMBER}/"


if os.path.exists(plots_output_dir):
    shutil.rmtree(plots_output_dir)
os.makedirs(plots_output_dir, exist_ok=True)

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

# output_dir = f"/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_outputs_metric_3dhp_v3/job_{JOB_CHUNK_NUMBER}/"
# plots_output_dir = f"/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_outputs_metric_plots_3dhp_v3/job_{JOB_CHUNK_NUMBER}/"


# output_dir = "/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_outputs_metric/"
# plots_output_dir = "/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_outputs_metric_plots/"




# Collect all subdirectories
subfolders = sorted([os.path.join(img_root, d) for d in os.listdir(img_root) if os.path.isdir(os.path.join(img_root, d))])

# Split into 8 chunks
subfolder_chunks = np.array_split(subfolders, NUM_JOBS)
assigned_subfolders = subfolder_chunks[JOB_CHUNK_NUMBER]

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}

# encoder = 'vitl' # or 'vits', 'vitb'
encoder = 'vitl'
dataset = 'hypersim' # 'hypersim' for indoor model, 'vkitti' for outdoor model
max_depth = 20 # 20 for indoor model, 80 for outdoor model

model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})

state_dict = torch.load(f'/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth', map_location='cpu')
missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
print(f"Missing keys: {missing_keys}")
print(f"Unexpected keys: {unexpected_keys}")

# model.load_state_dict(torch.load(f'/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth', map_location='cpu'))
# model.load_state_dict(torch.load(f'/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/checkpoints/depth_anything_v2_metric_hypersim_vits.pth', map_location='cpu'))
model.cuda()
model.eval()



def pad_resize(image, target_size=(384,288)):
    """Resize an image to target size while maintaining aspect ratio with padding."""
    h, w = image.shape[:2]
    target_w, target_h = target_size

    # Compute scale while preserving aspect ratio
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize while maintaining aspect ratio
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create a blank canvas and paste resized image
    padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    return padded, scale, pad_x, pad_y



def convert_coco_to_h36m_2d(keypoints):
    """
    Convert COCO 17-keypoint format to H36M 17-keypoint format.
    
    Creates missing H36M joints (Pelvis, Thorax, Spine, Head) by averaging
    corresponding COCO keypoints, then remaps the remaining joints.
    
    COCO (17 keypoints): nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles
    H36M (17 keypoints): pelvis, R-hip, R-knee, R-ankle, L-hip, L-knee, L-ankle, 
                         spine, thorax, neck, head, L-shoulder, L-elbow, L-wrist,
                         R-shoulder, R-elbow, R-wrist
    
    Args:
        keypoints: array of shape (..., 17, 2) in COCO format
        
    Returns:
        keypoints_h36m: array of shape (..., 17, 2) in H36M format
    """
    assert keypoints.shape[-2:] == (17, 2), "Input must have shape (..., 17, 2)"
    
    keypoints_h36m = np.zeros_like(keypoints)  # Ensure correct shape (N, 17, 2)

    # Compute missing joints
    keypoints_h36m[:, 0, :] = (keypoints[:, 11, :] + keypoints[:, 12, :]) / 2  # Pelvis
    keypoints_h36m[:, 8, :] = (keypoints[:, 5, :] + keypoints[:, 6, :]) / 2  # Thorax
    keypoints_h36m[:, 7, :] = (keypoints_h36m[:, 0, :] + keypoints_h36m[:, 8, :]) / 2  # Spine
    keypoints_h36m[:, 10, :] = (keypoints[:, 1, :] + keypoints[:, 2, :]) / 2  # Head

    # Reorder keypoints from COCO to H36M
    dest_indices = [1, 2, 3, 4, 5, 6, 9, 11, 12, 13, 14, 15, 16]
    source_indices = [12, 14, 16, 11, 13, 15, 0, 5, 7, 9, 6, 8, 10]
    
    keypoints_h36m[:, dest_indices, :] = keypoints[:, source_indices, :]

    return keypoints_h36m

    # 2. Reorder the remaining keypoints from COCO to H36M.
    # Mapping (destination H36M index ← source COCO index):
    #   1: Right Hip       ← 12
    #   2: Right Knee      ← 14
    #   3: Right Ankle     ← 16
    #   4: Left Hip        ← 11
    #   5: Left Knee       ← 13
    #   6: Left Ankle      ← 15
    #   9: Neck Base       ← 0   (using the COCO nose)
    #   11: Left Shoulder  ← 5
    #   12: Left Elbow     ← 7
    #   13: Left Wrist     ← 9
    #   14: Right Shoulder ← 6
    #   15: Right Elbow    ← 8
    #   16: Right Wrist    ← 10
    #

def compute_casp_descriptor(depth_samples, coords_samples):
    """
    Compute a compact distributional descriptor for CASP.
    
    Implements Step 2(c) from the CASP algorithm:
    - Central sample (median) as robust depth estimate
    - Robust order statistics (Q10, Q25, Q50, Q75, Q90)
    - Spread metric: MAD (Median Absolute Deviation)
    - Occlusion asymmetry: difference between Q10 and central
    
    The occlusion asymmetry metric captures whether there's a large positive gap
    between the lower quantile and the central value, which may indicate an 
    external occluder in front of the keypoint.
    
    Args:
        depth_samples: array of depth values sampled from disk around keypoint
        coords_samples: array of (x, y) coordinates corresponding to each depth sample
        
    Returns:
        dict with keys:
            - central: median depth (main estimate)
            - central_xy: (x, y) coordinate of the pixel with median depth
            - Q10, Q25, Q50, Q75, Q90: quantiles
            - Q10_xy, Q25_xy, Q50_xy, Q75_xy, Q90_xy: (x, y) coordinates for each quantile
            - mad: Median Absolute Deviation (spread)
            - occlusion_asymmetry: Q10 - central (negative = likely occluder)
        Returns None if no samples available
    """
    if len(depth_samples) == 0:
        return None
    
    depth_samples = np.array(depth_samples)
    coords_samples = np.array(coords_samples)
    
    # Central sample (median)
    d_j_ctr = np.median(depth_samples)
    central_idx = np.argmin(np.abs(depth_samples - d_j_ctr))
    central_xy = tuple(coords_samples[central_idx])
    
    # Robust order statistics (quantiles) with coordinates
    Q10 = np.percentile(depth_samples, 10)
    Q10_idx = np.argmin(np.abs(depth_samples - Q10))
    Q10_xy = tuple(coords_samples[Q10_idx])
    
    Q25 = np.percentile(depth_samples, 25)
    Q25_idx = np.argmin(np.abs(depth_samples - Q25))
    Q25_xy = tuple(coords_samples[Q25_idx])
    
    Q50 = np.percentile(depth_samples, 50)
    Q50_idx = np.argmin(np.abs(depth_samples - Q50))
    Q50_xy = tuple(coords_samples[Q50_idx])
    
    Q75 = np.percentile(depth_samples, 75)
    Q75_idx = np.argmin(np.abs(depth_samples - Q75))
    Q75_xy = tuple(coords_samples[Q75_idx])
    
    Q90 = np.percentile(depth_samples, 90)
    Q90_idx = np.argmin(np.abs(depth_samples - Q90))
    Q90_xy = tuple(coords_samples[Q90_idx])
    
    # Spread: MAD (Median Absolute Deviation)
    mad = np.median(np.abs(depth_samples - Q50))
    
    # Occlusion asymmetry: difference between lower quantile and central
    occlusion_asym = Q10 - d_j_ctr
    
    descriptor = {
        'central': d_j_ctr,
        'central_xy': central_xy,
        'Q10': Q10, 'Q10_xy': Q10_xy,
        'Q25': Q25, 'Q25_xy': Q25_xy,
        'Q50': Q50, 'Q50_xy': Q50_xy,
        'Q75': Q75, 'Q75_xy': Q75_xy,
        'Q90': Q90, 'Q90_xy': Q90_xy,
        'mad': mad,
        'occlusion_asymmetry': occlusion_asym
    }
    
    return descriptor


def get_casp_depth(depth_map, keypoint, confidence, person_scale, r_min=6, r_max=22, 
                   k=0.07, beta=1.0, gamma=1.0, num_samples=100):
    """
    CASP: Confidence-Adaptive & Scale-Adaptive Sampling Procedure for depth estimation.
    
    Implements enhanced CASP with scale awareness:
    
    For each keypoint j, set a confidence- and scale-adaptive radius:
        r_j = clip(r_min, r_max, k × S × [1 + β(1 - c_j)]^γ)
    
    where S is the person scale (torso length in pixels).
    
    Sample depths in a disk N_j of radius r_j centered at (x_j, y_j).
    Compute a compact distributional descriptor using robust statistics.
    
    RATIONALE:
    - High confidence → small radius → trust precise 2D localization
    - Low confidence → large radius → aggregate over wider neighborhood for robustness
    - Large person → larger radius to match body part size
    - Small person → smaller radius to avoid sampling outside body
    - Median is robust to outliers from occlusions and depth discontinuities
    
    Args:
        depth_map: (H, W) metric depth map in meters
        keypoint: (x, y) 2D pixel coordinates of keypoint
        confidence: confidence score c_j ∈ [0,1] from 2D pose detector
        person_scale: scale metric S (e.g., torso length or sqrt(bbox_area)) in pixels
        r_min: minimum sampling radius in pixels (default: 6)
        r_max: maximum sampling radius in pixels (default: 22)
        k: scale multiplier (default: 0.07)
        beta: confidence sensitivity (default: 1.0)
        gamma: confidence exponent (default: 1.0)
        num_samples: number of random points to sample in disk (default: 100)
        
    Returns:
        tuple: (descriptor_dict, central_depth)
            - descriptor_dict: contains central, quantiles, MAD, occlusion_asymmetry, radius_used
            - central_depth: median depth value (main estimate)
    
    Example:
        >>> depth_map = np.random.rand(480, 640) * 10  # mock depth map
        >>> kp = (320, 240)  # center pixel
        >>> conf = 0.9  # high confidence
        >>> scale = 200.0  # torso length = 200 pixels
        >>> desc, depth = get_casp_depth(depth_map, kp, conf, scale)
        >>> print(f"Depth: {depth:.2f}m, radius: {desc['radius_used']:.1f}px")
    """
    x, y = keypoint
    H, W = depth_map.shape
    
    # Confidence- and scale-adaptive radius
    # r = k × S × [1 + β(1 - c)]^γ
    confidence_factor = (1 + beta * (1 - confidence)) ** gamma
    r_j = k * person_scale * confidence_factor
    
    # Clip to [r_min, r_max]
    r_j = np.clip(r_j, r_min, r_max)
    
    # Sample depths in a disk of radius r_j
    depth_samples = []
    coords_samples = []  # Store (x, y) coordinates for each sample
    
    # Generate random points in disk using rejection sampling
    for _ in range(num_samples):
        # Random point in square [-r_j, r_j]
        dx = np.random.uniform(-r_j, r_j)
        dy = np.random.uniform(-r_j, r_j)
        
        # Check if inside disk
        if dx**2 + dy**2 <= r_j**2:
            nx, ny = int(x + dx), int(y + dy)
            if 0 <= nx < W and 0 <= ny < H:
                depth_samples.append(depth_map[ny, nx])
                coords_samples.append((nx, ny))
    
    # Compute distributional descriptor
    descriptor = compute_casp_descriptor(depth_samples, coords_samples)
    
    if descriptor is None:
        return None, depth_map[y, x]
    
    # Store the radius used in the descriptor for debugging/analysis
    descriptor['radius_used'] = r_j
    descriptor['person_scale'] = person_scale
    
    # Return the central (median) depth as the main estimate
    return descriptor, descriptor['central']

def convert_coco_to_h36m_confidence(scores):
    """
    Convert COCO confidence scores to H36M format.
    
    Averages confidence scores for synthetic H36M joints (Pelvis, Thorax, Spine, Head)
    from their constituent COCO joints, then remaps the remaining scores.
    
    Args:
        scores: array of shape (N, 17) with COCO confidence scores
        
    Returns:
        scores_h36m: array of shape (N, 17) with H36M confidence scores
    """
    # Create a new array for H36M (17 keypoints)
    scores_h36m = np.zeros((scores.shape[0], 17), dtype=scores.dtype)

    # Pelvis (root) = average of left hip (11) and right hip (12)
    scores_h36m[:, 0] = (scores[:, 11] + scores[:, 12]) / 2

    # Thorax = average of left shoulder (5) and right shoulder (6)
    scores_h36m[:, 8] = (scores[:, 5] + scores[:, 6]) / 2

    # Spine = average of thorax (8) and pelvis (0)
    scores_h36m[:, 7] = (scores_h36m[:, 0] + scores_h36m[:, 8]) / 2

    # Head = average of left eye (1) and right eye (2)
    scores_h36m[:, 10] = (scores[:, 1] + scores[:, 2]) / 2

    # Rearrange remaining joints according to COCO → H36M mapping
    coco_to_h36m_indices = [12, 14, 16, 11, 13, 15, 0, 5, 7, 9, 6, 8, 10]

    scores_h36m[:, [1, 2, 3, 4, 5, 6, 9, 11, 12, 13, 14, 15, 16]] = scores[:, coco_to_h36m_indices]

    return scores_h36m


def compute_person_scale(keypoints_h36m):
    """
    Compute person scale metric from H36M keypoints.
    
    Uses torso length (pelvis to thorax distance) as the scale metric.
    This provides a robust measure of person size in the image.
    
    H36M joint indices:
        0: Pelvis
        8: Thorax
    
    Args:
        keypoints_h36m: (17, 2) array of H36M keypoints in pixel coordinates
        
    Returns:
        float: torso length in pixels (scale metric S)
    """
    # Pelvis (root) - index 0
    pelvis = keypoints_h36m[0]
    
    # Thorax - index 8
    thorax = keypoints_h36m[8]
    
    # Compute Euclidean distance (torso length)
    torso_length = np.linalg.norm(thorax - pelvis)
    
    return torso_length





# # Ensure output directory exists
# os.makedirs(output_dir, exist_ok=True)
# os.makedirs(plots_output_dir, exist_ok=True)


# Collect image paths from assigned subfolders
image_paths = []
print(f"Job {JOB_CHUNK_NUMBER}: Collecting paths from {len(assigned_subfolders)} subfolders")
for subfolder in tqdm(assigned_subfolders, desc=f"Job {JOB_CHUNK_NUMBER} Processing"):
    for root, _, files in os.walk(subfolder):
        # ipdb.set_trace()
        for file in files:
            file = file.decode('utf-8') if isinstance(file, bytes) else file
            if file.lower().endswith(('.jpg', '.png')):
            # if file.decode('utf-8').lower().endswith(('.jpg', '.png')):
            # if file.lower().endswith(('.jpg', '.png')):
                image_paths.append(os.path.join(root, file))

# ipdb.set_trace()
# Save depth map visualization
vmin, vmax = 5, 7


target_size = (384,288)  # Use (384, 288) if RTMPose is set to that

# POSE_MODEL_CONFIG = '/srv/essa-lab/flash3/nwarner30/image_editing/hallucinating-scenes/mmpose/rtmpose-l_8xb256-420e_coco-256x192.py'
POSE_MODEL_CONFIG = '/srv/essa-lab/flash3/nwarner30/pose_estimation/mmpose/rtmpose-l_8xb256-420e_coco-384x288.py'


# POSE_MODEL_CHECKPOINT = '/srv/essa-lab/flash3/nwarner30/image_editing/hallucinating-scenes/mmpose/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-256x192-f016ffe0_20230126.pth'
POSE_MODEL_CHECKPOINT = '/srv/essa-lab/flash3/nwarner30/pose_estimation/mmpose/rtmpose-l_simcc-body7_pt-body7_420e-384x288-3f5a1437_20230504.pth'

SKELETON_STYLE='mmpose'

# Initialize device and pose model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pose_model = init_model(POSE_MODEL_CONFIG, POSE_MODEL_CHECKPOINT, device=device)

# Initialize the visualizer with configuration from the pose model
from mmpose.registry import VISUALIZERS
pose_model.cfg.visualizer.radius = 3
pose_model.cfg.visualizer.alpha = 0.8
pose_model.cfg.visualizer.line_width = 1
visualizer = VISUALIZERS.build(pose_model.cfg.visualizer)
visualizer.set_dataset_meta(pose_model.dataset_meta, skeleton_style='mmpose')

detection_config = '/srv/essa-lab/flash3/nwarner30/image_editing/hallucinating-scenes/mmpose/projects/rtmpose/rtmdet/person/rtmdet_m_640-8xb32_coco-person.py'
# detection_checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-m_8xb256-rsb-a1-600e_in1k-ecb3bbd9.pth'
detection_checkpoint = 'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth'
DETECTION_MODEL = init_detector(detection_config, detection_checkpoint, device=device)
DETECTION_MODEL.cfg = adapt_mmdet_pipeline(DETECTION_MODEL.cfg)

KPT_THRESH_COMPLETE = 0.3 # Confidence metric to determine if pose is complete/ contained in video. Depends on pose extracto
DET_CAT_ID = 0 # Human, check
# BBOX_THRESH = 0.03
BBOX_THRESH = 0.3 # fixed for highres
NMS_THRESH = 0.3

import uuid

# Dynamically set TEMP_FILE_NAME with a random UUID
TEMP_FILE_NAME = f"/tmp/temp_{uuid.uuid4().hex}.png"

# Debug: Set to True to save and visualize first few frames
DEBUG_MODE = True
DEBUG_FRAMES_COUNT = 75
processed_frames = 0

# Shuffle image paths in debug mode for better variation
if DEBUG_MODE:
    import random
    random.seed(42)  # Set seed for reproducibility
    random.shuffle(image_paths)
    print(f"Debug mode: Shuffled {len(image_paths)} images for variation")

# Process each image
for img_path in tqdm(image_paths, desc=f"Job {JOB_CHUNK_NUMBER} Processing Images"):
    raw_img = cv2.imread(img_path)

    # Run inference
    with torch.no_grad():
        depth_map = model.infer_image(raw_img)
        # Pad-resize image while preserving aspect ratio

        det_results = inference_detector(DETECTION_MODEL, raw_img)
        pred_np = det_results.pred_instances.detach().cpu().numpy()
        bboxes = np.concatenate((pred_np.bboxes, pred_np.scores[:, None]), axis=1)
        bboxes = bboxes[(pred_np.labels == DET_CAT_ID) & (pred_np.scores > BBOX_THRESH)]
        bboxes = bboxes[nms(bboxes, NMS_THRESH), :4]

        # Run top-down pose on the original image + boxes
        pose_results = inference_topdown(
            pose_model,
            raw_img,         # full-res H×W×3 array
            bboxes,
            # format='xyxy'
        )
        data_samples = merge_data_samples(pose_results)

        # Select the best single proposal
        keypoints = data_samples.pred_instances.keypoints
        keypoints_score = data_samples.pred_instances.keypoint_scores
        sel = keypoints_score.mean(axis=1).argmax()
        keypoints = keypoints[sel]
        keypoints_score = keypoints_score[sel]

        # Convert to H36M ordering
        keypoints_h36m      = convert_coco_to_h36m_2d(keypoints[None, ...])[0]
        keypoints_conf_h36m = convert_coco_to_h36m_confidence(keypoints_score[None, ...])[0]



    # ipdb.set_trace()
    # after you have keypoints_h36m as floats
    keypoints_h36m_int = keypoints_h36m.astype(np.int64)

    # clamp to valid indices
    H, W = depth_map.shape
    keypoints_h36m_int[:, 0] = np.clip(keypoints_h36m_int[:, 0], 0, W-1)
    keypoints_h36m_int[:, 1] = np.clip(keypoints_h36m_int[:, 1], 0, H-1)

    # now you can safely index
    keypoints_depth_exact = depth_map[
        keypoints_h36m_int[:, 1],
        keypoints_h36m_int[:, 0]
    ]

    # Compute person scale (torso length) for scale-adaptive CASP
    person_scale = compute_person_scale(keypoints_h36m)
    
    # Extract depth values using CASP (Confidence-Adaptive & Scale-Adaptive Sampling Procedure)
    keypoints_depth_casp = []
    casp_descriptors = []
    
    for kp, conf in zip(keypoints_h36m_int, keypoints_conf_h36m):
        descriptor, depth_value = get_casp_depth(
            depth_map, 
            kp, 
            confidence=conf,
            person_scale=person_scale,
            r_min=3,
            r_max=30,
            k=0.04,
            beta=3.0,
            gamma=1.5,
            num_samples=100
        )
        keypoints_depth_casp.append(depth_value)
        casp_descriptors.append(descriptor)
    
    keypoints_depth_casp = np.array(keypoints_depth_casp)

    # Debug visualization for first few frames
    if DEBUG_MODE and processed_frames < DEBUG_FRAMES_COUNT:
        # Convert BGR to RGB for plotting
        img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        
        # Find the joint with highest variability (MAD) for detailed analysis
        mad_values = [desc['mad'] if desc is not None else 0 for desc in casp_descriptors]
        highest_var_idx = np.argmax(mad_values)
        highest_var_val = mad_values[highest_var_idx]
        highest_var_conf = keypoints_conf_h36m[highest_var_idx]
        highest_var_desc = casp_descriptors[highest_var_idx]
        
        # Create figure with subplots (4 rows, 2 columns)
        fig, axes = plt.subplots(4, 2, figsize=(16, 24))
        
        # 1. Original image with keypoints
        axes[0, 0].imshow(img_rgb)
        axes[0, 0].scatter(keypoints_h36m_int[:, 0], keypoints_h36m_int[:, 1], 
                          c='red', s=50, marker='o', label='Keypoints')
        axes[0, 0].set_title('Original Image with Keypoints')
        axes[0, 0].axis('off')
        axes[0, 0].legend()
        
        # 2. Depth map with CASP circles - ENHANCED VISIBILITY with ZOOM
        # Calculate bounding box around all keypoints with padding
        x_min = keypoints_h36m_int[:, 0].min()
        x_max = keypoints_h36m_int[:, 0].max()
        y_min = keypoints_h36m_int[:, 1].min()
        y_max = keypoints_h36m_int[:, 1].max()
        
        # Add padding (e.g., 20% of the bbox size or 100 pixels, whichever is larger)
        x_range = x_max - x_min
        y_range = y_max - y_min
        padding = max(int(0.3 * max(x_range, y_range)), 100)
        
        x_min_padded = max(0, x_min - padding)
        x_max_padded = min(W, x_max + padding)
        y_min_padded = max(0, y_min - padding)
        y_max_padded = min(H, y_max + padding)
        
        # Crop depth map to zoomed region
        depth_map_cropped = depth_map[y_min_padded:y_max_padded, x_min_padded:x_max_padded]
        
        im1 = axes[0, 1].imshow(depth_map_cropped, cmap='viridis', 
                               extent=[x_min_padded, x_max_padded, y_max_padded, y_min_padded])
        
        # Note: Circles are in data coordinates (pixels), so they automatically scale with zoom
        # No manual scaling needed - matplotlib handles this correctly
        
        # Superimpose confidence- and scale-adaptive circles on depth map with maximum visibility
        for idx, (kp, conf) in enumerate(zip(keypoints_h36m_int, keypoints_conf_h36m)):
            # Use same formula as CASP
            k, beta, gamma = 0.04, 3.0, 1.5
            r_min, r_max = 3, 30
            confidence_factor = (1 + beta * (1 - conf)) ** gamma
            r_j = np.clip(k * person_scale * confidence_factor, r_min, r_max)
            
            # Highlight the highest variability joint differently
            if idx == highest_var_idx:
                circle = plt.Circle((kp[0], kp[1]), r_j, color='yellow', fill=False, linewidth=4, alpha=1.0)
            else:
                circle = plt.Circle((kp[0], kp[1]), r_j, color='red', fill=False, linewidth=3, alpha=1.0)
            axes[0, 1].add_patch(circle)
        
        # Use smaller white X markers with red edges (markers don't auto-scale, keep them normal)
        axes[0, 1].scatter(keypoints_h36m_int[:, 0], keypoints_h36m_int[:, 1], 
                          c='white', s=40, marker='x', linewidths=2, edgecolors='red')
        # Highlight highest variability joint
        axes[0, 1].scatter(keypoints_h36m_int[highest_var_idx, 0], keypoints_h36m_int[highest_var_idx, 1], 
                          c='yellow', s=60, marker='x', linewidths=3, edgecolors='black')
        
        axes[0, 1].set_xlim(x_min_padded, x_max_padded)
        axes[0, 1].set_ylim(y_max_padded, y_min_padded)  # Inverted for image coordinates
        axes[0, 1].set_title('Depth Map + CASP Sampling Radii (ZOOMED, yellow = highest var)')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        # 3. Confidence- and scale-adaptive radii visualization
        axes[1, 0].imshow(img_rgb)
        for idx, (kp, conf) in enumerate(zip(keypoints_h36m_int, keypoints_conf_h36m)):
            # Use same formula as CASP
            k, beta, gamma = 0.04, 3.0, 1.5
            r_min, r_max = 3, 30
            confidence_factor = (1 + beta * (1 - conf)) ** gamma
            r_j = np.clip(k * person_scale * confidence_factor, r_min, r_max)
            
            if idx == highest_var_idx:
                circle = plt.Circle((kp[0], kp[1]), r_j, color='yellow', fill=False, linewidth=3, alpha=0.8)
            else:
                circle = plt.Circle((kp[0], kp[1]), r_j, color='cyan', fill=False, linewidth=2, alpha=0.6)
            axes[1, 0].add_patch(circle)
        axes[1, 0].scatter(keypoints_h36m_int[:, 0], keypoints_h36m_int[:, 1], 
                          c='red', s=30, marker='o')
        # Highlight highest variability joint
        axes[1, 0].scatter(keypoints_h36m_int[highest_var_idx, 0], keypoints_h36m_int[highest_var_idx, 1], 
                          c='yellow', s=50, marker='o', edgecolors='black', linewidths=2)
        axes[1, 0].set_title(f'CASP Adaptive Radii (Scale={person_scale:.0f}px, Joint {highest_var_idx}, MAD={highest_var_val:.4f}m)')
        axes[1, 0].axis('off')
        
        # 4. Depth comparison plot with CASP ranges
        joint_indices = np.arange(len(keypoints_depth_casp))
        
        # Plot exact depth as single line
        axes[1, 1].plot(joint_indices, keypoints_depth_exact, 'o-', color='orange', 
                       label='Exact (single pixel)', markersize=8, linewidth=2, zorder=3)
        
        # Plot CASP median
        axes[1, 1].plot(joint_indices, keypoints_depth_casp, 's-', color='blue', 
                       label='CASP (median)', markersize=8, linewidth=2, zorder=3)
        
        # Add shaded region for CASP Q25-Q75 (IQR) and error bars for Q10-Q90
        q10_values = [desc['Q10'] if desc is not None else 0 for desc in casp_descriptors]
        q25_values = [desc['Q25'] if desc is not None else 0 for desc in casp_descriptors]
        q75_values = [desc['Q75'] if desc is not None else 0 for desc in casp_descriptors]
        q90_values = [desc['Q90'] if desc is not None else 0 for desc in casp_descriptors]
        
        # Shaded region for IQR (Q25-Q75)
        axes[1, 1].fill_between(joint_indices, q25_values, q75_values, 
                                alpha=0.3, color='blue', label='CASP IQR (Q25-Q75)')
        
        # Error bars for full range (Q10-Q90)
        lower_err = [keypoints_depth_casp[i] - q10_values[i] for i in range(17)]
        upper_err = [q90_values[i] - keypoints_depth_casp[i] for i in range(17)]
        axes[1, 1].errorbar(joint_indices, keypoints_depth_casp, 
                           yerr=[lower_err, upper_err],
                           fmt='none', ecolor='blue', alpha=0.5, capsize=3, 
                           linewidth=1.5, label='CASP Range (Q10-Q90)', zorder=2)
        
        # Highlight highest variability joint
        axes[1, 1].scatter(highest_var_idx, keypoints_depth_exact[highest_var_idx], 
                          s=200, marker='o', facecolors='none', edgecolors='red', linewidths=3, zorder=4)
        axes[1, 1].scatter(highest_var_idx, keypoints_depth_casp[highest_var_idx], 
                          s=200, marker='s', facecolors='none', edgecolors='red', linewidths=3, zorder=4)
        
        axes[1, 1].set_xlabel('Joint Index')
        axes[1, 1].set_ylabel('Depth (m)')
        axes[1, 1].set_title('Depth with CASP Distribution (red circles = highest variability)')
        axes[1, 1].legend(fontsize=8, loc='best')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 5. Distributional descriptor for highest variability joint
        if highest_var_desc is not None:
            desc = highest_var_desc
            quantile_names = ['Q10', 'Q25', 'Q50', 'Q75', 'Q90']
            quantile_values = [desc['Q10'], desc['Q25'], desc['Q50'], desc['Q75'], desc['Q90']]
            
            axes[2, 0].barh(quantile_names, quantile_values, color='steelblue', alpha=0.7)
            axes[2, 0].axvline(desc['central'], color='red', linestyle='--', linewidth=2, label=f"Central (median): {desc['central']:.3f}m")
            axes[2, 0].axvline(keypoints_depth_exact[highest_var_idx], color='orange', linestyle=':', linewidth=2, 
                              label=f"Exact: {keypoints_depth_exact[highest_var_idx]:.3f}m")
            axes[2, 0].set_xlabel('Depth (m)')
            axes[2, 0].set_title(f'CASP Descriptor for Joint {highest_var_idx} (MAD={highest_var_val:.4f}m)')
            axes[2, 0].legend(fontsize=9)
            axes[2, 0].grid(True, alpha=0.3, axis='x')
            
            # Add text annotations
            textstr = f"Confidence: {highest_var_conf:.3f}\nMAD (spread): {desc['mad']:.4f}m\nOcclusion asym: {desc['occlusion_asymmetry']:.4f}m"
            axes[2, 0].text(0.02, 0.98, textstr, transform=axes[2, 0].transAxes, 
                           fontsize=10, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 6. Histogram of all depth samples for highest variability joint
        # Re-sample for visualization purposes
        kp_highest_var = keypoints_h36m_int[highest_var_idx]
        conf_highest_var = keypoints_conf_h36m[highest_var_idx]
        
        # Use same formula as CASP
        k, beta, gamma = 0.04, 3.0, 1.5
        r_min, r_max = 3, 30
        confidence_factor = (1 + beta * (1 - conf_highest_var)) ** gamma
        r_j = np.clip(k * person_scale * confidence_factor, r_min, r_max)
        
        # Sample depths again for histogram
        depth_samples_viz = []
        for _ in range(500):  # More samples for better histogram
            dx = np.random.uniform(-r_j, r_j)
            dy = np.random.uniform(-r_j, r_j)
            if dx**2 + dy**2 <= r_j**2:
                nx, ny = int(kp_highest_var[0] + dx), int(kp_highest_var[1] + dy)
                if 0 <= nx < W and 0 <= ny < H:
                    depth_samples_viz.append(depth_map[ny, nx])
        
        if len(depth_samples_viz) > 0:
            axes[2, 1].hist(depth_samples_viz, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
            axes[2, 1].axvline(desc['central'], color='red', linestyle='--', linewidth=2, label=f"Median: {desc['central']:.3f}m")
            axes[2, 1].axvline(keypoints_depth_exact[highest_var_idx], color='orange', linestyle=':', linewidth=2, 
                              label=f"Exact: {keypoints_depth_exact[highest_var_idx]:.3f}m")
            axes[2, 1].set_xlabel('Depth (m)')
            axes[2, 1].set_ylabel('Frequency')
            axes[2, 1].set_title(f'Depth Distribution (r={r_j:.1f}px, n={len(depth_samples_viz)} samples)')
            axes[2, 1].legend()
            axes[2, 1].grid(True, alpha=0.3, axis='y')
        
        # 7. Box and whisker plot for all 17 joints
        # Collect depth samples for each joint
        all_joint_samples = []
        radius_values = []  # Track radius for each joint
        for idx, (kp, conf) in enumerate(zip(keypoints_h36m_int, keypoints_conf_h36m)):
            # Use same formula as CASP
            k, beta, gamma = 0.04, 3.0, 1.5
            r_min, r_max = 3, 30
            confidence_factor = (1 + beta * (1 - conf)) ** gamma
            r_j = np.clip(k * person_scale * confidence_factor, r_min, r_max)
            radius_values.append(r_j)
            
            joint_samples = []
            for _ in range(200):  # Sample for box plot
                dx = np.random.uniform(-r_j, r_j)
                dy = np.random.uniform(-r_j, r_j)
                if dx**2 + dy**2 <= r_j**2:
                    nx, ny = int(kp[0] + dx), int(kp[1] + dy)
                    if 0 <= nx < W and 0 <= ny < H:
                        joint_samples.append(depth_map[ny, nx])
            all_joint_samples.append(joint_samples)
        
        # Create box plot
        bp = axes[3, 0].boxplot(all_joint_samples, positions=range(17), widths=0.6, patch_artist=True,
                                showfliers=True, flierprops=dict(marker='o', markersize=3, alpha=0.5))
        # Color boxes
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        # Highlight highest variability joint
        bp['boxes'][highest_var_idx].set_facecolor('yellow')
        bp['boxes'][highest_var_idx].set_edgecolor('black')
        bp['boxes'][highest_var_idx].set_linewidth(2)
        
        # Overlay the CASP median values
        axes[3, 0].plot(range(17), keypoints_depth_casp, 'r^', markersize=8, label='CASP Median', zorder=3)
        # Overlay exact depth values
        axes[3, 0].plot(range(17), keypoints_depth_exact, 'o', color='orange', markersize=6, label='Exact Depth', zorder=3)
        
        axes[3, 0].set_xlabel('Joint Index')
        axes[3, 0].set_ylabel('Depth (m)')
        axes[3, 0].set_title('CASP Depth Distribution per Joint (yellow = highest variability)')
        axes[3, 0].legend()
        axes[3, 0].grid(True, alpha=0.3, axis='y')
        axes[3, 0].set_xticks(range(17))
        
        # 8. Adaptive Radius per joint
        axes[3, 1].bar(range(17), radius_values, color='steelblue', alpha=0.7, edgecolor='black')
        axes[3, 1].bar(highest_var_idx, radius_values[highest_var_idx], color='yellow', alpha=0.9, edgecolor='black', linewidth=2)
        
        # Add horizontal lines for r_min and r_max
        axes[3, 1].axhline(y=3, color='red', linestyle='--', linewidth=1, alpha=0.7, label=f'r_min=3px')
        axes[3, 1].axhline(y=30, color='red', linestyle='--', linewidth=1, alpha=0.7, label=f'r_max=30px')
        
        # Add text showing range and spread
        radius_range = max(radius_values) - min(radius_values)
        radius_ratio = max(radius_values) / max(min(radius_values), 0.1)  # Avoid divide by zero
        textstr = f"Person scale: {person_scale:.0f}px\nRadius range: [{min(radius_values):.1f}, {max(radius_values):.1f}]px\nSpread: {radius_range:.1f}px ({radius_ratio:.2f}x)\nMean: {np.mean(radius_values):.1f}px"
        axes[3, 1].text(0.02, 0.98, textstr, transform=axes[3, 1].transAxes, 
                       fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        axes[3, 1].set_xlabel('Joint Index')
        axes[3, 1].set_ylabel('Adaptive Radius [pixels]')
        axes[3, 1].set_title('CASP Adaptive Radius per Joint (β=3.0, γ=1.5)')
        axes[3, 1].legend(loc='upper right', fontsize=8)
        axes[3, 1].grid(True, alpha=0.3, axis='y')
        axes[3, 1].set_xticks(range(17))
        axes[3, 1].set_ylim(0, 35)  # Set y-limit to show full range
        
        plt.tight_layout()
        debug_path = os.path.join(plots_output_dir, f'debug_frame_{processed_frames:03d}.png')
        plt.savefig(debug_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\nDebug frame {processed_frames} saved to: {debug_path}")
        print(f"  Avg confidence: {keypoints_conf_h36m.mean():.3f}")
        print(f"  Depth range (exact): [{keypoints_depth_exact.min():.2f}, {keypoints_depth_exact.max():.2f}]")
        print(f"  Depth range (CASP):  [{keypoints_depth_casp.min():.2f}, {keypoints_depth_casp.max():.2f}]")
        
        processed_frames += 1
        
        # Add breakpoint after first frame for inspection
        if processed_frames == DEBUG_FRAMES_COUNT:
            ipdb.set_trace()

    # ipdb.set_trace()
    # convert BGR→RGB for plotting
    img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

    h, w = img_rgb.shape[:2]
    dpi = 100
    plt.figure(figsize=(w/dpi, h/dpi), dpi=dpi)
    plt.imshow(img_rgb)
    # keypoints_h36m_int is (17,2) array of [x,y] in original image coords
    xs = keypoints_h36m_int[:, 0]
    ys = keypoints_h36m_int[:, 1]

    # plot joints
    plt.scatter(xs, ys, c='r', s=40, marker='o', label='predicted')
    # optionally connect with skeleton lines if you have a list of edges:
    # for (i,j) in skeleton:
    #     plt.plot([xs[i], xs[j]], [ys[i], ys[j]], c='r')

    plt.axis('off')
    plt.legend(loc='lower right')
    plt.tight_layout()

    # build the output filename by replacing the extension
    kpt_fname = os.path.basename(img_path).replace(".jpg", "_kpts.jpg")
    # join with your plots directory
    kpt_path = os.path.join(plots_output_dir, kpt_fname)

    plt.savefig(kpt_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    # ipdb.set_trace()

    output_path = os.path.join(output_dir, os.path.basename(img_path).replace(".jpg", "_depth.npz"))
    
    # Create compact 10D summary descriptor for each joint
    # Format: [x, y, confidence, radius, Q10, Q25, Q50, Q75, Q90, exact_depth]
    summary_casp_descriptor_10d = []
    for kp, conf, desc, exact_depth in zip(keypoints_h36m, keypoints_conf_h36m, casp_descriptors, keypoints_depth_exact):
        if desc is not None:
            summary = [
                kp[0],              # x coordinate
                kp[1],              # y coordinate
                conf,               # confidence score
                desc['radius_used'], # adaptive radius
                desc['Q10'],        # 10th percentile depth
                desc['Q25'],        # 25th percentile depth
                desc['Q50'],        # median depth
                desc['Q75'],        # 75th percentile depth
                desc['Q90'],        # 90th percentile depth
                exact_depth         # exact single-pixel depth at keypoint
            ]
        else:
            # Fallback if descriptor is None
            summary = [kp[0], kp[1], conf, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, exact_depth]
        summary_casp_descriptor_10d.append(summary)
    
    summary_casp_descriptor_10d = np.array(summary_casp_descriptor_10d)  # Shape: (17, 10)
    
    # Prepare data to save
    save_data = {
        'keypoints': keypoints_h36m, 
        'keypoints_score': keypoints_conf_h36m, 
        'keypoints_depth': keypoints_depth_casp,  # Now using CASP depth
        'keypoints_depth_exact': keypoints_depth_exact,
        'casp_descriptors': casp_descriptors,  # Save full CASP statistics (list of dicts)
        'summary_casp_descriptor_10d': summary_casp_descriptor_10d  # Compact (17, 10) array
    }
    
    # Optionally include full depth map with optimizations
    if CACHE_FULL_DEPTH_MAPS:
        optimized_depth = depth_map.copy()
        
        # Downsample if requested
        if DOWNSAMPLE_DEPTH:
            optimized_depth = cv2.resize(optimized_depth, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        
        # Convert to float16 if requested
        if USE_FLOAT16:
            optimized_depth = optimized_depth.astype(np.float16)
        
        save_data['depth_map'] = optimized_depth
        # ipdb.set_trace()
    
    np.savez_compressed(output_path, **save_data)

import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import ipdb
from collections import defaultdict
import numpy as np

import sys
import importlib
import shutil
from PIL import Image

# Pose specific metric imports
from mmpose.apis import init_model, inference_topdown
from mmdet.apis import inference_detector, init_detector
from mmpose.utils import adapt_mmdet_pipeline
from mmpose.evaluation.functional import nms
from mmpose.structures import merge_data_samples, split_instances

# Remove existing imports if they were cached
if "depth_anything_v2" in sys.modules:
    del sys.modules["depth_anything_v2"]

# Insert the correct path to metric_depth
sys.path.insert(0, "/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-OfficialV2/Depth-Anything-V2/metric_depth")
sys.path.append('/srv/essa-lab/flash3/nwarner30/image_editing/baselines_quantitative_dec24/PIDM/')
# from jan25_utilities import process_frame_np

# Force reload after setting the correct path
import depth_anything_v2.dpt
importlib.reload(depth_anything_v2.dpt)

print("Loaded from:", depth_anything_v2.dpt.__file__)  # Verify correct path

# Import the model
from depth_anything_v2.dpt import DepthAnythingV2



# Directory containing images
img_root = "/srv/essa-lab/flash3/nwarner30/pose_estimation/3dpw_data/imageFiles"

JOB_CHUNK_NUMBER = int(os.getenv("JOB_CHUNK_NUMBER", 5))  # Default to 0 if not set
NUM_JOBS = 8
PLOT_EXAMPLES=True

USE_GT_FOR_DAV = False

output_dir = f"/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_outputs_metric_3dpw_train_v6_casp/job_{JOB_CHUNK_NUMBER}/"
plots_output_dir = f"/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_outputs_metric_plots_3dpw_train_v6_casp/job_{JOB_CHUNK_NUMBER}/"



if os.path.exists(plots_output_dir):
    shutil.rmtree(plots_output_dir)
os.makedirs(plots_output_dir, exist_ok=True)

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

#v6 try outdoor
# USE_GT_FOR_DAV = True
# output_dir = f"/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_outputs_metric_3dpw_train_v6_gtdav/job_{JOB_CHUNK_NUMBER}/" #correct bgr to rgb v3
# plots_output_dir = f"/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_outputs_metric_plots_3dpw_train_v6_gtdav/job_{JOB_CHUNK_NUMBER}/"


# USE_GT_FOR_DAV = False
# output_dir = f"/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_outputs_metric_3dpw_train_v6_det_dav/job_{JOB_CHUNK_NUMBER}/" #correct bgr to rgb v3
# plots_output_dir = f"/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_outputs_metric_plots_3dpw_train_v6_det_dav/job_{JOB_CHUNK_NUMBER}/"




# USE_GT_FOR_DAV = True
# output_dir = f"/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_outputs_metric_3dpw_train_v5_gtdav/job_{JOB_CHUNK_NUMBER}/" #correct bgr to rgb v3
# plots_output_dir = f"/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_outputs_metric_plots_3dpw_train_v5_gtdav/job_{JOB_CHUNK_NUMBER}/"

# USE_GT_FOR_DAV = False
# output_dir = f"/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_outputs_metric_3dpw_train_v5_det_dav/job_{JOB_CHUNK_NUMBER}/" #correct bgr to rgb v3
# plots_output_dir = f"/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_outputs_metric_plots_3dpw_train_v5_det_dav/job_{JOB_CHUNK_NUMBER}/"


# output_dir = f"/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_outputs_metric_3dpw_v3/job_{JOB_CHUNK_NUMBER}/" #v3 correct rgb and scale
# plots_output_dir = f"/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_outputs_metric_plots_3dpw/job_{JOB_CHUNK_NUMBER}/"


# output_dir = "/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_outputs_metric/"
# plots_output_dir = "/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_outputs_metric_plots/"


## Load detection models for later

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
BBOX_THRESH = 0.03
# BBOX_THRESH = 0.08
NMS_THRESH = 0.3

import uuid

# Dynamically set TEMP_FILE_NAME with a random UUID
TEMP_FILE_NAME = f"/tmp/temp_{uuid.uuid4().hex}.png"

# Debug: Set to True to save and visualize first few frames
DEBUG_MODE = False
DEBUG_FRAMES_COUNT = 50
processed_frames = 0



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
# dataset = 'vkitti'  # Change from 'hypersim' to 'vkitti' for outdoor model
max_depth = 20 # 20 for indoor model, 80 for outdoor model
# max_depth=80

model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
model.load_state_dict(torch.load(f'/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth', map_location='cpu'))
# model.load_state_dict(torch.load(f'/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/checkpoints/depth_anything_v2_metric_vkitti_vitl.pth', map_location='cpu'))
# model.load_state_dict(torch.load(f'/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/checkpoints/depth_anything_v2_metric_hypersim_vits.pth', map_location='cpu'))
model.cuda()
model.eval()


def keypoints_to_bboxes(gt_keypoints, image_shape, margin=150):
    """Derive bounding boxes from keypoints with margin, clipped to image bounds."""
    h, w = image_shape
    bboxes = []

    # Ensure gt_keypoints is a batch of keypoints
    if gt_keypoints.ndim == 2:
        gt_keypoints = gt_keypoints[np.newaxis, :]  # Add batch dimension if needed
    
    for kpts in gt_keypoints:
        x_min, y_min = kpts[:, 0].min(), kpts[:, 1].min()
        x_max, y_max = kpts[:, 0].max(), kpts[:, 1].max()

        # Apply margin and clip to image boundaries
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(w - 1, x_max + margin)
        y_max = min(h - 1, y_max + margin)

        bboxes.append([x_min, y_min, x_max, y_max])

    return np.array(bboxes)



# Function to process each framen
def process_frame_np(frame, gt_bboxes):
    
        
    if frame.shape[0] == 3:  # Check if input is (C, H, W)
        frame_np = frame.transpose(1, 2, 0)  # Convert to (H, W, C)
    else:
        frame_np = frame  # Already in (H, W, C), no need to transpose

    # Convert frame_np from RGB to BGR for RTMDet
    frame_np_bgr = frame_np[..., ::-1].copy()  # Reverse the last dimension to swap RGB to BGR

    # ipdb.set_trace()
    # Perform detection using BGR image
    # Use ground truth bounding boxes if provided
    if gt_bboxes is not None and len(gt_bboxes) > 0:
        # print("using provided bbox")
        bboxes = gt_bboxes
    else:
        # Fallback to detection if no ground truth boxes provided
        det_results = inference_detector(DETECTION_MODEL, frame_np_bgr)
        pred_instance = det_results.pred_instances.detach().cpu().numpy()
        bboxes = np.concatenate((pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
        bboxes = bboxes[np.logical_and(pred_instance.labels == DET_CAT_ID,
                                        pred_instance.scores > BBOX_THRESH)]
        bboxes = bboxes[nms(bboxes, NMS_THRESH), :4]


    
    # Skip if no detections
    if bboxes.size == 0:
        print("No bounding boxes detected for this frame.")
        return None, None, None

    if frame.shape[0] == 3:  # Assuming C, H, W format
        frame = frame.permute(1, 2, 0)  # Convert to H, W, C
    if frame.dtype == torch.float32:  # Assuming tensor is in [0, 1]
        frame = (frame * 255).byte()  # Scale to [0, 255] and convert to uint8

    # ipdb.set_trace()
    # Convert BGR back to RGB for pose estimation
    # frame_rgb = frame_np*255  # Reverse channels to convert BGR to RGB
    # Ensure the array is in the correct data type (uint8) and range (0-255)
    frame_rgb_uint8 = (frame_np * 255).astype(np.uint8)  # Scale to [0, 255] and convert to uint8
    # Save the RGB frame as an image for pose estimation
    pil_frame = Image.fromarray(frame_rgb_uint8)

    # pil_frame = Image.fromarray(frame.detach().cpu().numpy()*255)
    pil_frame.save(TEMP_FILE_NAME)
    pose_results = inference_topdown(pose_model, TEMP_FILE_NAME, bboxes)

    data_samples = merge_data_samples(pose_results)


    # Assume function to extract and preprocess bboxes from detections
    # Extract keypoints and scores
    keypoints = data_samples.pred_instances.keypoints
    keypoints_score = data_samples.pred_instances.keypoint_scores

    # ipdb.set_trace()
    # Find the proposal with the highest maximum score
    mean_scores = keypoints_score.mean(axis=1)  # Get mean score for each proposal
    argmax_index = mean_scores.argmax()  # Index of proposal with highest score

    # ipdb.set_trace()
    # Apply threshold check
    # if mean_scores[argmax_index] > 0.4: #some may be missing
    # if mean_scores:
    #     # Select keypoints and scores of the best proposal
    #     keypoints = keypoints[argmax_index]
    #     keypoints_score = keypoints_score[argmax_index]
    # else:
    #     # No valid detections
    #     keypoints = None
    #     keypoints_score = None
    keypoints = keypoints[argmax_index]
    keypoints_score = keypoints_score[argmax_index]

    # ipdb.set_trace()
    return keypoints, keypoints_score, data_samples

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

import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_keypoints_and_bboxes(img_path, gt_keypoints, pred_keypoints, gt_bboxes, output_path="2_18_test1.png"):
    """Plot GT keypoints (blue), predicted keypoints (red), and GT bounding boxes (red) on the image."""
    # Load the image
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Failed to load image: {img_path}")
        return

    # Convert BGR to RGB for plotting
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 10))
    plt.imshow(img_rgb)

    # Plot ground truth keypoints in blue
    if gt_keypoints is not None:
        for kp in gt_keypoints:
            plt.scatter(kp[0], kp[1], color='blue', s=40, label='GT Keypoints')

    # Plot predicted keypoints in red
    if pred_keypoints is not None:
        for kp in pred_keypoints:
            plt.scatter(kp[0], kp[1], color='red', s=40, label='Predicted Keypoints')

    # Plot ground truth bounding boxes in red
    if gt_bboxes is not None:
        for bbox in gt_bboxes:
            x_min, y_min, x_max, y_max = bbox
            rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                                 linewidth=2, edgecolor='red', facecolor='none', label='GT BBox')
            plt.gca().add_patch(rect)

    # Add legend
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    plt.legend(unique_labels.values(), unique_labels.keys(), loc='upper right')

    # Save the plotted image
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    # print(f"✅ Saved visualization to {output_path}")


def convert_coco_to_h36m_2d(keypoints):
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

def get_min_depth_from_circle(depth_map, keypoint, radius=3):
    """Extracts the minimum depth value from a circular region around a keypoint."""
    x, y = keypoint
    H, W = depth_map.shape
    neighbors = []

    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if dx**2 + dy**2 <= radius**2:  # Ensure it's within the circle
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H:  # Ensure within bounds
                    neighbors.append(depth_map[ny, nx])

    return min(neighbors) if neighbors else depth_map[y, x]  # Default to original depth if empty

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


def get_casp_depth(depth_map, keypoint, confidence, person_scale, r_min=3, r_max=30, 
                   k=0.04, beta=3.0, gamma=1.5, num_samples=100):
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
        r_min: minimum sampling radius in pixels (default: 3)
        r_max: maximum sampling radius in pixels (default: 30)
        k: scale multiplier (default: 0.04)
        beta: confidence sensitivity (default: 3.0)
        gamma: confidence exponent (default: 1.5)
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


def scale_bboxes_like_pad_resize(gt_bboxes, original_shape, target_size=(384, 288)):
    """Scale and pad bounding boxes using the same logic as pad_resize."""
    h, w = original_shape
    target_w, target_h = target_size

    # Compute scale from original image dimensions
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2

    # Apply the scaling and padding to the bounding boxes
    scaled_bboxes = gt_bboxes.copy()
    scaled_bboxes[:, 0] = gt_bboxes[:, 0] * scale + pad_x
    scaled_bboxes[:, 1] = gt_bboxes[:, 1] * scale + pad_y
    scaled_bboxes[:, 2] = gt_bboxes[:, 2] * scale + pad_x
    scaled_bboxes[:, 3] = gt_bboxes[:, 3] * scale + pad_y

    return scaled_bboxes

def unpad_and_scale_keypoints(pred_keypoints, original_shape, target_size=(384, 288)):
    """Scale and unpad predicted keypoints back to the original image size."""
    h, w = original_shape
    target_w, target_h = target_size

    # Compute the same scale and padding
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2

    # Reverse the scaling and padding
    original_keypoints = pred_keypoints.copy()
    original_keypoints[:, 0] = (pred_keypoints[:, 0] - pad_x) / scale
    original_keypoints[:, 1] = (pred_keypoints[:, 1] - pad_y) / scale

    return original_keypoints


# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plots_output_dir, exist_ok=True)


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

# Shuffle image paths in debug mode for better variation
if DEBUG_MODE:
    import random
    random.seed(42)  # Set seed for reproducibility
    print(f"\n🔍 DEBUG MODE ENABLED: Will process {DEBUG_FRAMES_COUNT} frames and break for inspection")
    random.shuffle(image_paths)
    print(f"📊 Shuffled {len(image_paths)} images for better variation\n")

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
BBOX_THRESH = 0.03
# BBOX_THRESH = 0.3 # fixed for highres
NMS_THRESH = 0.3

import uuid

# Dynamically set TEMP_FILE_NAME with a random UUID
TEMP_FILE_NAME = f"/tmp/temp_{uuid.uuid4().hex}.png"


### This code is to grab the groundtruth depths in v5. (Grab detected depths too?)
train_ann_file = '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_3dpw_all_v3.npz'
# train_ann_file = '/srv/essa-lab/flash3/nwarner30/pose_estimation/3dpw_data/processed_mmpose_shards/processed_3dpw_trainval_v2.npz'
train_annotations = np.load(train_ann_file, allow_pickle=True)
combined_imgnames = train_annotations['imgname']
combined_keypoints = train_annotations['part']
# Track files with exceeded keypoints
exceeding_files = []


# Track failed image reads
failed_images = []




def find_matching_participants(formatted_name, combined_imgnames):
    # Helper function to parse filename components
    def parse_filename(filename):
        parts = filename.split('_')
        setting = parts[0]
        action = parts[1]
        vidnum = parts[2]
        participant = parts[3]
        frame = parts[-1].replace('frame', '').replace('.jpg', '')
        return setting, action, vidnum, participant, frame

    # Parse the target filename
    target_setting, target_action, target_vidnum, _, target_frame = parse_filename(formatted_name)

    # Find matches with the same setting, action, vidnum, and frame
    matches = []
    for name in combined_imgnames:
        setting, action, vidnum, participant, frame = parse_filename(name)
        if (setting == target_setting and action == target_action 
            and vidnum == target_vidnum and frame == target_frame):
            matches.append(name)

    return matches


# Track files with exceeded keypoints
exceeding_files = []
failed_images = []


# Initialize a cache for frames with OOB keypoints
oob_frames = []

for i, img_path in enumerate(tqdm(image_paths, desc="Processing Images")):

    raw_img = cv2.imread(img_path)
    if raw_img is None:
        print(f"❌ Failed to read image: {img_path}")
        failed_images.append(img_path)
        continue

    # Compute depth map once per image
    with torch.no_grad():
        depth_map = model.infer_image(raw_img)

    img_name = os.path.basename(img_path)
    matches = find_matching_participants(img_name, combined_imgnames)

    if not matches:
        print(f"No matches found for {img_name}. Skipping frame.")
        continue

    # Precompute padded/resized image once
    resized_img, scale, pad_x, pad_y = pad_resize(raw_img, target_size=(384, 288))
    resized_img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

    for match in matches:
        if match not in combined_imgnames:
            print(f"Match {match} not in combined images, skipping.")
            continue

        idx = np.where(combined_imgnames == match)[0][0]
        gt_keypoints = combined_keypoints[idx][..., :-1].astype(int)

        image_shape = raw_img.shape[:2]
        gt_bboxes = keypoints_to_bboxes(gt_keypoints, image_shape, margin=150)
        scaled_gt_bboxes = scale_bboxes_like_pad_resize(gt_bboxes, raw_img.shape[:2])

        if not USE_GT_FOR_DAV:
            # det_results = inference_detector(DETECTION_MODEL, raw_img)
            # pred_np = det_results.pred_instances.detach().cpu().numpy()
            # bboxes = np.concatenate((pred_np.bboxes, pred_np.scores[:, None]), axis=1)
            # bboxes = bboxes[(pred_np.labels == DET_CAT_ID) & (pred_np.scores > BBOX_THRESH)]
            # bboxes = bboxes[nms(bboxes, NMS_THRESH), :4]
            bboxes = gt_bboxes  # shape (1,4) in raw-image coords
            # This avoids wrong person detections!

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

            # keypoints, keypoints_score, data_samples = process_frame_np(resized_img_rgb, scaled_gt_bboxes)
            # original_keypoints = unpad_and_scale_keypoints(keypoints, raw_img.shape[:2])

            # # Ensure keypoints are within bounds
            # H, W = depth_map.shape
            # original_keypoints[:, 0] = np.clip(original_keypoints[:, 0], 0, W - 1)
            # original_keypoints[:, 1] = np.clip(original_keypoints[:, 1], 0, H - 1)

            # keypoints_h36m = convert_coco_to_h36m_2d(original_keypoints[None, ...])[0]
            keypoints_h36m = keypoints_h36m.astype(int)
            # keypoints_conf_h36m = convert_coco_to_h36m_confidence(keypoints_score[None, ...])[0]

            if PLOT_EXAMPLES and i % 10 == 0:
                output_path = os.path.join(plots_output_dir, f"example_{i}.png")
                plot_keypoints_and_bboxes(img_path, gt_keypoints, keypoints, gt_bboxes, output_path=output_path)
                print("plotted", output_path)
                # ipdb.set_trace()

            # if PLOT_EXAMPLES and i % 10 == 0:

            #     output_path = f"/srv/essa-lab/flash3/nwarner30/pose_estimation/2_18_test_{int(i)}.png"
            #     plot_keypoints_and_bboxes(img_path, gt_keypoints, keypoints, gt_bboxes, output_path=output_path)
            #     ipdb.set_trace()
            
            # ─── CLIP TO DEPTH-MAP BOUNDS ───
            H, W = depth_map.shape
            keypoints_h36m[:, 0] = np.clip(keypoints_h36m[:, 0], 0, W - 1)
            keypoints_h36m[:, 1] = np.clip(keypoints_h36m[:, 1], 0, H - 1)

        else:
            keypoints_h36m = gt_keypoints
            keypoints_conf_h36m = None
            
            # Check if keypoints need clipping
            H, W = depth_map.shape
            x_exceed = (keypoints_h36m[:, 0] < 0) | (keypoints_h36m[:, 0] >= W)
            y_exceed = (keypoints_h36m[:, 1] < 0) | (keypoints_h36m[:, 1] >= H)

            if x_exceed.any() or y_exceed.any():
                print(f"⚠️ Clipping GT keypoints for {match}: X out-of-bounds: {keypoints_h36m[x_exceed, 0]}, Y out-of-bounds: {keypoints_h36m[y_exceed, 1]}")
                oob_frames.append({
                    "match": match,
                    "x_out_of_bounds": keypoints_h36m[x_exceed, 0].tolist(),
                    "y_out_of_bounds": keypoints_h36m[y_exceed, 1].tolist()
                })

            # Apply clipping
            keypoints_h36m[:, 0] = np.clip(keypoints_h36m[:, 0], 0, W - 1)
            keypoints_h36m[:, 1] = np.clip(keypoints_h36m[:, 1], 0, H - 1)

        keypoints_depth_exact = depth_map[keypoints_h36m[:, 1], keypoints_h36m[:, 0]]
        
        # Compute person scale (torso length) for scale-adaptive CASP
        person_scale = compute_person_scale(keypoints_h36m)
        
        # Extract depth values using CASP (Confidence-Adaptive & Scale-Adaptive Sampling Procedure)
        keypoints_depth_casp = []
        casp_descriptors = []
        
        # Handle case where confidence is None (GT keypoints)
        if keypoints_conf_h36m is None:
            # Use default confidence of 1.0 for GT keypoints
            keypoints_conf_h36m = np.ones(17)
        
        for kp, conf in zip(keypoints_h36m, keypoints_conf_h36m):
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
        
        # Extract depth values using a circular neighborhood (legacy method)
        keypoints_depth_min_circle = np.array([get_min_depth_from_circle(depth_map, kp) for kp in keypoints_h36m])
        
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
        
        # Debug visualization for first few frames (comprehensive 8-panel, only process first match per image)
        if DEBUG_MODE and processed_frames < DEBUG_FRAMES_COUNT:
            # Convert BGR to RGB for plotting
            img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
            
            # Find the joint with highest variability (MAD) for detailed analysis
            mad_values = [desc['mad'] if desc is not None else 0 for desc in casp_descriptors]
            highest_var_idx = np.argmax(mad_values)
            highest_var_val = mad_values[highest_var_idx]
            highest_var_conf = keypoints_conf_h36m[highest_var_idx]
            highest_var_desc = casp_descriptors[highest_var_idx]
            
            # Create comprehensive 8-panel visualization
            fig = plt.figure(figsize=(24, 16))
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
            
            # 1. Original image with keypoints (top-left)
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.imshow(img_rgb)
            ax1.scatter(keypoints_h36m[:, 0], keypoints_h36m[:, 1], 
                       c='red', s=80, marker='o', edgecolors='white', linewidths=2)
            for j_idx, kp in enumerate(keypoints_h36m):
                ax1.text(kp[0]+5, kp[1]-5, str(j_idx), color='yellow', fontsize=8, weight='bold')
            ax1.set_title(f'Original Image ({img_rgb.shape[1]}x{img_rgb.shape[0]})', fontsize=12)
            ax1.axis('off')
            
            # 2. Depth map visualization (top-middle)
            ax2 = fig.add_subplot(gs[0, 1])
            depth_display = ax2.imshow(depth_map, cmap='plasma', vmin=depth_map.min(), vmax=depth_map.max())
            
            # Show circles with sizes proportional to CASP radius
            for kp, desc in zip(keypoints_h36m, casp_descriptors):
                if desc is not None:
                    radius = desc['radius_used']
                    circle = plt.Circle((kp[0], kp[1]), radius, color='cyan', fill=False, linewidth=1.5, alpha=0.8)
                    ax2.add_patch(circle)
                    # Add small center marker
                    ax2.scatter(kp[0], kp[1], c='white', s=20, marker='+', linewidths=2)
            
            plt.colorbar(depth_display, ax=ax2, label='Depth (m)', fraction=0.046)
            ax2.set_title(f'Depth Map with CASP Radii (range: [{depth_map.min():.2f}, {depth_map.max():.2f}]m)', fontsize=12)
            ax2.axis('off')
            
            # 3. CASP sampling regions for highest variance joint (top-right)
            ax3 = fig.add_subplot(gs[0, 2])
            zoom_size = 100
            hv_kp = keypoints_h36m[highest_var_idx]
            x_min = max(0, int(hv_kp[0]) - zoom_size)
            x_max = min(depth_map.shape[1], int(hv_kp[0]) + zoom_size)
            y_min = max(0, int(hv_kp[1]) - zoom_size)
            y_max = min(depth_map.shape[0], int(hv_kp[1]) + zoom_size)
            
            depth_crop = depth_map[y_min:y_max, x_min:x_max]
            img_crop = img_rgb[y_min:y_max, x_min:x_max]
            
            ax3.imshow(img_crop, alpha=0.6)
            depth_overlay = ax3.imshow(depth_crop, cmap='plasma', alpha=0.4, 
                                       vmin=depth_map.min(), vmax=depth_map.max())
            
            if highest_var_desc is not None:
                radius = highest_var_desc['radius_used']
                circle = plt.Circle((hv_kp[0] - x_min, hv_kp[1] - y_min), radius, 
                                   color='cyan', fill=False, linewidth=2, label=f'CASP radius={radius:.1f}px')
                ax3.add_patch(circle)
                
                ax3.scatter(hv_kp[0] - x_min, hv_kp[1] - y_min, c='red', s=100, marker='x', linewidths=3)
                
                for q_name in ['Q10', 'Q25', 'Q50', 'Q75', 'Q90']:
                    q_xy = highest_var_desc[f'{q_name}_xy']
                    ax3.scatter(q_xy[0] - x_min, q_xy[1] - y_min, s=50, marker='o', 
                               edgecolors='white', linewidths=1.5, alpha=0.8)
            
            ax3.set_title(f'Joint {highest_var_idx} CASP Sampling\n(MAD={highest_var_val:.4f}m, conf={highest_var_conf:.2f})', fontsize=11)
            ax3.legend(fontsize=8)
            ax3.axis('off')
            
            # 4. Depth comparison across all joints (middle-left, spans 2 columns)
            ax4 = fig.add_subplot(gs[1, :2])
            joint_indices = np.arange(17)
            
            ax4.plot(joint_indices, keypoints_depth_exact, 'o-', color='orange', 
                    label='Exact (single pixel)', markersize=8, linewidth=2, zorder=3)
            ax4.plot(joint_indices, keypoints_depth_casp, 's-', color='blue', 
                    label='CASP (median)', markersize=8, linewidth=2, zorder=3)
            
            q10_vals = [desc['Q10'] if desc else 0 for desc in casp_descriptors]
            q25_vals = [desc['Q25'] if desc else 0 for desc in casp_descriptors]
            q75_vals = [desc['Q75'] if desc else 0 for desc in casp_descriptors]
            q90_vals = [desc['Q90'] if desc else 0 for desc in casp_descriptors]
            
            ax4.fill_between(joint_indices, q25_vals, q75_vals, alpha=0.3, color='blue', label='CASP IQR (Q25-Q75)')
            
            lower_err = [keypoints_depth_casp[i] - q10_vals[i] for i in range(17)]
            upper_err = [q90_vals[i] - keypoints_depth_casp[i] for i in range(17)]
            ax4.errorbar(joint_indices, keypoints_depth_casp, yerr=[lower_err, upper_err],
                        fmt='none', ecolor='blue', alpha=0.5, capsize=3, linewidth=1.5, 
                        label='CASP Range (Q10-Q90)', zorder=2)
            
            ax4.scatter(highest_var_idx, keypoints_depth_exact[highest_var_idx], 
                       s=300, marker='o', facecolors='none', edgecolors='red', linewidths=3, zorder=4)
            ax4.scatter(highest_var_idx, keypoints_depth_casp[highest_var_idx], 
                       s=300, marker='s', facecolors='none', edgecolors='red', linewidths=3, zorder=4)
            
            ax4.set_xlabel('Joint Index', fontsize=11)
            ax4.set_ylabel('Depth (m)', fontsize=11)
            ax4.set_title(f'Depth Estimates Across All Joints (Person Scale={person_scale:.0f}px)', fontsize=12)
            ax4.legend(fontsize=9, loc='best')
            ax4.grid(True, alpha=0.3)
            
            # 5. Confidence scores (middle-right)
            ax5 = fig.add_subplot(gs[1, 2])
            conf_colors = plt.cm.RdYlGn(keypoints_conf_h36m)
            bars = ax5.bar(joint_indices, keypoints_conf_h36m, color=conf_colors, edgecolor='black', linewidth=1)
            ax5.axhline(y=0.5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Low conf threshold')
            ax5.set_xlabel('Joint Index', fontsize=11)
            ax5.set_ylabel('Confidence Score', fontsize=11)
            ax5.set_title(f'2D Pose Confidence\n(Avg={keypoints_conf_h36m.mean():.3f})', fontsize=12)
            ax5.set_ylim([0, 1])
            ax5.legend(fontsize=9)
            ax5.grid(True, alpha=0.3, axis='y')
            
            # 6. Adaptive radius per joint (bottom-left)
            ax6 = fig.add_subplot(gs[2, 0])
            radii = [desc['radius_used'] if desc else 0 for desc in casp_descriptors]
            radius_colors = plt.cm.viridis((np.array(radii) - min(radii)) / (max(radii) - min(radii) + 1e-8))
            bars = ax6.bar(joint_indices, radii, color=radius_colors, edgecolor='black', linewidth=1)
            ax6.set_xlabel('Joint Index', fontsize=11)
            ax6.set_ylabel('CASP Radius (pixels)', fontsize=11)
            ax6.set_title(f'Adaptive Sampling Radius per Joint\n(range: [{min(radii):.1f}, {max(radii):.1f}]px)', fontsize=12)
            ax6.grid(True, alpha=0.3, axis='y')
            
            # 7. MAD (uncertainty) per joint (bottom-middle)
            ax7 = fig.add_subplot(gs[2, 1])
            mad_colors = plt.cm.Reds((np.array(mad_values) - min(mad_values)) / (max(mad_values) - min(mad_values) + 1e-8))
            bars = ax7.bar(joint_indices, mad_values, color=mad_colors, edgecolor='black', linewidth=1)
            ax7.bar(highest_var_idx, mad_values[highest_var_idx], color='red', edgecolor='black', linewidth=2)
            ax7.set_xlabel('Joint Index', fontsize=11)
            ax7.set_ylabel('MAD (m)', fontsize=11)
            ax7.set_title(f'Depth Uncertainty (MAD)\n(Joint {highest_var_idx} has highest: {highest_var_val:.4f}m)', fontsize=12)
            ax7.grid(True, alpha=0.3, axis='y')
            
            # 8. Occlusion asymmetry (bottom-right)
            ax8 = fig.add_subplot(gs[2, 2])
            occ_asym = [desc['occlusion_asymmetry'] if desc else 0 for desc in casp_descriptors]
            colors = ['red' if val < -0.1 else 'orange' if val < 0 else 'green' for val in occ_asym]
            bars = ax8.bar(joint_indices, occ_asym, color=colors, edgecolor='black', linewidth=1)
            ax8.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax8.axhline(y=-0.1, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Likely occluded')
            ax8.set_xlabel('Joint Index', fontsize=11)
            ax8.set_ylabel('Occlusion Asymmetry (Q10 - median)', fontsize=11)
            ax8.set_title('Occlusion Detection\n(negative = potential occluder)', fontsize=12)
            ax8.legend(fontsize=9)
            ax8.grid(True, alpha=0.3, axis='y')
            
            plt.suptitle(f'CASP Debug Frame {processed_frames} - 3DPW', fontsize=14, weight='bold', y=0.995)
            
            debug_path = os.path.join(plots_output_dir, f'debug_frame_{processed_frames:03d}.png')
            plt.savefig(debug_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"\n{'='*60}")
            print(f"Debug frame {processed_frames} saved to: {debug_path}")
            print(f"  Person scale: {person_scale:.1f}px")
            print(f"  Avg confidence: {keypoints_conf_h36m.mean():.3f}")
            print(f"  Depth range (exact): [{keypoints_depth_exact.min():.2f}, {keypoints_depth_exact.max():.2f}]m")
            print(f"  Depth range (CASP):  [{keypoints_depth_casp.min():.2f}, {keypoints_depth_casp.max():.2f}]m")
            print(f"  Highest variance joint: {highest_var_idx} (MAD={highest_var_val:.4f}m, conf={highest_var_conf:.2f})")
            print(f"{'='*60}\n")
            
            processed_frames += 1
            
            if processed_frames == DEBUG_FRAMES_COUNT:
                ipdb.set_trace()
                break  # Exit inner match loop after debug frames

        output_path = os.path.join(output_dir, f"{match.replace('.jpg', '_depth.npz')}")
        
        # Prepare data to save
        save_data = {
            'keypoints': keypoints_h36m, 
            'keypoints_score': keypoints_conf_h36m, 
            'keypoints_depth': keypoints_depth_casp,  # Now using CASP depth
            'keypoints_depth_exact': keypoints_depth_exact,
            'casp_descriptors': casp_descriptors,  # Save full CASP statistics (list of dicts)
            'summary_casp_descriptor_10d': summary_casp_descriptor_10d  # Compact (17, 10) array
        }
        
        np.savez_compressed(output_path, **save_data)
    
    # Break outer loop if debug frames complete
    if DEBUG_MODE and processed_frames >= DEBUG_FRAMES_COUNT:
        break

# Save results
exceeding_files_path = os.path.join(output_dir, "exceeding_keypoints.npz")
np.savez_compressed(exceeding_files_path, exceeding_files=exceeding_files)

failed_images_path = os.path.join(output_dir, "failed_images.txt")
with open(failed_images_path, "w") as f:
    f.writelines(f"{img}\n" for img in failed_images)

print(f"✅ Processing completed: {len(image_paths)} images")

# Save the OOB frames to a JSON file
import json

oob_frames_path = os.path.join(output_dir, "oob_frames.json")
with open(oob_frames_path, "w") as f:
    json.dump(oob_frames, f, indent=4)

print(f"✅ Cached {len(oob_frames)} OOB frames to {oob_frames_path}")

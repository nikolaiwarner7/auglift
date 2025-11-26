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
import uuid
import json

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

# Configuration: Set to True to cache full depth maps (warning: large file sizes ~8MB per image)
CACHE_FULL_DEPTH_MAPS = True
# Depth map optimization options (when CACHE_FULL_DEPTH_MAPS is True)
DOWNSAMPLE_DEPTH = True  # Downsample to 50% of original dimensions (75% size reduction)
USE_FLOAT16 = True       # Use float16 instead of float32 (50% size reduction)
                        # Combined: ~87.5% size reduction per depth map

# output_dir = f"/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_outputs_metric_3dpw_train_v5_det_dav_hr_test_dbg/job_{JOB_CHUNK_NUMBER}/" #correct bgr to rgb v3
# plots_output_dir = f"/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_outputs_metric_plots_3dpw_train_v5_det_dav_hr_test_dbg/job_{JOB_CHUNK_NUMBER}/"

output_dir = f"/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_outputs_metric_3dpw_train_v7_debug_oct_v2/job_{JOB_CHUNK_NUMBER}/" #correct bgr to rgb v3
plots_output_dir = f"/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_outputs_metric_plots_3dpw_train_v7_debug_oct_v2/job_{JOB_CHUNK_NUMBER}/"




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
model.load_state_dict(torch.load(f'/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth', map_location='cpu'), strict=False)
# model.load_state_dict(torch.load(f'/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/checkpoints/depth_anything_v2_metric_vkitti_vitl.pth', map_location='cpu'), strict=False)
# model.load_state_dict(torch.load(f'/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/checkpoints/depth_anything_v2_metric_hypersim_vits.pth', map_location='cpu'), strict=False)
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
        depth_map, dav2_feature_256 = model.infer_image_with_features(raw_img)
        det_feature_coarse = None  # Initialize
        
        # Run detection once per image to cache features
        det_results = inference_detector(DETECTION_MODEL, raw_img)
        pred_np = det_results.pred_instances.detach().cpu().numpy()
        
        # Cache detection feature maps (only coarsest scale)
        det_features = DETECTION_MODEL.cached_features
        if det_features is not None:
            # Only keep the coarsest feature map and convert to float16
            det_feature_coarse = det_features[-1].detach().cpu().numpy().astype(np.float16)
        else:
            det_feature_coarse = None

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
        keypoints_depth_min_circle = np.array([get_min_depth_from_circle(depth_map, kp) for kp in keypoints_h36m])

        output_path = os.path.join(output_dir, f"{match.replace('.jpg', '_depth.npz')}")
        
        # Prepare data to save
        save_data = {
            'keypoints': keypoints_h36m, 
            'keypoints_score': keypoints_conf_h36m, 
            'keypoints_depth': keypoints_depth_min_circle, 
            'keypoints_depth_exact': keypoints_depth_exact
        }
        
        # Add coarsest detection feature if available (fp16 for efficiency)
        if det_feature_coarse is not None:
            save_data['det_feature_coarse'] = det_feature_coarse
        
        # Add DAV2 256-D feature map (fp16)
        if USE_FLOAT16:
            save_data['dav2_feature_256'] = dav2_feature_256.astype(np.float16)
        else:
            save_data['dav2_feature_256'] = dav2_feature_256
        
        # Optionally include full depth map with optimizations
        if CACHE_FULL_DEPTH_MAPS:
            optimized_depth = depth_map.copy()
            
            # Downsample if requested (scale to 50% to save space)
            if DOWNSAMPLE_DEPTH:
                H, W = optimized_depth.shape
                # Scale to 50% of original dimensions
                new_h, new_w = int(H * 0.5), int(W * 0.5)
                optimized_depth = cv2.resize(optimized_depth, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # Convert to float16 if requested
            if USE_FLOAT16:
                optimized_depth = optimized_depth.astype(np.float16)
            
            save_data['depth_map'] = optimized_depth
        
        np.savez_compressed(output_path, **save_data)

# Save results
exceeding_files_path = os.path.join(output_dir, "exceeding_keypoints.npz")
np.savez_compressed(exceeding_files_path, exceeding_files=exceeding_files)

failed_images_path = os.path.join(output_dir, "failed_images.txt")
with open(failed_images_path, "w") as f:
    f.writelines(f"{img}\n" for img in failed_images)

print(f"✅ Processing completed: {len(image_paths)} images")

# Save the OOB frames to a JSON file
oob_frames_path = os.path.join(output_dir, "oob_frames.json")
with open(oob_frames_path, "w") as f:
    json.dump(oob_frames, f, indent=4)

print(f"✅ Cached {len(oob_frames)} OOB frames to {oob_frames_path}")

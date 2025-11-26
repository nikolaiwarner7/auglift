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
# from jan25_utilities import process_frame_np

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

# from pytorch_fid.fid_score import get_activations_from_images, calculate_frechet_distance
# from pytorch_fid.inception import InceptionV3


# Directory containing images
img_root = "/srv/essa-lab/flash3/nwarner30/pose_estimation/data/test_images_raw"

JOB_CHUNK_NUMBER = int(os.getenv("JOB_CHUNK_NUMBER", 0))  # Default to 0 if not set
NUM_JOBS = 8

# Configuration: Set to True to cache full depth maps (warning: large file sizes ~8MB per image)
CACHE_FULL_DEPTH_MAPS = True
# Depth map optimization options (when CACHE_FULL_DEPTH_MAPS is True)
DOWNSAMPLE_DEPTH = True  # Downsample to 1024x1024 (75% size reduction)
USE_FLOAT16 = True       # Use float16 instead of float32 (50% size reduction)
                        # Combined: ~85% size reduction (13MB → ~2MB per file)

# output_dir = f"/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_outputs_metric_3dhp_v4_highres/job_{JOB_CHUNK_NUMBER}/"
# output_dir = f"/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_outputs_metric_3dhp_v5_cache_full/job_{JOB_CHUNK_NUMBER}/"
# output_dir = f"/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_outputs_metric_3dhp_v8_cache_maps/job_{JOB_CHUNK_NUMBER}/"
output_dir = f"/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_outputs_metric_3dhp_v10_cache_maps/job_{JOB_CHUNK_NUMBER}/"

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
model.load_state_dict(torch.load(f'/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth', map_location='cpu'), strict=False)
# model.load_state_dict(torch.load(f'/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/checkpoints/depth_anything_v2_metric_hypersim_vits.pth', map_location='cpu'), strict=False)
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



# Process each image
for img_path in tqdm(image_paths, desc=f"Job {JOB_CHUNK_NUMBER} Processing Images"):
    raw_img = cv2.imread(img_path)

    # Run inference
    with torch.no_grad():
        depth_map, dav2_feature_256 = model.infer_image_with_features(raw_img)
        
        # Pad-resize image while preserving aspect ratio

        det_results = inference_detector(DETECTION_MODEL, raw_img)
        pred_np = det_results.pred_instances.detach().cpu().numpy()
        bboxes = np.concatenate((pred_np.bboxes, pred_np.scores[:, None]), axis=1)
        bboxes = bboxes[(pred_np.labels == DET_CAT_ID) & (pred_np.scores > BBOX_THRESH)]
        bboxes = bboxes[nms(bboxes, NMS_THRESH), :4]

        # If no person is detected, skip this image
        if len(bboxes) == 0:
            print(f"⚠️  Warning: No person detected in {img_path}, skipping.")
            continue

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
        selected_bbox = bboxes[sel]  # Store the bbox for this instance

        # Cache RTMPose feature maps (only coarsest scale from backbone)
        # NOTE: Features are extracted from the CROPPED bbox region, not full image!
        rtm_features = pose_model.cached_features
        if rtm_features is not None:
            # cached_features is a tuple: (feature_pyramid, ...) where feature_pyramid is a list/tuple
            # Extract coarsest feature map: feats[0][0] gives shape (1, 1024, 12, 9)
            if isinstance(rtm_features, (tuple, list)) and len(rtm_features) > 0:
                feature_pyramid = rtm_features[0]
                rtm_feature_coarse = feature_pyramid[0].detach().cpu().numpy().astype(np.float16)
            else:
                rtm_feature_coarse = None
        else:
            rtm_feature_coarse = None

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



    # Extract depth values using a circular neighborhood
    keypoints_depth_min_circle = np.array([get_min_depth_from_circle(depth_map, kp) for kp in keypoints_h36m_int])

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
    
    # Prepare data to save
    save_data = {
        'keypoints': keypoints_h36m, 
        'keypoints_score': keypoints_conf_h36m, 
        'keypoints_depth': keypoints_depth_min_circle, 
        'keypoints_depth_exact': keypoints_depth_exact,
        'original_img_shape': np.array(raw_img.shape[:2])  # (H, W) of full image
    }
    
    # Add RTMPose backbone feature if available (fp16 for efficiency)
    # Features are from CROPPED bbox patch, so we need bbox coordinates for proper sampling
    if rtm_feature_coarse is not None:
        save_data['rtm_feature_coarse'] = rtm_feature_coarse
        save_data['rtm_bbox'] = selected_bbox  # [x1, y1, x2, y2] in full image coords
    
    # Add DAV2 256-D feature map (fp16)
    save_data['dav2_feature_256'] = dav2_feature_256
    
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

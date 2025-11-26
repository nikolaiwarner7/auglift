import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import ipdb


# gt_ann_file = '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_fit3d_all_v3.npz'
# ann_file = '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_fit3d_all_v3_detections.npz'
# ann_file='/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_fit3d_all_v6_detdav_25hz.npz' #v6.
# ann_file='/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_fit3d_all_v6_detdav_25hz_dets.npz'
# ann_file='/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_fit3d_all_v5_detdav_25hz_hr.npz'
# ann_file='/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_fit3d_all_v6_casp_dets.npz'
# ann_file='/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_fit3d_all_v6_casp_dets.npz'
ann_file='/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_fit3d_all_v6_casp_fullcasp_dets.npz'


annotations = np.load(ann_file, allow_pickle=True)
# gt_annotations = np.load(gt_ann_file, allow_pickle=True)

imgnames = annotations['imgname']
gt_keypoints = annotations['part']

pred_keypoints = annotations['predicted_keypoints']


# ipdb.set_trace()
# Replace ground-truth keypoints with predicted keypoints
# Convert to a mutable dictionary
annotations_dict = {key: annotations[key] for key in annotations.files}

# ipdb.set_trace()

# Extract subjects
subjects = [name.split('_')[0] for name in imgnames]

# Count occurrences
unique_subjects, counts = np.unique(subjects, return_counts=True)

# Convert to dictionary
subject_counts = dict(zip(unique_subjects, counts))

print(subject_counts)
# {'s03': 42986, 's04': 119436, 's05': 99299, 's07': 51909, 's08': 94592, 's09': 108348, 's10': 120536, 's11': 127428}

# Define train and validation subjects
train_subjects = {'s03', 's04', 's05', 's07', 's08', 's09'}  # Train set
val_subjects = {'s10', 's11'}  # Validation set

# Create train/val masks
train_mask = np.isin(subjects, list(train_subjects))
val_mask = np.isin(subjects, list(val_subjects))

# Split data
train_data = {key: value[train_mask] for key, value in annotations.items()}
val_data = {key: value[val_mask] for key, value in annotations.items()}


ipdb.set_trace()
# Save new train/val datasets
# np.savez('/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/train_fit3d_v6.npz', **train_data)
# np.savez('/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/val_fit3d_v6.npz', **val_data)

# np.savez('/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/train_fit3d_v6_dets.npz', **train_data)
# np.savez('/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/val_fit3d_v6_dets.npz', **val_data)

# np.savez('/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/train_fit3d_v5_detdav_25hz_hr.npz', **train_data)
# np.savez('/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/val_fit3d_v5_detdav_25hz_hr.npz', **val_data)

# np.savez('/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/train_fit3d_v6_casp_dets.npz', **train_data)
# np.savez('/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/val_fit3d_v6_casp_dets.npz', **val_data)

np.savez('/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/train_fit3d_v6_casp_fullcasp_dets.npz', **train_data)
np.savez('/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/val_fit3d_v6_casp_fullcasp_dets.npz', **val_data)

print(f"Train size: {sum(train_mask)}, Val size: {sum(val_mask)}")




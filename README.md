# AugLift Git Commit CR 11-26
# Root relative paths, to clean up
# Fork off of main mmpose

# External Libraries
# mmpose: https://github.com/open-mmlab/mmpose
# depth anything v2: https://github.com/DepthAnything/Depth-Anything-V2

# Datasets
# 3dpw: 3dpw_data
# h36m: h36m_data
# 3dhp: data
# fit3d: fit3d_data

# Data Preprocessing
# mmpose preprocessing 3DHP/H36M/3DPW/fit3d
# Depth / 2D pose estimation
# 3dpw/h36m/3dhp/fit3d 1_30...

coarse_depth_experiments/Depth-Anything-OfficialV2/Depth-Anything-V2/metric_depth/1_30_estimate_depth_proto_loop_metric_h36m_v3_sampling_patch_statistics.py
coarse_depth_experiments/Depth-Anything-OfficialV2/Depth-Anything-V2/metric_depth/launch_depth_metric_loop_h36m.sh
coarse_depth_experiments/Depth-Anything-OfficialV2/Depth-Anything-V2/metric_depth/launch_depth_metric_loop_outer_24_h36m.sh

coarse_depth_experiments/Depth-Anything-OfficialV2/Depth-Anything-V2/metric_depth/1_30_estimate_depth_proto_loop_metric_fit3d_v3_sampling_patch_statistics.py
coarse_depth_experiments/Depth-Anything-OfficialV2/Depth-Anything-V2/metric_depth/launch_depth_metric_loop_fit3d.sh
coarse_depth_experiments/Depth-Anything-OfficialV2/Depth-Anything-V2/metric_depth/launch_depth_metric_loop_outer_24_fit3d.sh

coarse_depth_experiments/Depth-Anything-OfficialV2/Depth-Anything-V2/metric_depth/1_30_estimate_depth_proto_loop_metric_3dpw_v3_sampling_patch_statistics.py
coarse_depth_experiments/Depth-Anything-OfficialV2/Depth-Anything-V2/metric_depth/launch_depth_metric_loop_3dpw.sh
coarse_depth_experiments/Depth-Anything-OfficialV2/Depth-Anything-V2/metric_depth/launch_depth_metric_loop_outer_24_3dpw.sh

coarse_depth_experiments/Depth-Anything-OfficialV2/Depth-Anything-V2/metric_depth/1_30_estimate_depth_proto_loop_metric_3dhp_v3_sampling_patch_statistics.py
coarse_depth_experiments/Depth-Anything-OfficialV2/Depth-Anything-V2/metric_depth/launch_depth_metric_loop_3dhp_train.sh
coarse_depth_experiments/Depth-Anything-OfficialV2/Depth-Anything-V2/metric_depth/launch_depth_metric_loop_outer_24_3dhp_train.sh
coarse_depth_experiments/Depth-Anything-OfficialV2/Depth-Anything-V2/metric_depth/launch_depth_metric_loop_3dhp.sh
coarse_depth_experiments/Depth-Anything-OfficialV2/Depth-Anything-V2/metric_depth/launch_depth_metric_loop_outer_24_3dhp.sh

# merge files 2_3
2_3_merge_xycd_npz_3dhp_train.py
2_3_merge_xycd_npz_3dhp.py
2_3_merge_xycd_npz_3dpw.py
2_3_merge_xycd_npz_fit3d.py
2_25_split_fit3d_train_test.py
2_3_merge_xycd_npz_h36m.py

# Dataloaders
mmpose/mmpose/datasets/datasets/base/base_mocap_dataset.py
mmpose/mmpose/datasets/datasets/body3d/mpi_3dhp_inf_dataset.py

# Codecs
mmpose/mmpose/codecs/image_pose_lifting.py
mmpose/mmpose/codecs/poseformer_label.py

# Model Definitions
mmpose/mmpose/models/backbones/poseformer.py
mmpose/mmpose/models/heads/regression_heads/poseformer_regression_head.py

# Pyconfigs
mmpose/configs/body_3d_keypoint/image_pose_lift/h36m/image-pose-lift_tcn_8xb64-200e_h36m_oct25_casp.py
mmpose/configs/body_3d_keypoint/poseformer/h36m/poseformer_h36m_cross_eval_test_config.py

# Training script (.sh)
mmpose/tools/launch_train_cross_datasets_may_25_poseformer_test_outer.sh
mmpose/tools/launch_train_cross_datasets_may_25_pf.sh

# Analysis scripts
10_10_convert_pkl_to_npz_faster.py

# Uncertainty aware descriptors
mmpose/configs/body_3d_keypoint/poseformer/h36m/poseformer_h36m_config_img_depth_baselines.py

# Feature fusion scripts
coarse_depth_experiments/Depth-Anything-OfficialV2/Depth-Anything-V2/metric_depth/1_30_estimate_depth_proto_loop_metric_3dhp_v4_cache_img_aug_baseline_oct_2025.py
coarse_depth_experiments/Depth-Anything-OfficialV2/Depth-Anything-V2/metric_depth/1_30_estimate_depth_proto_loop_metric_3dpw_v4_cache_img_aug_baseline_oct_2025.py
coarse_depth_experiments/Depth-Anything-OfficialV2/Depth-Anything-V2/metric_depth/1_30_estimate_depth_proto_loop_metric_h36m_v4_cache_img_aug_baseline_oct_2025.py

# HRnet experiments
coarse_depth_experiments/Depth-Anything-OfficialV2/Depth-Anything-V2/metric_depth/launch_hrnet_detections_h36m.sh
5_24_merge_xy_npz_h36m_hrnet_v2_detections_add_vis.py

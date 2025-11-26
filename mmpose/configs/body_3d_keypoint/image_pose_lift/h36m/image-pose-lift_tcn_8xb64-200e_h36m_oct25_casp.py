_base_ = ['../../../_base_/default_runtime.py']

import os
PERSPECTIVE_METHOD = os.getenv('PERSPECTIVE_METHOD', '')
CASP_MODE = os.getenv('CASP_MODE', '')  # 'v0' or 'spatial'
print(f"CASP_MODE: {CASP_MODE}")
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '512') or 512)
USE_INVCONF_LOSS = os.getenv('USE_INVCONF_LOSS', 'false').lower() == 'true'
print(f"USE_INVCONF_LOSS: {USE_INVCONF_LOSS}")

if CASP_MODE == 'spatial':
    train_ann_file = '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_h36m_train_v6_casp_fullcasp_dets.npz'
    test_ann_file = '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_h36m_val_v6_casp_fullcasp_dets.npz'
    casp_spatial_flag = True
    NUM_CHANNELS = 17 * 10
elif CASP_MODE == 'v0':
    train_ann_file = '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_h36m_train_v6_casp_dets.npz'
    test_ann_file = '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_h36m_val_v6_casp_dets.npz'
    casp_spatial_flag = False
    NUM_CHANNELS = 17 * 6
else:
    if PERSPECTIVE_METHOD == 'xycd':
        train_ann_file = os.getenv('TRAIN_ANN_FILE', '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_h36m_train_v4_highres_dets.npz')
        test_ann_file = os.getenv('TEST_ANN_FILE', '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_h36m_val_v4_highres_dets.npz')
        NUM_CHANNELS = int(os.getenv('NUM_CHANNELS', 68))
    elif PERSPECTIVE_METHOD == 'xyc':
        train_ann_file = os.getenv('TRAIN_ANN_FILE', '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_h36m_train_v4_highres_dets.npz')
        test_ann_file = os.getenv('TEST_ANN_FILE', '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_h36m_val_v4_highres_dets.npz')
        NUM_CHANNELS = int(os.getenv('NUM_CHANNELS', 51))
    else:
        train_ann_file = os.getenv('TRAIN_ANN_FILE', '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_h36m_train_v4_highres_dets.npz')
        test_ann_file = os.getenv('TEST_ANN_FILE', '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_h36m_val_v4_highres_dets.npz')
        NUM_CHANNELS = int(os.getenv('NUM_CHANNELS', 34))
    casp_spatial_flag = False
    print("No casp!")

# Import the environment variable 'EXPERIMENT_NAME'
EXPERIMENT_NAME = os.getenv('EXPERIMENT_NAME', '')
# Print the experiment name (defaulting to an empty string if not set)
print(f"Experiment Name: {EXPERIMENT_NAME}")

WANDB_PROJECT_NAME = os.getenv('WANDB_PROJECT_NAME', '')
# Print the experiment name (defaulting to an empty string if not set)
print(f"Project Name: {WANDB_PROJECT_NAME}")

DROPOUT = float(os.getenv('DROPOUT', '0.25') or 0.25)
NUM_BLOCKS_ENV = int(os.getenv('NUM_BLOCKS', '2') or 2)
KERNEL_SIZES = tuple([1] * (NUM_BLOCKS_ENV + 1))  # keep per-frame MLP

vis_backends = [
    dict(type='LocalVisBackend'),
]

visualizer = dict(
    type='Pose3dLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# runtime
train_cfg = dict(max_epochs=200, val_interval=3)

# # optimizer
# optim_wrapper = dict(optimizer=dict(type='Adam', lr=1e-3))

# # learning policy
# param_scheduler = [
#     dict(type='StepLR', step_size=100000, gamma=0.96, end=80, by_epoch=False)
# ]

# optimizer
optim_wrapper = dict(optimizer=dict(type='Adam', lr=1e-4)) #1e-3 was 4mm off.

# learning policy
param_scheduler = [
    dict(type='ExponentialLR', gamma=0.98, end=200, by_epoch=True)
]

# auto_scale_lr = dict(base_batch_size=512)

# hooks
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        save_best='MPJPE',
        rule='less',
        max_keep_ckpts=1),
    logger=dict(type='LoggerHook', interval=20),
)


# Set CASP flags for codec
codec = {
    'type': 'ImagePoseLifting',
    'num_keypoints': 17,
    'root_index': 0,
    'remove_root': True,
    'dropout_scaling_fix': True,
}
if CASP_MODE == 'v0':
    codec['use_casp'] = True
if CASP_MODE == 'spatial':
    codec['use_casp_spatial'] = True

# Configure loss based on USE_INVCONF_LOSS
if USE_INVCONF_LOSS:
    loss_config = dict(
        type='InverseConfMPJPELoss',
        invconf_alpha=2.0,
        invconf_eps=1e-2,
        invconf_clip=(0.2, 1.0),
        normalize_weights=False,
        max_weight=10.0,
        weighting_mode='per_joint'
    )
else:
    loss_config = dict(type='MSELoss')

# model settings
model = dict(
    type='PoseLifter',
    backbone=dict(
        type='TCN',
        in_channels=NUM_CHANNELS,
        stem_channels=1024,
        num_blocks=NUM_BLOCKS_ENV,
        kernel_sizes=KERNEL_SIZES,
        dropout=DROPOUT,
    ),
    head=dict(
        type='TemporalRegressionHead',
        in_channels=1024,
        num_joints=16,
        loss=loss_config,
        decoder=codec,
    ))

# base dataset settings
dataset_type = 'H36M_Dataset_Custom'
data_root = 'data/h36m/'

# pipelines
train_pipeline = [
    dict(type='GenerateTarget', encoder=codec),
    dict(
        type='PackPoseInputs',
        meta_keys=('id', 'category_id', 'target_img_path', 'flip_indices',
           'target_root', 'target_root_index'))
]
val_pipeline = train_pipeline

# train_ann_file = '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_h36m_train_v4_highres_dets.npz'
train_camera_param_file = "/srv/essa-lab/flash3/nwarner30/pose_estimation/h36m_data/annotations/h36m_cameras_train_vanilla_h36m_25hz.pkl"
print(f"Annotation file: {train_ann_file}")
print(f"Camera parameter file: {train_camera_param_file}")
# test_ann_file = '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_h36m_val_v4_highres_dets.npz'
test_camera_param_file= '/srv/essa-lab/flash3/nwarner30/pose_estimation/h36m_data/annotations/h36m_cameras_test_vanilla_h36m_25hz.pkl'

# data loaders
train_dataloader = dict(
    batch_size=BATCH_SIZE,
    num_workers=0,
    # persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=train_ann_file,
        camera_param_file=train_camera_param_file,
        seq_len=1,
        causal=True,
        keypoint_2d_src='gt',
        data_root=data_root,
        data_prefix=dict(img='images/'),
        pipeline=train_pipeline,
        test_mode=True,
        perspective_method=PERSPECTIVE_METHOD, # Use R,T; Bone Lengths, and Ordinal Depth
    ))
val_dataloader = dict(
    batch_size=BATCH_SIZE,
    num_workers=0,
    # persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        ann_file=test_ann_file,
        camera_param_file=test_camera_param_file,
        seq_len=1,
        causal=True,
        keypoint_2d_src='gt',
        data_root=data_root,
        data_prefix=dict(img='images/'),
        pipeline=train_pipeline,
        test_mode=True,
        perspective_method=PERSPECTIVE_METHOD, # Use R,T; Bone Lengths, and Ordinal Depth
    ))

# (2) MPI-INF-3DHP test loader
test_dataloader = dict(
    batch_size=BATCH_SIZE,
    num_workers=0,
    # persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='MPI_INF_3DHP_Dataset',
        # ann_file='/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_3dhp_test_all_v3_det.npz',
        ann_file='/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_3dhp_test_all_v4_hr.npz',
        camera_param_file='/srv/essa-lab/flash3/nwarner30/pose_estimation/data/annotations/cameras_test.pkl',
        seq_len=1,
        causal=True,
        keypoint_2d_src='gt',
        data_root=data_root,
        data_prefix=dict(img='images/'),
        pipeline=train_pipeline,
        test_mode=True,
        config_method='mpi_test',
        perspective_method=PERSPECTIVE_METHOD,
    )
)

# (3) Fit3D test loader
test_dataloader_2 = dict(
    batch_size=BATCH_SIZE,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='Fit3D_Dataset',
        # ann_file='/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/val_fit3d_v6_dets.npz',
        ann_file='/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/val_fit3d_v5_detdav_25hz_hr.npz',
        camera_param_file='/srv/essa-lab/flash3/nwarner30/pose_estimation/fit3d_data/processed/_25hz_combine_train_allcameras.pkl',
        seq_len=1,
        causal=True,
        keypoint_2d_src='gt',
        data_root=data_root,
        data_prefix=dict(img='images/'),
        pipeline=train_pipeline,
        test_mode=True,
        dataset_type='fit3d',
        perspective_method=PERSPECTIVE_METHOD,
    )
)

# (4) 3DPW test loader
test_dataloader_3 = dict(
    batch_size=BATCH_SIZE,
    num_workers=0,
    # persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='_3DPW_Dataset',
        # ann_file='/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_3dpw_all_v5_detections.npz',
        ann_file='/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_3dpw_all_v5_det_dav_hr.npz',
        camera_param_file='/srv/essa-lab/flash3/nwarner30/pose_estimation/3dpw_data/processed_mmpose_shards/camera_params_all.pkl',
        seq_len=1,
        causal=True,
        keypoint_2d_src='gt',
        data_root=data_root,
        data_prefix=dict(img='images/'),
        pipeline=train_pipeline,
        test_mode=True,
        dataset_type='3dpw',
        perspective_method=PERSPECTIVE_METHOD,
    )
)

# evaluators
skip_list = [
    'S9_Greet', 'S9_SittingDown', 'S9_Wait_1', 'S9_Greeting', 'S9_Waiting_1'
]

# (A) H36M evaluation (unchanged from val)
val_evaluator = [
    dict(type='MPJPE', mode='mpjpe', dataset_type='h36m_custom', actionwise=True),
]

# (B) MPI-INF-3DHP evaluation
test_evaluator = [
    dict(type='MPJPE', mode='mpjpe', dataset_type='3dhp-inf'),
]

# (C) Fit3D evaluation
test_evaluator_2 = [
    dict(type='MPJPE', mode='mpjpe', dataset_type='fit3d'),
]

# (D) 3DPW evaluation
test_evaluator_3 = [
    dict(type='MPJPE', mode='mpjpe', dataset_type='3dpw'),
]


custom_hooks = [
    dict(
        type='WandbLoggerHook',
        project=WANDB_PROJECT_NAME,
        experiment_name=EXPERIMENT_NAME,
        val_interval=5, # default 1 (want si)
    ),
    # dict(
    #     type='RunCrossValHook',
    #     val_interval=1,
    # )
]

if CASP_MODE == 'spatial':
    test_dataloader['dataset']['ann_file'] = '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_3dhp_test_all_v6_casp_fullcasp_dets.npz'
    test_dataloader_2['dataset']['ann_file'] = '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/val_fit3d_v6_casp_fullcasp_dets.npz'
    test_dataloader_3['dataset']['ann_file'] = '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_3dpw_all_v6_casp_fullcasp_dets.npz'
else:
    test_dataloader['dataset']['ann_file'] = '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_3dhp_test_all_v6_casp_dets.npz'
    test_dataloader_2['dataset']['ann_file'] = '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/val_fit3d_v6_casp_dets.npz'
    test_dataloader_3['dataset']['ann_file'] = '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_3dpw_all_v6_casp_dets.npz'
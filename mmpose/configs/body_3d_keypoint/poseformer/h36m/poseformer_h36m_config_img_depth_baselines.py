_base_ = ['../../../_base_/default_runtime.py']


import os

# --- Environment & Experiment Setup ---
PERSPECTIVE_METHOD = os.getenv('PERSPECTIVE_METHOD', '')
EXPERIMENT_NAME    = os.getenv('EXPERIMENT_NAME', '')
WANDB_PROJECT_NAME = os.getenv('WANDB_PROJECT_NAME', '')
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 256))

# Ordinal loss toggles
USE_ORDINAL_LOSS = os.getenv('USE_ORDINAL_LOSS', 'false').lower() == 'true'
ORDINAL_W   = float(os.getenv('ORDINAL_W', '0.3'))      # λ
ORDINAL_EPS = float(os.getenv('ORDINAL_EPS_MM', '30'))  # mm
ORDINAL_TAU = float(os.getenv('ORDINAL_TAU_MM', '25'))  # mm

# Feature map projection toggles (NEW - all default OFF for backward compatibility)
FEAT_PROJ_DIM   = int(os.getenv('FEAT_PROJ_DIM', '16'))
USE_IMG_FEATS   = os.getenv('USE_IMG_FEATS', 'false').lower() == 'true'   # RTM 1024D
USE_DEPTH_FEATS = os.getenv('USE_DEPTH_FEATS', 'false').lower() == 'true' # DAV2 256D

# Feature normalization toggle (default: normalize features at runtime in codec)
NORMALIZE_FEATS = os.getenv('NORMALIZE_FEATS', 'true').lower() == 'true'

# Dataset selection based on keypoint source and sampling mode
# Options: '2d_det_kpt_2d_det_sample_nearest4', '2d_det_kpt_2d_det_sample_nearest2', 
#          '2d_det_kpt_2d_det_sample_nearest1', '2d_det_kpt_gt_sample_nearest4'
DATASET_MODE = os.getenv('DATASET_MODE', '2d_det_kpt_2d_det_sample_nearest4')

IMG_FEAT_DIM, DEPTH_FEAT_DIM = 1024, 256

print(f'Experiment Name: {EXPERIMENT_NAME}')
print(f'Project Name:    {WANDB_PROJECT_NAME}')
print(f'Perspective:     {PERSPECTIVE_METHOD}')
print(f'Batch:     {BATCH_SIZE}')
print(f'Dataset Mode:    {DATASET_MODE}')
print(f'Ordinal Loss:    {USE_ORDINAL_LOSS} (lambda={ORDINAL_W}, eps={ORDINAL_EPS}mm, tau={ORDINAL_TAU}mm)')
print(f'Use IMG Feats:   {USE_IMG_FEATS} (RTM 1024D -> {FEAT_PROJ_DIM}D)')
print(f'Use Depth Feats: {USE_DEPTH_FEATS} (DAV2 256D -> {FEAT_PROJ_DIM}D)')
print(f'Normalize Feats: {NORMALIZE_FEATS} (runtime normalization in codec)')
if USE_IMG_FEATS and USE_DEPTH_FEATS:
    print(f'WARNING: BOTH RTM+DAV2 features enabled! Total feature dims: {IMG_FEAT_DIM + DEPTH_FEAT_DIM} -> {2 * FEAT_PROJ_DIM}D')

# --- Sequence & Batch Size ---
SEQ_LEN = int(os.getenv('SEQ_LEN', '243'))
SEQ_STEP = 1 # by defualt.

# --- Transformer Input Dim ---
if PERSPECTIVE_METHOD == 'xycd':
    num_channels_transformer = 4
elif PERSPECTIVE_METHOD == 'xycdd':
    num_channels_transformer = 5
elif PERSPECTIVE_METHOD == 'xyc':
    num_channels_transformer = 3
elif PERSPECTIVE_METHOD == 'xyd':
    num_channels_transformer = 3
elif PERSPECTIVE_METHOD == 'casp10d':
    CASP_MODE = os.getenv('CASP_MODE', 'v0')
    if CASP_MODE == 'spatial':
        num_channels_transformer = 10
    else:
        num_channels_transformer = 6
else:
    num_channels_transformer = 2

# --- Visualization ---
vis_backends = [dict(type='LocalVisBackend')]
visualizer  = dict(
    type='Pose3dLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')

# --- Training & Validation Codecs ---
use_casp = (PERSPECTIVE_METHOD == 'casp10d')
use_casp_spatial = (PERSPECTIVE_METHOD == 'casp10d' and os.getenv('CASP_MODE', 'spatial') == 'spatial')

train_codec = dict(
    type='PoseFormerLabel',
    num_keypoints=17,
    concat_vis=False,
    mode='train',
    depth_jitter_sigma=0.00,
    concatenate_root_depth=(PERSPECTIVE_METHOD == 'xycdd'),
    use_casp=use_casp,
    use_casp_spatial=use_casp_spatial,
    use_xyd=(PERSPECTIVE_METHOD == 'xyd'),
    # Feature map parameters
    normalize_feats=NORMALIZE_FEATS,
    use_img_feats=USE_IMG_FEATS,
    use_depth_feats=USE_DEPTH_FEATS,
    img_feat_dim=IMG_FEAT_DIM,
    depth_feat_dim=DEPTH_FEAT_DIM,
)
val_codec = dict(
    type='PoseFormerLabel',
    num_keypoints=17,
    concat_vis=False,
    rootrel=True,
    concatenate_root_depth=(PERSPECTIVE_METHOD == 'xycdd'),
    use_casp=use_casp,
    use_casp_spatial=use_casp_spatial,
    use_xyd=(PERSPECTIVE_METHOD == 'xyd'),
    # Feature map parameters
    normalize_feats=NORMALIZE_FEATS,
    use_img_feats=USE_IMG_FEATS,
    use_depth_feats=USE_DEPTH_FEATS,
    img_feat_dim=IMG_FEAT_DIM,
    depth_feat_dim=DEPTH_FEAT_DIM,
)

#Adjust to official paper stuff
WIDTH = 96 if SEQ_LEN >= 81 else 32          # helper

USE_INVCONF_LOSS = os.getenv('USE_INVCONF_LOSS', 'false').lower() == 'true'

loss_type = 'InverseConfMPJPELoss' if USE_INVCONF_LOSS else 'MPJPELoss'

# Build loss config conditionally
if USE_INVCONF_LOSS:
    loss_cfg = dict(
        type=loss_type,
        invconf_alpha=2.0,
        invconf_eps=1e-2,
        invconf_clip=(0.2, 1.0),
        normalize_weights=False,
        max_weight=10.0,
        weighting_mode='per_joint'
    )
else:
    loss_cfg = dict(type=loss_type)

# Calculate raw input channels (base + optional feature maps)
raw_in_channels = (
    num_channels_transformer
    + (IMG_FEAT_DIM   if USE_IMG_FEATS   else 0)
    + (DEPTH_FEAT_DIM if USE_DEPTH_FEATS else 0)
)

model = dict(
    type='PoseLifter',
    backbone=dict(
        type='PoseFormer',
        in_channels=raw_in_channels,         # raw XY(C)(D) + raw feat maps
        feat_size     = WIDTH,
        spatial_depth = 4,
        temporal_depth= 4 if SEQ_LEN >= 81 else 4,
        num_heads     = 8,
        mlp_ratio     = 4.0,
        seq_len       = SEQ_LEN,
        num_keypoints = 17,
        dropout       = 0.0,
        drop_path_rate= 0.1,
        # NEW: tell the backbone how to fuse features
        use_img_feats   = USE_IMG_FEATS,
        use_depth_feats = USE_DEPTH_FEATS,
        img_feat_dim    = IMG_FEAT_DIM,
        depth_feat_dim  = DEPTH_FEAT_DIM,
        feat_proj_dim   = FEAT_PROJ_DIM,
    ),
    head=dict(
        type='PoseRegressionHead',
        in_channels = WIDTH * 17,            # <-- unchanged
        num_joints  = 17,
        out_channels= 3,
        loss=loss_cfg,
        ordinal_loss=(
            dict(
                type='PairwiseOrdinalLogisticLoss',
                epsilon_mm=ORDINAL_EPS,
                tau_mm=ORDINAL_TAU,
                weight_mode='absdz',
                loss_weight=ORDINAL_W
            ) if USE_ORDINAL_LOSS else None
        ),
        decoder=val_codec,
    ),
    test_cfg=dict(flip_test=True),
)

# --- Optimizer & LR Schedule (PoseFormer defaults) ---
optim_wrapper = dict(
    # optimizer=dict(type='AdamW', lr=2e-4, weight_decay=0.05),
    # optimizer = dict(type='AdamW', lr=2e-4, weight_decay=1e-4)
    optimizer = dict(type='AdamW', lr=2e-4, weight_decay=0.05) # Not sure why it was 1e-4
)




param_scheduler = [
    dict(  # simple per-epoch decay
        type='ExponentialLR',
        gamma=0.98,          # same as -lrd 0.99
        by_epoch=True
    )
]


# train_cfg = dict(max_epochs=130, val_interval=1)
train_cfg = dict(
    max_epochs=130,
    val_interval=1
)
val_cfg   = dict()
# Outer test_cfg (dataset-level) can be left empty
test_cfg  = dict()

# --- Hooks & Logging ---
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        save_best='MPJPE',
        rule='less',
        max_keep_ckpts=1
    ),
    logger=dict(type='LoggerHook', interval=20),
)

custom_hooks = [
    dict(
        type='WandbLoggerHook',
        project=WANDB_PROJECT_NAME,
        experiment_name=EXPERIMENT_NAME,
        val_interval=1,
    )
]

# --- Dataset & Pipelines ---
dataset_type = 'H36M_Dataset_Custom'
data_root = 'data/h36m/'

train_pipeline = [
    dict(type='GenerateTarget', encoder=train_codec),
    dict(
        type='RandomFlipAroundRoot',
        keypoints_flip_cfg=dict(center_mode='static', center_x=0.),
        target_flip_cfg=dict(center_mode='static', center_x=0.),
        flip_label=True
    ),
    dict(
        type='PackPoseInputs',
        meta_keys=('id', 'category_id', 'target_img_path', 'flip_indices', 'camera_param')
    )
]
val_pipeline = [
    dict(type='GenerateTarget', encoder=val_codec),
    dict(
        type='PackPoseInputs',
        meta_keys=('id', 'category_id', 'target_img_path', 'flip_indices', 'camera_param')
    )
]

# ------- Absolute NPZ paths - Select based on DATASET_MODE -------
# Dataset mode options (use underscore versions to match launch script):
#   - '2d_det_kpt_2d_det_sample_nearest_4': Detected keypoints, detected feature sampling, nearest_4
#   - '2d_det_kpt_2d_det_sample_nearest_2': Detected keypoints, detected feature sampling, nearest_2
#   - '2d_det_kpt_2d_det_sample_nearest_1': Detected keypoints, detected feature sampling, nearest_1
#   - '2d_det_kpt_gt_sample_nearest4': Detected keypoints, GT feature sampling, nearest_4

DATASET_PATHS = {
    # Detected 2D keypoints + Detected feature sampling (recommended for train/test consistency)
    '2d_det_kpt_2d_det_sample_nearest_4': {
        'train': '/srv/essa-lab/flash3/nwarner30/pose_estimation/h36m_data/annotations/merged_data_h36m_train_v10_cached_feats_rtm_dav2_nearest_4_dets.npz',
        'val': '/srv/essa-lab/flash3/nwarner30/pose_estimation/h36m_data/annotations/merged_data_h36m_val_v10_cached_feats_rtm_dav2_nearest_4_dets.npz',
        'test': '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_3dhp_test_all_v10_cached_feats_rtm_dav2_nearest_4_dets.npz',
    },
    '2d_det_kpt_2d_det_sample_nearest_2': {
        'train': '/srv/essa-lab/flash3/nwarner30/pose_estimation/h36m_data/annotations/merged_data_h36m_train_v10_cached_feats_rtm_dav2_nearest_2_dets.npz',
        'val': '/srv/essa-lab/flash3/nwarner30/pose_estimation/h36m_data/annotations/merged_data_h36m_val_v10_cached_feats_rtm_dav2_nearest_2_dets.npz',
        'test': '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_3dhp_test_all_v10_cached_feats_rtm_dav2_nearest_2_dets.npz',
    },
    '2d_det_kpt_2d_det_sample_nearest_1': {
        'train': '/srv/essa-lab/flash3/nwarner30/pose_estimation/h36m_data/annotations/merged_data_h36m_train_v10_cached_feats_rtm_dav2_nearest_1_dets.npz',
        'val': '/srv/essa-lab/flash3/nwarner30/pose_estimation/h36m_data/annotations/merged_data_h36m_val_v10_cached_feats_rtm_dav2_nearest_1_dets.npz',
        'test': '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_3dhp_test_all_v10_cached_feats_rtm_dav2_nearest_1_dets.npz',
    },
    # Detected 2D keypoints + GT feature sampling (for debugging only - creates train/test mismatch!)
    '2d_det_kpt_gt_sample_nearest_4': {
        'train': '/srv/essa-lab/flash3/nwarner30/pose_estimation/h36m_data/annotations/merged_data_h36m_train_v10_cached_feats_rtm_dav2_feat_map_gt_sample_nearest_4_dets.npz',
        'val': '/srv/essa-lab/flash3/nwarner30/pose_estimation/h36m_data/annotations/merged_data_h36m_val_v10_cached_feats_rtm_dav2_feat_map_gt_sample_nearest_4_dets.npz',
        'test': '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_3dhp_test_all_v10_cached_feats_rtm_dav2_feat_map_gt_sample_nearest_4.npz',
    },
    # GT 2D keypoints + GT feature sampling (upper bound - no train/test mismatch)
    '2d_gts_kpt_gt_sample_nearest_4': {
        'train': '/srv/essa-lab/flash3/nwarner30/pose_estimation/h36m_data/annotations/merged_data_h36m_train_v10_cached_feats_rtm_dav2_feat_map_gt_sample_nearest_4.npz',
        'val': '/srv/essa-lab/flash3/nwarner30/pose_estimation/h36m_data/annotations/merged_data_h36m_val_v10_cached_feats_rtm_dav2_feat_map_gt_sample_nearest_4.npz',
        'test': '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_3dhp_test_all_v10_cached_feats_rtm_dav2_feat_map_gt_sample_nearest_4.npz',
    },
}

# Select paths based on DATASET_MODE
assert DATASET_MODE in DATASET_PATHS, f"Invalid DATASET_MODE: {DATASET_MODE}. Choose from {list(DATASET_PATHS.keys())}"
TRAIN_ANN = DATASET_PATHS[DATASET_MODE]['train']
VAL_ANN = DATASET_PATHS[DATASET_MODE]['val']
TEST_ANN = DATASET_PATHS[DATASET_MODE]['test']
# ------- Dataloaders (single test dataset; testds2/testds3 removed) -------
train_dataloader = dict(
    batch_size=BATCH_SIZE,
    pin_memory=True,
    num_workers=0,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=TRAIN_ANN,
        seq_len=SEQ_LEN,
        seq_step=SEQ_STEP,
        causal=False,
        pad_video_seq=True,
        camera_param_file='/srv/essa-lab/flash3/nwarner30/pose_estimation/h36m_data/annotations/h36m_cameras_train_vanilla_h36m_25hz.pkl',
        data_root=data_root,
        data_prefix=dict(img='images/'),
        pipeline=train_pipeline,
        perspective_method=PERSPECTIVE_METHOD,
        # normalize_feats=NORMALIZE_FEATS,  # NEW: pass normalization flag
    )
)

val_dataloader = dict(
    batch_size=BATCH_SIZE,
    pin_memory=True,
    num_workers=0,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        ann_file=VAL_ANN,
        seq_len=SEQ_LEN,
        seq_step=SEQ_STEP,
        causal=False,
        pad_video_seq=True,
        camera_param_file='/srv/essa-lab/flash3/nwarner30/pose_estimation/h36m_data/annotations/h36m_cameras_test_vanilla_h36m_25hz.pkl',
        data_root=data_root,
        data_prefix=dict(img='images/'),
        pipeline=val_pipeline,
        test_mode=True,
        perspective_method=PERSPECTIVE_METHOD,
        # normalize_feats=NORMALIZE_FEATS,  # NEW: pass normalization flag
    )
)


test_dataloader = dict(
    batch_size=BATCH_SIZE,
    num_workers=0,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='MPI_INF_3DHP_Dataset',
        ann_file=TEST_ANN,
        seq_len=SEQ_LEN,
        causal=False,
        pad_video_seq=True,
        camera_param_file='/srv/essa-lab/flash3/nwarner30/pose_estimation/data/annotations/cameras_test.pkl',
        data_root=data_root,
        data_prefix=dict(img='images/'),
        pipeline=val_pipeline,
        test_mode=True,
        config_method='mpi_test',
        perspective_method=PERSPECTIVE_METHOD,
        # normalize_feats=NORMALIZE_FEATS,  # NEW: pass normalization flag
    )
)

# test_dataloader_2 = dict(
#     batch_size=BATCH_SIZE,
#     num_workers=0,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type='_3DPW_Dataset',
#         ann_file='/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_3dpw_v8_cached_feats_dets_fmap_pre_normalized.npz',
#         seq_len=SEQ_LEN,
#         causal=False,
#         pad_video_seq=True,
#         camera_param_file='/srv/essa-lab/flash3/nwarner30/pose_estimation/3dpw_data/processed_mmpose_shards/camera_params_all.pkl',
#         data_root=data_root,
#         data_prefix=dict(img='images/'),
#         pipeline=val_pipeline,
#         test_mode=True,
#         dataset_type='3dpw',
#         perspective_method=PERSPECTIVE_METHOD,
#         # normalize_feats=NORMALIZE_FEATS,  # NEW: pass normalization flag
#     )
# )
test_dataloader_2 = None

test_dataloader_3 = None

skip_list = ['S9_Greet', 'S9_SittingDown', 'S9_Wait_1', 'S9_Greeting', 'S9_Waiting_1']
val_evaluator = [
    dict(
        type='MPJPE',
        mode='mpjpe',
        dataset_type='h36m_custom',
        actionwise=True,
        skip_list=skip_list
    ),
    dict(type='PairwiseOrdinalAccuracy', epsilon_mm=30.0)
]
test_evaluator = [
    dict(type='MPJPE', mode='mpjpe', dataset_type='3dhp-inf'),
    dict(type='PairwiseOrdinalAccuracy', epsilon_mm=30.0)
]

# test_evaluator_2 = [
#     dict(type='MPJPE', mode='mpjpe', dataset_type='3dpw'),
#     dict(type='PairwiseOrdinalAccuracy', epsilon_mm=30.0)
# ]
test_evaluator_2 = None

test_evaluator_3 = None
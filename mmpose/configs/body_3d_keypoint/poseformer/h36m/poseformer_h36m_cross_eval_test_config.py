# filepath: /Users/nikolaiwarner/Desktop/local_vscode/poseformer_h36m_cross_eval_test_config.py
import os
import os.path as osp

_base_ = ['../../../_base_/default_runtime.py']

# Load perspective method from environment variable
PERSPECTIVE_METHOD = os.getenv('PERSPECTIVE_METHOD', '')
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 512))

# Import the environment variable 'EXPERIMENT_NAME'
EXPERIMENT_NAME = os.getenv('EXPERIMENT_NAME', '')
print(f"Experiment Name: {EXPERIMENT_NAME}")

WANDB_PROJECT_NAME = os.getenv('WANDB_PROJECT_NAME', '')
print(f"Project Name: {WANDB_PROJECT_NAME}")

vis_backends = [
    dict(type='LocalVisBackend'),
]

# SEQ_LEN
SEQ_LEN = int(os.getenv('SEQ_LEN', '27'))
print(f"Seqlen: {SEQ_LEN}")

SEQ_STEP = int(os.getenv('SEQ_STEP', '1'))
print(f"Seqstep: {SEQ_STEP}")

# Determine number of input channels based on perspective method
if PERSPECTIVE_METHOD == 'xycd':
    num_channels_transformer = 4
elif PERSPECTIVE_METHOD == 'xycdd':
    num_channels_transformer = 5
elif PERSPECTIVE_METHOD == 'xyc':
    num_channels_transformer = 3
elif PERSPECTIVE_METHOD == 'xyd':
    num_channels_transformer = 3
elif PERSPECTIVE_METHOD == '':
    num_channels_transformer = 2
else:
    print("Error in perspective")
    num_channels_transformer = 2

print(f"Input channels: {num_channels_transformer}")

visualizer = dict(
    type='Pose3dLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# No training configuration for test-only mode
train_dataloader = None
train_cfg = None
optim_wrapper = None

# Codec settings for validation/testing
val_codec = dict(
    type='PoseFormerLabel',
    num_keypoints=17,
    concat_vis=False,
    rootrel=True,
    concatenate_root_depth=(PERSPECTIVE_METHOD == 'xycdd'),
    use_xyd=(PERSPECTIVE_METHOD == 'xyd'),
)

# PoseFormer model configuration
WIDTH = 96 if SEQ_LEN >= 81 else 32

model = dict(
    type='PoseLifter',
    backbone=dict(
        type='PoseFormer',
        in_channels=num_channels_transformer,
        feat_size=WIDTH,
        spatial_depth=4,
        temporal_depth=4 if SEQ_LEN >= 81 else 4,
        num_heads=8,
        mlp_ratio=4.0,
        seq_len=SEQ_LEN,
        num_keypoints=17,
        dropout=0.0,
        drop_path_rate=0.1,
    ),
    head=dict(
        type='PoseRegressionHead',
        in_channels=WIDTH * 17,
        num_joints=17,
        out_channels=3,
        loss=dict(type='MPJPELoss'),
        decoder=val_codec,
    ),
    test_cfg=dict(flip_test=True),
)

# base dataset settings
dataset_type = 'H36M_Dataset_Custom'
data_root = 'data/h36m/'

# pipelines
val_pipeline = [
    dict(type='GenerateTarget', encoder=val_codec),
    dict(
        type='PackPoseInputs',
        meta_keys=('id', 'category_id', 'target_img_path', 'flip_indices', 'camera_param'))
]

# Validation dataloader (H36M in-distribution)
test_ann_file = '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_h36m_val_v4_highres_dets.npz'

val_dataloader = dict(
    batch_size=BATCH_SIZE,
    pin_memory=True,
    num_workers=0,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        ann_file=test_ann_file,
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
    ))

val_evaluator = [
    dict(type='MPJPE', mode='mpjpe'),
]

# ─────── INSERT TEST_DS / HR logic here ───────

# Read which test–dataset to use and the HR flag
TEST_DS = os.getenv('TEST_DS', '3dhp').lower()
print(f"TEST_DS = {TEST_DS}")
HR = os.getenv('HR', 'False').lower() in ('true', '1', 'yes')
HR = False
print(f"HR = {HR}")

# Base directory where all merged .npz live
base_dir = '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data'

# Prepare test_dataloader and test_evaluator based on TEST_DS
if TEST_DS == '3dhp':
    # ann_file = osp.join(base_dir, 'merged_data_3dhp_test_all_v4_hr.npz')
    ann_file ='/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_3dhp_test_v8_cached_feats_rtm_dav2.npz' # gt no fmap
    cam_file = '/srv/essa-lab/flash3/nwarner30/pose_estimation/data/annotations/cameras_test.pkl'
    test_dataloader = dict(
        batch_size=BATCH_SIZE,
        num_workers=0,
        sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
        dataset=dict(
            type='MPI_INF_3DHP_Dataset',
            ann_file=ann_file,
            seq_len=SEQ_LEN,
            seq_step=SEQ_STEP,
            causal=False,
            pad_video_seq=True,
            camera_param_file=cam_file,
            data_root=data_root,
            data_prefix=dict(img='images/'),
            pipeline=val_pipeline,
            test_mode=True,
            config_method='mpi_test',
            perspective_method=PERSPECTIVE_METHOD,
        )
    )
    test_evaluator = [
        dict(type='MPJPE', mode='mpjpe', dataset_type='3dhp-inf'),
        dict(type='PCK3D150mm'),
        dict(type='OrdinalDepthAccuracy'),
    ]

elif TEST_DS == 'fit3d':
    ann_file = osp.join(base_dir, 'val_fit3d_v5_detdav_25hz_hr.npz')
    cam_file = '/srv/essa-lab/flash3/nwarner30/pose_estimation/fit3d_data/processed/_25hz_combine_train_allcameras.pkl'
    test_dataloader = dict(
        batch_size=BATCH_SIZE,
        num_workers=0,
        sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
        dataset=dict(
            type='Fit3D_Dataset',
            ann_file=ann_file,
            seq_len=SEQ_LEN,
            seq_step=SEQ_STEP,
            causal=False,
            pad_video_seq=True,
            camera_param_file=cam_file,
            data_root=data_root,
            data_prefix=dict(img='images/'),
            pipeline=val_pipeline,
            test_mode=True,
            dataset_type='fit3d',
            perspective_method=PERSPECTIVE_METHOD,
        )
    )
    test_evaluator = [
        dict(type='MPJPE', mode='mpjpe', dataset_type='fit3d'),
    ]

elif TEST_DS == '3dpw':
    ann_file = osp.join(base_dir, 'merged_data_3dpw_all_v5_det_dav_hr.npz')
    cam_file = '/srv/essa-lab/flash3/nwarner30/pose_estimation/3dpw_data/processed_mmpose_shards/camera_params_all.pkl'
    test_dataloader = dict(
        batch_size=BATCH_SIZE,
        num_workers=0,
        sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
        dataset=dict(
            type='_3DPW_Dataset',
            ann_file=ann_file,
            seq_len=SEQ_LEN,
            seq_step=SEQ_STEP,
            causal=False,
            pad_video_seq=True,
            camera_param_file=cam_file,
            data_root=data_root,
            data_prefix=dict(img='images/'),
            pipeline=val_pipeline,
            test_mode=True,
            dataset_type='3dpw',
            perspective_method=PERSPECTIVE_METHOD,
        )
    )
    test_evaluator = [
        dict(type='MPJPE', mode='mpjpe', dataset_type='3dpw'),
    ]

elif TEST_DS == 'h36m_indistr':
    test_dataloader = dict(
        batch_size=BATCH_SIZE,
        pin_memory=True,
        num_workers=0,
        sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
        dataset=dict(
            type=dataset_type,
            ann_file=test_ann_file,
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
        )
    )
    test_evaluator = [
        dict(type='MPJPE', mode='mpjpe', dataset_type='h36m_custom'),
    ]

else:
    raise ValueError(f"Unsupported TEST_DS: {TEST_DS}. Must be one of ['3dhp','fit3d','3dpw','h36m_indistr'].")

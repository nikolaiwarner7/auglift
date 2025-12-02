# **AugLift Training Guide**

This guide covers training AugLift models for 3D human pose lifting with depth augmentation.

---

## **Prerequisites**

Before training, ensure you have:

1. ✅ Completed environment setup (see main README)
2. ✅ Preprocessed datasets with XYCD features (see [PREPROCESSING.md](PREPROCESSING.md))
3. ✅ Downloaded pretrained 2D detection models (if needed)

---

## **Model Architectures**

AugLift supports multiple backbone architectures:

### **PoseFormer (default)**
* Transformer-based temporal lifting
* Input: XYCD features (2D keypoints + confidence + depth)
* Supports both frame-based and sequence-based inputs

### **TCN (Temporal Convolutional Network)**
* Dilated convolutional temporal model
* Faster training than PoseFormer
* Good for real-time applications

---

## **Training Configurations**

Training configs are located in:

```
mmpose/configs/body_3d_keypoint/poseformer/h36m/
mmpose/configs/body_3d_keypoint/image_pose_lift/h36m/
```

### **Key config files:**

**PoseFormer configs:**
* `poseformer_h36m_config_img_depth_baselines.py` - Depth-augmented PoseFormer

**TCN configs:**
* `image-pose-lift_tcn_8xb64-200e_h36m_oct25_casp.py` - TCN with CASP depth features

---

## **Single-Dataset Training**

### **H36M only:**

```bash
python mmpose/tools/train.py \
    mmpose/configs/body_3d_keypoint/poseformer/h36m/poseformer_h36m_config_img_depth_baselines.py
```

### **Custom config:**

```bash
python mmpose/tools/train.py path/to/your/config.py
```

---

## **Cross-Dataset Training**

AugLift supports training on multiple datasets jointly (H36M + 3DPW + 3DHP + Fit3D).

### **Using launcher scripts:**

```bash
# PoseFormer cross-dataset
bash mmpose/tools/launch_train_cross_datasets_may_25_pf.sh

# PoseFormer outer loop (hyperparameter sweep)
bash mmpose/tools/launch_train_cross_datasets_may_25_poseformer_test_outer.sh
```

These scripts:
* Combine datasets with balanced sampling
* Handle dataset-specific coordinate systems
* Support multi-GPU distributed training

---

## **Custom Codecs & Heads**

### **Codecs:**

AugLift uses custom codecs to handle XYCD input format:

* `mmpose/mmpose/codecs/image_pose_lifting.py` - Base lifting codec
* `mmpose/mmpose/codecs/poseformer_label.py` - Modified for PoseFormer

**Key features:**
* Root-relative keypoint normalization
* COCO→H36M joint mapping
* Depth channel integration

### **Regression Heads:**

* `mmpose/mmpose/models/heads/regression_heads/poseformer_regression_head.py`

Modified to accept 4-channel input (X, Y, C, D) instead of standard 2-channel (X, Y).

---

## **Distributed Training**

For multi-GPU training:

```bash
# 8 GPUs
bash mmpose/tools/dist_train.sh \
    path/to/config.py \
    8 \
    --work-dir work_dirs/experiment_name
```

Or with SLURM:

```bash
GPUS=8 bash mmpose/tools/slurm_train.sh \
    partition_name \
    job_name \
    path/to/config.py \
    work_dirs/experiment_name
```

---

## **Training Parameters**

Common parameters to adjust in configs:

```python
# Batch size
train_dataloader = dict(batch_size=64)

# Learning rate
optim_wrapper = dict(
    optimizer=dict(lr=0.001)
)

# Training epochs
train_cfg = dict(max_epochs=200)

# Input features
codec = dict(
    type='ImagePoseLifting',
    num_input_channels=4,  # XYCD
    use_depth=True,
)
```

---

## **Evaluation**

### **During training:**

Validation runs automatically if configured:

```python
val_cfg = dict()
val_dataloader = dict(
    dataset=dict(ann_file='val_annotations.npz')
)
val_evaluator = dict(type='SimpleMeanMetric')
```

### **After training:**

```bash
python mmpose/tools/test.py \
    path/to/config.py \
    work_dirs/experiment_name/best_model.pth \
    --work-dir work_dirs/evaluation
```

---

## **Monitoring**

Training logs and checkpoints are saved to `work_dirs/`:

```
work_dirs/experiment_name/
├── config.py              # Full config used
├── last_checkpoint        # Latest checkpoint path
├── best_model.pth         # Best model by validation metric
├── epoch_*.pth            # Periodic checkpoints
└── logs/
    └── scalars.json       # Training metrics
```

View with TensorBoard:

```bash
tensorboard --logdir work_dirs/experiment_name
```

---

## **Model Variants**

### **Baseline (XY only):**
```python
codec = dict(num_input_channels=2, use_depth=False)
```

### **XYC (with confidence):**
```python
codec = dict(num_input_channels=3, use_depth=False, use_confidence=True)
```

### **XYCD (full AugLift):**
```python
codec = dict(num_input_channels=4, use_depth=True, use_confidence=True)
```

### **XYCD + Feature Maps (V2):**
```python
codec = dict(
    num_input_channels=4,
    use_depth=True,
    use_confidence=True,
    use_feature_maps=True,
)
```

---

## **Troubleshooting**

### **CUDA out of memory:**
* Reduce batch size
* Reduce sequence length (for temporal models)
* Use gradient checkpointing

### **NaN loss:**
* Check learning rate (may be too high)
* Verify data normalization
* Check for invalid depth values

### **Coordinate system errors:**
* Verify camera parameters are consistent
* Check root-relative normalization in codec
* Ensure COCO→H36M mapping is correct

---

## **Key Training Scripts**

* `mmpose/tools/train.py` - Main training script
* `mmpose/tools/test.py` - Evaluation script
* `mmpose/tools/launch_train_cross_datasets_may_25_pf.sh` - PoseFormer launcher
* `mmpose/tools/launch_train_cross_datasets_may_25_poseformer_test_outer.sh` - Hyperparameter sweep

---

**Next:** See [CHANGES.md](CHANGES.md) for detailed modifications vs MMPose base.

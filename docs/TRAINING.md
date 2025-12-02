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

AugLift supports **4 backbone architectures**, each with different tradeoffs:

### **1. PoseFormer (Transformer, Many-to-One)**
* Transformer-based temporal lifting
* **Many-to-one prediction:** Predicts single frame from sequence
* Input: XYCD features (2D keypoints + confidence + depth)
* Near SOTA performance
* Variable sequence lengths supported (27, 81, 243 frames)

### **2. TCN (Temporal Convolutional Network)**
* Dilated convolutional temporal model
* **Second oldest architecture**, very lightweight
* Faster training than PoseFormer
* Good for real-time applications
* Supports temporal sequences

### **3. MotionBERT (Transformer, Many-to-Many)**
* Transformer-based, near SOTA (particularly strong)
* **Many-to-many prediction:** Predicts multiple frames at once
* Uses **camera-invariant codec**
* Requires **GT root depth and focal length** for precise inference
* Variable sequence lengths

### **4. SimpleBaseline (No Temporal)**
* Single-frame baseline (no temporal features)
* Lightweight, fast inference
* Only one "sequence length" (frame-based)
* Good baseline for ablation studies

### **Sequence Length Studies**

We study multiple sequence lengths for each temporal architecture:
* **PoseFormer/MotionBERT/TCN:** 27, 81, 243 frames
* **SimpleBaseline:** N/A (single frame)

Per **Section 4.1** and **Section 5** of the paper.

---

## **Training Configurations**

Training configs are located in:

```
mmpose/configs/body_3d_keypoint/poseformer/h36m/
mmpose/configs/body_3d_keypoint/image_pose_lift/h36m/
mmpose/configs/body_3d_keypoint/motionbert/h36m/
mmpose/configs/body_3d_keypoint/video_pose_lift/h36m/
```

### **Key config files:**

**PoseFormer configs:**
* `poseformer_h36m_cross_eval_test_config.py` - Cross-dataset evaluation config
* `poseformer_h36m_config_img_depth_baselines.py` - Depth-augmented PoseFormer

**SimpleBaseline configs:**
* `image-pose-lift_tcn_8xb64-200e_h36m_oct25_casp.py` - SimpleBaseline with CASP depth features

**MotionBERT configs:**
* `motionbert_h36m_config.py` - MotionBERT with camera-invariant codec

**TCN/VideoPose3D configs:**
* `video_pose_lift_tcn_h36m.py` - TCN temporal model

### **Cross-Dataset Configurations**

We have configurations for:
* Each architecture (PoseFormer, TCN, MotionBERT, SimpleBaseline)
* Each source dataset (H36M, 3DPW, 3DHP, Fit3D)
* Cross-dataset evaluation (train on one, test on others)

Per **Section 4.1** of the paper.

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

**Inner training script:**
```bash
# PoseFormer cross-dataset (single configuration)
bash mmpose/tools/launch_train_cross_datasets_may_25_pf.sh
```

**Outer training script:**
```bash
# PoseFormer hyperparameter sweep across representations
bash mmpose/tools/launch_train_cross_datasets_may_25_poseformer_test_outer.sh
```

These scripts:
* Combine datasets with balanced sampling
* Handle dataset-specific coordinate systems
* Support multi-GPU distributed training via SLURM
* Sweep across sequence lengths and input representations

### **Input Representation Sweep (Section 5)**

We experiment across multiple input representations:

1. **XY** - 2D keypoints only (baseline)
2. **XYC** - 2D keypoints + confidence
3. **XYD** - 2D keypoints + depth
4. **XYCD** - AugLift V1 (full 4-channel)
5. **AugLift V2** - Richer feature fusion with RTMPose/DepthAnything features
6. **Image-Features Baseline** - Direct image feature extraction

**Note:** Due to time constraints, the full representation study is primarily conducted on **PoseFormer** (Section 5 of paper). Other architectures focus on key configurations.

---

## **Custom Codecs & Heads**

### **Codecs:**

AugLift uses custom codecs to handle XYCD input format:

* `mmpose/mmpose/codecs/image_pose_lifting.py` - Base lifting codec
* `mmpose/mmpose/codecs/poseformer_label.py` - Modified for PoseFormer
* `mmpose/mmpose/codecs/motionbert_label.py` - Camera-invariant codec for MotionBERT

**Key features:**
* Root-relative keypoint normalization
* COCO→H36M joint mapping
* Depth channel integration
* Input normalization for additional channels (C, D)
* Camera-invariant formulation (MotionBERT) requires GT root depth and focal length

### **Regression Heads:**

Modified to accept 4-channel input (X, Y, C, D) instead of standard 2-channel (X, Y):

* `mmpose/mmpose/models/heads/regression_heads/poseformer_regression_head.py`
* `mmpose/mmpose/models/heads/regression_heads/motionbert_regression_head.py`
* `mmpose/mmpose/models/heads/regression_heads/simple_regression_head.py`
* `mmpose/mmpose/models/heads/regression_heads/temporal_regression_head.py` (TCN)

**Modifications:**
* Input layer sizes modified per architecture
* Specified in pyconfig files
* Support variable input channels (2, 3, or 4)



---

## **Evaluation**

### **During training:**

Validation runs automatically if configured:

```python
val_cfg = dict()
val_dataloader = dict(
    dataset=dict(ann_file='val_annotations.npz')
)
val_evaluator = dict(type='ExampleMetric')
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
* `mmpose/tools/launch_train_cross_datasets_may_25_pf.sh` - PoseFormer inner loop (single config)
* `mmpose/tools/launch_train_cross_datasets_may_25_poseformer_test_outer.sh` - PoseFormer outer loop (representation sweep)

### **Dataset Loaders:**

* **Base class:** `mmpose/mmpose/datasets/datasets/base/base_mocap_dataset.py`
  * Key changes to load additional fields (C, D, feature maps)
  * Critical to AugLift method
* **H36M:** Uses base_mocap_dataset
* **3DHP:** `mmpose/mmpose/datasets/datasets/body3d/mpi_3dhp_inf_dataset.py`
* **3DPW:** `mmpose/mmpose/datasets/datasets/body3d/3dpw_dataset.py`
* **Fit3D:** `mmpose/mmpose/datasets/datasets/body3d/fit3d_dataset.py`

### **Model Definitions:**

* **PoseFormer:**
  * Backbone: `mmpose/mmpose/models/backbones/poseformer.py`
  * Head: `mmpose/mmpose/models/heads/regression_heads/poseformer_regression_head.py`
* **MotionBERT:**
  * Backbone: `mmpose/mmpose/models/backbones/motionbert.py`
  * Head: `mmpose/mmpose/models/heads/regression_heads/motionbert_regression_head.py`
* **SimpleBaseline:**
  * Head: `mmpose/mmpose/models/heads/regression_heads/simple_regression_head.py`
* **TCN:**
  * Backbone: `mmpose/mmpose/models/backbones/tcn.py`
  * Head: `mmpose/mmpose/models/heads/regression_heads/temporal_regression_head.py`

---

**Next:** See [CHANGES.md](CHANGES.md) for detailed modifications vs MMPose base.

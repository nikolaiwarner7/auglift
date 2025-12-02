# **AugLift Changes vs MMPose Base**

This document details all modifications made to the base MMPose framework to implement AugLift.

---

## **Overview**

AugLift extends MMPose with:

1. **Custom datasets** for 3DPW, 3DHP, Fit3D
2. **Depth-augmented codecs** (XYCD input)
3. **Modified backbones and heads** for 4-channel input
4. **Cross-dataset training** infrastructure
5. **Preprocessing pipelines** for depth + 2D detection

---

## **1. Dataset Classes**

### **New Dataset Classes:**

All based on the H36M dataset class but with custom naming and structure:

#### **`base_mocap_dataset.py`**
* **Location:** `mmpose/mmpose/datasets/datasets/base/base_mocap_dataset.py`
* **Changes:**
  * Base class for all motion capture datasets
  * **Key changes to load additional fields (C, D, feature maps) critical to AugLift method**
  * Handles camera coordinate systems
  * Supports XYCD input format
  * Root-relative keypoint handling
  * Feature map loading for V2 experiments

#### **`mpi_3dhp_inf_dataset.py`**
* **Location:** `mmpose/mmpose/datasets/datasets/body3d/mpi_3dhp_inf_dataset.py`
* **Changes:**
  * Custom loader for 3DHP `.mat` annotations
  * Camera parameter handling specific to 3DHP
  * Test set support for cross-dataset evaluation

#### **3DPW Dataset**
* **Location:** `mmpose/mmpose/datasets/datasets/body3d/pw3d_dataset.py`
* **Changes:**
  * Video-based dataset support
  * Integration with `dataset.json` format
  * Temporal sequence handling

#### **Fit3D Dataset**
* **Location:** `mmpose/mmpose/datasets/datasets/body3d/fit3d_dataset.py`
* **Changes:**
  * Custom folder structure support
  * Train/test split handling
  * Multi-environment support (indoor/outdoor)

---

## **2. Codecs**

### **`image_pose_lifting.py`**
* **Location:** `mmpose/mmpose/codecs/image_pose_lifting.py`
* **Changes:**
  * **4-channel input support** (XYCD vs standard XY)
  * COCO→H36M joint mapping
  * Root-relative normalization
  * Depth channel integration
  * Confidence score handling
  * **Input normalization for additional channels (C, D)**

### **`poseformer_label.py`**
* **Location:** `mmpose/mmpose/codecs/poseformer_label.py`
* **Changes:**
  * Modified for depth-augmented temporal sequences
  * Temporal window handling with 4 channels
  * Sequence padding for variable-length inputs
  * Input normalization adjustments

### **`motionbert_label.py`**
* **Location:** `mmpose/mmpose/codecs/motionbert_label.py`
* **Changes:**
  * Camera-invariant codec for MotionBERT
  * Requires GT root depth and focal length
  * Many-to-many prediction support
  * Input normalization for XYCD features

---

## **3. Model Architectures**

AugLift supports **4 architectures** with modified input layers for richer input formulations.

### **PoseFormer (Many-to-One Transformer)**
* **Backbone:** `mmpose/mmpose/models/backbones/poseformer.py`
* **Head:** `mmpose/mmpose/models/heads/regression_heads/poseformer_regression_head.py`
* **Changes:**
  * Input embedding layer modified for 4 channels
  * Positional encoding adapted for XYCD
  * Temporal attention with depth-aware features
  * Accepts 4-channel input
  * Optional depth-only pathway
  * Multi-task loss support (3D pose + depth consistency)
  * **Input layer sizes modified as specified in pyconfig**

### **MotionBERT (Many-to-Many Transformer)**
* **Backbone:** `mmpose/mmpose/models/backbones/motionbert.py`
* **Head:** `mmpose/mmpose/models/heads/regression_heads/motionbert_regression_head.py`
* **Changes:**
  * Input embedding layer modified for 4 channels
  * Camera-invariant formulation
  * Many-to-many prediction (multiple frames)
  * Requires GT root depth and focal length for inference
  * **Input layer sizes modified as specified in pyconfig**

### **TCN / VideoPose3D (Temporal Convolution)**
* **Backbone:** `mmpose/mmpose/models/backbones/tcn.py`
* **Head:** `mmpose/mmpose/models/heads/regression_heads/temporal_regression_head.py`
* **Changes:**
  * First conv layer modified for 4-channel input
  * Dilated convolutions with depth features
  * Lightweight, second oldest architecture
  * **Input layer sizes modified as specified in pyconfig**

### **SimpleBaseline (Single Frame)**
* **Head:** `mmpose/mmpose/models/heads/regression_heads/simple_regression_head.py`
* **Changes:**
  * Input layer modified for 4-channel input
  * No temporal features
  * Single-frame baseline
  * **Input layer sizes modified as specified in pyconfig**

---

## **4. Evaluators**

### **3D Pose Evaluator**
* **Location:** `mmpose/mmpose/evaluation/metrics/keypoint3d_metric.py`
* **Changes:**
  * Coordinate system alignment across datasets
  * Camera parameter transformation
  * Protocol #1 and Protocol #2 for H36M
  * Cross-dataset evaluation metrics

---

## **5. Data Pipelines**

### **Preprocessing Transforms**

#### **Depth Integration Transform**
* **Location:** `mmpose/mmpose/datasets/transforms/common_transforms.py`
* **Changes:**
  * Load depth from NPZ files
  * Normalize depth values
  * Align depth with 2D keypoints

#### **COCO→H36M Mapping Transform**
* **Location:** `mmpose/mmpose/datasets/transforms/formatting.py`
* **Changes:**
  * Joint reordering from COCO (17 joints) to H36M (17 joints)
  * Handle missing joints (padding with zeros)
  * Confidence score propagation

---

## **6. Training Configurations**

### **Cross-Dataset Configs**

#### **Location:** `mmpose/configs/body_3d_keypoint/`

**Key config files:**

* **PoseFormer:**
  * `poseformer_h36m_cross_eval_test_config.py` - Cross-dataset evaluation
  * `poseformer_h36m_config_img_depth_baselines.py` - Depth baselines
  * Configurations for each source dataset (H36M, 3DPW, 3DHP, Fit3D)
  * Multiple sequence lengths (27, 81, 243 frames)
  
* **SimpleBaseline:**
  * `image-pose-lift_tcn_8xb64-200e_h36m_oct25_casp.py` - CASP depth features
  * Single-frame configurations

* **MotionBERT:**
  * `motionbert_h36m_config.py` - Camera-invariant codec
  * Many-to-many prediction configs
  
* **TCN/VideoPose3D:**
  * `video_pose_lift_tcn_h36m.py` - Temporal convolution
  
* **Cross-Dataset:**
  * Joint training on H36M + 3DPW + 3DHP + Fit3D
  * Balanced sampling across datasets
  * Shared coordinate system handling
  * Per Section 4.1 of paper

---

## **7. Training Scripts**

### **Launcher Scripts**

* `mmpose/tools/launch_train_cross_datasets_may_25_pf.sh`
  * PoseFormer **inner loop** (single configuration)
  * Cross-dataset training
  * Multi-GPU distribution via SLURM
  
* `mmpose/tools/launch_train_cross_datasets_may_25_poseformer_test_outer.sh`
  * PoseFormer **outer loop** (hyperparameter sweep)
  * Sweep across:
    * Input representations (XY, XYC, XYD, XYCD, AugLift V2, Image-Features)
    * Sequence lengths (27, 81, 243 frames)
    * Datasets (H36M, 3DPW, 3DHP, Fit3D)
  * Per Section 5 of paper (representation study primarily on PoseFormer)

---

## **8. Preprocessing Scripts**

### **Depth Estimation**

* `coarse_depth_experiments/Depth-Anything-OfficialV2/Depth-Anything-V2/metric_depth/1_30_estimate_depth_proto_loop_metric_*.py`
  * Joint 2D pose + depth estimation
  * RTMDet → RTMPose → DepthAnything pipeline
  * Per-frame NPZ output
  * **Implements AugLift architecture details (Section 3.2):**
    * Sparse feature fusion
    * Uncertainty-aware descriptors (adaptive radius sampling)
    * CASP depth statistics (min, max, point)

### **XYCD Merging**

* `2_3_merge_xycd_npz_*.py`
  * Combine per-frame NPZ → unified training file
  * Feature map sparse sampling (V2)
  * Multiprocessing support

---

## **9. Key Dependencies**

### **Added to MMPose:**

```python
# In configs
'depth-anything-v2',  # Depth estimation
'rtmpose',            # 2D pose detection
'rtmdet',             # 2D bbox detection
```

---

## **10. Coordinate System Changes**

### **Camera Coordinate Systems**

**MMPose default:** Global coordinates for 3D poses

**AugLift:** Camera-viewpoint coordinates for all datasets

**Changes:**
* Modified dataset classes to output camera coordinates
* Evaluator converts to common coordinate frame for cross-dataset metrics
* Root-relative normalization in codec (centered at pelvis)

---

## **11. File Structure Differences**

```diff
mmpose/
  mmpose/
    codecs/
+     image_pose_lifting.py        # 4-channel XYCD codec
+     poseformer_label.py          # Temporal XYCD sequences
    datasets/
      datasets/
        base/
+         base_mocap_dataset.py    # Base for motion capture data
        body3d/
+         mpi_3dhp_inf_dataset.py  # 3DHP loader
+         pw3d_dataset.py          # 3DPW loader
+         fit3d_dataset.py         # Fit3D loader
      transforms/
+       depth_transforms.py        # Depth loading/normalization
    models/
      backbones/
+       poseformer.py              # Modified input layer
+       tcn.py                     # Modified for 4 channels
      heads/
        regression_heads/
+         poseformer_regression_head.py  # 4-channel head
    evaluation/
      metrics/
+       keypoint3d_metric.py       # Cross-dataset metrics
  configs/
    body_3d_keypoint/
      poseformer/
+       h36m/
+         poseformer_h36m_config_img_depth_baselines.py
+       cross_dataset/
+         poseformer_cross_dataset_config.py
  tools/
+   launch_train_cross_datasets_may_25_pf.sh
+   launch_train_cross_datasets_may_25_poseformer_test_outer.sh

+ coarse_depth_experiments/           # New directory
+   Depth-Anything-OfficialV2/
+     Depth-Anything-V2/
+       metric_depth/
+         1_30_estimate_depth_proto_loop_metric_*.py
+         launch_depth_metric_loop_*.sh

+ 2_3_merge_xycd_npz_*.py            # New scripts
+ 2_25_split_fit3d_train_test.py
```

---

## **12. Summary of Key Changes**

| Component | Change | Purpose |
|-----------|--------|---------|
| **Input Format** | XY → XYCD (4 channels) | Add depth + confidence cues |
| **Architectures** | +4 architectures (PF, MB, TCN, SB) | Multiple backbone options |
| **Datasets** | +3DPW, 3DHP, Fit3D | Cross-dataset training |
| **Coordinate System** | Global → Camera viewpoint | Consistent across datasets |
| **Preprocessing** | +Depth estimation pipeline | Generate XYCD features |
| **Codecs** | Root-relative + COCO→H36M + normalization | Handle format differences + richer inputs |
| **Backbones** | 2-channel → 4-channel input (per pyconfig) | Process depth information |
| **Evaluation** | +Cross-dataset metrics | Generalization testing |
| **Representations** | 6 variants (XY, XYC, XYD, XYCD, V2, Img-Feat) | Section 5 ablation study |
| **base_mocap_dataset** | Load C, D, feature maps | Critical for AugLift method |

---

## **13. Backward Compatibility**

AugLift maintains compatibility with standard MMPose by:

* Using `num_input_channels` flag to toggle XYCD vs XY
* Falling back to standard codecs when `use_depth=False`
* Supporting original H36M format without modification

---

**For questions about specific changes, see the individual files or open an issue.**

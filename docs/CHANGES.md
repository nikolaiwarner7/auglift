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
  * Handles camera coordinate systems
  * Supports XYCD input format
  * Root-relative keypoint handling

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

### **`poseformer_label.py`**
* **Location:** `mmpose/mmpose/codecs/poseformer_label.py`
* **Changes:**
  * Modified for depth-augmented temporal sequences
  * Temporal window handling with 4 channels
  * Sequence padding for variable-length inputs

---

## **3. Model Architectures**

### **PoseFormer Backbone**
* **Location:** `mmpose/mmpose/models/backbones/poseformer.py`
* **Changes:**
  * Input embedding layer modified for 4 channels
  * Positional encoding adapted for XYCD
  * Temporal attention with depth-aware features

### **PoseFormer Regression Head**
* **Location:** `mmpose/mmpose/models/heads/regression_heads/poseformer_regression_head.py`
* **Changes:**
  * Accepts 4-channel input
  * Optional depth-only pathway
  * Multi-task loss support (3D pose + depth consistency)

### **TCN Backbone**
* **Location:** `mmpose/mmpose/models/backbones/tcn.py`
* **Changes:**
  * First conv layer modified for 4-channel input
  * Dilated convolutions with depth features

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

* `poseformer_h36m_config_img_depth_baselines.py`
  * PoseFormer with XYCD input
  * H36M-only training
  
* `poseformer_cross_dataset_config.py`
  * Joint training on H36M + 3DPW + 3DHP + Fit3D
  * Balanced sampling across datasets
  * Shared coordinate system handling

* `image-pose-lift_tcn_8xb64-200e_h36m_oct25_casp.py`
  * TCN with CASP (min/max/point) depth features
  * Faster training than PoseFormer

---

## **7. Training Scripts**

### **Launcher Scripts**

* `mmpose/tools/launch_train_cross_datasets_may_25_pf.sh`
  * PoseFormer cross-dataset training
  * Multi-GPU distribution
  
* `mmpose/tools/launch_train_cross_datasets_may_25_poseformer_test_outer.sh`
  * Hyperparameter sweep
  * Outer loop over depth variants (point, min/max, CASP, feature maps)

---

## **8. Preprocessing Scripts**

### **Depth Estimation**

* `coarse_depth_experiments/Depth-Anything-OfficialV2/Depth-Anything-V2/metric_depth/1_30_estimate_depth_proto_loop_metric_*.py`
  * Joint 2D pose + depth estimation
  * RTMDet → RTMPose → DepthAnything pipeline
  * Per-frame NPZ output

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
| **Datasets** | +3DPW, 3DHP, Fit3D | Cross-dataset training |
| **Coordinate System** | Global → Camera viewpoint | Consistent across datasets |
| **Preprocessing** | +Depth estimation pipeline | Generate XYCD features |
| **Codecs** | Root-relative + COCO→H36M | Handle format differences |
| **Backbones** | 2-channel → 4-channel input | Process depth information |
| **Evaluation** | +Cross-dataset metrics | Generalization testing |

---

## **13. Backward Compatibility**

AugLift maintains compatibility with standard MMPose by:

* Using `num_input_channels` flag to toggle XYCD vs XY
* Falling back to standard codecs when `use_depth=False`
* Supporting original H36M format without modification

---

**For questions about specific changes, see the individual files or open an issue.**

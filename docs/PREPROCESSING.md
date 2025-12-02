# **AugLift Data Preprocessing**

This guide covers the full preprocessing pipeline for AugLift, from raw images to training-ready XYCD features.

---

## **Overview**

The preprocessing pipeline consists of three main stages:

1. **MMPose preprocessing** - Extract 2D/3D annotations in camera viewpoint coordinates
2. **Depth & 2D pose estimation** - Generate depth maps and 2D detections
3. **XYCD merging** - Combine all features into unified training files

---

## **Stage 1: MMPose Preprocessing**

### **What it does:**

* Processes raw image files and annotations
* Creates target 2D and 3D data in **camera viewpoint coordinates** (not global)
* Outputs are used as ground truth for training
* **Note:** We train on root-relative keypoints, but this is handled in the codec layer

### **Run preprocessing:**

```bash
# H36M
python mmpose/tools/dataset_converters/preprocess_h36m.py

# 3DPW
python mmpose/tools/dataset_converters/preprocess_3dpw.py

# 3DHP
python mmpose/tools/dataset_converters/preprocess_3dhp.py

# Fit3D
python mmpose/tools/dataset_converters/preprocess_fit3d.py
```

Expected outputs:
* Processed annotations in MMPose format
* Camera parameters
* Train/val/test splits

---

## **Stage 2: Depth Estimation & 2D Pose Detection**

### **What it does:**

The `1_30_estimate_depth_proto_loop_metric_*` scripts jointly perform:

1. **2D bounding box detection** (RTMDet)
2. **2D pose estimation** (RTMPose) 
3. **Monocular depth estimation** (DepthAnything V2)

This captures:
* Sparse depth estimates at keypoint locations
* 2D detected keypoints (to model real-world errors)
* (AugLift V2) Cached feature maps from RTMPose and DepthAnything

### **Format conversion:**

⚠️ **Important:** 2D detections are in **COCO format**, but MMPose 3D trainer expects **H36M format**.

Following previous literature, we convert COCO→H36M using a mapping. **Crucial to handle padding and resizing correctly** between:
* 2D bbox detection
* 2D pose estimation  
* Monocular depth models

### **Pipeline:**

```
Raw Image 
   ↓
RTMDet (2D bbox) 
   ↓
RTMPose (2D keypoints) + DepthAnything (depth map)
   ↓
Extract depth at keypoint locations
   ↓
Save to individual NPZ files (one per frame)
```

---

## **Distributed Execution (SLURM)**

Because datasets are large (especially with feature maps), we distribute processing across **10-20 SLURM nodes** to run overnight.

### **Launch scripts:**

Located in `coarse_depth_experiments/Depth-Anything-OfficialV2/Depth-Anything-V2/metric_depth/`:

```bash
# H36M
bash launch_depth_metric_loop_h36m.sh
bash launch_depth_metric_loop_outer_24_h36m.sh

# 3DPW
bash launch_depth_metric_loop_3dpw.sh

# 3DHP
bash launch_depth_metric_loop_3dhp.sh

# Fit3D
bash launch_depth_metric_loop_fit3d.sh
```

These scripts split the dataset and launch parallel jobs.

---

## **Example: H36M Depth Estimation**

```bash
# Single-node test
python metric_depth/1_30_estimate_depth_proto_loop_metric_h36m_v3_sampling_patch_statistics.py

# Distributed (SLURM)
bash metric_depth/launch_depth_metric_loop_h36m.sh
```

**Outputs:**
* Individual NPZ files per frame (can be several TB for feature maps)
* Each NPZ contains:
  * 2D keypoints (COCO format, converted to H36M)
  * Confidence scores
  * Depth estimates (point, min, max)
  * (V2) Feature maps from RTMPose and DepthAnything

---

## **Stage 3: XYCD Merging**

### **What it does:**

Combines individual per-frame NPZ files into a single unified training file per dataset (a few GB).

Merges:
* **X, Y**: 2D keypoint coordinates
* **C**: Confidence scores
* **D**: Depth estimates
* Dataset metadata (subject, action, camera, etc.)

### **Feature map handling (AugLift V2):**

Per Appendix G and Sections 5.1–5.2:
* Sparse sample feature maps
* Average pool to retain location-invariant local information
* Reduces storage from TB→GB scale

### **Merge scripts:**

```bash
# H36M
python 2_3_merge_xycd_npz_h36m.py

# 3DPW
python 2_3_merge_xycd_npz_3dpw.py

# 3DHP
python 2_3_merge_xycd_npz_3dhp.py

# Fit3D
python 2_3_merge_xycd_npz_fit3d.py
```

### **Processing modes:**

Both single-process and multiprocessing variants are available for faster merging.

---

## **Fit3D Train/Test Split**

Fit3D requires an additional split step:

```bash
python 2_25_split_fit3d_train_test.py
```

This creates `fit3d_train.npz` and `fit3d_test.npz`.

---

## **Depth Statistics Variant (CASP)**

We experimented with additional local depth statistics (mean, std, percentiles), but found that **just using min, max, and point depth** (called **CASP**) performed best.

The statistics variant is referenced in some scripts but is not used in the final models.

---

## **Output Files**

After preprocessing, you should have:

```
h36m_xycd_train.npz
h36m_xycd_val.npz
3dpw_xycd_train.npz
3dpw_xycd_test.npz
3dhp_xycd_train.npz
3dhp_xycd_test.npz
fit3d_xycd_train.npz
fit3d_xycd_test.npz
```

These are the inputs to AugLift training.

---

## **Key Scripts Reference**

### **Depth Estimation**
* `coarse_depth_experiments/Depth-Anything-OfficialV2/Depth-Anything-V2/metric_depth/1_30_estimate_depth_proto_loop_metric_h36m_v3_sampling_patch_statistics.py`
* `...metric_3dpw_v3_sampling_patch_statistics.py`
* `...metric_3dhp_v3_sampling_patch_statistics.py`
* `...metric_fit3d_v3_sampling_patch_statistics.py`

### **SLURM Launch Scripts**
* `launch_depth_metric_loop_h36m.sh`
* `launch_depth_metric_loop_outer_24_h36m.sh`
* Similar scripts for 3DPW, 3DHP, Fit3D

### **XYCD Merge Scripts**
* `2_3_merge_xycd_npz_h36m.py`
* `2_3_merge_xycd_npz_3dpw.py`
* `2_3_merge_xycd_npz_3dhp.py`
* `2_3_merge_xycd_npz_fit3d.py`
* `2_25_split_fit3d_train_test.py`

---

## **Troubleshooting**

### **COCO→H36M format issues:**
* Check joint mapping in `mmpose/mmpose/codecs/image_pose_lifting.py`
* Verify padding/resizing consistency across detection→pose→depth pipeline

### **Feature map storage:**
* Use sparse sampling + average pooling (V2 experiments)
* Consider distributed storage if working with full feature maps

### **SLURM jobs:**
* Check node availability and GPU allocation
* Monitor disk I/O for large writes

---

**Next:** See [TRAINING.md](TRAINING.md) for training instructions.

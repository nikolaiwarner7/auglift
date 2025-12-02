# **AugLift Post-Training Analysis**

This guide covers the post-training analysis pipeline for AugLift models, including quantitative metrics and qualitative visualizations for paper figures.

---

## **Overview**

After training models, we perform comprehensive post-hoc analysis to understand:

1. **Quantitative performance comparison** - Basic MPJPE and qualitative metrics
2. **Geometric Occlusion ("Pipe Man")** - How does physical self-occlusion affect performance?
3. **Pose Similarity (Novelty)** - How does out-of-distribution distance correlate with error?
4. **2D Detection Quality** - How robust is each model to noisy 2D inputs?
5. **Qualitative visualizations** - Generate paper-quality figures and skeleton overlays

**Note:** This analysis uses predictions from the **best in-distribution epoch** (selected via validation performance).

**Additional analyses** (motion speed, body parts, geometric features, visibility-based occlusion) are documented in the **Appendix** section below.

---

## **Analysis Scripts**

### **Main Analysis Script:**

```bash
python poseformer_metrics_simple.py --dataset DATASET --compare MODEL1 MODEL2 [OPTIONS]
```

**Location:** `poseformer_metrics_simple.py`

---

## **1. Basic Metrics Comparison**

### **Compute MPJPE and qualitative metrics for all models:**

```bash
# Single dataset
python poseformer_metrics_simple.py --dataset 3dhp --compare xy xycd

# All datasets
python poseformer_metrics_simple.py --dataset all --compare xy xycd
```

### **Metrics computed:**

**MPJPE percentiles:**
- Mean, median, p25, p50, p75, p90, p95, p99

**Qualitative metrics:**
- **PCK3D** (100mm, 150mm thresholds)
- **PCKt** (scale-invariant, 0.5× and 1.0× torso size)
- **Ordinal depth accuracy** (exact rank, ±1 slack)
- **Coarse ordinal accuracy** (100mm, 250mm depth buckets)
- **Pairwise ordinal accuracy** (depth ordering between all joint pairs)

**Input depth quality (occlusion proxy):**
- **DAV input ordinal accuracy** - Measures how well DepthAnything V2 captures depth ordering
  - Computed as pairwise depth relationship accuracy between DAV depths and GT depths
  - Higher accuracy = clear scene, good depth estimation
  - Lower accuracy = occluded/ambiguous scene, poor depth ordering
  - Used as proxy for occlusion level in downstream analysis

### **Output:**

- Console summary comparing models
- Per-dataset statistics saved to text files

---

## **2. Geometric Occlusion Analysis ("Pipe Man")**

### **Goal:**

Determine how well the model performs when body parts **physically block each other** using 3D geometry and 2D overlap detection.

### **Method:**

This analysis treats the body as a set of "pipes" (cylinders) with anatomically-grounded radii. A joint is occluded when another joint that is closer to the camera overlaps it in 2D space within its radius.

#### **Step 1: Define Body Shape**

Define thickness (radius in pixels) for different body parts at a reference person height of 300px:
- **Torso core:** Root=28px, Spine=26px, Thorax=25px
- **Head/neck:** Head=22px, Neck=20px
- **Upper limbs:** Shoulders=18px, Elbows=15px, Wrists=12px
- **Lower limbs:** Hips=22px, Knees=20px, Feet=15px

#### **Step 2: Normalize Scale**

For each frame:
1. Compute 2D bounding box from GT keypoints
2. Calculate `scale_factor = bbox_height / 300px`
3. Scale radii proportionally to person size

#### **Step 3: Detect Occlusions (The "Pipe" Logic)**

For every joint pair $(i, j)$ in each frame:
1. **Check Depth:** Is joint $i$ physically closer to camera than joint $j$ (by at least 10mm)?
2. **Check 2D Overlap:** Does the 2D projection of joint $j$ fall inside the scaled radius of joint $i$?
3. **Result:** If both conditions are true → joint $j$ is marked **occluded** (visibility = 0)

#### **Step 4: Stratify & Compare**

1. Calculate **occlusion score** per frame: $1.0 - \text{mean\_visibility}$
2. Sort frames into quartiles:
   - **Q1 (Clear):** Lowest occlusion scores (≤ 25th percentile)
   - **Q2 (Mild):** 25th-50th percentile
   - **Q3 (Moderate):** 50th-75th percentile
   - **Q4 (Occluded):** Highest occlusion scores (> 75th percentile)
3. Compute MPJPE for each model in each quartile

### **Usage:**

```bash
python poseformer_metrics_simple.py \
    --dataset 3dhp \
    --compare xy xycd \
    --occlusion_analysis \
    --occlusion_level geometric
```

### **Hypothesis:**

- AugLift should show **larger improvements** in Q3-Q4 (moderate to high occlusion)
- Baseline should struggle when depth cues are ambiguous due to self-occlusion

### **Output:**

- Per-quartile MPJPE statistics (mean, median, p90, p95)
- Per-joint occlusion rates (which joints are occluded most often)
- Comparison table showing Q4-Q1 performance gap for each model
- Visualizations of representative frames from each quartile

---

## **3. Pose Similarity Analysis (Novelty)**

### **Goal:**

Measure if the model fails on poses it **hasn't seen before** (Out-Of-Distribution) by comparing test poses to training distribution prototypes.

### **Method:**

#### **Step 1: Build Source Prototypes**

1. Load entire **H36M** training dataset (Ground Truth 3D poses)
2. Flatten poses into vectors: $(N, 51)$ dimensions (17 joints × 3 coords)
3. Run **K-Means Clustering** with $K=100$ to find 100 "Prototype Poses" that represent the training distribution

#### **Step 2: Measure Novelty**

For every frame in the **Test Set** (e.g., 3DPW, 3DHP, Fit3D):
1. Flatten the GT pose into a 51D vector
2. Calculate Euclidean distance to the **nearest** H36M prototype
3. This distance = "novelty score" (higher = more out-of-distribution)

#### **Step 3: Stratify by Similarity**

Sort all test frames by distance to prototypes:
- **Tier 1 (Most Similar 20%):** Frames that look like H36M (lowest distances)
- **Tier 2 (Middle 60%):** Moderate similarity
- **Tier 3 (Least Similar 20%):** Frames that look nothing like H36M (highest distances)

#### **Step 4: Compare Performance**

Compute each model's MPJPE for each tier to see if error spikes on Tier 3 (OOD poses).

### **Usage:**

```bash
python poseformer_metrics_simple.py \
    --dataset 3dpw \
    --compare xy xycd \
    --domain_similarity
```

**Note:** Analysis is performed on test datasets only (H36M excluded as it's the source domain).

### **Hypothesis:**

- Baseline (XY) should show **larger degradation** on Tier 3 (OOD poses)
- AugLift (XYCD) should **generalize better** due to richer geometric input (depth cues help on novel poses)

### **Output:**

- MPJPE statistics per similarity tier (mean, median, p90, p95)
- Tier 3 vs Tier 1 comparison (OOD penalty)
- Distance distribution statistics
- Visualizations of representative frames from each tier with distance annotations

---

## **4. 2D Pose Estimator Reliability**

### **Goal:**

Test if the 3D model is **robust when input 2D keypoints are noisy or wrong** by stratifying frames by 2D detection error.

### **Method:**

#### **Step 1: Load Data**

1. Load **Predicted 2D Keypoints** (what the 3D model sees as input)
2. Load **Ground Truth 2D Keypoints** (where joints actually are in the image)

#### **Step 2: Calculate Input Error**

1. Compute L2 distance (pixel error) between Predicted and GT for every joint
2. **Filter:** Ignore any detection > 100 pixels off (treat as invalid/outlier)
3. Calculate **mean 2D error** per frame (or per joint for joint-level analysis)

#### **Step 3: Stratify by Quality**

Sort frames into quartiles based on mean 2D error:
- **Q1 (Best 25%):** High-quality 2D inputs (low pixel error)
- **Q2 (Good 25%):** Medium-high quality
- **Q3 (Poor 25%):** Medium-low quality
- **Q4 (Worst 25%):** Noisy/jittery 2D inputs (high pixel error)

#### **Step 4: Compare Performance**

Compute 3D MPJPE for each quartile to see if the model survives bad inputs (Q4).

### **Usage:**

#### **Frame-Level Analysis:**

```bash
python poseformer_metrics_simple.py \
    --dataset 3dhp \
    --compare xy xycd \
    --detection_quality_analysis \
    --detection_quality_level frame
```

#### **Joint-Level Analysis:**

```bash
python poseformer_metrics_simple.py \
    --dataset 3dhp \
    --compare xy xycd \
    --detection_quality_analysis \
    --detection_quality_level joint
```

### **Hypothesis:**

- AugLift should be **more robust** to noisy 2D inputs (Q4) due to depth prior providing alternative signal
- Baseline should show **significant degradation** from Q1 → Q4 (heavily reliant on 2D quality)
- Q4-Q1 gap should be smaller for XYCD than XY

### **Output:**

- Per-quartile MPJPE statistics (mean, median, p90, p95)
- Per-quartile mean 2D error (shows input quality)
- Q4 vs Q1 comparison (robustness to noise)
- Visualizations of representative frames from each quartile with 2D error annotations

---

## **5. Visualizations**

### **5.1. Error Distribution Plots:**

Generates 3-panel plots:
- MPJPE histograms (XY vs XYCD)
- Absolute delta distribution
- Relative delta distribution

```bash
python poseformer_metrics_simple.py \
    --dataset 3dhp \
    --compare xy xycd \
    --visualize
```

### **5.2. Frame Visualizations (Top Improvements):**

### **Workflow 1: Full Analysis for Paper**

```bash
# 1. Basic metrics + visualizations
python poseformer_metrics_simple.py --dataset all --compare xy xycd --visualize

# 2. Geometric occlusion analysis ("Pipe Man")
python poseformer_metrics_simple.py --dataset 3dhp --compare xy xycd --occlusion_analysis --occlusion_level geometric

# 3. Pose similarity (novelty)
python poseformer_metrics_simple.py --dataset 3dpw --compare xy xycd --domain_similarity

# 4. 2D detection quality
python poseformer_metrics_simple.py --dataset 3dhp --compare xy xycd --detection_quality_analysis

# 5. Representative poses (for paper figures)
python poseformer_metrics_simple.py --dataset 3dhp --compare xy xycd --cluster --num_clusters 100
```

### **Workflow 2: Quick Comparison**

```bash
# Just metrics + top improvements visualization
python poseformer_metrics_simple.py \
    --dataset 3dhp \
    --compare xy xycd \
    --visualize \
    --num_vis 50
```

### **Workflow 3: Occlusion-Focused Analysis**

```bash
# Geometric occlusion analysis with visualizations
python poseformer_metrics_simple.py \
    --dataset 3dhp \
    --compare xy xycd \
    --occlusion_analysis \
    --occlusion_level geometric
```

---



## **APPENDIX: Additional Analyses**

The following analyses provide deeper insights but are not included in the main paper figures.

---

### **A1. Visibility-Based Occlusion Analysis**

Alternative occlusion metrics using 2D keypoint confidence scores instead of geometric computation.

#### **A1.1. Frame-Level (2D Visibility-Based):**

Uses mean 2D keypoint confidence per frame as occlusion proxy.

```bash
python poseformer_metrics_simple.py \
    --dataset 3dhp \
    --compare xy xycd \
    --occlusion_analysis \
    --occlusion_level frame
```

**Stratification:**
- Fixed visibility bins: ≤0.30, (0.30-0.40], ..., (0.90-1.00]
- Enables cross-dataset comparison with consistent thresholds

#### **A1.2. Joint-Level (2D Visibility-Based):**

Analyzes each joint independently based on its visibility score.

```bash
python poseformer_metrics_simple.py \
    --dataset 3dhp \
    --compare xy xycd \
    --occlusion_analysis \
    --occlusion_level joint
```

#### **A1.3. Depth-Based Occlusion (DAV Ordering Accuracy):**

Uses DAV input depth ordering accuracy as occlusion proxy.

```bash
python poseformer_metrics_simple.py \
    --dataset 3dhp \
    --compare xy xycd \
    --occlusion_analysis \
    --occlusion_level depth
```

---

### **A2. Motion Speed Analysis**

Analyze MPJPE stratified by local motion speed (mm/frame).

#### **Goal:**

Understand how model performance varies with motion dynamics.

#### **Method:**

1. Compute per-joint local speed (mm/frame) via finite differences
2. Smooth speed with centered moving average (window frames)
3. Respect clip boundaries (no smoothing across scene cuts)
4. Stratify frames/joints by speed bins (quantile-based or fixed edges)
5. Compute MPJPE statistics for each speed tier

#### **Speed Levels:**

**Frame-Level Motion:** Mean joint motion speed (relative to root) per frame.

```bash
python poseformer_metrics_simple.py \
    --dataset 3dhp \
    --compare xy xycd \
    --motion_speed_analysis \
    --speed_level frame \
    --speed_window 10
```

**Root-Level Motion:** Body translation speed through space (before root-centering).

```bash
python poseformer_metrics_simple.py \
    --dataset 3dhp \
    --compare xy xycd \
    --motion_speed_analysis \
    --speed_level root \
    --speed_window 10
```

**Joint-Level Motion:** Per-joint motion speed for all 17 joints.

```bash
python poseformer_metrics_simple.py \
    --dataset 3dhp \
    --compare xy xycd \
    --motion_speed_analysis \
    --speed_level joint \
    --speed_window 10
```

**Parameters:**
- `--speed_window`: Smoothing window length (default: 10 frames)
- `--speed_bins`: Number of speed bins (default: 5)
- `--speed_level`: Analysis level (frame, root, or joint)

**Hypothesis:**
- High-speed motion (rapid movement) may be harder to predict accurately
- AugLift may be more robust to motion dynamics due to richer input

**Output:**
- MPJPE statistics per speed tier (Q1-Q5 or custom bins)
- Comparison table showing performance across motion speeds
- Body-part-specific speed and error correlations (frame-level only)
- Visualizations of representative frames from each speed tier

---

### **A3. Body Part Analysis**

Identify which joint groups benefit most from augmented inputs.

#### **Goal:**

Determine which body parts show the largest improvements from depth augmentation, independent of motion speed.

#### **Method:**

1. Define body part groups:
   - Upper appendages (shoulders, elbows, wrists)
   - Lower appendages (hips, knees, feet)
   - Torso (spine, thorax)
   - Head (neck_base, head)
   - All appendages (arms + legs combined)
2. Compute per-body-part MPJPE for each model
3. Compare improvements (Δ MPJPE) across body parts

```bash
python poseformer_metrics_simple.py \
    --dataset 3dhp \
    --compare xy xycd \
    --body_part_analysis
```

**Hypothesis:**
- Distal joints (wrists, feet) may benefit more from depth cues
- Appendages may show larger improvements than torso/head

**Output:**
- Per-body-part MPJPE comparison (mean, median, p90, p95)
- Per-joint MPJPE statistics (all 17 joints)
- Ranked list of body parts by improvement magnitude
- Delta (mm) and relative improvement (%) for each body part

---

### **A4. Geometric Features Analysis**

Analyze MPJPE by geometric features that explain tail error patterns.

#### **Goal:**

Understand **why** augmented inputs help on certain frames by examining geometric properties.

#### **Method:**

Stratify frames by four key geometric dimensions:

1. **Bounding-box scale** (2D person size in pixels)
   - Proxy for camera distance and perspective strength
   - Larger scale → stronger perspective distortion

2. **Foreshortening ratio** (2D projected length / 3D true length)
   - Low ratio (<0.5) → severe foreshortening (limbs toward/away from camera)
   - Proxy for depth ambiguity from projection

3. **Torso pitch** (forward/backward lean in degrees)
   - Measures self-occlusion due to body inclination
   - Higher pitch → more self-occlusion

4. **Ordinal margin** (fraction of joint pairs with near-equal depth)
   - Threshold: |Δz| < 100mm
   - High margin → many ambiguous depth pairs → prone to ordering errors

```bash
python poseformer_metrics_simple.py \
    --dataset 3dhp \
    --compare xy xycd \
    --geometric_features
```

**Hypothesis:**
- AugLift should show larger improvements at:
  - Large bbox scales (strong perspective)
  - Low foreshortening ratios (severe distortion)
  - Moderate-to-high torso pitch (self-occlusion)
  - High ordinal margin (depth ambiguity)

**Output:**
- MPJPE statistics per quartile for each feature
- Comparison of Δp90 and Δp95 across quartiles
- Q4-Q1 span analysis (trend across feature range)
- Summary of key findings (which features correlate with improvements)

---

### **A5. Angle & Orientation Analysis**

Analyze bone orientation errors (alternative to Euclidean MPJPE).

```bash
python poseformer_metrics_simple.py \
    --dataset 3dhp \
    --compare xy xycd \
    --angles_orientations
```

**Metrics:**
- Per-bone X/Y angle errors (rotation in XZ and YZ planes)
- Torso orientation error (normal vector angular difference)



### **5.3. Contrast Visualizations (Large Improvement + Good Absolute Performance):**

Find frames where:
1. XYCD improves significantly over XY (large negative delta)
2. XYCD absolute error ≤ threshold (good performance)

**This highlights scenes where XYCD excels.**

```bash
python poseformer_metrics_simple.py \
    --dataset 3dhp \
    --compare xy xycd \
    --visualize_contrast \
    --contrast_threshold 150
```

### **5.4. Representative Poses (Clustering):**

Find **most diverse and representative poses** using K-means clustering in 3D GT pose space:

1. Cluster GT poses into 100 types
2. Select frame closest to each cluster center
3. Sort by cluster size (most common poses first)
4. Visualize each representative pose

**Purpose:** Show AugLift performance across diverse pose types (not biased by error).

```bash
python poseformer_metrics_simple.py \
    --dataset 3dhp \
    --compare xy xycd \
    --cluster \
    --num_clusters 100
```

**Output:** 100 representative frames with skeleton overlays (XY vs XYCD side-by-side)

---

## **6. Confidence Filtering**

### **Filter low-confidence 2D keypoints from metrics:**

```bash
python poseformer_metrics_simple.py \
    --dataset 3dhp \
    --compare xy xycd \
    --confidence_threshold 0.3
```

**Effect:**
- Joints with 2D confidence ≤ 0.3 are excluded from MPJPE computation
- Reduces bias from unreliable 2D detections
- Default: 0.3 (recommended for RTMPose)

---

## **7. Example Workflows**

### **Workflow 1: Full Analysis for Paper**

```bash
# 1. Basic metrics + visualizations
python poseformer_metrics_simple.py --dataset all --compare xy xycd --visualize

# 2. Geometric occlusion analysis ("Pipe Man")
python poseformer_metrics_simple.py --dataset 3dhp --compare xy xycd --occlusion_analysis --occlusion_level geometric

# 3. Pose similarity (novelty)
python poseformer_metrics_simple.py --dataset 3dpw --compare xy xycd --domain_similarity

# 4. 2D detection quality
python poseformer_metrics_simple.py --dataset 3dhp --compare xy xycd --detection_quality_analysis

# 5. Representative poses (for paper figures)
python poseformer_metrics_simple.py --dataset 3dhp --compare xy xycd --cluster --num_clusters 100
```

### **Workflow 2: Quick Comparison**

```bash
# Just metrics + top improvements visualization
python poseformer_metrics_simple.py \
    --dataset 3dhp \
    --compare xy xycd \
    --visualize \
    --num_vis 50
```

---

## **9. Output Structure**

### **Directory Layout:**

```
/srv/essa-lab/flash3/nwarner30/pose_estimation/poseformer_analysis/
├── xy_vs_xycd/
│   ├── error_distributions_3dhp_xy_vs_xycd.png
│   ├── selected_frames_3dhp_xy_vs_xycd_80_95.txt
│   └── frames_3dhp_xy_vs_xycd/
│       ├── rank_001_delta_-45.2mm_*.png
│       └── ...
├── xy_vs_xycd_contrast_improvements/
│   ├── contrast_frames_3dhp_xy_vs_xycd_threshold150mm.txt
│   └── frames_3dhp_xy_vs_xycd/
│       └── ...
├── xy_vs_xycd_representative_cluster_scenes/
│   ├── representative_poses_3dhp_xy_vs_xycd_k100.txt
│   └── frames_3dhp_xy_vs_xycd/
│       └── ...
├── xy_vs_xycd_domain_similarity/
│   ├── domain_similarity_3dpw_vs_h36m_xy.txt
│   ├── domain_similarity_3dpw_vs_h36m_xycd.txt
│   └── frames_3dpw_xy_vs_xycd_similarity_tiers/
│       └── ...
├── xy_vs_xycd_occlusion_analysis/
│   ├── geometric_occlusion_3dhp_xy.txt
│   ├── geometric_occlusion_3dhp_xycd.txt
│   ├── geometric_occlusion_comparison_3dhp_xy_vs_xycd.txt
│   └── frames_3dhp_xy_vs_xycd_occlusion_tiers/
│       └── ...
├── xy_vs_xycd_detection_quality_analysis/
│   ├── detection_quality_analysis_3dhp_xy.txt
│   ├── detection_quality_analysis_3dhp_xycd.txt
│   └── frames_3dhp_xy_vs_xycd_detection_quality/
│       └── ...
└── (additional appendix analyses below)
```

---

Visualize frames where XYCD improves most over XY:

- Top 50 improvements (sorted by relative delta)
- 150 random frames from percentile window
- 10-panel layout per frame:
  - Original image
  - Image + 2D pose overlay
  - 2D confidence visualization (color-coded by confidence)
  - Depth visualization (color-coded by GT depth)
  - 3D skeletons from 3 viewing angles (XY vs XYCD side-by-side)

```bash
python poseformer_metrics_simple.py \
    --dataset 3dhp \
    --compare xy xycd \
    --visualize \
    --num_vis 200 \
    --min_pct 80 \
    --max_pct 95
```

**Parameters:**
- `--num_vis`: Total frames to visualize (default: 200)
- `--min_pct`: Min percentile for XY error (default: 80)
- `--max_pct`: Max percentile for XY error (default: 95)



## **10. Citation & References**

If you use this analysis pipeline, please cite:

- **MMPose:** For evaluation metrics and dataset infrastructure
- **DepthAnything V2:** For depth ordering quality metric
- **RTMPose:** For 2D detection confidence scores

---

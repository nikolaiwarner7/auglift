# **AugLift Post-Training Analysis**

This guide covers the post-training analysis pipeline for AugLift models, including quantitative metrics and qualitative visualizations for paper figures.

---

## **Overview**

After training models, we perform comprehensive post-hoc analysis to understand:

1. **Quantitative performance comparison** across input representations (XY, XYC, XYD, XYCD)
2. **Occlusion sensitivity** - How does occlusion affect AugLift vs baseline?
3. **Domain similarity** - How does out-of-distribution distance correlate with error?
4. **2D detection quality** - How robust is each model to noisy 2D inputs?
5. **Qualitative visualizations** - Generate paper-quality figures and skeleton overlays

**Note:** This analysis uses predictions from the **best in-distribution epoch** (selected via validation performance).

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

## **2. Occlusion Analysis**

### **Goal:**

Stratify frames by occlusion level and analyze how AugLift performs vs baseline in occluded vs clear scenes.

### **Occlusion Metrics:**

We use **three levels** of occlusion analysis:

#### **2.1. Frame-Level (2D Visibility-Based):**

Uses mean 2D keypoint confidence per frame as occlusion proxy.

```bash
python poseformer_metrics_simple.py \
    --dataset 3dhp \
    --compare xy xycd \
    --occlusion_analysis \
    --occlusion_level frame
```

**Stratification:**
- **Tier 1:** Low occlusion (high mean visibility > 67th percentile)
- **Tier 2:** Medium occlusion (33rd-67th percentile)
- **Tier 3:** High occlusion (visibility 0.3 to 33rd percentile)
- **Tier 4:** Very high occlusion (visibility ≤ 0.3)

#### **2.2. Joint-Level (2D Visibility-Based):**

Analyzes each joint independently based on its visibility score.

```bash
python poseformer_metrics_simple.py \
    --dataset 3dhp \
    --compare xy xycd \
    --occlusion_analysis \
    --occlusion_level joint
```

**Use case:** Understand which joints benefit most from depth augmentation under occlusion.

#### **2.3. Depth-Based Occlusion (DAV Ordering Accuracy):**

Uses **DAV input depth ordering accuracy** as occlusion proxy (Section 1 metric).

```bash
python poseformer_metrics_simple.py \
    --dataset 3dhp \
    --compare xy xycd \
    --occlusion_analysis \
    --occlusion_level depth
```

**Intuition:**
- Frames where DAV correctly captures depth ordering (high pairwise accuracy) = **clear scenes**
- Frames where DAV fails at depth ordering (low pairwise accuracy) = **occluded/ambiguous scenes**

**Stratification:**
- **Q1 (best 25%):** High DAV accuracy (clear scene)
- **Q2 (good 25%):** Medium-high accuracy
- **Q3 (poor 25%):** Medium-low accuracy
- **Q4 (worst 25%):** Low DAV accuracy (occluded scene)

**Why this matters:**
- Tests hypothesis: AugLift should help more in **moderately occluded** scenes (Q3/Q4)
- Baseline should struggle more when depth cues are ambiguous

### **Output:**

- Statistics per occlusion tier (mean MPJPE, median, p90, p95)
- Comparison table (XY vs XYCD) for each tier
- Visualization of representative frames from each tier

---

## **3. Domain Similarity Analysis**

### **Goal:**

Measure how **out-of-distribution distance** affects generalization. We cluster H36M (source domain) poses and measure test frame similarity.

### **Method:**

1. **Cluster H36M GT poses** into ~100 representative pose types (K-means in 51D space: 17 joints × 3 coords)
2. For each **test frame**, compute distance to nearest H36M cluster center
3. **Stratify test frames** by similarity:
   - **Most similar 20%:** Poses very close to H36M distribution
   - **Middle 60%:** Moderate similarity
   - **Least similar 20%:** Out-of-distribution poses
4. Compare MPJPE across similarity tiers

### **Usage:**

```bash
python poseformer_metrics_simple.py \
    --dataset 3dpw \
    --compare xy xycd \
    --domain_similarity
```

**Note:** H36M is excluded (it's the source domain).

### **Hypothesis:**

- Baseline (XY) should degrade more on OOD poses (least similar 20%)
- AugLift (XYCD) should generalize better due to richer input

### **Output:**

- MPJPE statistics per similarity tier
- Comparison: Least similar vs Most similar
- Visualizations with similarity distance annotations

---

## **4. 2D Detection Quality Analysis**

### **Goal:**

Analyze robustness to **noisy 2D inputs** by stratifying frames by 2D detection error.

### **Method:**

1. Load **GT 2D keypoints** (from ground truth annotations)
2. Compute **2D detection error** = L2 distance between predicted 2D keypoints and GT 2D
3. Filter out detections > 100px error (invalid)
4. **Stratify frames** into quartiles by 2D error:
   - **Q1 (best 25%):** Lowest 2D error (high-quality detections)
   - **Q2-Q3:** Medium quality
   - **Q4 (worst 25%):** Highest 2D error (noisy detections)
5. Compare 3D MPJPE across detection quality tiers

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

- AugLift should be more robust to noisy 2D inputs (Q4) due to depth prior
- Baseline should degrade significantly with poor detections

### **Output:**

- MPJPE statistics per detection quality quartile
- Comparison: Worst vs Best detection quality
- Visualizations with 2D error annotations

---

## **5. Angle & Orientation Analysis**

### **Goal:**

Analyze **bone orientation errors** (alternative to Euclidean MPJPE) to understand geometric accuracy.

### **Method:**

1. **Per-bone angle errors:**
   - Compute X-angle (rotation in XZ plane, about Y-axis)
   - Compute Y-angle (rotation in YZ plane, about X-axis)
   - Compare predicted vs GT bone directions

2. **Torso orientation error:**
   - Compute torso normal vector (cross product of hip and shoulder vectors)
   - Measure angular difference between predicted and GT torso orientation

### **Usage:**

```bash
python poseformer_metrics_simple.py \
    --dataset 3dhp \
    --compare xy xycd \
    --angles_orientations
```

### **Output:**

- Per-bone X/Y angle errors (mean across frames)
- Overall angle error statistics
- Torso orientation error distribution
- Comparison table between models

---

## **6. Visualizations**

### **6.1. Error Distribution Plots:**

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

### **6.2. Frame Visualizations (Top Improvements):**

Visualize frames where XYCD improves most over XY:

- Top 50 improvements (sorted by relative delta)
- 150 random frames from percentile window
- 10-panel layout per frame:
  - Original image
  - Image + 2D pose overlay
  - 2D confidence visualization (color-coded by confidence)
  - Depth visualization (color-coded by GT depth)
  - 3D skeletons from 3 viewing angles (XY vs XYCD side-by-side)

**Filtering:**
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

**Output:** Saved to `/srv/essa-lab/flash3/nwarner30/pose_estimation/poseformer_analysis/xy_vs_xycd/frames_*/`

### **6.3. Contrast Visualizations (Large Improvement + Good Absolute Performance):**

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

### **6.4. Representative Poses (Clustering):**

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

## **7. Confidence Filtering**

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

## **8. Output Structure**

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
│   ├── occlusion_analysis_3dhp_xy.txt
│   ├── occlusion_analysis_3dhp_xycd.txt
│   ├── occlusion_comparison_3dhp_xy_vs_xycd.txt
│   └── frames_3dhp_xy_vs_xycd_occlusion_tiers/
│       └── ...
├── xy_vs_xycd_depth_ordering_occlusion/
│   ├── depth_ordering_occlusion_3dhp_xy.txt
│   ├── depth_ordering_occlusion_3dhp_xycd.txt
│   ├── depth_ordering_comparison_3dhp_xy_vs_xycd.txt
│   └── frames_3dhp_xy_vs_xycd_depth_ordering_tiers/
│       └── ...
├── xy_vs_xycd_detection_quality_analysis/
│   ├── detection_quality_analysis_3dhp_xy.txt
│   ├── detection_quality_analysis_3dhp_xycd.txt
│   └── frames_3dhp_xy_vs_xycd_detection_quality/
│       └── ...
└── xy_vs_xycd_angles_orientations/
    └── angles_orientations_3dhp_comparison.txt
```

---

## **9. Key Implementation Details**

### **9.1. Pairwise Depth Ordering Accuracy (Occlusion Proxy):**

```python
def pairwise_depth_accuracy(z_a, z_b):
    """Compute pairwise depth ordering accuracy.
    
    For each frame, computes:
    - For all joint pairs (i,j): does z_a predict correct front/behind relationship?
    - sign(z_a[i] - z_a[j]) == sign(z_b[i] - z_b[j])
    - Accuracy = fraction of pairs with matching order (0 to 1)
    
    Higher accuracy = better ordinal depth accuracy (clear scene)
    Lower accuracy = poor depth ordering (occluded/ambiguous scene)
    """
```

**Use cases:**
1. **Model pairwise ordinal accuracy:** Compare predicted Z vs GT Z
2. **DAV input quality:** Compare DAV depths vs GT Z (occlusion proxy)

### **9.2. Coarse Ordinal Depth:**

Groups joints into depth buckets (threshold-based):

```python
def compute_coarse_ordinal_depth(joints_3d, threshold_mm=250):
    """Assign same rank to joints within threshold_mm of each other.
    
    More forgiving than exact ordinal ranking.
    Useful for measuring depth understanding at coarser granularity.
    """
```

### **9.3. Angle & Orientation Errors:**

```python
def compute_angle_errors(pred_coords, gt_coords, mask):
    """Per-bone X/Y angle errors (rotation in XZ and YZ planes)."""

def compute_torso_orientation_error(pred_coords, gt_coords, mask):
    """Torso normal vector angular difference (cross product of hip/shoulder vectors)."""
```

---

## **10. Example Workflows**

### **Workflow 1: Full Analysis for Paper**

```bash
# 1. Basic metrics + visualizations
python poseformer_metrics_simple.py --dataset all --compare xy xycd --visualize

# 2. Occlusion analysis (all 3 levels)
python poseformer_metrics_simple.py --dataset 3dhp --compare xy xycd --occlusion_analysis --occlusion_level frame
python poseformer_metrics_simple.py --dataset 3dhp --compare xy xycd --occlusion_analysis --occlusion_level joint
python poseformer_metrics_simple.py --dataset 3dhp --compare xy xycd --occlusion_analysis --occlusion_level depth

# 3. Domain similarity
python poseformer_metrics_simple.py --dataset 3dpw --compare xy xycd --domain_similarity

# 4. Detection quality
python poseformer_metrics_simple.py --dataset 3dhp --compare xy xycd --detection_quality_analysis

# 5. Angle/orientation analysis
python poseformer_metrics_simple.py --dataset 3dhp --compare xy xycd --angles_orientations

# 6. Representative poses (for paper figures)
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
# Depth-based occlusion analysis with visualizations
python poseformer_metrics_simple.py \
    --dataset 3dhp \
    --compare xy xycd \
    --occlusion_analysis \
    --occlusion_level depth
```

---

## **11. Key Findings Expected**

Based on the analysis, you should observe:

1. **Occlusion:** AugLift (XYCD) should show **larger improvements** in medium-high occlusion tiers (Tier 3-4)
2. **Domain Similarity:** AugLift should **generalize better** on OOD poses (least similar 20%)
3. **Detection Quality:** AugLift should be **more robust** to noisy 2D inputs (Q4 worst detections)
4. **Depth Ordering:** In scenes where DAV depth is ambiguous (Q4), AugLift should still outperform XY
5. **Angles:** AugLift should show better **geometric accuracy** (lower bone orientation errors)

---

## **12. Future Improvements (TODO)**

Per the script header comment:

```
{TODO: Refactor into separate scripts}
```

**Suggested refactoring:**

1. `compute_metrics.py` - Core MPJPE and qualitative metrics
2. `occlusion_analysis.py` - Occlusion stratification (all 3 levels)
3. `domain_similarity.py` - OOD distance analysis
4. `detection_quality.py` - 2D input robustness
5. `visualizations.py` - Frame visualizations and plots
6. `angles_orientations.py` - Geometric error analysis

**Benefits:**
- Cleaner code organization
- Easier to run individual analyses
- Better maintainability

---

## **13. Citation & References**

If you use this analysis pipeline, please cite:

- **MMPose:** For evaluation metrics and dataset infrastructure
- **DepthAnything V2:** For depth ordering quality metric
- **RTMPose:** For 2D detection confidence scores

---

**For questions about specific metrics or visualizations, see the inline documentation in `poseformer_metrics_simple.py`.**

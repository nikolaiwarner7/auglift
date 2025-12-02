#!/usr/bin/env python3
"""Compute metrics for PoseFormer models - SIMPLIFIED VERSION.

PoseFormer makes single-frame predictions, so no complex sequence matching is needed.
This script computes MPJPE, PCK3D, PCKt, and ordinal depth accuracy metrics.

PSEUDOCODE:
-----------
1. Load predictions for specified model variants (XY, XYC, XYD, XYCD)
   - Read from npz files: june_25_poseformer_poseformer_testds_{variant}_sl27_{dataset}_results.npz
   - Flatten sequence format (1, 27, 17, 3) to frame-level (N, 17, 3)

2. Load ground truth 3D poses
   - Read from merged_data_{dataset}_v4_hr.npz
   - Extract GT poses (N, 17, 3) and image names

3. Align predictions with GT
   - Match lengths (min of pred and GT)
   - Center at root joint (subtract joint 0)
   - Flip Y-axis (multiply by -1)
   - Convert to millimeters (multiply by 1000)

4. Compute MPJPE metrics
   - Per-joint errors: L2 norm between pred and GT
   - Per-frame MPJPE: mean across joints
   - Statistics: mean, median, percentiles (p25, p50, p75, p90, p95, p99)

5. Compute qualitative metrics
   - PCK3D: percentage of joints within threshold (100mm, 150mm)
   - PCKt: scale-invariant accuracy relative to torso size (0.5, 1.0 ratios)
   - Ordinal depth accuracy: correct ranking of joint depths (exact, ±1)
   - Input depth ordinal accuracy (DAV vs GT):
     * Load DAV monocular depth estimates (predicted_da_depth)
     * Preprocess depths same as poses (root-center, y-flip, convert to mm)
     * For each frame, compute pairwise depth ordering accuracy:
       - For all joint pairs (i,j): does DAV predict correct front/behind relationship?
       - sign(dav_depth[i] - dav_depth[j]) == sign(gt_depth[i] - gt_depth[j])
       - Accuracy = fraction of pairs with matching order (0 to 1)
     * Higher accuracy = DAV correctly captures depth relationships (clear scene)
     * Lower accuracy = DAV fails at depth ordering (occluded/ambiguous scene)
     * Use as proxy for occlusion: low accuracy → likely occluded

6. Compare models
   - Compute delta: model2 - model1
   - Relative delta: (model2 - model1) / model1
   - Print comparison tables

7. Generate visualizations (optional)
   - 3-panel plots: MPJPE distributions, absolute delta, relative delta
   - Find frames where model2 improves most (in specified percentile window)
   - Save frame lists and statistics

USAGE:
------
# Basic comparison (XY vs XYCD on 3DHP, default seqlen=27)
python sep_2025_experiments/poseformer_metrics_simple_with_autorotate.py --dataset 3dhp --compare xy xycd --test_valid_only

#And cluster
python sep_2025_experiments/poseformer_metrics_simple.py --dataset 3dhp --compare xy xycd --cluster

#3dpw
python sep_2025_experiments/poseformer_metrics_simple.py --dataset 3dpw --compare xy xycd --visualize

# All datasets with visualizations
python poseformer_metrics_simple.py --dataset all --compare xy xycd --visualize

# Custom sequence length (if you have models trained with different seqlen)
python poseformer_metrics_simple.py --dataset 3dhp --compare xy xycd --seqlen 81


# Analysis of alternative joint angle error
# Run angle/orientation analysis for XY vs XYCD on 3DHP
python sep_2025_experiments/poseformer_metrics_simple.py  --dataset 3dhp --compare xy xycd --angles_orientations

# Body part analysis (which joint groups benefit most from augmented inputs)
# Independent of motion speed
python sep_2025_experiments/poseformer_metrics_simple.py --dataset 3dhp --compare xy xycd --body_part_analysis

# Geometric features analysis (bbox scale, foreshortening, torso pitch, ordinal margin)
# Reveals structural patterns behind tail error reduction
python sep_2025_experiments/poseformer_metrics_simple.py --dataset 3dhp --compare xy xycd --geometric_features

# Motion speed analysis with different levels
# Frame-level: mean joint motion (relative to root)
python sep_2025_experiments/poseformer_metrics_simple.py --dataset 3dhp --compare xy xycd --motion_speed_analysis --speed_level frame

# Root-level: body translation through space (before root-centering)
python sep_2025_experiments/poseformer_metrics_simple.py --dataset 3dhp --compare xy xycd --motion_speed_analysis --speed_level root

# Joint-level: per-joint motion (all joints)
python sep_2025_experiments/poseformer_metrics_simple.py --dataset 3dhp --compare xy xycd --motion_speed_analysis --speed_level joint


"""

import numpy as np
import os
import argparse
from scipy.stats import rankdata
import ipdb
import math

# --- Fast image-view auto-alignment helpers ---------------------------------
def _yaw_pitch_rotate_xyz(points_xyz: np.ndarray, yaw_deg: float, pitch_deg: float):
    """
    Rotate Nx3 points around Y (yaw) then X (pitch). Coordinates are [x, y, z] in mm.
    Returns rotated points (translation removed inside the caller).
    """
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)

    Ry = np.array([[ math.cos(yaw), 0.0,  math.sin(yaw)],
                   [ 0.0,           1.0,  0.0          ],
                   [-math.sin(yaw), 0.0,  math.cos(yaw)]], dtype=float)
    Rx = np.array([[1.0, 0.0,            0.0           ],
                   [0.0, math.cos(pitch),-math.sin(pitch)],
                   [0.0, math.sin(pitch), math.cos(pitch)]], dtype=float)

    R = Rx @ Ry
    return (points_xyz @ R.T)

def _procrustes_scale_translate(P: np.ndarray, Q: np.ndarray, mask: np.ndarray=None):
    """
    2D isotropic Procrustes: find s, t that minimize || s*P + t - Q ||.
    Returns aligned P and MSE. If mask provided, only those joints are used.
    """
    if mask is not None:
        P = P[mask]
        Q = Q[mask]
    if len(P) == 0 or len(Q) == 0:
        return None, np.inf

    Pc = P - P.mean(axis=0, keepdims=True)
    Qc = Q - Q.mean(axis=0, keepdims=True)

    denom = np.sum(Pc**2) + 1e-8
    s = float(np.sum(Pc * Qc) / denom)   # scalar scale
    P_aligned = s * Pc

    # Best translation after scaling
    t = Q.mean(axis=0, keepdims=True) - P_aligned.mean(axis=0, keepdims=True)
    P_aligned = P_aligned + t
    mse = float(np.mean((P_aligned - Q)**2))
    return P_aligned, mse

def find_best_view(points3d_xyz: np.ndarray,
                   keypoints2d,
                   vis_scores,
                   conf_thresh: float = 0.25,
                   n_iter: int = 8):
    """
    Returns (best_elev_deg, best_azim_deg) in Matplotlib view_init convention.
    Works even with root-relative 3D (we center & scale).
    If 2D is missing, falls back to a reasonable side view (5°, 90°).
    """
    if keypoints2d is None or len(keypoints2d) != points3d_xyz.shape[0]:
        return 5.0, 90.0  # default

    # mask by confidence if provided
    mask = None
    if vis_scores is not None and len(vis_scores) == len(keypoints2d):
        mask = (vis_scores >= conf_thresh)

    # center both spaces (translation is free anyway)
    P3 = points3d_xyz - points3d_xyz.mean(axis=0, keepdims=True)
    K2 = keypoints2d.copy()

    # coarse grid + quick local refinement around the best
    yaw_grid   = np.linspace(-150, 150, 7)   # [-150, ..., 150]
    pitch_grid = np.linspace(  -5,  55,  7)  # [-5, ..., 55]
    candidates = [(y, p) for y in yaw_grid for p in pitch_grid]

    best = (np.inf, 5.0, 90.0)  # (mse, elev, azim)
    rng = np.random.default_rng(123)

    def score(yaw_deg, pitch_deg):
        PR = _yaw_pitch_rotate_xyz(P3, yaw_deg, pitch_deg)
        # Orthographic projection to screen x,y
        proj2d = PR[:, [0, 1]]  # [x, y]
        _, mse = _procrustes_scale_translate(proj2d, K2, mask=mask)
        return mse

    # coarse
    for y, p in candidates[:]:
        mse = score(y, p)
        if mse < best[0]:
            best = (mse, p, y)

    # local random refinement around the coarse best
    for _ in range(max(0, n_iter-1)):
        y = float(best[2] + rng.uniform(-25, 25))
        p = float(best[1] + rng.uniform(-15, 15))
        mse = score(y, p)
        if mse < best[0]:
            best = (mse, p, y)

    best_elev, best_azim = float(best[1]), float(best[2])
    return best_elev, best_azim
# -----------------------------------------------------------------------------

def pairwise_depth_accuracy(z_a, z_b):
    """Compute pairwise depth ordering accuracy between two depth arrays.
    
    This metric measures how often the front/behind relationships between joints
    are preserved between two depth arrays (e.g., input DAV depths vs GT depths).
    
    Args:
        z_a: (N, 17) array of depth values (e.g., DAV input depths)
        z_b: (N, 17) array of depth values (e.g., GT depths)
    
    Returns:
        (N,) array of per-frame accuracy scores in [0, 1]
        Higher score = better ordinal depth accuracy (less depth ambiguity)
        Lower score = poor depth ordering (likely occluded/ambiguous scene)
    """
    N, J = z_a.shape
    acc = np.zeros(N, dtype=float)
    for i in range(N):
        za, zb = z_a[i], z_b[i]
        # Compute sign of pairwise differences (front/behind relationships)
        sa = np.sign(za[:, None] - za[None, :])   # (17, 17)
        sb = np.sign(zb[:, None] - zb[None, :])   # (17, 17)
        # Only use upper triangle (each joint pair once: i < j)
        mask = np.triu(np.ones((J, J), dtype=bool), 1)
        # Accuracy = fraction of pairs with matching front/behind order
        acc[i] = np.mean((sa[mask] == sb[mask]))
    return acc


def compute_root_motion_speed(coords_before_centering, imgnames, window=10):
    """Compute per-frame root joint motion speed (mm/frame) with smoothing.
    
    This measures how fast the entire body moves through space (translation).
    Must be called BEFORE root-centering to capture global motion.
    
    Args:
        coords_before_centering: (N, 17, 3) poses in original coordinate system (not root-centered)
        imgnames: List of image names (to respect clip boundaries)
        window: Smoothing window length in frames (default: 10)
    
    Returns:
        (N,) array of smoothed root motion speeds in mm/frame
    """
    root_joint = coords_before_centering[:, 0, :]  # (N, 3) - root position
    N = root_joint.shape[0]
    
    # Ensure imgnames length matches coords
    if len(imgnames) != N:
        print(f"  Warning: imgnames length ({len(imgnames)}) != coords length ({N}), truncating")
        imgnames = imgnames[:N]
    
    # Compute per-frame displacement
    displacement = np.zeros(N, dtype=float)
    speeds = np.zeros(N, dtype=float)
    
    # Compute within each clip segment (don't cross boundaries)
    for lo, hi in _segments_from_imgnames(imgnames):
        # Ensure segment boundaries are valid
        lo = max(0, min(lo, N))
        hi = max(0, min(hi, N))
        
        if hi <= lo:
            continue
        
        # Finite differences within segment
        for i in range(lo + 1, hi):
            if i < N and i - 1 >= 0:  # Bounds check
                d = root_joint[i] - root_joint[i - 1]  # (3,)
                displacement[i] = np.linalg.norm(d)  # scalar in mm
        
        # Smooth within segment only
        seg_len = hi - lo
        if window and window > 1 and seg_len > 1:
            k = np.ones(int(window), dtype=float) / float(window)
            seg_displacement = displacement[lo:hi]
            if len(seg_displacement) > 0:
                smoothed = np.convolve(seg_displacement, k, mode='same')
                speeds[lo:hi] = smoothed
        else:
            speeds[lo:hi] = displacement[lo:hi]
    
    return speeds


def compute_ordinal_depth(coords_3d):
    """Compute ordinal depth ranks for 3D coordinates."""
    depths = coords_3d[..., 2]  # Z-coordinates (depth)
    ranks = np.zeros_like(depths, dtype=float)
    for i in range(depths.shape[0]):
        ranks[i] = rankdata(depths[i], method='average')
    return ranks


# --- Geometric Feature Extraction for Tail Error Analysis ---
# 
# These four features characterize *why* augmented inputs (depth, confidence) help
# on certain frames by providing interpretable axes for analyzing tail error patterns:
#
# 1. **Bounding-box scale** - 2D person size (correlates with perspective strength)
#    - Larger scale → stronger perspective distortion (close to camera)
#    - Expectation: XYCD helps more at large scales
#
# 2. **Foreshortening ratio** - 2D projected length / 3D true length for bones
#    - Low ratio (<0.5) → severe foreshortening (limbs toward/away from camera)
#    - Expectation: XYCD helps when foreshortening is severe
#
# 3. **Torso pitch** - Forward/backward lean relative to camera z-axis
#    - Measures self-occlusion due to body inclination
#    - Expectation: Gains at moderate-to-high pitch angles (15-40°)
#
# 4. **Ordinal margin** - Fraction of joint pairs with near-equal depth (|Δz| < 100mm)
#    - High margin → many ambiguous depth pairs → prone to ordering errors
#    - Expectation: XYCD helps when ordinal margin is high (depth ambiguity)

def bbox_scale_2d(k2d):
    """Compute per-frame person scale from 2D keypoints.
    
    Measures the 2D spatial extent of the person in the image.
    Correlates with camera distance and perspective strength.
    
    Args:
        k2d: (N, 17, 2) array of 2D keypoints
    
    Returns:
        (N,) array of scale values (sqrt of bbox area in pixels)
    """
    mins = np.nanmin(k2d, axis=1)  # (N, 2)
    maxs = np.nanmax(k2d, axis=1)  # (N, 2)
    wh = maxs - mins  # (N, 2) - width and height
    # Scale = sqrt(width * height) for stability
    return np.sqrt(np.clip(wh[:, 0] * wh[:, 1], 1e-6, None))


def foreshortening_ratio(k2d, k3d, bone_pairs=None):
    """Compute ratio of 2D to 3D bone lengths (captures projection distortion).
    
    Quantifies how much perspective distortion or depth change exists.
    Low ratio = strong foreshortening (limbs pointed toward/away from camera).
    
    Args:
        k2d: (N, 17, 2) array of 2D keypoints
        k3d: (N, 17, 3) array of 3D keypoints in mm
        bone_pairs: List of (parent, child) joint index pairs (default: major limbs)
    
    Returns:
        (N,) array of mean foreshortening ratios per frame
    """
    if bone_pairs is None:
        # Major limb bones for foreshortening analysis
        bone_pairs = [
            (1, 2), (2, 3),    # right leg
            (4, 5), (5, 6),    # left leg
            (14, 15), (15, 16), # right arm
            (11, 12), (12, 13), # left arm
            (0, 7), (7, 8),    # torso
        ]
    
    # Extract bone vectors
    parent_indices = [a for a, b in bone_pairs]
    child_indices = [b for a, b in bone_pairs]
    
    # 2D bone vectors and lengths
    v2 = k2d[:, child_indices, :] - k2d[:, parent_indices, :]  # (N, n_bones, 2)
    l2 = np.linalg.norm(v2, axis=-1)  # (N, n_bones)
    
    # 3D bone vectors and lengths
    v3 = k3d[:, child_indices, :] - k3d[:, parent_indices, :]  # (N, n_bones, 3)
    l3 = np.linalg.norm(v3, axis=-1)  # (N, n_bones)
    
    # Ratio: 2D projected length / true 3D length
    # Values <1 indicate foreshortening
    ratio = l2 / (l3 + 1e-6)
    
    # Return mean ratio across bones per frame
    return np.nanmean(ratio, axis=1)


def torso_pitch_deg(k3d, left_sh=11, right_sh=14, left_hip=4, right_hip=1):
    """Estimate torso inclination relative to camera z-axis (degrees).
    
    Measures forward/backward lean of the torso.
    Captures viewpoint-related self-occlusion patterns.
    
    Args:
        k3d: (N, 17, 3) array of 3D keypoints in mm
        left_sh, right_sh, left_hip, right_hip: Joint indices
    
    Returns:
        (N,) array of pitch angles in degrees
    """
    # Compute torso midpoints
    shoulders = 0.5 * (k3d[:, left_sh, :] + k3d[:, right_sh, :])  # (N, 3)
    hips = 0.5 * (k3d[:, left_hip, :] + k3d[:, right_hip, :])      # (N, 3)
    
    # Torso vector (shoulder to hip)
    v = shoulders - hips  # (N, 3)
    
    # Projection onto XY plane (perpendicular to camera)
    vxy = np.linalg.norm(v[:, :2], axis=1)  # (N,)
    
    # Depth component (along camera z-axis)
    vz = np.abs(v[:, 2]) + 1e-6  # (N,)
    
    # Pitch angle: arctan(vz / vxy)
    # 0° = upright, 90° = lying down toward/away from camera
    return np.degrees(np.arctan2(vz, vxy))


def ordinal_margin(k3d, thresh_mm=100):
    """Compute fraction of joint pairs with near-equal depth (depth ambiguity).
    
    Indicates how many depth orderings are ambiguous (near-ties).
    High margin fraction = many ambiguous depth pairs = more prone to ordering errors.
    
    Args:
        k3d: (N, 17, 3) array of 3D keypoints in mm
        thresh_mm: Depth threshold for "near-equal" (default: 100mm)
    
    Returns:
        (N,) array of ordinal margin fractions [0, 1]
    """
    N, J, _ = k3d.shape
    z = k3d[:, :, 2]  # (N, 17) - depth coordinates
    
    # Count close pairs per frame
    margin_fractions = np.zeros(N, dtype=np.float32)
    
    for frame_idx in range(N):
        close_pairs = 0
        total_pairs = 0
        
        # Check all joint pairs
        for i in range(J):
            for j in range(i + 1, J):
                dz = np.abs(z[frame_idx, i] - z[frame_idx, j])
                if dz < thresh_mm:
                    close_pairs += 1
                total_pairs += 1
        
        margin_fractions[frame_idx] = close_pairs / total_pairs if total_pairs > 0 else 0.0
    
    return margin_fractions


def compute_joint_visibility_2dprox(k2d, k3d, joint_names=None, eps_mm=10):
    """
    Estimate per-joint occlusion using 3D depth ordering and 2D overlap,
    with adaptive radii per body region that scale with person size.

    A joint j is marked occluded if another joint i lies in front (smaller Z)
    and projects within i's radius on the 2D plane.

    Args:
        k2d: (N, J, 2) array of 2D GT keypoints (pixels)
        k3d: (N, J, 3) array of 3D GT keypoints (mm)
        joint_names: optional list of length J giving joint labels
        eps_mm: minimum depth gap (mm) to consider i in front of j
    Returns:
        vis: (N, J) uint8, 1=visible / 0=occluded
    """
    N, J, _ = k2d.shape
    vis = np.ones((N, J), dtype=np.uint8)

    # --- Anatomically-grounded base radii (for ~300px-tall person) ---
    # These scale dynamically with person's 2D bbox size
    radius_base = {
        'root': 28, 'spine': 26, 'thorax': 25,           # Torso core (thick cylinder)
        'neck_base': 20, 'head': 22,                      # Head/neck (wide silhouette)
        'left_shoulder': 18, 'right_shoulder': 18,        # Upper limb proximal
        'left_elbow': 15, 'right_elbow': 15,              # Upper limb mid
        'left_wrist': 12, 'right_wrist': 12,              # Upper limb distal
        'left_hip': 22, 'right_hip': 22,                  # Lower limb proximal (thicker than arms)
        'left_knee': 20, 'right_knee': 20,                # Lower limb mid
        'left_foot': 15, 'right_foot': 15,                # Lower limb distal
    }
    default_radius_base = 12

    for f in range(N):
        uvs = k2d[f]
        zs = k3d[f, :, 2]
        
        # Compute person scale from 2D bbox
        x_min, y_min = np.nanmin(uvs, axis=0)
        x_max, y_max = np.nanmax(uvs, axis=0)
        bbox_scale = np.sqrt(max((x_max - x_min) * (y_max - y_min), 1.0))
        scale_factor = bbox_scale / 300.0  # Normalize to 300px reference person
        
        for j in range(J):
            uj, vj = uvs[j]
            zj = zs[j]

            # Get base radius and scale by person size
            base_r = radius_base.get(
                joint_names[j], default_radius_base) if joint_names else default_radius_base
            radius_px = base_r * scale_factor

            for i in range(J):
                if i == j:
                    continue
                if zs[i] + eps_mm < zj:  # i is closer
                    du = uvs[i, 0] - uj
                    dv = uvs[i, 1] - vj
                    if (du * du + dv * dv) <= (radius_px * radius_px):
                        vis[f, j] = 0
                        break
    return vis


def summarize_self_occlusion(vis_flags):
    """Summarize per-frame and per-joint occlusion rates."""
    N, J = vis_flags.shape
    frame_scores = 1.0 - vis_flags.mean(axis=1)  # Higher score = more occluded
    joint_rates = 1.0 - vis_flags.mean(axis=0)
    return frame_scores, joint_rates


def ordinal_accuracy_with_slack(pred_ranks, gt_ranks, slack=0, bidirectional=False):
    """Calculate ordinal depth accuracy with optional slack tolerance."""
    if bidirectional:
        correct = np.abs(pred_ranks - gt_ranks) <= slack
    else:
        correct = pred_ranks == gt_ranks
    return np.mean(correct)


def compute_torso_orientation_error(pred_coords, gt_coords, mask):
    """Compute torso orientation error between predicted and GT poses.
    
    Torso orientation is defined by the normal vector to the plane formed by:
    - Hip vector: from right_hip to left_hip
    - Shoulder vector: from right_shoulder to left_shoulder
    
    The normal vector is computed via cross product: hip_vec × shoulder_vec
    Then we compute the angular difference between predicted and GT normal vectors.
    
    Args:
        pred_coords: (N, 17, 3) predicted 3D poses in mm
        gt_coords: (N, 17, 3) GT 3D poses in mm
        mask: (N, 17) boolean mask for valid joints
    
    Returns:
        dict with:
            - 'orientation_errors': (N,) array of orientation errors (degrees)
            - 'valid_mask': (N,) boolean mask for valid torso orientations
            - 'pred_normals': (N, 3) predicted torso normal vectors
            - 'gt_normals': (N, 3) GT torso normal vectors
    """
    # Joint indices for torso computation
    right_hip_idx = 1
    left_hip_idx = 4
    right_shoulder_idx = 14
    left_shoulder_idx = 11
    
    N = pred_coords.shape[0]
    
    # Check if all 4 torso joints are valid
    torso_joint_indices = [right_hip_idx, left_hip_idx, right_shoulder_idx, left_shoulder_idx]
    valid_mask = np.all(mask[:, torso_joint_indices], axis=1)  # (N,)
    
    # Initialize outputs
    orientation_errors = np.full(N, np.nan, dtype=float)
    pred_normals = np.zeros((N, 3), dtype=float)
    gt_normals = np.zeros((N, 3), dtype=float)
    
    # Compute hip and shoulder vectors
    # Hip vector: right_hip -> left_hip
    pred_hip_vec = pred_coords[:, left_hip_idx, :] - pred_coords[:, right_hip_idx, :]  # (N, 3)
    gt_hip_vec = gt_coords[:, left_hip_idx, :] - gt_coords[:, right_hip_idx, :]        # (N, 3)
    
    # Shoulder vector: right_shoulder -> left_shoulder
    pred_shoulder_vec = pred_coords[:, left_shoulder_idx, :] - pred_coords[:, right_shoulder_idx, :]  # (N, 3)
    gt_shoulder_vec = gt_coords[:, left_shoulder_idx, :] - gt_coords[:, right_shoulder_idx, :]        # (N, 3)
    
    # Compute normal vectors via cross product: hip_vec × shoulder_vec
    pred_normals = np.cross(pred_hip_vec, pred_shoulder_vec)  # (N, 3)
    gt_normals = np.cross(gt_hip_vec, gt_shoulder_vec)        # (N, 3)
    
    # Normalize the normal vectors
    pred_norm_magnitude = np.linalg.norm(pred_normals, axis=1, keepdims=True)  # (N, 1)
    gt_norm_magnitude = np.linalg.norm(gt_normals, axis=1, keepdims=True)      # (N, 1)
    
    # Avoid division by zero
    valid_pred = pred_norm_magnitude.squeeze() > 1e-6
    valid_gt = gt_norm_magnitude.squeeze() > 1e-6
    valid_mask = valid_mask & valid_pred & valid_gt
    
    # Normalize where valid
    pred_normals[valid_mask] = pred_normals[valid_mask] / pred_norm_magnitude[valid_mask]
    gt_normals[valid_mask] = gt_normals[valid_mask] / gt_norm_magnitude[valid_mask]
    
    # Compute angular difference using dot product
    # angle = arccos(dot(n1, n2))
    dot_product = np.sum(pred_normals * gt_normals, axis=1)  # (N,)
    
    # Clamp to [-1, 1] to avoid numerical issues with arccos
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    # Compute angle in radians, then convert to degrees
    angle_rad = np.arccos(np.abs(dot_product))  # Use abs to get smallest angle (ignore flip)
    orientation_errors[valid_mask] = np.degrees(angle_rad[valid_mask])
    
    return {
        'orientation_errors': orientation_errors,
        'valid_mask': valid_mask,
        'pred_normals': pred_normals,
        'gt_normals': gt_normals
    }


def compute_angle_errors(pred_coords, gt_coords, mask):
    """Compute angle errors between predicted and GT joint connections.
    
    For each bone (connection between two joints), compute:
    - X-angle error: angle difference in XZ plane (rotation about Y-axis)
    - Y-angle error: angle difference in YZ plane (rotation about X-axis)
    
    Args:
        pred_coords: (N, 17, 3) predicted 3D poses in mm
        gt_coords: (N, 17, 3) GT 3D poses in mm
        mask: (N, 17) boolean mask for valid joints
    
    Returns:
        dict with:
            - 'x_angle_errors': (N, 16) array of X-angle errors per bone (degrees)
            - 'y_angle_errors': (N, 16) array of Y-angle errors per bone (degrees)
            - 'bone_mask': (N, 16) boolean mask for valid bones (both endpoints visible)
            - 'bone_names': list of bone names
    """
    # Define skeleton connections (parent, child)
    skeleton_links = [
        (0, 4),   # root -> left_hip
        (4, 5),   # left_hip -> left_knee
        (5, 6),   # left_knee -> left_foot
        (0, 1),   # root -> right_hip
        (1, 2),   # right_hip -> right_knee
        (2, 3),   # right_knee -> right_foot
        (0, 7),   # root -> spine
        (7, 8),   # spine -> thorax
        (8, 9),   # thorax -> neck_base
        (9, 10),  # neck_base -> head
        (8, 11),  # thorax -> left_shoulder
        (11, 12), # left_shoulder -> left_elbow
        (12, 13), # left_elbow -> left_wrist
        (8, 14),  # thorax -> right_shoulder
        (14, 15), # right_shoulder -> right_elbow
        (15, 16), # right_elbow -> right_wrist
    ]
    
    bone_names = [
        'root-left_hip', 'left_hip-left_knee', 'left_knee-left_foot',
        'root-right_hip', 'right_hip-right_knee', 'right_knee-right_foot',
        'root-spine', 'spine-thorax', 'thorax-neck_base', 'neck_base-head',
        'thorax-left_shoulder', 'left_shoulder-left_elbow', 'left_elbow-left_wrist',
        'thorax-right_shoulder', 'right_shoulder-right_elbow', 'right_elbow-right_wrist'
    ]
    
    N = pred_coords.shape[0]
    n_bones = len(skeleton_links)
    
    x_angle_errors = np.zeros((N, n_bones), dtype=float)
    y_angle_errors = np.zeros((N, n_bones), dtype=float)
    bone_mask = np.zeros((N, n_bones), dtype=bool)
    
    for bone_idx, (parent_idx, child_idx) in enumerate(skeleton_links):
        # Check if both endpoints are valid
        bone_valid = mask[:, parent_idx] & mask[:, child_idx]
        bone_mask[:, bone_idx] = bone_valid
        
        # Compute bone vectors
        pred_bone = pred_coords[:, child_idx, :] - pred_coords[:, parent_idx, :]  # (N, 3)
        gt_bone = gt_coords[:, child_idx, :] - gt_coords[:, parent_idx, :]        # (N, 3)
        
        # X-angle: angle in XZ plane (rotation about Y-axis)
        # Use atan2(x, z) for both pred and GT
        pred_x_angle = np.arctan2(pred_bone[:, 0], pred_bone[:, 2])  # (N,)
        gt_x_angle = np.arctan2(gt_bone[:, 0], gt_bone[:, 2])        # (N,)
        
        # Y-angle: angle in YZ plane (rotation about X-axis)
        # Use atan2(y, z) for both pred and GT
        pred_y_angle = np.arctan2(pred_bone[:, 1], pred_bone[:, 2])  # (N,)
        gt_y_angle = np.arctan2(gt_bone[:, 1], gt_bone[:, 2])        # (N,)
        
        # Compute angular errors (handle wrapping: -pi to pi)
        x_angle_diff = pred_x_angle - gt_x_angle
        x_angle_diff = np.arctan2(np.sin(x_angle_diff), np.cos(x_angle_diff))  # Wrap to [-pi, pi]
        
        y_angle_diff = pred_y_angle - gt_y_angle
        y_angle_diff = np.arctan2(np.sin(y_angle_diff), np.cos(y_angle_diff))  # Wrap to [-pi, pi]
        
        # Convert to degrees and take absolute value
        x_angle_errors[:, bone_idx] = np.abs(np.degrees(x_angle_diff))
        y_angle_errors[:, bone_idx] = np.abs(np.degrees(y_angle_diff))
        
        # Set invalid bones to NaN
        x_angle_errors[~bone_valid, bone_idx] = np.nan
        y_angle_errors[~bone_valid, bone_idx] = np.nan
    
    return {
        'x_angle_errors': x_angle_errors,
        'y_angle_errors': y_angle_errors,
        'bone_mask': bone_mask,
        'bone_names': bone_names
    }


def compute_pck3d(pred_coords, gt_coords, mask, threshold_mm=150):
    """Compute PCK3D metric: percentage of keypoints within threshold distance."""
    errors = np.linalg.norm(pred_coords - gt_coords, axis=-1)
    within_threshold = (errors < threshold_mm) & mask
    return np.mean(within_threshold)


def compute_torso_size(coords_3d, left_hip_idx=4, right_hip_idx=1, thorax_idx=8):
    """Compute torso size for each frame."""
    pelvis = (coords_3d[:, left_hip_idx, :] + coords_3d[:, right_hip_idx, :]) / 2.0
    thorax = coords_3d[:, thorax_idx, :]
    return np.linalg.norm(thorax - pelvis, axis=-1)


def compute_pckt(pred_coords, gt_coords, mask, threshold_ratio=0.2):
    """Compute PCKt metric: scale-invariant accuracy relative to torso size."""
    errors = np.linalg.norm(pred_coords - gt_coords, axis=-1)
    torso_sizes = compute_torso_size(gt_coords)
    thresholds = torso_sizes[:, np.newaxis] * threshold_ratio
    within_threshold = (errors < thresholds) & mask
    return np.mean(within_threshold)


def load_poseformer_predictions(dataset, model_variant, seqlen=27):
    """Load PoseFormer prediction results and per-prediction image basenames."""
    import os
    path_template = f"/srv/essa-lab/flash3/nwarner30/pose_estimation/test_results/june_25_poseformer_poseformer_testds_{{variant}}_sl{seqlen}_{{dataset}}_results.npz"

    # Normalize dataset name
    if '3dhp' in dataset.lower():
        dataset_key = '3dhp'
    elif 'h36m' in dataset.lower():
        dataset_key = 'h36m'
    elif '3dpw' in dataset.lower():
        dataset_key = '3dpw'
    elif 'fit3d' in dataset.lower():
        dataset_key = 'fit3d'
    else:
        raise ValueError(f"Unrecognized dataset: {dataset}")

    # Normalize model variant name
    variant_map = {'xy': '', 'base': '', 'xyc': 'xyc', 'xyd': 'xyd', 'xycd': 'xycd'}
    variant_key = variant_map.get(model_variant.lower(), '')

    # Construct path (note the double underscore for XY baseline)
    if variant_key == '':
        results_path = path_template.replace('testds_{variant}_', 'testds__').format(dataset=dataset_key, variant='')
    else:
        results_path = path_template.format(variant=variant_key, dataset=dataset_key)

    if not results_path or not os.path.exists(results_path):
        print(f"Error: Could not find results for {dataset} - {model_variant}")
        print(f"  Expected path: {results_path}")
        return None

    print(f"Loading predictions from: {results_path}")
    pred_results = np.load(results_path, allow_pickle=True)
    pred_data = pred_results['data']

    pred_list = []
    name_list = []

    for rec in pred_data:
        # PoseFormer outputs were saved per-sample with a single target frame.
        # Keypoints may be [1,17,3] or [17,3]. We take the single target frame.
        kp = rec['pred_instances']['keypoints']
        kp = np.asarray(kp)
        if kp.ndim == 4:
            kp = np.squeeze(kp, axis=0)     # [1,17,3] -> [1,17,3]
        if kp.ndim == 3 and kp.shape[0] > 1:
            kp = kp[-1]                      # keep target frame only
        elif kp.ndim == 3 and kp.shape[0] == 1:
            kp = kp[0]                       # [1,17,3] -> [17,3]
        # else: already [17,3]

        pred_list.append(kp)

        # Extract the target image basename
        tgt = rec.get('target_img_path', None)
        if isinstance(tgt, (list, tuple)) and len(tgt) > 0:
            tgt = tgt[0]
        name_list.append(os.path.basename(str(tgt)) if tgt is not None else None)

    pred_coords = np.stack(pred_list, axis=0)           # (Npred,17,3)
    pred_imgnames = np.array(name_list, dtype=object)   # (Npred,)

    print(f"  Loaded {len(pred_coords)} predictions with shape {pred_coords.shape}")
    if pred_imgnames[0] is not None:
        print(f"  Example pred basename: {pred_imgnames[0]}")

    return pred_coords, pred_imgnames


def load_ground_truth(dataset, use_test_valid_3dhp=False):
    """Load ground truth 3D poses.
    
    Args:
        dataset: Dataset name ('3dhp', 'h36m', '3dpw', 'fit3d')
        use_test_valid_3dhp: If True and dataset is 3DHP, filter to test_valid subset (2,875 frames)
    """
    path_map = {
        '3dhp': "/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_3dhp_test_all_v4_hr.npz",
        'h36m': "/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_h36m_val_v4_highres_dets.npz",
        '3dpw': "/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_3dpw_all_v5_det_dav_hr.npz",
        'fit3d': "/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_fit3d_all_v3.npz",
    }
    
    if '3dhp' in dataset.lower():
        dataset_key = '3dhp'
    elif 'h36m' in dataset.lower():
        dataset_key = 'h36m'
    elif '3dpw' in dataset.lower():
        dataset_key = '3dpw'
    elif 'fit3d' in dataset.lower():
        dataset_key = 'fit3d'
    else:
        raise ValueError(f"Unrecognized dataset name: {dataset}")
    input_data_path = path_map[dataset_key]
    
    # Load test_valid subset for 3DHP if requested
    test_valid_imgnames_set = None
    if use_test_valid_3dhp and dataset_key == '3dhp':
        test_valid_path = '/srv/essa-lab/flash3/nwarner30/pose_estimation/data/annotations/mpi_inf_3dhp_test_valid.npz'
        print(f"Loading 3DHP test_valid subset filter from: {test_valid_path}")
        test_valid_data = np.load(test_valid_path, allow_pickle=True)
        test_valid_imgnames = test_valid_data['imgname']
        test_valid_imgnames_set = set(test_valid_imgnames)
        print(f"  test_valid contains {len(test_valid_imgnames)} frames (11.6% of test_all)")
    
    print(f"Loading ground truth from: {input_data_path}")
    input_data = np.load(input_data_path, allow_pickle=True)
    all_gt_3d_poses = input_data['S'][..., :3]  # (N, 17, 3)
    all_imgnames = input_data['imgname']
    
    # Load 2D keypoints and visibility scores if available
    all_input_2d_keypoints = input_data.get('predicted_keypoints', None)  # (N, 17, 2)
    all_visibility_scores = input_data.get('predicted_keypoints_score', None)  # (N, 17)
    
    # Load DAV input depths if available (for occlusion analysis)
    all_dav_depths = None
    possible_depth_keys = ['predicted_da_depth', 'predicted_depths_dav', 'dav_depth', 'input_depths', 'dav_depths']
    
    for key in possible_depth_keys:
        if key in input_data:
            all_dav_depths = input_data[key]  # (N, 17) or (N, 17, 1)
            if all_dav_depths.ndim == 3:
                all_dav_depths = all_dav_depths[..., 0]  # Flatten to (N, 17)
            print(f"  Loaded DAV input depths from key '{key}', shape: {all_dav_depths.shape}")
            break
    
    # Filter to test_valid subset if requested for 3DHP
    if test_valid_imgnames_set is not None:
        print(f"\n  Filtering to test_valid subset...")
        valid_indices = []
        for i, imgname in enumerate(all_imgnames):
            if imgname in test_valid_imgnames_set:
                valid_indices.append(i)
        
        valid_indices = np.array(valid_indices, dtype=int)
        print(f"  Matched {len(valid_indices)} / {len(test_valid_imgnames_set)} frames from test_valid")
        
        # Extract filtered data
        gt_3d_poses = all_gt_3d_poses[valid_indices]
        imgnames = all_imgnames[valid_indices]
        input_2d_keypoints = all_input_2d_keypoints[valid_indices] if all_input_2d_keypoints is not None else None
        visibility_scores = all_visibility_scores[valid_indices] if all_visibility_scores is not None else None
        dav_depths = all_dav_depths[valid_indices] if all_dav_depths is not None else None
    else:
        # Use full dataset
        gt_3d_poses = all_gt_3d_poses
        imgnames = all_imgnames
        input_2d_keypoints = all_input_2d_keypoints
        visibility_scores = all_visibility_scores
        dav_depths = all_dav_depths
    
    # Print loaded data summary
    if input_2d_keypoints is not None:
        if visibility_scores is not None:
            print(f"  Loaded {len(gt_3d_poses)} GT frames with 2D keypoints and visibility scores")
        else:
            print(f"  Loaded {len(gt_3d_poses)} GT frames with 2D keypoints (no visibility scores)")
    else:
        print(f"  Loaded {len(gt_3d_poses)} GT frames (no 2D keypoints available)")
    
    if dav_depths is None:
        print(f"  No DAV input depths found (tried keys: {possible_depth_keys})")
    
    # Return the test_valid_imgnames_set if it was used, otherwise None
    return gt_3d_poses, imgnames, input_2d_keypoints, visibility_scores, dav_depths, test_valid_imgnames_set


def align_fit3d_predictions_with_gt(pred_imgnames, gt_imgnames):
    """
    Align FIT3D predictions with GT using image filenames embedded in the .npz.

    FIT3D filenames follow pattern:
        s{subject}_{video_id}_{exercise}.mp4_{frame:06d}.jpg
    Example:
        s11_50591643_walk_the_box.mp4_002486.jpg

    We align by sorting both sets and mapping prediction frames to GT order
    within each (subject, video_id, exercise) sequence.
    """
    from collections import defaultdict
    import os

    print(f"\n  Aligning FIT3D predictions to GT via filename matching...")

    # --- Parse both GT and prediction filenames ---
    def parse_key(imgname):
        base = os.path.basename(str(imgname))
        parts = base.split('_')
        if len(parts) < 3:
            return base, None
        subject = parts[0]
        video_id = parts[1]
        rest = '_'.join(parts[2:])
        exercise = rest.split('.mp4_')[0] if '.mp4_' in rest else rest.split('.')[0]
        return f"{subject}_{video_id}_{exercise}", base

    # Build lookup from GT: key → list of frame names and indices
    gt_lookup = defaultdict(list)
    for idx, imgname in enumerate(gt_imgnames):
        key, base = parse_key(imgname)
        gt_lookup[key].append((base, idx))

    # Sort GT frames within each sequence
    for k in gt_lookup:
        gt_lookup[k].sort(key=lambda x: x[0])

    # Build aligned prediction index order
    aligned_indices = []
    unmatched = 0
    for pred_name in pred_imgnames:
        key, base = parse_key(pred_name)
        if key not in gt_lookup:
            unmatched += 1
            continue
        # Find closest matching frame (string match)
        for frame_name, gt_idx in gt_lookup[key]:
            if frame_name == base:
                aligned_indices.append(gt_idx)
                break
        else:
            unmatched += 1

    print(f"    Matched {len(aligned_indices)} / {len(pred_imgnames)} frames")
    if unmatched > 0:
        print(f"    Warning: {unmatched} prediction frames had no GT match")

    return np.array(aligned_indices, dtype=int)


def align_3dpw_predictions_with_gt(pred_coords, gt_imgnames):
    """Align 3DPW predictions with GT by EXACTLY matching the dataset loader's logic.
    
    This function reconstructs the EXACT sequence ordering from _3dpw_dataset.py:
    1. Parse imgnames using _parse_h36m_imgname() logic
    2. Group by (subj, action, camera) where subj=participant, action=action, camera='0'
    3. Sort groups by key (matching dataset loader's sorted(video_frames.items()))
    4. For each frame in each group (in order), create one prediction
    """
    from collections import defaultdict
    
    print(f"\n  Reconstructing 3DPW prediction ordering (matching dataset loader)...")
    
    # Step 1: Parse imgnames EXACTLY like _3dpw_dataset.py does
    # Format: {location}_{action}_{action_num}_{participant}_frame{XXXXX}.jpg
    # Example: downtown_windowShopping_00_participant0_frame01503.jpg
    
    video_frames_old = defaultdict(list)
    for idx, imgname in enumerate(gt_imgnames):
        # Parse using EXACT logic from _parse_h36m_imgname()
        parts = imgname.split('_')
        if len(parts) >= 4:
            # location, action, action_num, participant = parts[0], parts[1], parts[2], parts[3]
            
            # CRITICAL: Parse participant ID to integer for correct sorting
            # 'participant10' comes before 'participant2' in string sort.
            # We need to sort numerically.
            try:
                subj_id = int(parts[3].replace('participant', ''))
            except (ValueError, IndexError):
                subj_id = -1 # Fallback for unexpected formats

            subj = parts[3]    # participant (e.g., "participant0")
            action = parts[1]  # action (e.g., "windowShopping")
            camera = '0'       # Always '0' for 3DPW
            
            # Group by (subj_id, action, camera) to ensure correct numeric sorting
            key = (subj_id, action, camera)
            video_frames_old[key].append(idx)
    
    print(f"    Found {len(video_frames_old)} groups")
    
    # Step 2: Iterate through groups in sorted(items()) order
    # CRITICAL: sorted() on dict.items() sorts by KEY (the tuple)
    # Dataset loader: for _, _indices in sorted(video_frames_old.items())
    # This gives us prediction order: all frames from group1, then group2, etc.
    pred_idx_to_gt_idx = []
    for group_key, frame_indices in sorted(video_frames_old.items()):
        # Within each group, frames appear in the order they were appended
        # (which is the order they appear in gt_imgnames)
        for frame_idx in frame_indices:
            pred_idx_to_gt_idx.append(frame_idx)
    
    print(f"    Prediction count: {len(pred_coords)}")
    print(f"    Mapping count: {len(pred_idx_to_gt_idx)}")
    print(f"    GT count: {len(gt_imgnames)}")
    
    # DEBUG: Print first 10 mappings
    print(f"\n    DEBUG: First 10 prediction→GT mappings:")
    for i in range(min(10, len(pred_idx_to_gt_idx))):
        gt_idx = pred_idx_to_gt_idx[i]
        print(f"      pred[{i}] → gt[{gt_idx}] = {gt_imgnames[gt_idx]}")
    
    # Step 3: Align lengths
    if len(pred_coords) != len(pred_idx_to_gt_idx):
        print(f"    WARNING: Prediction count mismatch! Using min length.")
        n = min(len(pred_coords), len(pred_idx_to_gt_idx))
        pred_coords = pred_coords[:n]
        pred_idx_to_gt_idx = pred_idx_to_gt_idx[:n]
    
    # Step 4: Create aligned predictions array
    aligned_preds = np.zeros((len(gt_imgnames), 17, 3), dtype=pred_coords.dtype)
    gt_has_prediction = np.zeros(len(gt_imgnames), dtype=bool)
    
    for pred_idx, gt_idx in enumerate(pred_idx_to_gt_idx):
        if gt_idx < len(gt_imgnames):
            aligned_preds[gt_idx] = pred_coords[pred_idx]
            gt_has_prediction[gt_idx] = True
    
    print(f"    ✓ Aligned {np.sum(gt_has_prediction)}/{len(gt_imgnames)} GT frames with predictions")
    
    return aligned_preds, gt_has_prediction


def compute_metrics(dataset, condition, seqlen=27, confidence_threshold=0.3, use_test_valid_3dhp=False, filter_by_confidence=False):
    """Compute all metrics for a dataset-condition pair.
    
    Args:
        dataset: Dataset name
        condition: Model variant ('xy', 'xyc', 'xyd', 'xycd')
        seqlen: Sequence length
        confidence_threshold: Minimum 2D keypoint confidence
        use_test_valid_3dhp: If True and dataset is 3DHP, filter to test_valid subset
        filter_by_confidence: If True, apply joint-level confidence filtering
    """
    print(f"\n{'='*60}")
    print(f"Computing metrics: {dataset} - {condition} (seqlen={seqlen})")
    print(f"Confidence threshold: {confidence_threshold}")
    if use_test_valid_3dhp and '3dhp' in dataset.lower():
        print(f"Using 3DHP test_valid subset (2,875 frames)")
    print(f"{'='*60}")
    
    # Load data
    result = load_poseformer_predictions(dataset, condition, seqlen=seqlen)
    if result is None:
        return None
    
    # Handle both old (single return) and new (tuple return) formats
    if isinstance(result, tuple):
        pred_coords, pred_imgnames = result
    else:
        pred_coords = result
        pred_imgnames = None
    
    gt_coords, gt_imgnames, input_2d_keypoints, visibility_scores, dav_depths, test_valid_set = load_ground_truth(dataset, use_test_valid_3dhp=use_test_valid_3dhp)
    
    # For 3DHP with test_valid filter: align predictions to match GT subset
    if use_test_valid_3dhp and '3dhp' in dataset.lower() and pred_imgnames is not None:
        print(f"\n  Filtering predictions to test_valid subset...")
        
        # Use the test_valid_set directly (it's the authoritative filter)
        if test_valid_set is not None:
            gt_imgnames_set = test_valid_set
            print(f"  Using test_valid_set with {len(test_valid_set)} frames as filter")
        else:
            # Fallback: use the already-filtered gt_imgnames
            gt_imgnames_set = set(gt_imgnames)
            print(f"  Fallback: using filtered gt_imgnames as filter")
        
        pred_valid_indices = []
        for i, imgname in enumerate(pred_imgnames):
            if imgname in gt_imgnames_set:
                pred_valid_indices.append(i)
        
        pred_valid_indices = np.array(pred_valid_indices, dtype=int)
        print(f"  Matched {len(pred_valid_indices)} / {len(pred_imgnames)} prediction frames")
        
        # Filter predictions
        pred_coords = pred_coords[pred_valid_indices]
        pred_imgnames = pred_imgnames[pred_valid_indices]
    
    # For 3DPW: Align predictions with GT using sequence reconstruction
    if dataset == '3dpw':
        pred_coords, gt_has_prediction = align_3dpw_predictions_with_gt(
            pred_coords, gt_imgnames
        )
        # Filter to only frames with predictions
        valid_mask = gt_has_prediction
        pred_coords = pred_coords[valid_mask]
        gt_coords = gt_coords[valid_mask]
        gt_imgnames = gt_imgnames[valid_mask]
        if input_2d_keypoints is not None:
            input_2d_keypoints = input_2d_keypoints[valid_mask]
        if visibility_scores is not None:
            visibility_scores = visibility_scores[valid_mask]
        if dav_depths is not None:
            dav_depths = dav_depths[valid_mask]
    
    print(f"\n{'='*60}")
    print(f"Alignment Check (assuming shuffle=False in dataloader)")
    print(f"{'='*60}")
    print(f"  Predictions: {len(pred_coords)} frames")
    print(f"  GT: {len(gt_coords)} frames")
    print(f"  2D keypoints available: {input_2d_keypoints is not None}")
    
    # For 3DPW: Print sequence structure to debug alignment
    if dataset == '3dpw':
        print(f"\n  3DPW Sequence Structure Analysis:")
        # Group GT by sequence
        sequence_groups = {}
        for idx, imgname in enumerate(gt_imgnames):
            # Extract sequence ID (location_action_num)
            parts = imgname.split('_')
            if len(parts) >= 3:
                seq_id = '_'.join(parts[:3])  # e.g., "downtown_windowShopping_00"
                if seq_id not in sequence_groups:
                    sequence_groups[seq_id] = []
                sequence_groups[seq_id].append((idx, imgname))
        
        print(f"  Found {len(sequence_groups)} unique sequences in GT")
        print(f"  Sequence lengths:")
        for seq_id, frames in sorted(sequence_groups.items())[:5]:  # Show first 5
            print(f"    {seq_id}: {len(frames)} frames")
        print(f"    ... ({len(sequence_groups) - 5} more sequences)")
    
    # Check if imgnames align (first 10 and last 10) if available
    if pred_imgnames is not None:
        n_check = min(10, len(pred_imgnames), len(gt_imgnames))
        print(f"\n  First {n_check} imgname comparison:")
        matches_first = 0
        for i in range(n_check):
            match = "✓" if pred_imgnames[i] == gt_imgnames[i] else "✗"
            print(f"    [{i}] Pred: {pred_imgnames[i]:50s} | GT: {gt_imgnames[i]:50s} {match}")
            if pred_imgnames[i] == gt_imgnames[i]:
                matches_first += 1
        
        print(f"\n  Last {n_check} imgname comparison:")
        matches_last = 0
        for i in range(-n_check, 0):
            match = "✓" if pred_imgnames[i] == gt_imgnames[i] else "✗"
            print(f"    [{len(pred_imgnames)+i}] Pred: {pred_imgnames[i]:50s} | GT: {gt_imgnames[i]:50s} {match}")
            if pred_imgnames[i] == gt_imgnames[i]:
                matches_last += 1
        
        print(f"\n  Alignment summary:")
        print(f"    First {n_check} frames: {matches_first}/{n_check} match")
        print(f"    Last {n_check} frames: {matches_last}/{n_check} match")
    else:
        print(f"\n  Warning: Prediction imgnames not available, cannot verify alignment")
    
    # --- Name-based alignment for FIT3D ---
    fit3d_aligned = False
    if dataset.lower() == 'fit3d' and pred_imgnames is not None:
        import os
        # Build lookup from GT basename -> GT index
        gt_base = np.array([os.path.basename(x) for x in gt_imgnames], dtype=object)
        idx_map = {name: i for i, name in enumerate(gt_base)}

        matched_pred_indices = []
        matched_gt_indices = []
        missing = 0
        for i, nm in enumerate(pred_imgnames):
            if nm is None:
                missing += 1
                continue
            gi = idx_map.get(nm, None)
            if gi is None:
                missing += 1
                continue
            matched_pred_indices.append(i)
            matched_gt_indices.append(gi)

        if not matched_pred_indices:
            print("  ERROR: No FIT3D imgname matches found; falling back to truncation.")
            fit3d_aligned = False
        else:
            matched_pred_indices = np.array(matched_pred_indices, dtype=int)
            matched_gt_indices = np.array(matched_gt_indices, dtype=int)

            # Reorder to prediction order
            pred_coords = pred_coords[matched_pred_indices]
            gt_coords   = gt_coords[matched_gt_indices]
            imgnames    = gt_imgnames[matched_gt_indices]

            if input_2d_keypoints is not None:
                input_2d_keypoints = input_2d_keypoints[matched_gt_indices]
            if visibility_scores is not None:
                visibility_scores = visibility_scores[matched_gt_indices]
            if dav_depths is not None:
                dav_depths = dav_depths[matched_gt_indices]

            print(f"\n  FIT3D name-match alignment:")
            print(f"    matched: {len(matched_pred_indices)}")
            print(f"    missing: {missing}")
            fit3d_aligned = True
    
    # Fallback: truncate to min length (for non-FIT3D datasets or when name-matching fails)
    if not fit3d_aligned and dataset.lower() != '3dpw':
        n_frames = min(len(pred_coords), len(gt_coords))
        pred_coords = pred_coords[:n_frames]
        gt_coords   = gt_coords[:n_frames]
        imgnames    = gt_imgnames[:n_frames]
        if input_2d_keypoints is not None:
            input_2d_keypoints = input_2d_keypoints[:n_frames]
        if visibility_scores is not None:
            visibility_scores = visibility_scores[:n_frames]
        if dav_depths is not None:
            dav_depths = dav_depths[:n_frames]
        print(f"\n  Using {n_frames} aligned frames (truncated to min length)")
    
    # After aligning by filename (FIT3D) or any other dataset-specific alignment,
    # finalize the working slices and establish n_frames for downstream code.
    assert len(pred_coords) == len(gt_coords), "Pred/GT length mismatch after alignment"
    n_frames = int(len(pred_coords))

    # Ensure imgnames points to the aligned GT names (used by later prints/analyses)
    imgnames = gt_imgnames

    # If you carry optional per-frame arrays, align them too (they were masked above;
    # this is a no-op here but keeps intent clear).
    if input_2d_keypoints is not None:
        assert len(input_2d_keypoints) == n_frames
    if visibility_scores is not None:
        assert len(visibility_scores) == n_frames
    if dav_depths is not None:
        assert len(dav_depths) == n_frames

    print(f"\n  Using {n_frames} frames after alignment")
    
    # Sanity check: show first/last few aligned names
    print(f"  First 3 aligned frames: {imgnames[0]}, {imgnames[1]}, {imgnames[2]}")
    print(f"  Last 3 aligned frames: {imgnames[n_frames-3]}, {imgnames[n_frames-2]}, {imgnames[n_frames-1]}")
    
    print(f"{'='*60}")
    
    # --- Root motion analysis (BEFORE root-centering) ---
    # NOTE: Predictions from PoseFormer are already root-centered (root at origin)
    # So we compute root motion on GT only, and skip predictions
    # Compute root motion speed on GT (how fast the body moves through space)
    gt_root_speed = compute_root_motion_speed(gt_coords, imgnames, window=10)
    
    print(f"\nRoot Motion Speed Statistics (GT only, predictions are pre-centered):")
    print(f"  Mean GT:   {np.mean(gt_root_speed):.2f} mm/frame")
    print(f"  Median GT:   {np.median(gt_root_speed):.2f} mm/frame")
    print(f"  Max GT:   {np.max(gt_root_speed):.2f} mm/frame")
    
    # Preprocessing: Center at root, flip Y so up is +Y (pred & GT)
    pred_centered = pred_coords - pred_coords[:, 0:1, :]
    gt_centered = gt_coords - gt_coords[:, 0:1, :]
    
    pred_centered[:, :, 1] *= -1.0
    gt_centered[:, :, 1] *= -1.0
    
    # --- FIT3D axis auto-alignment (choose axis mapping that minimizes MPJPE on a sample) ---
    if dataset.lower() == 'fit3d':
        def _transform(P, mode):
            if mode == 'identity':
                return P
            if mode == 'swap_xz':         # (x,y,z) -> (z,y,x)
                return np.stack([P[..., 2], P[..., 1], P[..., 0]], axis=-1)
            if mode == 'flip_z':          # (x,y,z) -> (x,y,-z)
                return P * np.array([1.0, 1.0, -1.0])[None, None, :]
            if mode == 'swap_xz_flipz':   # (x,y,z) -> (z,y,-x)
                return np.stack([P[..., 2], P[..., 1], -P[..., 0]], axis=-1)
            if mode == 'flip_x':          # (x,y,z) -> (-x,y,z)
                return P * np.array([-1.0, 1.0, 1.0])[None, None, :]
            if mode == 'flip_xz':         # (x,y,z) -> (-x,y,-z)
                return P * np.array([-1.0, 1.0, -1.0])[None, None, :]
            return P
        
        sample = slice(0, min(5000, pred_centered.shape[0]))
        if visibility_scores is not None:
            joint_mask = (visibility_scores[sample] > 0.3)
        else:
            joint_mask = np.ones(pred_centered[sample, :, 0].shape, dtype=bool)
        
        def _mpjpe_mean(P):
            E = np.linalg.norm(P - gt_centered[sample], axis=-1)
            E[~joint_mask] = np.nan
            return float(np.nanmean(E))
        
        candidates = ['identity', 'swap_xz', 'flip_z', 'swap_xz_flipz', 'flip_x', 'flip_xz']
        scores = {m: _mpjpe_mean(_transform(pred_centered[sample], m)) for m in candidates}
        best = min(scores, key=scores.get)
        pred_centered = _transform(pred_centered, best)
        print(f"  FIT3D axis auto-align: {best} (sample MPJPE {scores[best]*1000.0:.1f} mm)")
    
    # Preprocess DAV depths: root-center only (already in camera frame, no Y-flip needed)
    dav_mm = None
    if dav_depths is not None:
        # DAV depths are already in camera frame, just need root-centering and unit conversion
        dav_root_centered = dav_depths - dav_depths[:, 0:1]  # Root-center: (N, 17)
        dav_mm = dav_root_centered * 1000.0  # Convert to mm, shape (N, 17)
        print(f"  Preprocessed DAV input depths: {dav_mm.shape}")
    
    # 3DPW: rotate predictions -90° about Y (x,y,z)->(-z,y,x)
    if dataset == '3dpw':
        x = pred_centered[:, :, 0].copy()
        z = pred_centered[:, :, 2].copy()
        pred_centered[:, :, 0] = -z
        pred_centered[:, :, 2] = x

        # Apply additional 90° CW: (-z, y, x) -> (x, y, z)
        x = pred_centered[:, :, 0].copy()  # This is -z
        z = pred_centered[:, :, 2].copy()  # This is x
        pred_centered[:, :, 0] = z         # new_x = x
        pred_centered[:, :, 2] = -x        # new_z = z

    # Convert to mm
    pred_mm = pred_centered * 1000.0
    gt_mm = gt_centered * 1000.0
    
    # Create confidence mask for joint-level filtering
    joint_mask = np.ones((n_frames, 17), dtype=bool)
    n_filtered_joints = 0
    
    if filter_by_confidence and confidence_threshold > 0.0 and visibility_scores is not None:
        joint_mask = visibility_scores > confidence_threshold
        n_filtered_joints = np.sum(~joint_mask)
        n_total_joints = joint_mask.size
        
        print(f"\n  Joint-level confidence filtering (threshold={confidence_threshold}):")
        print(f"    Filtered {n_filtered_joints}/{n_total_joints} joints ({n_filtered_joints/n_total_joints*100:.2f}%)")
        print(f"    Remaining joints: {np.sum(joint_mask)}")
        
        # Check frames with all joints filtered
        frames_with_no_joints = np.sum(joint_mask, axis=1) == 0
        n_empty_frames = np.sum(frames_with_no_joints)
        if n_empty_frames > 0:
            print(f"    WARNING: {n_empty_frames} frames have NO valid joints after filtering")
    elif confidence_threshold > 0.0 and visibility_scores is not None:
        print(f"\n  Joint-level confidence filtering: DISABLED (use --filter_by_confidence to enable)")
        print(f"    Confidence threshold set to {confidence_threshold}, but filtering is off")
    elif confidence_threshold > 0.0:
        print(f"\n  WARNING: Confidence threshold={confidence_threshold} set but no visibility scores available")
        print(f"    Proceeding without joint-level filtering")
    
    # Compute MPJPE with joint-level masking
    per_joint_errors = np.linalg.norm(pred_mm - gt_mm, axis=-1)  # (N, 17)
    
    # Apply mask: set filtered joints to NaN
    per_joint_errors_masked = per_joint_errors.copy()
    per_joint_errors_masked[~joint_mask] = np.nan
    
    # Compute per-frame MPJPE (mean over valid joints only)
    # Use warnings.catch_warnings to suppress RuntimeWarning for empty slices
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
        per_frame_mpjpe = np.nanmean(per_joint_errors_masked, axis=1)  # (N,)
    
    # ALSO compute global MPJPE (matching mmpose keypoint_mpjpe implementation)
    # This flattens all valid joints and computes mean globally (not per-frame)
    global_mpjpe = np.mean(per_joint_errors[joint_mask])  # Mean over all valid (frame, joint) pairs
    
    # Model pairwise ordinal accuracy (pred Z vs GT Z)
    pred_depths = pred_mm[..., 2]  # (N, 17)
    gt_depths = gt_mm[..., 2]  # (N, 17)
    pairwise_ordinal_per_frame = pairwise_depth_accuracy(pred_depths, gt_depths)
    pairwise_ordinal = np.mean(pairwise_ordinal_per_frame)
    
    # MPJPE statistics
    results = {
        'dataset': dataset,
        'condition': condition,
        'n_frames': n_frames,
        'mean_mpjpe': np.nanmean(per_frame_mpjpe),  # Per-frame average (for analysis)
        'global_mpjpe': global_mpjpe,  # Global average (matches mmpose evaluator)
        'median_mpjpe': np.nanmedian(per_frame_mpjpe),
        'p25': np.nanpercentile(per_frame_mpjpe, 25),
        'p50': np.nanpercentile(per_frame_mpjpe, 50),
        'p75': np.nanpercentile(per_frame_mpjpe, 75),
        'p90': np.nanpercentile(per_frame_mpjpe, 90),
        'p95': np.nanpercentile(per_frame_mpjpe, 95),
        'p99': np.nanpercentile(per_frame_mpjpe, 99),
        'per_frame_mpjpe': per_frame_mpjpe,
        'pred_mm': pred_mm,
        'gt_mm': gt_mm,
        'imgnames': imgnames,
        'input_2d_keypoints': input_2d_keypoints,  # Add 2D keypoints to results
        'visibility_scores': visibility_scores,  # Add visibility scores to results
        'joint_mask': joint_mask,  # Add joint mask to results
        'confidence_threshold': confidence_threshold,  # Record threshold used
        'n_filtered_joints': n_filtered_joints if confidence_threshold > 0 else 0,  # Record filtering stats
        'root_motion_speed': {
            'pred_speed': np.zeros_like(gt_root_speed),  # Placeholder: predictions are pre-centered
            'gt_speed': gt_root_speed,
            'mean_pred': 0.0,  # Placeholder: predictions are pre-centered
            'mean_gt': np.mean(gt_root_speed),
        },
    }
    
    print(f"\nMPJPE Metrics:")
    print(f"  Global MPJPE (mmpose-style): {results['global_mpjpe']:.2f} mm")
    print(f"  Mean (per-frame avg): {results['mean_mpjpe']:.2f} mm")
    print(f"  Median: {results['median_mpjpe']:.2f} mm")
    print(f"  p75: {results['p75']:.2f} mm")
    print(f"  p90: {results['p90']:.2f} mm")
    print(f"  p95: {results['p95']:.2f} mm")
    
    # Compute qualitative metrics (use joint_mask for confidence filtering)
    pck3d_100 = compute_pck3d(pred_mm, gt_mm, joint_mask, threshold_mm=100)
    pck3d_150 = compute_pck3d(pred_mm, gt_mm, joint_mask, threshold_mm=150)
    
    pckt_050 = compute_pckt(pred_mm, gt_mm, joint_mask, threshold_ratio=0.5)
    pckt_100 = compute_pckt(pred_mm, gt_mm, joint_mask, threshold_ratio=1.0)
    
    pred_ranks = compute_ordinal_depth(pred_mm)
    gt_ranks = compute_ordinal_depth(gt_mm)
    
    # Apply joint mask to ordinal depth calculations
    pred_ranks_masked = pred_ranks.copy()
    gt_ranks_masked = gt_ranks.copy()
    pred_ranks_masked[~joint_mask] = np.nan
    gt_ranks_masked[~joint_mask] = np.nan
    
    # Compute ordinal accuracy only on valid joints
    valid_rank_pairs = joint_mask  # Only consider masked joints
    if np.sum(valid_rank_pairs) > 0:
        ordinal_exact = np.mean((pred_ranks[valid_rank_pairs] == gt_ranks[valid_rank_pairs]))
        ordinal_pm1 = np.mean(np.abs(pred_ranks[valid_rank_pairs] - gt_ranks[valid_rank_pairs]) <= 1)
    else:
        ordinal_exact = 0.0
        ordinal_pm1 = 0.0
    
    # Coarse ordinal depth accuracy with different thresholds
    # Threshold = max depth difference to be considered "same depth bucket"
    # 100mm = 0.1m, 250mm = 0.25m
    pred_coarse_100 = compute_coarse_ordinal_depth(pred_mm, threshold_mm=100)
    gt_coarse_100 = compute_coarse_ordinal_depth(gt_mm, threshold_mm=100)
    if np.sum(joint_mask) > 0:
        coarse_ordinal_100 = np.mean((pred_coarse_100[joint_mask] == gt_coarse_100[joint_mask]))
    else:
        coarse_ordinal_100 = 0.0

    pred_coarse_250 = compute_coarse_ordinal_depth(pred_mm, threshold_mm=250)
    gt_coarse_250 = compute_coarse_ordinal_depth(gt_mm, threshold_mm=250)
    if np.sum(joint_mask) > 0:
        coarse_ordinal_250 = np.mean((pred_coarse_250[joint_mask] == gt_coarse_250[joint_mask]))
    else:
        coarse_ordinal_250 = 0.0
    
    # Compute input depth ordinal accuracy (DAV vs GT)
    input_depth_ordinal_accuracy = None
    input_depth_ordinal_accuracy_per_frame = None
    if dav_mm is not None:
        gt_depths = gt_mm[..., 2]  # (N, 17) - GT Z coords in mm
        input_depth_ordinal_accuracy_per_frame = pairwise_depth_accuracy(dav_mm, gt_depths)  # (N,)
        input_depth_ordinal_accuracy = np.mean(input_depth_ordinal_accuracy_per_frame)
        
        print(f"\n  Input Depth Ordinal Accuracy (DAV vs GT): {input_depth_ordinal_accuracy:.4f}")
        print(f"    Mean accuracy: {input_depth_ordinal_accuracy:.4f}")
        print(f"    Median accuracy: {np.median(input_depth_ordinal_accuracy_per_frame):.4f}")
        print(f"    Min accuracy: {np.min(input_depth_ordinal_accuracy_per_frame):.4f}")
        print(f"    Max accuracy: {np.max(input_depth_ordinal_accuracy_per_frame):.4f}")
    
    results['qual_metrics'] = {
        'PCK3D-100mm': pck3d_100,
        'PCK3D-150mm': pck3d_150,
        'PCKt@0.5': pckt_050,
        'PCKt@1.0': pckt_100,
        'ordinal_depth_accuracy_exact': ordinal_exact,
        'ordinal_depth_accuracy_±1': ordinal_pm1,
        'coarse_ordinal_accuracy_100mm': coarse_ordinal_100,
        'coarse_ordinal_accuracy_250mm': coarse_ordinal_250,
        'pairwise_ordinal_accuracy': pairwise_ordinal,
    }
    
    # Add input depth ordinal accuracy to results if available
    if input_depth_ordinal_accuracy is not None:
        results['qual_metrics']['input_depth_ordinal_accuracy'] = input_depth_ordinal_accuracy
        results['input_depth_ordinal_accuracy_per_frame'] = input_depth_ordinal_accuracy_per_frame
    
    # Add preprocessed DAV depths to results for visualization
    if dav_mm is not None:
        results['dav_mm'] = dav_mm
    
    print(f"\nQualitative Metrics:")
    print(f"  PCK3D-100mm: {pck3d_100:.4f}")
    print(f"  PCK3D-150mm: {pck3d_150:.4f}")
    print(f"  PCKt@0.5: {pckt_050:.4f}")
    print(f"  PCKt@1.0: {pckt_100:.4f}")
    print(f"\nDepth Understanding Metrics:")
    print(f"  Ordinal Depth Accuracy (exact): {ordinal_exact:.4f}")
    print(f"  Ordinal Depth Accuracy (±1): {ordinal_pm1:.4f}")
    print(f"  Coarse Ordinal Accuracy (100mm): {coarse_ordinal_100:.4f}")
    print(f"  Coarse Ordinal Accuracy (250mm): {coarse_ordinal_250:.4f}")
    print(f"  Pairwise Ordinal Accuracy (pred vs GT): {pairwise_ordinal:.4f}")
    if input_depth_ordinal_accuracy is not None:
        print(f"\nInput Depth Quality (Occlusion Proxy):")
        print(f"  DAV Input Ordinal Accuracy: {input_depth_ordinal_accuracy:.4f}")
        print(f"  (Lower = more depth ambiguity/occlusion)")
    
    return results



def main():
    parser = argparse.ArgumentParser(description='Compute metrics for PoseFormer models')
    parser.add_argument('--dataset', type=str, default='all',
                       choices=['h36m', '3dhp', '3dpw', 'fit3d', 'all'],
                       help='Dataset to analyze (default: all)')
    parser.add_argument('--models', type=str, nargs='+', default=['xy', 'xycd'],
                       choices=['xy', 'xyc', 'xyd', 'xycd'],
                       help='Model variants to analyze (default: xy xycd)')
    parser.add_argument('--compare', type=str, nargs=2, default=['xy', 'xycd'],
                       choices=['xy', 'xyc', 'xyd', 'xycd'],
                       help='Two models to compare (default: xy xycd)')
    parser.add_argument('--seqlen', type=int, default=27,
                       help='Sequence length (default: 27)')
    parser.add_argument('--confidence_threshold', type=float, default=0.3,
                       help='Minimum 2D keypoint confidence to include joint in metrics (default: 0.3, set to 0 to disable filtering)')
    parser.add_argument('--filter_by_confidence', action='store_true',
                       help='Enable joint-level confidence filtering (default: False). When enabled, joints below --confidence_threshold are excluded from metrics.')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate error distribution plots and visualizations')
    parser.add_argument('--visualize_contrast', action='store_true',
                       help='Visualize frames where model2 improves most over model1, but model2 error is still ≤150mm (contrast between good absolute performance and large improvement)')
    parser.add_argument('--contrast_threshold', type=float, default=150.0,
                       help='Maximum error threshold (mm) for model2 in contrast visualization (default: 150)')
    parser.add_argument('--contrast_mode', type=str, default='base', choices=['base', 'with_casp'],
                       help='Contrast visualization mode: base (original 4-panel) or with_casp (add CASP depth sampling visualization). Default: base')
    parser.add_argument('--cluster', action='store_true',
                       help='Find and visualize most representative poses using clustering')
    parser.add_argument('--num_clusters', type=int, default=100,
                       help='Number of representative poses to find via clustering (default: 100)')
    parser.add_argument('--num_vis', type=int, default=200,
                       help='Number of frames to visualize: top 50 improvements + random frames (default: 200)')
    parser.add_argument('--min_pct', type=int, default=80,
                       help='Minimum percentile for frame selection (default: 80)')
    parser.add_argument('--max_pct', type=int, default=95,
                       help='Maximum percentile for frame selection (default: 95)')
    parser.add_argument('--domain_similarity', action='store_true',
                       help='Analyze MPJPE stratified by similarity to source domain (H36M)')
    parser.add_argument('--occlusion_analysis', action='store_true',
                       help='Analyze MPJPE stratified by occlusion level')
    parser.add_argument('--occlusion_level', type=str, default='frame', choices=['frame', 'joint', 'depth', 'geometric'],
                       help='Occlusion analysis level: frame (per-frame mean visibility), joint (per-joint visibility), depth (DAV depth ordering accuracy), or geometric (3D-based self-occlusion). Default: frame')
    parser.add_argument('--detection_quality_analysis', action='store_true',
                       help='Analyze MPJPE stratified by 2D detection quality (distance from pred 2D to GT 2D keypoints)')
    parser.add_argument('--detection_quality_level', type=str, default='frame', choices=['frame', 'joint'],
                       help='Detection quality analysis level: frame (per-frame mean 2D error) or joint (per-joint 2D error). Default: frame')
    parser.add_argument('--angles_orientations', action='store_true',
                       help='Analyze angle and orientation errors between models (per-bone X/Y angles)')
    parser.add_argument('--motion_speed_analysis', action='store_true',
                       help='Analyze MPJPE stratified by smoothed local motion speed')
    parser.add_argument('--speed_window', type=int, default=10,
                       help='Smoothing window length for motion speed analysis (default: 10 frames)')
    parser.add_argument('--speed_bins', type=int, default=5,
                       help='Number of speed bins for motion analysis (default: 5)')
    parser.add_argument('--speed_level', type=str, default='frame', choices=['frame', 'joint', 'root'],
                       help='Motion speed analysis level: frame (per-frame mean speed), joint (per-joint speed), or root (root translation speed). Default: frame')
    parser.add_argument('--body_part_analysis', action='store_true',
                       help='Analyze MPJPE improvements by body part (which joint groups benefit most from augmented inputs)')
    parser.add_argument('--geometric_features', action='store_true',
                       help='Analyze MPJPE by geometric features: bbox scale, foreshortening, torso pitch, ordinal margin')
    parser.add_argument('--test_valid_only_3dhp', action='store_true',
                       help='For 3DHP dataset, use only test_valid subset (2,875 frames) instead of test_all (24,888 frames)')
    args = parser.parse_args()
    
    # Select datasets
    if args.dataset == 'all':
        datasets = ['h36m', '3dhp', '3dpw', 'fit3d']
    else:
        datasets = [args.dataset]
    
    print(f"\n{'='*60}")
    print(f"PoseFormer Metrics Analysis")
    print(f"{'='*60}")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Comparison: {args.compare[0]} vs {args.compare[1]}")
    print(f"Sequence length: {args.seqlen}\n")
    
    # Compute metrics for all dataset-model pairs
    all_results = {}
    for dataset in datasets:
        all_results[dataset] = {}
        for model in args.models:
            results = compute_metrics(
                dataset, 
                model, 
                seqlen=args.seqlen, 
                confidence_threshold=args.confidence_threshold,
                use_test_valid_3dhp=args.test_valid_only_3dhp,
                filter_by_confidence=args.filter_by_confidence
            )
            if results:
                all_results[dataset][model] = results
    
    # Print summary comparing the two specified models
    print(f"\n{'='*60}")
    print(f"SUMMARY: Metric Comparison ({args.compare[0].upper()} vs {args.compare[1].upper()})")
    print(f"{'='*60}\n")
    
    model1, model2 = args.compare[0], args.compare[1]
    
    for dataset in datasets:
        if dataset not in all_results:
            continue
        
        if model1 not in all_results[dataset] or model2 not in all_results[dataset]:
            print(f"\n{dataset.upper()}: Missing model results, skipping comparison")
            continue
        
        print(f"\n{dataset.upper()}")
        print(f"{'-'*60}")
        
        res1 = all_results[dataset][model1]
        res2 = all_results[dataset][model2]
        
        # MPJPE percentiles
        print(f"\nMPJPE Percentiles:")
        print(f"  {'Metric':<20} {model1.upper():<15} {model2.upper():<15} {'Delta':<15}")
        print(f"  {'-'*65}")
        
        # Add mean MPJPE first
        mean1 = res1['mean_mpjpe']
        mean2 = res2['mean_mpjpe']
        delta_mean = mean2 - mean1
        print(f"  {'mean':<20} {mean1:>10.2f} mm   {mean2:>10.2f} mm   {delta_mean:>+10.2f} mm")
        
        # Then percentiles
        for p in ['p25', 'p50', 'p75', 'p90', 'p95', 'p99']:
            val1 = res1[p]
            val2 = res2[p]
            delta = val2 - val1
            print(f"  {p:<20} {val1:>10.2f} mm   {val2:>10.2f} mm   {delta:>+10.2f} mm")
        
        # Qualitative metrics
        print(f"\nQualitative Metrics:")
        print(f"  {'Metric':<30} {model1.upper():<15} {model2.upper():<15} {'Delta':<15}")
        print(f"  {'-'*75}")
        
        for metric_name in res1['qual_metrics'].keys():
            val1 = res1['qual_metrics'][metric_name]
            val2 = res2['qual_metrics'][metric_name]
            delta = val2 - val1
            print(f"  {metric_name:<30} {val1:>10.4f}      {val2:>10.4f}      {delta:>+10.4f}")
    
    # Generate visualizations if requested
    if args.visualize:
        import shutil
        
        base_output_dir = "/srv/essa-lab/flash3/nwarner30/pose_estimation/poseformer_analysis"
        comparison_name = f"{model1}_vs_{model2}"
        output_dir = os.path.join(base_output_dir, comparison_name)
        
        print(f"\n{'='*60}")
        print(f"Generating Visualizations")
        print(f"{'='*60}")
        print(f"Output directory: {output_dir}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        print(f"  ✓ Created/verified output directory\n")
        
        for dataset in datasets:
            if dataset not in all_results:
                continue
            
            # Check if both comparison models are available
            if model1 not in all_results[dataset] or model2 not in all_results[dataset]:
                print(f"\nSkipping visualizations for {dataset} (missing model results)")
                continue
            
            print(f"\nProcessing {dataset}...")
            res1 = all_results[dataset][model1]
            res2 = all_results[dataset][model2]
            
            # Plot error distributions
            plot_error_distributions(res1, res2, dataset, output_dir, model1, model2)
            
            # Find and list interesting frames
            find_and_visualize_frames(res1, res2, dataset, output_dir, 
                                     num_frames=args.num_vis,
                                     min_pct=args.min_pct,
                                     max_pct=args.max_pct,
                                     model1=model1,
                                     model2=model2)
    
    # Generate contrast visualizations if requested
    if args.visualize_contrast:
        import shutil
        
        base_output_dir = "/srv/essa-lab/flash3/nwarner30/pose_estimation/poseformer_analysis"
        comparison_name = f"{model1}_vs_{model2}_contrast_improvements"
        output_dir = os.path.join(base_output_dir, comparison_name)
        
        print(f"\n{'='*60}")
        print(f"Generating Contrast Visualizations")
        print(f"{'='*60}")
        print(f"Output directory: {output_dir}")
        print(f"Finding frames where {model2.upper()} improves most over {model1.upper()},")
        print(f"but {model2.upper()} error ≤ {args.contrast_threshold:.0f}mm")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        print(f"  ✓ Created/verified output directory\n")
        
        for dataset in datasets:
            if dataset not in all_results:
                continue
            
            # Check if both comparison models are available
            if model1 not in all_results[dataset] or model2 not in all_results[dataset]:
                print(f"\nSkipping contrast visualizations for {dataset} (missing model results)")
                continue
            
            print(f"\nProcessing {dataset}...")
            res1 = all_results[dataset][model1]
            res2 = all_results[dataset][model2]
            
            # Find contrast frames: large improvement but good absolute performance
            find_and_visualize_contrast_frames(res1, res2, dataset, output_dir,
                                              model1=model1,
                                              model2=model2,
                                              contrast_threshold=args.contrast_threshold,
                                              num_frames=args.num_vis,
                                              contrast_mode=args.contrast_mode)
    
    # Generate cluster-based visualizations if requested
    if args.cluster:
        import shutil
        
        base_output_dir = "/srv/essa-lab/flash3/nwarner30/pose_estimation/poseformer_analysis"
        comparison_name = f"{model1}_vs_{model2}_representative_cluster_scenes"
        output_dir = os.path.join(base_output_dir, comparison_name)
        
        print(f"\n{'='*60}")
        print(f"Generating Cluster-Based Representative Visualizations")
        print(f"{'='*60}")
        print(f"Output directory: {output_dir}")
        
        # Clean up existing output directory if it exists
        if os.path.exists(output_dir):
            print(f"  Removing existing output directory...")
            shutil.rmtree(output_dir)
            print(f"  ✓ Cleaned up existing visualizations")
        
        # Create fresh output directory
        os.makedirs(output_dir, exist_ok=True)
        print(f"  ✓ Created fresh output directory\n")
        
        for dataset in datasets:
            if dataset not in all_results:
                continue
            
            # Check if both comparison models are available
            if model1 not in all_results[dataset] or model2 not in all_results[dataset]:
                print(f"\nSkipping cluster visualizations for {dataset} (missing model results)")
                continue
            
            print(f"\nProcessing {dataset} with clustering...")
            res1 = all_results[dataset][model1]
            res2 = all_results[dataset][model2]
            
            # Find and visualize representative poses using clustering
            find_and_visualize_representative_poses(res1, res2, dataset, output_dir,
                                                   num_clusters=args.num_clusters,
                                                   model1=model1,
                                                   model2=model2)
    
    # Generate domain similarity analysis if requested
    if args.domain_similarity:
        base_output_dir = "/srv/essa-lab/flash3/nwarner30/pose_estimation/poseformer_analysis"
        comparison_name = f"{model1}_vs_{model2}_domain_similarity"
        output_dir = os.path.join(base_output_dir, comparison_name)
        
        print(f"\n{'='*60}")
        print(f"Generating Domain Similarity Analysis")
        print(f"{'='*60}")
        print(f"Output directory: {output_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"  ✓ Created/verified output directory\n")
        
        # Analyze each test dataset (exclude h36m as it's the source)
        for dataset in datasets:
            if dataset == 'h36m':
                print(f"\nSkipping H36M (source domain)")
                continue
            
            if dataset not in all_results:
                continue
            
            # Check if both models are available for comparison
            if model1 not in all_results[dataset] or model2 not in all_results[dataset]:
                print(f"\nSkipping {dataset} (missing one or both models)")
                continue
            
            # Analyze both models
            results_by_model = {}
            for model in [model1, model2]:
                print(f"\nAnalyzing {dataset} - {model.upper()}...")
                tier_stats, vis_frames = analyze_domain_similarity(
                    all_results[dataset][model], 
                    dataset, 
                    output_dir, 
                    model_name=model
                )
                results_by_model[model] = (tier_stats, vis_frames)
            
            # Visualize frames with side-by-side comparison
            if model1 in results_by_model and model2 in results_by_model:
                _, vis_frames = results_by_model[model1]  # Use model1's frame selection
                
                if vis_frames:
                    print(f"\n  Visualizing {len(vis_frames)} representative frames (comparing {model1.upper()} vs {model2.upper()})...")
                    frames_output_dir = os.path.join(output_dir, f'frames_{dataset}_{model1}_vs_{model2}_similarity_tiers')
                    visualize_frame_skeletons(
                        vis_frames,
                        all_results[dataset][model1],
                        all_results[dataset][model2],  # Compare model1 vs model2
                        dataset,
                        frames_output_dir,
                        model1,
                        model2
                    )
    
    # Generate occlusion analysis if requested
    if args.occlusion_analysis:
        base_output_dir = "/srv/essa-lab/flash3/nwarner30/pose_estimation/poseformer_analysis"
        comparison_name = f"{model1}_vs_{model2}_occlusion_analysis"
        output_dir = os.path.join(base_output_dir, comparison_name)
        
        print(f"\n{'='*60}")
        print(f"Generating Occlusion Analysis")
        print(f"{'='*60}")
        print(f"Output directory: {output_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"  ✓ Created/verified output directory\n")
        
        # Analyze all datasets
        for dataset in datasets:
            if dataset not in all_results:
                continue
            
            # Check if both models are available for comparison
            if model1 not in all_results[dataset] or model2 not in all_results[dataset]:
                print(f"\nSkipping {dataset} (missing one or both models)")
                continue
            
            # Analyze both models
            results_by_model = {}
            for model in [model1, model2]:
                print(f"\nAnalyzing {dataset} - {model.upper()}...")
                # Use different analysis function based on occlusion level
                if args.occlusion_level == 'depth':
                    result = analyze_depth_ordering_occlusion(
                        all_results[dataset][model], 
                        dataset, 
                        output_dir, 
                        model_name=model
                    )
                elif args.occlusion_level == 'geometric':
                    result = analyze_geometric_occlusion(
                        all_results[dataset][model], 
                        dataset, 
                        output_dir, 
                        model_name=model
                    )
                else:
                    result = analyze_occlusion_levels(
                        all_results[dataset][model], 
                        dataset, 
                        output_dir, 
                        model_name=model,
                        level=args.occlusion_level
                    )
                if result is not None:
                    results_by_model[model] = result
            
            # Visualize frames with side-by-side comparison if both models succeeded
            if model1 in results_by_model and model2 in results_by_model:
                _, vis_frames = results_by_model[model1]  # Use model1's frame selection
                
                if vis_frames:
                    print(f"\n  Visualizing {len(vis_frames)} representative frames (comparing {model1.upper()} vs {model2.upper()})...")
                    frames_output_dir = os.path.join(output_dir, f'frames_{dataset}_{model1}_vs_{model2}_occlusion_tiers')
                    visualize_frame_skeletons(
                        vis_frames,
                        all_results[dataset][model1],
                        all_results[dataset][model2],  # Compare model1 vs model2
                        dataset,
                        frames_output_dir,
                        model1,
                        model2
                    )
            
            # Generate comparison table (only for frame-level analysis)
            if model1 in results_by_model and model2 in results_by_model and args.occlusion_level != 'joint':
                tier_stats_1, _ = results_by_model[model1]
                tier_stats_2, _ = results_by_model[model2]
                
                print(f"\n{'='*80}")
                print(f"Occlusion Analysis Comparison: {model1.upper()} vs {model2.upper()} ({dataset})")
                print(f"{'='*80}")
                
                for tier_name in tier_stats_1.keys():
                    if tier_name not in tier_stats_2:
                        continue
                    
                    tier_label = tier_name.replace('_', ' ').title()
                    stats1 = tier_stats_1[tier_name]
                    stats2 = tier_stats_2[tier_name]
                    
                    mpjpe1 = stats1['mean_mpjpe']
                    mpjpe2 = stats2['mean_mpjpe']
                    delta = mpjpe2 - mpjpe1
                    rel_delta = (delta / mpjpe1 * 100) if mpjpe1 > 0 else 0
                    pct_benefit = -rel_delta  # Negative delta = improvement
                    
                    # Tail metrics
                    p90_1 = stats1['p90']
                    p90_2 = stats2['p90']
                    delta_p90 = p90_2 - p90_1
                    rel_delta_p90 = (delta_p90 / p90_1 * 100) if p90_1 > 0 else 0
                    
                    p95_1 = stats1['p95']
                    p95_2 = stats2['p95']
                    delta_p95 = p95_2 - p95_1
                    rel_delta_p95 = (delta_p95 / p95_1 * 100) if p95_1 > 0 else 0
                    
                    print(f"\n{tier_label} (N={stats1['n_frames']} frames):")
                    if args.occlusion_level == 'depth':
                        print(f"  Mean DAV ordering accuracy: {stats1.get('mean_dav_accuracy', 0):.4f}")
                    else:
                        print(f"  Mean visibility: {stats1.get('mean_visibility', 0):.3f}")
                    print(f"  {model1.upper()} Mean: {mpjpe1:.2f} mm, p90: {p90_1:.2f} mm, p95: {p95_1:.2f} mm")
                    print(f"  {model2.upper()} Mean: {mpjpe2:.2f} mm, p90: {p90_2:.2f} mm, p95: {p95_2:.2f} mm")
                    print(f"  Mean Δ: {delta:+.2f} mm ({rel_delta:+.1f}%) | Benefit: {pct_benefit:+.1f}%")
                    print(f"  p90 Δ: {delta_p90:+.2f} mm ({rel_delta_p90:+.1f}%)")
                    print(f"  p95 Δ: {delta_p95:+.2f} mm ({rel_delta_p95:+.1f}%)")
                
                # Save comparison table
                comparison_path = os.path.join(output_dir, f'occlusion_comparison_{dataset}_{model1}_vs_{model2}.txt')
                with open(comparison_path, 'w') as f:
                    f.write(f"Occlusion Analysis Model Comparison: {model1.upper()} vs {model2.upper()}\n")
                    f.write(f"Dataset: {dataset}\n")
                    if args.occlusion_level == 'depth':
                        f.write(f"Occlusion metric: DAV input depth ordering accuracy (lower = more occluded)\n")
                    else:
                        f.write(f"Occlusion metric: 2D keypoint visibility (lower = more occluded)\n")
                    f.write("="*80 + "\n\n")
                    
                    for tier_name in tier_stats_1.keys():
                        if tier_name not in tier_stats_2:
                            continue
                        
                        tier_label = tier_name.replace('_', ' ').title()
                        stats1 = tier_stats_1[tier_name]
                        stats2 = tier_stats_2[tier_name]
                        
                        f.write(f"{tier_label}:\n")
                        f.write(f"  N frames: {stats1['n_frames']}\n")
                        if args.occlusion_level == 'depth':
                            f.write(f"  Mean DAV ordering accuracy: {stats1.get('mean_dav_accuracy', 0):.4f}\n")
                        else:
                            f.write(f"  Mean visibility: {stats1.get('mean_visibility', 0):.3f}\n")
                        f.write(f"\n")
                        
                        f.write(f"  {model1.upper()} Metrics:\n")
                        f.write(f"    Mean MPJPE: {stats1['mean_mpjpe']:.2f} mm\n")
                        f.write(f"    Median MPJPE: {stats1['median_mpjpe']:.2f} mm\n")
                        f.write(f"    p80: {stats1.get('p80', stats1['p75']):.2f} mm\n")
                        f.write(f"    p90: {stats1['p90']:.2f} mm\n")
                        f.write(f"    p95: {stats1['p95']:.2f} mm\n\n")
                        
                        f.write(f"  {model2.upper()} Metrics:\n")
                        f.write(f"    Mean MPJPE: {stats2['mean_mpjpe']:.2f} mm\n")
                        f.write(f"    Median MPJPE: {stats2['median_mpjpe']:.2f} mm\n")
                        f.write(f"    p80: {stats2.get('p80', stats2['p75']):.2f} mm\n")
                        f.write(f"    p90: {stats2['p90']:.2f} mm\n")
                        f.write(f"    p95: {stats2['p95']:.2f} mm\n\n")
                        
                        mpjpe1 = stats1['mean_mpjpe']
                        mpjpe2 = stats2['mean_mpjpe']
                        delta = mpjpe2 - mpjpe1
                        rel_delta = (delta / mpjpe1 * 100) if mpjpe1 > 0 else 0
                        pct_benefit = -rel_delta
                        
                        p90_1 = stats1['p90']
                        p90_2 = stats2['p90']
                        delta_p90 = p90_2 - p90_1
                        rel_delta_p90 = (delta_p90 / p90_1 * 100) if p90_1 > 0 else 0
                        
                        p95_1 = stats1['p95']
                        p95_2 = stats2['p95']
                        delta_p95 = p95_2 - p95_1
                        rel_delta_p95 = (delta_p95 / p95_1 * 100) if p95_1 > 0 else 0
                        
                        f.write(f"  Comparison (Mean):\n")
                        f.write(f"    Absolute delta: {delta:+.2f} mm\n")
                        f.write(f"    Relative delta: {rel_delta:+.1f}%\n")
                        f.write(f"    Percentage benefit: {pct_benefit:+.1f}%\n\n")
                        
                        f.write(f"  Comparison (Tail - p90):\n")
                        f.write(f"    {model1.upper()}: {p90_1:.2f} mm\n")
                        f.write(f"    {model2.upper()}: {p90_2:.2f} mm\n")
                        f.write(f"    Absolute delta: {delta_p90:+.2f} mm\n")
                        f.write(f"    Relative delta: {rel_delta_p90:+.1f}%\n\n")
                        
                        f.write(f"  Comparison (Tail - p95):\n")
                        f.write(f"    {model1.upper()}: {p95_1:.2f} mm\n")
                        f.write(f"    {model2.upper()}: {p95_2:.2f} mm\n")
                        f.write(f"    Absolute delta: {delta_p95:+.2f} mm\n")
                        f.write(f"    Relative delta: {rel_delta_p95:+.1f}%\n\n")
                        f.write("="*80 + "\n\n")
                
                print(f"\n  Saved comparison table: {comparison_path}")
            elif args.occlusion_level == 'joint':
                print(f"\n  Note: Skipping comparison table for joint-level analysis (use frame-level for model comparison)")
    
    # Generate depth ordering occlusion analysis if requested
    if args.occlusion_analysis and args.occlusion_level == 'depth':
        base_output_dir = "/srv/essa-lab/flash3/nwarner30/pose_estimation/poseformer_analysis"
        comparison_name = f"{model1}_vs_{model2}_depth_ordering_occlusion"
        output_dir = os.path.join(base_output_dir, comparison_name)
        
        print(f"\n{'='*60}")
        print(f"Generating Depth Ordering Occlusion Analysis")
        print(f"{'='*60}")
        print(f"Output directory: {output_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"  ✓ Created/verified output directory\n")
        
        # Analyze all datasets
        for dataset in datasets:
            if dataset not in all_results:
                continue
            
            # Check if both models are available for comparison
            if model1 not in all_results[dataset] or model2 not in all_results[dataset]:
                print(f"\nSkipping {dataset} (missing one or both models)")
                continue
            
            # Analyze both models
            results_by_model = {}
            for model in [model1, model2]:
                print(f"\nAnalyzing {dataset} - {model.upper()}...")
                result = analyze_depth_ordering_occlusion(
                    all_results[dataset][model], 
                    dataset, 
                    output_dir, 
                    model_name=model
                )
                if result is not None:
                    results_by_model[model] = result
            
            # Visualize frames with side-by-side comparison if both models succeeded
            if model1 in results_by_model and model2 in results_by_model:
                _, vis_frames = results_by_model[model1]  # Use model1's frame selection
                
                if vis_frames:
                    print(f"\n  Visualizing {len(vis_frames)} representative frames (comparing {model1.upper()} vs {model2.upper()})...")
                    frames_output_dir = os.path.join(output_dir, f'frames_{dataset}_{model1}_vs_{model2}_depth_ordering_tiers')
                    visualize_frame_skeletons(
                        vis_frames,
                        all_results[dataset][model1],
                        all_results[dataset][model2],  # Compare model1 vs model2
                        dataset,
                        frames_output_dir,
                        model1,
                        model2
                    )
            
            # Generate comparison table
            if model1 in results_by_model and model2 in results_by_model:
                tier_stats_1, _ = results_by_model[model1]
                tier_stats_2, _ = results_by_model[model2]
                
                print(f"\n{'='*80}")
                print(f"Depth Ordering Occlusion Comparison: {model1.upper()} vs {model2.upper()} ({dataset})")
                print(f"{'='*80}")
                
                for tier_name in tier_stats_1.keys():
                    if tier_name not in tier_stats_2:
                        continue
                    
                    tier_label = tier_name.replace('_', ' ').title()
                    stats1 = tier_stats_1[tier_name]
                    stats2 = tier_stats_2[tier_name]
                    
                    mpjpe1 = stats1['mean_mpjpe']
                    mpjpe2 = stats2['mean_mpjpe']
                    delta = mpjpe2 - mpjpe1
                    rel_delta = (delta / mpjpe1 * 100) if mpjpe1 > 0 else 0
                    
                    print(f"\n{tier_label} (N={stats1['n_frames']} frames):")
                    print(f"  {model1.upper()} Mean MPJPE: {mpjpe1:.2f} mm")
                    print(f"  {model2.upper()} Mean MPJPE: {mpjpe2:.2f} mm")
                    print(f"  Delta: {delta:+.2f} mm ({rel_delta:+.1f}%)")
                
                # Save comparison table
                comparison_path = os.path.join(output_dir, f'depth_ordering_comparison_{dataset}_{model1}_vs_{model2}.txt')
                with open(comparison_path, 'w') as f:
                    f.write(f"Depth Ordering Occlusion Model Comparison: {model1.upper()} vs {model2.upper()}\n")
                    f.write(f"Dataset: {dataset}\n")
                    f.write(f"Occlusion metric: DAV input depth ordering accuracy (lower = more occluded)\n")
                    f.write("="*80 + "\n\n")
                    
                    for tier_name in tier_stats_1.keys():
                        if tier_name not in tier_stats_2:
                            continue
                        
                        tier_label = tier_name.replace('_', ' ').title()
                        stats1 = tier_stats_1[tier_name]
                        stats2 = tier_stats_2[tier_name]
                        
                        f.write(f"{tier_label}:\n")
                        f.write(f"  N frames: {stats1['n_frames']}\n")
                        f.write(f"  Mean DAV ordering accuracy: {stats1.get('mean_dav_accuracy', 0):.4f}\n")
                        f.write(f"  Mean MPJPE: {stats1['mean_mpjpe']:.2f} mm\n")
                        f.write(f"  Median MPJPE: {stats1['median_mpjpe']:.2f} mm\n")
                        f.write(f"  Std MPJPE: {stats1['std_mpjpe']:.2f} mm\n")
                        f.write(f"  25th percentile: {stats1['p25']:.2f} mm\n")
                        f.write(f"  75th percentile: {stats1['p75']:.2f} mm\n")
                        f.write(f"  90th percentile: {stats1['p90']:.2f} mm\n")
                        f.write(f"  95th percentile: {stats1['p95']:.2f} mm\n\n")
                        
                        f.write(f"  {model2.upper()} Metrics:\n")
                        f.write(f"    Mean MPJPE: {stats2['mean_mpjpe']:.2f} mm\n")
                        f.write(f"    Median MPJPE: {stats2['median_mpjpe']:.2f} mm\n")
                        f.write(f"    p90: {stats2['p90']:.2f} mm\n")
                        f.write(f"    p95: {stats2['p95']:.2f} mm\n\n")
                        
                        mpjpe1 = stats1['mean_mpjpe']
                        mpjpe2 = stats2['mean_mpjpe']
                        delta = mpjpe2 - mpjpe1
                        rel_delta = (delta / mpjpe1 * 100) if mpjpe1 > 0 else 0
                        
                        f.write(f"  Comparison:\n")
                        f.write(f"    Absolute delta: {delta:+.2f} mm\n")
                        f.write(f"    Relative delta: {rel_delta:+.1f}%\n")
                        f.write(f"    {model2.upper()} improvement: {-delta:.2f} mm\n\n")
                        f.write("="*80 + "\n\n")
                
                print(f"\n  Saved comparison table: {comparison_path}")
    
    # Generate detection quality analysis if requested
    if args.detection_quality_analysis:
        base_output_dir = "/srv/essa-lab/flash3/nwarner30/pose_estimation/poseformer_analysis"
        comparison_name = f"{model1}_vs_{model2}_detection_quality_analysis"
        output_dir = os.path.join(base_output_dir, comparison_name)
        
        print(f"\n{'='*60}")
        print(f"Generating 2D Detection Quality Analysis")
        print(f"{'='*60}")
        print(f"Output directory: {output_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"  ✓ Created/verified output directory\n")
        
        # Analyze all datasets
        for dataset in datasets:
            if dataset not in all_results:
                continue
            
            # Check if both models are available for comparison
            if model1 not in all_results[dataset] or model2 not in all_results[dataset]:
                print(f"\nSkipping {dataset} (missing one or both models)")
                continue
            
            # Analyze both models
            results_by_model = {}
            for model in [model1, model2]:
                print(f"\nAnalyzing {dataset} - {model.upper()}...")
                result = analyze_detection_quality(
                    all_results[dataset][model], 
                    dataset, 
                    output_dir, 
                    model_name=model,
                    level=args.detection_quality_level
                )
                if result is not None:
                    results_by_model[model] = result
            
            # Visualize frames with side-by-side comparison if both models succeeded
            if model1 in results_by_model and model2 in results_by_model:
                _, vis_frames = results_by_model[model1]  # Use model1's frame selection
                
                if vis_frames:
                    print(f"\n  Visualizing {len(vis_frames)} representative frames (comparing {model1.upper()} vs {model2.upper()})...")
                    frames_output_dir = os.path.join(output_dir, f'frames_{dataset}_{model1}_vs_{model2}_detection_quality')
                    visualize_frame_skeletons(
                        vis_frames,
                        all_results[dataset][model1],
                        all_results[dataset][model2],  # Compare model1 vs model2
                        dataset,
                        frames_output_dir,
                        model1,
                        model2
                    )
    
    print(f"\n{'='*60}")
    print(f"Analysis complete!")
    print(f"{'='*60}")


    # Generate angle/orientation analysis if requested
    if args.angles_orientations:
        base_output_dir = "/srv/essa-lab/flash3/nwarner30/pose_estimation/poseformer_analysis"
        comparison_name = f"{model1}_vs_{model2}_angles_orientations"
        output_dir = os.path.join(base_output_dir, comparison_name)
        
        print(f"\n{'='*60}")
        print(f"Generating Angle & Orientation Analysis")
        print(f"{'='*60}")
        print(f"Output directory: {output_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"  ✓ Created/verified output directory\n")
        
        # Analyze all datasets where both models are available
        for dataset in datasets:
            if dataset not in all_results:
                continue
            
            # Check if both comparison models are available
            if model1 not in all_results[dataset] or model2 not in all_results[dataset]:
                print(f"\nSkipping {dataset} (missing one or both models)")
                continue
            
            print(f"\nProcessing {dataset}...")
            
            # Pass both models' results for comparison
            results_by_model = {
                model1: all_results[dataset][model1],
                model2: all_results[dataset][model2]
            }
            
            analyze_angles_orientations(
                results_by_model,
                dataset,
                output_dir,
                [model1, model2]
            )
    
    # Generate motion speed analysis if requested
    if args.motion_speed_analysis:
        base_output_dir = "/srv/essa-lab/flash3/nwarner30/pose_estimation/poseformer_analysis"
        comparison_name = f"{model1}_vs_{model2}_motion_speed_analysis"
        output_dir = os.path.join(base_output_dir, comparison_name)
        
        print(f"\n{'='*60}")
        print(f"Generating Motion Speed Analysis")
        print(f"{'='*60}")
        print(f"Output directory: {output_dir}")
        print(f"Window: {args.speed_window} frames, Bins: {args.speed_bins}, Level: {args.speed_level}")
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"  ✓ Created/verified output directory\n")
        
        # Analyze all datasets
        for dataset in datasets:
            if dataset not in all_results:
                continue
            
            # Check if both models are available for comparison
            if model1 not in all_results[dataset] or model2 not in all_results[dataset]:
                print(f"\nSkipping {dataset} (missing one or both models)")
                continue
            
            # Analyze both models
            results_by_model = {}
            for model in [model1, model2]:
                print(f"\nAnalyzing {dataset} - {model.upper()}...")
                result = analyze_motion_speed(
                    all_results[dataset][model], 
                    dataset, 
                    output_dir, 
                    model_name=model,
                    level=args.speed_level,
                    window=args.speed_window,
                    num_bins=args.speed_bins
                )
                if result is not None:
                    results_by_model[model] = result
            
            # Visualize frames with side-by-side comparison if both models succeeded
            if model1 in results_by_model and model2 in results_by_model and args.speed_level == 'frame':
                _, vis_frames = results_by_model[model1]  # Use model1's frame selection
                
                if vis_frames:
                    print(f"\n  Visualizing {len(vis_frames)} representative frames (comparing {model1.upper()} vs {model2.upper()})...")
                    frames_output_dir = os.path.join(output_dir, f'frames_{dataset}_{model1}_vs_{model2}_speed_tiers')
                    visualize_frame_skeletons(
                        vis_frames,
                        all_results[dataset][model1],
                        all_results[dataset][model2],  # Compare model1 vs model2
                        dataset,
                        frames_output_dir,
                        model1,
                        model2
                    )
            
            # Generate comparison table
            if model1 in results_by_model and model2 in results_by_model:
                tier_stats_1, _ = results_by_model[model1]
                tier_stats_2, _ = results_by_model[model2]
                
                if tier_stats_1 and tier_stats_2:
                    print(f"\n{'='*80}")
                    print(f"Motion Speed Analysis Comparison: {model1.upper()} vs {model2.upper()} ({dataset})")
                    print(f"{'='*80}")
                    
                    # Get common bin keys
                    common_bins = [k for k in tier_stats_1.keys() if k in tier_stats_2]
                    
                    for bin_name in common_bins:
                        stats1 = tier_stats_1[bin_name]
                        stats2 = tier_stats_2[bin_name]
                        
                        if args.speed_level in ['frame', 'root']:
                            mpjpe1 = stats1['mean_mpjpe']
                            mpjpe2 = stats2['mean_mpjpe']
                            delta = mpjpe2 - mpjpe1
                            rel_delta = (delta / mpjpe1 * 100) if mpjpe1 > 0 else 0
                            
                            print(f"\n{bin_name} (N1={stats1['n_frames']}, N2={stats2['n_frames']} frames):")
                            print(f"  Mean speed: {stats1['mean_speed']:.2f} mm/frame")
                            print(f"  {model1.upper()} Mean MPJPE: {mpjpe1:.2f} mm")
                            print(f"  {model2.upper()} Mean MPJPE: {mpjpe2:.2f} mm")
                            print(f"  Delta: {delta:+.2f} mm ({rel_delta:+.1f}%)")
                        else:  # joint level
                            err1 = stats1['mean_error']
                            err2 = stats2['mean_error']
                            delta = err2 - err1
                            rel_delta = (delta / err1 * 100) if err1 > 0 else 0
                            
                            print(f"\n{bin_name} (N1={stats1['n_joints']}, N2={stats2['n_joints']} joints):")
                            print(f"  Mean speed: {stats1['mean_speed']:.2f} mm/frame")
                            print(f"  {model1.upper()} Mean Error: {err1:.2f} mm")
                            print(f"  {model2.upper()} Mean Error: {err2:.2f} mm")
                            print(f"  Delta: {delta:+.2f} mm ({rel_delta:+.1f}%)")
    
    # Generate body part analysis if requested
    if args.body_part_analysis:
        base_output_dir = "/srv/essa-lab/flash3/nwarner30/pose_estimation/poseformer_analysis"
        comparison_name = f"{model1}_vs_{model2}_body_part_analysis"
        output_dir = os.path.join(base_output_dir, comparison_name)
        
        print(f"\n{'='*60}")
        print(f"Generating Body Part Analysis")
        print(f"{'='*60}")
        print(f"Output directory: {output_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"  ✓ Created/verified output directory\n")
        
        # Analyze all datasets
        for dataset in datasets:
            if dataset not in all_results:
                continue
            
            # Check if both models are available for comparison
            if model1 not in all_results[dataset] or model2 not in all_results[dataset]:
                print(f"\nSkipping {dataset} (missing one or both models)")
                continue
            
            print(f"\nAnalyzing {dataset}...")
            analyze_body_part_improvements(
                all_results[dataset][model1],
                all_results[dataset][model2],
                dataset,
                output_dir,
                model1,
                model2
            )
    
    # Generate geometric features analysis if requested
    if args.geometric_features:
        base_output_dir = "/srv/essa-lab/flash3/nwarner30/pose_estimation/poseformer_analysis"
        comparison_name = f"{model1}_vs_{model2}_geometric_features"
        output_dir = os.path.join(base_output_dir, comparison_name)
        
        print(f"\n{'='*60}")
        print(f"Generating Geometric Features Analysis")
        print(f"{'='*60}")
        print(f"Output directory: {output_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"  ✓ Created/verified output directory\n")
        
        # Analyze all datasets
        for dataset in datasets:
            if dataset not in all_results:
                continue
            
            # Check if both models are available for comparison
            if model1 not in all_results[dataset] or model2 not in all_results[dataset]:
                print(f"\nSkipping {dataset} (missing one or both models)")
                continue
            
            print(f"\nAnalyzing {dataset}...")
            analyze_geometric_features(
                all_results[dataset][model1],
                all_results[dataset][model2],
                dataset,
                output_dir,
                model1,
                model2
            )

    print(f"\n{'='*60}")
    print(f"Analysis complete!")
    print(f"{'='*60}")


def plot_error_distributions(results_model1, results_model2, dataset, output_dir, model1='base', model2='xycd'):
    """Plot error distribution comparisons between two models."""
    import matplotlib.pyplot as plt
    
    os.makedirs(output_dir, exist_ok=True)
    
    errors1 = results_model1['per_frame_mpjpe']
    errors2 = results_model2['per_frame_mpjpe']
    
    # Align by length
    n = min(len(errors1), len(errors2))
    errors1 = errors1[:n]
    errors2 = errors2[:n]
    
    # Filter out NaN values
    valid_mask = np.isfinite(errors1) & np.isfinite(errors2)
    n_invalid = np.sum(~valid_mask)
    if n_invalid > 0:
        print(f"  Filtering out {n_invalid} frames with NaN MPJPE for visualization")
    
    if np.sum(valid_mask) == 0:
        print(f"  No valid frames for plotting")
        return
    
    errors1 = errors1[valid_mask]
    errors2 = errors2[valid_mask]
    
    # Compute deltas
    deltas = errors2 - errors1
    rel_deltas = deltas / (errors1 + 1e-8)
    
    # Create 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Panel 1: MPJPE distributions
    axes[0].hist(errors1, bins=50, alpha=0.6, label=f'{model1.upper()}', color='red')
    axes[0].hist(errors2, bins=50, alpha=0.6, label=f'{model2.upper()}', color='blue')
    axes[0].axvline(np.median(errors1), color='red', linestyle='--', alpha=0.8, label=f'{model1.upper()} median: {np.median(errors1):.1f}mm')
    axes[0].axvline(np.median(errors2), color='blue', linestyle='--', alpha=0.8, label=f'{model2.upper()} median: {np.median(errors2):.1f}mm')
    axes[0].set_xlabel('MPJPE (mm)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'MPJPE Distribution - {dataset}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Panel 2: Absolute delta distribution
    axes[1].hist(deltas, bins=60, color='green', alpha=0.7)
    axes[1].axvline(0, color='black', linestyle='--', linewidth=1.5, label='No change')
    axes[1].axvline(np.median(deltas), color='red', linestyle='--', alpha=0.8, label=f'Median: {np.median(deltas):.2f}mm')
    axes[1].set_xlabel(f'{model2.upper()} - {model1.upper()} (mm)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Absolute Error Delta')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Panel 3: Relative delta distribution
    axes[2].hist(rel_deltas, bins=60, color='purple', alpha=0.7)
    axes[2].axvline(0, color='black', linestyle='--', linewidth=1.5, label='No change')
    axes[2].axvline(np.median(rel_deltas), color='red', linestyle='--', alpha=0.8, label=f'Median: {np.median(rel_deltas):.3f}')
    axes[2].set_xlabel(f'({model2.upper()} - {model1.upper()}) / {model1.upper()}')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Relative Error Delta')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'error_distributions_{dataset}_{model1}_vs_{model2}.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved error distribution plot: {output_path}")
    
    # Print statistics
    print(f"\n  Delta Statistics:")
    print(f"    Mean absolute delta: {np.mean(deltas):.2f} mm")
    print(f"    Median absolute delta: {np.median(deltas):.2f} mm")
    print(f"    Mean relative delta: {np.mean(rel_deltas):.4f}")
    print(f"    Median relative delta: {np.median(rel_deltas):.4f}")
    print(f"    Frames where {model2.upper()} improves: {np.sum(deltas < 0)} ({np.mean(deltas < 0)*100:.1f}%)")
    print(f"    Frames where {model2.upper()} degrades: {np.sum(deltas > 0)} ({np.mean(deltas > 0)*100:.1f}%)")


def find_and_visualize_frames(results_model1, results_model2, dataset, output_dir, num_frames=50, min_pct=80, max_pct=95, model1='base', model2='xycd'):
    """Find and visualize frames where model2 shows largest improvements/degradations compared to model1."""
    print(f"\n{'='*60}")
    print(f"Finding interesting frames for {dataset}")
    print(f"{'='*60}")
    
    errors1 = results_model1['per_frame_mpjpe']
    errors2 = results_model2['per_frame_mpjpe']
    imgnames = results_model1['imgnames']
    
    # Align by length
    n = min(len(errors1), len(errors2), len(imgnames))
    errors1 = errors1[:n]
    errors2 = errors2[:n]
    imgnames = imgnames[:n]
    
    # Filter out NaN values (frames with all joints filtered by confidence threshold)
    valid_mask = np.isfinite(errors1) & np.isfinite(errors2)
    n_invalid = np.sum(~valid_mask)
    if n_invalid > 0:
        print(f"  Filtering out {n_invalid} frames with NaN MPJPE (all joints filtered by confidence threshold)")
    
    if np.sum(valid_mask) == 0:
        print(f"  No valid frames remaining after filtering NaN values")
        return []
    
    # Apply valid mask
    errors1_valid = errors1[valid_mask]
    errors2_valid = errors2[valid_mask]
    imgnames_valid = imgnames[valid_mask]
    valid_indices = np.where(valid_mask)[0]
    
    # Filter to percentile window (based on model1 errors)
    pmin = np.nanpercentile(errors1_valid, min_pct)
    pmax = np.nanpercentile(errors1_valid, max_pct)
    mask = (errors1_valid >= pmin) & (errors1_valid < pmax)
    
    print(f"  {model1.upper()} error percentile window: [{min_pct}, {max_pct}) = [{pmin:.1f}, {pmax:.1f}] mm")
    print(f"  Frames in window: {np.sum(mask)} / {len(errors1_valid)}")
    
    # Build frame info list
    frames = []
    for i in range(len(errors1_valid)):
        if not mask[i]:
            continue
        
        delta = errors2_valid[i] - errors1_valid[i]
        rel_delta = delta / (errors1_valid[i] + 1e-8)
        
        frames.append({
            'idx': valid_indices[i],  # Use original index for later lookup
            'imgname': imgnames_valid[i],
            f'{model1}_error': errors1_valid[i],
            f'{model2}_error': errors2_valid[i],
            'delta': delta,
            'rel_delta': rel_delta
        })
    
    if not frames:
        print("  No frames found in percentile window")
        return
    
    # Find frames with largest improvements (most negative rel_delta)
    frames_sorted = sorted(frames, key=lambda x: x['rel_delta'])
    
    # Select top improvements while avoiding consecutive frames (within 10 frames)
    num_top = min(50, num_frames)  # Top 50 improvements
    num_random = max(0, num_frames - num_top)  # Remaining frames (e.g., 150 for total of 200)
    
    best_improvements = []
    min_frame_gap = 10
    used_indices = set()
    
    for frame in frames_sorted:
        if len(best_improvements) >= num_top:
            break
        
        frame_idx = frame['idx']
        
        # Check if this frame is too close to any already selected frame
        is_too_close = False
        for used_idx in used_indices:
            if abs(frame_idx - used_idx) < min_frame_gap:
                is_too_close = True
                break
        
        if not is_too_close:
            best_improvements.append(frame)
            used_indices.add(frame_idx)
    
    print(f"  Selected {len(best_improvements)} top improvements (non-consecutive, min gap={min_frame_gap} frames)")
    
    # Add random frames from the remaining frames in the percentile window
    # Also ensure they are non-consecutive with respect to each other AND top improvements
    if num_random > 0 and len(frames_sorted) > num_top:
        remaining_frames = [f for f in frames_sorted if f not in best_improvements]
        
        # Filter remaining frames to exclude those too close to already selected frames
        filtered_remaining = []
        for frame in remaining_frames:
            frame_idx = frame['idx']
            is_too_close = False
            for used_idx in used_indices:
                if abs(frame_idx - used_idx) < min_frame_gap:
                    is_too_close = True
                    break
            if not is_too_close:
                filtered_remaining.append(frame)
        
        # Randomly select from filtered pool
        rng = np.random.RandomState(42)  # Fixed seed for reproducibility
        num_random_actual = min(num_random, len(filtered_remaining))
        
        if num_random_actual > 0:
            # Shuffle and select frames one by one, ensuring they're non-consecutive
            rng.shuffle(filtered_remaining)
            random_frames = []
            
            for frame in filtered_remaining:
                if len(random_frames) >= num_random_actual:
                    break
                
                frame_idx = frame['idx']
                is_too_close = False
                for used_idx in used_indices:
                    if abs(frame_idx - used_idx) < min_frame_gap:
                        is_too_close = True
                        break
                
                if not is_too_close:
                    random_frames.append(frame)
                    used_indices.add(frame_idx)
        else:
            random_frames = []
        
        # Combine top improvements + random frames
        frames_to_visualize = best_improvements + random_frames
        print(f"  Selected {len(best_improvements)} top improvements + {len(random_frames)} random frames (total: {len(frames_to_visualize)})")
        print(f"  All frames are non-consecutive (min gap={min_frame_gap} frames)")
    else:
        frames_to_visualize = best_improvements
        random_frames = []
        print(f"  Selected {len(frames_to_visualize)} top improvements")
    
    print(f"\n  Top 10 improvements:")
    for i, f in enumerate(best_improvements[:10]):
        print(f"    {i+1}. {f['imgname']}: {model1}={f[f'{model1}_error']:.1f}mm, {model2}={f[f'{model2}_error']:.1f}mm, delta={f['delta']:.1f}mm ({f['rel_delta']*100:.1f}%)")
    
    # Save frame list (all selected frames)
    output_path = os.path.join(output_dir, f'selected_frames_{dataset}_{model1}_vs_{model2}_{min_pct}_{max_pct}.txt')
    with open(output_path, 'w') as f:
        f.write(f"Selected frames for visualization: {len(frames_to_visualize)} total\n")
        f.write(f"  - Top {len(best_improvements)} improvements (sorted by relative delta)\n")
        if num_random > 0:
            f.write(f"  - {len(random_frames)} random frames from percentile window\n")
        f.write(f"Dataset: {dataset}\n")
        f.write(f"{model1.upper()} error percentile window: [{min_pct}, {max_pct})\n")
        f.write("="*80 + "\n\n")
        
        # Write top improvements first
        f.write("=== TOP IMPROVEMENTS ===\n\n")
        for i, frame in enumerate(best_improvements):
            f.write(f"{i+1}. {frame['imgname']}\n")
            f.write(f"   {model1.upper()} error: {frame[f'{model1}_error']:.2f} mm\n")
            f.write(f"   {model2.upper()} error: {frame[f'{model2}_error']:.2f} mm\n")
            f.write(f"   Absolute delta: {frame['delta']:.2f} mm\n")
            f.write(f"   Relative delta: {frame['rel_delta']*100:.2f}%\n\n")
        
        # Write random frames if any
        if num_random > 0 and len(random_frames) > 0:
            f.write("\n=== RANDOM FRAMES ===\n\n")
            for i, frame in enumerate(random_frames):
                f.write(f"{len(best_improvements) + i + 1}. {frame['imgname']}\n")
                f.write(f"   {model1.upper()} error: {frame[f'{model1}_error']:.2f} mm\n")
                f.write(f"   {model2.upper} error: {frame[f'{model2}_error']:.2f} mm\n")
                f.write(f"   Absolute delta: {frame['delta']:.2f} mm\n")
                f.write(f"   Relative delta: {frame['rel_delta']*100:.2f}%\n\n")
    
    print(f"  Saved frame list: {output_path}")
    
    # Visualize frames with skeleton overlays
    visualize_frame_skeletons(frames_to_visualize, results_model1, results_model2, dataset, output_dir, model1, model2)
    
    return frames_to_visualize

# // ...existing code...

def analyze_geometric_features(results_model1, results_model2, dataset, output_dir, model1='xy', model2='xycd'):
    """Analyze MPJPE stratified by geometric features that explain tail error patterns.
    
    This function examines four key dimensions that reveal *why* augmented inputs help:
    1. Bounding-box scale (2D person size → perspective strength)
    2. Foreshortening ratio (2D/3D bone length → distortion)
    3. Torso pitch (body inclination → self-occlusion)
    4. Ordinal margin (depth ambiguity → near-tie errors)
    
    For each feature, frames are stratified into quartiles and MPJPE improvements
    (Δp90, Δp95) are computed per bin to reveal structural patterns.
    
    Args:
        results_model1: Dict with model1 predictions and metadata
        results_model2: Dict with model2 predictions and metadata
        dataset: Dataset name
        output_dir: Directory to save analysis results
        model1: First model name (baseline)
        model2: Second model name (augmented)
    """
    print(f"\n{'='*60}")
    print(f"Geometric Feature Analysis: {dataset}")
    print(f"Comparing {model1.upper()} vs {model2.upper()}")
    print(f"{'='*60}")
    
    # Extract data
    pred_mm_1 = results_model1['pred_mm']
    pred_mm_2 = results_model2['pred_mm']
    gt_mm = results_model1['gt_mm']
    input_2d = results_model1.get('input_2d_keypoints', None)
    per_frame_mpjpe_1 = results_model1['per_frame_mpjpe']
    per_frame_mpjpe_2 = results_model2['per_frame_mpjpe']
    
    if input_2d is None:
        print(f"  ✗ 2D keypoints not available, cannot compute scale/foreshortening")
        return None
    
    N = min(len(pred_mm_1), len(pred_mm_2), len(gt_mm), len(input_2d))
    print(f"  Analyzing {N} frames")
    
    # Compute geometric features
    print(f"\n  Computing geometric features...")
    
    # 1. Bounding-box scale (pixels)
    bbox_scales = bbox_scale_2d(input_2d[:N])
    print(f"    ✓ Bounding-box scale (range: [{np.min(bbox_scales):.1f}, {np.max(bbox_scales):.1f}] px)")
    
    # 2. Foreshortening ratio (2D/3D bone length)
    # Use GT 3D poses for foreshortening computation (true geometry)
    foreshort_ratios = foreshortening_ratio(input_2d[:N], gt_mm[:N])
    print(f"    ✓ Foreshortening ratio (range: [{np.min(foreshort_ratios):.3f}, {np.max(foreshort_ratios):.3f}])")
    
    # 3. Torso pitch (degrees)
    pitch_angles = torso_pitch_deg(gt_mm[:N])
    print(f"    ✓ Torso pitch (range: [{np.min(pitch_angles):.1f}, {np.max(pitch_angles):.1f}]°)")
    
    # 4. Ordinal margin (depth ambiguity fraction)
    margins = ordinal_margin(gt_mm[:N], thresh_mm=100)
    print(f"    ✓ Ordinal margin (range: [{np.min(margins):.3f}, {np.max(margins):.3f}])")
    
    # Compute deltas
    delta_mpjpe = per_frame_mpjpe_2[:N] - per_frame_mpjpe_1[:N]
    
    # Stratify by each feature (quartiles)
    features = {
        'bbox_scale': (bbox_scales, 'Bounding-Box Scale (px)', 'px'),
        'foreshortening': (foreshort_ratios, 'Foreshortening Ratio', ''),
        'torso_pitch': (pitch_angles, 'Torso Pitch (degrees)', '°'),
        'ordinal_margin': (margins, 'Ordinal Margin (depth ambiguity)', ''),
    }
    
    analysis_results = {}
    
    for feature_key, (feature_values, feature_label, unit) in features.items():
        print(f"\n  {'='*60}")
        print(f"  Analyzing: {feature_label}")
        print(f"  {'='*60}")
        
        # Compute quartiles
        q25 = np.percentile(feature_values, 25)
        q50 = np.percentile(feature_values, 50)
        q75 = np.percentile(feature_values, 75)
        
        # Create quartile masks
        q1_mask = feature_values <= q25
        q2_mask = (feature_values > q25) & (feature_values <= q50)
        q3_mask = (feature_values > q50) & (feature_values <= q75)
        q4_mask = feature_values > q75
        
        quartile_stats = {}
        
        print(f"\n  Quartile breakdown:")
        for q_name, mask, q_label in [
            ('Q1', q1_mask, 'Q1 (lowest 25%)'),
            ('Q2', q2_mask, 'Q2 (25-50%)'),
            ('Q3', q3_mask, 'Q3 (50-75%)'),
            ('Q4', q4_mask, 'Q4 (highest 25%)'),
        ]:
            if np.sum(mask) == 0:
                continue
            
            # Feature range
            feat_min = np.min(feature_values[mask])
            feat_max = np.max(feature_values[mask])
            feat_mean = np.mean(feature_values[mask])
            
            # MPJPE statistics for each model
            mpjpe1 = per_frame_mpjpe_1[:N][mask]
            mpjpe2 = per_frame_mpjpe_2[:N][mask]
            deltas = delta_mpjpe[mask]
            
            # Filter out NaN values (frames with all joints filtered by confidence threshold)
            valid_mask = np.isfinite(mpjpe1) & np.isfinite(mpjpe2) & np.isfinite(deltas)
            if np.sum(valid_mask) == 0:
                print(f"    {q_label:<20}: N={np.sum(mask):>5} (all NaN, skipped)")
                continue
            
            mpjpe1_valid = mpjpe1[valid_mask]
            mpjpe2_valid = mpjpe2[valid_mask]
            deltas_valid = deltas[valid_mask]
            
            # Compute tail metrics
            p90_1 = np.percentile(mpjpe1_valid, 90)
            p90_2 = np.percentile(mpjpe2_valid, 90)
            delta_p90 = p90_2 - p90_1
            
            p95_1 = np.percentile(mpjpe1_valid, 95)
            p95_2 = np.percentile(mpjpe2_valid, 95)
            delta_p95 = p95_2 - p95_1
            
            mean1 = np.mean(mpjpe1_valid)
            mean2 = np.mean(mpjpe2_valid)
            median1 = np.median(mpjpe1_valid)
            median2 = np.median(mpjpe2_valid)
            delta_mean = mean2 - mean1
            delta_median = median2 - median1
            
            quartile_stats[q_name] = {
                'n_frames': np.sum(mask),
                'n_valid_frames': np.sum(valid_mask),
                'feature_range': (feat_min, feat_max),
                'feature_mean': feat_mean,
                'mean_mpjpe_1': mean1,
                'mean_mpjpe_2': mean2,
                'median_mpjpe_1': median1,
                'median_mpjpe_2': median2,
                'p90_1': p90_1,
                'p90_2': p90_2,
                'delta_p90': delta_p90,
                'p95_1': p95_1,
                'p95_2': p95_2,
                'delta_p95': delta_p95,
                'mean_delta': delta_mean,
                'median_delta': delta_median,
            }
            
            print(f"    {q_label:<20}: N={np.sum(valid_mask):>5} valid (of {np.sum(mask):>5}), "
                  f"range=[{feat_min:>7.2f}, {feat_max:>7.2f}]{unit}")
            print(f"      {model1.upper():>4}: mean={mean1:>6.1f}mm, median={median1:>6.1f}mm, "
                  f"p90={p90_1:>6.1f}mm, p95={p95_1:>6.1f}mm")
            print(f"      {model2.upper():>4}: mean={mean2:>6.1f}mm, median={median2:>6.1f}mm, "
                  f"p90={p90_2:>6.1f}mm, p95={p95_2:>6.1f}mm")
            print(f"      Δ({model2.upper()}-{model1.upper()}): mean={delta_mean:>+6.1f}mm ({(delta_mean/mean1*100):>+5.1f}%), "
                  f"median={delta_median:>+6.1f}mm ({(delta_median/median1*100):>+5.1f}%), "
                  f"p90={delta_p90:>+7.2f}mm, p95={delta_p95:>+7.2f}mm")
        
        analysis_results[feature_key] = {
            'label': feature_label,
            'unit': unit,
            'quartiles': quartile_stats,
        }
    
    # Save detailed results
    output_path = os.path.join(output_dir, f'geometric_features_{dataset}_{model1}_vs_{model2}.txt')
    with open(output_path, 'w') as f:
        f.write(f"Geometric Feature Analysis: {dataset}\n")
        f.write(f"Comparing {model1.upper()} vs {model2.upper()}\n")
        f.write(f"Total frames: {N}\n")
        f.write("="*80 + "\n\n")
        
        f.write("Four Key Dimensions:\n")
        f.write("  1. Bounding-box scale → perspective strength (camera distance)\n")
        f.write("  2. Foreshortening ratio → projection distortion (limb orientation)\n")
        f.write("  3. Torso pitch → self-occlusion (body inclination)\n")
        f.write("  4. Ordinal margin → depth ambiguity (near-tie pairs)\n\n")
        
        for feature_key, feature_data in analysis_results.items():
            label = feature_data['label']
            unit = feature_data['unit']
            quartiles = feature_data['quartiles']
            
            f.write("="*80 + "\n")
            f.write(f"{label}\n")
            f.write("="*80 + "\n\n")
            
            for q_name in ['Q1', 'Q2', 'Q3', 'Q4']:
                if q_name not in quartiles:
                    continue
                
                stats = quartiles[q_name]
                f.write(f"{q_name} (N={stats['n_frames']} frames):\n")
                f.write(f"  Feature range: [{stats['feature_range'][0]:.2f}, {stats['feature_range'][1]:.2f}]{unit}\n")
                f.write(f"  Feature mean: {stats['feature_mean']:.2f}{unit}\n\n")
                
                f.write(f"  {model1.upper()} MPJPE:\n")
                f.write(f"    Mean: {stats['mean_mpjpe_1']:.2f} mm\n")
                f.write(f"    p90: {stats['p90_1']:.2f} mm\n")
                f.write(f"    p95: {stats['p95_1']:.2f} mm\n\n")
                
                f.write(f"  {model2.upper()} MPJPE:\n")
                f.write(f"    Mean: {stats['mean_mpjpe_2']:.2f} mm\n")
                f.write(f"    p90: {stats['p90_2']:.2f} mm\n")
                f.write(f"    p95: {stats['p95_2']:.2f} mm\n\n")
                
                f.write(f"  Tail Error Improvement:\n")
                f.write(f"    Δp90: {stats['delta_p90']:+.2f} mm\n")
                f.write(f"    Δp95: {stats['delta_p95']:+.2f} mm\n")
                f.write(f"    Mean Δ: {stats['mean_delta']:+.2f} mm\n\n")
            
            # Compute trend across quartiles
            delta_p90_trend = [quartiles[q]['delta_p90'] for q in ['Q1', 'Q2', 'Q3', 'Q4'] if q in quartiles]
            if len(delta_p90_trend) >= 3:
                # Check if monotonic improvement/degradation
                increasing = all(delta_p90_trend[i] <= delta_p90_trend[i+1] for i in range(len(delta_p90_trend)-1))
                decreasing = all(delta_p90_trend[i] >= delta_p90_trend[i+1] for i in range(len(delta_p90_trend)-1))
                
                f.write(f"  Trend Summary:\n")
                if increasing:
                    f.write(f"    ✓ Monotonic: {model2.upper()} helps MORE as {label.lower()} increases\n")
                elif decreasing:
                    f.write(f"    ✓ Monotonic: {model2.upper()} helps MORE as {label.lower()} decreases\n")
                else:
                    f.write(f"    • Non-monotonic trend across quartiles\n")
                
                f.write(f"    Δp90 range: [{min(delta_p90_trend):+.2f}, {max(delta_p90_trend):+.2f}] mm\n")
                f.write(f"    Δp90 span: {max(delta_p90_trend) - min(delta_p90_trend):.2f} mm\n\n")
    
    print(f"\n  Saved geometric feature analysis: {output_path}")
    
    # Print summary
    print(f"\n  {'='*60}")
    print(f"  Summary: Key Findings")
    print(f"  {'='*60}")
    
    for feature_key, feature_data in analysis_results.items():
        label = feature_data['label']
        quartiles = feature_data['quartiles']
        
        # Compare Q1 vs Q4
        if 'Q1' in quartiles and 'Q4' in quartiles:
            delta_p90_q1 = quartiles['Q1']['delta_p90']
            delta_p90_q4 = quartiles['Q4']['delta_p90']
            diff = delta_p90_q4 - delta_p90_q1
            
            print(f"\n  {label}:")
            print(f"    Q1 Δp90: {delta_p90_q1:+.2f} mm")
            print(f"    Q4 Δp90: {delta_p90_q4:+.2f} mm")
            print(f"    Q4-Q1 span: {diff:+.2f} mm")
            
            if abs(diff) > 5.0:
                direction = "higher" if diff > 0 else "lower"
                print(f"    → {model2.upper()} helps more at {direction} {label.lower()}")
    
    return analysis_results

def find_and_visualize_contrast_frames(results_model1, results_model2, dataset, output_dir, model1='xy', model2='xycd', contrast_threshold=150.0, num_frames=50, contrast_mode='base'):
    """Find and visualize frames where model2 shows large improvement over model1,
    but model2 error is still ≤ contrast_threshold (good absolute performance).
    
    Args:
        contrast_mode: 'base' for original visualization, 'with_casp' to include CASP depth sampling
    """
    print(f"\n{'='*60}")
    print(f"Finding contrast frames for {dataset}")
    print(f"Visualization mode: {contrast_mode}")
    print(f"{'='*60}")
    print(f"Criteria:")
    print(f"  1. {model2.upper()} improves over {model1.upper()} (negative delta)")
    print(f"  2. {model2.upper()} absolute error ≤ {contrast_threshold:.0f}mm")

    # Safely extract arrays and imgnames
    errors1 = np.array(results_model1.get('per_frame_mpjpe', []), dtype=float)
    errors2 = np.array(results_model2.get('per_frame_mpjpe', []), dtype=float)
    imgnames_raw = results_model1.get('imgnames', results_model2.get('imgnames', []))
    imgnames = [str(x) for x in list(imgnames_raw)]

    # Replace NaNs/infs so they don't pass filters
    errors1 = np.nan_to_num(errors1, nan=np.inf, posinf=np.inf, neginf=np.inf)
    errors2 = np.nan_to_num(errors2, nan=np.inf, posinf=np.inf, neginf=np.inf)

    # Align lengths
    n = min(len(errors1), len(errors2), len(imgnames))
    errors1 = errors1[:n]
    errors2 = errors2[:n]
    imgnames = imgnames[:n]

    # Filters
    good_absolute_mask = errors2 <= contrast_threshold
    improvement_mask = errors2 < errors1
    contrast_mask = good_absolute_mask & improvement_mask

    print(f"\n  Filtering results:")
    print(f"    Total frames: {n}")
    print(f"    Frames where {model2.upper()} ≤ {contrast_threshold:.0f}mm: {np.sum(good_absolute_mask)}")
    print(f"    Frames where {model2.upper()} improves: {np.sum(improvement_mask)}")
    print(f"    Frames meeting both criteria: {np.sum(contrast_mask)}")

    if np.sum(contrast_mask) == 0:
        print(f"  ✗ No frames found meeting contrast criteria")
        return []

    # Build safe frame list
    frames = []
    for i in range(n):
        if not contrast_mask[i]:
            continue
        delta = float(errors2[i] - errors1[i])
        rel_delta = float(delta / (errors1[i] + 1e-8))
        improvement_magnitude = float(errors1[i] - errors2[i])
        frames.append({
            'idx': int(i),
            'imgname': imgnames[i],
            f'{model1}_error': float(errors1[i]),
            f'{model2}_error': float(errors2[i]),
            'delta': delta,
            'rel_delta': rel_delta,
            'improvement': improvement_magnitude
        })

    # Sort by improvement magnitude desc
    frames_sorted = sorted(frames, key=lambda x: x['improvement'], reverse=True)

    # Select non-consecutive frames (min gap)
    min_frame_gap = 10
    selected_frames = []
    used_indices = set()
    for frame in frames_sorted:
        if len(selected_frames) >= num_frames:
            break
        idx = frame['idx']
        if any(abs(idx - u) < min_frame_gap for u in used_indices):
            continue
        selected_frames.append(frame)
        used_indices.add(idx)

    print(f"\n  Selected {len(selected_frames)} contrast frames (min gap={min_frame_gap})")
    for i, f in enumerate(selected_frames[:10]):
        print(f"    {i+1}. {f['imgname']}: {model1}={f[f'{model1}_error']:.1f}mm, {model2}={f[f'{model2}_error']:.1f}mm, improvement={f['improvement']:.1f}mm")

    # Save frame list
    output_path = os.path.join(output_dir, f'contrast_frames_{dataset}_{model1}_vs_{model2}_threshold{contrast_threshold:.0f}mm.txt')
    with open(output_path, 'w') as fh:
        fh.write(f"Contrast frames for visualization: {len(selected_frames)} total\n")
        fh.write(f"Dataset: {dataset}\n")
        fh.write(f"Criteria: model2 improves and model2 <= {contrast_threshold:.0f}mm\n")
        fh.write("="*80 + "\n\n")
        for i, frame in enumerate(selected_frames):
            fh.write(f"{i+1}. {frame['imgname']}\n")
            fh.write(f"   {model1.upper()} error: {frame[f'{model1}_error']:.2f} mm\n")
            fh.write(f"   {model2.upper()} error: {frame[f'{model2}_error']:.2f} mm\n")
            fh.write(f"   Improvement: {frame['improvement']:.2f} mm\n")
            fh.write(f"   Delta: {frame['delta']:.2f} mm\n\n")
    print(f"  Saved contrast frame list: {output_path}")

    # Load CASP data if in with_casp mode
    casp_data = None
    if contrast_mode == 'with_casp':
        casp_data = load_casp_data_for_visualization(dataset)
    
    # Visualize with robust skeleton function
    visualize_frame_skeletons(selected_frames, results_model1, results_model2, dataset, output_dir, model1, model2, casp_data=casp_data)
    return selected_frames


def load_casp_data_for_visualization(dataset):
    """Load CASP precomputed data for enhanced visualization."""
    casp_paths = {
        '3dhp': "/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_3dhp_test_all_v6_casp.npz",
        # '3dpw': "/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_3dpw_all_v6_gtdav.npz",
        # 'fit3d': "/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_fit3d_all_v6_casp.npz",
    }
    
    if dataset not in casp_paths:
        print(f"  ✗ No CASP data path configured for {dataset}")
        return None
    
    casp_path = casp_paths[dataset]
    if not os.path.exists(casp_path):
        print(f"  ✗ CASP data file not found: {casp_path}")
        return None
    
    print(f"  Loading CASP data from: {casp_path}")
    casp_npz = np.load(casp_path, allow_pickle=True)
    
    casp_data = {
        'imgnames': casp_npz['imgname'],
        'casp_descriptors': casp_npz.get('summary_casp_descriptor_10d', None),
        'dav_depths': casp_npz.get('predicted_da_depth', None),
        'keypoints_2d': casp_npz.get('predicted_keypoints', None),
        'confidence': casp_npz.get('predicted_keypoints_score', None),
    }
    
    # Build imgname to index mapping for quick lookup
    casp_data['imgname_to_idx'] = {name: i for i, name in enumerate(casp_data['imgnames'])}
    print(f"  ✓ Loaded CASP data for {len(casp_data['imgnames'])} frames")
    
    return casp_data

def visualize_frame_skeletons(frames, results_model1, results_model2, dataset, output_dir, model1, model2, casp_data=None):
    """Visualize selected frames with 3D pose skeleton overlays."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import cv2
    except ImportError as e:
        print(f"  Warning: Could not import visualization libraries: {e}")
        print(f"  Skipping frame visualizations")
        return
    
    # Create subdirectory for frame visualizations
    # Use a unique directory name to avoid conflicts
    if model1 == model2:
        # Single model visualization (for tier-based analysis)
        frames_dir = output_dir
    else:
        # Model comparison visualization
        frames_dir = os.path.join(output_dir, f'frames_{dataset}_{model1}_vs_{model2}')
    
    # Clean up existing frames directory if it exists (only for comparison mode)
    if model1 != model2 and os.path.exists(frames_dir):
        import shutil
        shutil.rmtree(frames_dir)
        print(f"  Cleaned up existing frames directory: {frames_dir}")
    
    os.makedirs(frames_dir, exist_ok=True)
    
    # Define skeleton structure
    keypoint_index = {
        'root': 0, 'right_hip': 1, 'right_knee': 2, 'right_foot': 3,
        'left_hip': 4, 'left_knee': 5, 'left_foot': 6, 'spine': 7,
        'thorax': 8, 'neck_base': 9, 'head': 10, 'left_shoulder': 11,
        'left_elbow': 12, 'left_wrist': 13, 'right_shoulder': 14,
        'right_elbow': 15, 'right_wrist': 16
    }
    
    skeleton_links = [
        ('root', 'left_hip'), ('left_hip', 'left_knee'), ('left_knee', 'left_foot'),
        ('root', 'right_hip'), ('right_hip', 'right_knee'), ('right_knee', 'right_foot'),
        ('root', 'spine'), ('spine', 'thorax'), ('thorax', 'neck_base'),
        ('neck_base', 'head'), ('thorax', 'left_shoulder'), ('left_shoulder', 'left_elbow'),
        ('left_elbow', 'left_wrist'), ('thorax', 'right_shoulder'), ('right_shoulder', 'right_elbow'),
        ('right_elbow', 'right_wrist')
    ]
    
    # Get image base paths for different datasets
    image_base_paths = {
        '3dhp': '/srv/essa-lab/flash3/nwarner30/pose_estimation/data/test_images_raw',
        'h36m': '/srv/essa-lab/flash3/nwarner30/pose_estimation/data/h36m/images',  # Placeholder
        '3dpw': '/srv/essa-lab/flash3/nwarner30/pose_estimation/3dpw_data/imageFiles',
    }
    
    image_base_path = image_base_paths.get(dataset, '')
    
    # Extract prediction and GT data
    pred_mm_1 = results_model1['pred_mm']
    pred_mm_2 = results_model2['pred_mm']
    gt_mm = results_model1['gt_mm']
    input_2d_keypoints = results_model1.get('input_2d_keypoints', None)
    
    # Retrieve preprocessed DAV depths from results (already root-centered and in mm)
    dav_mm = results_model1.get('dav_mm', None)
    if dav_mm is not None:
        print(f"  ✓ DAV depths available for visualization: {dav_mm.shape}")
    else:
        print(f"  ✗ DAV depths not available in results")
    
    visualized_count = 0
    for i, frame in enumerate(frames):
        imgname = frame['imgname']
        frame_idx = frame['idx']
        
        if frame_idx >= len(pred_mm_1) or frame_idx >= len(pred_mm_2) or frame_idx >= len(gt_mm):
            continue
        
        # Get 3D poses for this frame
        pred_pose_1 = pred_mm_1[frame_idx]  # (17, 3) in mm
        pred_pose_2 = pred_mm_2[frame_idx]  # (17, 3) in mm
        gt_pose = gt_mm[frame_idx]  # (17, 3) in mm
        
        # Try to load the image
        img_rgb = None
        if dataset == '3dhp':
            # Extract camera from imgname (e.g., "TS1_004023.jpg" -> "TS1")
            camera = imgname.split('_')[0]
            
            # Try with and without "_img_" prefix
            if '_img_' not in imgname:
                parts = imgname.split('_')
                if len(parts) == 2:
                    imgname_with_img = f"{parts[0]}_img_{parts[1]}"
                else:
                    imgname_with_img = imgname
            else:
                imgname_with_img = imgname
            
            possible_paths = [
                os.path.join(image_base_path, camera, imgname_with_img),
                os.path.join(image_base_path, camera, imgname),
            ]
            
            for img_path in possible_paths:
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    if img is not None:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        break
        
        # elif dataset == '3dpw':
        #     # 3DPW format in data: "downtown_windowShopping_00_participant0_frame01503.jpg"
        #     # 3DPW format on disk: "downtown_windowShopping_00_image_01503.jpg"
        #     # Pattern: {location}_{action}_{number}_participant{N}_frame{XXXXX}.jpg -> {location}_{action}_{number}_image_{XXXXX}.jpg
        #     parts = imgname.split('_')
        #     ipdb.set_trace()
        #     if len(parts) >= 5 and parts[-2].startswith('frame'):
        #         # Extract components
        #         sequence_name = '_'.join(parts[:3])  # e.g., "downtown_windowShopping_00"
        #         frame_number = parts[-1].replace('frame', '').replace('.jpg', '')  # e.g., "01503"
                
        #         # Construct actual filename on disk
        #         img_path = os.path.join(image_base_path, sequence_name, f"{sequence_name}_image_{frame_number}.jpg")
                
        #         if os.path.exists(img_path):
        #             img = cv2.imread(img_path)
        #             if img is not None:
        #                 img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        elif dataset == '3dpw':
            # 3DPW format in data: "downtown_windowShopping_00_participant0_frame01503.jpg"
            # 3DPW format on disk: "downtown_windowShopping_00_image_01503.jpg"
            # Pattern: {location}_{action}_{number}_participant{N}_frame{XXXXX}.jpg -> {location}_{action}_{number}_image_{XXXXX}.jpg
            parts = imgname.split('_')
            
            # Find where 'frame' starts in the parts
            frame_part_idx = None
            for idx, part in enumerate(parts):
                if part.startswith('frame') or (idx > 0 and parts[idx-1].startswith('participant')):
                    frame_part_idx = idx
                    break
            
            if frame_part_idx is not None:
                # Everything before the frame part is the sequence name
                sequence_name = '_'.join(parts[:frame_part_idx-1])  # Exclude 'participantX' part
                
                # Extract frame number from the last part (e.g., "frame01503.jpg" -> "01503")
                frame_str = parts[-1]  # e.g., "frame01503.jpg"
                frame_number = frame_str.replace('frame', '').replace('.jpg', '')  # "01503"
                
                # Construct actual imgname on disk
                img_path = os.path.join(image_base_path, sequence_name, f"{sequence_name}_image_{frame_number}.jpg")
                
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    if img is not None:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        elif dataset == 'h36m':
            # TODO: Implement image loading for H36M
            pass
            
        # Create visualization (with or without image)
        if img_rgb is not None:
            # Layout depends on CASP mode
            if casp_data is not None:
                # Enhanced layout with CASP: 3 rows (4 input panels + 6 3D views + 4 CASP panels)
                fig = plt.figure(figsize=(28, 15))
                num_rows = 3
            else:
                # Standard layout: 2 rows (4 input panels + 6 3D views)
                fig = plt.figure(figsize=(28, 10))
                num_rows = 2
            
            # Top row: 4 input panels
            # Use colspan to evenly distribute: [1.5, 1.5, 1.5, 1.5] spans across 6 columns
            
            # Compute cropped/zoomed bbox for panels 1 & 2 (60% padding)
            img_cropped_wide = img_rgb
            crop_offset_x = 0
            crop_offset_y = 0
            
            if input_2d_keypoints is not None and frame_idx < len(input_2d_keypoints):
                input_2d_pose = input_2d_keypoints[frame_idx]  # (17, 2)
                
                # Compute bbox around keypoints with 60% padding
                x_min, y_min = input_2d_pose.min(axis=0)
                x_max, y_max = input_2d_pose.max(axis=0)
                
                # Make bbox square
                width = x_max - x_min
                height = y_max - y_min
                size = max(width, height)
                
                # Add 60% padding
                padding = size * 0.60
                
                # Center the square bbox
                cx = (x_min + x_max) / 2
                cy = (y_min + y_max) / 2
                half_size = (size + 2 * padding) / 2
                
                crop_x_min = max(0, int(cx - half_size))
                crop_x_max = min(img_rgb.shape[1], int(cx + half_size))
                crop_y_min = max(0, int(cy - half_size))
                crop_y_max = min(img_rgb.shape[0], int(cy + half_size))
                
                # Validate crop bounds before cropping
                if crop_x_max > crop_x_min and crop_y_max > crop_y_min:
                    # Crop image
                    img_cropped_wide = img_rgb[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
                    crop_offset_x = crop_x_min
                    crop_offset_y = crop_y_min
                else:
                    # Invalid crop, use full image
                    img_cropped_wide = img_rgb
            
            # Panel 1: Original image (cols 0-1.5) - zoomed with 60% padding
            ax1 = plt.subplot2grid((num_rows, 6), (0, 0), colspan=1)
            ax1.imshow(img_cropped_wide)
            ax1.set_title("Original Image", fontsize=10)
            ax1.axis('off')
            
            # Panel 2: Image with 2D pose overlay (cols 1.5-3) - zoomed with 60% padding
            ax2 = plt.subplot2grid((num_rows, 6), (0, 1), colspan=2)
            ax2.imshow(img_cropped_wide)
            
            # Draw 2D skeleton if available
            if input_2d_keypoints is not None and frame_idx < len(input_2d_keypoints):
                input_2d_pose = input_2d_keypoints[frame_idx]  # (17, 2)
                
                # Adjust keypoint coordinates for cropped image
                input_2d_pose_cropped = input_2d_pose.copy()
                input_2d_pose_cropped[:, 0] -= crop_offset_x
                input_2d_pose_cropped[:, 1] -= crop_offset_y
                
                # Draw 2D skeleton links
                for link in skeleton_links:
                    start_name, end_name = link
                    start_idx = keypoint_index[start_name]
                    end_idx = keypoint_index[end_name]
                    
                    start_pt = input_2d_pose_cropped[start_idx]
                    end_pt = input_2d_pose_cropped[end_idx]
                    
                    ax2.plot([start_pt[0], end_pt[0]], [start_pt[1], end_pt[1]], 
                            'r-', linewidth=2, alpha=0.7)
                
                # Draw 2D keypoints
                ax2.scatter(input_2d_pose_cropped[:, 0], input_2d_pose_cropped[:, 1], 
                           c='yellow', s=30, edgecolors='red', linewidths=1.5, zorder=3)
                
                ax2.set_title("Image + 2D Pose", fontsize=10)
            else:
                ax2.set_title("Image (2D data N/A)", fontsize=10)
            
            ax2.axis('off')
            
            # Panel 3: 2D Keypoint Confidence Visualization (cols 3-4.5)
            ax3 = plt.subplot2grid((num_rows, 6), (0, 3), colspan=1)
            
            # Draw 2D skeleton with confidence coloring if available
            if input_2d_keypoints is not None and frame_idx < len(input_2d_keypoints):
                input_2d_pose = input_2d_keypoints[frame_idx]  # (17, 2)
                
                # Compute bbox around keypoints with 25% padding
                x_min, y_min = input_2d_pose.min(axis=0)
                x_max, y_max = input_2d_pose.max(axis=0)
                
                # Make bbox square
                width = x_max - x_min
                height = y_max - y_min
                size = max(width, height)
                
                # Add 25% padding
                padding = size * 0.25
                
                # Center the square bbox
                cx = (x_min + x_max) / 2
                cy = (y_min + y_max) / 2
                half_size = (size + 2 * padding) / 2
                
                crop_x_min = max(0, int(cx - half_size))
                crop_x_max = min(img_rgb.shape[1], int(cx + half_size))
                crop_y_min = max(0, int(cy - half_size))
                crop_y_max = min(img_rgb.shape[0], int(cy + half_size))
                
                # Validate crop bounds before cropping
                if crop_x_max > crop_x_min and crop_y_max > crop_y_min:
                    # Crop image
                    img_cropped = img_rgb[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
                    ax3.imshow(img_cropped)
                else:
                    # Invalid crop, use full image
                    ax3.imshow(img_rgb)
                
                # Adjust keypoint coordinates for cropped image
                input_2d_pose_cropped = input_2d_pose.copy()
                input_2d_pose_cropped[:, 0] -= crop_x_min
                input_2d_pose_cropped[:, 1] -= crop_y_min
                
                # Get visibility scores if available
                if results_model1.get('visibility_scores') is not None and frame_idx < len(results_model1['visibility_scores']):
                    vis_scores = results_model1['visibility_scores'][frame_idx]  # (17,)
                    
                    # Draw skeleton links colored by average confidence of endpoints
                    for link in skeleton_links:
                        start_name, end_name = link
                        start_idx = keypoint_index[start_name]
                        end_idx = keypoint_index[end_name]
                        
                        start_pt = input_2d_pose_cropped[start_idx]
                        end_pt = input_2d_pose_cropped[end_idx]
                        
                        # Average confidence of endpoints
                        avg_conf = (vis_scores[start_idx] + vis_scores[end_idx]) / 2.0
                        
                        # Color: white (conf=0.5) to green (conf=1.0)
                        # Map [0.5, 1.0] -> [0.0, 1.0] for colormap
                        color_val = (avg_conf - 0.5) / 0.5 if avg_conf >= 0.5 else 0.0
                        color_val = np.clip(color_val, 0.0, 1.0)
                        color = plt.cm.RdYlGn(color_val)
                        ax3.plot([start_pt[0], end_pt[0]], [start_pt[1], end_pt[1]], 
                                color=color, linewidth=2, alpha=0.8)
                    
                    # Draw keypoints colored by confidence
                    for joint_idx in range(17):
                        pt = input_2d_pose_cropped[joint_idx]
                        conf = vis_scores[joint_idx]
                        color_val = (conf - 0.5) / 0.5 if conf >= 0.5 else 0.0
                        color_val = np.clip(color_val, 0.0, 1.0)
                        color = plt.cm.RdYlGn(color_val)
                        ax3.scatter(pt[0], pt[1], c=[color], s=80, edgecolors='black', 
                                   linewidths=1.5, zorder=3)
                    
                    # Add colorbar legend
                    from matplotlib.cm import ScalarMappable
                    from matplotlib.colors import Normalize
                    sm = ScalarMappable(cmap=plt.cm.RdYlGn, norm=Normalize(vmin=0.5, vmax=1.0))
                    sm.set_array([])
                    cbar = plt.colorbar(sm, ax=ax3, fraction=0.046, pad=0.04)
                    cbar.set_label('Confidence', fontsize=8)
                    cbar.ax.tick_params(labelsize=7)
                else:
                    # No confidence scores, draw in default color
                    for link in skeleton_links:
                        start_name, end_name = link
                        start_idx = keypoint_index[start_name]
                        end_idx = keypoint_index[end_name]
                        
                        start_pt = input_2d_pose_cropped[start_idx]
                        end_pt = input_2d_pose_cropped[end_idx]
                        ax3.plot([start_pt[0], end_pt[0]], [start_pt[1], end_pt[1]], 
                                'gray', linewidth=2, alpha=0.7)
                    
                    ax3.scatter(input_2d_pose_cropped[:, 0], input_2d_pose_cropped[:, 1], 
                               c='yellow', s=30, edgecolors='black', linewidths=1.5, zorder=3)
                
                ax3.set_title("2D Confidence (0.5=White, 1.0=Green)", fontsize=10)
            else:
                ax3.imshow(img_rgb)
                ax3.set_title("2D Confidence (N/A)", fontsize=10)
            
            ax3.axis('off')
            
            # Panel 4: DAV Input Depth Visualization (cols 4.5-6)
            ax4 = plt.subplot2grid((num_rows, 6), (0, 4), colspan=2)
            
            # Draw 2D skeleton colored by DAV input depth if available
            if dav_mm is not None and input_2d_keypoints is not None and frame_idx < len(input_2d_keypoints):
                input_2d_pose = input_2d_keypoints[frame_idx]  # (17, 2)
                
                # Compute bbox around keypoints with 25% padding (same as confidence panel)
                x_min, y_min = input_2d_pose.min(axis=0)
                x_max, y_max = input_2d_pose.max(axis=0)
                
                # Make bbox square
                width = x_max - x_min
                height = y_max - y_min
                size = max(width, height)
                
                # Add 25% padding
                padding = size * 0.25
                
                # Center the square bbox
                cx = (x_min + x_max) / 2
                cy = (y_min + y_max) / 2
                half_size = (size + 2 * padding) / 2
                
                crop_x_min = max(0, int(cx - half_size))
                crop_x_max = min(img_rgb.shape[1], int(cx + half_size))
                crop_y_min = max(0, int(cy - half_size))
                crop_y_max = min(img_rgb.shape[0], int(cy + half_size))
                
                # Validate crop bounds before cropping
                if crop_x_max > crop_x_min and crop_y_max > crop_y_min:
                    # Crop image
                    img_cropped = img_rgb[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
                    ax4.imshow(img_cropped)
                else:
                    # Invalid crop, use full image
                    ax4.imshow(img_rgb)
                
                # Adjust keypoint coordinates for cropped image
                input_2d_pose_cropped = input_2d_pose.copy()
                input_2d_pose_cropped[:, 0] -= crop_x_min
                input_2d_pose_cropped[:, 1] -= crop_y_min
                
                # Use DAV input depth predictions (root-relative, in mm)
                dav_depths = dav_mm[frame_idx]  # (17,) - DAV predicted Z coordinates in mm
                
                # Normalize depths for colormap (root = 0)
                depth_range = max(abs(dav_depths.min()), abs(dav_depths.max()))
                if depth_range > 0:
                    # Map to [0, 1] range: closer (negative) = blue, root (0) = white, further (positive) = red
                    norm_depths = (dav_depths + depth_range) / (2 * depth_range)
                else:
                    norm_depths = np.ones_like(dav_depths) * 0.5
                
                # Draw skeleton links colored by average depth
                for link in skeleton_links:
                    start_name, end_name = link
                    start_idx = keypoint_index[start_name]
                    end_idx = keypoint_index[end_name]
                    
                    start_pt = input_2d_pose_cropped[start_idx]
                    end_pt = input_2d_pose_cropped[end_idx]
                    
                    # Average depth
                    avg_depth_norm = (norm_depths[start_idx] + norm_depths[end_idx]) / 2.0
                    
                    # Color: blue (closer) -> white (root) -> red (further)
                    color = plt.cm.RdBu_r(avg_depth_norm)
                    ax4.plot([start_pt[0], end_pt[0]], [start_pt[1], end_pt[1]], 
                            color=color, linewidth=2, alpha=0.8)
                
                # Draw keypoints colored by depth
                for joint_idx in range(17):
                    pt = input_2d_pose_cropped[joint_idx]
                    depth_norm = norm_depths[joint_idx]
                    color = plt.cm.RdBu_r(depth_norm)
                    ax4.scatter(pt[0], pt[1], c=[color], s=80, edgecolors='black', 
                               linewidths=1.5, zorder=3)
                
                # Add colorbar legend for depth
                from matplotlib.cm import ScalarMappable
                from matplotlib.colors import Normalize
                # Show depth range in mm
                sm = ScalarMappable(cmap=plt.cm.RdBu_r, norm=Normalize(vmin=-depth_range, vmax=depth_range))
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax4, fraction=0.046, pad=0.04)
                cbar.set_label('DAV Depth (mm)', fontsize=8)
                cbar.ax.tick_params(labelsize=7)
                
                ax4.set_title("DAV Input Depth (Blue=Closer, White=Root, Red=Further)", fontsize=10)
            else:
                ax4.imshow(img_rgb)
                ax4.set_title("DAV Input Depth (N/A)", fontsize=10)
            
            ax4.axis('off')
            
            # Bottom row: 3D views - 3 viewing angles, each with both models side-by-side
            # Format: [Front-XY, Front-XYCD | Side(L)-XY, Side(L)-XYCD | Auto-XY, Auto-XYCD]
            # Front view (elevation=45°)
            ax3_xy_front = plt.subplot2grid((num_rows, 6), (1, 0), projection='3d')
            ax3_xycd_front = plt.subplot2grid((num_rows, 6), (1, 1), projection='3d')
            
            # Side-Left view (elevation=12°, lower camera)
            ax3_xy_left = plt.subplot2grid((num_rows, 6), (1, 2), projection='3d')
            ax3_xycd_left = plt.subplot2grid((num_rows, 6), (1, 3), projection='3d')
            
            # Auto-aligned view (matches image perspective)
            ax3_xy_right = plt.subplot2grid((num_rows, 6), (1, 4), projection='3d')
            ax3_xycd_right = plt.subplot2grid((num_rows, 6), (1, 5), projection='3d')
            
            # --- Compute auto-aligned view that matches 2D image perspective ---
            input_2d_pose_for_align = None
            vis_for_align = None
            if input_2d_keypoints is not None and frame_idx < len(input_2d_keypoints):
                input_2d_pose_for_align = input_2d_keypoints[frame_idx]
            if results_model1.get('visibility_scores') is not None and frame_idx < len(results_model1['visibility_scores']):
                vis_for_align = results_model1['visibility_scores'][frame_idx]
            
            # Use the stronger depth model for view matching if available; fallback to model1
            points3d_for_align = pred_pose_2 if isinstance(pred_pose_2, np.ndarray) else pred_pose_1
            
            best_elev_deg, best_azim_deg = find_best_view(points3d_for_align, input_2d_pose_for_align, vis_for_align)
            
            # Group by angle: [(ax_model1, ax_model2), view_name, elevation, azimuth]
            view_configs = [
                ((ax3_xy_front, ax3_xycd_front), "Front", 45, 0),
                ((ax3_xy_left, ax3_xycd_left), "Side (L)", 12, -90),
                # Last panel uses auto-aligned view that best matches the image's 2D pose
                ((ax3_xy_right, ax3_xycd_right), f"Auto (image view)", best_elev_deg, best_azim_deg)
            ]
        else:
            # 6-panel: 3 angles × 2 models side-by-side
            fig = plt.figure(figsize=(18, 10))
            
            # Front view
            ax3_xy_front = fig.add_subplot(2, 3, 1, projection='3d')
            ax3_xycd_front = fig.add_subplot(2, 3, 2, projection='3d')
            
            # Side-Left view
            ax3_xy_left = fig.add_subplot(2, 3, 3, projection='3d')
            ax3_xycd_left = fig.add_subplot(2, 3, 4, projection='3d')
            
            # Auto-aligned view
            ax3_xy_right = fig.add_subplot(2, 3, 5, projection='3d')
            ax3_xycd_right = fig.add_subplot(2, 3, 6, projection='3d')
            
            # --- Compute auto-aligned view that matches 2D image perspective ---
            input_2d_pose_for_align = None
            vis_for_align = None
            if input_2d_keypoints is not None and frame_idx < len(input_2d_keypoints):
                input_2d_pose_for_align = input_2d_keypoints[frame_idx]
            if results_model1.get('visibility_scores') is not None and frame_idx < len(results_model1['visibility_scores']):
                vis_for_align = results_model1['visibility_scores'][frame_idx]
            
            # Use the stronger depth model for view matching if available; fallback to model1
            points3d_for_align = pred_pose_2 if isinstance(pred_pose_2, np.ndarray) else pred_pose_1
            
            best_elev_deg, best_azim_deg = find_best_view(points3d_for_align, input_2d_pose_for_align, vis_for_align)
            
            view_configs = [
                ((ax3_xy_front, ax3_xycd_front), "Front", 45, 0),
                ((ax3_xy_left, ax3_xycd_left), "Side (L)", 12, -90),
                ((ax3_xy_right, ax3_xycd_right), f"Auto (image view)", best_elev_deg, best_azim_deg)
            ]
        
        # Draw 3D skeletons for each viewing angle
        for (ax_model1, ax_model2), view_name, elev, azim in view_configs:
            # Draw Model1 (XY) skeleton
            for link in skeleton_links:
                start_name, end_name = link
                start_idx = keypoint_index[start_name]
                end_idx = keypoint_index[end_name]
                
                # GT in black dashed
                start_pt_gt = gt_pose[start_idx]
                end_pt_gt = gt_pose[end_idx]
                ax_model1.plot([start_pt_gt[0], end_pt_gt[0]], 
                              [start_pt_gt[2], end_pt_gt[2]], 
                              [start_pt_gt[1], end_pt_gt[1]], 
                              'k--', linewidth=1.5, alpha=0.7)
                
                # Model1 prediction in red
                start_pt = pred_pose_1[start_idx]
                end_pt = pred_pose_1[end_idx]
                ax_model1.plot([start_pt[0], end_pt[0]], 
                              [start_pt[2], end_pt[2]], 
                              [start_pt[1], end_pt[1]], 
                              'r-', linewidth=2, alpha=0.8)
            
            # Draw keypoints for Model1
            ax_model1.scatter(gt_pose[:, 0], gt_pose[:, 2], gt_pose[:, 1], 
                            c='black', s=15, alpha=0.7, marker='x', label='GT')
            ax_model1.scatter(pred_pose_1[:, 0], pred_pose_1[:, 2], pred_pose_1[:, 1], 
                            c='red', s=25, alpha=0.8, label=f'{model1.upper()}')
            
            ax_model1.set_title(f'{model1.upper()} - {view_name}\nError: {frame[f"{model1}_error"]:.1f}mm', fontsize=9)
            ax_model1.set_xlabel('X (mm)', fontsize=8)
            ax_model1.set_ylabel('Z (mm)', fontsize=8)
            ax_model1.set_zlabel('Y (mm)', fontsize=8)
            ax_model1.set_xlim(-700, 700)
            ax_model1.set_ylim(-700, 700)
            ax_model1.set_zlim(-700, 700)
            ax_model1.view_init(elev=elev, azim=azim)
            # Keep perspective scale consistent across views
            if hasattr(ax_model1, "dist"):
                ax_model1.dist = 7
            if view_name == "Front":
                ax_model1.legend(loc='upper right', fontsize=7)
            
            # Draw Model2 (XYCD) skeleton
            for link in skeleton_links:
                start_name, end_name = link
                start_idx = keypoint_index[start_name]
                end_idx = keypoint_index[end_name]
                
                # GT in black dashed
                start_pt_gt = gt_pose[start_idx]
                end_pt_gt = gt_pose[end_idx]
                ax_model2.plot([start_pt_gt[0], end_pt_gt[0]], 
                              [start_pt_gt[2], end_pt_gt[2]], 
                              [start_pt_gt[1], end_pt_gt[1]], 
                              'k--', linewidth=1.5, alpha=0.7)
                
                # Model2 prediction in blue
                start_pt = pred_pose_2[start_idx]
                end_pt = pred_pose_2[end_idx]
                ax_model2.plot([start_pt[0], end_pt[0]], 
                              [start_pt[2], end_pt[2]], 
                              [start_pt[1], end_pt[1]], 
                              'b-', linewidth=2, alpha=0.8)
            
            # Draw keypoints for Model2
            ax_model2.scatter(gt_pose[:, 0], gt_pose[:, 2], gt_pose[:, 1], 
                            c='black', s=15, alpha=0.7, marker='x', label='GT')
            ax_model2.scatter(pred_pose_2[:, 0], pred_pose_2[:, 2], pred_pose_2[:, 1], 
                            c='blue', s=25, alpha=0.8, label=f'{model2.upper()}')
            
            # Get model2 error from results if not in frame dict
            model2_error = frame.get(f'{model2}_error', 0.0)
            if model2_error == 0.0 and frame_idx < len(results_model2['per_frame_mpjpe']):
                model2_error = results_model2['per_frame_mpjpe'][frame_idx]
            
            ax_model2.set_title(f'{model2.upper()} - {view_name}\nError: {model2_error:.1f}mm', fontsize=9)
            ax_model2.set_xlabel('X (mm)', fontsize=8)
            ax_model2.set_ylabel('Z (mm)', fontsize=8)
            ax_model2.set_zlabel('Y (mm)', fontsize=8)
            ax_model2.set_xlim(-700, 700)
            ax_model2.set_ylim(-700, 700)
            ax_model2.set_zlim(-700, 700)
            ax_model2.view_init(elev=elev, azim=azim)
            # Keep perspective scale consistent across views
            if hasattr(ax_model2, "dist"):
                ax_model2.dist = 7
            if view_name == "Front":
                ax_model2.legend(loc='upper right', fontsize=7)
        
        # Add CASP visualization panels if casp_data is provided
        if casp_data is not None and imgname in casp_data['imgname_to_idx']:
            casp_idx = casp_data['imgname_to_idx'][imgname]
            casp_keypoints_2d = casp_data['keypoints_2d'][casp_idx] if casp_data['keypoints_2d'] is not None else None
            casp_confidence = casp_data['confidence'][casp_idx] if casp_data['confidence'] is not None else None
            casp_descriptors = casp_data['casp_descriptors'][casp_idx] if casp_data['casp_descriptors'] is not None else None
            
            # Select 4-5 joints with varying radii for CASP visualization
            # First, compute radii for all joints to find diverse examples
            all_radii = []
            if casp_descriptors is not None and len(casp_descriptors.shape) >= 2:
                for j_idx in range(17):
                    desc = casp_descriptors[j_idx]
                    if len(desc) == 10:
                        all_radii.append((j_idx, float(desc[3])))  # (joint_idx, radius)
            
            # Select joints with diverse radii (smallest, small-mid, mid, mid-large, largest)
            if len(all_radii) >= 5:
                all_radii_sorted = sorted(all_radii, key=lambda x: x[1])
                # Pick: 0%, 25%, 50%, 75%, 100% of sorted radii
                indices = [0, len(all_radii_sorted)//4, len(all_radii_sorted)//2, 
                          3*len(all_radii_sorted)//4, len(all_radii_sorted)-1]
                casp_joints = [all_radii_sorted[i][0] for i in indices]
            else:
                # Fallback: use first 4-5 joints
                casp_joints = list(range(min(5, 17)))
            
            # Get joint names
            joint_names_h36m = [
                'root', 'right_hip', 'right_knee', 'right_foot',
                'left_hip', 'left_knee', 'left_foot', 'spine',
                'thorax', 'neck_base', 'head', 'left_shoulder',
                'left_elbow', 'left_wrist', 'right_shoulder',
                'right_elbow', 'right_wrist'
            ]
            casp_joint_names = [joint_names_h36m[j] for j in casp_joints]
            
            # FIXED CROP SIZE - use the largest radius among selected joints for all panels
            max_radius = max([all_radii_sorted[i][1] for i in indices]) if len(all_radii_sorted) >= 5 else 30.0
            fixed_crop_size = int(max_radius * 7)  # Use 7x largest radius as fixed crop size
            print(f"  CASP visualization: Using fixed crop size {fixed_crop_size}px (based on max radius {max_radius:.0f}px)")
            
            for panel_idx, (joint_idx, joint_name) in enumerate(zip(casp_joints, casp_joint_names)):
                ax_casp = plt.subplot2grid((num_rows, 6), (2, panel_idx + 1), colspan=1)
                
                if casp_keypoints_2d is not None and casp_confidence is not None:
                    kp = casp_keypoints_2d[joint_idx]
                    conf = casp_confidence[joint_idx]
                    
                    # Extract depth statistics from pre-computed CASP descriptors
                    # CASP descriptor format (compact 10D):
                    # [x, y, confidence, radius, Q10, Q25, Q50, Q75, Q90, exact_depth]
                    depth_stats = {}
                    radius = 10.0  # Default fallback
                    
                    if casp_descriptors is not None and len(casp_descriptors.shape) >= 2:
                        desc = casp_descriptors[joint_idx]  # (10,) compact format
                        
                        if len(desc) == 10:
                            # Compact 10D CASP format
                            radius = float(desc[3])  # Use radius from descriptor
                            
                            # Extract 3 depth samples: min (Q10), max (Q90), pointwise (exact_depth)
                            # Sample ALL positions at random radii within [0, radius] to show sampling disk coverage
                            # Use deterministic randomness (seeded by joint position + frame index)
                            seed_val = int(kp[0] + kp[1] + joint_idx + frame_idx) % (2**32 - 1)
                            np.random.seed(seed_val)
                            
                            # Sample three random angles for the three depth samples
                            angles = np.random.uniform(0, 2 * np.pi, size=3)
                            
                            # Sample random radii for ALL samples (within sampling disk)
                            # Use sqrt for uniform distribution in disk area
                            radii = radius * np.sqrt(np.random.uniform(0, 1, size=3))
                            
                            # Min depth sample (Q10) - random position within disk
                            x_min = float(kp[0] + radii[0] * np.cos(angles[0]))
                            y_min = float(kp[1] + radii[0] * np.sin(angles[0]))
                            
                            # Max depth sample (Q90) - random position within disk
                            x_max = float(kp[0] + radii[1] * np.cos(angles[1]))
                            y_max = float(kp[1] + radii[1] * np.sin(angles[1]))
                            
                            # Pointwise depth sample (exact) - random position within disk
                            x_point = float(kp[0] + radii[2] * np.cos(angles[2]))
                            y_point = float(kp[1] + radii[2] * np.sin(angles[2]))
                            
                            depth_stats = {
                                'radius': radius,
                                'min': {'depth': float(desc[4]), 'x': x_min, 'y': y_min, 'r': radii[0]},      # Q10 = min (random pos in disk)
                                'max': {'depth': float(desc[8]), 'x': x_max, 'y': y_max, 'r': radii[1]},      # Q90 = max (random pos in disk)
                                'pointwise': {'depth': float(desc[9]), 'x': x_point, 'y': y_point, 'r': radii[2]},  # exact depth (random pos in disk)
                            }
                        else:
                            # Fallback for other formats
                            depth_stats = None
                    else:
                        depth_stats = None
                
                # FIXED CROP SIZE - all panels use same crop dimensions
                pad = fixed_crop_size
                x_min = max(0, int(kp[0] - pad))
                x_max = min(img_rgb.shape[1], int(kp[0] + pad))
                y_min = max(0, int(kp[1] - pad))
                y_max = min(img_rgb.shape[0], int(kp[1] + pad))
                
                if x_max > x_min and y_max > y_min:
                    img_crop = img_rgb[y_min:y_max, x_min:x_max]
                    ax_casp.imshow(img_crop)
                    
                    # Adjust keypoint coordinates to cropped image
                    kp_crop = kp.copy()
                    kp_crop[0] -= x_min
                    kp_crop[1] -= y_min
                    
                    # Draw sampling disk (cyan circle)
                    circle = plt.Circle(kp_crop, radius, fill=False, color='cyan', linewidth=2.5)
                    ax_casp.add_patch(circle)
                    
                    # Draw keypoint center (red cross)
                    ax_casp.scatter([kp_crop[0]], [kp_crop[1]], c='red', s=80, marker='x', linewidths=3)
                    
                    # Plot 3 depth samples with different colors
                    sample_colors = {
                        'min': 'blue',       # min depth - blue
                        'max': 'red',        # max depth - red  
                        'pointwise': 'lime', # exact center - lime (already as cross, just mark)
                    }
                    
                    if depth_stats:
                        for sample_name, sample_data in depth_stats.items():
                            if sample_name == 'radius':
                                continue
                            if isinstance(sample_data, dict) and 'x' in sample_data and 'y' in sample_data:
                                if sample_data['x'] is None or sample_data['y'] is None:
                                    continue
                                # Adjust to cropped coordinates
                                s_x_crop = sample_data['x'] - x_min
                                s_y_crop = sample_data['y'] - y_min
                                color = sample_colors.get(sample_name, 'white')
                                
                                # Pointwise is already drawn as red cross, skip or use different marker
                                if sample_name != 'pointwise':
                                    ax_casp.scatter([s_x_crop], [s_y_crop], c=color, s=120, 
                                                  marker='o', edgecolors='black', linewidths=2.5,
                                                  zorder=10, alpha=0.95)
                    
                    # Build info text with depth statistics from CASP descriptors
                    info_lines = [
                        f'{joint_name}',
                        f'Conf={conf:.2f}',
                    ]
                    
                    # Add radius from CASP descriptor
                    info_lines.append(f'R={radius:.0f}px')
                    
                    # Add 3 depth samples: min, max, pointwise
                    if depth_stats:
                        for sample_name in ['min', 'max', 'pointwise']:
                            if sample_name in depth_stats and isinstance(depth_stats[sample_name], dict):
                                sample_data = depth_stats[sample_name]
                                sample_label = sample_name.capitalize()
                                info_lines.append(f'{sample_label}={sample_data["depth"]:.2f}m')
                    
                    info_text = '\n'.join(info_lines)
                    ax_casp.text(0.02, 0.98, info_text, transform=ax_casp.transAxes,
                               fontsize=8, verticalalignment='top', family='monospace',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, pad=0.5))
                else:
                    # Invalid crop bounds
                    ax_casp.text(0.5, 0.5, f'{joint_name}\n(Invalid crop)', 
                               ha='center', va='center', fontsize=10)
            else:
                # No CASP data for this joint
                ax_casp.text(0.5, 0.5, f'{joint_name}\n(No CASP data)', 
                           ha='center', va='center', fontsize=10)
            
            ax_casp.axis('off')
        
        # Add title with appropriate context
        if 'tier' in frame:
            tier_label = frame['tier'].replace('_', ' ').title()
            if 'dav_ordering_accuracy' in frame:
                fig.suptitle(f"{tier_label} | DAV Ordering Acc: {frame['dav_ordering_accuracy']:.4f} | Error: {frame[f'{model1}_error']:.1f}mm\n{imgname}", 
                            fontsize=14, fontweight='bold')
            else:
                # Fallback
                fig.suptitle(f"{tier_label} | {imgname} | Error: {frame[f'{model1}_error']:.1f}mm", 
                            fontsize=14, fontweight='bold')
        else:
            # Original comparison visualization
            fig.suptitle(f"Rank {i+1}: {imgname} | Delta: {frame['delta']:+.1f}mm ({frame['rel_delta']*100:+.1f}%)", 
                        fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save visualization with appropriate filename
        if 'tier' in frame:
            # Tier-based visualization
            tier_name = frame['tier']
            if 'similarity_distance' in frame:
                output_filename = f"{tier_name}_dist_{frame['similarity_distance']:.1f}_{os.path.basename(imgname).replace('.jpg', '.png')}"
            elif 'mean_visibility' in frame:
                output_filename = f"{tier_name}_vis_{frame['mean_visibility']:.3f}_{os.path.basename(imgname).replace('.jpg', '.png')}"
            elif 'detection_2d_error' in frame:
                output_filename = f"{tier_name}_2derr_{frame['detection_2d_error']:.2f}px_{os.path.basename(imgname).replace('.jpg', '.png')}"
            elif 'dav_ordering_accuracy' in frame:
                output_filename = f"{tier_name}_davacc_{frame['dav_ordering_accuracy']:.4f}_{os.path.basename(imgname).replace('.jpg', '.png')}"
            else:
                output_filename = f"{tier_name}_{os.path.basename(imgname).replace('.jpg', '.png')}"
        else:
            # Original comparison visualization
            output_filename = f"rank_{i+1:03d}_delta_{frame['delta']:+.1f}mm_{os.path.basename(imgname).replace('.jpg', '.png')}"
        
        output_path = os.path.join(frames_dir, output_filename)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
        # Save a black background version (replace images with black in top row)
        if img_rgb is not None:
            # Replace top row panels with black backgrounds
            # Panel 1: Black background instead of image
            ax1.clear()
            ax1.imshow(np.zeros_like(img_cropped_wide))
            ax1.set_title("Original Image", fontsize=10)
            ax1.axis('off')
            
            # Panel 2: Black background with 2D pose overlay
            ax2.clear()
            ax2.imshow(np.zeros_like(img_cropped_wide))
            
            # Redraw 2D skeleton on black background
            if input_2d_keypoints is not None and frame_idx < len(input_2d_keypoints):
                input_2d_pose = input_2d_keypoints[frame_idx]
                input_2d_pose_cropped = input_2d_pose.copy()
                input_2d_pose_cropped[:, 0] -= crop_offset_x
                input_2d_pose_cropped[:, 1] -= crop_offset_y
                
                for link in skeleton_links:
                    start_name, end_name = link
                    start_idx = keypoint_index[start_name]
                    end_idx = keypoint_index[end_name]
                    start_pt = input_2d_pose_cropped[start_idx]
                    end_pt = input_2d_pose_cropped[end_idx]
                    ax2.plot([start_pt[0], end_pt[0]], [start_pt[1], end_pt[1]], 
                            'r-', linewidth=2, alpha=0.7)
                
                ax2.scatter(input_2d_pose_cropped[:, 0], input_2d_pose_cropped[:, 1], 
                           c='yellow', s=30, edgecolors='red', linewidths=1.5, zorder=3)
                ax2.set_title("Image + 2D Pose", fontsize=10)
            else:
                ax2.set_title("Image (2D data N/A)", fontsize=10)
            ax2.axis('off')
            
            # Panel 3: Black background with confidence coloring
            ax3.clear()
            if input_2d_keypoints is not None and frame_idx < len(input_2d_keypoints):
                input_2d_pose = input_2d_keypoints[frame_idx]
                crop_x_min = max(0, int(cx - half_size))
                crop_x_max = min(img_rgb.shape[1], int(cx + half_size))
                crop_y_min = max(0, int(cy - half_size))
                crop_y_max = min(img_rgb.shape[0], int(cy + half_size))
                
                if crop_x_max > crop_x_min and crop_y_max > crop_y_min:
                    black_crop = np.zeros((crop_y_max - crop_y_min, crop_x_max - crop_x_min, 3), dtype=np.uint8)
                    ax3.imshow(black_crop)
                else:
                    ax3.imshow(np.zeros_like(img_rgb))
                
                input_2d_pose_cropped = input_2d_pose.copy()
                input_2d_pose_cropped[:, 0] -= crop_x_min
                input_2d_pose_cropped[:, 1] -= crop_y_min
                
                if results_model1.get('visibility_scores') is not None and frame_idx < len(results_model1['visibility_scores']):
                    vis_scores = results_model1['visibility_scores'][frame_idx]
                    
                    for link in skeleton_links:
                        start_name, end_name = link
                        start_idx = keypoint_index[start_name]
                        end_idx = keypoint_index[end_name]
                        start_pt = input_2d_pose_cropped[start_idx]
                        end_pt = input_2d_pose_cropped[end_idx]
                        avg_conf = (vis_scores[start_idx] + vis_scores[end_idx]) / 2.0
                        color_val = (avg_conf - 0.5) / 0.5 if avg_conf >= 0.5 else 0.0
                        color_val = np.clip(color_val, 0.0, 1.0)
                        color = plt.cm.RdYlGn(color_val)
                        ax3.plot([start_pt[0], end_pt[0]], [start_pt[1], end_pt[1]], 
                                color=color, linewidth=2, alpha=0.8)
                    
                    for joint_idx in range(17):
                        pt = input_2d_pose_cropped[joint_idx]
                        conf = vis_scores[joint_idx]
                        color_val = (conf - 0.5) / 0.5 if conf >= 0.5 else 0.0
                        color_val = np.clip(color_val, 0.0, 1.0)
                        color = plt.cm.RdYlGn(color_val)
                        ax3.scatter(pt[0], pt[1], c=[color], s=80, edgecolors='black', 
                                   linewidths=1.5, zorder=3)
                    
                    from matplotlib.cm import ScalarMappable
                    from matplotlib.colors import Normalize
                    sm = ScalarMappable(cmap=plt.cm.RdYlGn, norm=Normalize(vmin=0.5, vmax=1.0))
                    sm.set_array([])
                    cbar = plt.colorbar(sm, ax=ax3, fraction=0.046, pad=0.04)
                    cbar.set_label('Confidence', fontsize=8)
                    cbar.ax.tick_params(labelsize=7)
                else:
                    for link in skeleton_links:
                        start_name, end_name = link
                        start_idx = keypoint_index[start_name]
                        end_idx = keypoint_index[end_name]
                        start_pt = input_2d_pose_cropped[start_idx]
                        end_pt = input_2d_pose_cropped[end_idx]
                        ax3.plot([start_pt[0], end_pt[0]], [start_pt[1], end_pt[1]], 
                                'gray', linewidth=2, alpha=0.7)
                    ax3.scatter(input_2d_pose_cropped[:, 0], input_2d_pose_cropped[:, 1], 
                               c='yellow', s=30, edgecolors='black', linewidths=1.5, zorder=3)
                ax3.set_title("2D Confidence (0.5=White, 1.0=Green)", fontsize=10)
            else:
                ax3.imshow(np.zeros_like(img_rgb))
                ax3.set_title("2D Confidence (N/A)", fontsize=10)
            ax3.axis('off')
            
            # Panel 4: Black background with DAV depth
            ax4.clear()
            if dav_mm is not None and input_2d_keypoints is not None and frame_idx < len(input_2d_keypoints):
                input_2d_pose = input_2d_keypoints[frame_idx]
                x_min, y_min = input_2d_pose.min(axis=0)
                x_max, y_max = input_2d_pose.max(axis=0)
                width = x_max - x_min
                height = y_max - y_min
                size = max(width, height)
                padding = size * 0.25
                cx = (x_min + x_max) / 2
                cy = (y_min + y_max) / 2
                half_size = (size + 2 * padding) / 2
                crop_x_min = max(0, int(cx - half_size))
                crop_x_max = min(img_rgb.shape[1], int(cx + half_size))
                crop_y_min = max(0, int(cy - half_size))
                crop_y_max = min(img_rgb.shape[0], int(cy + half_size))
                
                if crop_x_max > crop_x_min and crop_y_max > crop_y_min:
                    black_crop = np.zeros((crop_y_max - crop_y_min, crop_x_max - crop_x_min, 3), dtype=np.uint8)
                    ax4.imshow(black_crop)
                else:
                    ax4.imshow(np.zeros_like(img_rgb))
                
                input_2d_pose_cropped = input_2d_pose.copy()
                input_2d_pose_cropped[:, 0] -= crop_x_min
                input_2d_pose_cropped[:, 1] -= crop_y_min
                
                dav_depths = dav_mm[frame_idx]
                depth_range = max(abs(dav_depths.min()), abs(dav_depths.max()))
                if depth_range > 0:
                    norm_depths = (dav_depths + depth_range) / (2 * depth_range)
                else:
                    norm_depths = np.ones_like(dav_depths) * 0.5
                
                for link in skeleton_links:
                    start_name, end_name = link
                    start_idx = keypoint_index[start_name]
                    end_idx = keypoint_index[end_name]
                    start_pt = input_2d_pose_cropped[start_idx]
                    end_pt = input_2d_pose_cropped[end_idx]
                    avg_depth_norm = (norm_depths[start_idx] + norm_depths[end_idx]) / 2.0
                    color = plt.cm.RdBu_r(avg_depth_norm)
                    ax4.plot([start_pt[0], end_pt[0]], [start_pt[1], end_pt[1]], 
                            color=color, linewidth=2, alpha=0.8)
                
                for joint_idx in range(17):
                    pt = input_2d_pose_cropped[joint_idx]
                    depth_norm = norm_depths[joint_idx]
                    color = plt.cm.RdBu_r(depth_norm)
                    ax4.scatter(pt[0], pt[1], c=[color], s=80, edgecolors='black', 
                               linewidths=1.5, zorder=3)
                
                from matplotlib.cm import ScalarMappable
                from matplotlib.colors import Normalize
                sm = ScalarMappable(cmap=plt.cm.RdBu_r, norm=Normalize(vmin=-depth_range, vmax=depth_range))
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax4, fraction=0.046, pad=0.04)
                cbar.set_label('DAV Depth (mm)', fontsize=8)
                cbar.ax.tick_params(labelsize=7)
                ax4.set_title("DAV Input Depth (Blue=Closer, White=Root, Red=Further)", fontsize=10)
            else:
                ax4.imshow(np.zeros_like(img_rgb))
                ax4.set_title("DAV Input Depth (N/A)", fontsize=10)
            ax4.axis('off')
            
            # Save black background version
            black_bg_filename = output_filename.replace('.png', '_black_background.png')
            black_bg_path = os.path.join(frames_dir, black_bg_filename)
            plt.savefig(black_bg_path, dpi=150, bbox_inches='tight')
        
        plt.close()
        
        visualized_count += 1
    
    print(f"  ✓ Visualized {visualized_count} frames")
    if dataset == 'h36m':
        print(f"  Note: Image loading not implemented for H36M dataset yet")
    else:
        print(f"  Note: Image loading implemented for {dataset.upper()} dataset")


def find_and_visualize_representative_poses(results_model1, results_model2, dataset, output_dir, num_clusters=100, model1='xy', model2='xycd'):
    """Find and visualize representative poses using K-means clustering in 3D GT pose space.
    
    This function identifies the most diverse and representative poses in a dataset by:
    1. Clustering 3D ground truth poses using K-means in 51D space (17 joints × 3 coords)
    2. Finding the frame closest to each cluster center (most representative of that pose type)
    3. Sorting clusters by size to show most common poses first
    4. Visualizing one representative frame per cluster
    
    Args:
        results_model1: Dict containing model1 results (pred_mm, gt_mm, errors, imgnames)
        results_model2: Dict containing model2 results
        dataset: Dataset name ('3dhp', 'h36m', '3dpw')
        output_dir: Directory to save visualizations and text summaries
        num_clusters: Number of pose clusters to find (default: 100)
        model1: Name of first model variant (e.g., 'xy')
        model2: Name of second model variant (e.g., 'xycd')
    
    Returns:
        List of dicts containing representative frame info, sorted by cluster size
    
    Algorithm:
        1. Extract GT 3D poses from results_model1 (N, 17, 3)
        2. Flatten to feature vectors (N, 17*3) where 17*3 = 51
        3. Run K-means clustering to partition poses into num_clusters groups
        4. For each cluster:
           - Find the frame whose pose is closest to cluster center (L2 distance)
           - This frame is the most "representative" of that pose type
        5. Sort clusters by size (descending) so most common poses appear first
        6. Visualize each representative frame with both model predictions
    
    Why this is useful:
        - Unlike random sampling, ensures diverse pose coverage
        - Unlike error-based selection, not biased toward difficult poses
        - Shows typical poses weighted by their frequency in the dataset
        - Each cluster represents a distinct pose type (e.g., walking, sitting, reaching)
        - Larger clusters = more common poses in the dataset
    
    Example output:
        Cluster 0 (500 frames): Standing upright, arms at sides
        Cluster 1 (350 frames): Walking, mid-stride
        Cluster 2 (200 frames): Reaching overhead
        ...
    """
    from sklearn.cluster import KMeans
    
    print(f"\n{'='*60}")
    print(f"Finding {num_clusters} representative poses for {dataset} via clustering")
    print(f"{'='*60}")
    
    # Extract GT poses and flatten for clustering
    gt_mm = results_model1['gt_mm']  # (N, 17, 3)
    errors1 = results_model1['per_frame_mpjpe']
    errors2 = results_model2['per_frame_mpjpe']
    imgnames = results_model1['imgnames']
    
    n = len(gt_mm)
    
    # Flatten 3D poses to feature vectors: (N, 17*3)
    gt_flat = gt_mm.reshape(n, -1)
    
    print(f"  Total frames: {n}")
    print(f"  Feature dimension: {gt_flat.shape[1]} (17 joints × 3 coords)")
    
    # Perform K-means clustering
    print(f"  Running K-means clustering with {num_clusters} clusters...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10, max_iter=300, verbose=0)
    cluster_labels = kmeans.fit_predict(gt_flat)
    
    print(f"  ✓ Clustering complete")
    
    # For each cluster, find the frame closest to the cluster center
    representative_frames = []
    
    print(f"  Selecting representative frame from each cluster...")
    for cluster_id in range(num_clusters):
        # Get indices of frames in this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        
        if len(cluster_indices) == 0:
            continue
        
        # Get cluster center
        cluster_center = kmeans.cluster_centers_[cluster_id]
        
        # Compute distances from cluster center for frames in this cluster
        cluster_poses = gt_flat[cluster_indices]
        distances = np.linalg.norm(cluster_poses - cluster_center, axis=1)
        
        # Find frame closest to center (most representative)
        closest_idx_in_cluster = np.argmin(distances)
        frame_idx = cluster_indices[closest_idx_in_cluster]
        
        # Compute deltas
        delta = errors2[frame_idx] - errors1[frame_idx]
        rel_delta = delta / (errors1[frame_idx] + 1e-8)
        
        representative_frames.append({
            'idx': frame_idx,
            'cluster_id': cluster_id,
            'cluster_size': len(cluster_indices),
            'distance_to_center': distances[closest_idx_in_cluster],
            'imgname': imgnames[frame_idx],
            f'{model1}_error': errors1[frame_idx],
            f'{model2}_error': errors2[frame_idx],
            'delta': delta,
            'rel_delta': rel_delta
        })
    
    print(f"  ✓ Selected {len(representative_frames)} representative frames")
    
    # Sort by cluster size (descending) to show most common poses first
    representative_frames.sort(key=lambda x: x['cluster_size'], reverse=True)
    
    # Print top 10 most common pose types
    print(f"\n  Top 10 most common pose types:")
    for i, f in enumerate(representative_frames[:10]):
        print(f"    {i+1}. Cluster {f['cluster_id']}: {f['cluster_size']} frames, "
              f"{model1}={f[f'{model1}_error']:.1f}mm, {model2}={f[f'{model2}_error']:.1f}mm, "
              f"delta={f['delta']:.1f}mm ({f['rel_delta']*100:.1f}%)")
    
    # Save representative frame list
    output_path = os.path.join(output_dir, f'representative_poses_{dataset}_{model1}_vs_{model2}_k{num_clusters}.txt')
    with open(output_path, 'w') as f:
        f.write(f"Representative poses via K-means clustering: {len(representative_frames)} clusters\n")
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Clustering performed in 3D GT pose space (17 joints × 3 coords)\n")
        f.write(f"Frames sorted by cluster size (most common poses first)\n")
        f.write("="*80 + "\n\n")
        
        for i, frame in enumerate(representative_frames):
            f.write(f"{i+1}. Cluster {frame['cluster_id']} (size: {frame['cluster_size']} frames)\n")
            f.write(f"   Representative frame: {frame['imgname']}\n")
            f.write(f"   Distance to cluster center: {frame['distance_to_center']:.2f}\n")
            f.write(f"   {model1.upper()} error: {frame[f'{model1}_error']:.2f} mm\n")
            f.write(f"   {model2.upper()} error: {frame[f'{model2}_error']:.2f} mm\n")
            f.write(f"   Absolute delta: {frame['delta']:.2f} mm\n")
            f.write(f"   Relative delta: {frame['rel_delta']*100:.2f}%\n\n")
    
    print(f"  Saved representative poses list: {output_path}")
    
    # Visualize representative frames with skeleton overlays
    visualize_frame_skeletons(representative_frames, results_model1, results_model2, dataset, output_dir, model1, model2)
    
    return representative_frames

def analyze_angles_orientations(results_by_model, dataset, output_dir, model_names):
    """Analyze angle/orientation errors between models.
    
    This function:
    1. Computes angle errors for each bone (joint connection)
    2. Reports per-bone and overall mean angle errors
    3. Compares angle errors across models
    4. Only uses joints/bones that pass visibility filtering
    
    Args:
        results_by_model: Dict mapping model name to results dict
        dataset: Dataset name
        output_dir: Directory to save analysis results
        model_names: List of model names to compare
    """
    print(f"\n{'='*60}")
    print(f"Angle & Orientation Analysis: {dataset}")
    print(f"{'='*60}")
    
    # Compute angle errors for each model
    angle_results = {}
    
    for model_name in model_names:
        if model_name not in results_by_model:
            print(f"  ✗ Model {model_name} not available, skipping")
            continue
        
        results = results_by_model[model_name]
        pred_mm = results['pred_mm']
        gt_mm = results['gt_mm']
        joint_mask = results.get('joint_mask', np.ones_like(pred_mm[:, :, 0], dtype=bool))
        
        print(f"\n  Processing {model_name.upper()}...")
        
        # Compute angle errors
        angle_data = compute_angle_errors(pred_mm, gt_mm, joint_mask)
        
        x_angle_errors = angle_data['x_angle_errors']  # (N, 16)
        y_angle_errors = angle_data['y_angle_errors']  # (N, 16)
        bone_mask = angle_data['bone_mask']  # (N, 16)
        bone_names = angle_data['bone_names']
        
        # Compute torso orientation errors
        torso_data = compute_torso_orientation_error(pred_mm, gt_mm, joint_mask)
        
        torso_orientation_errors = torso_data['orientation_errors']  # (N,)
        torso_valid_mask = torso_data['valid_mask']  # (N,)
        
        # Compute statistics
        n_valid_bones = np.sum(bone_mask)
        n_total_bones = bone_mask.size
        n_valid_torso = np.sum(torso_valid_mask)
        n_total_torso = len(torso_valid_mask)
        
        print(f"    Valid bones: {n_valid_bones} / {n_total_bones} ({n_valid_bones/n_total_bones*100:.1f}%)")
        print(f"    Valid torso orientations: {n_valid_torso} / {n_total_torso} ({n_valid_torso/n_total_torso*100:.1f}%)")
        
        # Overall statistics (across all bones and frames)
        overall_x_angle_error = np.nanmean(x_angle_errors)
        overall_y_angle_error = np.nanmean(y_angle_errors)
        overall_angle_error = np.nanmean([x_angle_errors, y_angle_errors])
        overall_torso_orientation_error = np.nanmean(torso_orientation_errors)
        
        print(f"    Overall X-angle error: {overall_x_angle_error:.2f}°")
        print(f"    Overall Y-angle error: {overall_y_angle_error:.2f}°")
        print(f"    Overall mean angle error: {overall_angle_error:.2f}°")
        print(f"    Overall torso orientation error: {overall_torso_orientation_error:.2f}°")
        
        # Per-bone statistics (mean across frames for each bone)
        per_bone_x_errors = np.nanmean(x_angle_errors, axis=0)  # (16,)
        per_bone_y_errors = np.nanmean(y_angle_errors, axis=0)  # (16,)
        per_bone_mean_errors = (per_bone_x_errors + per_bone_y_errors) / 2.0
        
        angle_results[model_name] = {
            'x_angle_errors': x_angle_errors,
            'y_angle_errors': y_angle_errors,
            'bone_mask': bone_mask,
            'bone_names': bone_names,
            'overall_x_angle_error': overall_x_angle_error,
            'overall_y_angle_error': overall_y_angle_error,
            'overall_angle_error': overall_angle_error,
            'per_bone_x_errors': per_bone_x_errors,
            'per_bone_y_errors': per_bone_y_errors,
            'per_bone_mean_errors': per_bone_mean_errors,
            'n_valid_bones': n_valid_bones,
            'n_total_bones': n_total_bones,
            'torso_orientation_errors': torso_orientation_errors,
            'torso_valid_mask': torso_valid_mask,
            'overall_torso_orientation_error': overall_torso_orientation_error,
            'n_valid_torso': n_valid_torso,
            'n_total_torso': n_total_torso,
        }
    
    if len(angle_results) == 0:
        print(f"  ✗ No angle results available")
        return
    
    # Print per-bone comparison table
    print(f"\n{'='*80}")
    print(f"Per-Bone Angle Error Comparison")
    print(f"{'='*80}")
    
    bone_names = angle_results[model_names[0]]['bone_names']
    
    # Header
    header = f"  {'Bone':<25}"
    for model_name in model_names:
        if model_name in angle_results:
            header += f" {model_name.upper()+'_X':<10} {model_name.upper()+'_Y':<10} {model_name.upper()+'_Mean':<10}"
    print(header)
    print(f"  {'-'*len(header)}")
    
    # Per-bone rows
    for bone_idx, bone_name in enumerate(bone_names):
        row = f"  {bone_name:<25}"
        for model_name in model_names:
            if model_name in angle_results:
                x_err = angle_results[model_name]['per_bone_x_errors'][bone_idx]
                y_err = angle_results[model_name]['per_bone_y_errors'][bone_idx]
                mean_err = angle_results[model_name]['per_bone_mean_errors'][bone_idx]
                row += f" {x_err:>8.2f}°  {y_err:>8.2f}°  {mean_err:>8.2f}° "
        print(row)
    
    # Overall row
    print(f"  {'-'*len(header)}")
    row = f"  {'OVERALL':<25}"
    for model_name in model_names:
        if model_name in angle_results:
            x_err = angle_results[model_name]['overall_x_angle_error']
            y_err = angle_results[model_name]['overall_y_angle_error']
            mean_err = angle_results[model_name]['overall_angle_error']
            row += f" {x_err:>8.2f}°  {y_err:>8.2f}°  {mean_err:>8.2f}° "
    print(row)
    
    # Torso orientation comparison
    print(f"\n{'='*80}")
    print(f"Torso Orientation Error Comparison")
    print(f"{'='*80}")
    
    header_torso = f"  {'Model':<15} {'N Valid':<12} {'Mean Error':<15} {'Median Error':<15} {'p90':<10} {'p95':<10}"
    print(header_torso)
    print(f"  {'-'*len(header_torso)}")
    
    for model_name in model_names:
        if model_name in angle_results:
            res = angle_results[model_name]
            torso_errors = res['torso_orientation_errors']
            valid_mask = res['torso_valid_mask']
            
            mean_err = np.nanmean(torso_errors)
            median_err = np.nanmedian(torso_errors)
            p90_err = np.nanpercentile(torso_errors, 90)
            p95_err = np.nanpercentile(torso_errors, 95)
            
            row = f"  {model_name.upper():<15} {res['n_valid_torso']:<12} {mean_err:>10.2f}°    {median_err:>10.2f}°    {p90_err:>6.2f}°  {p95_err:>6.2f}°"
            print(row)
    
    # Save detailed results
    output_path = os.path.join(output_dir, f'angles_orientations_{dataset}_comparison.txt')
    with open(output_path, 'w') as f:
        f.write(f"Angle & Orientation Error Analysis: {dataset}\n")
        f.write(f"Models: {', '.join([m.upper() for m in model_names if m in angle_results])}\n")
        f.write("="*80 + "\n\n")
        
        f.write("Angle error computation:\n")
        f.write("  - X-angle: rotation about Y-axis (angle in XZ plane)\n")
        f.write("  - Y-angle: rotation about X-axis (angle in YZ plane)\n")
        f.write("  - Computed for each bone (joint connection)\n")
        f.write("  - Only valid bones (both endpoints pass visibility filter) are included\n\n")
        
        # Overall statistics
        f.write("="*80 + "\n")
        f.write("Overall Angle Errors (across all bones and frames)\n")
        f.write("="*80 + "\n\n")
        
        for model_name in model_names:
            if model_name not in angle_results:
                continue
            
            res = angle_results[model_name]
            f.write(f"{model_name.upper()}:\n")
            f.write(f"  Valid bones: {res['n_valid_bones']} / {res['n_total_bones']} ")
            f.write(f"({res['n_valid_bones']/res['n_total_bones']*100:.1f}%)\n")
            f.write(f"  Overall X-angle error: {res['overall_x_angle_error']:.2f}°\n")
            f.write(f"  Overall Y-angle error: {res['overall_y_angle_error']:.2f}°\n")
            f.write(f"  Overall mean angle error: {res['overall_angle_error']:.2f}°\n\n")
        
        # Torso orientation statistics
        f.write("="*80 + "\n")
        f.write("Torso Orientation Errors\n")
        f.write("="*80 + "\n\n")
        f.write("Torso orientation computed from cross product of:\n")
        f.write("  - Hip vector (right_hip → left_hip)\n")
        f.write("  - Shoulder vector (right_shoulder → left_shoulder)\n")
        f.write("Normal vector = hip_vec × shoulder_vec\n")
        f.write("Error = angular difference between predicted and GT normal vectors\n\n")
        
        for model_name in model_names:
            if model_name not in angle_results:
                continue
            
            res = angle_results[model_name]
            torso_errors = res['torso_orientation_errors']
            
            f.write(f"{model_name.upper()}:\n")
            f.write(f"  Valid frames: {res['n_valid_torso']} / {res['n_total_torso']} ")
            f.write(f"({res['n_valid_torso']/res['n_total_torso']*100:.1f}%)\n")
            f.write(f"  Mean orientation error: {np.nanmean(torso_errors):.2f}°\n")
            f.write(f"  Median orientation error: {np.nanmedian(torso_errors):.2f}°\n")
            f.write(f"  Std orientation error: {np.nanstd(torso_errors):.2f}°\n")
            f.write(f"  90th percentile: {np.nanpercentile(torso_errors, 90):.2f}°\n")
            f.write(f"  95th percentile: {np.nanpercentile(torso_errors, 95):.2f}°\n\n")
        
        # Per-bone statistics
        f.write("="*80 + "\n")
        f.write("Per-Bone Angle Errors (mean across frames)\n")
        f.write("="*80 + "\n\n")
        
        for bone_idx, bone_name in enumerate(bone_names):
            f.write(f"{bone_name}:\n")
            for model_name in model_names:
                if model_name not in angle_results:
                    continue
                
                res = angle_results[model_name]
                x_err = res['per_bone_x_errors'][bone_idx]
                y_err = res['per_bone_y_errors'][bone_idx]
                mean_err = res['per_bone_mean_errors'][bone_idx]
                
                f.write(f"  {model_name.upper()}: X={x_err:.2f}°, Y={y_err:.2f}°, Mean={mean_err:.2f}°\n")
            f.write("\n")
        
        # Comparison (if multiple models)
        if len(angle_results) >= 2:
            f.write("="*80 + "\n")
            f.write(f"Comparison: {model_names[1].upper()} vs {model_names[0].upper()}\n")
            f.write("="*80 + "\n\n")
            
            res1 = angle_results[model_names[0]]
            res2 = angle_results[model_names[1]]
            
            overall_delta = res2['overall_angle_error'] - res1['overall_angle_error']
            overall_rel_delta = overall_delta / res1['overall_angle_error'] if res1['overall_angle_error'] > 0 else 0
            
            f.write(f"Overall angle error:\n")
            f.write(f"  {model_names[0].upper()}: {res1['overall_angle_error']:.2f}°\n")
            f.write(f"  {model_names[1].upper()}: {res2['overall_angle_error']:.2f}°\n")
            f.write(f"  Absolute delta: {overall_delta:+.2f}°\n")
            f.write(f"  Relative delta: {overall_rel_delta*100:+.1f}%\n\n")
            
            f.write(f"Per-bone deltas (mean angle error):\n")
            for bone_idx, bone_name in enumerate(bone_names):
                mean1 = res1['per_bone_mean_errors'][bone_idx]
                mean2 = res2['per_bone_mean_errors'][bone_idx]
                delta = mean2 - mean1
                rel_delta = delta / mean1 if mean1 > 0 else 0
                f.write(f"  {bone_name}: {delta:+.2f}° ({rel_delta*100:+.1f}%)\n")
    
    print(f"\n  Saved angle/orientation analysis: {output_path}")
    print(f"  ✓ Analysis complete")

def analyze_domain_similarity(results_test, dataset, output_dir, model_name='xy'):
    """Analyze MPJPE stratified by similarity to source domain (H36M).
    
    This function:
    1. Loads H36M GT poses as the source domain
    2. Clusters H36M poses to capture the distribution
    3. For each test frame, computes distance to nearest H36M cluster center
    4. Stratifies test frames by similarity (most similar 20%, middle 60%, least similar 20%)
    5. Computes MPJPE statistics for each similarity tier
    """
    from sklearn.cluster import KMeans
    
    print(f"\n{'='*60}")
    print(f"Domain Similarity Analysis: {dataset} vs H36M (source)")
    print(f"{'='*60}")
    
    # Load H36M GT poses as source domain
    h36m_gt, _, _, _, _ = load_ground_truth('h36m')
    
    # Center and preprocess H36M GT (same as test preprocessing)
    h36m_centered = h36m_gt - h36m_gt[:, 0:1, :]
    h36m_centered[:, :, 1] = -h36m_centered[:, :, 1]
    h36m_mm = h36m_centered * 1000.0
    
    # Flatten to feature vectors for clustering
    h36m_flat = h36m_mm.reshape(len(h36m_mm), -1)  # (N_h36m, 51)
    
    print(f"  H36M source: {len(h36m_mm)} frames")
    print(f"  Test domain ({dataset}): {len(results_test['gt_mm'])} frames")
    
    # Cluster H36M poses to capture source domain distribution
    # Use ~100 clusters to capture diversity
    n_clusters = min(100, len(h36m_mm) // 50)  # At least 50 frames per cluster
    print(f"  Clustering H36M into {n_clusters} pose clusters...")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300, verbose=0)
    kmeans.fit(h36m_flat)
    
    print(f"  ✓ H36M clustering complete")
    
    # For each test frame, compute distance to nearest H36M cluster center
    test_gt_mm = results_test['gt_mm']
    imgnames = results_test['imgnames']
    per_frame_mpjpe = results_test['per_frame_mpjpe']
    test_flat = test_gt_mm.reshape(len(test_gt_mm), -1)  # (N_test, 51)
    
    # Compute distances to all cluster centers for each test frame
    distances_to_centers = kmeans.transform(test_flat)  # (N_test, n_clusters)
    
    # Distance to nearest cluster = similarity metric (lower = more similar to H36M)
    min_distances = np.min(distances_to_centers, axis=1)  # (N_test,)
    
    print(f"  ✓ Computed similarity scores for {len(min_distances)} test frames")
    print(f"    Distance range: [{np.min(min_distances):.1f}, {np.max(min_distances):.1f}]")
    print(f"    Mean distance: {np.mean(min_distances):.1f}")
    
    # Stratify test frames by similarity (lower distance = more similar)
    # Tier 1: Most similar 20% (lowest distances)
    # Tier 2: Middle 60%
    # Tier 3: Least similar 20% (highest distances)
    p20 = np.percentile(min_distances, 20)
    p80 = np.percentile(min_distances, 80)
    
    tier1_mask = min_distances <= p20  # Most similar
    tier2_mask = (min_distances > p20) & (min_distances <= p80)  # Middle
    tier3_mask = min_distances > p80  # Least similar
    
    print(f"\n  Similarity stratification:")
    print(f"    Tier 1 (most similar 20%): {np.sum(tier1_mask)} frames, distance ≤ {p20:.1f}")
    print(f"    Tier 2 (middle 60%):       {np.sum(tier2_mask)} frames, distance {p20:.1f} to {p80:.1f}")
    print(f"    Tier 3 (least similar 20%): {np.sum(tier3_mask)} frames, distance > {p80:.1f}")
    
    # Compute MPJPE statistics for each tier
    # Note: imgnames and per_frame_mpjpe already extracted above
    
    tier_stats = {}
    for tier_name, mask in [('most_similar_20', tier1_mask), 
                            ('middle_60', tier2_mask), 
                            ('least_similar_20', tier3_mask)]:
        if np.sum(mask) == 0:
            continue
        
        tier_errors = per_frame_mpjpe[mask]
        # Use nanmean/nanmedian to handle frames with all joints filtered out
        tier_stats[tier_name] = {
            'n_frames': np.sum(mask),
            'mean_mpjpe': np.nanmean(tier_errors),
            'median_mpjpe': np.nanmedian(tier_errors),
            'std_mpjpe': np.nanstd(tier_errors),
            'p25': np.nanpercentile(tier_errors, 25),
            'p75': np.nanpercentile(tier_errors, 75),
            'p90': np.nanpercentile(tier_errors, 90),
            'p95': np.nanpercentile(tier_errors, 95),
        }
    
    # Print comparison
    print(f"\n  MPJPE by Domain Similarity Tier:")
    print(f"  {'Tier':<25} {'N Frames':<12} {'Mean MPJPE':<15} {'Median MPJPE':<15}")
    print(f"  {'-'*67}")
    
    for tier_name, stats in tier_stats.items():
        tier_label = tier_name.replace('_', ' ').title()
        print(f"  {tier_label:<25} {stats['n_frames']:<12} {stats['mean_mpjpe']:>10.2f} mm   {stats['median_mpjpe']:>10.2f} mm")
    
    # Save detailed results
    output_path = os.path.join(output_dir, f'domain_similarity_{dataset}_vs_h36m_{model_name}.txt')
    with open(output_path, 'w') as f:
        f.write(f"Domain Similarity Analysis: {dataset} vs H36M\n")
        f.write(f"Model: {model_name.upper()}\n")
        f.write(f"H36M (source) frames: {len(h36m_mm)}\n")
        f.write(f"Test ({dataset}) frames: {len(results_test['gt_mm'])}\n")
        f.write(f"H36M clusters: {n_clusters}\n")
        f.write("="*80 + "\n\n")
        
        f.write("Similarity metric: Distance to nearest H36M cluster center (in 3D GT pose space)\n")
        f.write(f"  - Lower distance = more similar to H36M poses\n")
        f.write(f"  - Higher distance = less similar (out-of-distribution)\n\n")
        
        f.write(f"Distance statistics:\n")
        f.write(f"  Min: {np.min(min_distances):.2f}\n")
        f.write(f"  Max: {np.max(min_distances):.2f}\n")
        f.write(f"  Mean: {np.mean(min_distances):.2f}\n")
        f.write(f"  Median: {np.median(min_distances):.2f}\n")
        f.write(f"  20th percentile: {p20:.2f}\n")
        f.write(f"  80th percentile: {p80:.2f}\n\n")
        
        f.write("="*80 + "\n")
        f.write("MPJPE by Similarity Tier\n")
        f.write("="*80 + "\n\n")
        
        for tier_name, stats in tier_stats.items():
            tier_label = tier_name.replace('_', ' ').title()
            f.write(f"{tier_label}:\n")
            f.write(f"  N frames: {stats['n_frames']}\n")
            f.write(f"  Mean MPJPE: {stats['mean_mpjpe']:.2f} mm\n")
            f.write(f"  Median MPJPE: {stats['median_mpjpe']:.2f} mm\n")
            f.write(f"  Std MPJPE: {stats['std_mpjpe']:.2f} mm\n")
            f.write(f"  25th percentile: {stats['p25']:.2f} mm\n")
            f.write(f"  75th percentile: {stats['p75']:.2f} mm\n")
            f.write(f"  90th percentile: {stats['p90']:.2f} mm\n")
            f.write(f"  95th percentile: {stats['p95']:.2f} mm\n\n")
        
        # Compute relative differences
        if 'most_similar_20' in tier_stats and 'least_similar_20' in tier_stats:
            most_sim = tier_stats['most_similar_20']['mean_mpjpe']
            least_sim = tier_stats['least_similar_20']['mean_mpjpe']
            delta = least_sim - most_sim
            rel_delta = delta / most_sim if most_sim > 0 else 0
            
            f.write("="*80 + "\n")
            f.write("Comparison: Least Similar vs Most Similar\n")
            f.write("="*80 + "\n")
            f.write(f"  Most similar 20% mean MPJPE: {most_sim:.2f} mm\n")
            f.write(f"  Least similar 20% mean MPJPE: {least_sim:.2f} mm\n")
            f.write(f"  Absolute difference: {delta:+.2f} mm\n")
            f.write(f"  Relative difference: {rel_delta*100:+.1f}%\n")
    
    print(f"  Saved domain similarity analysis: {output_path}")
    
    # Visualize representative frames from each tier
    print(f"\n  Selecting frames for visualization...")
    
    # Select 50 frames per tier with minimum gap of 20 frames between selections
    vis_frames = []
    min_frame_gap = 20
    num_samples_per_tier = 50
    
    for tier_name, mask in [('most_similar_20', tier1_mask), 
                            ('middle_60', tier2_mask), 
                            ('least_similar_20', tier3_mask)]:
        if np.sum(mask) == 0:
            continue
        
        tier_indices = np.where(mask)[0]
        tier_distances = min_distances[mask]
        
        print(f"  Processing {tier_name}: {len(tier_indices)} frames available")
        
        # Sort frames by priority (distance metric)
        if tier_name == 'most_similar_20':
            # Sort by distance (ascending) - prioritize closest to H36M
            sort_order = np.argsort(tier_distances)
        elif tier_name == 'least_similar_20':
            # Sort by distance (descending) - prioritize farthest from H36M
            sort_order = np.argsort(tier_distances)[::-1]
        else:
            # Middle tier: sample evenly across the distance range
            sort_order = np.argsort(tier_distances)
        
        # Select frames while maintaining minimum gap
        selected_indices = []
        used_frame_indices = set()
        
        for idx in sort_order:
            if len(selected_indices) >= num_samples_per_tier:
                break
            
            frame_idx = tier_indices[idx]
            
            # Check if this frame is too close to any already selected frame
            is_too_close = False
            for used_idx in used_frame_indices:
                if abs(frame_idx - used_idx) < min_frame_gap:
                    is_too_close = True
                    break
            
            if not is_too_close:
                selected_indices.append(idx)
                used_frame_indices.add(frame_idx)
        
        print(f"    Selected {len(selected_indices)} frames (min gap={min_frame_gap})")
        
        # Add selected frames to visualization list
        for idx in selected_indices:
            frame_idx = tier_indices[idx]
            vis_frames.append({
                'idx': frame_idx,
                'tier': tier_name,
                'similarity_distance': tier_distances[idx],
                'imgname': imgnames[frame_idx],
                f'{model_name}_error': per_frame_mpjpe[frame_idx],
                'delta': 0.0,  # Placeholder for compatibility
                'rel_delta': 0.0  # Placeholder for compatibility
            })
    
    print(f"  Total selected: {len(vis_frames)} frames for visualization")
    
    return tier_stats, vis_frames

def analyze_occlusion_levels(results_test, dataset, output_dir, model_name='xy', level='frame'):
    """Analyze MPJPE stratified by occlusion level using 2D keypoint confidence.
    
    This function:
    1. Extracts 2D keypoint confidence scores (if available)
    2. Stratifies frames/joints by FIXED visibility bins: ≤0.30, (0.30-0.40], ..., (0.90-1.00]
    3. Computes MPJPE statistics for each visibility bin
    4. Provides consistent cross-dataset comparison (no percentile-based bins)
    
    Args:
        results_test: Dict containing test dataset results (pred_mm, gt_mm, errors, input_2d_keypoints)
        dataset: Test dataset name
        output_dir: Directory to save analysis results
        model_name: Model variant name (default: 'xy')
        level: Occlusion analysis level ('frame' or 'joint')
    
    Returns:
        Tuple of (tier_stats, vis_frames) for visualization
    """
    print(f"\n{'='*60}")
    print(f"Occlusion Analysis: {dataset} (level={level})")
    print(f"{'='*60}")
    
    # Check if visibility scores are available
    visibility_scores = results_test.get('visibility_scores', None)
    
    if visibility_scores is None:
        print(f"  ✗ No visibility scores available for {dataset}")
        print(f"  Skipping occlusion analysis")
        return None
    
    print(f"  ✓ Visibility scores available")
    print(f"  Shape: {visibility_scores.shape}")
    print(f"  Analysis level: {level}")
    
    # Get data for tier statistics
    per_frame_mpjpe = results_test['per_frame_mpjpe']
    pred_mm = results_test['pred_mm']
    gt_mm = results_test['gt_mm']
    imgnames = results_test['imgnames']
    
    # Fixed edges and labels: ≤0.30, (0.30–0.40], …, (0.90–1.00]
    edges  = [0.0, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    labels = ["vis_≤0.30",
              "vis_0.30_0.40", "vis_0.40_0.50", "vis_0.50_0.60", "vis_0.60_0.70",
              "vis_0.70_0.80", "vis_0.80_0.90", "vis_0.90_1.00"]
    
    # Handle per-joint analysis separately
    if level == 'joint':
        # ---------- Per-Joint OCCLUSION BINS (fixed intervals) ----------
        flat_visibility = visibility_scores.flatten()  # (N*J,)
        print(f"\n  Per-joint visibility statistics:")
        print(f"    Visibility range: [{np.min(flat_visibility):.3f}, {np.max(flat_visibility):.3f}]")
        print(f"    Mean visibility: {np.mean(flat_visibility):.3f}")
        print(f"    % joints with vis ≤ 0.3: {np.mean(flat_visibility <= 0.3)*100:.1f}%")

        # Clamp for binning, but keep raw vis for reporting means
        vis_for_bins = np.clip(flat_visibility, 0.0, 1.0)

        # Errors at joint level
        pred_mm_flat = pred_mm.reshape(-1, 3)  # (N*J,3)
        gt_mm_flat   = gt_mm.reshape(-1, 3)    # (N*J,3)
        joint_errors = np.linalg.norm(pred_mm_flat - gt_mm_flat, axis=-1)

        tier_stats_joint = {}
        ordered_keys = []

        print(f"\n  Joint-Level Error by Occlusion Bins (fixed):")
        print(f"  {'Bin':<16} {'N Joints':>9} {'Mean Vis':>10} {'Mean Err':>11} {'Median':>9} {'p90':>9} {'p95':>9}")
        print(f"  {'-'*76}")

        for i, lab in enumerate(labels):
            if i == 0:
                mask = (vis_for_bins <= edges[1])
            else:
                lower, upper = edges[i], edges[i+1]
                mask = (vis_for_bins > lower) & (vis_for_bins <= upper)

            n = int(np.sum(mask))
            if n == 0:
                continue

            errs = joint_errors[mask]
            vis  = flat_visibility[mask]  # use raw (not-clipped) for reporting

            stats = {
                'n_joints': n,
                'mean_visibility': float(np.mean(vis)),
                'mean_error': float(np.mean(errs)),
                'median_error': float(np.median(errs)),
                'std_error': float(np.std(errs)),
                'p25': float(np.percentile(errs, 25)),
                'p75': float(np.percentile(errs, 75)),
                'p90': float(np.percentile(errs, 90)),
                'p95': float(np.percentile(errs, 95)),
            }
            tier_stats_joint[lab] = stats
            ordered_keys.append(lab)

            print(f"  {lab:<16} {n:>9} {stats['mean_visibility']:>10.3f} "
                  f"{stats['mean_error']:>11.2f} {stats['median_error']:>9.2f} "
                  f"{stats['p90']:>9.2f} {stats['p95']:>9.2f}")

        # Save to file
        output_path_joint = os.path.join(output_dir, f'occlusion_analysis_joint_{dataset}_{model_name}.txt')
        with open(output_path_joint, 'w') as f:
            f.write(f"Per-Joint Occlusion Analysis (fixed bins 0.30→1.00): {dataset}\n")
            f.write(f"Model: {model_name.upper()}\n")
            total_joints = pred_mm.shape[0] * pred_mm.shape[1]
            f.write(f"Total joints: {total_joints}\n")
            f.write("="*80 + "\n\n")
            for k in ordered_keys:
                s = tier_stats_joint[k]
                f.write(f"{k}:\n")
                f.write(f"  N joints: {s['n_joints']}\n")
                f.write(f"  Mean visibility: {s['mean_visibility']:.3f}\n")
                f.write(f"  Mean error: {s['mean_error']:.2f} mm\n")
                f.write(f"  Median error: {s['median_error']:.2f} mm\n")
                f.write(f"  Std error: {s['std_error']:.2f} mm\n")
                f.write(f"  p25: {s['p25']:.2f} mm | p75: {s['p75']:.2f} mm | "
                        f"p90: {s['p90']:.2f} mm | p95: {s['p95']:.2f} mm\n\n")

        print(f"  Saved per-joint occlusion analysis: {output_path_joint}")
        return tier_stats_joint, []
    
    # ---------- PER-FRAME ANALYSIS with FIXED BINS ----------
    mean_visibility = np.mean(visibility_scores, axis=1)  # (N,)
    
    print(f"\n  Per-frame visibility statistics:")
    print(f"    Mean visibility range: [{np.min(mean_visibility):.3f}, {np.max(mean_visibility):.3f}]")
    print(f"    Mean of means: {np.mean(mean_visibility):.3f}")
    
    # Clamp visibility for binning, but keep raw for reporting means
    vis_for_bins = np.clip(mean_visibility, 0.0, 1.0)
    
    print(f"\n  Occlusion stratification (fixed bins, mean frame visibility):")
    
    # Create tier masks and compute stats
    tier_stats = {}
    ordered_keys = []
    
    for i, lab in enumerate(labels):
        if i == 0:
            mask = (vis_for_bins <= edges[1])
        else:
            lower, upper = edges[i], edges[i+1]
            mask = (vis_for_bins > lower) & (vis_for_bins <= upper)
        
        n = int(np.sum(mask))
        if n == 0:
            continue
        
        mean_vis_in_bin = np.mean(mean_visibility[mask])
        print(f"    {lab:<16}: {n:>6} frames, mean vis = {mean_vis_in_bin:.3f}")
        ordered_keys.append(lab)
        
        tier_errors = per_frame_mpjpe[mask]
        tier_visibility = mean_visibility[mask]
        
        # Compute per-joint errors for this tier
        tier_pred = pred_mm[mask]
        tier_gt = gt_mm[mask]
        tier_joint_errors = np.linalg.norm(tier_pred - tier_gt, axis=-1)  # (N_tier, 17)
        
        # Compute visibility-weighted MPJPE
        tier_joint_vis = visibility_scores[mask]  # (N_tier, 17)
        eps = 0.01
        inv_weights = 1.0 / np.maximum(tier_joint_vis, eps)  # (N_tier, 17)
        weighted_joint_errors = tier_joint_errors * inv_weights  # (N_tier, 17)
        weighted_mpjpe = np.mean(weighted_joint_errors, axis=1)  # (N_tier,)
        
        tier_stats[lab] = {
            'n_frames': n,
            'mean_visibility': mean_vis_in_bin,
            'mean_mpjpe': float(np.mean(tier_errors)),
            'median_mpjpe': float(np.median(tier_errors)),
            'std_mpjpe': float(np.std(tier_errors)),
            'p25': float(np.percentile(tier_errors, 25)),
            'p75': float(np.percentile(tier_errors, 75)),
            'p80': float(np.percentile(tier_errors, 80)),
            'p90': float(np.percentile(tier_errors, 90)),
            'p95': float(np.percentile(tier_errors, 95)),
            'mean_weighted_mpjpe': float(np.mean(weighted_mpjpe)),
            'median_weighted_mpjpe': float(np.median(weighted_mpjpe)),
        }
    
    # Print comparison
    print(f"\n  MPJPE by Occlusion Level (fixed bins):")
    print(f"  {'Bin':<20} {'N Frames':<10} {'Mean Vis':<12} {'MPJPE':<12} {'p90':<10} {'p95':<10}")
    print(f"  {'-'*74}")
    
    for tier_name in ordered_keys:
        stats = tier_stats[tier_name]
        tier_label = tier_name.replace('_', ' ').title()
        print(f"  {tier_label:<20} {stats['n_frames']:<10} {stats['mean_visibility']:<12.3f} "
              f"{stats['mean_mpjpe']:>8.2f} mm   {stats['p90']:>6.2f} mm {stats['p95']:>6.2f} mm")
    
    # Save detailed results
    output_path = os.path.join(output_dir, f'occlusion_analysis_{dataset}_{model_name}.txt')
    with open(output_path, 'w') as f:
        f.write(f"Occlusion Analysis (fixed bins 0.30→1.00): {dataset}\n")
        f.write(f"Model: {model_name.upper()}\n")
        f.write(f"Total frames: {len(per_frame_mpjpe)}\n")
        f.write("="*80 + "\n\n")
        
        f.write("Occlusion metric: Mean 2D keypoint visibility score (per frame)\n")
        f.write(f"  - Higher visibility = less occluded / more visible\n")
        f.write(f"  - Lower visibility = more occluded / less visible\n")
        f.write(f"  - Fixed bins enable cross-dataset comparison\n\n")
        
        f.write(f"Visibility statistics:\n")
        f.write(f"  Min mean visibility: {np.min(mean_visibility):.3f}\n")
        f.write(f"  Max mean visibility: {np.max(mean_visibility):.3f}\n")
        f.write(f"  Overall mean visibility: {np.mean(mean_visibility):.3f}\n")
        f.write(f"  Median mean visibility: {np.median(mean_visibility):.3f}\n\n")
        
        f.write("="*80 + "\n")
        f.write("MPJPE by Occlusion Bin (Fixed Intervals)\n")
        f.write("="*80 + "\n\n")
        
        for tier_name in ordered_keys:
            stats = tier_stats[tier_name]
            tier_label = tier_name.replace('_', ' ').title()
            f.write(f"{tier_label}:\n")
            f.write(f"  N frames: {stats['n_frames']}\n")
            f.write(f"  Mean visibility: {stats['mean_visibility']:.3f}\n")
            f.write(f"  Mean MPJPE: {stats['mean_mpjpe']:.2f} mm\n")
            f.write(f"  Median MPJPE: {stats['median_mpjpe']:.2f} mm\n")
            f.write(f"  Std MPJPE: {stats['std_mpjpe']:.2f} mm\n")
            f.write(f"  25th percentile: {stats['p25']:.2f} mm\n")
            f.write(f"  75th percentile: {stats['p75']:.2f} mm\n")
            f.write(f"  80th percentile: {stats['p80']:.2f} mm\n")
            f.write(f"  90th percentile: {stats['p90']:.2f} mm\n")
            f.write(f"  95th percentile: {stats['p95']:.2f} mm\n")
            f.write(f"  Mean weighted MPJPE: {stats['mean_weighted_mpjpe']:.2f} mm\n")
            f.write(f"  Median weighted MPJPE: {stats['median_weighted_mpjpe']:.2f} mm\n\n")
    
    print(f"  Saved occlusion analysis: {output_path}")
    
    # Visualize representative frames from each bin
    print(f"\n  Selecting frames for visualization...")
    
    vis_frames = []
    for tier_name in ordered_keys:
        mask = None
        # Reconstruct mask for this bin
        i = labels.index(tier_name)
        if i == 0:
            mask = (vis_for_bins <= edges[1])
        else:
            lower, upper = edges[i], edges[i+1]
            mask = (vis_for_bins > lower) & (vis_for_bins <= upper)
        
        if np.sum(mask) == 0:
            continue
        
        tier_indices = np.where(mask)[0]
        tier_vis = mean_visibility[mask]
        
        # Select 10 representative frames from this bin
        if tier_name == "vis_≤0.30":
            # Lowest visibility bin: select frames with lowest visibility
            sorted_idx = np.argsort(tier_vis)[:10]
        elif tier_name == "vis_0.90_1.00":
            # Highest visibility bin: select frames with highest visibility
            sorted_idx = np.argsort(tier_vis)[-10:][::-1]
        else:
            # Middle bins: sample evenly
            n_samples = min(10, len(tier_indices))
            sorted_idx = np.linspace(0, len(tier_indices)-1, n_samples, dtype=int)
        
        for idx in sorted_idx:
            frame_idx = tier_indices[idx]
            vis_frames.append({
                'idx': frame_idx,
                'tier': tier_name,
                'mean_visibility': tier_vis[idx],
                'imgname': imgnames[frame_idx],
                f'{model_name}_error': per_frame_mpjpe[frame_idx],
                'delta': 0.0,
                'rel_delta': 0.0
            })
    
    print(f"  Selected {len(vis_frames)} frames for visualization")
    
    return tier_stats, vis_frames


def compute_coarse_ordinal_depth(joints_3d, threshold_mm=250):
    """Compute coarse ordinal depth from 3D joint coordinates based on depth thresholding.
    
    Joints within threshold_mm of each other are assigned the same rank (bucket).
    This is more forgiving than exact ordinal ranking.
    
    Args:
        joints_3d: (N, 17, 3) array of 3D poses in mm
        threshold_mm: Depth threshold for same bucket (default: 250mm = 0.25m)
    
    Returns:
        (N, 17) array of coarse ordinal ranks
    """
    z_coords = joints_3d[..., 2]  # (N, 17)
    
    # Handle single frame case
    if z_coords.ndim == 1:
        z_coords = z_coords[None, :]
        squeeze_output = True
    else:
        squeeze_output = False
    
    ordinal_depth = np.zeros_like(z_coords, dtype=int)
    for i in range(z_coords.shape[0]):
        z_frame = z_coords[i]
        sorted_indices = np.argsort(z_frame)
        rank = 1
        bucket_start_depth = z_frame[sorted_indices[0]]
        for j in sorted_indices:
            if abs(z_frame[j] - bucket_start_depth) > threshold_mm:
                rank += 1
                bucket_start_depth = z_frame[j]
            ordinal_depth[i, j] = rank
    
    # Squeeze back if input was single frame
    if squeeze_output:
        ordinal_depth = ordinal_depth.squeeze(0)
    
    return ordinal_depth
    
    # Stratify by occlusion level into 4 tiers
    if level == 'frame':
        # PER-FRAME STRATIFICATION
        # Tier 1: Low occlusion (top 33% confidence among frames with vis > 0.3)
        # Tier 2: Medium occlusion (middle 34% among frames with vis > 0.3)
        # Tier 3: High occlusion (bottom 33% among frames with vis > 0.3)
        # Tier 4: Very high occlusion (visibility ≤ 0.3)
        
        # First, separate frames by the 0.3 threshold
        very_high_occ_mask = mean_visibility <= 0.3
        above_threshold_mask = mean_visibility > 0.3
        
        if np.sum(above_threshold_mask) > 0:
            visibility_above_threshold = mean_visibility[above_threshold_mask]
            p33_above = np.percentile(visibility_above_threshold, 33)
            p67_above = np.percentile(visibility_above_threshold, 67)
            
            # Create tier masks
            tier1_mask = (mean_visibility > p67_above)  # High visibility (low occlusion)
            tier2_mask = (mean_visibility > p33_above) & (mean_visibility <= p67_above)  # Medium visibility
            tier3_mask = (mean_visibility > 0.3) & (mean_visibility <= p33_above)  # Low visibility (high occlusion)
            tier4_mask = very_high_occ_mask  # Very low visibility (very high occlusion)
            
            print(f"\n  Occlusion stratification (by mean visibility):")
            print(f"    Tier 1 (low occlusion):       {np.sum(tier1_mask)} frames, mean vis > {p67_above:.3f}")
            print(f"    Tier 2 (medium occlusion):    {np.sum(tier2_mask)} frames, mean vis {p33_above:.3f} to {p67_above:.3f}")
            print(f"    Tier 3 (high occlusion):      {np.sum(tier3_mask)} frames, mean vis 0.3 to {p33_above:.3f}")
            print(f"    Tier 4 (very high occlusion): {np.sum(tier4_mask)} frames, mean vis ≤ 0.3")
        else:
            # All frames have visibility ≤ 0.3 (edge case)
            tier1_mask = np.zeros(len(mean_visibility), dtype=bool)
            tier2_mask = np.zeros(len(mean_visibility), dtype=bool)
            tier3_mask = np.zeros(len(mean_visibility), dtype=bool)
            tier4_mask = very_high_occ_mask
            
            print(f"\n  Occlusion stratification (by mean visibility):")
            print(f"    All {np.sum(tier4_mask)} frames have visibility ≤ 0.3")
    
    else:  # level == 'joint'
        # PER-JOINT STRATIFICATION
        # Separate joints by the 0.3 threshold
        very_high_occ_mask = mean_visibility <= 0.3
        above_threshold_mask = mean_visibility > 0.3
    
    # Get data for tier statistics
    per_frame_mpjpe = results_test['per_frame_mpjpe']
    pred_mm = results_test['pred_mm']
    gt_mm = results_test['gt_mm']
    imgnames = results_test['imgnames']
    
    # Handle per-joint analysis separately
    if level == 'joint':
        # ---------- Per-Joint OCCLUSION BINS (fixed intervals) ----------
        flat_visibility = visibility_scores.flatten()  # (N*J,)
        print(f"\n  Per-joint visibility statistics:")
        print(f"    Visibility range: [{np.min(flat_visibility):.3f}, {np.max(flat_visibility):.3f}]")
        print(f"    Mean visibility: {np.mean(flat_visibility):.3f}")
        print(f"    % joints with vis ≤ 0.3: {np.mean(flat_visibility <= 0.3)*100:.1f}%")

        # Clamp for binning, but keep raw vis for reporting means
        vis_for_bins = np.clip(flat_visibility, 0.0, 1.0)

        # Fixed edges and labels: ≤0.30, (0.30–0.40], …, (0.90–1.00]
        edges  = [0.0, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
        labels = ["vis_≤0.30",
                  "vis_0.30_0.40", "vis_0.40_0.50", "vis_0.50_0.60", "vis_0.60_0.70",
                  "vis_0.70_0.80", "vis_0.80_0.90", "vis_0.90_1.00"]

        # Errors at joint level
        pred_mm_flat = pred_mm.reshape(-1, 3)  # (N*J,3)
        gt_mm_flat   = gt_mm.reshape(-1, 3)    # (N*J,3)
        joint_errors = np.linalg.norm(pred_mm_flat - gt_mm_flat, axis=-1)

        tier_stats_joint = {}
        ordered_keys = []

        print(f"\n  Joint-Level Error by Occlusion Bins (fixed):")
        print(f"  {'Bin':<16} {'N Joints':>9} {'Mean Vis':>10} {'Mean Err':>11} {'Median':>9} {'p90':>9} {'p95':>9}")
        print(f"  {'-'*70}")

        for i, lab in enumerate(labels):
            if i == 0:
                mask = (vis_for_bins <= edges[1])
            else:
                lower, upper = edges[i], edges[i+1]
                mask = (vis_for_bins > lower) & (vis_for_bins <= upper)

            n = int(np.sum(mask))
            if n == 0:
                continue

            errs = joint_errors[mask]
            vis  = flat_visibility[mask]  # use raw (not-clipped) for reporting

            stats = {
                'n_joints': n,
                'mean_visibility': float(np.mean(vis)),
                'mean_error': float(np.mean(errs)),
                'median_error': float(np.median(errs)),
                'std_error': float(np.std(errs)),
                'p25': float(np.percentile(errs, 25)),
                'p75': float(np.percentile(errs, 75)),
                'p90': float(np.percentile(errs, 90)),
                'p95': float(np.percentile(errs, 95)),
            }
            tier_stats_joint[lab] = stats
            ordered_keys.append(lab)

            print(f"  {lab:<16} {n:>9} {stats['mean_visibility']:>10.3f} "
                  f"{stats['mean_error']:>11.2f} {stats['median_error']:>9.2f} "
                  f"{stats['p90']:>9.2f} {stats['p95']:>9.2f}")

        # Save to file
        output_path_joint = os.path.join(output_dir, f'occlusion_analysis_joint_{dataset}_{model_name}.txt')
        with open(output_path_joint, 'w') as f:
            f.write(f"Per-Joint Occlusion Analysis (fixed bins 0.30→1.00): {dataset}\n")
            f.write(f"Model: {model_name.upper()}\n")
            total_joints = pred_mm.shape[0] * pred_mm.shape[1]
            f.write(f"Total joints: {total_joints}\n")
            f.write("="*80 + "\n\n")
            for k in ordered_keys:
                s = tier_stats_joint[k]
                f.write(f"{k}:\n")
                f.write(f"  N joints: {s['n_joints']}\n")
                f.write(f"  Mean visibility: {s['mean_visibility']:.3f}\n")
                f.write(f"  Mean error: {s['mean_error']:.2f} mm\n")
                f.write(f"  Median error: {s['median_error']:.2f} mm\n")
                f.write(f"  Std error: {s['std_error']:.2f} mm\n")
                f.write(f"  p25: {s['p25']:.2f} mm | p75: {s['p75']:.2f} mm | "
                        f"p90: {s['p90']:.2f} mm | p95: {s['p95']:.2f} mm\n\n")

        print(f"  Saved per-joint occlusion analysis: {output_path_joint}")
        return tier_stats_joint, []
    
    tier_stats = {}
    for tier_name, mask in [('low_occlusion', tier1_mask), 
                            ('medium_occlusion', tier2_mask), 
                            ('high_occlusion', tier3_mask),
                            ('very_high_occlusion', tier4_mask)]:
        if np.sum(mask) == 0:
            continue
        
        tier_errors = per_frame_mpjpe[mask]
        tier_visibility = mean_visibility[mask]
        
        # Compute per-joint errors for this tier
        tier_pred = pred_mm[mask]
        tier_gt = gt_mm[mask]
        tier_joint_errors = np.linalg.norm(tier_pred - tier_gt, axis=-1)  # (N_tier, 17)
        
        # Compute visibility-weighted MPJPE
        tier_joint_vis = visibility_scores[mask]  # (N_tier, 17)
        # Weight by inverse visibility (low visibility = higher weight in error)
        # Use max(vis, eps) to avoid division by zero
        eps = 0.01
        inv_weights = 1.0 / np.maximum(tier_joint_vis, eps)  # (N_tier, 17)
        weighted_joint_errors = tier_joint_errors * inv_weights  # (N_tier, 17)
        weighted_mpjpe = np.mean(weighted_joint_errors, axis=1)  # (N_tier,)
        
        tier_stats[tier_name] = {
            'n_frames': np.sum(mask),
            'mean_visibility': np.mean(tier_visibility),
            'mean_mpjpe': np.mean(tier_errors),
            'median_mpjpe': np.median(tier_errors),
            'std_mpjpe': np.std(tier_errors),
            'p25': np.percentile(tier_errors, 25),
            'p75': np.percentile(tier_errors, 75),
            'p90': np.percentile(tier_errors, 90),
            'p95': np.percentile(tier_errors, 95),
            'mean_weighted_mpjpe': np.mean(weighted_mpjpe),
            'median_weighted_mpjpe': np.median(weighted_mpjpe),
        }
    
    # Print comparison
    print(f"\n  MPJPE by Occlusion Level:")
    print(f"  {'Tier':<20} {'N Frames':<10} {'Mean Vis':<12} {'MPJPE':<12} {'Weighted MPJPE':<15}")
    print(f"  {'-'*69}")
    
    for tier_name, stats in tier_stats.items():
        tier_label = tier_name.replace('_', ' ').title()
        print(f"  {tier_label:<20} {stats['n_frames']:<10} {stats['mean_visibility']:<12.3f} "
              f"{stats['mean_mpjpe']:>8.2f} mm   {stats['mean_weighted_mpjpe']:>10.2f} mm")
    
    # Save detailed results
    output_path = os.path.join(output_dir, f'occlusion_analysis_{dataset}_{model_name}.txt')
    with open(output_path, 'w') as f:
        f.write(f"Occlusion Analysis: {dataset}\n")
        f.write(f"Model: {model_name.upper()}\n")
        f.write(f"Total frames: {len(per_frame_mpjpe)}\n")
        f.write("="*80 + "\n\n")
        
        f.write("Occlusion metric: Mean 2D keypoint visibility score (per frame)\n")
        f.write(f"  - Higher visibility = less occluded / more visible\n")
        f.write(f"  - Lower visibility = more occluded / less visible\n\n")
        
        f.write(f"Visibility statistics:\n")
        f.write(f"  Min mean visibility: {np.min(mean_visibility):.3f}\n")
        f.write(f"  Max mean visibility: {np.max(mean_visibility):.3f}\n")
        f.write(f"  Overall mean visibility: {np.mean(mean_visibility):.3f}\n")
        f.write(f"  Median mean visibility: {np.median(mean_visibility):.3f}\n")
        if np.sum(above_threshold_mask) > 0:
            f.write(f"  33rd percentile (above 0.3): {p33_above:.3f}\n")
            f.write(f"  67th percentile (above 0.3): {p67_above:.3f}\n")
        f.write(f"  Frames with visibility ≤ 0.3: {np.sum(very_high_occ_mask)} ({np.mean(very_high_occ_mask)*100:.1f}%)\n\n")
        
        f.write("="*80 + "\n")
        f.write("MPJPE by Occlusion Tier\n")
        f.write("="*80 + "\n\n")
        
        for tier_name, stats in tier_stats.items():
            tier_label = tier_name.replace('_', ' ').title()
            f.write(f"{tier_label}:\n")
            f.write(f"  N frames: {stats['n_frames']}\n")
            f.write(f"  Mean visibility: {stats['mean_visibility']:.3f}\n")
            f.write(f"  Mean MPJPE: {stats['mean_mpjpe']:.2f} mm\n")
            f.write(f"  Median MPJPE: {stats['median_mpjpe']:.2f} mm\n")
            f.write(f"  Std MPJPE: {stats['std_mpjpe']:.2f} mm\n")
            f.write(f"  25th percentile: {stats['p25']:.2f} mm\n")
            f.write(f"  75th percentile: {stats['p75']:.2f} mm\n")
            f.write(f"  90th percentile: {stats['p90']:.2f} mm\n")
            f.write(f"  95th percentile: {stats['p95']:.2f} mm\n")
            f.write(f"  Mean weighted MPJPE (inverse visibility): {stats['mean_weighted_mpjpe']:.2f} mm\n")
            f.write(f"  Median weighted MPJPE: {stats['median_weighted_mpjpe']:.2f} mm\n\n")
        
        # Compute relative differences
        if 'low_occlusion' in tier_stats and 'high_occlusion' in tier_stats:
            low_occ = tier_stats['low_occlusion']['mean_mpjpe']
            high_occ = tier_stats['high_occlusion']['mean_mpjpe']
            delta = high_occ - low_occ
            rel_delta = delta / low_occ if low_occ > 0 else 0
            
            low_occ_weighted = tier_stats['low_occlusion']['mean_weighted_mpjpe']
            high_occ_weighted = tier_stats['high_occlusion']['mean_weighted_mpjpe']
            delta_weighted = high_occ_weighted - low_occ_weighted
            rel_delta_weighted = delta_weighted / low_occ_weighted if low_occ_weighted > 0 else 0
            
            f.write("="*80 + "\n")
            f.write("Comparison: High Occlusion vs Low Occlusion\n")
            f.write("="*80 + "\n")
            f.write(f"  Low occlusion mean MPJPE: {low_occ:.2f} mm\n")
            f.write(f"  High occlusion mean MPJPE: {high_occ:.2f} mm\n")
            f.write(f"  Absolute difference: {delta:+.2f} mm\n")
            f.write(f"  Relative difference: {rel_delta*100:+.1f}%\n\n")
            
            f.write(f"Visibility-weighted MPJPE:\n")
            f.write(f"  Low occlusion: {low_occ_weighted:.2f} mm\n")
            f.write(f"  High occlusion: {high_occ_weighted:.2f} mm\n")
            f.write(f"  Absolute difference: {delta_weighted:+.2f} mm\n")
            f.write(f"  Relative difference: {rel_delta_weighted*100:+.1f}%\n")
    
    print(f"  Saved occlusion analysis: {output_path}")
    
    # Visualize representative frames from each tier
    print(f"\n  Selecting frames for visualization...")
    
    # Select top 10 frames from each tier
    vis_frames = []
    for tier_name, mask in [('low_occlusion', tier1_mask), 
                            ('medium_occlusion', tier2_mask), 
                            ('high_occlusion', tier3_mask),
                            ('very_high_occlusion', tier4_mask)]:
        if np.sum(mask) == 0:
            continue
        
        tier_indices = np.where(mask)[0]
        tier_vis = mean_visibility[mask]
        
        # For low occlusion: select frames with highest visibility
        # For high occlusion: select frames with lowest visibility
        # For very high occlusion: select frames with lowest visibility (all ≤ 0.3)
        if tier_name == 'low_occlusion':
            # Sort by visibility (descending) and take top 10
            sorted_idx = np.argsort(tier_vis)[-10:][::-1]
        elif tier_name in ['high_occlusion', 'very_high_occlusion']:
            # Sort by visibility (ascending) and take top 10
            sorted_idx = np.argsort(tier_vis)[:10]
        else:
            # Middle tier: sample 10 frames evenly spaced
            n_samples = min(10, len(tier_indices))
            sorted_idx = np.linspace(0, len(tier_indices)-1, n_samples, dtype=int)
        
        for idx in sorted_idx:
            frame_idx = tier_indices[idx]
            vis_frames.append({
                'idx': frame_idx,
                'tier': tier_name,
                'mean_visibility': tier_vis[idx],
                'imgname': imgnames[frame_idx],
                f'{model_name}_error': per_frame_mpjpe[frame_idx],
                'delta': 0.0,  # Placeholder for compatibility
                'rel_delta': 0.0  # Placeholder for compatibility
            })
    
    print(f"  Selected {len(vis_frames)} frames for visualization")
    
    return tier_stats, vis_frames


def compute_coarse_ordinal_depth(joints_3d, threshold_mm=250):
    """Compute coarse ordinal depth from 3D joint coordinates based on depth thresholding.
    
    Joints within threshold_mm of each other are assigned the same rank (bucket).
    This is more forgiving than exact ordinal ranking.
    
    Args:
        joints_3d: (N, 17, 3) array of 3D poses in mm
        threshold_mm: Depth threshold for same bucket (default: 250mm = 0.25m)
    
    Returns:
        (N, 17) array of coarse ordinal ranks
    """
    z_coords = joints_3d[..., 2]  # (N, 17)
    
    # Handle single frame case
    if z_coords.ndim == 1:
        z_coords = z_coords[None, :]
        squeeze_output = True
    else:
        squeeze_output = False
    
    ordinal_depth = np.zeros_like(z_coords, dtype=int)
    for i in range(z_coords.shape[0]):
        z_frame = z_coords[i]
        sorted_indices = np.argsort(z_frame)
        rank = 1
        bucket_start_depth = z_frame[sorted_indices[0]]
        for j in sorted_indices:
            if abs(z_frame[j] - bucket_start_depth) > threshold_mm:
                rank += 1
                bucket_start_depth = z_frame[j]
            ordinal_depth[i, j] = rank
    
    # Squeeze back if input was single frame
    if squeeze_output:
        ordinal_depth = ordinal_depth.squeeze(0)
    
    return ordinal_depth


def analyze_depth_ordering_occlusion(results_test, dataset, output_dir, model_name='xy'):
    """Analyze MPJPE stratified by DAV depth ordering accuracy (occlusion proxy).
    
    This function uses the pre-computed input_depth_ordinal_accuracy_per_frame metric
    to stratify frames by depth ordering quality:
    - High accuracy = clear scene, good depth ordering
    - Low accuracy = occluded/ambiguous scene, poor depth ordering
    
    Args:
        results_test: Dict containing test dataset results with input_depth_ordinal_accuracy_per_frame
        dataset: Test dataset name
        output_dir: Directory to save analysis results
        model_name: Model variant name (default: 'xy')
    
    Returns:
        Tuple of (tier_stats, vis_frames) for visualization
    """
    print(f"\n{'='*60}")
    print(f"Depth Ordering Occlusion Analysis: {dataset}")
    print(f"{'='*60}")
    
    # Check if DAV depth ordering accuracy is available
    dav_accuracy_per_frame = results_test.get('input_depth_ordinal_accuracy_per_frame', None)
    
    if dav_accuracy_per_frame is None:
        print(f"  ✗ DAV depth ordering accuracy not available for {dataset}")
        print(f"  Skipping depth ordering occlusion analysis")
        return None
    
    print(f"  ✓ DAV depth ordering accuracy available")
    print(f"  Shape: {dav_accuracy_per_frame.shape}")
    
    # Print statistics
    print(f"\n  DAV Depth Ordering Accuracy Statistics:")
    print(f"    Mean: {np.mean(dav_accuracy_per_frame):.4f}")
    print(f"    Median: {np.median(dav_accuracy_per_frame):.4f}")
    print(f"    Min: {np.min(dav_accuracy_per_frame):.4f}")
    print(f"    Max: {np.max(dav_accuracy_per_frame):.4f}")
    print(f"    Interpretation: Higher = better depth ordering (less occlusion)")
    
    # Stratify into quartiles by depth ordering quality
    # Q1 = best depth ordering (highest accuracy, least occluded)
    # Q4 = worst depth ordering (lowest accuracy, most occluded)
    q25 = np.percentile(dav_accuracy_per_frame, 25)
    q50 = np.percentile(dav_accuracy_per_frame, 50)
    q75 = np.percentile(dav_accuracy_per_frame, 75)
    
    q1_mask = dav_accuracy_per_frame >= q75  # Best 25% (high accuracy)
    q2_mask = (dav_accuracy_per_frame >= q50) & (dav_accuracy_per_frame < q75)
    q3_mask = (dav_accuracy_per_frame >= q25) & (dav_accuracy_per_frame < q50)
    q4_mask = dav_accuracy_per_frame < q25  # Worst 25% (low accuracy)
    
    print(f"\n  Depth Ordering Stratification:")
    print(f"    Q1 (best, least occluded):    {np.sum(q1_mask)} frames, accuracy ≥ {q75:.4f}")
    print(f"    Q2 (good):                    {np.sum(q2_mask)} frames, accuracy {q50:.4f} to {q75:.4f}")
    print(f"    Q3 (poor):                    {np.sum(q3_mask)} frames, accuracy {q25:.4f} to {q50:.4f}")
    print(f"    Q4 (worst, most occluded):    {np.sum(q4_mask)} frames, accuracy < {q25:.4f}")
    
    # Get data for tier statistics
    per_frame_mpjpe = results_test['per_frame_mpjpe']
    imgnames = results_test['imgnames']
    
    # Compute MPJPE statistics for each quartile
    tier_stats = {}
    for tier_name, mask in [('Q1_clear', q1_mask), 
                            ('Q2_mild', q2_mask), 
                            ('Q3_moderate', q3_mask),
                            ('Q4_occluded', q4_mask)]:
        if np.sum(mask) == 0:
            continue
        
        tier_errors = per_frame_mpjpe[mask]
        tier_dav_accuracy = dav_accuracy_per_frame[mask]
        
        tier_stats[tier_name] = {
            'n_frames': np.sum(mask),
            'mean_dav_accuracy': np.mean(tier_dav_accuracy),
            'median_dav_accuracy': np.median(tier_dav_accuracy),
            'mean_mpjpe': np.mean(tier_errors),
            'median_mpjpe': np.median(tier_errors),
            'std_mpjpe': np.std(tier_errors),
            'p25': np.percentile(tier_errors, 25),
            'p75': np.percentile(tier_errors, 75),
            'p90': np.percentile(tier_errors, 90),
            'p95': np.percentile(tier_errors, 95),
        }
    
    # Print comparison
    print(f"\n  MPJPE by Depth Ordering Quality:")
    print(f"  {'Tier':<20} {'N Frames':<10} {'Mean DAV Acc':<15} {'Mean MPJPE':<12} {'Median MPJPE':<15}")
    print(f"  {'-'*72}")
    
    for tier_name, stats in tier_stats.items():
        tier_label = tier_name.replace('_', ' ').title()
        print(f"  {tier_label:<20} {stats['n_frames']:<10} {stats['mean_dav_accuracy']:<15.4f} "
              f"{stats['mean_mpjpe']:>8.2f} mm   {stats['median_mpjpe']:>10.2f} mm")
    
    # Save detailed results
    output_path = os.path.join(output_dir, f'depth_ordering_occlusion_{dataset}_{model_name}.txt')
    with open(output_path, 'w') as f:
        f.write(f"Depth Ordering Occlusion Analysis: {dataset}\n")
        f.write(f"Model: {model_name.upper()}\n")
        f.write(f"Total frames: {len(per_frame_mpjpe)}\n")
        f.write("="*80 + "\n\n")
        
        f.write("Occlusion metric: DAV input depth ordering accuracy (pairwise depth relationships)\n")
        f.write(f"  - Higher accuracy = better depth ordering (clear scene, less occluded)\n")
        f.write(f"  - Lower accuracy = poor depth ordering (ambiguous/occluded scene)\n\n")
        
        f.write(f"DAV Depth Ordering Accuracy Statistics:\n")
        f.write(f"  Mean: {np.mean(dav_accuracy_per_frame):.4f}\n")
        f.write(f"  Median: {np.median(dav_accuracy_per_frame):.4f}\n")
        f.write(f"  25th percentile: {q25:.4f}\n")
        f.write(f"  50th percentile: {q50:.4f}\n")
        f.write(f"  75th percentile: {q75:.4f}\n\n")
        
        f.write("="*80 + "\n")
        f.write("MPJPE by Depth Ordering Quality\n")
        f.write("="*80 + "\n\n")
        
        for tier_name, stats in tier_stats.items():
            tier_label = tier_name.replace('_', ' ').title()
            f.write(f"{tier_label}:\n")
            f.write(f"  N frames: {stats['n_frames']}\n")
            f.write(f"  Mean DAV accuracy: {stats['mean_dav_accuracy']:.4f}\n")
            f.write(f"  Median DAV accuracy: {stats['median_dav_accuracy']:.4f}\n")
            f.write(f"  Mean MPJPE: {stats['mean_mpjpe']:.2f} mm\n")
            f.write(f"  Median MPJPE: {stats['median_mpjpe']:.2f} mm\n")
            f.write(f"  Std MPJPE: {stats['std_mpjpe']:.2f} mm\n")
            f.write(f"  25th percentile: {stats['p25']:.2f} mm\n")
            f.write(f"  75th percentile: {stats['p75']:.2f} mm\n")
            f.write(f"  90th percentile: {stats['p90']:.2f} mm\n")
            f.write(f"  95th percentile: {stats['p95']:.2f} mm\n\n")
        
        # Compute relative differences
        if 'Q1_clear' in tier_stats and 'Q4_occluded' in tier_stats:
            clear = tier_stats['Q1_clear']['mean_mpjpe']
            occluded = tier_stats['Q4_occluded']['mean_mpjpe']
            delta = occluded - clear
            rel_delta = delta / clear if clear > 0 else 0
            
            f.write("="*80 + "\n")
            f.write("Comparison: Occluded (Q4) vs Clear (Q1)\n")
            f.write("="*80 + "\n")
            f.write(f"  Q1 (clear) mean MPJPE: {clear:.2f} mm\n")
            f.write(f"  Q4 (occluded) mean MPJPE: {occluded:.2f} mm\n")
            f.write(f"  Absolute difference: {delta:+.2f} mm\n")
            f.write(f"  Relative difference: {rel_delta*100:+.1f}%\n")
    
    print(f"  Saved depth ordering occlusion analysis: {output_path}")
    
    # Visualize representative frames from each tier
    print(f"\n  Selecting frames for visualization...")
    
    # Select top 10 frames from each tier
    vis_frames = []
    for tier_name, mask in [('Q1_clear', q1_mask), 
                            ('Q2_mild', q2_mask), 
                            ('Q3_moderate', q3_mask),
                            ('Q4_occluded', q4_mask)]:
        if np.sum(mask) == 0:
            continue
        
        tier_indices = np.where(mask)[0]
        tier_dav_acc = dav_accuracy_per_frame[mask]
        
        # For clear (Q1): select frames with highest accuracy
        # For occluded (Q4): select frames with lowest accuracy
        # For Q2/Q3: sample 10 frames evenly spaced
        if tier_name == 'Q1_clear':
            sorted_idx = np.argsort(tier_dav_acc)[-10:][::-1]
        elif tier_name == 'Q4_occluded':
            sorted_idx = np.argsort(tier_dav_acc)[:10]
        else:
            n_samples = min(10, len(tier_indices))
            sorted_idx = np.linspace(0, len(tier_indices)-1, n_samples, dtype=int)
        
        for idx in sorted_idx:
            frame_idx = tier_indices[idx]
            vis_frames.append({
                'idx': frame_idx,
                'tier': tier_name,
                'dav_ordering_accuracy': tier_dav_acc[idx],
                'imgname': imgnames[frame_idx],
                f'{model_name}_error': per_frame_mpjpe[frame_idx],
                'delta': 0.0,  # Placeholder for compatibility
                'rel_delta': 0.0  # Placeholder for compatibility
            })
    
    print(f"  Selected {len(vis_frames)} frames for visualization")
    
    return tier_stats, vis_frames


def analyze_geometric_occlusion(results_test, dataset, output_dir, model_name='xy'):
    """Analyze MPJPE stratified by geometric self-occlusion (3D-based).
    
    This function uses 3D GT poses and 2D GT keypoints to compute physical
    self-occlusion: a joint is occluded if a closer joint (smaller Z) overlaps
    it in the 2D image plane within a joint-specific radius.
    
    Args:
        results_test: Dict containing test dataset results
        dataset: Test dataset name
        output_dir: Directory to save analysis results
        model_name: Model variant name (default: 'xy')
    
    Returns:
        Tuple of (tier_stats, vis_frames) for visualization
    """
    print(f"\n{'='*60}")
    print(f"Geometric Self-Occlusion Analysis: {dataset}")
    print(f"{'='*60}")
    
    # Get GT 3D poses and 2D keypoints
    gt_mm = results_test['gt_mm']
    input_2d_keypoints = results_test.get('input_2d_keypoints', None)
    
    if input_2d_keypoints is None:
        print(f"  ✗ 2D keypoints not available for {dataset}")
        print(f"  Skipping geometric occlusion analysis")
        return None
    
    print(f"  ✓ 2D keypoints and 3D GT poses available")
    
    # Define joint names for adaptive radius lookup
    joint_names = [
        'root', 'right_hip', 'right_knee', 'right_foot',
        'left_hip', 'left_knee', 'left_foot', 'spine',
        'thorax', 'neck_base', 'head', 'left_shoulder',
        'left_elbow', 'left_wrist', 'right_shoulder',
        'right_elbow', 'right_wrist'
    ]
    
    # Compute geometric self-occlusion visibility
    print(f"  Computing geometric self-occlusion (3D depth + 2D overlap)...")
    vis_flags = compute_joint_visibility_2dprox(
        input_2d_keypoints, 
        gt_mm, 
        joint_names=joint_names,
        eps_mm=10  # Minimum depth gap to consider occlusion
    )
    
    # Sanitize vis_flags to ensure binary and complete
    vis_flags = np.nan_to_num(vis_flags, nan=1.0)
    vis_flags = (vis_flags > 0).astype(np.uint8)
    
    # Summarize occlusion
    occ_frame_scores, occ_joint_rates = summarize_self_occlusion(vis_flags)
    
    print(f"  ✓ Computed geometric occlusion for {len(occ_frame_scores)} frames")
    print(f"\n  Geometric Self-Occlusion Statistics:")
    print(f"    Mean occluded fraction (per frame): {occ_frame_scores.mean():.3f}")
    print(f"    Median occluded fraction: {np.median(occ_frame_scores):.3f}")
    print(f"    Min occluded fraction: {occ_frame_scores.min():.3f}")
    print(f"    Max occluded fraction: {occ_frame_scores.max():.3f}")
    print(f"\n  Per-Joint Occlusion Rates:")
    for j, joint_name in enumerate(joint_names):
        print(f"    {joint_name:<20}: {occ_joint_rates[j]:.3f}")
    
    # Stratify into quartiles by occlusion score
    # Q1 = least occluded (lowest score)
    # Q4 = most occluded (highest score)
    
    # Avoid degenerate quantiles when many frames have same score (e.g., all zeros)
    unique_vals = np.unique(occ_frame_scores)
    if len(unique_vals) <= 2:
        # Degenerate case: use min, max, and single quantile
        q25 = occ_frame_scores.min()
        q50 = occ_frame_scores.min()
        q75 = np.percentile(occ_frame_scores, 75)
    else:
        q25 = np.percentile(occ_frame_scores, 25)
        q50 = np.percentile(occ_frame_scores, 50)
        q75 = np.percentile(occ_frame_scores, 75)
    
    q1_mask = occ_frame_scores <= q25  # Least occluded
    q2_mask = (occ_frame_scores > q25) & (occ_frame_scores <= q50)
    q3_mask = (occ_frame_scores > q50) & (occ_frame_scores <= q75)
    q4_mask = occ_frame_scores > q75  # Most occluded
    
    print(f"\n  Geometric Occlusion Stratification:")
    print(f"    Q1 (least occluded):  {np.sum(q1_mask)} frames, score ≤ {q25:.3f}")
    print(f"    Q2 (mild):            {np.sum(q2_mask)} frames, score {q25:.3f} to {q50:.3f}")
    print(f"    Q3 (moderate):        {np.sum(q3_mask)} frames, score {q50:.3f} to {q75:.3f}")
    print(f"    Q4 (most occluded):   {np.sum(q4_mask)} frames, score > {q75:.3f}")
    
    # Compute MPJPE statistics for each quartile
    per_frame_mpjpe = results_test['per_frame_mpjpe']
    imgnames = results_test['imgnames']
    
    tier_stats = {}
    for tier_name, mask in [('Q1_clear', q1_mask), 
                            ('Q2_mild', q2_mask), 
                            ('Q3_moderate', q3_mask),
                            ('Q4_occluded', q4_mask)]:
        if np.sum(mask) == 0:
            continue
        
        tier_errors = per_frame_mpjpe[mask]
        tier_occ_scores = occ_frame_scores[mask]
        
        # Filter out NaN errors (frames with all joints filtered by confidence threshold)
        valid_mask = np.isfinite(tier_errors)
        if not np.any(valid_mask):
            # All frames in this tier have NaN errors - skip tier
            print(f"    Warning: {tier_name} has no valid frames (all NaN MPJPE)")
            continue
        
        tier_errors_valid = tier_errors[valid_mask]
        tier_occ_scores_valid = tier_occ_scores[valid_mask]
        
        tier_stats[tier_name] = {
            'n_frames': int(np.sum(mask)),
            'n_valid_frames': int(np.sum(valid_mask)),
            'mean_occlusion': float(np.mean(tier_occ_scores_valid)),
            'median_occlusion': float(np.median(tier_occ_scores_valid)),
            'mean_mpjpe': float(np.mean(tier_errors_valid)),
            'median_mpjpe': float(np.median(tier_errors_valid)),
            'std_mpjpe': float(np.std(tier_errors_valid)),
            'p25': float(np.percentile(tier_errors_valid, 25)),
            'p75': float(np.percentile(tier_errors_valid, 75)),
            'p90': float(np.percentile(tier_errors_valid, 90)),
            'p95': float(np.percentile(tier_errors_valid, 95)),
        }
    
    # Print comparison
    print(f"\n  MPJPE by Geometric Occlusion Level:")
    print(f"  {'Tier':<20} {'N Frames':<10} {'Mean Occ':<12} {'Mean MPJPE':<12} {'Median MPJPE':<15}")
    print(f"  {'-'*69}")
    
    for tier_name, stats in tier_stats.items():
        tier_label = tier_name.replace('_', ' ').title()
        print(f"  {tier_label:<20} {stats['n_frames']:<10} {stats['mean_occlusion']:<12.3f} "
              f"{stats['mean_mpjpe']:>8.2f} mm   {stats['median_mpjpe']:>10.2f} mm")
    
    # Save detailed results
    output_path = os.path.join(output_dir, f'geometric_occlusion_{dataset}_{model_name}.txt')
    with open(output_path, 'w') as f:
        f.write(f"Geometric Self-Occlusion Analysis: {dataset}\n")
        f.write(f"Model: {model_name.upper()}\n")
        f.write(f"Total frames: {len(per_frame_mpjpe)}\n")
        f.write("="*80 + "\n\n")
        
        f.write("Occlusion metric: 3D-based geometric self-occlusion\n")
        f.write(f"  - A joint is occluded if a closer joint (smaller depth Z) overlaps it\n")
        f.write(f"  - Overlap determined by 2D proximity within joint-specific radius\n")
        f.write(f"  - Higher score = more occluded (more joints blocked by closer joints)\n\n")
        
        f.write(f"Geometric Occlusion Statistics:\n")
        f.write(f"  Mean occluded fraction: {occ_frame_scores.mean():.3f}\n")
        f.write(f"  Median occluded fraction: {np.median(occ_frame_scores):.3f}\n")
        f.write(f"  25th percentile: {q25:.3f}\n")
        f.write(f"  50th percentile: {q50:.3f}\n")
        f.write(f"  75th percentile: {q75:.3f}\n\n")
        
        f.write("Per-Joint Occlusion Rates:\n")
        for j, joint_name in enumerate(joint_names):
            f.write(f"  {joint_name:<20}: {occ_joint_rates[j]:.3f}\n")
        f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("MPJPE by Geometric Occlusion Level\n")
        f.write("="*80 + "\n\n")
        
        for tier_name, stats in tier_stats.items():
            tier_label = tier_name.replace('_', ' ').title()
            f.write(f"{tier_label}:\n")
            f.write(f"  N frames: {stats['n_frames']}\n")
            f.write(f"  Mean occlusion score: {stats['mean_occlusion']:.3f}\n")
            f.write(f"  Median occlusion score: {stats['median_occlusion']:.3f}\n")
            f.write(f"  Mean MPJPE: {stats['mean_mpjpe']:.2f} mm\n")
            f.write(f"  Median MPJPE: {stats['median_mpjpe']:.2f} mm\n")
            f.write(f"  Std MPJPE: {stats['std_mpjpe']:.2f} mm\n")
            f.write(f"  25th percentile: {stats['p25']:.2f} mm\n")
            f.write(f"  75th percentile: {stats['p75']:.2f} mm\n")
            f.write(f"  90th percentile: {stats['p90']:.2f} mm\n")
            f.write(f"  95th percentile: {stats['p95']:.2f} mm\n\n")
        
        # Compute relative differences
        if 'Q1_clear' in tier_stats and 'Q4_occluded' in tier_stats:
            clear = tier_stats['Q1_clear']['mean_mpjpe']
            occluded = tier_stats['Q4_occluded']['mean_mpjpe']
            delta = occluded - clear
            rel_delta = delta / clear if clear > 0 else 0
            
            f.write("="*80 + "\n")
            f.write("Comparison: Most Occluded (Q4) vs Least Occluded (Q1)\n")
            f.write("="*80 + "\n")
            f.write(f"  Q1 (clear) mean MPJPE: {clear:.2f} mm\n")
            f.write(f"  Q4 (occluded) mean MPJPE: {occluded:.2f} mm\n")
            f.write(f"  Absolute difference: {delta:+.2f} mm\n")
            f.write(f"  Relative difference: {rel_delta*100:+.1f}%\n")
    
    print(f"  Saved geometric occlusion analysis: {output_path}")
    
    # Visualize representative frames from each tier
    print(f"\n  Selecting frames for visualization...")
    
    # Select top 10 frames from each tier
    vis_frames = []
    for tier_name, mask in [('Q1_clear', q1_mask), 
                            ('Q2_mild', q2_mask), 
                            ('Q3_moderate', q3_mask),
                            ('Q4_occluded', q4_mask)]:
        if np.sum(mask) == 0:
            continue
        
        tier_indices = np.where(mask)[0]
        tier_occ = occ_frame_scores[mask]
        
        # For clear (Q1): select frames with lowest occlusion
        # For occluded (Q4): select frames with highest occlusion
        # For Q2/Q3: sample 10 frames evenly spaced
        if tier_name == 'Q1_clear':
            sorted_idx = np.argsort(tier_occ)[:10]
        elif tier_name == 'Q4_occluded':
            sorted_idx = np.argsort(tier_occ)[-10:][::-1]
        else:
            n_samples = min(10, len(tier_indices))
            sorted_idx = np.linspace(0, len(tier_indices)-1, n_samples, dtype=int)
        
        for idx in sorted_idx:
            frame_idx = tier_indices[idx]
            vis_frames.append({
                'idx': frame_idx,
                'tier': tier_name,
                'geometric_occlusion': tier_occ[idx],
                'imgname': imgnames[frame_idx],
                f'{model_name}_error': per_frame_mpjpe[frame_idx],
                'delta': 0.0,  # Placeholder for compatibility
                'rel_delta': 0.0  # Placeholder for compatibility
            })
    
    print(f"  Selected {len(vis_frames)} frames for visualization")
    
    return tier_stats, vis_frames


def _sequence_key_from_imgname(name: str) -> str:
    """
    Heuristic: strip the trailing frame token so consecutive frames share a key.
    Works for names like '..._frame01503.jpg' or '.../seq123/frame_000123.png'.
    Falls back to the basename without extension if 'frame' not present.
    """
    base = os.path.basename(str(name))
    if 'frame' in base:
        return base.rsplit('frame', 1)[0]
    return os.path.splitext(base)[0]


def robust_local_speed(coords_mm: np.ndarray,
                       imgnames,
                       window: int = 10,
                       exclude_root: bool = True) -> np.ndarray:
    """
    Per-frame local motion speed (mm/frame) computed within each clip segment.
    Uses mean joint displacement (excluding root by default) and smooths
    with a centered moving average per segment.
    """
    coords = coords_mm[:, 1:, :] if exclude_root else coords_mm  # (N, J, 3)
    N = coords.shape[0]
    disp = np.zeros(N, dtype=float)
    speeds = np.zeros(N, dtype=float)

    for lo, hi in _segments_from_imgnames(imgnames):  # segment = [lo, hi)
        # finite differences within the segment
        for i in range(lo + 1, hi):
            d = coords[i] - coords[i - 1]                      # (J, 3)
            disp[i] = np.nanmean(np.linalg.norm(d, axis=-1))   # scalar

        # smooth within the segment only
        if window and window > 1 and (hi - lo) > 1:
            k = np.ones(int(window), dtype=float) / float(window)
            speeds[lo:hi] = np.convolve(disp[lo:hi], k, mode='same')
        else:
            speeds[lo:hi] = disp[lo:hi]

    return speeds


def robust_local_speed_per_joint(coords_mm: np.ndarray,
                                 imgnames,
                                 window: int = 10,
                                 exclude_root: bool = True) -> np.ndarray:
    """
    Per-joint local motion speed (mm/frame), computed and smoothed per segment.
    Returns (N, J) where J = 16 if exclude_root=True else 17.
    """
    coords = coords_mm[:, 1:, :] if exclude_root else coords_mm  # (N, J, 3)
    N, J, _ = coords.shape
    disp = np.zeros((N, J), dtype=float)
    smoothed = np.zeros((N, J), dtype=float)

    for lo, hi in _segments_from_imgnames(imgnames):
        for i in range(lo + 1, hi):
            d = coords[i] - coords[i - 1]                 # (J, 3)
            disp[i] = np.linalg.norm(d, axis=-1)          # (J,)

        if window and window > 1 and (hi - lo) > 1:
            k = np.ones(int(window), dtype=float) / float(window)
            for j in range(J):
                smoothed[lo:hi, j] = np.convolve(disp[lo:hi, j], k, mode='same')
        else:
            smoothed[lo:hi] = disp[lo:hi]

    return smoothed


# def _clip_key(name):
#     """Extract clip/sequence ID from image filename for motion analysis."""
#     import os
#     d = os.path.dirname(name)
#     return d if d else name.split('/')[0]
def _clip_key(name: str):
    """Extract sequence prefix (clip ID) from filename for motion-speed grouping."""
    import os
    base = os.path.basename(name)
    # Example: "TS1_000001.jpg" → "TS1"
    if '_' in base:
        return base.split('_')[0]
    # fallback to directory name if present
    d = os.path.dirname(name)
    return d if d else base



def _segments_from_imgnames(imgnames):
    """Return list of (start, end) index pairs for contiguous frames in same clip."""
    segs = []
    if len(imgnames) == 0:
        return segs
    # ipdb.set_trace()
    cur_key = _clip_key(imgnames[0])
    start = 0
    for i in range(1, len(imgnames)):
        k = _clip_key(imgnames[i])
        if k != cur_key:
            segs.append((start, i))   # [start, i)
            start = i
            cur_key = k
    segs.append((start, len(imgnames)))
    return segs


def _finite_diff_speed_mm(gt_mm, lo, hi):
    """
    Finite-diff speed (mm/frame) per joint for a single segment [lo, hi).
    Central difference where possible, one-sided at ends.
    Returns speed_seg with shape (L, J)
    """
    seg = gt_mm[lo:hi]         # (L, J, 3)
    L, J, _ = seg.shape
    speed = np.zeros((L, J), dtype=np.float32)
    if L == 1:
        return speed
    # central difference interior
    if L >= 3:
        v_mid = (seg[2:] - seg[:-2]) * 0.5         # (L-2, J, 3)
        speed[1:-1] = np.linalg.norm(v_mid, axis=-1)
    # forward/backward ends
    v0  = seg[1]  - seg[0]      # (J, 3)
    vN1 = seg[-1] - seg[-2]     # (J, 3)
    speed[0]  = np.linalg.norm(v0,  axis=-1)
    speed[-1] = np.linalg.norm(vN1, axis=-1)
    return speed


def _moving_average_same_2d(x, window):
    """
    Centered moving average along time for all joints in a segment.
    x: (L, J) array; window: int ≥1. Uses true variable-length divisor at edges.
    Returns smoothed (L, J).
    """
    L, J = x.shape
    if window <= 1 or L == 0:
        return x.copy()
    half_left  = window // 2
    half_right = window - half_left - 1
    # prefix sums for O(1) window sums per t
    csum = np.concatenate([np.zeros((1, J), dtype=x.dtype), np.cumsum(x, axis=0)], axis=0)  # (L+1, J)
    out = np.empty_like(x)
    for t in range(L):
        lo = max(0, t - half_left)
        hi = min(L - 1, t + half_right)
        # prefix indices are +1
        s  = csum[hi + 1] - csum[lo]
        n  = (hi - lo + 1)
        out[t] = s / max(1, n)
    return out


def _smoothed_local_speed_mm(gt_mm, imgnames, window=10):
    """
    Compute per-joint local speed (mm/frame), then smooth with a centered
    moving average of length 'window', without crossing clip boundaries.
    Returns (N, J) array.
    """
    N, J, _ = gt_mm.shape
    speed = np.zeros((N, J), dtype=np.float32)
    for lo, hi in _segments_from_imgnames(imgnames):
        sp = _finite_diff_speed_mm(gt_mm, lo, hi)            # (L, J)
        sp = _moving_average_same_2d(sp, window=window)      # (L, J)
        speed[lo:hi] = sp
    return speed


def analyze_motion_speed(results_test, dataset, output_dir, model_name='xy', level='frame', 
                         window=10, num_bins=5, fixed_edges=None):
    """
    Analyze MPJPE stratified by smoothed local motion speed.
    
    This function:
    1. Computes local speed (mm/frame) via finite differences
    2. Smooths speed with centered moving average (window frames)
    3. Stratifies frames/joints by speed bins
    4. Computes MPJPE statistics for each speed tier
    
    Args:
        results_test: Dict containing test dataset results
        dataset: Test dataset name
        output_dir: Directory to save analysis results
        model_name: Model variant name (default: 'xy')
        level: Analysis level ('frame' or 'joint')
        window: Smoothing window length in frames (default: 10)
        num_bins: Number of quantile bins if fixed_edges is None (default: 5)
        fixed_edges: Optional list of fixed bin edges in mm/frame
    
    Returns:
        Tuple of (tier_stats, vis_frames) for visualization
    """
    print(f"\n{'='*60}")
    print(f"Motion Speed Analysis (smoothed): {dataset} (level={level}, window={window})")
    print(f"{'='*60}")

    gt_mm   = results_test['gt_mm']              # (N, J, 3) - root-centered
    pred_mm = results_test['pred_mm']            # (N, J, 3) - root-centered
    names   = results_test['imgnames']           # list[str] len N
    pf_mpj  = results_test.get('per_frame_mpjpe')  # (N,) or None

    # 1) Compute smoothed local speed using robust sequence-aware method
    print(f"  Computing smoothed local speed (window={window} frames)...")
    
    # Define body part groups (joint indices)
    body_parts = {
        'upper_appendages': [11, 12, 13, 14, 15, 16],  # shoulders, elbows, wrists
        'lower_appendages': [1, 2, 3, 4, 5, 6],        # hips, knees, feet
        'torso': [7, 8],                                # spine, thorax
        'head': [9, 10],                                # neck_base, head
        'all_appendages': [1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16],  # arms + legs
    }
    
    # Initialize body part dicts - always compute for use in later sections
    body_part_speeds = {}
    body_part_errors = {}
    
    if level == 'frame':
        # Compute per-body-part speeds for detailed analysis
        sp_joint_all = robust_local_speed_per_joint(
            gt_mm,
            names,
            window=window,
            exclude_root=False  # Include root to get all 17 joints
        )  # (N, 17)
        
        # Compute body-part speeds and errors for all body parts
        for part_name, joint_indices in body_parts.items():
            # Average speed across joints in this body part
            body_part_speeds[part_name] = np.nanmean(sp_joint_all[:, joint_indices], axis=1)  # (N,)
            
            # Compute body-part-specific errors (mean 3D error across joints in this part)
            pred_part = pred_mm[:, joint_indices, :]  # (N, n_joints, 3)
            gt_part = gt_mm[:, joint_indices, :]      # (N, n_joints, 3)
            part_errors = np.linalg.norm(pred_part - gt_part, axis=-1)  # (N, n_joints)
            body_part_errors[part_name] = np.nanmean(part_errors, axis=1)  # (N,)
    
    if level == 'frame':
        # Per-frame speed (mean across joints, excluding root)
        speeds = robust_local_speed(
            gt_mm,                   # use GT to measure true motion
            names,
            window=window,
            exclude_root=True
        )
        speed_metric = speeds  # (N,)
        unit = "frames"
        
    elif level == 'root':
        # Root motion speed (body translation through space)
        # Use pre-computed root motion speeds from results
        if 'root_motion_speed' in results_test and 'gt_speed' in results_test['root_motion_speed']:
            speed_metric = results_test['root_motion_speed']['gt_speed']  # (N-1,) - one less due to diff
            # Pad with first value to match frame count
            speed_metric = np.concatenate([[speed_metric[0]], speed_metric])  # (N,)
            unit = "frames"
            print(f"  Using pre-computed root motion speeds")
        else:
            print(f"  ✗ Root motion speeds not available in results")
            print(f"  Skipping root motion analysis")
            return None, []
        
    else:
        # Per-joint speed
        sp_joint = robust_local_speed_per_joint(
            gt_mm,
            names,
            window=window,
            exclude_root=True
        )
        speed_metric = sp_joint.reshape(-1)  # (N*J,) where J=16
        unit = "joints"
    
    # 2) Basic stats and degenerate check
    sp_min = float(np.nanmin(speed_metric)) if len(speed_metric) else 0.0
    sp_max = float(np.nanmax(speed_metric)) if len(speed_metric) else 0.0
    sp_mean = float(np.nanmean(speed_metric)) if len(speed_metric) else 0.0
    sp_median = float(np.nanmedian(speed_metric)) if len(speed_metric) else 0.0
    
    print(f"  Speed statistics (overall):")
    print(f"    Min: {sp_min:.2f} mm/frame")
    print(f"    Max: {sp_max:.2f} mm/frame")
    print(f"    Mean: {sp_mean:.2f} mm/frame")
    print(f"    Median: {sp_median:.2f} mm/frame")
    
    # Print body-part-specific speeds if available
    if level == 'frame' and body_part_speeds is not None:
        print(f"\n  Body-part speed statistics:")
        for part_name in ['upper_appendages', 'lower_appendages', 'torso', 'head', 'all_appendages']:
            part_speed = body_part_speeds[part_name]
            print(f"    {part_name.replace('_', ' ').title():<20}: "
                  f"mean={np.nanmean(part_speed):>6.2f} mm/frame, "
                  f"median={np.nanmedian(part_speed):>6.2f} mm/frame, "
                  f"max={np.nanmax(part_speed):>6.2f} mm/frame")
    
    if not np.isfinite(sp_max) or (sp_max - sp_min) < 1e-6:
        print("  ✗ Degenerate smoothed speed distribution.")
        return None, []

    # 3) build bin edges (quantiles by default; or use fixed_edges if provided)
    valid = np.isfinite(speed_metric)
    if not np.any(valid):
        print("  ✗ No valid smoothed speed values.")
        return None, []

    sp_valid = speed_metric[valid]
    
    if fixed_edges is not None and len(fixed_edges) >= 2:
        edges = np.array(fixed_edges, dtype=np.float32)
        print(f"  Using fixed bin edges")
    else:
        q = np.linspace(0, 1, num_bins + 1)
        edges = np.quantile(sp_valid, q)
        edges = np.unique(edges)
        if len(edges) < 2:
            print("  ✗ Degenerate smoothed speed distribution.")
            return None, []
        print(f"  Using {len(edges)-1} quantile-based bins")

    print("\n  Smoothed speed bin edges (mm/frame):")
    for i in range(len(edges) - 1):
        print(f"    Bin {i+1}: [{edges[i]:.2f}, {edges[i+1]:.2f}]")

    def _mask_for_bin(sp, lo, hi, first, last):
        if first:
            return (sp >= lo) & (sp <= hi)
        else:
            return (sp >  lo) & (sp <= hi)

    # 4) compute stats per bin (frame, root, or joint)
    tier_stats = {}
    ordered_bins = []
    if level in ['frame', 'root']:
        # Frame-level or root-level: speed_metric is (N,)
        # ensure per_frame_mpjpe exists
        if pf_mpj is None or not np.any(np.isfinite(pf_mpj)):
            # compute from pred/gt if missing
            frame_errs = np.linalg.norm(pred_mm - gt_mm, axis=-1).mean(axis=-1)  # (N,)
        else:
            frame_errs = pf_mpj
        
        # Ensure frame_errs and speed_metric have same length
        n = min(len(frame_errs), len(speed_metric))
        frame_errs = frame_errs[:n]
        speed_metric_aligned = speed_metric[:n]

        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i+1]
            m = _mask_for_bin(speed_metric_aligned, lo, hi, i == 0, i == len(edges) - 2)
            m &= np.isfinite(frame_errs)
            n_valid = int(np.sum(m))
            if n_valid == 0:
                continue
            errs = frame_errs[m]
            sp   = speed_metric_aligned[m]
            key  = f"speed_bin_{i+1}"
            ordered_bins.append(key)
            tier_stats[key] = {
                'n_frames': n_valid,
                'mean_speed': float(np.mean(sp)),
                'median_speed': float(np.median(sp)),
                'mean_mpjpe': float(np.mean(errs)),
                'median_mpjpe': float(np.median(errs)),
                'std_mpjpe': float(np.std(errs)),
                'p25': float(np.percentile(errs, 25)),
                'p75': float(np.percentile(errs, 75)),
                'p90': float(np.percentile(errs, 90)),
                'p95': float(np.percentile(errs, 95)),
            }

        level_label = "Root Motion" if level == 'root' else "Frame-Level"
        print(f"\n  MPJPE by Smoothed Speed ({level_label}):")
        print(f"  {'Bin':<14} {'N Frames':>9} {'Mean Spd':>10} {'MPJPE':>10} {'p90':>10} {'p95':>10}")
        print("  " + "-"*63)
        for k in ordered_bins:
            s = tier_stats[k]
            print(f"  {k:<14} {s['n_frames']:>9} {s['mean_speed']:>10.2f} {s['mean_mpjpe']:>10.2f} "
                  f"{s['p90']:>10.2f} {s['p95']:>10.2f}")

        suffix = 'root' if level == 'root' else 'frame'
        out = os.path.join(output_dir, f"speed_analysis_{dataset}_{model_name}_{suffix}.txt")

    elif level == 'joint':
        # joint-level MPJPE
        joint_errs = np.linalg.norm(pred_mm - gt_mm, axis=-1).reshape(-1)  # (N*J,)
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i+1]
            m = _mask_for_bin(speed_metric, lo, hi, i == 0, i == len(edges) - 2)
            m &= np.isfinite(joint_errs)
            n = int(np.sum(m))
            if n == 0:
                continue
            errs = joint_errs[m]
            sp   = speed_metric[m]
            key  = f"speed_bin_{i+1}"
            ordered_bins.append(key)
            tier_stats[key] = {
                'n_joints': n,
                'mean_speed': float(np.mean(sp)),
                'median_speed': float(np.median(sp)),
                'mean_error': float(np.mean(errs)),
                'median_error': float(np.median(errs)),
                'std_error': float(np.std(errs)),
                'p25': float(np.percentile(errs, 25)),
                'p75': float(np.percentile(errs, 75)),
                'p90': float(np.percentile(errs, 90)),
                'p95': float(np.percentile(errs, 95)),
            }

        print(f"\n  Joint Error by Smoothed Speed (joint-level):")
        print(f"  {'Bin':<14} {'N Joints':>9} {'Mean Spd':>10} {'Mean Err':>10} {'Median':>10} {'p90':>10} {'p95':>10}")
        print("  " + "-"*75)
        for k in ordered_bins:
            s = tier_stats[k]
            print(f"  {k:<14} {s['n_joints']:>9} {s['mean_speed']:>10.2f} {s['mean_error']:>10.2f} "
                  f"{s['median_error']:>10.2f} {s['p90']:>10.2f} {s['p95']:>10.2f}")

        out = os.path.join(output_dir, f"speed_analysis_{dataset}_{model_name}_joint.txt")

    # 5) save
    with open(out, "w") as f:
        f.write(f"Motion Speed Analysis (Smoothed): {dataset}\n")
        f.write(f"Model: {model_name.upper()} | Level: {level} | Window: {window} frames\n")
        f.write("="*80 + "\n\n")
        f.write("Speed computation:\n")
        f.write("  - Local speed via finite differences (central diff where possible)\n")
        f.write(f"  - Smoothed with centered moving average (window={window} frames)\n")
        f.write("  - Speed computed per joint, respects clip boundaries\n\n")
        
        # Write body-part-specific speed statistics if available
        if level == 'frame' and body_part_speeds is not None:
            f.write("Body-Part Speed Statistics:\n")
            f.write("="*80 + "\n")
            for part_name in ['upper_appendages', 'lower_appendages', 'torso', 'head', 'all_appendages']:
                part_speed = body_part_speeds[part_name]
                f.write(f"\n{part_name.replace('_', ' ').title()}:\n")
                f.write(f"  Mean: {np.nanmean(part_speed):.2f} mm/frame\n")
                f.write(f"  Median: {np.nanmedian(part_speed):.2f} mm/frame\n")
                f.write(f"  Min: {np.nanmin(part_speed):.2f} mm/frame\n")
                f.write(f"  Max: {np.nanmax(part_speed):.2f} mm/frame\n")
                f.write(f"  Std: {np.nanstd(part_speed):.2f} mm/frame\n")
            f.write("\n" + "="*80 + "\n\n")
        
        f.write("Overall Speed Bin Edges (mm/frame):\n")
        for i in range(len(edges) - 1):
            f.write(f"  Bin {i+1}: [{edges[i]:.4f}, {edges[i+1]:.4f}]\n")
        f.write("\n")
        for k in ordered_bins:
            s = tier_stats[k]
            if level in ['frame', 'root']:
                f.write(f"{k}:\n")
                f.write(f"  N frames: {s['n_frames']}\n")
                f.write(f"  Mean speed: {s['mean_speed']:.2f} mm/frame\n")
                f.write(f"  Median speed: {s['median_speed']:.2f} mm/frame\n")
                f.write(f"  Mean MPJPE: {s['mean_mpjpe']:.2f} mm\n")
                f.write(f"  Median MPJPE: {s['median_mpjpe']:.2f} mm\n")
                f.write(f"  Std MPJPE: {s['std_mpjpe']:.2f} mm\n")
                f.write(f"  25th percentile: {s['p25']:.2f} mm\n")
                f.write(f"  75th percentile: {s['p75']:.2f} mm\n")
                f.write(f"  90th percentile: {s['p90']:.2f} mm\n")
                f.write(f"  95th percentile: {s['p95']:.2f} mm\n\n")
            else:  # level == 'joint'
                f.write(f"{k}:\n")
                f.write(f"  N joints: {s['n_joints']}\n")
                f.write(f"  Mean speed: {s['mean_speed']:.2f} mm/frame\n")
                f.write(f"  Median speed: {s['median_speed']:.2f} mm/frame\n")
                f.write(f"  Mean error: {s['mean_error']:.2f} mm\n")
                f.write(f"  Median error: {s['median_error']:.2f} mm\n")
                f.write(f"  Std error: {s['std_error']:.2f} mm\n")
                f.write(f"  25th percentile: {s['p25']:.2f} mm\n")
                f.write(f"  75th percentile: {s['p75']:.2f} mm\n")
                f.write(f"  90th percentile: {s['p90']:.2f} mm\n")
                f.write(f"  95th percentile: {s['p95']:.2f} mm\n\n")
        
        # Add body-part-stratified analysis if available
        if level == 'frame' and body_part_speeds is not None:
            f.write("\n" + "="*80 + "\n")
            f.write("Body-Part MPJPE Stratified by Body-Part Motion Speed\n")
            f.write("="*80 + "\n\n")
            f.write("For each body part, stratify frames by that part's speed (tertiles)\n")
            f.write("and report BODY-PART-SPECIFIC MPJPE (not overall frame MPJPE) for each speed tier.\n")
            f.write("This shows how each body part's error changes with its own motion speed.\n\n")
            
            for part_name in ['upper_appendages', 'lower_appendages', 'torso', 'head', 'all_appendages']:
                if part_name not in body_part_speeds or part_name not in body_part_errors:
                    continue
                part_speed = body_part_speeds[part_name]
                part_error = body_part_errors[part_name]  # Body-part-specific errors
                part_label = part_name.replace('_', ' ').title()
                
                f.write(f"\n{part_label}:\n")
                f.write("="*60 + "\n")
                
                # Tertile stratification (low/med/high speed)
                valid = np.isfinite(part_speed) & np.isfinite(part_error)
                if not np.any(valid):
                    f.write("  (No valid data)\n")
                    continue
                
                t33 = np.percentile(part_speed[valid], 33)
                t67 = np.percentile(part_speed[valid], 67)
                
                low_mask = valid & (part_speed <= t33)
                med_mask = valid & (part_speed > t33) & (part_speed <= t67)
                high_mask = valid & (part_speed > t67)
                
                for tier_name, mask in [('Low speed', low_mask), ('Medium speed', med_mask), ('High speed', high_mask)]:
                    if not np.any(mask):
                        continue
                    errs = part_error[mask]  # Use body-part-specific errors
                    spd = part_speed[mask]
                    f.write(f"  {tier_name} (N={np.sum(mask)} frames):\n")
                    f.write(f"    Speed range: [{np.min(spd):.2f}, {np.max(spd):.2f}] mm/frame\n")
                    f.write(f"    Mean speed: {np.mean(spd):.2f} mm/frame\n")
                    f.write(f"    Mean {part_label} MPJPE: {np.mean(errs):.2f} mm\n")
                    f.write(f"    Median {part_label} MPJPE: {np.median(errs):.2f} mm\n")
                    f.write(f"    p90 {part_label} MPJPE: {np.percentile(errs, 90):.2f} mm\n")
                    f.write(f"    p95 {part_label} MPJPE: {np.percentile(errs, 95):.2f} mm\n\n")

    print(f"  Saved motion speed analysis: {out}")
    
    # Print body-part-specific MPJPE by speed tiers (tertiles)
    if level == 'frame' and body_part_speeds is not None:
        print(f"\n  Body-Part MPJPE by Speed Tier (Tertiles):")
        print(f"  " + "="*80)
        
        for part_name in ['upper_appendages', 'lower_appendages', 'torso', 'head', 'all_appendages']:
            part_speed = body_part_speeds[part_name]
            part_error = body_part_errors[part_name]
            part_label = part_name.replace('_', ' ').title()
            
            print(f"\n  {part_label}:")
            print(f"  {'Tier':<15} {'N Frames':<10} {'Speed Range':<20} {'Mean':<12} {'Median':<12} {'p90':<10} {'p95':<10}")
            print(f"  {'-'*89}")
            
            # Tertile stratification
            valid = np.isfinite(part_speed) & np.isfinite(part_error)
            if not np.any(valid):
                print(f"    (No valid data)")
                continue
            
            t33 = np.percentile(part_speed[valid], 33)
            t67 = np.percentile(part_speed[valid], 67)
            
            low_mask = valid & (part_speed <= t33)
            med_mask = valid & (part_speed > t33) & (part_speed <= t67)
            high_mask = valid & (part_speed > t67)
            
            for tier_name, mask in [('Low', low_mask), ('Medium', med_mask), ('High', high_mask)]:
                if not np.any(mask):
                    continue
                errs = part_error[mask]
                spd = part_speed[mask]
                speed_range = f"[{np.min(spd):.1f}, {np.max(spd):.1f}]"
                print(f"  {tier_name:<15} {np.sum(mask):<10} {speed_range:<20} "
                      f"{np.mean(errs):>8.2f} mm {np.median(errs):>8.2f} mm {np.percentile(errs, 90):>6.2f} mm {np.percentile(errs, 95):>6.2f} mm")
        
        print(f"\n  Body-part motion vs body-part MPJPE correlation:")
        print(f"  {'Body Part':<25} {'Speed→Error Corr':<20} {'Interpretation'}")
        print(f"  {'-'*75}")
        
        for part_name in ['upper_appendages', 'lower_appendages', 'torso', 'head', 'all_appendages']:
            part_speed = body_part_speeds[part_name]
            part_error = body_part_errors[part_name]
            valid = np.isfinite(part_speed) & np.isfinite(part_error)
            if np.sum(valid) > 10:
                corr = np.corrcoef(part_speed[valid], part_error[valid])[0, 1]
                if corr > 0.3:
                    interp = "Strong positive (faster→harder)"
                elif corr > 0.1:
                    interp = "Moderate positive"
                elif corr > -0.1:
                    interp = "Weak/no correlation"
                elif corr > -0.3:
                    interp = "Moderate negative"
                else:
                    interp = "Strong negative (faster→easier)"
                
                part_label = part_name.replace('_', ' ').title()
                print(f"  {part_label:<25} {corr:>8.3f}              {interp}")
        print()
    
    # For frame-level analysis, select representative frames from each bin
    vis_frames = []
    if level == 'frame':
        print(f"\n  Selecting frames for visualization...")
        for i, k in enumerate(ordered_bins):
            lo, hi = edges[i], edges[i+1]
            m = _mask_for_bin(speed_metric, lo, hi, i == 0, i == len(edges) - 2)
            m &= np.isfinite(frame_errs)
            if not np.any(m):
                continue
            
            tier_indices = np.where(m)[0]
            tier_speeds = speed_metric[m]
            
            # Select 10 representative frames from this bin
            if i == 0:
                # Lowest speed bin: select frames with lowest speed
                sorted_idx = np.argsort(tier_speeds)[:10]
            elif i == len(ordered_bins) - 1:
                # Highest speed bin: select frames with highest speed
                sorted_idx = np.argsort(tier_speeds)[-10:][::-1]
            else:
                # Middle bins: sample evenly
                n_samples = min(10, len(tier_indices))
                sorted_idx = np.linspace(0, len(tier_indices)-1, n_samples, dtype=int)
            
            for idx in sorted_idx:
                frame_idx = tier_indices[idx]
                vis_frames.append({
                    'idx': frame_idx,
                    'tier': k,
                    'mean_speed': tier_speeds[idx],
                    'imgname': names[frame_idx],
                    f'{model_name}_error': frame_errs[frame_idx],
                    'delta': 0.0,
                    'rel_delta': 0.0
                })
        
        print(f"  Selected {len(vis_frames)} frames for visualization")
    
    return tier_stats, vis_frames


def analyze_detection_quality(results_test, dataset, output_dir, model_name='xy', level='frame'):
    """
    This function:
    1. Loads GT 2D keypoints
    2. Computes distance between predicted 2D and GT 2D keypoints
    3. Stratifies frames by detection quality (quartiles: Q1=best, Q4=worst)
    4. Computes MPJPE statistics for each quality tier
    
    Args:
        results_test: Dict containing test dataset results
        dataset: Test dataset name
        output_dir: Directory to save analysis results
        model_name: Model variant name (default: 'xy')
    
    Returns:
        Tuple of (tier_stats, vis_frames) for visualization
    """
    print(f"\n{'='*60}")
    print(f"2D Detection Quality Analysis: {dataset} (level={level})")
    print(f"{'='*60}")
    
    # GT 2D keypoints path
    gt_2d_path_map = {
        '3dhp': "/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_3dhp_test_all_v6_casp.npz",
        '3dpw': "/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_3dpw_all_v6_gtdav.npz",
        'fit3d': "/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_fit3d_all_v6_casp.npz", #gt 2d
    }
    
    dataset_key = '3dhp' if '3dhp' in dataset.lower() else '3dpw'
    if dataset_key not in gt_2d_path_map:
        print(f"  ✗ GT 2D keypoints not available for {dataset}")
        print(f"  Skipping detection quality analysis")
        return None
    
    gt_2d_path = gt_2d_path_map[dataset_key]
    
    # Load GT 2D keypoints from 'part' field (first 2 channels are x, y)
    try:
        print(f"  Loading GT 2D keypoints from: {gt_2d_path}")
        gt_2d_data = np.load(gt_2d_path, allow_pickle=True)
        part_data = gt_2d_data['part']  # Shape: (N, 17, 3) where channels are [x, y, vis]
        gt_2d_keypoints = part_data[..., :2]  # Extract first 2 channels: (N, 17, 2)
        print(f"  ✓ Loaded GT 2D keypoints from 'part' field, shape: {gt_2d_keypoints.shape}")
    except Exception as e:
        print(f"  ✗ Failed to load GT 2D keypoints: {e}")
        print(f"  Skipping detection quality analysis")
        return None
    
    # Get predicted 2D keypoints from results
    pred_2d_keypoints = results_test.get('input_2d_keypoints', None)
    if pred_2d_keypoints is None:
        print(f"  ✗ Predicted 2D keypoints not available in results")
        print(f"  Skipping detection quality analysis")
        return None
    
    print(f"  ✓ Predicted 2D keypoints available, shape: {pred_2d_keypoints.shape}")
    print(f"  Analysis level: {level}")
    
    # Align lengths
    n_frames = min(len(gt_2d_keypoints), len(pred_2d_keypoints), len(results_test['per_frame_mpjpe']))
    gt_2d_keypoints = gt_2d_keypoints[:n_frames]
    pred_2d_keypoints = pred_2d_keypoints[:n_frames]
    
    # Compute 2D detection error (Euclidean distance per joint)
    detection_errors_per_joint = np.linalg.norm(pred_2d_keypoints - gt_2d_keypoints, axis=-1)  # (N, 17)
    
    # Filter out detections > 100px off (mark as invalid)
    max_2d_error_px = 100.0
    valid_detections = detection_errors_per_joint <= max_2d_error_px  # (N, 17)
    n_invalid_joints = np.sum(~valid_detections)
    n_total_joints = detection_errors_per_joint.size
    
    print(f"\n  Filtering detections > {max_2d_error_px:.0f} px:")
    print(f"    Invalid joints: {n_invalid_joints} / {n_total_joints} ({n_invalid_joints/n_total_joints*100:.2f}%)")
    
    # Set invalid detections to NaN for exclusion from statistics
    detection_errors_per_joint_filtered = detection_errors_per_joint.copy()
    detection_errors_per_joint_filtered[~valid_detections] = np.nan
    
    if level == 'frame':
        # PER-FRAME ANALYSIS: Mean 2D error per frame (excluding invalid joints)
        detection_error_per_frame = np.nanmean(detection_errors_per_joint_filtered, axis=1)  # (N,)
        
        # Filter out frames where all joints are invalid
        valid_frames = ~np.isnan(detection_error_per_frame)
        n_invalid_frames = np.sum(~valid_frames)
        
        print(f"    Invalid frames (all joints > {max_2d_error_px:.0f}px): {n_invalid_frames} / {n_frames}")
        
        if np.sum(valid_frames) == 0:
            print(f"  ✗ No valid frames remaining after filtering")
            return None
        
        print(f"\n  2D Detection Error Statistics (per-frame, filtered):")
        print(f"    Mean: {np.nanmean(detection_error_per_frame):.2f} pixels")
        print(f"    Median: {np.nanmedian(detection_error_per_frame):.2f} pixels")
        print(f"    Min: {np.nanmin(detection_error_per_frame):.2f} pixels")
        print(f"    Max: {np.nanmax(detection_error_per_frame):.2f} pixels")
    else:  # level == 'joint'
        # PER-JOINT ANALYSIS: Flatten to (N*17,) for joint-level stratification
        detection_error_flat = detection_errors_per_joint_filtered.flatten()  # (N*17,)
        valid_joints_flat = ~np.isnan(detection_error_flat)
        
        if np.sum(valid_joints_flat) == 0:
            print(f"  ✗ No valid joints remaining after filtering")
            return None
        
        print(f"\n  2D Detection Error Statistics (per-joint, filtered):")
        print(f"    Mean: {np.nanmean(detection_error_flat):.2f} pixels")
        print(f"    Median: {np.nanmedian(detection_error_flat):.2f} pixels")
        print(f"    Min: {np.nanmin(detection_error_flat):.2f} pixels")
        print(f"    Max: {np.nanmax(detection_error_flat):.2f} pixels")
    
    # Stratify into quartiles by detection quality
    if level == 'frame':
        # PER-FRAME STRATIFICATION
        # Use only valid frames for percentile computation
        valid_errors = detection_error_per_frame[valid_frames]
        
        q25 = np.nanpercentile(valid_errors, 25)
        q50 = np.nanpercentile(valid_errors, 50)
        q75 = np.nanpercentile(valid_errors, 75)
        
        # Create masks for valid frames only
        q1_mask = valid_frames & (detection_error_per_frame <= q25)
        q2_mask = valid_frames & (detection_error_per_frame > q25) & (detection_error_per_frame <= q50)
        q3_mask = valid_frames & (detection_error_per_frame > q50) & (detection_error_per_frame <= q75)
        q4_mask = valid_frames & (detection_error_per_frame > q75)
        
        unit = "frames"
        print(f"\n  Detection Quality Stratification (by mean 2D error per frame):")
        print(f"    Q1 (best 25%):   {np.sum(q1_mask)} {unit}, 2D error ≤ {q25:.2f} px")
        print(f"    Q2 (good 25%):   {np.sum(q2_mask)} {unit}, 2D error {q25:.2f} to {q50:.2f} px")
        print(f"    Q3 (poor 25%):   {np.sum(q3_mask)} {unit}, 2D error {q50:.2f} to {q75:.2f} px")
        print(f"    Q4 (worst 25%):  {np.sum(q4_mask)} {unit}, 2D error > {q75:.2f} px")
    else:  # level == 'joint'
        # PER-JOINT STRATIFICATION
        valid_errors_flat = detection_error_flat[valid_joints_flat]
        
        q25 = np.nanpercentile(valid_errors_flat, 25)
        q50 = np.nanpercentile(valid_errors_flat, 50)
        q75 = np.nanpercentile(valid_errors_flat, 75)
        
        # Create masks for valid joints only
        q1_mask = valid_joints_flat & (detection_error_flat <= q25)
        q2_mask = valid_joints_flat & (detection_error_flat > q25) & (detection_error_flat <= q50)
        q3_mask = valid_joints_flat & (detection_error_flat > q50) & (detection_error_flat <= q75)
        q4_mask = valid_joints_flat & (detection_error_flat > q75)
        
        unit = "joints"
        print(f"\n  Detection Quality Stratification (by 2D error per joint):")
        print(f"    Q1 (best 25%):   {np.sum(q1_mask)} {unit}, 2D error ≤ {q25:.2f} px")
        print(f"    Q2 (good 25%):   {np.sum(q2_mask)} {unit}, 2D error {q25:.2f} to {q50:.2f} px")
        print(f"    Q3 (poor 25%):   {np.sum(q3_mask)} {unit}, 2D error {q50:.2f} to {q75:.2f} px")
        print(f"    Q4 (worst 25%):  {np.sum(q4_mask)} {unit}, 2D error > {q75:.2f} px")
    
    # Compute MPJPE statistics for each quartile
    per_frame_mpjpe = results_test['per_frame_mpjpe'][:n_frames]
    imgnames = results_test['imgnames'][:n_frames]
    
    if level == 'frame':
        # PER-FRAME STATISTICS
        tier_stats = {}
        for tier_name, mask in [('Q1_best', q1_mask), 
                                ('Q2_good', q2_mask), 
                                ('Q3_poor', q3_mask),
                                ('Q4_worst', q4_mask)]:
            if np.sum(mask) == 0:
                continue
            
            tier_errors = per_frame_mpjpe[mask]
            tier_2d_errors = detection_error_per_frame[mask]
            
            tier_stats[tier_name] = {
                'n_frames': np.sum(mask),
                'mean_2d_error': np.nanmean(tier_2d_errors),
                'mean_mpjpe': np.mean(tier_errors),
                'median_mpjpe': np.median(tier_errors),
                'std_mpjpe': np.std(tier_errors),
                'p25': np.percentile(tier_errors, 25),
                'p75': np.percentile(tier_errors, 75),
                'p90': np.percentile(tier_errors, 90),
                'p95': np.percentile(tier_errors, 95),
            }
    else:  # level == 'joint'
        # PER-JOINT STATISTICS
        # Get 3D data
        pred_mm = results_test['pred_mm'][:n_frames]
        gt_mm = results_test['gt_mm'][:n_frames]
        
        # Flatten to joint level
        pred_mm_flat = pred_mm.reshape(-1, 3)  # (N*17, 3)
        gt_mm_flat = gt_mm.reshape(-1, 3)      # (N*17, 3)
        
        # Compute per-joint 3D errors
        joint_errors_3d = np.linalg.norm(pred_mm_flat - gt_mm_flat, axis=-1)  # (N*17,)
        
        tier_stats = {}
        for tier_name, mask in [('Q1_best', q1_mask), 
                                ('Q2_good', q2_mask), 
                                ('Q3_poor', q3_mask),
                                ('Q4_worst', q4_mask)]:
            if np.sum(mask) == 0:
                continue
            
            tier_errors_3d = joint_errors_3d[mask]
            tier_2d_errors = detection_error_flat[mask]
            
            tier_stats[tier_name] = {
                'n_joints': np.sum(mask),
                'mean_2d_error': np.nanmean(tier_2d_errors),
                'mean_error': np.mean(tier_errors_3d),
                'median_error': np.median(tier_errors_3d),
                'std_error': np.std(tier_errors_3d),
                'p25': np.percentile(tier_errors_3d, 25),
                'p75': np.percentile(tier_errors_3d, 75),
                'p90': np.percentile(tier_errors_3d, 90),
                'p95': np.percentile(tier_errors_3d, 95),
            }
    
    # Print comparison
    if level == 'frame':
        print(f"\n  MPJPE by 2D Detection Quality:")
        print(f"  {'Quartile':<15} {'N Frames':<10} {'Mean 2D Err':<13} {'Mean MPJPE':<12} {'Median MPJPE':<15}")
        print(f"  {'-'*65}")
        
        for tier_name, stats in tier_stats.items():
            tier_label = tier_name.replace('_', ' ').upper()
            print(f"  {tier_label:<15} {stats['n_frames']:<10} {stats['mean_2d_error']:>9.2f} px   "
                  f"{stats['mean_mpjpe']:>8.2f} mm   {stats['median_mpjpe']:>10.2f} mm")
    else:  # level == 'joint'
        print(f"\n  Joint-Level Error by 2D Detection Quality:")
        print(f"  {'Quartile':<15} {'N Joints':<10} {'Mean 2D Err':<13} {'Mean Error':<12} {'Median Error':<15}")
        print(f"  {'-'*65}")
        
        for tier_name, stats in tier_stats.items():
            tier_label = tier_name.replace('_', ' ').upper()
            print(f"  {tier_label:<15} {stats['n_joints']:<10} {stats['mean_2d_error']:>9.2f} px   "
                  f"{stats['mean_error']:>8.2f} mm   {stats['median_error']:>10.2f} mm")
    
    # Save detailed results
    suffix = "_joint" if level == 'joint' else ""
    output_path = os.path.join(output_dir, f'detection_quality_analysis{suffix}_{dataset}_{model_name}.txt')
    with open(output_path, 'w') as f:
        f.write(f"2D Detection Quality Analysis: {dataset} (level={level})\n")
        f.write(f"Model: {model_name.upper()}\n")
        f.write(f"Total frames: {n_frames}\n")
        f.write(f"Filtered out detections > {max_2d_error_px:.0f} px\n")
        f.write("="*80 + "\n\n")
        
        if level == 'frame':
            f.write("Analysis: Stratify frames by 2D detection quality (mean pred 2D vs GT 2D error per frame)\n")
            f.write(f"  - Lower 2D error = better detection quality\n")
            f.write(f"  - Higher 2D error = worse detection quality (noisy input)\n\n")
            
            f.write(f"2D Detection Error Statistics (per-frame, filtered):\n")
            f.write(f"  Mean: {np.nanmean(detection_error_per_frame):.2f} pixels\n")
            f.write(f"  Median: {np.nanmedian(detection_error_per_frame):.2f} pixels\n")
            f.write(f"  25th percentile: {q25:.2f} pixels\n")
            f.write(f"  50th percentile: {q50:.2f} pixels\n")
            f.write(f"  75th percentile: {q75:.2f} pixels\n\n")
        else:  # level == 'joint'
            f.write("Analysis: Stratify joints by 2D detection quality (pred 2D vs GT 2D error per joint)\n")
            f.write(f"  - Lower 2D error = better detection quality\n")
            f.write(f"  - Higher 2D error = worse detection quality (noisy input)\n\n")
            
            f.write(f"2D Detection Error Statistics (per-joint, filtered):\n")
            f.write(f"  Mean: {np.nanmean(detection_error_flat):.2f} pixels\n")
            f.write(f"  Median: {np.nanmedian(detection_error_flat):.2f} pixels\n")
            f.write(f"  25th percentile: {q25:.2f} pixels\n")
            f.write(f"  50th percentile: {q50:.2f} pixels\n")
            f.write(f"  75th percentile: {q75:.2f} pixels\n\n")
        
        f.write("="*80 + "\n")
        if level == 'frame':
            f.write("MPJPE by Detection Quality Quartile\n")
        else:
            f.write("Joint-Level 3D Error by Detection Quality Quartile\n")
        f.write("="*80 + "\n\n")
        
        for tier_name, stats in tier_stats.items():
            tier_label = tier_name.replace('_', ' ').upper()
            f.write(f"{tier_label}:\n")
            
            if level == 'frame':
                f.write(f"  N frames: {stats['n_frames']}\n")
                f.write(f"  Mean 2D error: {stats['mean_2d_error']:.2f} pixels\n")
                f.write(f"  Mean MPJPE: {stats['mean_mpjpe']:.2f} mm\n")
                f.write(f"  Median MPJPE: {stats['median_mpjpe']:.2f} mm\n")
                f.write(f"  Std MPJPE: {stats['std_mpjpe']:.2f} mm\n")
                f.write(f"  25th percentile: {stats['p25']:.2f} mm\n")
                f.write(f"  75th percentile: {stats['p75']:.2f} mm\n")
                f.write(f"  90th percentile: {stats['p90']:.2f} mm\n")
                f.write(f"  95th percentile: {stats['p95']:.2f} mm\n\n")
            else:  # level == 'joint'
                f.write(f"  N joints: {stats['n_joints']}\n")
                f.write(f"  Mean 2D error: {stats['mean_2d_error']:.2f} pixels\n")
                f.write(f"  Mean error: {stats['mean_error']:.2f} mm\n")
                f.write(f"  Median error: {stats['median_error']:.2f} mm\n")
                f.write(f"  Std error: {stats['std_error']:.2f} mm\n")
                f.write(f"  25th percentile: {stats['p25']:.2f} mm\n")
                f.write(f"  75th percentile: {stats['p75']:.2f} mm\n")
                f.write(f"  90th percentile: {stats['p90']:.2f} mm\n")
                f.write(f"  95th percentile: {stats['p95']:.2f} mm\n\n")
        
        # Compute relative differences
        if 'Q1_best' in tier_stats and 'Q4_worst' in tier_stats:
            if level == 'frame':
                best_mpjpe = tier_stats['Q1_best']['mean_mpjpe']
                worst_mpjpe = tier_stats['Q4_worst']['mean_mpjpe']
                delta = worst_mpjpe - best_mpjpe
                rel_delta = delta / best_mpjpe if best_mpjpe > 0 else 0
                
                f.write("="*80 + "\n")
                f.write("Comparison: Worst Detection Quality vs Best Detection Quality\n")
                f.write("="*80 + "\n")
                f.write(f"  Q1 (best) mean MPJPE: {best_mpjpe:.2f} mm\n")
                f.write(f"  Q4 (worst) mean MPJPE: {worst_mpjpe:.2f} mm\n")
                f.write(f"  Absolute difference: {delta:+.2f} mm\n")
                f.write(f"  Relative difference: {rel_delta*100:+.1f}%\n")
            else:  # level == 'joint'
                best_error = tier_stats['Q1_best']['mean_error']
                worst_error = tier_stats['Q4_worst']['mean_error']
                delta = worst_error - best_error
                rel_delta = delta / best_error if best_error > 0 else 0
                
                f.write("="*80 + "\n")
                f.write("Comparison: Worst Detection Quality vs Best Detection Quality\n")
                f.write("="*80 + "\n")
                f.write(f"  Q1 (best) mean error: {best_error:.2f} mm\n")
                f.write(f"  Q4 (worst) mean error: {worst_error:.2f} mm\n")
                f.write(f"  Absolute difference: {delta:+.2f} mm\n")
                f.write(f"  Relative difference: {rel_delta*100:+.1f}%\n")
    
    print(f"  Saved detection quality analysis: {output_path}")
    
    # Select frames for visualization (only for per-frame analysis)
    if level == 'frame':
        print(f"\n  Selecting frames for visualization...")
        vis_frames = []
        
        for tier_name, mask in [('Q1_best', q1_mask), 
                                ('Q2_good', q2_mask), 
                                ('Q3_poor', q3_mask),
                                ('Q4_worst', q4_mask)]:
            if np.sum(mask) == 0:
                continue
            
            tier_indices = np.where(mask)[0]
            tier_2d_errors = detection_error_per_frame[mask]
            
            # For Q1: select frames with lowest 2D errors
            # For Q4: select frames with highest 2D errors
            # For Q2/Q3: sample 10 frames evenly spaced
            if tier_name == 'Q1_best':
                sorted_idx = np.argsort(tier_2d_errors)[:10]
            elif tier_name == 'Q4_worst':
                sorted_idx = np.argsort(tier_2d_errors)[-10:][::-1]
            else:
                n_samples = min(10, len(tier_indices))
                sorted_idx = np.linspace(0, len(tier_indices)-1, n_samples, dtype=int)
            
            for idx in sorted_idx:
                frame_idx = tier_indices[idx]
                vis_frames.append({
                    'idx': frame_idx,
                    'tier': tier_name,
                    'detection_2d_error': tier_2d_errors[idx],
                    'imgname': imgnames[frame_idx],
                    f'{model_name}_error': per_frame_mpjpe[frame_idx],
                    'delta': 0.0,  # Placeholder for compatibility
                    'rel_delta': 0.0  # Placeholder for compatibility
                })
        
        print(f"  Selected {len(vis_frames)} frames for visualization")
        
        return tier_stats, vis_frames
    else:
        # Per-joint level: no visualization
        return tier_stats, []


def analyze_body_part_improvements(results_model1, results_model2, dataset, output_dir, model1='xy', model2='xycd'):
    """Analyze MPJPE improvements by body part (independent of motion speed).
    
    This function compares per-body-part errors between two models to identify
    which joint groups benefit most from augmented inputs (e.g., depth, confidence).
    
    Body parts analyzed:
    - Upper appendages (shoulders, elbows, wrists)
    - Lower appendages (hips, knees, feet)
    - Torso (spine, thorax)
    - Head (neck_base, head)
    - All appendages (arms + legs combined)
    """
    print(f"\n{'='*60}")
    print(f"Body Part Analysis: {dataset}")
    print(f"Comparing {model1.upper()} vs {model2.upper()}")
    print(f"{'='*60}")
    
    # Define body part groups (joint indices)
    body_parts = {
        'upper_appendages': [11, 12, 13, 14, 15, 16],
        'lower_appendages': [1, 2, 3, 4, 5, 6],
        'torso': [7, 8],
        'head': [9, 10],
        'all_appendages': [1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16],
    }
    
    joint_names = [
        'root', 'right_hip', 'right_knee', 'right_foot',
        'left_hip', 'left_knee', 'left_foot', 'spine',
        'thorax', 'neck_base', 'head', 'left_shoulder',
        'left_elbow', 'left_wrist', 'right_shoulder',
        'right_elbow', 'right_wrist'
    ]
    
    # Get 3D poses and compute per-joint errors
    pred_mm_1 = results_model1['pred_mm']
    pred_mm_2 = results_model2['pred_mm']
    gt_mm = results_model1['gt_mm']
    
    # Per-joint errors for each model (N, 17)
    joint_errors_1 = np.linalg.norm(pred_mm_1 - gt_mm, axis=-1)
    joint_errors_2 = np.linalg.norm(pred_mm_2 - gt_mm, axis=-1)
    
    # Compute body part statistics
    body_part_stats = {}
    
    print(f"\n  Body Part Error Statistics:")
    print(f"  {'Body Part':<25} {model1.upper():<15} {model2.upper():<15} {'Delta':<15} {'Rel %':<10}")
    print(f"  {'-'*80}")
    
    for part_name, joint_indices in body_parts.items():
        # Mean error across joints in this body part, per frame
        part_errors_1 = np.nanmean(joint_errors_1[:, joint_indices], axis=1)
        part_errors_2 = np.nanmean(joint_errors_2[:, joint_indices], axis=1)
        
        # Overall statistics
        mean_1 = np.nanmean(part_errors_1)
        mean_2 = np.nanmean(part_errors_2)
        delta = mean_2 - mean_1
        rel_delta = (delta / mean_1 * 100) if mean_1 > 0 else 0
        
        body_part_stats[part_name] = {
            'mean_error_1': mean_1,
            'mean_error_2': mean_2,
            'median_error_1': np.nanmedian(part_errors_1),
            'median_error_2': np.nanmedian(part_errors_2),
            'std_error_1': np.nanstd(part_errors_1),
            'std_error_2': np.nanstd(part_errors_2),
            'p90_1': np.nanpercentile(part_errors_1, 90),
            'p90_2': np.nanpercentile(part_errors_2, 90),
            'p95_1': np.nanpercentile(part_errors_1, 95),
            'p95_2': np.nanpercentile(part_errors_2, 95),
            'delta': delta,
            'rel_delta': rel_delta,
            'n_frames': np.sum(np.isfinite(part_errors_1)),
        }
        
        part_label = part_name.replace('_', ' ').title()
        print(f"  {part_label:<25} {mean_1:>10.2f} mm   {mean_2:>10.2f} mm   {delta:>+10.2f} mm   {rel_delta:>+7.1f}%")
    
    # Per-joint analysis
    print(f"\n  Per-Joint Error Statistics:")
    print(f"  {'Joint':<20} {model1.upper():<15} {model2.upper():<15} {'Delta':<15} {'Rel %':<10}")
    print(f"  {'-'*75}")
    
    per_joint_stats = {}
    for joint_idx in range(17):
        joint_name = joint_names[joint_idx]
        mean_1 = np.nanmean(joint_errors_1[:, joint_idx])
        mean_2 = np.nanmean(joint_errors_2[:, joint_idx])
        delta = mean_2 - mean_1
        rel_delta = (delta / mean_1 * 100) if mean_1 > 0 else 0
        
        per_joint_stats[joint_name] = {
            'mean_error_1': mean_1,
            'mean_error_2': mean_2,
            'median_error_1': np.nanmedian(joint_errors_1[:, joint_idx]),
            'median_error_2': np.nanmedian(joint_errors_2[:, joint_idx]),
            'std_error_1': np.nanstd(joint_errors_1[:, joint_idx]),
            'std_error_2': np.nanstd(joint_errors_2[:, joint_idx]),
            'delta': delta,
            'rel_delta': rel_delta,
        }
        
        print(f"  {joint_name:<20} {mean_1:>10.2f} mm   {mean_2:>10.2f} mm   {delta:>+10.2f} mm   {rel_delta:>+7.1f}%")
    
    # Find body parts with largest improvements
    sorted_parts = sorted(body_part_stats.items(), key=lambda x: x[1]['delta'])
    
    print(f"\n  Body Parts Ranked by Improvement (largest improvement first):")
    for i, (part_name, stats) in enumerate(sorted_parts):
        part_label = part_name.replace('_', ' ').title()
        print(f"    {i+1}. {part_label:<25}: {stats['delta']:>+8.2f} mm ({stats['rel_delta']:>+6.1f}%)")
    
    # Save detailed results
    output_path = os.path.join(output_dir, f'body_part_analysis_{dataset}_{model1}_vs_{model2}.txt')
    with open(output_path, 'w') as f:
        f.write(f"Body Part Analysis: {dataset}\n")
        f.write(f"Comparing {model1.upper()} vs {model2.upper()}\n")
        f.write(f"Total frames: {len(pred_mm_1)}\n")
        f.write("="*80 + "\n\n")
        f.write("Body Part Definitions:\n")
        f.write("  - Upper appendages: shoulders, elbows, wrists\n")
        f.write("  - Lower appendages: hips, knees, feet\n")
        f.write("  - Torso: spine, thorax\n")
        f.write("  - Head: neck_base, head\n\n")
        
        # ...write detailed stats...
        for part_name in ['upper_appendages', 'lower_appendages', 'torso', 'head', 'all_appendages']:
            if part_name not in body_part_stats:
                continue
            stats = body_part_stats[part_name]
            part_label = part_name.replace('_', ' ').title()
            f.write(f"{part_label}:\n")
            f.write(f"  {model1.upper()}: {stats['mean_error_1']:.2f} mm\n")
            f.write(f"  {model2.upper()}: {stats['mean_error_2']:.2f} mm\n")
            f.write(f"  Delta: {stats['delta']:+.2f} mm ({stats['rel_delta']:+.1f}%)\n\n")
    
    print(f"\n  Saved body part analysis: {output_path}")


if __name__ == "__main__":
    main()

#!/bin/bash
export NUM_GPUS=1        # single-GPU runs

LAUNCH=/srv/essa-lab/flash3/nwarner30/pose_estimation/mmpose/tools/launch_train_cross_datasets_may_25_pf.sh
DATASET=h36m_det

# ────────────────  SEQLEN = 1  ────────────────────────────────────────────────
# sbatch --gpus=a40:1 $LAUNCH ''    34 $DATASET pf_h36m_pnone_sl1   1
# # sbatch --gpus=a40:1 $LAUNCH xyc   51 $DATASET pf_h36m_xyc_sl1     1
# sbatch --gpus=a40:1 $LAUNCH xycd  68 $DATASET pf_h36m_xycd_sl1    1

# # ────────────────  SEQLEN = 9 ────────────────────────────────────────────────
# sbatch --gpus=a40:1 $LAUNCH ''    34 $DATASET pf_h36m_pnone_sl9  9
# # sbatch --gpus=a40:1 $LAUNCH xyc   51 $DATASET pf_h36m_xyc_sl9    9
# sbatch --gpus=a40:1 $LAUNCH xycd  68 $DATASET pf_h36m_xycd_sl9   9

# # # ────────────────  SEQLEN = 27 ────────────────────────────────────────────────
# sbatch --gpus=a40:1 $LAUNCH ''    34 $DATASET pf_h36m_pnone_sl27  27
# # sbatch --gpus=a40:1 $LAUNCH xyc   51 $DATASET pf_h36m_xyc_sl27    27
# sbatch --gpus=a40:1 $LAUNCH xycd  68 $DATASET pf_h36m_xycd_sl27   27

# ────────────────  SEQLEN = 81 ────────────────────────────────────────────────
# sbatch --gpus=a40:1 $LAUNCH ''    34 $DATASET pf_h36m_pnone_sl81  81
# sbatch --gpus=a40:1 $LAUNCH xyc   51 $DATASET pf_h36m_xyc_sl81    81
# sbatch --gpus=a40:1 $LAUNCH xycd  68 $DATASET pf_h36m_xycd_sl81   81

# ────────────────  SEQLEN = 243 ───────────────────────────────────────────────
# sbatch --gpus=a40:1 $LAUNCH ''    34 $DATASET pf_h36m_pnone_sl243 243
# sbatch --gpus=a40:1 $LAUNCH xyc   51 $DATASET pf_h36m_xyc_sl243   243
# sbatch --gpus=a40:1 $LAUNCH xycd  68 $DATASET pf_h36m_xycd_sl243  243

# bash /srv/essa-lab/flash3/nwarner30/pose_estimation/mmpose/tools/launch_train_cross_datasets_may_25_pf.sh \
#   xycd 68 h36m_det pf_h36m_jan_25_nwarner_all_cross_evals_train_h36m_det 243

# ────────────────  SEQLEN = 1, 9, 27 CASP ────────────────────────────────────────────────
# CASP v0
# sbatch --gpus=a40:1 --export=CASP_MODE=v0 $LAUNCH casp10d 68 $DATASET pf_h36m_casp10d_v0_sl1   1
# sbatch --gpus=a40:1 --export=CASP_MODE=v0 $LAUNCH casp10d 68 $DATASET pf_h36m_casp10d_v0_sl9   9
# sbatch --gpus=a40:1 --export=CASP_MODE=v0 $LAUNCH casp10d 68 $DATASET pf_h36m_casp10d_v0_sl27 27
# sbatch --gpus=a40:1 --export=CASP_MODE=v0 $LAUNCH casp10d 68 $DATASET pf_h36m_casp10d_v0_sl81 81
# # CASP spatial
# sbatch --gpus=a40:1 --export=CASP_MODE=spatial $LAUNCH casp10d 68 $DATASET pf_h36m_casp10d_spatial_sl1   1
# sbatch --gpus=a40:1 --export=CASP_MODE=spatial $LAUNCH casp10d 68 $DATASET pf_h36m_casp10d_spatial_sl9   9
# sbatch --gpus=a40:1 --export=CASP_MODE=spatial $LAUNCH casp10d 68 $DATASET pf_h36m_casp10d_spatial_sl27 27

#debug
# bash /srv/essa-lab/flash3/nwarner30/pose_estimation/mmpose/tools/launch_train_cross_datasets_may_25_pf.sh '' 34 h36m_det pf_h36m_pnone_sl9 9
# bash /srv/essa-lab/flash3/nwarner30/pose_estimation/mmpose/tools/launch_train_cross_datasets_may_25_pf.sh xycd 68 h36m_det pf_h36m_xycd_sl9 9
# CASP_MODE=v0 bash /srv/essa-lab/flash3/nwarner30/pose_estimation/mmpose/tools/launch_train_cross_datasets_may_25_pf.sh casp10d 68 h36m_det pf_h36m_casp10d_v0_sl9 9
# CASP_MODE=spatial bash /srv/essa-lab/flash3/nwarner30/pose_estimation/mmpose/tools/launch_train_cross_datasets_may_25_pf.sh casp10d 68 h36m_det pf_h36m_casp10d_spatial_sl9 9


# Example: Run slen9 poseformer with invconf loss

# sbatch --gpus=a40:1 --export=USE_INVCONF_LOSS=true /srv/essa-lab/flash3/nwarner30/pose_estimation/mmpose/tools/launch_train_cross_datasets_may_25_pf.sh xycd 68 h36m_det pf_h36m_xycd_invconf_sl1 1
# sbatch --gpus=a40:1 --export=USE_INVCONF_LOSS=true /srv/essa-lab/flash3/nwarner30/pose_estimation/mmpose/tools/launch_train_cross_datasets_may_25_pf.sh xycd 68 h36m_det pf_h36m_xycd_invconf_sl27 27
# sbatch --gpus=a40:1 --export=USE_INVCONF_LOSS=true /srv/essa-lab/flash3/nwarner30/pose_estimation/mmpose/tools/launch_train_cross_datasets_may_25_pf.sh xycd 68 h36m_det pf_h36m_xycd_invconf_sl81 81
# sbatch --gpus=a40:1 --export=USE_INVCONF_LOSS=true /srv/essa-lab/flash3/nwarner30/pose_estimation/mmpose/tools/launch_train_cross_datasets_may_25_pf.sh xycd 68 h36m_det pf_h36m_xycd_invconf_sl9 9
# USE_INVCONF_LOSS=true bash /srv/essa-lab/flash3/nwarner30/pose_estimation/mmpose/tools/launch_train_cross_datasets_may_25_pf.sh xycd 68 h36m_det pf_h36m_xycd_invconf_sl9 9

# Testing both casp_v0 and invconf
# USE_INVCONF_LOSS=true CASP_MODE=v0 bash /srv/essa-lab/flash3/nwarner30/pose_estimation/mmpose/tools/launch_train_cross_datasets_may_25_pf.sh casp10d 68 h36m_det pf_h36m_casp10d_v0_invconf_sl9 9

# sbatch --gpus=a40:1 --export=USE_INVCONF_LOSS=true,CASP_MODE=v0 $LAUNCH casp10d 68 $DATASET pf_h36m_casp10d_v0_invconf_sl1 1
# sbatch --gpus=a40:1 --export=USE_INVCONF_LOSS=true,CASP_MODE=v0 $LAUNCH casp10d 68 $DATASET pf_h36m_casp10d_v0_invconf_sl9 9
# sbatch --gpus=a40:1 --export=USE_INVCONF_LOSS=true,CASP_MODE=v0 $LAUNCH casp10d 68 $DATASET pf_h36m_casp10d_v0_invconf_sl27 27

### XY, XYC, XYD perspective methods (standard loss, no casp) ###

# XY (base 2D), slen=1
# echo "Launching: pf_h36m_xy_sl1_v6dets"
# sbatch --gpus=a40:1 $LAUNCH "" 34 $DATASET pf_h36m_xy_sl1_v6dets 1

# # XY (base 2D), slen=9
# echo "Launching: pf_h36m_xy_sl9_v6dets"
# sbatch --gpus=a40:1 $LAUNCH "" 34 $DATASET pf_h36m_xy_sl9_v6dets 9

# # XY (base 2D), slen=27
# echo "Launching: pf_h36m_xy_sl27_v6dets"
# sbatch --gpus=a40:1 $LAUNCH "" 34 $DATASET pf_h36m_xy_sl27_v6dets 27

# # XYC (xy + confidence), slen=1
# echo "Launching: pf_h36m_xyc_sl1_v6dets"
# sbatch --gpus=a40:1 $LAUNCH xyc 51 $DATASET pf_h36m_xyc_sl1_v6dets 1

# # XYC (xy + confidence), slen=9
# echo "Launching: pf_h36m_xyc_sl9_v6dets"
# sbatch --gpus=a40:1 $LAUNCH xyc 51 $DATASET pf_h36m_xyc_sl9_v6dets 9

# # XYC (xy + confidence), slen=27
# echo "Launching: pf_h36m_xyc_sl27_v6dets"
# sbatch --gpus=a40:1 $LAUNCH xyc 51 $DATASET pf_h36m_xyc_sl27_v6dets 27

# # XYD (xy + depth), slen=1
# echo "Launching: pf_h36m_xyd_sl1_v6dets"
# sbatch --gpus=a40:1 $LAUNCH xyd 51 $DATASET pf_h36m_xyd_sl1_v6dets 1

# # XYD (xy + depth), slen=9
# echo "Launching: pf_h36m_xyd_sl9_v6dets"
# sbatch --gpus=a40:1 $LAUNCH xyd 51 $DATASET pf_h36m_xyd_sl9_v6dets 9

# # XYD (xy + depth), slen=27
# echo "Launching: pf_h36m_xyd_sl27_v6dets"
# sbatch --gpus=a40:1 $LAUNCH xyd 51 $DATASET pf_h36m_xyd_sl27_v6dets 27


#DEBUGGING
# XY/XYC/XYD debug commands
# bash /srv/essa-lab/flash3/nwarner30/pose_estimation/mmpose/tools/launch_train_cross_datasets_may_25_pf.sh "" 34 h36m_det pf_h36m_xy_sl9_v6dets 9
# bash /srv/essa-lab/flash3/nwarner30/pose_estimation/mmpose/tools/launch_train_cross_datasets_may_25_pf.sh xyc 51 h36m_det pf_h36m_xyc_sl9_v6dets 9
# bash /srv/essa-lab/flash3/nwarner30/pose_estimation/mmpose/tools/launch_train_cross_datasets_may_25_pf.sh xyd 51 h36m_det pf_h36m_xyd_sl9_v6dets 9

### ORDINAL LOSS EXPERIMENTS (XYCD + pairwise depth regularizer) ###
# # XYCD + ordinal loss (λ=0.3, ε=30mm, τ=25mm), slen=1
# echo "Launching: pf_h36m_xycd_ord_sl1_v6dets"
# sbatch --gpus=a40:1 --export=USE_ORDINAL_LOSS=true,ORDINAL_W=0.3,ORDINAL_EPS_MM=30,ORDINAL_TAU_MM=25 \
#   $LAUNCH xycd 68 $DATASET pf_h36m_xycd_ord_sl1_v6dets 1

# # XYCD + ordinal loss (λ=0.3, ε=30mm, τ=25mm), slen=9
# echo "Launching: pf_h36m_xycd_ord_sl9_v6dets"
# sbatch --gpus=a40:1 --export=USE_ORDINAL_LOSS=true,ORDINAL_W=0.3,ORDINAL_EPS_MM=30,ORDINAL_TAU_MM=25 \
#   $LAUNCH xycd 68 $DATASET pf_h36m_xycd_ord_sl9_v6dets 9

# echo "Launching: pf_h36m_xycd_ord_sl9_v6dets"
# sbatch --gpus=a40:1 --export=USE_ORDINAL_LOSS=true,ORDINAL_W=2,ORDINAL_EPS_MM=25,ORDINAL_TAU_MM=20 \
#   $LAUNCH xycd 68 $DATASET pf_h36m_xycd_ord_sl9_v6dets 9


# # XYCD + ordinal loss (λ=10, ε=25mm, τ=20mm), slen=9
# echo "Launching: pf_h36m_xycd_ord_sl9_v6dets"
# sbatch --gpus=a40:1 --export=USE_ORDINAL_LOSS=true,ORDINAL_W=10,ORDINAL_EPS_MM=25,ORDINAL_TAU_MM=20 \
#   $LAUNCH xycd 68 $DATASET pf_h36m_xycd_ord_sl9_v6dets 9

# XYCD + ordinal loss (λ=200, ε=25mm, τ=20mm), slen=9
# echo "Launching: pf_h36m_xycd_ord_sl9_v6dets"
# sbatch --gpus=a40:1 --export=USE_ORDINAL_LOSS=true,ORDINAL_W=200,ORDINAL_EPS_MM=25,ORDINAL_TAU_MM=20 \
#   $LAUNCH xycd 68 $DATASET pf_h36m_xycd_ord_sl9_v6dets 9



# # XYCD baseline (no ordinal), slen=9 - for comparison
# echo "Launching: pf_h36m_xycd_sl9_v6dets"
# sbatch --gpus=a40:1 $LAUNCH xycd 68 $DATASET pf_h36m_xycd_sl9_v6dets 9
# # Debug: bash /srv/essa-lab/flash3/nwarner30/pose_estimation/mmpose/tools/launch_train_cross_datasets_may_25_pf.sh xycd 68 h36m_det pf_h36m_xycd_sl9_v6dets 9

# # XY baseline (no ordinal, no depth), slen=9 - for comparison
# echo "Launching: pf_h36m_xy_sl9_v6dets"
# sbatch --gpus=a40:1 $LAUNCH "" 34 $DATASET pf_h36m_xy_sl9_v6dets 9
# # Debug: bash /srv/essa-lab/flash3/nwarner30/pose_estimation/mmpose/tools/launch_train_cross_datasets_may_25_pf.sh "" 34 h36m_det pf_h36m_xy_sl9_v6dets 9

# XYCD + ordinal loss (λ=0.3, ε=30mm, τ=25mm), slen=27
# echo "Launching: pf_h36m_xycd_ord_sl27_v6dets"
# sbatch --gpus=a40:1 --export=USE_ORDINAL_LOSS=true,ORDINAL_W=0.3,ORDINAL_EPS_MM=30,ORDINAL_TAU_MM=25 \
#   $LAUNCH xycd 68 $DATASET pf_h36m_xycd_ord_sl27_v6dets 27

# DEBUGGING ordinal loss

# # XYCD + ordinal: 
# DEBUG_MODE=true USE_ORDINAL_LOSS=true ORDINAL_W=0.3 ORDINAL_EPS_MM=30 ORDINAL_TAU_MM=25 bash /srv/essa-lab/flash3/nwarner30/pose_estimation/mmpose/tools/launch_train_cross_datasets_may_25_pf.sh xycd 68 h36m_det pf_h36m_xycd_ord_sl9_v6dets 9
# # XYCD baseline (no ordinal): 
# DEBUG_MODE=true bash /srv/essa-lab/flash3/nwarner30/pose_estimation/mmpose/tools/launch_train_cross_datasets_may_25_pf.sh xycd 68 h36m_det pf_h36m_xycd_sl9_v6dets 9

# # XYCD + ordinal (VERY HEAVY 10): 
# XYCD + ordinal (nearly pure): 
# DEBUG_MODE=true USE_ORDINAL_LOSS=true ORDINAL_W=10.0 ORDINAL_EPS_MM=25 ORDINAL_TAU_MM=20 bash /srv/essa-lab/flash3/nwarner30/pose_estimation/mmpose/tools/launch_train_cross_datasets_may_25_pf.sh xycd 68 h36m_det pf_h36m_xycd_ord_only_sl9 9


### ============================================================================
### FEATURE MAP EXPERIMENTS (v10 cached features) - DYNAMIC SEQUENCE LENGTHS
### Sweeps across: sequence lengths (1, 9, 27), perspectives (XY, XYCD), 
###                features (baseline, RTM, DAV2), sampling modes (nearest_1/2/4)
### ============================================================================

# # Old v8 experiments (commented out - replaced by v10 below)
# SEQ_LENS=(1 9)   # Add more (e.g., 27 81) if needed
# FEAT_PROJ_DIM=16
# DATASET=h36m_det
# 
# for SLEN in "${SEQ_LENS[@]}"; do
#   echo "────────────────────────────────────────────"
#   echo "Launching jobs for sequence length = ${SLEN}"
#   echo "────────────────────────────────────────────"
# 
#   # # # ---- XY BASELINE ----
#   echo "Launching: pf_h36m_xy_sl${SLEN}_v8_baseline"
#   sbatch --gpus=a40:1 --job-name=pf_h36m_xy_sl${SLEN}_v8_baseline \
#     --export=USE_IMG_FEATS=false,USE_DEPTH_FEATS=false \
#     $LAUNCH '' 68 $DATASET pf_h36m_xy_sl${SLEN}_v8_baseline ${SLEN}
# 
#   # # ---- XY + RTM IMAGE FEATURES ----
#   # echo "Launching: pf_h36m_xy_sl${SLEN}_v8_rtm"
#   # sbatch --gpus=a40:1 --job-name=pf_h36m_xy_sl${SLEN}_v8_rtm \
#   #   --export=USE_IMG_FEATS=true,USE_DEPTH_FEATS=false,FEAT_PROJ_DIM=${FEAT_PROJ_DIM} \
#   #   $LAUNCH '' 68 $DATASET pf_h36m_xy_sl${SLEN}_v8_rtm ${SLEN}
# 
#   # # # ---- XY + DAV2 DEPTH FEATURES ----
#   echo "Launching: pf_h36m_xy_sl${SLEN}_v8_dav2"
#   sbatch --gpus=a40:1 --job-name=pf_h36m_xy_sl${SLEN}_v8_dav2 \
#     --export=USE_IMG_FEATS=false,USE_DEPTH_FEATS=true,FEAT_PROJ_DIM=${FEAT_PROJ_DIM} \
#     $LAUNCH '' 68 $DATASET pf_h36m_xy_sl${SLEN}_v8_dav2 ${SLEN}
# 
#   # # # ---- XYCD BASELINE ----
#   echo "Launching: pf_h36m_xycd_sl${SLEN}_v8_baseline"
#   sbatch --gpus=a40:1 --job-name=pf_h36m_xycd_sl${SLEN}_v8_baseline \
#     --export=USE_IMG_FEATS=false,USE_DEPTH_FEATS=false \
#     $LAUNCH xycd 68 $DATASET pf_h36m_xycd_sl${SLEN}_v8_baseline ${SLEN}
# 
#   # # ---- XYCD + RTM IMAGE FEATURES ----
#   # echo "Launching: pf_h36m_xycd_sl${SLEN}_v8_rtm"
#   # sbatch --gpus=a40:1 --job-name=pf_h36m_xycd_sl${SLEN}_v8_rtm \
#   #   --export=USE_IMG_FEATS=true,USE_DEPTH_FEATS=false,FEAT_PROJ_DIM=${FEAT_PROJ_DIM} \
#   #   $LAUNCH xycd 68 $DATASET pf_h36m_xycd_sl${SLEN}_v8_rtm ${SLEN}
# 
#   # # # ---- XYCD + DAV2 DEPTH FEATURES ----
#   echo "Launching: pf_h36m_xycd_sl${SLEN}_v8_dav2"
#   sbatch --gpus=a40:1 --job-name=pf_h36m_xycd_sl${SLEN}_v8_dav2 \
#     --export=USE_IMG_FEATS=false,USE_DEPTH_FEATS=true,FEAT_PROJ_DIM=${FEAT_PROJ_DIM} \
#     $LAUNCH xycd 68 $DATASET pf_h36m_xycd_sl${SLEN}_v8_dav2 ${SLEN}
# 
#   # # ---- XYCD + BOTH FEATURES (OPTIONAL) ----
#   # echo "Launching: pf_h36m_xycd_sl${SLEN}_v8_both"
#   # sbatch --gpus=a40:1 --job-name=pf_h36m_xycd_sl${SLEN}_v8_both \
#   #   --export=USE_IMG_FEATS=true,USE_DEPTH_FEATS=true,FEAT_PROJ_DIM=${FEAT_PROJ_DIM} \
#   #   $LAUNCH xycd 68 $DATASET pf_h36m_xycd_sl${SLEN}_v8_both ${SLEN}
# done

# ============================================================================
# V10 EXPERIMENTS - Sweep across sampling modes (nearest_1, nearest_2, nearest_4)
# Options: '2d_det_kpt_2d_det_sample_nearest4', '2d_det_kpt_2d_det_sample_nearest2', 
#          '2d_det_kpt_2d_det_sample_nearest1', '2d_det_kpt_gt_sample_nearest4'
# ============================================================================

# SEQ_LENS=(1 9 27)
# SEQ_LENS=(1 9)
SEQ_LENS=(1)
FEAT_PROJ_DIM=16
DATASET=h36m_det

# Dataset modes to test
# Format: "dataset_mode:short_name" for cleaner job names
DATASET_MODES=(
  "2d_det_kpt_2d_det_sample_nearest_4:n4_det"
  ## "2d_det_kpt_2d_det_sample_nearest_2:n2_det" just do the 4s
  ## "2d_det_kpt_2d_det_sample_nearest_1:n1_det"
  # "2d_det_kpt_gt_sample_nearest_4:n4_gt"          # Detected kpts + GT sampling
  # "2d_gts_kpt_gt_sample_nearest_4:n4_gt_gt"          # GT kpts + GT sampling (upper bound)
)


for SLEN in "${SEQ_LENS[@]}"; do
  for MODE_PAIR in "${DATASET_MODES[@]}"; do
    # Split into dataset_mode and short_name
    DATASET_MODE="${MODE_PAIR%%:*}"
    SHORT_NAME="${MODE_PAIR##*:}"
    
    echo "════════════════════════════════════════════════════════════════"
    echo "Launching jobs: SeqLen=${SLEN}, Mode=${SHORT_NAME} (${DATASET_MODE})"
    echo "════════════════════════════════════════════════════════════════"
    
    # # # ---- XY BASELINE ----
    # echo "Launching: pf_h36m_xy_sl${SLEN}_v10_${SHORT_NAME}_baseline"
    # sbatch --gpus=a40:1 --job-name=pf_xy_sl${SLEN}_${SHORT_NAME}_base \
    #   --export=USE_IMG_FEATS=false,USE_DEPTH_FEATS=false,DATASET_MODE=${DATASET_MODE} \
    #   $LAUNCH '' 68 $DATASET pf_h36m_xy_sl${SLEN}_v10_${SHORT_NAME}_baseline ${SLEN}
    
    # # # ---- XY + RTM IMAGE FEATURES ----
    # echo "Launching: pf_h36m_xy_sl${SLEN}_v10_${SHORT_NAME}_rtm"
    # sbatch --gpus=a40:1 --job-name=pf_xy_sl${SLEN}_${SHORT_NAME}_rtm \
    #   --export=USE_IMG_FEATS=true,USE_DEPTH_FEATS=false,FEAT_PROJ_DIM=${FEAT_PROJ_DIM},DATASET_MODE=${DATASET_MODE} \
    #   $LAUNCH '' 68 $DATASET pf_h36m_xy_sl${SLEN}_v10_${SHORT_NAME}_rtm ${SLEN}
    
    # # # ---- XY + DAV2 DEPTH FEATURES ----
    # echo "Launching: pf_h36m_xy_sl${SLEN}_v10_${SHORT_NAME}_dav2"
    # sbatch --gpus=a40:1 --job-name=pf_xy_sl${SLEN}_${SHORT_NAME}_dav2 \
    #   --export=USE_IMG_FEATS=false,USE_DEPTH_FEATS=true,FEAT_PROJ_DIM=${FEAT_PROJ_DIM},DATASET_MODE=${DATASET_MODE} \
    #   $LAUNCH '' 68 $DATASET pf_h36m_xy_sl${SLEN}_v10_${SHORT_NAME}_dav2 ${SLEN}
    
    # # ---- XYCD BASELINE ----
    # echo "Launching: pf_h36m_xycd_sl${SLEN}_v10_${SHORT_NAME}_baseline"
    # sbatch --gpus=a40:1 --job-name=pf_xycd_sl${SLEN}_${SHORT_NAME}_base \
    #   --export=USE_IMG_FEATS=false,USE_DEPTH_FEATS=false,DATASET_MODE=${DATASET_MODE} \
    #   $LAUNCH xycd 68 $DATASET pf_h36m_xycd_sl${SLEN}_v10_${SHORT_NAME}_baseline ${SLEN}
    
    # # ---- XYCD + RTM IMAGE FEATURES ----
    # echo "Launching: pf_h36m_xycd_sl${SLEN}_v10_${SHORT_NAME}_rtm"
    # sbatch --gpus=a40:1 --job-name=pf_xycd_sl${SLEN}_${SHORT_NAME}_rtm \
    #   --export=USE_IMG_FEATS=true,USE_DEPTH_FEATS=false,FEAT_PROJ_DIM=${FEAT_PROJ_DIM},DATASET_MODE=${DATASET_MODE} \
    #   $LAUNCH xycd 68 $DATASET pf_h36m_xycd_sl${SLEN}_v10_${SHORT_NAME}_rtm ${SLEN}
    
    # # # ---- XYCD + DAV2 DEPTH FEATURES ----
    # echo "Launching: pf_h36m_xycd_sl${SLEN}_v10_${SHORT_NAME}_dav2"
    # sbatch --gpus=a40:1 --job-name=pf_xycd_sl${SLEN}_${SHORT_NAME}_dav2 \
    #   --export=USE_IMG_FEATS=false,USE_DEPTH_FEATS=true,FEAT_PROJ_DIM=${FEAT_PROJ_DIM},DATASET_MODE=${DATASET_MODE} \
    #   $LAUNCH xycd 68 $DATASET pf_h36m_xycd_sl${SLEN}_v10_${SHORT_NAME}_dav2 ${SLEN}
    
    # ---- XYCD + BOTH FEATURES (RTM + DAV2) ----
    echo "Launching: pf_h36m_xycd_sl${SLEN}_v10_${SHORT_NAME}_both"
    sbatch --gpus=a40:1 --job-name=pf_xycd_sl${SLEN}_${SHORT_NAME}_both \
      --export=USE_IMG_FEATS=true,USE_DEPTH_FEATS=true,FEAT_PROJ_DIM=${FEAT_PROJ_DIM},DATASET_MODE=${DATASET_MODE} \
      $LAUNCH xycd 68 $DATASET pf_h36m_xycd_sl${SLEN}_v10_${SHORT_NAME}_both ${SLEN}
    
    # # ---- XY + BOTH FEATURES (RTM + DAV2) ----
    # echo "Launching: pf_h36m_xy_sl${SLEN}_v10_${SHORT_NAME}_both"
    # sbatch --gpus=a40:1 --job-name=pf_xy_sl${SLEN}_${SHORT_NAME}_both \
    #   --export=USE_IMG_FEATS=true,USE_DEPTH_FEATS=true,FEAT_PROJ_DIM=${FEAT_PROJ_DIM},DATASET_MODE=${DATASET_MODE} \
    #   $LAUNCH '' 68 $DATASET pf_h36m_xy_sl${SLEN}_v10_${SHORT_NAME}_both ${SLEN}
    
    echo ""
  done
done

# launch debug;
# sbatch --gpus=a40:1 --job-name=pf_xycd_sl1_n2_det_rtm \
#   --export=USE_IMG_FEATS=true,USE_DEPTH_FEATS=false,FEAT_PROJ_DIM=16,DATASET_MODE=2d_det_kpt_2d_det_sample_nearest_2 \
#   /srv/essa-lab/flash3/nwarner30/pose_estimation/mmpose/tools/launch_train_cross_datasets_may_25_pf.sh \
#   xycd 68 h36m_det pf_h36m_xycd_sl1_v10_n2_det_rtm 1

# DEBUG_MODE=true \ 
# USE_IMG_FEATS=true \
# USE_DEPTH_FEATS=false \
# FEAT_PROJ_DIM=16 \
# DATASET_MODE=2d_det_kpt_2d_det_sample_nearest_2 \
# bash /srv/essa-lab/flash3/nwarner30/pose_estimation/mmpose/tools/launch_train_cross_datasets_may_25_pf.sh \
#   xycd 68 h36m_det pf_h36m_xycd_sl1_v10_n2_det_rtm 1

### ============================================================================
### DEBUG COMMANDS (local run)
### ============================================================================n

# DEBUG_MODE=true USE_IMG_FEATS=false USE_DEPTH_FEATS=false \
#   bash $LAUNCH '' 68 h36m_det pf_h36m_xy_sl1_v8_baseline 1

# DEBUG_MODE=true USE_IMG_FEATS=true USE_DEPTH_FEATS=false FEAT_PROJ_DIM=16 \
#   bash $LAUNCH '' 68 h36m_det pf_h36m_xy_sl1_v8_rtm 1

# DEBUG_MODE=true USE_IMG_FEATS=false USE_DEPTH_FEATS=true FEAT_PROJ_DIM=16 \
#   bash $LAUNCH '' 68 h36m_det pf_h36m_xy_sl1_v8_dav2 1

# DEBUG_MODE=true USE_IMG_FEATS=false USE_DEPTH_FEATS=false \
#   bash $LAUNCH xycd 68 h36m_det pf_h36m_xycd_sl1_v8_baseline 1

# DEBUG_MODE=true USE_IMG_FEATS=true USE_DEPTH_FEATS=false FEAT_PROJ_DIM=16 \
#   bash $LAUNCH xycd 68 h36m_det pf_h36m_xycd_sl1_v8_rtm 1

# DEBUG_MODE=true USE_IMG_FEATS=false USE_DEPTH_FEATS=true FEAT_PROJ_DIM=16 \
#   bash $LAUNCH xycd 68 h36m_det pf_h36m_xycd_sl1_v8_dav2 1

# DEBUG_MODE=true USE_IMG_FEATS=true USE_DEPTH_FEATS=true FEAT_PROJ_DIM=16 \
#   bash $LAUNCH xycd 68 h36m_det pf_h36m_xycd_sl1_v8_both 1


# DEBUG_MODE=true USE_IMG_FEATS=true USE_DEPTH_FEATS=false FEAT_PROJ_DIM=16 \
#   bash $LAUNCH xycd 68 h36m_det pf_h36m_xycd_sl9_v8_rtm 9

#!/bin/bash
#SBATCH --job-name=poseformer_default
#SBATCH --partition=essa-lab
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=12
#SBATCH --exclude=voltron,xaea-12,ig-88,samantha,qt-1,cyborg
#SBATCH --output=%x_%j.out

PERSPECTIVE_METHOD="$1"
NUM_CHANNELS="$2"
DATASET="$3"
EXPERIMENT_NAME="$4"
SEQ_LEN="${5:-1}"     # Default to 1
SEQ_STEP="${6:-1}"    # Default to 1 if not provided

# Loss configuration
export USE_INVCONF_LOSS=${USE_INVCONF_LOSS:-false}

# Feature map configuration (v8 cached features)
export USE_IMG_FEATS=${USE_IMG_FEATS:-false}       # RTM image features (192D)
export USE_DEPTH_FEATS=${USE_DEPTH_FEATS:-false}   # DAV2 depth features (256D)
export FEAT_PROJ_DIM=${FEAT_PROJ_DIM:-16}          # Projection dimensione_reimpliment_poseformer


# Load Conda environment
source /srv/essa-lab/flash3/nwarner30/miniconda3/etc/profile.d/conda.sh
conda activate openmmlab

# Get SLURM job ID and node name
JOB_ID=${SLURM_JOB_ID}
NODE_NAME=${SLURM_NODELIST}

# Check if CUDA is available
if python -c "import torch; exit(0) if torch.cuda.is_available() else exit(1)"; then
    echo "CUDA is available on node ${NODE_NAME} for SLURM job ${JOB_ID}. Proceeding with training..."
else
    echo "CUDA not available on node ${NODE_NAME} for SLURM job ${JOB_ID}. Exiting..."
    exit 1
fi

# Accept parameters from the outer script
PERSPECTIVE_METHOD="$1"
NUM_CHANNELS="$2"
DATASET="$3"
EXPERIMENT_NAME="$4"
SEQ_LEN="${5:-1}"     # Default to 1
SEQ_STEP="${6:-1}"    # Default to 1 if not provided
# InverseConfMPJPELoss environment variable (default: false, can be overridden by sbatch --export)
export USE_INVCONF_LOSS=${USE_INVCONF_LOSS:-false}



# Debug options
export WANDB_MODE=offline

# Enable debug mode with DEBUG_MODE=true
if [[ "${DEBUG_MODE:-false}" == "true" ]]; then
  export DEBUG_DATASET=True
  export NUM_DEBUG_FRAMES=200000
  echo "DEBUG MODE ENABLED: Using only ${NUM_DEBUG_FRAMES} frames"
fi

export SEQ_LEN
export BATCH_SIZE=512
export WANDB_PROJECT_NAME='train_other_datasets_cross_evals_june25'
export PERSPECTIVE_METHOD
export NUM_CHANNELS
export EXPERIMENT_NAME
export SEQ_STEP

# Set CASP_MODE for CASP runs
if [[ "$PERSPECTIVE_METHOD" == "casp10d" ]]; then
  export CASP_MODE=${CASP_MODE:-v0}
fi

CONFIG="/srv/essa-lab/flash3/nwarner30/pose_estimation/mmpose/configs/body_3d_keypoint/poseformer/h36m/poseformer_h36m_config_img_depth_baselines.py"
WORK_DIR="/srv/essa-lab/flash3/nwarner30/pose_estimation/test_runs/${EXPERIMENT_NAME}"

# Create WORK_DIR if it doesn't exist
mkdir -p ${WORK_DIR}

echo "Running experiment: ${EXPERIMENT_NAME} with ${PERSPECTIVE_METHOD} and dataset ${DATASET}"
echo "Feature map settings: USE_IMG_FEATS=${USE_IMG_FEATS}, USE_DEPTH_FEATS=${USE_DEPTH_FEATS}, FEAT_PROJ_DIM=${FEAT_PROJ_DIM}"



# rendezvous setup (needed only if you use torchrun, harmless otherwise)
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

# Allow the outer script to override NUM_GPUS; default to 1
NUM_GPUS=${NUM_GPUS:-1}

if [ "$NUM_GPUS" -gt 1 ]; then
  # multi-GPU: use DDP
  torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="$NUM_GPUS" \
    /srv/essa-lab/flash3/nwarner30/pose_estimation/mmpose/tools/train.py \
    "${CONFIG}" \
    --work-dir "${WORK_DIR}" \
    --launcher pytorch \
    # --cfg-options data.samples_per_gpu=16
else
  # single-GPU: fall back to non-distributed launch
  python -u /srv/essa-lab/flash3/nwarner30/pose_estimation/mmpose/tools/train.py \
    "${CONFIG}" \
    --work-dir "${WORK_DIR}"
fi


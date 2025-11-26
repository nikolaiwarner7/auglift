#!/bin/bash
#SBATCH --job-name=generate_OD_data
#SBATCH --partition=essa-lab  # Adjust as needed
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --gpus=a40:1 # Adjust GPU as needed
#SBATCH --cpus-per-task=12
#SBATCH --exclude=voltron,xaea-12
#SBATCH --output=%x_%j.out  # Output file


# Load Conda environment
source /srv/essa-lab/flash3/nwarner30/miniconda3/etc/profile.d/conda.sh
conda activate openmmlab

# Get SLURM job ID and node name
JOB_ID=${SLURM_JOB_ID}
NODE_NAME=${SLURM_NODELIST}

JOB_CHUNK_NUMBER=${1:-1}
export JOB_CHUNK_NUMBER

# Check if CUDA is available
if python -c "import torch; exit(0) if torch.cuda.is_available() else exit(1)"; then
    echo "CUDA is available on node ${NODE_NAME} for SLURM job ${JOB_ID}. Proceeding with training..."
    echo "JOB CHUNK set to ${JOB_CHUNK_NUMBER}"
else
    echo "CUDA not available on node ${NODE_NAME} for SLURM job ${JOB_ID}. Re-submitting..."
    NEW_JOB_ID=$(sbatch "$0" "$@" | awk '{print $NF}')
    echo "Resubmitted as SLURM job ${NEW_JOB_ID}"
    exit 1
fi

# python -u /srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-OfficialV2/Depth-Anything-V2/metric_depth/1_30_estimate_depth_proto_loop_metric_h36m.py
#Uppdated high res stuff
# python -u /srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-OfficialV2/Depth-Anything-V2/metric_depth/1_30_estimate_depth_proto_loop_metric_h36m_v2_may25.py
# python -u /srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-OfficialV2/Depth-Anything-V2/metric_depth/1_30_estimate_depth_proto_loop_metric_h36m_v3_sampling_patch_statistics.py
python -u /srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-OfficialV2/Depth-Anything-V2/metric_depth/1_30_estimate_depth_proto_loop_metric_h36m_v4_cache_img_aug_baseline_oct_2025.py
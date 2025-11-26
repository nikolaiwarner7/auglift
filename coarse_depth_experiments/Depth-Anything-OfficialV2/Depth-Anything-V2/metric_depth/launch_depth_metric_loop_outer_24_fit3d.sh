#!/bin/bash

# Number of jobs to launch
NUM_JOBS=16

# Submit each job with a unique JOB_CHUNK_NUMBER
for ((i=0; i<$NUM_JOBS; i++)); do
    sbatch /srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-OfficialV2/Depth-Anything-V2/metric_depth/launch_depth_metric_loop_fit3d.sh $i
    echo "Submitted job with JOB_CHUNK_NUMBER=$i"
done

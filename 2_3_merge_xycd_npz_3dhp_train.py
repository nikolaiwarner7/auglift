import numpy as np
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import ipdb
import os
import pickle
from glob import glob

OD_STYLE = 'buckets' # running or buckets
# Running: 10cm between each pred, not accumulated. Buckets seems more intuitive.
# OD_COARSE_THRESHOLD = 0.01  # 10cm
OD_COARSE_THRESHOLD = 0.25  # 10cm


OUTPUT_NAME = f"{OD_STYLE}_threshold_{OD_COARSE_THRESHOLD}"
# Paths for the old and new .npz files
COMPUTE_GT_OD = True


#h36m train aug
# KEEP:
original_npz_path = '/srv/essa-lab/flash3/nwarner30/pose_estimation/data/annotations/mpi_inf_3dhp_train__all.npz'

#######################
### v3 run again ###### v3 corrected for rgb.
#######################


output_npz_path = '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/test_outputs_metric_3dhp_train_v7_highres.npz'
depth_root = "/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_outputs_metric_3dhp_train_v7_highres/" #fix rgb
# cache_path = "/srv/essa-lab/flash3/nwarner30/pose_estimation/depth_file_map_3dhp_test.pkl"

# Load original and new data
original_data = np.load(original_npz_path, allow_pickle=True)
# ipdb.set_trace()
#first load the new data in 24 npz

import re
from collections import defaultdict

train_frame_nums = {
    (1, 1): 6416, (1, 2): 12430, (2, 1): 6502, (2, 2): 6081, (3, 1): 12488, (3, 2): 12283,
    (4, 1): 6171, (4, 2): 6675, (5, 1): 12820, (5, 2): 12312, (6, 1): 6188, (6, 2): 6145,
    (7, 1): 6239, (7, 2): 6320, (8, 1): 6468, (8, 2): 6054
}



num_samples = len(original_data['S'])



# Check if cached depth file map exists

print("Building depth file map...")
depth_file_map = {}
for job_dir in tqdm(os.listdir(depth_root)):
    job_path = os.path.join(depth_root, job_dir)
    # ipdb.set_trace()
    if os.path.isdir(job_path):
        for depth_file in glob(os.path.join(job_path, "*.npz")):
            base_name = os.path.basename(depth_file).replace("_depth.npz", "")
            # if 'TS1' in base_name:
            #     # ipdb.set_trace()
            #     # print(base_name)
            depth_file_map[base_name] = depth_file


# Initialize storage arrays
predicted_da_depths = []
predicted_keypoints_scores = []
predicted_keypoints_list = []

# Function to process a single sample
def process_sample(args):
    i, individual_imgname = args  # Unpack passed data


    # img_parts = individual_imgname.replace(".jpg", "").split("_")
    # ipdb.set_trace()
    # TS1_img_000003_depth.npz
    # Convert 'TS1_001001.jpg' → 'TS1_img_001001_depth.npz'
    img_base_name = individual_imgname.replace(".jpg", "")
    # img_base_name = f"{img_parts[0]}_img_{int(img_parts[1]):06d}"

    # print(img_base_name)
    # Match image name to depth file
    if img_base_name in depth_file_map:
        depth_data = np.load(depth_file_map[img_base_name], allow_pickle=True)
        predicted_da_depth = depth_data['keypoints_depth']
        predicted_keypoints_score = depth_data['keypoints_score']
        predicted_keypoints = depth_data['keypoints']
    else:
        # if 'TS1' not in img_base_name:
        #     ipdb.set_trace()
        print(img_base_name)
        predicted_da_depth = None
        predicted_keypoints_score = None
        predicted_keypoints = None

    return predicted_da_depth, predicted_keypoints_score, predicted_keypoints



# np.savez(output_npz_path, **original_data_dict)
# print(f"Updated data successfully saved to {output_npz_path}")

# import multiprocessing as mp
from tqdm.contrib.concurrent import process_map  # For better multiprocessing tqdm integration


# Extract image names into a list to avoid repeated disk access
imgnames = original_data['imgname']
num_samples = len(imgnames)
# num_samples = 10 # debug

# Prepare list of (index, imgname) tuples with a progress bar
task_list = [(i, imgnames[i]) for i in tqdm(range(num_samples), desc="Preparing tasks")]

# # Prepare list of (index, imgname) tuples before multiprocessing
# task_list = [(i, original_data['imgname'][i]) for i in tqdm(range(num_samples))]

# Use multiprocessing to parallelize
print("Matching with multiprocessing...")
# Use multiprocessing
# with mp.Pool(processes=4) as pool:
# ipdb.set_trace()
results = process_map(process_sample, task_list, max_workers=8, chunksize=1)

# print("Matching in single process mode for debugging...")
# results = []
# for task in tqdm(task_list, desc="Processing"):
#     results.append(process_sample(task))  # Direct function call without multiprocessing

# Unpack results
predicted_da_depths, predicted_keypoints_scores, predicted_keypoints_list = zip(*results)


# Initialize original_data_dict with all data from original_data except 'imgname' (if you need to copy other data)
original_data_dict = {key: original_data[key] for key in original_data.files}

# Unpack results and convert to arrays
# predicted_da_depths, predicted_keypoints_scores, predicted_keypoints_list = zip(*results)
# ipdb.set_trace()

# ipdb.set_trace()

none_indices = {
    "depth": [i for i, depth in enumerate(predicted_da_depths) if depth is None],
    "keypoints_score": [i for i, score in enumerate(predicted_keypoints_scores) if score is None],
    "keypoints": [i for i, keypoints in enumerate(predicted_keypoints_list) if keypoints is None],
}

print(f"None values found: { {k: len(v) for k, v in none_indices.items()} }")
ipdb.set_trace()
# ipdb.set_trace()
# Define default values based on expected shapes
depth_shape = (17,)  # Replace with the actual expected shape
keypoints_score_shape = (17,)  # Replace with actual expected shape
keypoints_shape = (17, 2)  # Replace with actual expected shape

default_depth = np.zeros(depth_shape)
default_keypoints_score = np.zeros(keypoints_score_shape)
default_keypoints = np.zeros(keypoints_shape)

ipdb.set_trace()
# Replace None values with default values
predicted_da_depths = np.array([depth if depth is not None else default_depth for depth in predicted_da_depths])
# Convert to object array and replace None with default
# Convert to a list and replace None with a default array
# Fix by unwrapping items and replacing None
# Safely handle both array(None) and direct None
# predicted_keypoints_scores = np.array([
#     np.zeros(17) if (x is None or (isinstance(x, np.ndarray) and x.item() is None)) else x.item()
#     for x in predicted_keypoints_scores
# ])
predicted_keypoints_scores = np.array([
    np.zeros(17) if x is None else x
    for x in predicted_keypoints_scores
])


# predicted_keypoints_scores = np.array([score if score is not None else default_keypoints_score for score in predicted_keypoints_scores])

predicted_keypoints_list = np.array([keypoints if keypoints is not None else default_keypoints for keypoints in predicted_keypoints_list])

# ipdb.set_trace()

original_data_dict['predicted_da_depth'] = predicted_da_depths
original_data_dict['predicted_keypoints_score'] = predicted_keypoints_scores
original_data_dict['predicted_keypoints'] = predicted_keypoints_list

np.savez(output_npz_path, **original_data_dict)
print(f"Updated data successfully saved to {output_npz_path}")

import numpy as np
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import ipdb
import os
import pickle
from glob import glob

SAVE_FULL_CASP = True  # Set to True to save full casp_descriptors, False for summary only

# Original base annotations (GT only)
# original_npz_path = '/srv/essa-lab/flash3/nwarner30/pose_estimation/data/annotations/fit3d_all.npz'
original_npz_path = '/srv/essa-lab/flash3/nwarner30/pose_estimation/fit3d_data/processed/fit3d_annotations_all__25hz_combine_train_all.npz'

# Output paths
# output_npz_path = '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_fit3d_all_v3.npz'
# depth_root = '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_outputs_metric_fit3d_v3/'
# cache_path = "/srv/essa-lab/flash3/nwarner30/pose_estimation/depth_file_map_fit3d_v3.pkl"

# Active: v6_casp version
output_npz_path = '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_merge_xycd_data/merged_data_fit3d_all_v6_casp.npz'
# depth_root = '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_outputs_metric_fit3d_v6_casp/'
# depth_root = '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_outputs_metric_fit3d_v6_casp/'
depth_root = '/srv/essa-lab/flash3/nwarner30/pose_estimation/coarse_depth_experiments/Depth-Anything-V2/test_outputs_metric_fit3d_v6_casp/'
cache_path = "/srv/essa-lab/flash3/nwarner30/pose_estimation/depth_file_map_fit3d_v6_casp.pkl"


# Load original and new data
original_data = np.load(original_npz_path, allow_pickle=True)
# ipdb.set_trace()
# ipdb.set_trace()
#first load the new data in 24 npz

# Initialize an empty dictionary for new_data_map
new_data_map = {}

# ipdb> print(list(original_data.keys()))
# ['imgname', 'center', 'scale', 'part', 'S', 'ordinal_depths', 'bone_lengths', 'camera_params_array']
# ipdb> 


num_samples = len(original_data['S'])


# Check if cached depth file map exists
if os.path.exists(cache_path):
    print(f"Loading cached depth file map from {cache_path}...")
    with open(cache_path, 'rb') as f:
        depth_file_map = pickle.load(f)
    print(f"Loaded {len(depth_file_map)} entries from cache")
else:
    print("Building depth file map...")
    depth_file_map = {}
    for job_dir in tqdm(os.listdir(depth_root)):
        job_path = os.path.join(depth_root, job_dir)
        if os.path.isdir(job_path):
            for depth_file in glob(os.path.join(job_path, "*.npz")):
                base_name = os.path.basename(depth_file).replace("_depth.npz", "")
                depth_file_map[base_name] = depth_file
    
    # Save cache for future runs
    print(f"Saving depth file map cache to {cache_path}...")
    with open(cache_path, 'wb') as f:
        pickle.dump(depth_file_map, f)
    print(f"Cached {len(depth_file_map)} entries")
# Initialize storage arrays
predicted_da_depths = []
predicted_keypoints_scores = []
predicted_keypoints_list = []
casp_descriptors_list = []
summary_casp_descriptor_10d_list = []


# ipdb.set_trace()

# Function to process a single sample
_debug_printed = [False]  # Add this BEFORE the function

def process_sample(args):
    i, individual_imgname = args
    
    # Apply SAME transformation as in filtering loop
    img_parts = individual_imgname.replace(".jpg", "").split("_")
    view_num_strip = img_parts[1].split("-")[0]
    suffix = "_".join(img_parts[2:])
    suffix = suffix.replace(".mp4", "")
    suffix_parts = suffix.rsplit("_", 1)
    
    if len(suffix_parts) == 2 and suffix_parts[1].isdigit():
        suffix = f"{suffix_parts[0]}_{int(suffix_parts[1]) + 1:06d}"
    
    img_base_name = f"{img_parts[0]}_{view_num_strip}_{suffix}"
    
    # Now this will match!
    if img_base_name in depth_file_map:
        depth_data = np.load(depth_file_map[img_base_name], allow_pickle=True)
        
        # 🔍 DEBUG: Print once only
        if not _debug_printed[0]:
            _debug_printed[0] = True
            print(f"\n🔍 DEBUG: First Fit3D depth file")
            print(f"   File: {depth_file_map[img_base_name]}")
            print(f"   Keys: {list(depth_data.keys())}")
            print(f"   Has 'casp_descriptors': {'casp_descriptors' in depth_data}")
            print(f"   Has 'summary_casp_descriptor_10d': {'summary_casp_descriptor_10d' in depth_data}")
            if 'summary_casp_descriptor_10d' in depth_data:
                summary_check = depth_data['summary_casp_descriptor_10d']
                print(f"   summary type: {type(summary_check)}")
                if summary_check is not None:
                    print(f"   summary shape: {summary_check.shape if hasattr(summary_check, 'shape') else 'N/A'}")
            print()
            # ipdb.set_trace()
        
        predicted_da_depth = depth_data['keypoints_depth']
        predicted_keypoints_score = depth_data['keypoints_score']
        predicted_keypoints = depth_data['keypoints']
        
        # Check for CASP fields (backwards compatible)
        casp_descriptors = depth_data['casp_descriptors'] if 'casp_descriptors' in depth_data else None
        summary_casp_descriptor_10d = depth_data['summary_casp_descriptor_10d'] if 'summary_casp_descriptor_10d' in depth_data else None
        
    else:
        predicted_da_depth = None
        predicted_keypoints_score = None
        predicted_keypoints = None
        casp_descriptors = None
        summary_casp_descriptor_10d = None

    return predicted_da_depth, predicted_keypoints_score, predicted_keypoints, casp_descriptors, summary_casp_descriptor_10d


# np.savez(output_npz_path, **original_data_dict)
# print(f"Updated data successfully saved to {output_npz_path}")

# import multiprocessing as mp
from tqdm.contrib.concurrent import process_map  # For better multiprocessing tqdm integration


# Extract image names from the NPZ
imgnames = original_data['imgname']

# # **Step 1: Filter imgnames to match 10Hz depth data**
# task_list = []

# for i, individual_imgname in enumerate(tqdm(imgnames, desc="Filtering")):
#     # **Apply transformation to match depth_file_map format**

#     img_parts = individual_imgname.replace(".jpg", "").split("_")
#     view_num_strip = img_parts[1].split("-")[0]  # Strip `-0` if present
#     suffix = "_".join(img_parts[2:])

#     suffix = suffix.replace(".mp4", "")  # for v6 only
#     suffix_parts = suffix.rsplit("_", 1)  # Split at last underscore

#     if len(suffix_parts) == 2 and suffix_parts[1].isdigit():
#         suffix = f"{suffix_parts[0]}_{int(suffix_parts[1]) + 1:06d}"  # Increment and zero-pad


#     img_base_name = f"{img_parts[0]}_{view_num_strip}_{suffix}"

#     # ipdb.set_trace()
#     # **Check if transformed name exists in depth_file_map**
#     if img_base_name in depth_file_map:
#         task_list.append((i, individual_imgname))

# print(f"Filtered task list from {len(imgnames)} → {len(task_list)} (10Hz matched)")

# ...existing code...

# **Step 1: Filter imgnames to match 10Hz depth data**
task_list = []

# 🔍 DEBUG: Track what's happening
sample_imgnames = []
sample_transforms = []
sample_matches = []

for i, individual_imgname in enumerate(tqdm(imgnames, desc="Filtering")):
    # **Apply transformation to match depth_file_map format**
    img_parts = individual_imgname.replace(".jpg", "").split("_")
    view_num_strip = img_parts[1].split("-")[0]  # Strip `-0` if present
    suffix = "_".join(img_parts[2:])
    suffix = suffix.replace(".mp4", "")  # for v6 only
    
    # Try WITHOUT incrementing first (frame numbers might already match)
    suffix_parts = suffix.rsplit("_", 1)  # Split at last underscore
    
    # Option 1: Don't increment (try direct match first)
    img_base_name_direct = f"{img_parts[0]}_{view_num_strip}_{suffix}"
    
    # Option 2: Increment frame number (your current logic)
    if len(suffix_parts) == 2 and suffix_parts[1].isdigit():
        suffix_incremented = f"{suffix_parts[0]}_{int(suffix_parts[1]) + 1:06d}"
        img_base_name_incremented = f"{img_parts[0]}_{view_num_strip}_{suffix_incremented}"
    else:
        img_base_name_incremented = img_base_name_direct
    
    # Try both strategies
    matched = False
    img_base_name = None
    
    if img_base_name_direct in depth_file_map:
        img_base_name = img_base_name_direct
        matched = True
    elif img_base_name_incremented in depth_file_map:
        img_base_name = img_base_name_incremented
        matched = True
    
    # 🔍 Collect debug samples (first 5)
    if i < 5:
        sample_imgnames.append(individual_imgname)
        sample_transforms.append({
            'direct': img_base_name_direct,
            'incremented': img_base_name_incremented,
            'matched': img_base_name if matched else None
        })
        sample_matches.append(matched)
    
    if matched:
        task_list.append((i, individual_imgname))

# 🔍 Print debug info
print(f"\n{'='*80}")
print(f"🔍 FILENAME MATCHING DEBUG:")
print(f"{'='*80}")
for idx, (orig, trans, match) in enumerate(zip(sample_imgnames, sample_transforms, sample_matches)):
    print(f"\nSample {idx + 1}:")
    print(f"  Original:    {orig}")
    print(f"  Direct:      {trans['direct']}")
    print(f"  Incremented: {trans['incremented']}")
    print(f"  Matched:     {trans['matched'] or 'NO MATCH'}")
    print(f"  Status:      {'✅ MATCH' if match else '❌ NO MATCH'}")

print(f"\n{'='*80}")
print(f"First 5 depth file keys (for comparison):")
for i, key in enumerate(list(depth_file_map.keys())[:5]):
    print(f"  {i+1}. {key}")
print(f"{'='*80}\n")

print(f"Filtered task list from {len(imgnames)} → {len(task_list)} (10Hz matched)")

# Add safety check
if len(task_list) == 0:
    print("\n❌ ERROR: No samples matched! Check filename format mismatch.")
    print("   Run the script again to see the debug output above.")
    import sys
    sys.exit(1)

# print(f"Filtered task list from {len(imgnames)} → {len(task_list)} (10Hz matched)")


num_samples = len(imgnames)
# num_samples = 10 # debug

# Prepare list of (index, imgname) tuples with a progress bar
# task_list = [(i, imgnames[i]) for i in tqdm(range(num_samples), desc="Preparing tasks")]

# # Prepare list of (index, imgname) tuples before multiprocessing
# task_list = [(i, original_data['imgname'][i]) for i in tqdm(range(num_samples))]

# Use multiprocessing to parallelize
print("Matching with multiprocessing...")
# Use multiprocessing
# with mp.Pool(processes=4) as pool:
# ipdb.set_trace()
DEBUG = False
if not DEBUG:
    results = process_map(process_sample, task_list, max_workers=8, chunksize=1)

    # print("Matching in single process mode for debugging...")
    # results = []
    # # ipdb.set_trace()
    # for task in tqdm(task_list, desc="Processing"):
    #     results.append(process_sample(task))  # Direct function call without multiprocessing

    # Unpack results
    predicted_da_depths, predicted_keypoints_scores, predicted_keypoints_list, casp_descriptors_list, summary_casp_descriptor_10d_list = zip(*results)

    # Initialize original_data_dict with all data from original_data
    original_data_dict = {key: original_data[key] for key in original_data.files}

    # ============================================================================
    # NEW: Add CASP defaults (same as 3DPW script)
    # ============================================================================
    none_indices = {
        "depth": [i for i, depth in enumerate(predicted_da_depths) if depth is None],
        "keypoints_score": [i for i, score in enumerate(predicted_keypoints_scores) if score is None],
        "keypoints": [i for i, keypoints in enumerate(predicted_keypoints_list) if keypoints is None],
    }

    # print(f"None values found: {{{k: len(v) for k, v in none_indices.items()}}}")
    print(f"None values found: { {k: len(v) for k, v in none_indices.items()} }")

    # Define defaults
    default_depth = np.zeros((17,))
    default_keypoints_score = np.zeros((17,))
    default_keypoints = np.zeros((17, 2))

    # NEW: Add default for casp_descriptors (list of 17 dicts)
    default_descriptor_dict = {
        'central': 0.0, 'central_xy': (0, 0),
        'Q10': 0.0, 'Q10_xy': (0, 0),
        'Q25': 0.0, 'Q25_xy': (0, 0),
        'Q50': 0.0, 'Q50_xy': (0, 0),
        'Q75': 0.0, 'Q75_xy': (0, 0),
        'Q90': 0.0, 'Q90_xy': (0, 0),
        'mad': 0.0,
        'occlusion_asymmetry': 0.0,
        'radius_used': 0.0,
        'person_scale': 0.0
    }
    default_casp_descriptor = [default_descriptor_dict.copy() for _ in range(17)]

    predicted_da_depths = np.array([d if d is not None else default_depth for d in predicted_da_depths])
    predicted_keypoints_scores = np.array([s if s is not None else default_keypoints_score for s in predicted_keypoints_scores])
    predicted_keypoints_list = np.array([k if k is not None else default_keypoints for k in predicted_keypoints_list])

    # NEW: Handle None values in casp_descriptors_list
    casp_descriptors_list = [c if c is not None else default_casp_descriptor for c in casp_descriptors_list]

    original_data_dict['predicted_da_depth'] = predicted_da_depths
    original_data_dict['predicted_keypoints_score'] = predicted_keypoints_scores
    original_data_dict['predicted_keypoints'] = predicted_keypoints_list

    # ============================================================================
    # Check if CASP data exists and add to output if present
    # ============================================================================
    has_casp_descriptors = any(x is not None for x in casp_descriptors_list)
    has_casp_summary = any(x is not None for x in summary_casp_descriptor_10d_list)

    print(f"🔍 CASP data check:")
    print(f"  - casp_descriptors: {sum(1 for x in casp_descriptors_list if x is not None)}/{len(casp_descriptors_list)} non-None")
    print(f"  - summary_casp_descriptor_10d: {sum(1 for x in summary_casp_descriptor_10d_list if x is not None)}/{len(summary_casp_descriptor_10d_list)} non-None")

    if has_casp_summary:
        print("✅ CASP summary data detected - adding summary_casp_descriptor_10d")
        
        default_summary = np.zeros((17, 10), dtype=np.float16)
        summary_casp_descriptor_10d_array = np.array([
            s if s is not None else default_summary
            for s in summary_casp_descriptor_10d_list
        ], dtype=np.float16)

        non_zero_count = np.count_nonzero(summary_casp_descriptor_10d_array.sum(axis=(1,2)))

        # original_data_dict['summary_casp_descriptor_10d'] = summary_casp_descriptor_10d_array
        # print(f"  - summary_casp_descriptor_10d shape: {summary_casp_descriptor_10d_array.shape} (float16)")
        # print(f"  - Non-zero summaries: {non_zero_count}/{len(summary_casp_descriptor_10d_array)}")
        # print(f"  ⚠️  Skipping 'casp_descriptors' (full dicts) to save space—compact only")

        if has_casp_summary or has_casp_descriptors:
            if SAVE_FULL_CASP and has_casp_descriptors:
                # Save [depth, x, y] for each percentile, plus person_scale as flat float16 array
                casp_percentiles = ['Q10', 'Q25', 'Q50', 'Q75', 'Q90', 'central']
                def extract_casp_features(joint_dict):
                    features = []
                    for k in casp_percentiles:
                        depth = joint_dict[k] if joint_dict and k in joint_dict else 0.0
                        xy = joint_dict[k + '_xy'] if joint_dict and (k + '_xy') in joint_dict else [0.0, 0.0]
                        features.extend([depth, xy[0], xy[1]])
                    person_scale = joint_dict['person_scale'] if joint_dict and 'person_scale' in joint_dict else 0.0
                    features.append(person_scale)
                    return np.array(features, dtype=np.float16)

                num_features = len(casp_percentiles) * 3 + 1  # 6*3 + 1 = 19

                casp_descriptors_array = np.array([
                    np.array([
                        extract_casp_features(d)
                        for d in (c if c is not None else [None]*17)
                    ], dtype=np.float16)
                    for c in casp_descriptors_list
                ], dtype=np.float16)  # shape: (N, 17, 19)

                original_data_dict['casp_descriptors'] = casp_descriptors_array
                print(f"  - casp_descriptors shape: {casp_descriptors_array.shape} (float16)")
                print(f"  ✅ Saving full casp_descriptors as float16 array")
                output_npz_path = output_npz_path.replace('.npz', '_fullcasp.npz')
            else:
                original_data_dict['summary_casp_descriptor_10d'] = summary_casp_descriptor_10d_array
                print(f"  - summary_casp_descriptor_10d shape: {summary_casp_descriptor_10d_array.shape} (float16)")
                print(f"  ⚠️  Skipping 'casp_descriptors' (full dicts) to save space—compact only")
                    
        if non_zero_count == 0:
            print("❌ WARNING: All CASP summaries are zero! Depth processing may have failed.")
    else:
        print("❌ No CASP summary data found - CASP field will be missing from output")
        print("   This will cause the '⚠️ CASP summary not found' warning during training")

    # 7) Save out the merged NPZ (with updated ‘part’):
    np.savez(output_npz_path, **original_data_dict)
    print(f"Updated data successfully saved to {output_npz_path}")


    # ============================================================================
    # STEP 2: Create detection-based version (replacing GT 2D with detected 2D)
    # ============================================================================
    print("\n" + "="*80)
    print("Creating detection-based version (replacing GT 2D with detected 2D)...")
    print("="*80)

    # Make a copy
    det_data_dict = original_data_dict.copy()

    # Get GT keypoints from FILTERED data (not original unfiltered data)
    gt_keypoints = original_data_dict['part']  # (M, 17, 3) - already filtered
    visibility = gt_keypoints[:, :, 2:]        # (M, 17, 1)

    # Concatenate predicted keypoints with GT visibility
    # pred_keypoints_with_visibility = np.concatenate([predicted_keypoints_arr, visibility], axis=-1)
    # CORRECT:
    pred_keypoints_with_visibility = np.concatenate([predicted_keypoints_list, visibility], axis=-1)

    # Replace 'part' with detected keypoints
    det_data_dict['part'] = pred_keypoints_with_visibility.copy()

    # Save detection-based version
    det_output_path = output_npz_path.replace('.npz', '_dets.npz')
    np.savez(det_output_path, **det_data_dict)
    print(f"✅ Detection-based annotations saved to: {det_output_path}")
    print(f"   'part' now contains: detected (x,y) + GT visibility")
    print("="*80 + "\n")
elif DEBUG:
    # REPLACE lines 263-275 with:

    # Use multiprocessing to parallelize
    print("Matching with multiprocessing...")

    DEBUG = True
    if DEBUG:
        print("\n🔍 DEBUG MODE: Processing only first 100 samples for testing...")
        debug_task_list = task_list[:100]
        results = process_map(process_sample, debug_task_list, max_workers=8, chunksize=1)
        
        # ✅ Test save immediately
        print("\n" + "="*80)
        print("🔍 DEBUG: Testing NPZ save with 100 samples...")
        print("="*80)
        
        try:
            # Unpack debug results
            debug_depths, debug_scores, debug_keypoints, debug_casp, debug_summary = zip(*results)
            
            # Convert to arrays (with None handling)
            debug_depths_arr = np.array([d if d is not None else np.zeros((17,)) for d in debug_depths])
            debug_scores_arr = np.array([s if s is not None else np.zeros((17,)) for s in debug_scores])
            debug_keypoints_arr = np.array([k if k is not None else np.zeros((17, 2)) for k in debug_keypoints])
            debug_summary_arr = np.array([s if s is not None else np.zeros((17, 10)) for s in debug_summary], dtype=np.float16)
            
            # Print shapes and types
            print(f"✅ Unpacked {len(results)} samples:")
            print(f"   depths shape: {debug_depths_arr.shape}, dtype: {debug_depths_arr.dtype}")
            print(f"   scores shape: {debug_scores_arr.shape}, dtype: {debug_scores_arr.dtype}")
            print(f"   keypoints shape: {debug_keypoints_arr.shape}, dtype: {debug_keypoints_arr.dtype}")
            print(f"   summary shape: {debug_summary_arr.shape}, dtype: {debug_summary_arr.dtype}")
            
            # Test save
            debug_output = output_npz_path.replace('.npz', '_debug_test.npz')
            test_dict = {
                'depths': debug_depths_arr,
                'scores': debug_scores_arr,
                'keypoints': debug_keypoints_arr,
                'summary_casp_descriptor_10d': debug_summary_arr
            }
            
            print(f"\n🔍 Attempting test save to: {debug_output}")
            np.savez(debug_output, **test_dict)
            print(f"✅ Test save SUCCESSFUL!")
            
            # Verify reload
            reloaded = np.load(debug_output, allow_pickle=True)
            print(f"✅ Test reload successful, keys: {list(reloaded.keys())}")
            print(f"   Reloaded summary shape: {reloaded['summary_casp_descriptor_10d'].shape}")
            print(f"   Reloaded summary dtype: {reloaded['summary_casp_descriptor_10d'].dtype}")
            
            # Clean up test file
            # os.remove(debug_output)
            print(f"✅ Cleaned up test file")
            
            print("\n🎉 DEBUG: Save test PASSED!")
            print("   Set DEBUG=False to process all samples.")
            print("="*80 + "\n")
            
            import sys
            sys.exit(0)  # Stop here in debug mode
            
        except Exception as e:
            print(f"\n❌ DEBUG: Save test FAILED!")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            print("\n   Checking first 5 results for type mismatches:")
            
            for i, result in enumerate(results[:5]):
                print(f"\n   Sample {i}:")
                for j, item in enumerate(result):
                    item_type = type(item)
                    item_shape = item.shape if hasattr(item, 'shape') else 'N/A'
                    item_dtype = item.dtype if hasattr(item, 'dtype') else 'N/A'
                    print(f"     Field {j}: type={item_type}, shape={item_shape}, dtype={item_dtype}")
            
            import sys
            sys.exit(1)

import pickle
import numpy as np
import ipdb
import torch

# Paths for the original .pkl and new .npz files
# pkl_path = '/srv/essa-lab/flash3/nwarner30/pose_estimation/test_runs/part3_11_1/best_pose_aug_h36m_on_3dhp/test_11_1.pkl'
# pkl_path = '/srv/essa-lab/flash3/nwarner30/pose_estimation/test_runs/part3_11_1/5e_test/train_11_1.pkl'
# pkl_path = '/srv/essa-lab/flash3/nwarner30/pose_estimation/test_runs/11_2_correct_OD_test/5e_test/11_2_correct_OD_test_split.pkl'
# pkl_path= '/srv/essa-lab/flash3/nwarner30/pose_estimation/test_runs/11_2_correct_OD_test/5e_train_split/11_2_correct_OD_train_split.pkl'
# pkl_path='/srv/essa-lab/flash3/nwarner30/pose_estimation/test_runs/11_2_correct_OD_test/3dhp_test_split/11_2_correct_OD_test_split.pkl'
# pkl_path = '/srv/essa-lab/flash3/nwarner30/pose_estimation/test_11_5_53_mpjpe_OD.pkl'
# pkl_path = '/srv/essa-lab/flash3/nwarner30/pose_estimation/test_11_5_53_mpjpe_pose_aug_no_OD.pkl'
# pkl_path = '/srv/essa-lab/flash3/nwarner30/pose_estimation/test_11_9_coarse25_OD.pkl'
# pkl_path = '/srv/essa-lab/flash3/nwarner30/pose_estimation/test_11_9_coarse25_pose_aug.pkl'
# pkl_path='/srv/essa-lab/flash3/nwarner30/pose_estimation/test_results/dump_3dhp_test_preds__results.pkl'
# pkl_path='/srv/essa-lab/flash3/nwarner30/pose_estimation/test_results/dump_3dhp_test_preds_xycd_results.pkl'
# pkl_path='/srv/essa-lab/flash3/nwarner30/pose_estimation/june_9_sb_3dpw_xy.pkl'
# pkl_path='/srv/essa-lab/flash3/nwarner30/pose_estimation/june_9_sb_3dpw_xycd.pkl'

# Define list of (pkl_path, npz_path) pairs to convert
conversion_pairs = [
    # New sl27 files (default - uncomment others if needed)
    # ('/srv/essa-lab/flash3/nwarner30/pose_estimation/test_results/july_25_videopose_lift_h36m_jan_25_nwarner_all_cross_evals_train_h36m_det_xycd_sl27_3dhpresults.pkl',
    #  '/srv/essa-lab/flash3/nwarner30/pose_estimation/test_results/july_25_videopose_lift_h36m_jan_25_nwarner_all_cross_evals_train_h36m_det_xycd_sl27_3dhpresults.npz'),
    # ('/srv/essa-lab/flash3/nwarner30/pose_estimation/test_results/july_25_videopose_lift_h36m_jan_25_nwarner_all_cross_evals_train_h36m_det__sl27_3dhpresults.pkl',
    #  '/srv/essa-lab/flash3/nwarner30/pose_estimation/test_results/july_25_videopose_lift_h36m_jan_25_nwarner_all_cross_evals_train_h36m_det__sl27_3dhpresults.npz'),
    # ('/srv/essa-lab/flash3/nwarner30/pose_estimation/test_results/july_25_videopose_lift_h36m_jan_25_nwarner_all_cross_evals_train_h36m_det__sl27_h36m_indistrresults.pkl',
    #  '/srv/essa-lab/flash3/nwarner30/pose_estimation/test_results/july_25_videopose_lift_h36m_jan_25_nwarner_all_cross_evals_train_h36m_det__sl27_h36m_indistrresults.npz'),
    # ('/srv/essa-lab/flash3/nwarner30/pose_estimation/test_results/july_25_videopose_lift_h36m_jan_25_nwarner_all_cross_evals_train_h36m_det_xycd_sl27_h36m_indistrresults.pkl',
    #  '/srv/essa-lab/flash3/nwarner30/pose_estimation/test_results/july_25_videopose_lift_h36m_jan_25_nwarner_all_cross_evals_train_h36m_det_xycd_sl27_h36m_indistrresults.npz'),
    # ('/srv/essa-lab/flash3/nwarner30/pose_estimation/test_results/june_25_poseformer_poseformer_testds__sl27_3dpw_results.pkl',
    # '/srv/essa-lab/flash3/nwarner30/pose_estimation/test_results/june_25_poseformer_poseformer_testds__sl27_3dpw_results.npz'),
    # ('/srv/essa-lab/flash3/nwarner30/pose_estimation/test_results/june_25_poseformer_poseformer_testds_xycd_sl27_3dpw_results.pkl',
    # '/srv/essa-lab/flash3/nwarner30/pose_estimation/test_results/june_25_poseformer_poseformer_testds_xycd_sl27_3dpw_results.npz'),
]

# All 12 cross-dataset test results (4 models × 3 datasets)
conversion_pairs_all_cross_dataset = [
    # # XY baseline model on all datasets
    # ('/srv/essa-lab/flash3/nwarner30/pose_estimation/test_results/june_25_poseformer_poseformer_testds__sl27_3dpw_results.pkl',
    #  '/srv/essa-lab/flash3/nwarner30/pose_estimation/test_results/june_25_poseformer_poseformer_testds__sl27_3dpw_results.npz'),
    # ('/srv/essa-lab/flash3/nwarner30/pose_estimation/test_results/june_25_poseformer_poseformer_testds__sl27_3dhp_results.pkl',
    #  '/srv/essa-lab/flash3/nwarner30/pose_estimation/test_results/june_25_poseformer_poseformer_testds__sl27_3dhp_results.npz'),
    # ('/srv/essa-lab/flash3/nwarner30/pose_estimation/test_results/june_25_poseformer_poseformer_testds__sl27_h36m_results.pkl',
    #  '/srv/essa-lab/flash3/nwarner30/pose_estimation/test_results/june_25_poseformer_poseformer_testds__sl27_h36m_results.npz'),
    
    # # XYC model on all datasets
    # ('/srv/essa-lab/flash3/nwarner30/pose_estimation/test_results/june_25_poseformer_poseformer_testds_xyc_sl27_3dpw_results.pkl',
    #  '/srv/essa-lab/flash3/nwarner30/pose_estimation/test_results/june_25_poseformer_poseformer_testds_xyc_sl27_3dpw_results.npz'),
    # ('/srv/essa-lab/flash3/nwarner30/pose_estimation/test_results/june_25_poseformer_poseformer_testds_xyc_sl27_3dhp_results.pkl',
    #  '/srv/essa-lab/flash3/nwarner30/pose_estimation/test_results/june_25_poseformer_poseformer_testds_xyc_sl27_3dhp_results.npz'),
    # ('/srv/essa-lab/flash3/nwarner30/pose_estimation/test_results/june_25_poseformer_poseformer_testds_xyc_sl27_h36m_results.pkl',
    #  '/srv/essa-lab/flash3/nwarner30/pose_estimation/test_results/june_25_poseformer_poseformer_testds_xyc_sl27_h36m_results.npz'),
    
    # # XYD model on all datasets
    # ('/srv/essa-lab/flash3/nwarner30/pose_estimation/test_results/june_25_poseformer_poseformer_testds_xyd_sl27_3dpw_results.pkl',
    #  '/srv/essa-lab/flash3/nwarner30/pose_estimation/test_results/june_25_poseformer_poseformer_testds_xyd_sl27_3dpw_results.npz'),
    # ('/srv/essa-lab/flash3/nwarner30/pose_estimation/test_results/june_25_poseformer_poseformer_testds_xyd_sl27_3dhp_results.pkl',
    #  '/srv/essa-lab/flash3/nwarner30/pose_estimation/test_results/june_25_poseformer_poseformer_testds_xyd_sl27_3dhp_results.npz'),
    # ('/srv/essa-lab/flash3/nwarner30/pose_estimation/test_results/june_25_poseformer_poseformer_testds_xyd_sl27_h36m_results.pkl',
    #  '/srv/essa-lab/flash3/nwarner30/pose_estimation/test_results/june_25_poseformer_poseformer_testds_xyd_sl27_h36m_results.npz'),
    
    # # XYCD model on all datasets
    # ('/srv/essa-lab/flash3/nwarner30/pose_estimation/test_results/june_25_poseformer_poseformer_testds_xycd_sl27_3dpw_results.pkl',
    #  '/srv/essa-lab/flash3/nwarner30/pose_estimation/test_results/june_25_poseformer_poseformer_testds_xycd_sl27_3dpw_results.npz'),
    # ('/srv/essa-lab/flash3/nwarner30/pose_estimation/test_results/june_25_poseformer_poseformer_testds_xycd_sl27_3dhp_results.pkl',
    #  '/srv/essa-lab/flash3/nwarner30/pose_estimation/test_results/june_25_poseformer_poseformer_testds_xycd_sl27_3dhp_results.npz'),
    # ('/srv/essa-lab/flash3/nwarner30/pose_estimation/test_results/june_25_poseformer_poseformer_testds_xycd_sl27_h36m_results.pkl',
    #  '/srv/essa-lab/flash3/nwarner30/pose_estimation/test_results/june_25_poseformer_poseformer_testds_xycd_sl27_h36m_results.npz'),
    
    # Fit3D results (4 models)
    # ('/srv/essa-lab/flash3/nwarner30/pose_estimation/test_results/june_25_poseformer_poseformer_testds__sl27_fit3d_results.pkl',
    #  '/srv/essa-lab/flash3/nwarner30/pose_estimation/test_results/june_25_poseformer_poseformer_testds__sl27_fit3d_results.npz'),
    # ('/srv/essa-lab/flash3/nwarner30/pose_estimation/test_results/june_25_poseformer_poseformer_testds_xyc_sl27_fit3d_results.pkl',
    #  '/srv/essa-lab/flash3/nwarner30/pose_estimation/test_results/june_25_poseformer_poseformer_testds_xyc_sl27_fit3d_results.npz'),
    # ('/srv/essa-lab/flash3/nwarner30/pose_estimation/test_results/june_25_poseformer_poseformer_testds_xyd_sl27_fit3d_results.pkl',
    #  '/srv/essa-lab/flash3/nwarner30/pose_estimation/test_results/june_25_poseformer_poseformer_testds_xyd_sl27_fit3d_results.npz'),
    # ('/srv/essa-lab/flash3/nwarner30/pose_estimation/test_results/june_25_poseformer_poseformer_testds_xycd_sl27_fit3d_results.pkl',
    #  '/srv/essa-lab/flash3/nwarner30/pose_estimation/test_results/june_25_poseformer_poseformer_testds_xycd_sl27_fit3d_results.npz'),

    # ('/srv/essa-lab/flash3/nwarner30/pose_estimation/test_results/june_25_poseformer_poseformer_testds_xycd_sl9_3dhp_results.pkl',
    # '/srv/essa-lab/flash3/nwarner30/pose_estimation/test_results/june_25_poseformer_poseformer_testds_xycd_sl9_3dhp_results.npz'),
    ('/srv/essa-lab/flash3/nwarner30/pose_estimation/test_results/june_25_poseformer_poseformer_testds_xycd_sl1_3dhp_results.pkl',
    '/srv/essa-lab/flash3/nwarner30/pose_estimation/test_results/june_25_poseformer_poseformer_testds_xycd_sl1_3dhp_results.npz')
]
# npz_path = '/srv/essa-lab/flash3/nwarner30/pose_estimation/test_runs/part3_11_1/best_pose_aug_h36m_on_3dhp/converted_test_11_1.npz'
# npz_path = '/srv/essa-lab/flash3/nwarner30/pose_estimation/test_runs/part3_11_1/5e_test/converted_train_11_1.npz'
# npz_path = '/srv/essa-lab/flash3/nwarner30/pose_estimation/test_runs/11_2_correct_OD_test/5e_test/converted_test_11_2.npz'
# npz_path= '/srv/essa-lab/flash3/nwarner30/pose_estimation/test_runs/11_2_correct_OD_test/5e_train_split/converted_train_11_2.npz'
# npz_path='/srv/essa-lab/flash3/nwarner30/pose_estimation/test_runs/11_2_correct_OD_test/3dhp_test_split/converted_test_11_2.npz'
# npz_path='/srv/essa-lab/flash3/nwarner30/pose_estimation/test_11_5_53_mpjpe_OD.npz'
# npz_path = '/srv/essa-lab/flash3/nwarner30/pose_estimation/test_11_5_53_mpjpe_pose_aug_no_OD.npz'
# npz_path = '/srv/essa-lab/flash3/nwarner30/pose_estimation/test_11_5_53_mpjpe_vanilla_no_OD.npz'
# npz_path = '/srv/essa-lab/flash3/nwarner30/pose_estimation/test_11_9_coarse25_OD.npz'
# npz_path = '/srv/essa-lab/flash3/nwarner30/pose_estimation/test_11_9_coarse25_pose_aug.npz'

# npz_path = '/srv/essa-lab/flash3/nwarner30/pose_estimation/test_results/dump_3dhp_test_preds__results.npz'
# npz_path = '/srv/essa-lab/flash3/nwarner30/pose_estimation/test_results/dump_3dhp_test_preds_xycd_results.npz'
# npz_path='/srv/essa-lab/flash3/nwarner30/pose_estimation/june_9_sb_3dpw_xy.npz'
# npz_path='/srv/essa-lab/flash3/nwarner30/pose_estimation/june_9_sb_3dpw_xycd.npz'

# Loop through all conversion pairs
for i, (pkl_path, npz_path) in enumerate(conversion_pairs_all_cross_dataset):
    print(f"\n=== Converting {i+1}/{len(conversion_pairs_all_cross_dataset)} ===")
    print(f"From: {pkl_path}")
    print(f"To:   {npz_path}")
    
    try:
        # Step 1: Load the pickle file
        with open(pkl_path, 'rb') as pkl_file:
            data = pickle.load(pkl_file)

        # Step 2: Convert data to a suitable format (e.g., a dictionary of numpy arrays)
        # Assuming `data` is a list of dictionaries, and we want to save key arrays for faster access
        data_dict = {}

        # Extract key arrays and store them in a new dictionary for saving
        # Ensure all elements are numpy-compatible (optional step)
        for entry in data:
            if isinstance(entry['gt_instance_labels']['lifting_target_weight'], torch.Tensor):
                entry['gt_instance_labels']['lifting_target_weight'] = entry['gt_instance_labels']['lifting_target_weight'].numpy()
            if isinstance(entry['gt_instance_labels']['lifting_target_label'], torch.Tensor):
                entry['gt_instance_labels']['lifting_target_label'] = entry['gt_instance_labels']['lifting_target_label'].numpy()

        # Step 3: Save the data as a .npz file
        np.savez(npz_path, data=data)

        print(f"✓ Successfully converted to {npz_path}")
        
    except Exception as e:
        print(f"✗ Error converting {pkl_path}: {str(e)}")
        continue

print(f"\n=== Conversion Complete ===")
print(f"Processed {len(conversion_pairs_all_cross_dataset)} files")

"""
"""
# Check convertedimport numpy as np

# # Path to your .npz file
# npz_path = '/srv/essa-lab/flash3/nwarner30/pose_estimation/converted_10_9_pkl_data.npz'

# # Load the .npz file
# loaded_data = np.load(npz_path, allow_pickle=True)

# # List the keys in the .npz file
# ipdb.set_trace()
"""
Given the 50hz versions
I ran these before remembering to concat vis all 1s
So lets write a minimal SEPARATE script
Loads the 50hz versions (or whichever)
Adds the vis channel to part
Saves.

"""

import numpy as np

# Fixed paths (50hz versions, update if needed)
TRAIN_FILE = "/srv/essa-lab/flash3/nwarner30/pose_estimation/h36m_data/annotations/h36m_train_vanilla_50hz_h36m_50hz_hrnet_preds_xy_with_confs.npz"
TEST_FILE  = "/srv/essa-lab/flash3/nwarner30/pose_estimation/h36m_data/annotations/h36m_test_vanilla_50hz_h36m_50hz_hrnet_preds_xy_with_confs.npz"

def add_vis_channel(path):
    data = np.load(path, allow_pickle=True)
    data_dict = {k: data[k] for k in data.files}

    if data_dict["part"].shape[-1] == 2:
        print(f"[INFO] Adding visibility channel to {path}")
        N = data_dict["part"].shape[0]
        ones = np.ones((N, 17, 1), dtype=data_dict["part"].dtype)
        data_dict["part"] = np.concatenate([data_dict["part"], ones], axis=2)
        np.savez(path, **data_dict)
        print(f"[✓] Updated and saved: {path}")
    else:
        print(f"[✓] Already has visibility: {path}")

add_vis_channel(TRAIN_FILE)
add_vis_channel(TEST_FILE)

# Copyright (c) OpenMMLab. All rights reserved.
import itertools
import logging
import os.path as osp
from copy import deepcopy
from itertools import filterfalse, groupby
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import os
import numpy as np  # ← NEW: needed for concatenate

import cv2
from mmengine.dataset import BaseDataset, force_full_init
from mmengine.fileio import exists, get_local_path, load
from mmengine.logging import print_log
from mmengine.utils import is_abs

from mmpose.registry import DATASETS
from ..utils import parse_pose_metainfo
import ipdb
from contextlib import AbstractContextManager

DEBUG_DATASET = os.getenv('DEBUG_DATASET', 'False').lower() == 'true'
print("DEBUGGING DATASET:", DEBUG_DATASET)
DATASET_TYPE = os.getenv('DATASET_TYPE', '')  # Replace 'default_value' with a fallback if needed
print(DATASET_TYPE)
NUM_DEBUG_FRAMES = int(os.getenv('NUM_DEBUG_FRAMES', 3000))
print("Num debug frames:,", NUM_DEBUG_FRAMES)
LONG_SEQ_SIXTHS = os.getenv('LONG_SEQ_SIXTHS', None)
print("Long sequence sixths:,", LONG_SEQ_SIXTHS)

import matplotlib.pyplot as plt
import cv2
def save_keypoints_on_image(image_path, keypoints, save_path):
    """
    Load the image, overlay keypoints, and save it.
    
    Args:
        image_path (str): Path to the image file.
        keypoints (np.array): Keypoints array (N, 2) where N is number of keypoints.
        save_path (str): Path to save the overlayed image.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Image not found at {image_path}")
        return
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    
    # Plot each keypoint
    for (x, y) in keypoints:
        plt.scatter(x, y, c='red', s=20)  # Ground truth keypoints in red
    
    plt.title(f"Keypoints on {os.path.basename(image_path)}")
    plt.axis('off')

    # Save the image with keypoints
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved: {save_path}")

@DATASETS.register_module()
class BaseMocapDataset(BaseDataset):
    """Base class for 3d body datasets.

    Args:
        ann_file (str): Annotation file path. Default: ''.
        seq_len (int): Number of frames in a sequence. Default: 1.
        multiple_target (int): If larger than 0, merge every
            ``multiple_target`` sequence together. Default: 0.
        causal (bool): If set to ``True``, the rightmost input frame will be
            the target frame. Otherwise, the middle input frame will be the
            target frame. Default: ``True``.
        subset_frac (float): The fraction to reduce dataset size. If set to 1,
            the dataset size is not reduced. Default: 1.
        camera_param_file (str): Cameras' parameters file. Default: ``None``.
        data_mode (str): Specifies the mode of data samples: ``'topdown'`` or
            ``'bottomup'``. In ``'topdown'`` mode, each data sample contains
            one instance; while in ``'bottomup'`` mode, each data sample
            contains all instances in a image. Default: ``'topdown'``
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Default: ``None``.
        data_root (str, optional): The root directory for ``data_prefix`` and
            ``ann_file``. Default: ``None``.
        data_prefix (dict, optional): Prefix for training data.
            Default: ``dict(img='')``.
        filter_cfg (dict, optional): Config for filter data. Default: `None`.
        indices (int or Sequence[int], optional): Support using first few
            data in annotation file to facilitate training/testing on a smaller
            dataset. Default: ``None`` which means using all ``data_infos``.
        serialize_data (bool, optional): Whether to hold memory using
            serialized objects, when enabled, data loader workers can use
            shared RAM from master process instead of making a copy.
            Default: ``True``.
        pipeline (list, optional): Processing pipeline. Default: [].
        test_mode (bool, optional): ``test_mode=True`` means in test phase.
            Default: ``False``.
        lazy_init (bool, optional): Whether to load annotation during
            instantiation. In some cases, such as visualization, only the meta
            information of the dataset is needed, which is not necessary to
            load annotation file. ``Basedataset`` can skip load annotations to
            save time by set ``lazy_init=False``. Default: ``False``.
        max_refetch (int, optional): If ``Basedataset.prepare_data`` get a
            None img. The maximum extra number of cycles to get a valid
            image. Default: 1000.
    """

    METAINFO: dict = dict()

    def __init__(self,
                 ann_file: str = '',
                 seq_len: int = 1,
                 multiple_target: int = 0,
                 causal: bool = True,
                 subset_frac: float = 1.0,
                 camera_param_file: Optional[str] = None,
                 data_mode: str = 'topdown',
                 metainfo: Optional[dict] = None,
                 data_root: Optional[str] = None,
                 data_prefix: dict = dict(img=''),
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = True,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000,
                 perspective_method: str='',
                 dataset_type: str = '',
                 normalize_feats: bool = False):  # NEW: normalization flag
        self.dataset_type = dataset_type
        self.perspective_method = perspective_method
        self.normalize_feats = normalize_feats  # NEW: store flag

        if data_mode not in {'topdown', 'bottomup'}:
            raise ValueError(
                f'{self.__class__.__name__} got invalid data_mode: '
                f'{data_mode}. Should be "topdown" or "bottomup".')
        self.data_mode = data_mode

        _ann_file = ann_file
        if not is_abs(_ann_file):
            _ann_file = osp.join(data_root, _ann_file)
        assert exists(_ann_file), (
            f'Annotation file `{_ann_file}` does not exist.')

        self._load_ann_file(_ann_file)

        self.camera_param_file = camera_param_file
        if self.camera_param_file:
            if not is_abs(self.camera_param_file):
                self.camera_param_file = osp.join(data_root,
                                                  self.camera_param_file)
            assert exists(self.camera_param_file), (
                f'Camera parameters file `{self.camera_param_file}` does not '
                'exist.')
            self.camera_param = load(self.camera_param_file)

        self.seq_len = seq_len
        self.causal = causal

        self.multiple_target = multiple_target
        if self.multiple_target:
            assert (self.seq_len == 1), (
                'Multi-target data sample only supports seq_len=1.')

        assert 0 < subset_frac <= 1, (
            f'Unsupported `subset_frac` {subset_frac}. Supported range '
            'is (0, 1].')
        self.subset_frac = subset_frac

        self.sequence_indices = self.get_sequence_indices()

        super().__init__(
            ann_file=ann_file,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            filter_cfg=filter_cfg,
            indices=indices,
            serialize_data=serialize_data,
            pipeline=pipeline,
            test_mode=test_mode,
            lazy_init=lazy_init,
            max_refetch=max_refetch)

    def _load_ann_file(self, ann_file: str) -> dict:
        """Load annotation file with optional debug subsetting for fast iteration."""
        # Determine subset range BEFORE loading
        subset_range = None
        if DEBUG_DATASET:
            subset_range = NUM_DEBUG_FRAMES
            print(f"DEBUG MODE: Will load only first {subset_range} frames")
        else:
            print("Loading full dataset annotations")
        
        with get_local_path(ann_file) as local_path:
            raw = np.load(local_path, allow_pickle=True)
            
            # If debug mode, slice BEFORE copying to save memory
            if subset_range is not None:
                self.ann_data = {}
                for k in raw.files:
                    arr = raw[k]
                    # Only slice arrays that have frame dimension
                    if arr.ndim > 0 and len(arr) >= subset_range:
                        self.ann_data[k] = arr[:subset_range].copy()
                    else:
                        self.ann_data[k] = arr.copy()
                print(f"Loaded {len(self.ann_data['imgname'])} frames (debug subset)")
            else:
                # Full dataset: copy everything
                self.ann_data = {k: raw[k].copy() for k in raw.files}
                print(f"Loaded {len(self.ann_data['imgname'])} frames (full dataset)")
            
            raw.close()

    # def _load_ann_file(self, ann_file: str):
    #     """Memory-map the .npz once and keep the ctx alive for workers."""
    #     print('Loading annotations fully (mmap_mode="r")')

    #     # 1) Create the context manager
    #     ctx: AbstractContextManager = get_local_path(ann_file)

    #     # 2) Enter it MANUALLY so we get the real file path,
    #     #    but **do not** exit; store ctx on `self` so it
    #     #    lives for the lifetime of the dataset object.
    #     local_path = ctx.__enter__()
    #     self._local_path_ctx = ctx          # prevents garbage-collection

    #     # 3) Open the NPZ with mem-map — now all workers share these pages
    #     self.ann_data = np.load(local_path, allow_pickle=True, mmap_mode='r')


    # def _load_ann_file(self, ann_file: str) -> dict:
    #     """Load annotation file to get image information.

    #     Args:
    #         ann_file (str): Annotation file path.

    #     Returns:
    #         dict: Annotation information.
    #     """

    #     with get_local_path(ann_file) as local_path:
    #         self.ann_data = np.load(local_path, allow_pickle=True)
    #         # self.ann_data = np.load(local_path, allow_pickle=True, mmap_mode='r')
            

    @classmethod
    def _load_metainfo(cls, metainfo: dict = None) -> dict:
        """Collect meta information from the dictionary of meta.

        Args:
            metainfo (dict): Raw data of pose meta information.

        Returns:
            dict: Parsed meta information.
        """

        if metainfo is None:
            metainfo = deepcopy(cls.METAINFO)

        if not isinstance(metainfo, dict):
            raise TypeError(
                f'metainfo should be a dict, but got {type(metainfo)}')

        # parse pose metainfo if it has been assigned
        if metainfo:
            metainfo = parse_pose_metainfo(metainfo)
        return metainfo

    @force_full_init
    def prepare_data(self, idx) -> Any:
        """
        Now that “_load_annotations” has already filled self.precomputed_key2d
        and self.precomputed_key3d for every mode, we simply slice out the
        243×17×C block for this particular sample.

        1. Fetch info = self.data_list[idx], then fids = info['frame_ids'] (length=T).
        2. Look up mode = self.perspective_method or 'none'.  Then do a single
           NumPy fancy‐index into the precomputed 2D/3D arrays:  key2d_block =
           self.precomputed_key2d[mode][ fids ], etc.
        3. Split the final channel off as “visibility,” and package everything
           into data_info exactly as your old pipeline expects.
        """
        # ipdb.set_trace()
        info = self.data_list[idx]
        fids = info['frame_ids']      # e.g. array([100, 101, …, 342]) length=T

        # Select which dictionary key to use
        mode = self.perspective_method or 'none'

        # 1) Slice out the 2D block for those frame‐IDs
        all_k2d = self.precomputed_key2d[mode]    # shape = (M, 17, C2)
        block2d  = all_k2d[fids, ...].astype(np.float32)  # (T,17,C2)
        # split off the last channel as visibility
        vis2d_block = block2d[..., -1]             # (T,17)
        key2d_block = block2d[..., :-1].astype(np.float32)  # (T,17,C2d)

        # --- NEW: append cached per-joint features (raw), window-aligned ---
        # Concatenation order: [base (xy/xyc/xycd)] + [RTM 1024D if enabled] + [DAV2 256D if enabled]
        extras = []
        if os.getenv('USE_IMG_FEATS', 'false').lower() == 'true' and 'rtm_joint_feats' in self.ann_data:
            img_block = self.ann_data['rtm_joint_feats'][fids, ...].astype(np.float32)   # (T,17,1024)
            extras.append(img_block)
        if os.getenv('USE_DEPTH_FEATS', 'false').lower() == 'true' and 'dav2_joint_feats' in self.ann_data:
            dep_block = self.ann_data['dav2_joint_feats'][fids, ...].astype(np.float32)  # (T,17,256)
            extras.append(dep_block)
        if len(extras) > 0:
            key2d_block = np.concatenate([key2d_block] + extras, axis=-1)  # (T,17,C2d+…)

        # 2) Slice out the 3D block for those frame‐IDs
        all_k3d = self.precomputed_key3d[mode]    # shape = (M, 17, C3)
        block3d = all_k3d[fids, ...].astype(np.float32)  # (T,17,C3)
        # split off last channel as visibility
        vis3d_block = block3d[..., -1]             # (T,17)
        key3d_block  = block3d[..., :-1]           # (T,17,C3–1) e.g. (T,17,3) or (T,17,4…) etc.

        # 3) Determine target indices for “lifting_target”
        # if self.causal:
        #     tgt_idx = [-1]
        # else:
        #     tgt_idx = [len(key3d_block) // 2]
        # if self.multiple_target:
        #     tgt_idx = list(range(self.multiple_target))
        # 3) Determine target indices for “lifting_target”
        if self.causal:
            tgt_idx = [-1]
        else:
            tgt_idx = [len(key3d_block) // 2]

        # if self.multiple_target:
        #     # Use the actual sequence length instead of the configured multiple_target
        #     T_actual = key3d_block.shape[0]
        #     tgt_idx = list(range(T_actual))

        if self.multiple_target:
            # always pick exactly `multiple_target` targets (never shorter)
            tgt_idx = list(range(self.multiple_target))


        # inside prepare_data, replace both lines

        # old:
        #   'img_paths': [ self.ann_data['imgname'][f] for f in fids ],
        #   'target_img_path': self.ann_data['imgname'][fids[tgt_idx[0]]],

        # new vectorized version:
        block_paths = self.ann_data['imgname'][fids]      # (243,) object‐array slice in one call




        # 4) Build final data_info exactly as your pipeline expects
        data_info = {
            'id'                      : info['id'],
            'num_keypoints'           : self.metainfo['num_keypoints'],
            'keypoints'               : key2d_block,                    # (T,17,C2–1)
            'keypoints_visible'       : vis2d_block,                    # (T,17)
            'keypoints_3d'            : key3d_block,                    # (T,17,C3–1)
            'keypoints_3d_visible'    : vis3d_block,                    # (T,17)
            'scale'                   : info['scale'],                  # float32
            'center'                  : info['center'],                 # (1,2)
            'lifting_target'          : key3d_block[tgt_idx],           # (ntgt,17,C3–1)
            'lifting_target_visible'  : vis3d_block[tgt_idx],           # (ntgt,17)
            # 'img_paths'               : [ self.ann_data['imgname'][f] for f in fids ],
            'img_paths': block_paths.tolist(),
            'img_ids'                 : fids,                           # array length=T
            # 'target_img_path'         : self.ann_data['imgname'][ fids[tgt_idx[0]] ],
            'target_img_path': [block_paths[tgt_idx[0]].item()],
            'camera_param'            : info['camera_param'],           # small dict or None
        }

        meta_info = self.get_data_info(idx)
        for key in ['upper_body_ids','lower_body_ids','flip_pairs','dataset_keypoint_weights','flip_indices','skeleton_links']:
            data_info[key] = deepcopy(meta_info[key])


        # 5) Send data_info through the rest of the pipeline
        return self.pipeline(data_info)

    # @force_full_init
    # def prepare_data(self, idx) -> Any:
    #     """
    #     1. Read only the needed 243 rows from mmaped .npz
    #     2. Apply your old “dual/xyd/xyc/xycd/OD_estimation” code on that slice
    #     3. Build exactly the same data_info dict your pipelines expect
    #     """
    #     info = self.data_list[idx]
    #     fids = info['frame_ids']              # e.g. array([100,101,…,342])

    #     ann = self.ann_data  # (already opened with mmap_mode='r')

    #     # 1── Slice only the raw 2D/3D for these 243 frame_ids:
    #     k2d_raw = ann['part'][fids].astype(np.float32)   # (243,17,3)
    #     k3d_raw = ann['S'   ][fids].astype(np.float32)   # (243,17,4_or_3)

    #     # Split into “keypoints + visibility”:
    #     key2d = k2d_raw[..., :2]        # (243,17,2)
    #     vis2d = k2d_raw[...,  2]        # (243,17)
    #     if k3d_raw.shape[-1] == 4:
    #         key3d = k3d_raw[..., :3]    # (243,17,3)
    #         vis3d = k3d_raw[...,  3]    # (243,17)
    #     else:
    #         key3d = k3d_raw            # (243,17,3) if no explicit 4th channel
    #         vis3d = vis2d              # fallback to 2D visibility

    #     # ── Slice only those same 243 rows from any auxiliary arrays,
    #     #     squeeze any existing singleton dimension, then add exactly one new axis:
    #     raw_scores = ann['predicted_keypoints_score'][fids].astype(np.float32)  # e.g. (243,17) or (243,17,1)
    #     if raw_scores.ndim == 3:
    #         raw_scores = raw_scores.squeeze(-1)  # becomes (243,17)
    #     # pred_scores = raw_scores[..., None]       # now guaranteed (243,17,1)
    #     pred_scores = raw_scores

    #     raw_depth = ann['predicted_da_depth'][fids].astype(np.float32)         # e.g. (243,17) or (243,17,1)
    #     if raw_depth.ndim == 3:
    #         raw_depth = raw_depth.squeeze(-1)   # becomes (243,17)
    #     # da_depth = raw_depth[..., None]          # now guaranteed (243,17,1)
    #     da_depth = raw_depth


    #     # 2── Now branch into each perspective mode, but ONLY on this (243×17×C) block:
    #     if self.perspective_method == 'dual_perspectives':
    #         part_v2 = ann['part_v2'][fids].astype(np.float32)  # (243,17,3)

    #         inter = np.empty((2 * len(fids), 17, 3), dtype=np.float32)
    #         inter[0::2] = k2d_raw
    #         inter[1::2] = part_v2
    #         key2d = inter[..., :2]
    #         vis2d = inter[...,  2]

    #         key3d = np.repeat(key3d, 2, axis=0)   # (2*243,17,3)
    #         vis3d = np.repeat(vis3d, 2, axis=0)
    #         fids  = np.repeat(fids, 2, axis=0)

    #     elif self.perspective_method == 'xyd':
    #         if self.dataset_type == '3dhp-train':
    #             valid_idx = np.load("/…/2_9_3dhp_skip_valid_indices.npy")
    #             key2d = key2d[valid_idx]
    #             key3d = key3d[valid_idx]
    #             vis2d = vis2d[valid_idx]
    #             vis3d = vis3d[valid_idx]
    #             fids  = fids[valid_idx]

    #         # pass only the 243×17×1 “da_depth” slice to concatenator
    #         k2d_plus = concatenate_kpts_2d_with_auxiliary_xyd(
    #             key2d, {'predicted_da_depth': da_depth}, perspective_method='xyd'
    #         )
    #         key2d      = k2d_plus[..., : k2d_plus.shape[-1] - 1]
    #         vis2d      = k2d_plus[..., -1]

    #         ord_depth = load_auxiliary_params_xycd(self.ann_data, len(key2d), self.random_indices)['ordinal_depths']
    #         ord_depth = ord_depth[..., None]     # (243,17,1)
    #         key3d = np.concatenate([key3d, ord_depth], axis=-1)
    #         # vis3d remains unchanged

    #     elif self.perspective_method == 'xyc':
    #         if self.dataset_type == '3dhp-train':
    #             valid_idx = np.load("/…/2_9_3dhp_skip_valid_indices.npy")
    #             key2d     = key2d[valid_idx]
    #             key3d     = key3d[valid_idx]
    #             vis2d     = vis2d[valid_idx]
    #             vis3d     = vis3d[valid_idx]
    #             fids      = fids[valid_idx]

    #         k2d_plus = concatenate_kpts_2d_with_auxiliary_xyc(
    #             key2d, {'predicted_keypoints_score': pred_scores}, perspective_method='xyc'
    #         )
    #         key2d = k2d_plus[..., : k2d_plus.shape[-1] - 1]
    #         vis2d = k2d_plus[..., -1]
    #         # keep key3d and vis3d unchanged

    #     elif self.perspective_method == 'xycd':
    #         if self.dataset_type == '3dhp-train':
    #             valid_idx = np.load("/…/2_9_3dhp_skip_valid_indices.npy")
    #             key2d     = key2d[valid_idx]
    #             key3d     = key3d[valid_idx]
    #             vis2d     = vis2d[valid_idx]
    #             vis3d     = vis3d[valid_idx]
    #             fids      = fids[valid_idx]

    #         k2d_plus = concatenate_kpts_2d_with_auxiliary_xycd(
    #             key2d,
    #             {
    #               'predicted_keypoints_score': pred_scores,
    #               'predicted_da_depth'      : da_depth
    #             },
    #             perspective_method='xycd'
    #         )
    #         key2d = k2d_plus[..., : k2d_plus.shape[-1] - 1]
    #         vis2d = k2d_plus[..., -1]

    #         ord_depth = load_auxiliary_params_xycd(self.ann_data, len(key2d), self.random_indices)['ordinal_depths'][..., None]
    #         root_z   = key3d[...,  2]  # (243,17)
    #         root_z   = root_z[..., None]  # (243,17,1)
    #         key3d = np.concatenate([
    #             key3d[..., :3],   # (243,17,3)
    #             ord_depth,        # (243,17,1)
    #             root_z,           # (243,17,1)
    #         ], axis=-1)          # final (243,17,5)
    #         # vis3d stays the same

    #     elif 'OD_estimation' in self.perspective_method:
    #         aux = load_auxiliary_params(self.ann_data, len(key2d), self.random_indices)
    #         if 'BL' in self.perspective_method and 'RT' in self.perspective_method:
    #             k2d_plus = concatenate_kpts_2d_with_auxiliary(
    #                 key2d, aux, perspective_method='BL_RT')
    #         elif 'BL' in self.perspective_method:
    #             k2d_plus = concatenate_kpts_2d_with_auxiliary(
    #                 key2d, aux, perspective_method='BL')
    #         else:
    #             k2d_plus = key2d  # fallback to raw 2D

    #         if 'RD' in self.perspective_method:
    #             root_z = key3d[..., 2]        # (243,17)
    #             root_z = root_z[..., None]     # (243,17,1)
    #             k2d_plus = np.concatenate([k2d_plus, root_z], axis=-1)

    #         key2d = k2d_plus[..., : k2d_plus.shape[-1] - 1]
    #         vis2d = k2d_plus[..., -1]

    #         ord_depth = aux['ordinal_depths'][..., None]  # (243,17,1)
    #         gt2d_xy   = key2d[..., :2]                    # (243,17,2)
    #         key3d = np.concatenate([
    #             key3d[..., :3],    # (243,17,3)
    #             ord_depth,         # (243,17,1)
    #             gt2d_xy,           # (243,17,2)
    #             vis3d[..., None],  # (243,17,1)
    #         ], axis=-1)            # final (243,17,7)
    #         vis3d = key3d[..., -1]  # final visibility channel

    #     else:
    #         # no special perspective
    #         key2d, vis2d = key2d, vis2d
    #         key3d, vis3d = key3d, vis3d

    #     # 3── Determine target indices ([-1] if causal else [mid], or multiple_target):
    #     if self.causal:
    #         tgt_idx = [-1]
    #     else:
    #         tgt_idx = [len(key3d) // 2]
    #     if self.multiple_target:
    #         tgt_idx = list(range(self.multiple_target))

    #     # 4── Build the final data_info exactly as your existing pipeline expects:
    #     data_info = {
    #         'id':                     info['id'],
    #         'num_keypoints':          self.metainfo['num_keypoints'],
    #         'keypoints':              key2d,                     # (T,17,C2)
    #         'keypoints_visible':      vis2d,                     # (T,17)
    #         'keypoints_3d':           key3d,                     # (T,17,C3)
    #         'keypoints_3d_visible':   vis3d,                     # (T,17)
    #         'scale':                  info['scale'],             # float32
    #         'center':                 info['center'],            # (1,2)
    #         'lifting_target':         key3d[tgt_idx],            # (ntgt,17,C3)
    #         'lifting_target_visible': vis3d[tgt_idx],            # (ntgt,17)
    #         'img_paths':              [ann['imgname'][f] for f in fids], 
    #         'img_ids':                fids,                      # array of length T
    #         'target_img_path':        ann['imgname'][fids[tgt_idx[0]]],
    #         'camera_param':           info['camera_param'],      # small dict or None
    #     }
    #     return self.pipeline(data_info)

    # @force_full_init
    # def prepare_data(self, idx) -> Any:
    #     """Get data processed by ``self.pipeline``.

    #     :class:`BaseCocoStyleDataset` overrides this method from
    #     :class:`mmengine.dataset.BaseDataset` to add the metainfo into
    #     the ``data_info`` before it is passed to the pipeline.

    #     Args:
    #         idx (int): The index of ``data_info``.

    #     Returns:
    #         Any: Depends on ``self.pipeline``.
    #     """
    #     data_info = self.get_data_info(idx)

    #     return self.pipeline(data_info)

    def get_data_info(self, idx: int) -> dict:
        """Get data info by index.

        Args:
            idx (int): Index of data info.

        Returns:
            dict: Data info.
        """
        data_info = super().get_data_info(idx)

        # Add metainfo items that are required in the pipeline and the model
        metainfo_keys = [
            'upper_body_ids', 'lower_body_ids', 'flip_pairs',
            'dataset_keypoint_weights', 'flip_indices', 'skeleton_links'
        ]

        for key in metainfo_keys:
            assert key not in data_info, (
                f'"{key}" is a reserved key for `metainfo`, but already '
                'exists in the `data_info`.')

            data_info[key] = deepcopy(self._metainfo[key])

        # === OPTIONAL attachments (safe no-ops when not present) ===
        # Attach cached feature maps if they exist in the NPZ
        if isinstance(self.ann_data, dict):
            # Extract frame indices for this sequence
            frame_ids = data_info.get('img_ids', None)
            if frame_ids is not None:
                # RTM detection features (17x192 per frame)
                if 'rtm_joint_feats' in self.ann_data:
                    data_info['rtm_joint_feats'] = self.ann_data['rtm_joint_feats'][frame_ids].astype(np.float32)
                # DAV2 depth features (17x256 per frame)
                if 'dav2_joint_feats' in self.ann_data:
                    data_info['dav2_joint_feats'] = self.ann_data['dav2_joint_feats'][frame_ids].astype(np.float32)
                # Feature map grid dimensions (for reference)
                if 'rtm_grid_hw' in self.ann_data:
                    data_info['rtm_grid_hw'] = self.ann_data['rtm_grid_hw']
                if 'dav2_grid_hw' in self.ann_data:
                    data_info['dav2_grid_hw'] = self.ann_data['dav2_grid_hw']
        
        # === NEW: Append raw feature maps to 2D keypoints (before codec) ===
        # Concatenate cached per-joint features if flags are enabled
        if isinstance(self.ann_data, dict):
            frame_ids = data_info.get('img_ids', None)
            if frame_ids is not None:
                kpts_2d = data_info.get('keypoints', None)  # (T, K, C2d)
                if kpts_2d is not None:
                    extras = []
                    
                    # RTM image features (192-D per joint)
                    if os.getenv('USE_IMG_FEATS', 'false').lower() == 'true':
                        if 'rtm_joint_feats' in self.ann_data:
                            img_block = self.ann_data['rtm_joint_feats'][frame_ids].astype(np.float32)
                            extras.append(img_block)
                    
                    # DAV2 depth features (256-D per joint)
                    if os.getenv('USE_DEPTH_FEATS', 'false').lower() == 'true':
                        if 'dav2_joint_feats' in self.ann_data:
                            dep_block = self.ann_data['dav2_joint_feats'][frame_ids].astype(np.float32)
                            extras.append(dep_block)
                    
                    # Concatenate: (T,K,C2d) + [(T,K,192)] + [(T,K,256)] → (T,K,C_total)
                    if len(extras) > 0:
                        data_info['keypoints'] = np.concatenate([kpts_2d] + extras, axis=-1)
        
        return data_info

    def load_data_list(self) -> List[dict]:
        """Load data list from COCO annotation file or person detection result
        file."""

        instance_list, image_list = self._load_annotations()

        if self.data_mode == 'topdown':
            data_list = self._get_topdown_data_infos(instance_list)
        else:
            data_list = self._get_bottomup_data_infos(instance_list,
                                                      image_list)

        return data_list

    def get_img_info(self, img_idx, img_name):
        try:
            with get_local_path(osp.join(self.data_prefix['img'],
                                         img_name)) as local_path:
                im = cv2.imread(local_path)
                h, w, _ = im.shape
        except:  # noqa: E722
            print_log(
                f'Failed to read image {img_name}.',
                logger='current',
                level=logging.DEBUG)
            return None

        img = {
            'file_name': img_name,
            'height': h,
            'width': w,
            'id': img_idx,
            'img_id': img_idx,
            'img_path': osp.join(self.data_prefix['img'], img_name),
        }
        return img

    def get_sequence_indices(self) -> List[List[int]]:
        """Build sequence indices.

        The default method creates sample indices that each sample is a single
        frame (i.e. seq_len=1). Override this method in the subclass to define
        how frames are sampled to form data samples.

        Outputs:
            sample_indices: the frame indices of each sample.
                For a sample, all frames will be treated as an input sequence,
                and the ground-truth pose of the last frame will be the target.
        """
        sequence_indices = []
        if self.seq_len == 1:
            num_imgs = len(self.ann_data['imgname'])
        else:
            raise NotImplementedError('Multi-frame data sample unsupported!')

        if self.multiple_target > 0:
            sequence_indices_merged = []
            for i in range(0, len(sequence_indices), self.multiple_target):
                if i + self.multiple_target > len(sequence_indices):
                    break
                sequence_indices_merged.append(
                    list(
                        itertools.chain.from_iterable(
                            sequence_indices[i:i + self.multiple_target])))
            sequence_indices = sequence_indices_merged
        return sequence_indices

    def _load_annotations(self) -> Tuple[List[dict], List[dict]]:
        """
        1. Load the raw "part" (2D) and "S" (3D) arrays from self.ann_data (mmap_mode='r').
        2. Split raw arrays into coords + visibility, then build one large "None"‐mode array.
        3. For each perspective mode (dual_perspectives, xyd, xyc, xycd, OD_estimation), build
           one big NumPy array of shape either (N,17,C) or (2*N,17,C).  Store all of these inside
           two dictionaries: self.precomputed_key2d and self.precomputed_key3d.
        4. Build instance_list exactly as before (only storing 'frame_ids', 'center', 'scale', 'camera_param'),
           never storing any of the big 243×… blocks.
        """
        num_keypoints = self.metainfo['num_keypoints']
        img_names = self.ann_data['imgname']       # shape = (N,)
        num_imgs = len(img_names)                 # N

        # ——————————————————————————————————————————————————————————————————————————————
        # STEP 0: Apply min-max normalization to feature maps (OPTIONAL)
        # ——————————————————————————————————————————————————————————————————————————————
        # Only normalize if normalize_feats=True (default: False, expects pre-normalized files)
        
        if self.normalize_feats:
            print("WARNING: Runtime feature normalization enabled (slow!)")
            print("   Consider using pre-normalized files instead:")
            print("   python precompute_feature_normalization.py --all")
            
            # Skip normalization if features are already normalized (check for typical range)
            def is_already_normalized(x, threshold=0.8):
                """Check if array is already in [-1, 1] range."""
                if x.size == 0:
                    return True
                sample_size = min(10000, x.size)
                sample = x.flat[:sample_size]
                in_range = np.sum((sample >= -1.1) & (sample <= 1.1)) / len(sample)
                return in_range > threshold
            
            def normalize_minmax_fast(x, eps=1e-6):
                """
                Fast robust min-max normalization to [-1, 1] using approximate percentiles.
                Uses np.quantile on a 50k-sample subset for speed on large arrays.
                """
                # For very large arrays (>1M elements), estimate percentiles from a sample
                if x.size > 1_000_000:
                    # Use uniform random indices instead of np.random.choice for 10x speedup
                    np.random.seed(42)
                    indices = np.random.randint(0, x.size, size=50000)
                    sample = x.flat[indices]
                    minv, maxv = np.percentile(sample, [1, 99])
                else:
                    minv, maxv = np.percentile(x, [1, 99])
                
                # Clip outliers, normalize to [0,1], then shift to [-1,1]
                return 2 * (np.clip(x, minv, maxv) - minv) / (maxv - minv + eps) - 1
            
            # Normalize RTM image features (17x192 per frame) - in-place for memory efficiency
            if 'rtm_joint_feats' in self.ann_data:
                rtm_data = self.ann_data['rtm_joint_feats']
                if is_already_normalized(rtm_data):
                    print(f"RTM features already normalized ({rtm_data.nbytes / 1e9:.2f} GB) - skipping")
                else:
                    print(f"Normalizing RTM features ({rtm_data.nbytes / 1e9:.2f} GB)...")
                    self.ann_data['rtm_joint_feats'] = normalize_minmax_fast(rtm_data).astype(np.float32)
            
            # Normalize DAV2 depth features (17x256 per frame) - in-place for memory efficiency
            if 'dav2_joint_feats' in self.ann_data:
                dav2_data = self.ann_data['dav2_joint_feats']
                if is_already_normalized(dav2_data):
                    print(f"DAV2 features already normalized ({dav2_data.nbytes / 1e9:.2f} GB) - skipping")
                else:
                    print(f"Normalizing DAV2 features ({dav2_data.nbytes / 1e9:.2f} GB)...")
                    self.ann_data['dav2_joint_feats'] = normalize_minmax_fast(dav2_data).astype(np.float32)
        else:
            print("OK: Using pre-normalized features (normalize_feats=False)")

        # ——————————————————————————————————————————————————————————————————————————————
        # STEP A: Load raw 2D and 3D arrays from disk (mmap) and split coords vs. visibility
        # ——————————————————————————————————————————————————————————————————————————————

        # (1) Raw 2D keypoints (part): shape = (N, 17, 3)
        raw_part = self.ann_data['part'].astype(np.float32)     # (N,17,3)
        coords2d = raw_part[..., :2]                            # (N,17,2)
        vis2d    = raw_part[..., 2]                             # (N,17)

        # (2) Raw 3D keypoints (S): shape = (N, 17, 4) or (N, 17, 3)
        raw_S = self.ann_data['S'].astype(np.float32)           # (N,17,4_or_3)
        coords3d = raw_S[..., :3]                               # (N,17,3)
        if raw_S.shape[-1] == 4:
            vis3d = raw_S[..., 3]                               # (N,17)
        else:
            # fallback: if there is no explicit 4th channel, use 2D visibility
            vis3d = vis2d                                       # (N,17)

        N = num_imgs

        # Build the “none” mode arrays (just raw coords + vis)
        #   key2d_none  shape = (N, 17, 3)   ; last channel = visibility
        #   key3d_none  shape = (N, 17, 4)   ; last channel = visibility
        key2d_none = np.concatenate([coords2d, vis2d[..., None]], axis=-1)   # (N,17,3)
        key3d_none = np.concatenate([coords3d, vis3d[..., None]], axis=-1)   # (N,17,4)

        # Initialize the dictionaries
        self.precomputed_key2d = {
            'none': key2d_none
        }
        self.precomputed_key3d = {
            'none': key3d_none
        }

        # ——————————————————————————————————————————————————————————————————————————————
        # STEP B: Build “dual_perspectives” mode if requested
        # ——————————————————————————————————————————————————————————————————————————————
        if getattr(self, 'perspective_method', '') == 'dual_perspectives' or 'dual_perspectives' in getattr(self, 'perspective_method', ''):
            # Load the second set of part_v2 from disk
            part_v2 = self.ann_data['part_v2'].astype(np.float32)   # (N,17,3)
            coords2d_v2 = part_v2[..., :2]                          # (N,17,2)
            vis2d_v2    = part_v2[..., 2]                           # (N,17)

            # We want to interleave [0,1,2,...,N-1] with [0,1,2,...,N-1] → array of length 2N.
            # Build an empty container of shape (2N,17,3).
            dual_key2d = np.empty((2 * N, num_keypoints, 3), dtype=np.float32)
            # Interleave coords2d + vis2d
            dual_coords2d = np.empty((2 * N, num_keypoints, 2), dtype=np.float32)
            dual_vis2d    = np.empty((2 * N, num_keypoints), dtype=np.float32)

            dual_coords2d[0::2] = coords2d                          # even positions
            dual_coords2d[1::2] = coords2d_v2                        # odd positions
            dual_vis2d   [0::2] = vis2d
            dual_vis2d   [1::2] = vis2d_v2

            dual_key2d = np.concatenate([dual_coords2d, dual_vis2d[..., None]], axis=-1)  # (2N,17,3)
            self.precomputed_key2d['dual_perspectives'] = dual_key2d

            # For 3D: just repeat coords3d and vis3d twice
            dual_coords3d = np.repeat(coords3d, 2, axis=0)        # (2N,17,3)
            dual_vis3d    = np.repeat(vis3d,    2, axis=0)        # (2N,17)
            dual_key3d = np.concatenate([dual_coords3d, dual_vis3d[..., None]], axis=-1)  # (2N,17,4)
            self.precomputed_key3d['dual_perspectives'] = dual_key3d

        # ——————————————————————————————————————————————————————————————————————————————
        # STEP C: Build “xyd” mode if requested
        # ——————————————————————————————————————————————————————————————————————————————
        if self.perspective_method == 'xyd' or 'xyd' in getattr(self, 'perspective_method', ''):
            # (1) Load the “predicted_da_depth” from disk, shape = (N,17) or (N,17,1).
            raw_da = self.ann_data['predicted_da_depth'].astype(np.float32)    # (N,17) or (N,17,1)
            if raw_da.ndim == 3:
                raw_da = raw_da.squeeze(-1)        # (N,17)

            # Build the per‐frame 2D+aux tensor.  We need to feed into the same function
            # as before, which expects an input of shape (N,17,C) where the last channel
            # is visibility.  So first reassemble coords2d+vis2d exactly as “none” mode:
            raw_k2d_for_xyd = np.concatenate([coords2d, vis2d[..., None]], axis=-1)  # (N,17,3)

            # Now call your helper to concatenate da_depth → shape = (N,17,Cxyd) where
            # the final channel is visibility.
            k2d_xyd_full = concatenate_kpts_2d_with_auxiliary_xyd(
                raw_k2d_for_xyd,
                {'predicted_da_depth': raw_da},
                perspective_method='xyd'
            )
            self.precomputed_key2d['xyd'] = k2d_xyd_full  # (N,17,Cxyd)

            # For 3D in “xyc” mode, we just keep raw coords3d+vis3d = (N,17,4):
            k3d_xyc_full = np.concatenate([coords3d, vis3d[..., None]], axis=-1)  # (N,17,4)
            self.precomputed_key3d['xyd'] = k3d_xyc_full

        # ——————————————————————————————————————————————————————————————————————————————
        # STEP D: Build “xyc” mode if requested
        # ——————————————————————————————————————————————————————————————————————————————
        if self.perspective_method == 'xyc' or 'xyc' in getattr(self, 'perspective_method', ''):
            # (1) Load “predicted_keypoints_score” from disk, shape = (N,17) or (N,17,1).
            raw_conf = self.ann_data['predicted_keypoints_score'].astype(np.float32)  # (N,17) or (N,17,1)
            if raw_conf.ndim == 3:
                raw_conf = raw_conf.squeeze(-1)       # (N,17)

            # Reassemble coords2d+vis2d = shape (N,17,3)
            raw_k2d_for_xyc = np.concatenate([coords2d, vis2d[..., None]], axis=-1)  # (N,17,3)

            # Call the helper that expects 2D+(last channel=vis) plus “predicted_keypoints_score”
            k2d_xyc_full = concatenate_kpts_2d_with_auxiliary_xyc(
                raw_k2d_for_xyc,
                {'predicted_keypoints_score': raw_conf},
                perspective_method='xyc'
            )
            self.precomputed_key2d['xyc'] = k2d_xyc_full  # (N,17,Cxyc)

            # For 3D in “xyc” mode, we just keep raw coords3d+vis3d = (N,17,4):
            k3d_xyc_full = np.concatenate([coords3d, vis3d[..., None]], axis=-1)  # (N,17,4)
            self.precomputed_key3d['xyc'] = k3d_xyc_full

        # ——————————————————————————————————————————————————————————————————————————————
        # STEP E: Build “xycd” mode if requested
        # ——————————————————————————————————————————————————————————————————————————————
        if self.perspective_method == 'xycd' or 'xycd' in getattr(self, 'perspective_method', ''):
            # (1) Load “predicted_keypoints_score” and “predicted_da_depth”
            raw_conf = self.ann_data['predicted_keypoints_score'].astype(np.float32)  # (N,17) or (N,17,1)
            if raw_conf.ndim == 3:
                raw_conf = raw_conf.squeeze(-1)       # (N,17)

            raw_da = self.ann_data['predicted_da_depth'].astype(np.float32)         # (N,17) or (N,17,1)
            if raw_da.ndim == 3:
                raw_da = raw_da.squeeze(-1)           # (N,17)

            # Reassemble coords2d+vis2d = shape (N,17,3)
            raw_k2d_for_xycd = np.concatenate([coords2d, vis2d[..., None]], axis=-1)  # (N,17,3)

            # Call the helper that takes both score and depth
            k2d_xycd_full = concatenate_kpts_2d_with_auxiliary_xycd(
                raw_k2d_for_xycd,
                {
                  'predicted_keypoints_score': raw_conf,
                  'predicted_da_depth'       : raw_da
                },
                perspective_method='xycd'
            )
            self.precomputed_key2d['xycd'] = k2d_xycd_full  # (N,17,Cxycd)

            # For 3D in “xycd” mode, we just keep raw coords3d+vis3d = (N,17,4):
            k3d_xyc_full = np.concatenate([coords3d, vis3d[..., None]], axis=-1)  # (N,17,4)
            self.precomputed_key3d['xycd'] = k3d_xyc_full

        # —————————————————————————————————————————————
        # STEP E2: Build "casp10d" mode if requested
        # —————————————————————————————————————————————
        # —————————————————————————————————————————————
        # ipdb.set_trace()
        if self.perspective_method == 'casp10d' or 'casp10d' in getattr(self, 'perspective_method', ''):
            casp_mode = os.getenv('CASP_MODE', 'v0')  # 'v0' or 'spatial'
            if casp_mode == 'spatial':
                casp_key = 'casp_descriptors'
            else:
                casp_key = 'summary_casp_descriptor_10d'
            casp_summary = self.ann_data.get(casp_key, None)
            if casp_summary is None:
                print(f"WARNING: CASP summary not found in annotations for key '{casp_key}', defaulting to zeros")
                casp_summary = np.zeros((N, num_keypoints, 10), dtype=np.float32)
            else:
                casp_summary = casp_summary.astype(np.float32)
            # ipdb.set_trace()
            raw_k2d_for_casp = np.concatenate([coords2d, vis2d[..., None]], axis=-1)  # (N,17,3)
            conf = self.ann_data.get('predicted_keypoints_score', None)
            if conf is not None:
                conf = conf.astype(np.float32)
            d_pt = self.ann_data.get('predicted_da_depth', None)
            if d_pt is not None:
                d_pt = d_pt.astype(np.float32)
            aux_dict = {'casp_descriptors': casp_summary}
            if casp_mode == 'spatial':
                aux_dict['predicted_keypoints_score'] = conf
                aux_dict['predicted_da_depth'] = d_pt
            k2d_casp_full = concatenate_kpts_2d_with_auxiliary_casp(
                raw_k2d_for_casp,
                aux_dict,
                casp_mode=casp_mode
            )
            self.precomputed_key2d['casp10d'] = k2d_casp_full  # (N,17,...) depends on mode
            k3d_casp_full = np.concatenate([coords3d, vis3d[..., None]], axis=-1)  # (N,17,4)
            self.precomputed_key3d['casp10d'] = k3d_casp_full
        # ——————————————————————————————————————————————————————————————————————————————
        # STEP F: Build “OD_estimation” mode if requested
        # ——————————————————————————————————————————————————————————————————————————————
        if 'OD_estimation' in getattr(self, 'perspective_method', ''):
            # (1) Load bone_lengths, ordinal_depths, camera_params_array from ann_data
            aux = load_auxiliary_params(self.ann_data, N, self.random_indices)
            # We will also reassemble coords2d+vis2d = (N,17,3)
            raw_k2d_for_OD = np.concatenate([coords2d, vis2d[..., None]], axis=-1)  # (N,17,3)

            # Decide which concatenation sub‐case: BL, BL_RT, or fallback to raw
            if 'BL' in self.perspective_method and 'RT' in self.perspective_method:
                k2d_od_full = concatenate_kpts_2d_with_auxiliary(
                    raw_k2d_for_OD, aux, perspective_method='BL_RT'
                )
            elif 'BL' in self.perspective_method:
                k2d_od_full = concatenate_kpts_2d_with_auxiliary(
                    raw_k2d_for_OD, aux, perspective_method='BL'
                )
            else:
                k2d_od_full = raw_k2d_for_OD.copy()   # fallback: no RT, no BL

            # If “RD” (root depth) is requested, tile root_z onto every joint
            if 'RD' in self.perspective_method:
                root_z_full = coords3d[..., 2]            # (N,17)
                root_z_full = root_z_full[..., None]      # (N,17,1)
                k2d_od_full = np.concatenate([k2d_od_full, root_z_full], axis=-1)

            self.precomputed_key2d['OD_estimation'] = k2d_od_full  # (N,17,COd)

            # Build the 3D output for OD: [ coords3d | ordinal_depth | raw_2d_xy | vis3d ]
            ord_depth = aux['ordinal_depths']                    # (N,17)
            ord_depth = ord_depth[..., None]                      # (N,17,1)
            raw_xy    = coords2d                                  # (N,17,2)
            vis3d_all = vis3d[..., None]                          # (N,17,1)
            k3d_od_full = np.concatenate([
                coords3d,          # (N,17,3)
                ord_depth,         # (N,17,1)
                raw_xy,            # (N,17,2)
                vis3d_all          # (N,17,1)
            ], axis=-1)               # final shape (N,17,7)
            self.precomputed_key3d['OD_estimation'] = k3d_od_full

        # ——————————————————————————————————————————————————————————————————————————————
        # STEP G: Now that all “precomputed_key2d” and “precomputed_key3d” exist,
        # build instance_list exactly as before (WITHOUT storing any big blocks).
        # ——————————————————————————————————————————————————————————————————————————————

        # First, optionally apply DEBUG_DATASET filtering to self.sequence_indices:
        if DEBUG_DATASET:
            if LONG_SEQ_SIXTHS:
                sixth_len = num_imgs // 6
                sixth_idx = int(LONG_SEQ_SIXTHS) - 1
                start_idx = sixth_idx * sixth_len
                end_idx   = (sixth_idx + 1) * sixth_len
                new_seq_idxs = []
                for seq in self.sequence_indices:
                    if all((fid >= start_idx and fid < end_idx) for fid in seq):
                        new_seq_idxs.append(seq)
                self.sequence_indices = new_seq_idxs
            else:
                new_seq_idxs = []
                for seq in self.sequence_indices:
                    if all(fid < NUM_DEBUG_FRAMES for fid in seq):
                        new_seq_idxs.append(seq)
                self.sequence_indices = new_seq_idxs

        # Load only per‐frame metadata (centers/scales)
        centers = self.ann_data.get('center', np.zeros((num_imgs, 2), dtype=np.float32))
        scales  = self.ann_data.get('scale',  np.zeros((num_imgs,),    dtype=np.float32))

        instance_list = []
        image_list    = []

        for idx, fids in enumerate(self.sequence_indices):
            inst = {
                'id'         : idx,
                # Keep frame‐IDs as int32 array
                'frame_ids'  : np.asarray(fids, dtype=np.int32),
                # Use center/scale of the FIRST frame in the sequence
                'center'     : centers[fids[0]].reshape(1, -1).astype(np.float32),
                'scale'      : np.array(scales[fids[0]], dtype=np.float32),
                'camera_param':
                    (self.get_camera_param(img_names[fids[0]])
                     if self.camera_param_file else None),
            }
            instance_list.append(inst)

        # If bottomup mode is requested, also build image_list for every frame
        if self.data_mode == 'bottomup':
            image_list = [
                self.get_img_info(i, img_names[i])
                for i in range(num_imgs)
            ]

        return instance_list, image_list

    # def _load_annotations(self) -> Tuple[List[dict], List[dict]]:
    #     """
    #     Only record, for each sequence:
    #     - 'id':          integer sequence index
    #     - 'frame_ids':   np.int32 array of length=self.seq_len (or multiple_target)
    #     - 'center':      (1×2) float32
    #     - 'scale':       single float32
    #     - 'camera_param': small dict or None
    #     Never load the full 'part' or 'S' arrays here.
    #     """
    #     num_keypoints = self.metainfo['num_keypoints']
    #     img_names = self.ann_data['imgname']          # shape = (num_imgs,)
    #     num_imgs = len(img_names)

    #     # --- DEBUG_DATASET filtering (optional) ---
    #     if DEBUG_DATASET:
    #         if LONG_SEQ_SIXTHS:
    #             # Keep only sequences whose frame_ids lie entirely within one sixth of the video
    #             sixth_len = num_imgs // 6
    #             sixth_idx = int(LONG_SEQ_SIXTHS) - 1
    #             start_idx = sixth_idx * sixth_len
    #             end_idx = (sixth_idx + 1) * sixth_len
    #             valid_set = set(range(start_idx, end_idx))
    #             new_seq_idxs = []
    #             for seq in self.sequence_indices:
    #                 # all(frame_id in [start_idx, end_idx))
    #                 if all((fid >= start_idx and fid < end_idx) for fid in seq):
    #                     new_seq_idxs.append(seq)
    #             self.sequence_indices = new_seq_idxs
    #         else:
    #             # Keep only sequences whose frame_ids all < NUM_DEBUG_FRAMES
    #             new_seq_idxs = []
    #             for seq in self.sequence_indices:
    #                 if all(fid < NUM_DEBUG_FRAMES for fid in seq):
    #                     new_seq_idxs.append(seq)
    #             self.sequence_indices = new_seq_idxs

    #     # --- Load only per-frame metadata (no big slicing) ---
    #     centers = self.ann_data.get('center',
    #             np.zeros((num_imgs, 2), dtype=np.float32))
    #     scales  = self.ann_data.get('scale',
    #             np.zeros((num_imgs,), dtype=np.float32))

    #     instance_list = []
    #     image_list = []

    #     for idx, fids in enumerate(self.sequence_indices):
    #         inst = {
    #             'id': idx,
    #             # Keep the frame‐IDs as int32 (length=self.seq_len, e.g. 243):
    #             'frame_ids': np.asarray(fids, dtype=np.int32),
    #             # Use the center/scale of the FIRST frame in this sequence:
    #             'center': centers[fids[0]].reshape(1, -1).astype(np.float32),  
    #             'scale':  np.array(scales[fids[0]], dtype=np.float32),
    #             'camera_param':
    #                 (self.get_camera_param(img_names[fids[0]])
    #                 if self.camera_param_file else None),
    #         }
    #         instance_list.append(inst)

    #     # If bottomup mode is requested, we also build an image_list of every frame (rarely used for 3D lifting)
    #     if self.data_mode == 'bottomup':
    #         image_list = [
    #             self.get_img_info(i, img_names[i])
    #             for i in range(num_imgs)
    #         ]

    #     # ipdb.set_trace()
    #     return instance_list, image_list


    def get_camera_param(self, imgname):
        """Get camera parameters of a frame by its image name.

        Override this method to specify how to get camera parameters.
        """
        raise NotImplementedError

    @staticmethod
    def _is_valid_instance(data_info: Dict) -> bool:
        """Check a data info is an instance with valid bbox and keypoint
        annotations."""
        # crowd annotation
        if 'iscrowd' in data_info and data_info['iscrowd']:
            return False
        # invalid keypoints
        if 'num_keypoints' in data_info and data_info['num_keypoints'] == 0:
            return False
        # invalid keypoints
        if 'keypoints' in data_info:
            if np.max(data_info['keypoints']) <= 0:
                return False
        return True

    def _get_topdown_data_infos(self, instance_list: List[Dict]) -> List[Dict]:
        """Organize the data list in top-down mode."""
        # sanitize data samples
        data_list_tp = list(filter(self._is_valid_instance, instance_list))

        return data_list_tp

    def _get_bottomup_data_infos(self, instance_list: List[Dict],
                                 image_list: List[Dict]) -> List[Dict]:
        """Organize the data list in bottom-up mode."""

        # bottom-up data list
        data_list_bu = []

        used_img_ids = set()

        # group instances by img_id
        for img_ids, data_infos in groupby(instance_list,
                                           lambda x: x['img_ids']):
            for img_id in img_ids:
                used_img_ids.add(img_id)
            data_infos = list(data_infos)

            # image data
            img_paths = data_infos[0]['img_paths']
            data_info_bu = {
                'img_ids': img_ids,
                'img_paths': img_paths,
            }

            for key in data_infos[0].keys():
                if key not in data_info_bu:
                    seq = [d[key] for d in data_infos]
                    if isinstance(seq[0], np.ndarray):
                        seq = np.concatenate(seq, axis=0)
                    data_info_bu[key] = seq

            # The segmentation annotation of invalid objects will be used
            # to generate valid region mask in the pipeline.
            invalid_segs = []
            for data_info_invalid in filterfalse(self._is_valid_instance,
                                                 data_infos):
                if 'segmentation' in data_info_invalid:
                    invalid_segs.append(data_info_invalid['segmentation'])
            data_info_bu['invalid_segs'] = invalid_segs

            data_list_bu.append(data_info_bu)

        # add images without instance for evaluation
        if self.test_mode:
            for img_info in image_list:
                if img_info['img_id'] not in used_img_ids:
                    data_info_bu = {
                        'img_ids': [img_info['img_id']],
                        'img_path': [img_info['img_path']],
                        'id': list(),
                    }
                    data_list_bu.append(data_info_bu)

        return data_list_bu

def load_auxiliary_params_xycd(ann_data, num_imgs, random_indices):
    """Load auxiliary parameters and return a dictionary."""
    keys = ['predicted_da_depth', 'predicted_keypoints_score', 'predicted_keypoints']
    aux_params = {key: ann_data.get(key, np.zeros((num_imgs, 17), dtype=np.float32)) for key in keys}

    # Ensure all auxiliary parameter shapes match the number of images
    # for key, value in aux_params.items():
        # if DEBUG_DATASET: # should be after loading
        #     aux_params[key] = value[random_indices]  # Apply random indices here
        # assert aux_params[key].shape[0] == len(random_indices), f"{key} mismatch with num_imgs"
    
    return aux_params

def concatenate_kpts_2d_with_auxiliary_xyc(kpts_2d, aux_params, perspective_method):
    """Concatenate 2D keypoints with auxiliary parameters based on the perspective method."""
    kpts_2d_coords = kpts_2d[:, :, :-1]  # Exclude visibility
    kpts_2d_visibility = kpts_2d[:, :, -1:]  # Extract visibility

    # Initialize components to concatenate
    components = [kpts_2d_coords]


    components.append(aux_params['predicted_keypoints_score'][..., None])
    print("keypoint_conf")

    components.append(kpts_2d_visibility)  # Append visibility last

    # ipdb.set_trace()
    # Concatenate all components along the last axis
    return np.concatenate(components, axis=-1)

def concatenate_kpts_2d_with_auxiliary_xyd(kpts_2d, aux_params, perspective_method):
    """Concatenate 2D keypoints with auxiliary parameters based on the perspective method."""
    kpts_2d_coords = kpts_2d[:, :, :-1]  # Exclude visibility
    kpts_2d_visibility = kpts_2d[:, :, -1:]  # Extract visibility

    # Initialize components to concatenate
    components = [kpts_2d_coords]


    components.append(aux_params['predicted_da_depth'][..., None])
    print("keypoint_dav2_depth")

    components.append(kpts_2d_visibility)  # Append visibility last

    # Concatenate all components along the last axis
    return np.concatenate(components, axis=-1)

def concatenate_kpts_2d_with_auxiliary_xycd(kpts_2d, aux_params, perspective_method):
    """Concatenate 2D keypoints with auxiliary parameters based on the perspective method."""
    kpts_2d_coords = kpts_2d[:, :, :-1]  # Exclude visibility
    kpts_2d_visibility = kpts_2d[:, :, -1:]  # Extract visibility

    # Initialize components to concatenate
    components = [kpts_2d_coords]

    # Prepare predicted keypoints scores
    pred_scores = aux_params.get('predicted_keypoints_score', None)
    # Check if pred_scores has the expected shape; otherwise, default to zeros
    
    if pred_scores is None or pred_scores.shape != (kpts_2d.shape[0], 17):
        # print(" Predicted keypoints scores missing or incorrectly shaped; defaulting to zeros.")
        pred_scores = np.zeros((kpts_2d.shape[0], 17))

    components.append(pred_scores[..., None])
    print("keypoint_conf")
    components.append(aux_params['predicted_da_depth'][..., None])
    print("keypoint_dav2_depth")

    components.append(kpts_2d_visibility)  # Append visibility last

    # Concatenate all components along the last axis
    return np.concatenate(components, axis=-1)

def load_auxiliary_params(ann_data, num_imgs, random_indices):
    """Load auxiliary parameters and return a dictionary."""
    keys = ['camera_params_array', 'bone_lengths', 'ordinal_depths']
    aux_params = {key: ann_data.get(key, np.zeros((num_imgs, 17), dtype=np.float32)) for key in keys}

    # Ensure all auxiliary parameter shapes match the number of images
    # for key, value in aux_params.items():
        # if DEBUG_DATASET:
        #     aux_params[key] = value[random_indices]  # Apply random indices here
        # assert aux_params[key].shape[0] == len(random_indices), f"{key} mismatch with num_imgs"
    
    return aux_params


def concatenate_kpts_2d_with_auxiliary(kpts_2d, aux_params, perspective_method):
    """Concatenate 2D keypoints with auxiliary parameters based on the perspective method."""
    kpts_2d_coords = kpts_2d[:, :, :-1]  # Exclude visibility
    kpts_2d_visibility = kpts_2d[:, :, -1:]  # Extract visibility

    # Initialize components to concatenate
    components = [kpts_2d_coords]

    if 'BL' in perspective_method:
        components.append(aux_params['bone_lengths'][..., None])
        print("BL")
    if 'OD' in perspective_method:
        components.append(aux_params['ordinal_depths'][..., None])
        print("OD")
    if 'RT' in perspective_method:
        components.append(aux_params['camera_params_array'][..., None])
        print("RT")

    components.append(kpts_2d_visibility)  # Append visibility last

    # Concatenate all components along the last axis
    return np.concatenate(components, axis=-1)

def concatenate_kpts_2d_with_auxiliary_casp(kpts_2d, aux_params, casp_mode='v0'):
    """
    Concatenate 2D keypoints with CASP descriptor.
    For spatial mode, extract [x, y, c, d_pt, dmin, x_dmin, y_dmin, dmax, x_dmax, y_dmax]
    from full CASP (19 features):
        x
        y
      c = predicted_keypoints_score
      d_pt 
      dmin = Q10 (0)
      x_dmin = Q10_xy_x (1)
      y_dmin = Q10_xy_y (2)
      dmax = Q90 (12)
      x_dmax = Q90_xy_x (13)
      y_dmax = Q90_xy_y (14)
    """
    kpts_2d_coords = kpts_2d[:, :, :2]
    kpts_2d_visibility = kpts_2d[:, :, 2:]
    casp_summary = aux_params.get('casp_descriptors', None)
    conf = aux_params.get('predicted_keypoints_score', None)
    d_pt = aux_params.get('predicted_da_depth', None)
    if casp_summary is None:
        print("No casp data")
        casp_features = np.zeros((kpts_2d.shape[0], 17, 10), dtype=np.float32)
        components = [kpts_2d_coords, casp_features, kpts_2d_visibility]
    else:
        if casp_mode == 'spatial' and casp_summary.shape[-1] == 19:
            casp_features = np.zeros((casp_summary.shape[0], casp_summary.shape[1], 8), dtype=np.float32)
            casp_features[..., 0] = conf                  # confidence
            casp_features[..., 1] = d_pt
            casp_features[..., 2] = casp_summary[..., 0]  # dmin
            casp_features[..., 3] = casp_summary[..., 1]  # x_dmin
            casp_features[..., 4] = casp_summary[..., 2]  # y_dmin
            casp_features[..., 5] = casp_summary[..., 12] # dmax
            casp_features[..., 6] = casp_summary[..., 13] # x_dmax
            casp_features[..., 7] = casp_summary[..., 14] # y_dmax
            components = [kpts_2d_coords, casp_features, kpts_2d_visibility]
        # elif casp_mode == 'v0':
        #     casp_features = casp_summary[..., [0,1,2,3,4,5]]
        #     kpts_2d_coords = kpts_2d[:, :, :2]
        #     components = [kpts_2d_coords, casp_features, kpts_2d_visibility]
        elif casp_mode == 'v0':
            conf_channel   = casp_summary[..., 2:3]   # confidence
            Q10_channel    = casp_summary[..., 4:5]   # Q10
            Q90_channel    = casp_summary[..., 8:9]   # Q90
            exact_depth    = casp_summary[..., 9:10]  # depth_point
            kpts_2d_coords = kpts_2d[:, :, :2]        # x, y
            kpts_2d_visibility = kpts_2d[:, :, 2:]    # visibility
            # Stack: [x, y, conf, depth_point, Q10, Q90, visibility]
            components = [
                kpts_2d_coords,
                conf_channel,
                exact_depth,
                Q10_channel,
                Q90_channel,
                kpts_2d_visibility
            ]
        else:
            casp_features = casp_summary
            kpts_2d_coords = kpts_2d[:, :, :2]
            components = [kpts_2d_coords, casp_features, kpts_2d_visibility]
    return np.concatenate(components, axis=-1)

# def concatenate_kpts_2d_with_auxiliary_casp(kpts_2d, aux_params, casp_mode='v0'):
#     """
#     Concatenate 2D keypoints with CASP descriptor.
#     For both 'spatial' and 'v0' modes, always concatenate [x, y] from original keypoints,
#     then the selected CASP features, then visibility.
#     For spatial mode with 19 features, extract:
#       [x, y] (from kpts_2d), c=1.0, d_pt, dmin, x_dmin, y_dmin, dmax, x_dmax, y_dmax (from CASP), visibility
#     For v0 mode, use [x, y] (from kpts_2d), [0,1,2,3,4,5] from CASP, visibility
#     """
#     kpts_2d_coords = kpts_2d[:, :, :2]
#     kpts_2d_visibility = kpts_2d[:, :, 2:]
#     casp_summary = aux_params.get('casp_descriptors', None)
#     if casp_summary is None:
#         if casp_mode == 'spatial':
#             casp_features = np.zeros((kpts_2d.shape[0], 17, 8), dtype=np.float32)
#             c_channel = np.ones((kpts_2d.shape[0], 17, 1), dtype=np.float32)
#             components = [kpts_2d_coords, c_channel, casp_features, kpts_2d_visibility]
#         elif casp_mode == 'v0':
#             casp_features = np.zeros((kpts_2d.shape[0], 17, 6), dtype=np.float32)
#             components = [kpts_2d_coords, casp_features, kpts_2d_visibility]
#         else:
#             casp_features = np.zeros((kpts_2d.shape[0], 17, 19), dtype=np.float32)
#             components = [kpts_2d_coords, casp_features, kpts_2d_visibility]
#     else:
#         if casp_mode == 'spatial' and casp_summary.shape[-1] == 19:
#             # Extract [c, d_pt, dmin, x_dmin, y_dmin, dmax, x_dmax, y_dmax] from CASP
#             c_channel = np.ones((casp_summary.shape[0], casp_summary.shape[1], 1), dtype=np.float32)
#             casp_features = np.zeros((casp_summary.shape[0], casp_summary.shape[1], 8), dtype=np.float32)
#             casp_features[..., 0] = casp_summary[..., 15]  # d_pt
#             casp_features[..., 1] = casp_summary[..., 0]   # dmin
#             casp_features[..., 2] = casp_summary[..., 1]   # x_dmin
#             casp_features[..., 3] = casp_summary[..., 2]   # y_dmin
#              casp_features[..., 4] = casp_summary[..., 12]  # dmax
#              casp_features[..., 5] = casp_summary[..., 13]  # x_dmax
#              casp_features[..., 6] = casp_summary[..., 14]  # y_dmax
#              casp_features[..., 7] = casp_summary[..., 16]  # x (central_xy_x)
#              # Note: y (central_xy_y) is already in kpts_2d_coords
#              components = [kpts_2d_coords, c_channel, casp_features, kpts_2d_visibility]
#         elif casp_mode == 'v0':
#             casp_features = casp_summary[..., [0,1,2,3,4,5]]
#             components = [kpts_2d_coords, casp_features, kpts_2d_visibility]
#         else:
#             casp_features = casp_summary
#             components = [kpts_2d_coords, casp_features, kpts_2d_visibility]
#     return np.concatenate(components, axis=-1)

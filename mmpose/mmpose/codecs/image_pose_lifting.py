# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import numpy as np

from mmpose.registry import KEYPOINT_CODECS
from .base import BaseKeypointCodec
import ipdb

@KEYPOINT_CODECS.register_module()
class ImagePoseLifting(BaseKeypointCodec):
    r"""Generate keypoint coordinates for pose lifter.

    Note:

        - instance number: N
        - keypoint number: K
        - keypoint dimension: D
        - pose-lifitng target dimension: C

    Args:
        num_keypoints (int): The number of keypoints in the dataset.
        root_index (Union[int, List]): Root keypoint index in the pose.
        remove_root (bool): If true, remove the root keypoint from the pose.
            Default: ``False``.
        save_index (bool): If true, store the root position separated from the
            original pose. Default: ``False``.
        reshape_keypoints (bool): If true, reshape the keypoints into shape
            (-1, N). Default: ``True``.
        concat_vis (bool): If true, concat the visibility item of keypoints.
            Default: ``False``.
        keypoints_mean (np.ndarray, optional): Mean values of keypoints
            coordinates in shape (K, D).
        keypoints_std (np.ndarray, optional): Std values of keypoints
            coordinates in shape (K, D).
        target_mean (np.ndarray, optional): Mean values of pose-lifitng target
            coordinates in shape (K, C).
        target_std (np.ndarray, optional): Std values of pose-lifitng target
            coordinates in shape (K, C).
    """


    auxiliary_encode_keys = {
        'lifting_target', 'lifting_target_visible', 'camera_param'
    }


    instance_mapping_table = dict(
        lifting_target='lifting_target',
        lifting_target_visible='lifting_target_visible',
    )
    label_mapping_table = dict(
        trajectory_weights='trajectory_weights',
        lifting_target_label='lifting_target_label',
        lifting_target_weight='lifting_target_weight',
        confidence='confidence')  # Add confidence to label mapping

    def __init__(self,
                 num_keypoints: int,
                 root_index: Union[int, List] = 0,
                 remove_root: bool = False,
                 save_index: bool = False,
                 reshape_keypoints: bool = True,
                 concat_vis: bool = False,
                 keypoints_mean: Optional[np.ndarray] = None,
                 keypoints_std: Optional[np.ndarray] = None,
                 target_mean: Optional[np.ndarray] = None,
                 target_std: Optional[np.ndarray] = None,
                 additional_encode_keys: Optional[List[str]] = None,
                 dropout_scaling_fix: bool = False,
                 use_casp: bool = False,
                 use_casp_spatial: bool = False):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.dropout_scaling_fix = dropout_scaling_fix
        if isinstance(root_index, int):
            root_index = [root_index]
        self.root_index = root_index
        self.remove_root = remove_root
        self.save_index = save_index
        self.reshape_keypoints = reshape_keypoints
        self.concat_vis = concat_vis
        self.use_casp = use_casp
        self.use_casp_spatial = use_casp_spatial
        if keypoints_mean is not None:
            assert keypoints_std is not None, 'keypoints_std is None'
            keypoints_mean = np.array(
                keypoints_mean,
                dtype=np.float32).reshape(1, num_keypoints, -1)
            keypoints_std = np.array(
                keypoints_std, dtype=np.float32).reshape(1, num_keypoints, -1)

            assert keypoints_mean.shape == keypoints_std.shape, (
                f'keypoints_mean.shape {keypoints_mean.shape} != '
                f'keypoints_std.shape {keypoints_std.shape}')
        if target_mean is not None:
            assert target_std is not None, 'target_std is None'
            target_dim = num_keypoints - 1 if remove_root else num_keypoints
            target_mean = np.array(
                target_mean, dtype=np.float32).reshape(1, target_dim, -1)
            target_std = np.array(
                target_std, dtype=np.float32).reshape(1, target_dim, -1)

            assert target_mean.shape == target_std.shape, (
                f'target_mean.shape {target_mean.shape} != '
                f'target_std.shape {target_std.shape}')
        self.keypoints_mean = keypoints_mean
        self.keypoints_std = keypoints_std
        self.target_mean = target_mean
        self.target_std = target_std

        if additional_encode_keys is not None:
            self.auxiliary_encode_keys.update(additional_encode_keys)

    def encode(self,
            keypoints: np.ndarray,
            keypoints_visible: Optional[np.ndarray] = None,
            lifting_target: Optional[np.ndarray] = None,
            lifting_target_visible: Optional[np.ndarray] = None,
            camera_param: Optional[dict] = None) -> dict:

        """Encoding keypoints from input image space to normalized space.

        Args:
            keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D).
            keypoints_visible (np.ndarray, optional): Keypoint visibilities in
                shape (N, K).
            lifting_target (np.ndarray, optional): 3d target coordinate in
                shape (T, K, C).
            lifting_target_visible (np.ndarray, optional): Target coordinate in
                shape (T, K, ).

        Returns:
            encoded (dict): Contains the following items:

                - keypoint_labels (np.ndarray): The processed keypoints in
                  shape like (N, K, D) or (K * D, N).
                - keypoint_labels_visible (np.ndarray): The processed
                  keypoints' weights in shape (N, K, ) or (N-1, K, ).
                - lifting_target_label: The processed target coordinate in
                  shape (K, C) or (K-1, C).
                - lifting_target_weight (np.ndarray): The target weights in
                  shape (K, ) or (K-1, ).
                - trajectory_weights (np.ndarray): The trajectory weights in
                  shape (K, ).
                - target_root (np.ndarray): The root coordinate of target in
                  shape (C, ).

                In addition, there are some optional items it may contain:

                - target_root (np.ndarray): The root coordinate of target in
                  shape (C, ). Exists if ``zero_center`` is ``True``.
                - target_root_removed (bool): Indicate whether the root of
                  pose-lifitng target is removed. Exists if
                  ``remove_root`` is ``True``.
                - target_root_index (int): An integer indicating the index of
                  root. Exists if ``remove_root`` and ``save_index``
                  are ``True``.
        """
        if keypoints_visible is None:
            keypoints_visible = np.ones(keypoints.shape[:2], dtype=np.float32)

        if lifting_target is None:
            lifting_target = [keypoints[0]]

        # set initial value for `lifting_target_weight`
        # and `trajectory_weights`
        if lifting_target_visible is None:
            lifting_target_visible = np.ones(
                lifting_target.shape[:-1], dtype=np.float32)
            lifting_target_weight = lifting_target_visible
            trajectory_weights = (1 / lifting_target[:, 2])
        else:
            valid = lifting_target_visible > 0.5
            lifting_target_weight = np.where(valid, 1., 0.).astype(np.float32)
            trajectory_weights = lifting_target_weight

        encoded = dict()

        # Zero-center the target pose around a given root keypoint
        assert (lifting_target.ndim >= 2 and
                lifting_target.shape[-2] > max(self.root_index)), \
            f'Got invalid joint shape {lifting_target.shape}'

        root = np.mean(
            lifting_target[..., self.root_index, :], axis=-2, dtype=np.float32)
        lifting_target_label = lifting_target - root[np.newaxis, ...]

        # ─── Extract confidence BEFORE normalization (for inverse confidence loss) ───
        confidence = None
        if keypoints.shape[-1] >= 3:  # Has confidence channel
            # Confidence is always at channel 2 for all formats (xycd, casp_v0, casp_spatial)
            confidence = keypoints[..., 2].copy()

        if self.remove_root and len(self.root_index) == 1:
            root_index = self.root_index[0]
            lifting_target_label = np.delete(
                lifting_target_label, root_index, axis=-2)
            lifting_target_visible = np.delete(
                lifting_target_visible, root_index, axis=-2)
            assert lifting_target_weight.ndim in {
                2, 3
            }, (f'lifting_target_weight.ndim {lifting_target_weight.ndim} '
                'is not in {2, 3}')

            axis_to_remove = -2 if lifting_target_weight.ndim == 3 else -1
            lifting_target_weight = np.delete(
                lifting_target_weight, root_index, axis=axis_to_remove)
            
            # Also remove root from confidence if it exists
            if confidence is not None:
                confidence = np.delete(confidence, root_index, axis=-1)
            
            # Add a flag to avoid latter transforms that rely on the root
            # joint or the original joint index
            encoded['target_root_removed'] = True

            # Save the root index which is necessary to restore the global pose
            if self.save_index:
                encoded['target_root_index'] = root_index

        # Normalize the 2D keypoint coordinate with mean and std
        keypoint_labels = keypoints.copy()

        assert 'w' in camera_param and 'h' in camera_param, \
            'camera_param must contain width (w) and height (h)'

        center = np.array([0.5 * camera_param['w'], 0.5 * camera_param['h']], dtype=np.float32)
        scale = np.array(0.5 * camera_param['w'], dtype=np.float32)

        keypoint_labels[..., :2] = (keypoint_labels[..., :2] - center) / scale

        # ─── Conditional normalization ─────────────────────────────────────────────
        # if self.dropout_scaling_fix:
        #     # 3-channel case: [x, y, c]
        #     if keypoint_labels.shape[-1] == 3:
        #         keypoint_labels[..., 2] = 2 * keypoint_labels[..., 2] - 1
        #     # 4-channel case: [x, y, c, d]
        #     elif keypoint_labels.shape[-1] >= 4:
        #         keypoint_labels[..., -2] = 2 * keypoint_labels[..., -2] - 1
        #         depth = keypoint_labels[..., -1]
        #         depth_rel = depth - depth.min(axis=-1, keepdims=True)
        #         depth_clp = np.clip(depth_rel, 0.0, 2.0)
        #         keypoint_labels[..., -1] = 2 * (depth_clp / 2.0) - 1
        # ipdb.set_trace()
        if self.use_casp and keypoint_labels.shape[-1] == 6:
            # CASP 6D format: [x, y, conf, depth_point, Q10, Q90]
            keypoint_labels[..., 4] -= keypoint_labels[..., 3]  # Q10 - depth_point
            keypoint_labels[..., 5] -= keypoint_labels[..., 3]  # Q90 - depth_point
            keypoint_labels[..., 3] -= keypoint_labels[..., 3].min(axis=-1, keepdims=True)
            for idx in [4, 5]:
                keypoint_labels[..., idx] = np.clip(keypoint_labels[..., idx], -1.0, 1.0)
        elif self.use_casp_spatial and keypoint_labels.shape[-1] == 10:
            # CASP spatial format: [x, y, c, d_pt, dmin, x_dmin, y_dmin, dmax, x_dmax, y_dmax]
            w, h = camera_param['w'], camera_param['h']
            keypoint_labels[..., 5] = keypoint_labels[..., 5] / w * 2 - 1  # x_dmin
            keypoint_labels[..., 6] = keypoint_labels[..., 6] / h * 2 - h / w  # y_dmin
            keypoint_labels[..., 8] = keypoint_labels[..., 8] / w * 2 - 1  # x_dmax
            keypoint_labels[..., 9] = keypoint_labels[..., 9] / h * 2 - h / w  # y_dmax
            keypoint_labels[..., 4] -= keypoint_labels[..., 3]  # dmin - d_pt
            keypoint_labels[..., 7] -= keypoint_labels[..., 3]  # dmax - d_pt
            keypoint_labels[..., 3] -= keypoint_labels[..., 3].min(axis=-1, keepdims=True)
            for idx in [4, 7]:
                keypoint_labels[..., idx] = np.clip(keypoint_labels[..., idx], -1.0, 1.0)
        elif keypoint_labels.shape[-1] >= 4:
            keypoint_labels[..., -1] -= keypoint_labels[..., -1].min(axis=-1, keepdims=True)


        # Generate reshaped keypoint coordinates
        assert keypoint_labels.ndim in {
            2, 3
        }, (f'keypoint_labels.ndim {keypoint_labels.ndim} is not in {2, 3}')
        if keypoint_labels.ndim == 2:
            keypoint_labels = keypoint_labels[None, ...]

        if self.concat_vis:
            keypoints_visible_ = keypoints_visible
            if keypoints_visible.ndim == 2:
                keypoints_visible_ = keypoints_visible[..., None]
            keypoint_labels = np.concatenate(
                (keypoint_labels, keypoints_visible_), axis=2)

        if self.reshape_keypoints:
            N = keypoint_labels.shape[0]
            keypoint_labels = keypoint_labels.transpose(1, 2, 0).reshape(-1, N)

        encoded['keypoint_labels'] = keypoint_labels
        encoded['keypoint_labels_visible'] = keypoints_visible
        encoded['lifting_target_label'] = lifting_target_label
        encoded['lifting_target_weight'] = lifting_target_weight
        encoded['trajectory_weights'] = trajectory_weights
        encoded['target_root'] = root
        
        # Add confidence if available (for inverse confidence loss)
        if confidence is not None:
            encoded['confidence'] = confidence

        return encoded

    def decode(self,
               encoded: np.ndarray,
               target_root: Optional[np.ndarray] = None
               ) -> Tuple[np.ndarray, np.ndarray]:
        """Decode keypoint coordinates from normalized space to input image
        space.

        Args:
            encoded (np.ndarray): Coordinates in shape (N, K, C).
            target_root (np.ndarray, optional): The target root coordinate.
                Default: ``None``.

        Returns:
            keypoints (np.ndarray): Decoded coordinates in shape (N, K, C).
            scores (np.ndarray): The keypoint scores in shape (N, K).
        """
        keypoints = encoded.copy()

        if self.target_mean is not None and self.target_std is not None:
            assert self.target_mean.shape == keypoints.shape, (
                f'self.target_mean.shape {self.target_mean.shape} '
                f'!= keypoints.shape {keypoints.shape}')
            keypoints = keypoints * self.target_std + self.target_mean

        if target_root is not None and target_root.size > 0:
            keypoints = keypoints + target_root
            if self.remove_root and len(self.root_index) == 1:
                keypoints = np.insert(
                    keypoints, self.root_index, target_root, axis=1)
        scores = np.ones(keypoints.shape[:-1], dtype=np.float32)

        return keypoints, scores

# poseformer_regression_head.py

from collections import OrderedDict
from typing import Dict, Optional, List    # ← add List here

import torch
import torch.nn as nn
from torch import Tensor

from mmpose.evaluation.functional import keypoint_mpjpe
from mmpose.registry import MODELS, KEYPOINT_CODECS
# from mmpose.structures import InstanceData
from mmpose.utils.typing import Predictions, OptSampleList, ConfigType
from mmpose.utils.tensor_utils import to_numpy
from mmpose.models.heads.base_head import BaseHead
import ipdb
import numpy as np

@MODELS.register_module()
class PoseRegressionHead(BaseHead):
    """Regression head for PoseFormer, with optional MPJPE + ordinal regularizer."""

    def __init__(self,
                 in_channels: int = 544,
                 num_joints: int = 17,
                 out_channels: int = 3,
                 loss: Dict = dict(type='MPJPELoss'),
                 ordinal_loss: Optional[Dict] = None,
                 decoder: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        # Use BaseHead's init to register init_cfg
        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.num_joints = num_joints
        self.out_channels = out_channels

        # Build the loss via MODELS so MPJPEVelocityJointLoss is found
        self.loss_module = MODELS.build(loss)

        # Optional ordinal pairwise loss (built only if provided)
        self.ordinal_loss = MODELS.build(ordinal_loss) if ordinal_loss is not None else None

        # Optional decoder (if you need to invert normalization)
        if decoder is not None:
            self.decoder = KEYPOINT_CODECS.build(decoder)
        else:
            self.decoder = None

        # Pre-logits MLP (identity, but kept for symmetry with MotionHead)
        self.pre_logits = nn.Identity()

        # Final regression layer: (in_channels → num_joints*out_channels)
        self.fc = nn.Linear(in_channels, self.num_joints * self.out_channels)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """
        Args
            feats: (B, F, 544) from PoseFormer
                   or (B, F, K, C) if a neck adds per-joint tokens.
        Returns
            (B, 17, 3) centre-frame 3-D joint coords.
        """
        if feats.dim() == 4:                  # (B, F, K, C)  →  flatten joints
            B, F, K, C = feats.shape
            feats = feats.view(B, F, -1)      # (B, F, K*C)
        elif feats.dim() == 3:                # (B, F, 544)
            B, F, C = feats.shape
        else:
            raise ValueError('Expected 3- or 4-D feats tensor.')

        centre = feats[:, F // 2]             # (B, 544)
        coords  = self.fc(centre).view(
            -1, self.num_joints, self.out_channels)  # (B, 17, 3)
        return coords


    # def forward(self, feats: Tensor) -> Tensor:
    #     # feats: (B, F, K, C)
    #     if feats.dim() == 3:
    #         feats = feats[:, None]
    #     B, F, K, C = feats.shape
    #     # center = feats[:, F // 2]               # (B, K, C)
    #     # x = self.pre_logits(center)             # (B, K, C) or (B,K,embed)
    #     # coords = self.fc(x)                     # (B, K, 3)
    #     # feats: (B, F, 544) from backbone
    #     center = feats[:, F // 2]                     # (B, 544)
    #     ipdb.set_trace()
    #     coords  = self.fc(center).view(               # → (B, K, 3)
    #                 -1, self.num_joints, self.out_channels)
    #     return coords



    def loss(self,
             feats: Tensor,
             batch_data_samples: List,
             train_cfg: Optional[dict] = None) -> dict:
        """
        Args:
            feats: (B, F, K, C) or (B, F, C)
            batch_data_samples: list of DataSample with .gt_instance_labels
        Returns:
            dict with keys 'loss_pose3d' and 'mpjpe'
        """
        # 1) Forward to get center‐frame predictions: (B, K, 3)
        preds = self.forward(feats)

        # 2) Gather full-sequence GT and weights: both (B, F, K, …)
        full_gt = torch.stack([
            ds.gt_instance_labels.lifting_target_label
            for ds in batch_data_samples
        ])  # shape (B, F, K, 3)
        full_wt = torch.stack([
            ds.gt_instance_labels.lifting_target_weight
            for ds in batch_data_samples
        ])  # shape (B, F, K)
        full_wt = full_wt.unsqueeze(-1)  # → (B, F, K, 1)

        # 3) Pick the middle frame
        F = full_gt.shape[1]
        gt     = full_gt[:, F // 2]      # (B, K, 3)
        weight = full_wt[:, F // 2]      # (B, K, 1)

        # 4) Primary regression loss (MPJPE / variant)
        base_loss = self.loss_module(preds, gt, weight)
        losses = dict(loss_pose3d=base_loss)

        # 5) Optional: ordinal pairwise regularizer on Z
        if self.ordinal_loss is not None:
            ord_loss = self.ordinal_loss(preds, gt, weight)  # (uses (B,K,3) and visibility)
            # Sum into main loss so checkpoints reflect total objective
            losses['loss_pose3d'] = base_loss + ord_loss
            losses['loss_ordinal'] = ord_loss

        # 6) Log MPJPE for monitoring (unchanged)
        from mmpose.evaluation.functional import keypoint_mpjpe
        from mmpose.utils.tensor_utils import to_numpy

        mpjpe_val = keypoint_mpjpe(
            pred = to_numpy(preds),
            gt   = to_numpy(gt),
            mask = to_numpy(weight[..., 0]) > 0
        )
        losses['mpjpe'] = torch.tensor(mpjpe_val, device=gt.device)

        return losses

    def predict(self,
                feats: Tensor,
                batch_data_samples: OptSampleList,
                test_cfg: ConfigType = {}) -> Predictions:
        """Predict and decode 3D poses for the centre frame of each clip.

        Returns:
            preds: Tuple[np.ndarray, np.ndarray] containing
                - keypoints: (N, K, 3) in pixel + meter space
                - scores:    (N, K)
        """
        # ipdb.set_trace()
        # 1) Centre-frame regression
        if test_cfg.get('flip_test', False):
            assert isinstance(feats, list) and len(feats) == 2, \
                'TTA expects feats=[orig, flipped]'
            flip_indices = batch_data_samples[0].metainfo['flip_indices']
            orig, flip = feats
            coords_orig = self.forward(orig)    # (B, K, 3)
            coords_flip = self.forward(flip)    # (B, K, 3)
            coords_flip[..., 0] *= -1
            coords_flip = coords_flip[:, flip_indices]
            batch_coords = 0.5 * (coords_orig + coords_flip)
        else:
            batch_coords = self.forward(feats)  # (B, K, 3)

        # 2) Gather image sizes
        camera_param = batch_data_samples[0].metainfo.get('camera_param', None)
        if camera_param is not None:
            w = np.array([d.metainfo['camera_param']['w']
                          for d in batch_data_samples],
                         dtype=np.float32)
            h = np.array([d.metainfo['camera_param']['h']
                          for d in batch_data_samples],
                         dtype=np.float32)
        else:
            w = np.array([d.metainfo['camera_param']['w']
                          for d in batch_data_samples],
                         dtype=np.float32)
            h = np.array([d.metainfo['camera_param']['h']
                          for d in batch_data_samples],
                         dtype=np.float32)



        # 3) Decode normalized coords back to pixel+meter space
        coords_np = batch_coords.detach().cpu().numpy()

        # ipdb.set_trace()
        kps, scores = self.decoder.decode(coords_np)

        # ipdb.set_trace()
        # from mmengine.structures import InstanceData
        # preds = [InstanceData(keypoints=torch.from_numpy(k),
        #                     keypoint_scores=torch.from_numpy(s))
        #         for k, s in zip(kps, scores)]
        # return preds, None

    
        # preds = self.decode(coords_np, w=w, h=h)
        # return preds
        from mmengine.structures import InstanceData
        pred_instances = []
        for kp, sc in zip(kps, scores):
            inst = InstanceData()
            inst.keypoints       = torch.from_numpy(kp)[None, ...]   #  (1, 17, 3)
            inst.keypoint_scores = torch.from_numpy(sc)[None, ...]   #  (1, 17)
            pred_instances.append(inst)

        return pred_instances, None

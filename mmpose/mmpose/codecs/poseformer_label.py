# poseformer_label.py

from copy import deepcopy
import numpy as np
from typing import Optional, Union, Dict, List, Tuple
from mmpose.registry import KEYPOINT_CODECS
from .base import BaseKeypointCodec
import ipdb

@KEYPOINT_CODECS.register_module()
class PoseFormerLabel(BaseKeypointCodec):
    """Minimal-diff MotionBERTLabel → PoseFormerLabel:
       - no camera_to_image_coord
       - no factor
       - 2D normalised by image_size
       - 3D root-relative in meters
       - same API & outputs as MotionBERTLabel
    """

    auxiliary_encode_keys = {
        'lifting_target', 'lifting_target_visible', 'camera_param', 'factor'
    }

    instance_mapping_table = dict(
        lifting_target='lifting_target',
        lifting_target_visible='lifting_target_visible',
    )
    label_mapping_table = dict(
        trajectory_weights='trajectory_weights',
        lifting_target_label='lifting_target_label',
        lifting_target_weight='lifting_target_weight')

    def __init__(self,
                 num_keypoints: int,
                 root_index: Union[int, List[int]] = 0,
                 remove_root: bool = False,
                 save_index: bool = False,
                 concat_vis: bool = False,
                 rootrel: bool = False,
                 mode: str = 'test',
                 depth_jitter_sigma: float = 0.0,
                 concatenate_root_depth: bool = False,
                 use_casp: bool = False,
                 use_casp_spatial: bool = False,
                 use_xyd: bool = False,
                 # NEW: Feature map normalization parameters
                 normalize_feats: bool = True,
                 use_img_feats: bool = False,
                 use_depth_feats: bool = False,
                 img_feat_dim: int = 1024,
                 depth_feat_dim: int = 256,
                 # Global feature statistics (pre-computed from entire dataset)
                 global_rtm_stats: Optional[Dict[str, np.ndarray]] = None,
                 global_dav2_stats: Optional[Dict[str, np.ndarray]] = None):
        super().__init__()
        # === preserve MotionBERTLabel args ===
        self.num_keypoints = num_keypoints
        self.root_index = [root_index] if isinstance(root_index, int) else root_index
        self.remove_root = remove_root
        self.save_index = save_index
        self.concat_vis = concat_vis
        self.rootrel = rootrel
        assert mode in {'train', 'test'}
        self.mode = mode
        self.depth_jitter_sigma = depth_jitter_sigma
        self.concatenate_root_depth = concatenate_root_depth
        self.use_casp = use_casp
        self.use_casp_spatial = use_casp_spatial
        self.use_xyd = use_xyd
        
        # NEW: Feature map parameters
        self.normalize_feats = normalize_feats
        self.use_img_feats = use_img_feats
        self.use_depth_feats = use_depth_feats
        self.img_feat_dim = img_feat_dim
        self.depth_feat_dim = depth_feat_dim
        
        # Store global statistics (if provided, use these instead of per-batch stats)
        self.global_rtm_stats = global_rtm_stats
        self.global_dav2_stats = global_dav2_stats

    def encode(self,
               keypoints: np.ndarray,
               keypoints_visible: Optional[np.ndarray] = None,
               lifting_target: Optional[np.ndarray] = None,
               lifting_target_visible: Optional[np.ndarray] = None,
               camera_param: Optional[dict] = None,
               factor: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Args:
            keypoints: (T, K, C) or (K, C) array of 2D+depth (if provided)
            keypoints_visible: (T, K) visibility flags
            lifting_target: (T, K, 3) ground-truth 3D
            lifting_target_visible: (T, K) flags for 3D validity
            camera_param: dict possibly containing
               - 'intrinsics_wo_distortion': {'c':..., 'f':...}
               - 'w': image width, 'h': image height
            factor: unused for PoseFormer
        Returns:
            dict with keys:
              'keypoint_labels': (T, K, C′) normalized 2D+depth
              'keypoint_labels_visible': (T, K)
              'lifting_target_label': (T, K, 3) root-relative 3D
              'lifting_target_weight': (T, K)
              'lifting_target': same as label
              'lifting_target_visible': (T, K)
        """
        # 1) Handle shapes
        kp = keypoints.copy()
        if kp.ndim == 2:
            kp = kp[None, ...]
        T, K, C = kp.shape

        # 2) Visibility → weights
        kv = (np.ones((T, K), np.float32)
              if keypoints_visible is None else keypoints_visible.copy())
        ltv = (np.ones((T, K), np.float32)
               if lifting_target_visible is None else lifting_target_visible.copy())
        lt_weight = (ltv > 0.5).astype(np.float32)

        # 3) Ground-truth 3D
        assert lifting_target is not None, 'lifting_target required'
        lt = lifting_target.copy()
        # lt = lifting_target.copy() / 1000.0  # Convert mm → meters


        # 4) Depth jitter (train-only)
        if (self.mode == 'train'
            and self.depth_jitter_sigma > 0
            and C >= 4):
            EXT = [10, 12, 13, 15, 16, 2, 3, 5, 6]
            noise_seq = np.random.uniform(
                -self.depth_jitter_sigma,
                self.depth_jitter_sigma,
                size=(1, len(EXT))
            )
            noise = np.tile(noise_seq, (T, 1))
            kp[..., EXT, -1] += noise
            lt[..., EXT, 2] += noise

        # 5) Prepare camera parameters
        cam = dict() if camera_param is None else deepcopy(camera_param)
        if 'intrinsics_wo_distortion' in cam:
            cam['c'] = np.array(cam['intrinsics_wo_distortion']['c'])
            cam['f'] = np.array(cam['intrinsics_wo_distortion']['f'])
        # fallback to image_size if w/h not in cam
        default_w, default_h = (None, None)
        # if self.image_size is not None:
        #     default_w, default_h = self.image_size
        cam['w'] = cam.get('w', default_w)
        cam['h'] = cam.get('h', default_h)
        assert cam['w'] is not None and cam['h'] is not None, (
            'Image size or camera_param must provide w,h')
        w, h = cam['w'], cam['h']
        # if w ==None:
        #     ipdb.set_trace()

        # 6) Normalize 2D → [-1,1]
        # kp[..., 0] = kp[..., 0] / w * 2.0 - 1.0
        # kp[..., 1] = kp[..., 1] / h * 2.0 - (h / w)
        # Normalize 2D → [-1,1]
        # kp[..., 0] = kp[..., 0] / w * 2.0 - 1.0
        # kp[..., 1] = kp[..., 1] / h * 2.0 - 1.0
        # — replace your existing x/y normalisation with this:
        # max_wh = w        # note: use width to preserve aspect ratio
        # kp[..., 0] = kp[..., 0] / max_wh * 2.0 - 1.0
        # kp[..., 1] = kp[..., 1] / max_wh * 2.0 - (h / w)

        # 6) Normalise 2-D, confidence and depth
        max_wh = w                      # keep current x/y rule
        kp[..., 0] = kp[..., 0] / max_wh * 2.0 - 1.0
        kp[..., 1] = kp[..., 1] / max_wh * 2.0 - (h / w)

        # ----------------  c and d to [-1, 1]  ----------------
        if kp.shape[-1] >= 3:                         # confidence
            kp[..., 2] = kp[..., 2] * 2.0 - 1.0

        # ipdb.set_trace()
        # CASP-specific normalization
        if hasattr(self, 'use_casp') and self.use_casp and kp.shape[-1] == 6:
            # CASP 6D format: [x, y, conf, depth_point, Q10, Q90]
            kp[..., 4] -= kp[..., 3]  # Q10 - depth_point
            kp[..., 5] -= kp[..., 3]  # Q90 - depth_point
            kp[..., 3] -= kp[..., 3].min(axis=-1, keepdims=True)
            for idx in [4, 5]:
                kp[..., idx] = np.clip(kp[..., idx], -1.0, 1.0)
        elif hasattr(self, 'use_casp_spatial') and self.use_casp_spatial and kp.shape[-1] == 10:
            # CASP spatial format: [x, y, c, d_pt, dmin, x_dmin, y_dmin, dmax, x_dmax, y_dmax]
            kp[..., 5] = kp[..., 5] / w * 2 - 1  # x_dmin
            kp[..., 6] = kp[..., 6] / h * 2 - h / w  # y_dmin
            kp[..., 8] = kp[..., 8] / w * 2 - 1  # x_dmax
            kp[..., 9] = kp[..., 9] / h * 2 - h / w  # y_dmax
            kp[..., 4] -= kp[..., 3]  # dmin - d_pt
            kp[..., 7] -= kp[..., 3]  # dmax - d_pt
            kp[..., 3] -= kp[..., 3].min(axis=-1, keepdims=True)
            for idx in [4, 7]:
                kp[..., idx] = np.clip(kp[..., idx], -1.0, 1.0)
        elif hasattr(self, 'use_xyd') and self.use_xyd and kp.shape[-1] == 3:
            # XYD format: [x, y, depth_point]
            # Make depth_point root-relative (subtract min across joints)
            kp[..., 2] -= kp[..., 2].min(axis=-1, keepdims=True)
        elif kp.shape[-1] == 3:
            # XYC format: [x, y, confidence]
            # No additional normalization needed - confidence is already in [0, 1]
            pass
        elif kp.shape[-1] >= 4:
            root_rel  = kp[..., 3] - kp[..., 3].min(axis=-1, keepdims=True)
            depth_clp = np.clip(root_rel, 0.0, 2.0)   # same 0-2 m window
            kp[..., 3] = 2.0 * (depth_clp / 2.0) - 1.0  # ← new line

        # 6.5) Normalize feature maps (RTM and/or DAV2) if enabled
        # Feature maps are appended after base channels (xy/xyc/xycd/etc)
        # Dataset concatenation order: [base] + [RTM 1024D if enabled] + [DAV2 256D if enabled]
        # Layout: [base_channels] [RTM if enabled] [DAV2 if enabled]
        if self.normalize_feats and (self.use_img_feats or self.use_depth_feats):
            # Determine base channel count (xy/xyc/xycd/casp/etc)
            # Work backwards: total - feature dims = base
            base_channels = kp.shape[-1]
            if self.use_img_feats:
                base_channels -= self.img_feat_dim
            if self.use_depth_feats:
                base_channels -= self.depth_feat_dim
            
            # Debug: Verify channel layout when both features are enabled
            if self.use_img_feats and self.use_depth_feats:
                expected_total = base_channels + self.img_feat_dim + self.depth_feat_dim
                assert kp.shape[-1] == expected_total, (
                    f"Channel mismatch! Expected {expected_total} "
                    f"(base={base_channels} + RTM={self.img_feat_dim} + DAV2={self.depth_feat_dim}), "
                    f"got {kp.shape[-1]}. "
                    f"Channel layout should be: [0:{base_channels}]=base, "
                    f"[{base_channels}:{base_channels + self.img_feat_dim}]=RTM, "
                    f"[{base_channels + self.img_feat_dim}:{expected_total}]=DAV2"
                )
            
            # Normalize RTM features (1024D) to [-1, 1] using global percentile-based min-max
            if self.use_img_feats:
                # RTM comes immediately after base channels
                feat_start = base_channels
                feat_end = feat_start + self.img_feat_dim
                rtm_feats = kp[..., feat_start:feat_end]
                
                # Global percentile-based normalization (across all samples in batch)
                minv = np.percentile(rtm_feats, 1)   # Single global min
                maxv = np.percentile(rtm_feats, 99)  # Single global max
                
                # Normalize: clip → scale to [0,1] → shift to [-1,1]
                rtm_normalized = 2 * (np.clip(rtm_feats, minv, maxv) - minv) / (maxv - minv + 1e-6) - 1
                kp[..., feat_start:feat_end] = rtm_normalized
            
            # Normalize DAV2 features (256D) to [-1, 1]
            if self.use_depth_feats:
                # DAV2 comes after base + (RTM if enabled)
                feat_start = base_channels + (self.img_feat_dim if self.use_img_feats else 0)
                feat_end = feat_start + self.depth_feat_dim
                dav2_feats = kp[..., feat_start:feat_end]
                
                # Global percentile-based normalization
                minv = np.percentile(dav2_feats, 1)   # Single global min
                maxv = np.percentile(dav2_feats, 99)  # Single global max
                
                # Normalize: clip → scale to [0,1] → shift to [-1,1]
                dav2_normalized = 2 * (np.clip(dav2_feats, minv, maxv) - minv) / (maxv - minv + 1e-6) - 1
                kp[..., feat_start:feat_end] = dav2_normalized




        # 7) Root-relative 3D (meters)
        root = lt[..., self.root_index, :].mean(axis=-2, keepdims=True)
        lt_rel = lt - root
        if self.remove_root:
            lt_rel = np.delete(lt_rel, self.root_index, axis=1)

        # 8) Optional concat visibility
        if self.concat_vis:
            kv_exp = kv[..., None]
            kp = np.concatenate([kp, kv_exp], axis=2)

        return {
            'keypoint_labels': kp.astype(np.float32),
            'keypoint_labels_visible': kv.astype(np.float32),
            'lifting_target_label': lt_rel.astype(np.float32),
            'lifting_target_weight': lt_weight.astype(np.float32),
            'lifting_target': lt_rel.astype(np.float32),
            'lifting_target_visible': ltv.astype(np.float32),
        }

    def decode(self,
            encoded: np.ndarray,
            target_root: Optional[np.ndarray] = None
            ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert root-relative 3-D metres back to absolute pose.

        Args:
            encoded (np.ndarray): (N, K, 3) root-relative poses in metres.
            target_root (np.ndarray | None): (N, 1, 3) absolute root coordinates
                in metres.  If None, poses remain root-centred.

        Returns:
            keypoints (np.ndarray): (N, K, 3) absolute poses in metres.
            scores    (np.ndarray): (N, K)   dummy 1.0 confidences.
        """
        keypoints = encoded.copy()

        # add the true root back if supplied
        if target_root is not None and target_root.size > 0:
            keypoints += target_root                           # (N,K,3) + (N,1,3)
            if self.remove_root and len(self.root_index) == 1:
                # restore the explicit root joint that was dropped during encode
                keypoints = np.insert(
                    keypoints, self.root_index, target_root, axis=1)

        scores = np.ones(keypoints.shape[:-1], dtype=np.float32)
        return keypoints, scores

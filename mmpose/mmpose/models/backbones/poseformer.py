"""Single‑stream Spatial‑Temporal Transformer backbone – PoseFormer."""

import torch
import torch.nn as nn
from mmengine.model.weight_init import trunc_normal_
from mmpose.registry import MODELS
from mmpose.models.backbones.base_backbone import BaseBackbone
from mmcv.cnn.bricks.drop import DropPath  # already a dep of MMEngine

class EncoderBlock(nn.Module):
    """Wrapper that adds residual DropPath around a standard TransformerEncoderLayer."""
    def __init__(self, d_model, nhead, dim_feedforward, dropout, drop_prob):
        super().__init__()
        self.core   = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.dp     = DropPath(drop_prob) if drop_prob > 0 else nn.Identity()

    def forward(self, x):
        y = self.core(x)               # same shape, already contains its own residual
        return x + self.dp(y - x)      # stochastic‑depth on the *delta*


class SpatialTransformer(nn.Module):
    def __init__(
        self,
        in_channels,
        feat_size: int = 32,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        drop_path_rate: float = 0.2,
        # NEW: feature fusion parameters
        use_img_feats: bool = False,
        use_depth_feats: bool = False,
        img_feat_dim: int = 1024,  # RTM features (was 192, now 1024)
        depth_feat_dim: int = 256,
        feat_proj_dim: int = 16,
    ):
        super().__init__()
        
        # Store flags/sizes for fusion
        self.use_img_feats   = use_img_feats
        self.use_depth_feats = use_depth_feats
        self.img_feat_dim    = img_feat_dim
        self.depth_feat_dim  = depth_feat_dim
        self.proj_dim        = feat_proj_dim
        
        # Work out base (XY/XYC/XYCD/...) channel count from total in_channels
        base_dim = in_channels \
                   - (img_feat_dim   if use_img_feats   else 0) \
                   - (depth_feat_dim if use_depth_feats else 0)
        self.base_dim = base_dim
        
        # Per-stream projectors: 192→16, 256→16 (only created when used)
        self.img_proj   = nn.Linear(img_feat_dim,   feat_proj_dim) if use_img_feats   else nn.Identity()
        self.depth_proj = nn.Linear(depth_feat_dim, feat_proj_dim) if use_depth_feats else nn.Identity()
        
        # Final per-joint embed after concatenation: [base + (16?) + (16?)] → feat_size
        effective_in = base_dim \
                       + (feat_proj_dim if use_img_feats   else 0) \
                       + (feat_proj_dim if use_depth_feats else 0)
        self.embed_in = nn.Linear(effective_in, feat_size)
        
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, 17, feat_size))  # (1,1,K,D)

        # same TransformerEncoderLayer, but now we carry drop_path_rate
        # encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=feat_size,
        #     nhead=num_heads,
        #     dim_feedforward=int(feat_size * mlp_ratio),
        #     dropout=dropout,
        #     batch_first=True
        # )
        # self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        # --- build depth‑dependent DropPath schedule ---
        dpr = torch.linspace(0, drop_path_rate, depth).tolist()       # layer‑wise probs
        self.blocks = nn.ModuleList([
            EncoderBlock(feat_size, num_heads,
                        int(feat_size * mlp_ratio),
                        dropout, dpr[i])
            for i in range(depth)
        ])

        # store it so you can apply DropPath to each layer if you swap in a block
        self.drop_path_rate = drop_path_rate

        self.norm = nn.LayerNorm(feat_size)
        trunc_normal_(self.pos_embed, std=0.02)


    def forward(self, x):          # x: (B, F, K, C_in = base + [192] + [256])
        B, F, K, Cin = x.shape
        
        # === Slice, project, and concatenate features ===
        base_end = self.base_dim
        base = x[..., :base_end]                           # (B,F,K, base)
        cursor = base_end
        
        if self.use_img_feats:
            img_raw = x[..., cursor : cursor + self.img_feat_dim]   # (B,F,K,192)
            cursor += self.img_feat_dim
            img_proj = self.img_proj(img_raw)                       # (B,F,K,16)
            base = torch.cat([base, img_proj], dim=-1)
        
        if self.use_depth_feats:
            dep_raw = x[..., cursor : cursor + self.depth_feat_dim] # (B,F,K,256)
            cursor += self.depth_feat_dim
            dep_proj = self.depth_proj(dep_raw)                     # (B,F,K,16)
            base = torch.cat([base, dep_proj], dim=-1)
        
        # Embed fused features
        x = self.embed_in(base)                                     # (B,F,K, D)
        x = x + self.pos_embed.expand(B, F, K, -1)
        
        # Apply transformer blocks
        x = x.view(B * F, K, -1)          # flatten batch×frames
        for blk in self.blocks:           # ↙ stochastic depth happens here
            x = blk(x)
        x = x.view(B, F, K, -1)
        x = self.norm(x)
        return x
    # def forward(self, x):  # (B,F,K,C)
    #     B, F, K, C = x.shape
    #     x = self.embed(x) + self.pos_embed[:, :, :K]  # (B,F,K,D)
    #     x = self.encoder(x.view(B*F, K, -1)).view(B, F, K, -1)
    #     x = self.norm(x)
    #     return x


class TemporalTransformer(nn.Module):
    def __init__(
        self,
        feat_size: int = 544,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,         # ← changed from 4.0
        dropout: float = 0.1,
        seq_len: int = 243,
        drop_path_rate: float = 0.2     # ← new argument
    ):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, 1, feat_size))

        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList([
            EncoderBlock(feat_size, num_heads,
                        int(feat_size * mlp_ratio),
                        dropout, dpr[i])
            for i in range(depth)
        ])

        # encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=feat_size,
        #     nhead=num_heads,
        #     dim_feedforward=int(feat_size * mlp_ratio),
        #     dropout=dropout,
        #     batch_first=True
        # )
        # self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # again, hold onto drop_path_rate for later block‐level use
        self.drop_path_rate = drop_path_rate

        self.norm = nn.LayerNorm(feat_size)
        trunc_normal_(self.pos_embed, std=0.02)


    def forward(self, x):          # x: (B, F, D)
        B, F, D = x.shape
        x = x + self.pos_embed[:, :F, 0, :]               # add temporal pos‑embed
        for blk in self.blocks:
            x = blk(x)                    # (B, F, D) – same shape each layer
        x = self.norm(x)
        return x                                          # (B, F, D)

    # def forward(self, x):          # x: (B, F, D)
    #     B, F, D = x.shape
    #     x = x + self.pos_embed[:, :F, 0, :]        # broadcast pos-embed (1,F,D)
    #     x = self.encoder(x)                        # (B, F, D)
    #     x = self.norm(x)
    #     return x                                   # (B, F, 544)


@MODELS.register_module()
class PoseFormer(BaseBackbone):
    """PoseFormer Backbone – 2-stage Spatial+Temporal Transformer"""

    def __init__(
        self,
        in_channels: int = 2,
        feat_size: int = 32,          # spatial embedding dim per joint
        num_heads: int = 8,
        mlp_ratio: float = 2.0,       # ← changed from 4.0
        drop_path_rate: float = 0.2,  # ← new parameter
        spatial_depth: int = 4,
        temporal_depth: int = 4,
        seq_len: int = 243,
        num_keypoints: int = 17,
        dropout: float = 0.1,
        init_cfg=None,
        # NEW: feature fusion parameters (passed from config)
        use_img_feats: bool = False,
        use_depth_feats: bool = False,
        img_feat_dim: int = 192,
        depth_feat_dim: int = 256,
        feat_proj_dim: int = 16,
    ):
        super().__init__(init_cfg)

        self.num_keypoints = num_keypoints
        spatial_dim = feat_size
        temporal_dim = spatial_dim * num_keypoints  # 32*17 = 544
        
        # Store fusion params for passthrough
        self.use_img_feats   = use_img_feats
        self.use_depth_feats = use_depth_feats
        self.img_feat_dim    = img_feat_dim
        self.depth_feat_dim  = depth_feat_dim
        self.feat_proj_dim   = feat_proj_dim

        # Spatial Transformer: pass fusion params
        self.spatial_encoder = SpatialTransformer(
            in_channels=in_channels,
            feat_size=spatial_dim,
            depth=spatial_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
            # NEW: passthrough fusion params
            use_img_feats=use_img_feats,
            use_depth_feats=use_depth_feats,
            img_feat_dim=img_feat_dim,
            depth_feat_dim=depth_feat_dim,
            feat_proj_dim=feat_proj_dim,
        )

        # Frame projection (collapses joints → one token per frame)
        self.frame_proj = nn.Identity()

        # Temporal Transformer: also pass mlp_ratio & drop_path_rate
        self.temporal_encoder = TemporalTransformer(
            feat_size=temporal_dim,
            depth=temporal_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            seq_len=seq_len,
            drop_path_rate=drop_path_rate   # ← now wired through
        )

        # learnable weighted-mean pooling over the sequence dimension
        self.weighted_mean = nn.Conv1d(seq_len, 1, kernel_size=1)


    def forward(self, x):  # x: (B, F, K, C)
        B, F, K, C = x.shape

        # 1) Spatial
        x = self.spatial_encoder(x)              # (B, F, K, spatial_dim)

        # 2) Flatten each frame into a single 544-D token
        x = x.view(B, F, -1)          # (B, F, 544)  ← no per-joint replication


        # 4) Temporal
        # --- PoseFormer.forward() ---
        x = self.temporal_encoder(x)   # (B, F, D) with F == seq_len
        # x: (B, F, D)
        x = self.weighted_mean(x)      # (B, 1, D)
        return x                       # (B, 1, D)


        # # x: (B, F, D)
        # # 1) grab the raw frame‐pool weights (shape [1, F, 1]) and squeeze to a 1D tensor [F]
        # raw_w = self.weighted_mean.weight.squeeze(0).squeeze(-1)   # → (F,)

        # # 2) normalize them into a convex combination
        # norm_w = torch.softmax(raw_w, dim=0)                       # → (F,)

        # # 3) apply those weights across the frame dimension
        # #    (broadcast to [B, F, D], multiply, then sum over F)
        # x = (x * norm_w.view(1, F, 1)).sum(dim=1, keepdim=True)    # → (B, 1, D)

        # return x                           
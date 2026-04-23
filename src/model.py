"""Graph-based keypoint diffusion model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch import Tensor

from src.data import (
    CROWDPOSE_EDGE_LIST,
    CROWDPOSE_JOINT_NAMES,
    PoseSample,
    PoseGraphBatch,
    prepare_sample,
    prepare_pose_graph_batch,
    read_image_size,
)
from src.metrics import KeypointAPMetric, canonical_to_pixel, compute_mpjpe, compute_pck


@dataclass
class KeypointDiffusionConfig:
    """Hyperparameters for :class:`KeypointDiffusionModel`."""

    name: str = "keypoint_diffusion"

    # Feature dimensions
    dim_pos: int = 128  # position-embedding output dim
    dim_index: int = 128  # keypoint-index embedding dim
    dim_time: int = 128  # timestep-embedding dim
    dim_edge: int = 64   # edge-feature encoding dim (relative joint positions)

    # GNN backbone
    dim_hidden: int = 128
    num_layers: int = 4
    num_heads: int = 4

    # Diffusion schedule (VE / EDM-style: noisy = x + sigma * epsilon)
    num_train_steps: int = 100
    sigma_min: float = 0.002  # noise level at t=1 (nearly clean)
    sigma_max: float = 0.5    # noise level at t=T  (≈ image_scale / 2 in canonical space)

    # DINO image conditioning
    use_image_cond: bool = False
    dim_image_cond: int = 128  # projected DINO feature dim per node

    # Global image conditioning: pooled DINO per-image signal combined with timestep
    use_global_image_cond: bool = False
    dim_global_image_cond: int = 128  # projected pooled-DINO dim

    # Training augmentations
    aug_joint_drop: bool = True
    p_joint_drop: float = 0.1  # probability of dropping each non-root joint independently
    aug_pose_drop: bool = True
    p_pose_drop: float = 0.1  # probability of dropping an entire person
    aug_joint_rand: bool = True
    p_joint_rand: float = 0.1  # probability of replacing each non-root joint with a random position
    aug_jitter: bool = True
    p_jitter_px: float = 4.0  # std-dev (pixels) of Gaussian keypoint coordinate noise
    aug_img_drop: bool = True
    p_img_drop: float = 0.1  # fraction of DINO node embeddings zeroed out

    # Architecture ablations
    use_message_passing: bool = True  # if False, use per-node MLP (no graph aggregation or bone embedding)
    zero_delta: bool = False          # if True, always return delta=0 (identity refinement; for sanity checks)

    # Regularisation
    dropout_rate: float = 0.1       # dropout probability in MLPs
    attn_dropout_rate: float = 0.1  # dropout probability on attention weights

    # Loss terms
    use_joint_loss: bool = True
    w_joint_loss: float = 1.0
    use_bone_loss: bool = True
    w_bone_loss: float = 1.0


class TimestepSinusoidalEmbedding(torch.nn.Module):
    """Sinusoidal positional encoding for a batch of integer timesteps."""

    def __init__(self, dim: int, max_period: int = 10_000) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim must be even, got {dim}")
        self.dim = dim
        self.max_period = max_period

    def forward(self, timestep: Tensor) -> Tensor:
        half_dim = self.dim // 2
        freqs = torch.exp(
            -torch.arange(half_dim, dtype=torch.float32, device=timestep.device)
            * np.log(self.max_period)
            / half_dim
        )  # (half_dim,)
        args = timestep[:, None].float() * freqs[None]  # (B, half_dim)
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, dim)


def pixel_to_grid_sample_coords(
    kpt_pixel: Tensor,  # (..., 2)  pixel-space (x, y)
    img_hw: Tensor,     # (..., 2)  original image (H, W) — broadcastable to kpt_pixel
) -> Tensor:            # (..., 2)  grid_sample coords in [-1, 1]
    # With resize-only preprocessing (do_center_crop=False,
    # size={"shortest_edge": S}), a scale s = S/min(H,W) maps pixel coords
    # to feature-map coords and also scales the feature-map size by the same
    # factor, so s cancels in the normalisation:
    #   x_feat = x * s / stride
    #   x_norm = 2 * x_feat / (W_orig * s / stride) - 1
    #          = 2 * x / W_orig - 1
    # Normalising against the original image dimensions is therefore exact.
    img_h = img_hw[..., 0].float()
    img_w = img_hw[..., 1].float()
    x_norm = (kpt_pixel[..., 0].float() / img_w) * 2 - 1
    y_norm = (kpt_pixel[..., 1].float() / img_h) * 2 - 1
    return torch.stack([x_norm, y_norm], dim=-1)


class DinoConditioner(torch.nn.Module):
    """Per-keypoint ConvNeXt DINO features, sampled from precomputed caches.

    Two feature levels are sampled at each keypoint location and concatenated:
      spatial:  stage1 (96ch @ 56×56) — fine-grained spatial texture
      semantic: stage4 (768ch @ 7×7)  — high-level semantics
    The 864-d vector is linearly projected to out_dim.

    A pre-pooled global image vector (768-d) is also available via
    fetch_global_feats() for use as a per-image global conditioning signal.

    Caches must be precomputed with scripts/precompute_dino.py --backbone-type convnext.
    """

    # Architectural constants for facebook/dinov3-convnext-small-pretrain-lvd1689m.
    SPATIAL_DIM: int = 96   # stage1 channels
    HIDDEN_DIM: int  = 768  # stage4 / pooler channels

    def __init__(
        self,
        model_name: str = "facebook/dinov3-convnext-small-pretrain-lvd1689m",
        out_dim: int = 128,
        cache_paths: list[str | Path] | None = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.spatial_dim: int = self.SPATIAL_DIM
        self.hidden_dim: int  = self.HIDDEN_DIM
        self.proj = torch.nn.Linear(self.spatial_dim + self.hidden_dim, out_dim)

        # In-memory cache: {image_id: {'spatial': ..., 'semantic': ..., 'global': ...}}
        # Populated lazily on first access via _load_image().
        self._cache: dict[int, dict] = {}
        # Directories to search for {image_id}.pt files.
        self._cache_dirs: list[Path] = []
        if cache_paths:
            self.load_cache(*cache_paths)

    def load_cache(self, *paths: str | Path) -> None:
        """Register one or more cache directories to load images from on demand."""
        for path in paths:
            p = Path(path)
            if not p.exists():
                # Backward-compat: old checkpoints store 'split.pt' paths; the new
                # per-image format lives in the same-named directory without the suffix.
                if p.suffix == ".pt":
                    candidate = p.with_suffix("")
                    if candidate.is_dir():
                        print(
                            f"DinoConditioner: '{p}' not found; "
                            f"using per-image directory '{candidate}' instead."
                        )
                        p = candidate
                    else:
                        print(f"DinoConditioner: cache path not found, skipping: {p}")
                        continue
                else:
                    print(f"DinoConditioner: cache directory not found, skipping: {p}")
                    continue
            if not p.is_dir():
                raise RuntimeError(
                    f"Cache path {p} is a file, not a directory. "
                    "Re-run scripts/precompute_dino.py to generate per-image .pt files."
                )
            self._cache_dirs.append(p)
        print(f"DinoConditioner: registered {len(self._cache_dirs)} cache dir(s)")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_dino_feature(self, iid: int) -> dict:
        """Return the cached feature dict for one image, loading from disk if needed."""
        if iid not in self._cache:
            if not self._cache_dirs:
                raise RuntimeError(
                    "Feature cache not loaded. "
                    "Call load_cache() with the directory produced by "
                    "scripts/precompute_dino.py --backbone-type convnext."
                )
            for cache_dir in self._cache_dirs:
                pt_path = cache_dir / f"{iid}.pt"
                if pt_path.exists():
                    self._cache[iid] = torch.load(pt_path, map_location="cpu", weights_only=False)
                    break
            else:
                raise RuntimeError(
                    f"Image id {iid} not found in any cache directory: "
                    + ", ".join(str(d) for d in self._cache_dirs)
                )
        return self._cache[iid]

    def _load_maps(
        self, image_ids: list[int], device: torch.device
    ) -> tuple[list[Tensor], list[Tensor]]:
        """Load spatial and semantic feature maps from the cache for a batch.

        Returns lists (not stacked) because images may have different spatial
        dimensions when preprocessed with aspect-ratio-preserving resize.

            spatial_maps:  list of B tensors, each (96, H_i/4, W_i/4)
            semantic_maps: list of B tensors, each (768, H_i/32, W_i/32)
        """
        spatial  = [self._load_dino_feature(iid)["spatial"].to(device)  for iid in image_ids]
        semantic = [self._load_dino_feature(iid)["semantic"].to(device) for iid in image_ids]
        return spatial, semantic

    def _sample_at_keypoints(
        self,
        feat_maps: list[Tensor],  # B tensors, each (C, H_i, W_i)
        kpt_pixel: Tensor,        # (B, N, K, 2)  pixel-space (x, y)
        img_hw: Tensor,           # (B, 2)  original image (H, W)
    ) -> Tensor:                  # (B, N, K, C)
        """Bilinearly sample a feature map at each keypoint's pixel location.

        Processes one image at a time so that variable spatial dimensions
        (from aspect-ratio-preserving resize) are handled without padding.
        Keypoints outside the image boundary return zero features.
        """
        B = len(feat_maps)
        N, K = kpt_pixel.shape[1], kpt_pixel.shape[2]
        device = kpt_pixel.device
        results = []
        for i in range(B):
            feat = feat_maps[i].to(device)              # (C, H_i, W_i)
            C = feat.shape[0]
            coords = pixel_to_grid_sample_coords(
                kpt_pixel[i].unsqueeze(0),              # (1, N, K, 2)
                img_hw[i].to(device).view(1, 1, 1, 2),
            )  # (1, N, K, 2)
            grid = coords.reshape(1, N * K, 1, 2)
            sampled = F.grid_sample(
                feat.unsqueeze(0).contiguous(), grid,
                mode="bilinear", align_corners=False, padding_mode="zeros",
            )  # (1, C, N*K, 1)
            results.append(sampled.squeeze(0).squeeze(-1).permute(1, 0).view(N, K, C))
        return torch.stack(results)  # (B, N, K, C)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_global_feats(self, image_ids: list[int], device: torch.device) -> Tensor:
        """Load the pre-pooled global image vector from the cache → (B, hidden_dim)."""
        return torch.stack(
            [self._load_dino_feature(iid)["global"] for iid in image_ids]
        ).to(device)

    def forward(
        self,
        kpt_pixel: Tensor,    # (B, N, K, 2) pixel-space (x, y)
        img_hw: Tensor,       # (B, 2) original image (H, W)
        image_ids: list[int], # image IDs to look up in the cache
    ) -> Tensor:              # (B, N, K, out_dim)
        spatial_maps, semantic_maps = self._load_maps(image_ids, kpt_pixel.device)
        spatial_feats  = self._sample_at_keypoints(spatial_maps,  kpt_pixel, img_hw)  # (B, N, K, 96)
        semantic_feats = self._sample_at_keypoints(semantic_maps, kpt_pixel, img_hw)  # (B, N, K, 768)
        return self.proj(torch.cat([spatial_feats, semantic_feats], dim=-1))           # (B, N, K, out_dim)


class AttentionDinoConditioner(DinoConditioner):
    """Per-keypoint DINO conditioning via cross-attention over all patch embeddings.

    Architecture:
      1. Fuse stage1 (96ch) and upsampled stage4 (768ch) patches → inner_dim via FF.
      2. Project normalised joint positions → node query features (inner_dim).
      3. Cross-attention (Q=nodes, K/V=patches) with a per-head relative-position
         bias (patch_pos − node_pos → Linear → num_heads) folded into attn_mask.
         Padded patch slots (variable P_i across the batch) are masked to −inf.
      4. Residual + LayerNorm + linear.
      5. Add absolute joint position encoding.
      6. Self-attention over K joints per person; absent joints masked via
         key_padding_mask and zeroed in the final output.
      7. Residual + LayerNorm + linear → out_proj → out_dim.

    Patch padding: images in a batch may have different numbers of stage-1 patches
    (variable spatial size after aspect-ratio-preserving resize).  All images are
    padded to max_P with zeros; padded positions receive −inf attention bias so
    they never contribute to the pooled output.

    NOTE: cross-attention memory scales as O(N·K·max_P).  For shortest_edge ≥ 448
    (stage1 ≈ 112×112 = 12 544 patches) this may be significant.
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov3-convnext-small-pretrain-lvd1689m",
        out_dim: int = 128,
        inner_dim: int = 256,
        num_heads: int = 4,
        dropout_rate: float = 0.1,
        cache_paths: list[str | Path] | None = None,
    ) -> None:
        super().__init__(model_name=model_name, out_dim=out_dim, cache_paths=cache_paths)
        assert inner_dim % num_heads == 0, (
            f"inner_dim ({inner_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.inner_dim = inner_dim
        self.num_heads = num_heads
        self.out_dim = out_dim

        # Fuse stage1 (96ch) + upsampled stage4 (768ch) → inner_dim
        self.patch_fusion = torch.nn.Sequential(
            torch.nn.Linear(self.SPATIAL_DIM + self.HIDDEN_DIM, inner_dim * 2),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(inner_dim * 2, inner_dim),
        )

        # Project normalised joint position (2D) → node query features
        self.node_pos_proj = torch.nn.Linear(2, inner_dim)

        # Relative-position bias for cross-attention: 2D offset → per-head scalar
        self.cross_rel_pos_bias = torch.nn.Linear(2, num_heads, bias=False)

        # Cross-attention block (nodes attend to patches)
        self.cross_attn = torch.nn.MultiheadAttention(
            inner_dim, num_heads, batch_first=True, dropout=dropout_rate
        )
        self.cross_norm    = torch.nn.LayerNorm(inner_dim)
        self.cross_linear  = torch.nn.Linear(inner_dim, inner_dim)
        self.cross_ff_norm = torch.nn.LayerNorm(inner_dim)

        # Absolute joint position encoding (added inside self-attention block)
        self.joint_pos_proj = torch.nn.Linear(2, inner_dim)

        # Self-attention block over K joints per person
        self.self_attn    = torch.nn.MultiheadAttention(
            inner_dim, num_heads, batch_first=True, dropout=dropout_rate
        )
        self.self_norm    = torch.nn.LayerNorm(inner_dim)
        self.self_linear  = torch.nn.Linear(inner_dim, inner_dim)
        self.self_ff_norm = torch.nn.LayerNorm(inner_dim)

        # Final projection
        self.out_proj = torch.nn.Linear(inner_dim, out_dim)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fuse_patches(
        self,
        spatial_map:  Tensor,  # (96,  H_s,     W_s    )  stage1, stride-4
        semantic_map: Tensor,  # (768, H_s//8,  W_s//8 )  stage4, stride-32
    ) -> tuple[Tensor, Tensor]:  # (P, inner_dim), (P, 2)
        C_s, H_s, W_s = spatial_map.shape
        P = H_s * W_s

        # Upsample semantic map to stage1 spatial resolution
        semantic_up = F.interpolate(
            semantic_map.unsqueeze(0),
            size=(H_s, W_s),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)  # (768, H_s, W_s)

        # Flatten patches and concatenate both levels: (P, 96+768)
        spatial_flat  = spatial_map.permute(1, 2, 0).reshape(P, C_s)
        semantic_flat = semantic_up.permute(1, 2, 0).reshape(P, self.HIDDEN_DIM)
        combined = torch.cat([spatial_flat, semantic_flat], dim=-1)  # (P, 864)

        patch_feats = self.patch_fusion(combined)  # (P, inner_dim)

        # Patch centre positions normalised to [-1, 1] (same space as node positions)
        device = spatial_map.device
        ys = (torch.arange(H_s, device=device).float() + 0.5) / H_s * 2 - 1
        xs = (torch.arange(W_s, device=device).float() + 0.5) / W_s * 2 - 1
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (H_s, W_s)
        patch_pos = torch.stack([grid_x, grid_y], dim=-1).reshape(P, 2)  # (P, 2) as (x, y)

        return patch_feats, patch_pos

    def _build_patch_batch(
        self,
        all_patch_feats: list[Tensor],  # B tensors, each (P_i, inner_dim)
        all_patch_pos:   list[Tensor],  # B tensors, each (P_i, 2)
    ) -> tuple[Tensor, Tensor, Tensor]:
        # Returns padded_feats (B, max_P, d), padded_pos (B, max_P, 2),
        #         patch_key_mask (B, max_P) bool — True = padded slot.
        B      = len(all_patch_feats)
        max_P  = max(f.shape[0] for f in all_patch_feats)
        device = all_patch_feats[0].device
        dtype  = all_patch_feats[0].dtype

        padded_feats   = torch.zeros(B, max_P, self.inner_dim, device=device, dtype=dtype)
        padded_pos     = torch.zeros(B, max_P, 2,              device=device, dtype=dtype)
        patch_key_mask = torch.ones( B, max_P, dtype=torch.bool, device=device)  # True = padded

        for i, (feats, pos) in enumerate(zip(all_patch_feats, all_patch_pos)):
            P_i = feats.shape[0]
            padded_feats[i, :P_i]   = feats
            padded_pos[i,   :P_i]   = pos
            patch_key_mask[i, :P_i] = False  # real patches

        return padded_feats, padded_pos, patch_key_mask

    def _cross_attend(
        self,
        node_feats:     Tensor,  # (B, NK, inner_dim)
        patch_feats:    Tensor,  # (B, max_P, inner_dim)
        node_pos:       Tensor,  # (B, NK, 2)
        patch_pos:      Tensor,  # (B, max_P, 2)
        patch_key_mask: Tensor,  # (B, max_P) bool — True = padded
    ) -> Tensor:                 # (B, NK, inner_dim)
        B, NK, _ = node_feats.shape
        max_P = patch_feats.shape[1]
        H     = self.num_heads

        # Relative offset of every patch to every node: (B, NK, max_P, 2)
        rel_pos  = patch_pos.unsqueeze(1) - node_pos.unsqueeze(2)

        # Per-head relative-position bias: (B, NK, max_P, H) → (B*H, NK, max_P)
        rel_bias = self.cross_rel_pos_bias(rel_pos)                        # (B, NK, max_P, H)
        rel_bias = rel_bias.permute(0, 3, 1, 2).reshape(B * H, NK, max_P) # (B*H, NK, max_P)

        # Fold padding mask into attn_mask: padded positions → −inf attention weight
        pad_inf = (
            patch_key_mask
            .view(B, 1, 1, max_P)
            .expand(B, H, NK, max_P)
            .reshape(B * H, NK, max_P)
        )
        rel_bias = rel_bias.masked_fill(pad_inf, float("-inf"))

        attn_out, _ = self.cross_attn(
            node_feats,   # Q: (B, NK, inner_dim)
            patch_feats,  # K: (B, max_P, inner_dim)
            patch_feats,  # V: (B, max_P, inner_dim)
            attn_mask=rel_bias,  # (B*H, NK, max_P)
        )  # (B, NK, inner_dim)
        h = self.cross_norm(node_feats + attn_out)
        h = self.cross_ff_norm(h + self.cross_linear(h))
        return h

    def _self_attend_joints(
        self,
        h:          Tensor,        # (BN, K, inner_dim)
        joint_pos:  Tensor,        # (BN, K, 2)  normalised [-1, 1]
        joint_mask: Tensor | None, # (BN, K) bool — True = absent; or None
    ) -> Tensor:                   # (BN, K, inner_dim)
        q = h + self.joint_pos_proj(joint_pos)  # add absolute position encoding

        # Guard: all-absent rows would produce NaN in softmax.
        safe_mask = None
        if joint_mask is not None:
            all_absent = joint_mask.all(dim=-1)  # (BN,)
            safe_mask  = joint_mask.clone()
            if all_absent.any():
                safe_mask[all_absent, 0] = False

        attn_out, _ = self.self_attn(q, q, q, key_padding_mask=safe_mask)  # (BN, K, inner_dim)
        h = self.self_norm(q + attn_out)
        h = self.self_ff_norm(h + self.self_linear(h))
        return h

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        kpt_pixel:  Tensor,            # (B, N, K, 2)  pixel-space (x, y)
        img_hw:     Tensor,            # (B, 2)         original image (H, W)
        image_ids:  list[int],
        joint_mask: Tensor | None = None,  # (B, N, K) bool — True = absent joint
    ) -> Tensor:                       # (B, N, K, out_dim)
        B, N, K, _ = kpt_pixel.shape
        device = kpt_pixel.device

        spatial_maps, semantic_maps = self._load_maps(image_ids, device)
        all_patch_feats: list[Tensor] = []
        all_patch_pos:   list[Tensor] = []
        for i in range(B):
            feats, pos = self._fuse_patches(spatial_maps[i], semantic_maps[i])
            all_patch_feats.append(feats)
            all_patch_pos.append(pos)

        padded_feats, padded_pos, patch_key_mask = self._build_patch_batch(
            all_patch_feats, all_patch_pos
        )  # (B, max_P, d), (B, max_P, 2), (B, max_P)

        node_pos_norm   = pixel_to_grid_sample_coords(
            kpt_pixel, img_hw.view(B, 1, 1, 2)
        )  # (B, N, K, 2)  in [-1, 1]
        node_pos_flat   = node_pos_norm.view(B, N * K, 2)
        h               = self.node_pos_proj(node_pos_flat)  # (B, NK, inner_dim)

        h = h + self._cross_attend(h, padded_feats, node_pos_flat, padded_pos, patch_key_mask)

        joint_pos       = node_pos_norm.view(B * N, K, 2)
        h               = h.view(B * N, K, self.inner_dim)
        joint_mask_flat = joint_mask.view(B * N, K) if joint_mask is not None else None
        h = h + self._self_attend_joints(h, joint_pos, joint_mask_flat)

        out = self.out_proj(h)
        if joint_mask is not None:
            out = out.masked_fill(joint_mask_flat.unsqueeze(-1), 0.0)
        return out.view(B, N, K, self.out_dim)


class FiLMLayer(torch.nn.Module):
    """Feature-wise Linear Modulation conditioned on a global vector: h → (1 + scale) * h + shift."""

    def __init__(self, cond_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(cond_dim, 2 * feature_dim)

    def forward(self, h: Tensor, g: Tensor) -> Tensor:
        # h: (M, feature_dim)   g: (M, cond_dim)
        params = self.proj(g)                        # (M, 2 * feature_dim)
        scale, shift = params.chunk(2, dim=-1)
        return (1.0 + scale) * h + shift


class PoseGraphLayer(MessagePassing):
    """One graph attention layer: FiLM conditioning → edge-attentive message passing → residual update.

    When use_message_passing=False the attention is skipped and the update MLP receives only [h_i, v_i].
    """

    def __init__(
        self,
        dim: int,
        dim_edge: int,
        dim_vis: int,
        dim_global: int,
        num_heads: int = 4,
        use_message_passing: bool = True,
        dropout_rate: float = 0.1,
        attn_dropout_rate: float = 0.1,
    ) -> None:
        super().__init__(aggr="add")  # weighted sum handled inside message()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        self.use_message_passing = use_message_passing
        self.attn_dropout = attn_dropout_rate

        if use_message_passing:
            self.W_q = torch.nn.Linear(dim, dim, bias=False)
            self.W_k = torch.nn.Linear(dim, dim, bias=False)
            self.W_v = torch.nn.Linear(dim, dim, bias=False)

            msg_in = dim + dim_edge + dim_vis
            self.msg_mlp = torch.nn.Sequential(
                torch.nn.LayerNorm(msg_in),
                torch.nn.Linear(msg_in, dim * 2),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout_rate),
                torch.nn.Linear(dim * 2, dim),
            )

            self.edge_bias = torch.nn.Linear(dim_edge, num_heads, bias=False)

        update_in = dim + (dim if use_message_passing else 0) + dim_vis
        self.update_mlp = torch.nn.Sequential(
            torch.nn.LayerNorm(update_in),
            torch.nn.Linear(update_in, dim * 2),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(dim * 2, dim),
        )

        self.film = FiLMLayer(dim_global, dim)

    def message(
        self,
        q_i: Tensor,        # (E, dim)      — destination query
        k_j: Tensor,        # (E, dim)      — source key
        v_state_j: Tensor,  # (E, dim)      — source value-state
        edge_feat: Tensor,  # (E, dim_edge) — per-edge relative-position features
        vis_j: Tensor,      # (E, dim_vis)  — source visual features (sliced by PyG)
        edge_dst: Tensor,   # (E,)          — destination node index for softmax
    ) -> Tensor:            # (E, dim) attention-weighted messages
        E = q_i.shape[0]
        H, D = self.num_heads, self.head_dim

        msg = self.msg_mlp(torch.cat([v_state_j, edge_feat, vis_j], dim=-1))  # (E, dim)

        q_h = q_i.view(E, H, D)
        k_h = k_j.view(E, H, D)
        dot   = (q_h * k_h).sum(-1) / math.sqrt(D)       # (E, H)
        score = dot + self.edge_bias(edge_feat)            # (E, H)
        alpha = softmax(score, index=edge_dst, num_nodes=self._num_nodes)  # (E, H)
        alpha = F.dropout(alpha, p=self.attn_dropout, training=self.training)  # (E, H)

        msg_h    = msg.view(E, H, D)
        weighted = (msg_h * alpha.unsqueeze(-1)).view(E, self.dim)  # (E, dim)
        return weighted

    def forward(
        self,
        h: Tensor,           # (M, dim)      M = B*N*K
        edge_index: Tensor,  # (2, E)
        e: Tensor,           # (E, dim_edge) per-edge relative-position features
        v: Tensor,           # (M, dim_vis)  per-node local visual features
        g: Tensor,           # (M, dim_global) global conditioning (timestep …)
    ) -> Tensor:
        h = self.film(h, g)

        if self.use_message_passing:
            q       = self.W_q(h)
            k       = self.W_k(h)
            v_state = self.W_v(h)

            _, dst = edge_index[0], edge_index[1]
            # Pass per-edge and per-node tensors through propagate kwargs.
            # Avoids stashing activations as module attributes, which confuses
            # torch.compile (dynamo captures stashed dtype and produces stale kernels
            # when bf16-mixed autocast changes activation dtypes).
            self._num_nodes = int(h.shape[0])  # plain Python int — safe for compile
            agg = self.propagate(
                edge_index,
                q=q, k=k, v_state=v_state,
                edge_feat=e,   # (E, dim_edge) — per-edge, passed as-is to message()
                vis=v,         # (M, dim_vis)  — PyG slices vis_j = v[src] automatically
                edge_dst=dst,  # (E,)          — destination indices for softmax
            )  # (M, dim)
            h = h + self.update_mlp(torch.cat([h, agg, v], dim=-1))
        else:
            h = h + self.update_mlp(torch.cat([h, v], dim=-1))

        return h


class PoseGraphDecoder(torch.nn.Module):
    """Stacked PoseGraphLayers followed by a LayerNorm + linear coordinate prediction head."""

    def __init__(
        self,
        dim_edge: int,
        dim_vis: int,
        dim_global: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int = 4,
        out_dim: int = 2,
        use_message_passing: bool = True,
        dropout_rate: float = 0.1,
        attn_dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([
            PoseGraphLayer(
                dim=hidden_dim,
                dim_edge=dim_edge,
                dim_vis=dim_vis,
                dim_global=dim_global,
                num_heads=num_heads,
                use_message_passing=use_message_passing,
                dropout_rate=dropout_rate,
                attn_dropout_rate=attn_dropout_rate,
            )
            for _ in range(num_layers)
        ])
        self.output_proj = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, out_dim),
        )

        # Initialise the final linear layer to zero so the untrained model predicts zero displacement.
        self.output_proj[1].weight.data.zero_()
        self.output_proj[1].bias.data.zero_()

    def forward(
        self,
        h: Tensor,           # (M, hidden_dim)  — h_i = LN(pos_emb + id_emb)
        edge_index: Tensor,  # (2, E)
        e: Tensor,           # (E, dim_edge)  per-edge geometry features
        v: Tensor,           # (M, dim_vis)   per-node local visual features
        g: Tensor,           # (M, dim_global) global conditioning
    ) -> Tensor:
        for layer in self.layers:
            h = layer(h, edge_index, e, v, g)
        return self.output_proj(h)        # (M, out_dim)


class KeypointDiffusionModel(torch.nn.Module):
    """Graph diffusion model over pose keypoints."""

    def __init__(
        self,
        config: KeypointDiffusionConfig,
        num_keypoints: int = 14,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_keypoints = num_keypoints

        self.position_embedding = torch.nn.Sequential(
            torch.nn.Linear(2, config.dim_pos),
            torch.nn.GELU(),
            torch.nn.Dropout(config.dropout_rate),
            torch.nn.Linear(config.dim_pos, config.dim_hidden),
        )
        self.node_idx_embedding = torch.nn.Embedding(num_keypoints, config.dim_hidden)
        self.node_init_norm = torch.nn.LayerNorm(config.dim_hidden)

        self.timestep_embedding = TimestepSinusoidalEmbedding(config.dim_time)
        global_in = config.dim_time + (
            config.dim_global_image_cond if config.use_global_image_cond else 0
        )
        self.global_cond_mlp = torch.nn.Sequential(
            torch.nn.Linear(global_in, config.dim_time),
            torch.nn.GELU(),
            torch.nn.Dropout(config.dropout_rate),
            torch.nn.Linear(config.dim_time, config.dim_time),
        )

        self.edge_encoder = torch.nn.Sequential(
            torch.nn.Linear(2, config.dim_edge),
            torch.nn.GELU(),
            torch.nn.Dropout(config.dropout_rate),
            torch.nn.Linear(config.dim_edge, config.dim_edge),
        )

        # Visual features (v_i) are zeros when use_image_cond=False; dim is always dim_image_cond.
        dim_vis = config.dim_image_cond

        self.decoder = PoseGraphDecoder(
            dim_edge=config.dim_edge,
            dim_vis=dim_vis,
            dim_global=config.dim_time,
            hidden_dim=config.dim_hidden,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            use_message_passing=config.use_message_passing,
            dropout_rate=config.dropout_rate,
            attn_dropout_rate=config.attn_dropout_rate,
        )

        # VE noise schedule (EDM-style): sigmas[0]=0 (clean sentinel used at the end
        # of inference), sigmas[1..T] increase log-linearly from sigma_min to sigma_max.
        T = config.num_train_steps
        sigmas = torch.zeros(T + 1)
        sigmas[1:] = torch.exp(
            torch.linspace(math.log(config.sigma_min), math.log(config.sigma_max), T)
        )
        self.register_buffer("_sigmas", sigmas)

    def add_noise(
        self,
        node_pos: Tensor,
        timestep: Tensor,
        batch_idx: Tensor,
        generator: torch.Generator | None = None,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        noise = torch.zeros_like(node_pos).normal_(generator=generator)
        t_node = timestep[batch_idx]           # broadcast (B,) → (B*N*K,)
        sigma = self._sigmas[t_node].unsqueeze(-1)  # (B*N*K, 1)
        return node_pos + sigma * noise, noise

    def backward_process(
        self,
        noisy_node_pos: Tensor,
        edge_index: Tensor,
        kpt_type_idx: Tensor,
        batch_idx: Tensor,
        timestep: Tensor,
        image_cond: Tensor | None = None,         # (B*N*K, dim_image_cond) or None
        global_image_cond: Tensor | None = None,  # (B*N*K, dim_global_image_cond) or None
        **kwargs,
    ) -> Tensor:
        M = noisy_node_pos.shape[0]  # B*N*K

        if self.config.zero_delta:
            return torch.zeros_like(noisy_node_pos)

        pos_emb = self.position_embedding(noisy_node_pos)  # (M, dim_hidden)
        idx_emb = self.node_idx_embedding(kpt_type_idx)    # (M, dim_hidden)
        h = self.node_init_norm(pos_emb + idx_emb)         # (M, dim_hidden)

        src, dst = edge_index[0], edge_index[1]
        delta_ij = noisy_node_pos[src] - noisy_node_pos[dst]  # (E, 2)
        e = self.edge_encoder(delta_ij)                        # (E, dim_edge)

        if image_cond is not None:
            v = image_cond
        else:
            v = torch.zeros(
                M, self.config.dim_image_cond,
                device=noisy_node_pos.device, dtype=noisy_node_pos.dtype,
            )

        t_emb = self.timestep_embedding(timestep)[batch_idx]  # (M, dim_time)
        if global_image_cond is not None:
            t_emb = torch.cat([t_emb, global_image_cond], dim=-1)  # (M, dim_time + dim_global)
        g = self.global_cond_mlp(t_emb)                           # (M, dim_time)

        return self.decoder(h, edge_index, e, v, g)  # (M, 2)

    @torch.no_grad()
    def ddim_sample(
        self,
        edge_index: Tensor,
        kpt_type_idx: Tensor,
        batch_idx: Tensor,
        num_nodes: int,
        num_inference_steps: int | None = None,
        start_t: int | None = None,
        device: torch.device | None = None,
        x_init: Tensor | None = None,
        image_cond: Tensor | None = None,         # (B*N*K, dim_image_cond) or None
        global_image_cond: Tensor | None = None,  # (B*N*K, dim_global_image_cond) or None
        guidance_scale: float = 1.0,              # CFG scale; 1.0 = no guidance
    ) -> Tensor:
        T = self.config.num_train_steps
        t_start = start_t if start_t is not None else T
        num_steps = num_inference_steps if num_inference_steps is not None else t_start
        device = device or edge_index.device

        timesteps = torch.linspace(
            t_start, 1, num_steps, dtype=torch.long, device=device
        )

        if x_init is not None:
            x = x_init.to(device)
        else:
            x = torch.randn(num_nodes, 2, device=device)

        prev_timestep_list = torch.cat(
            [timesteps[1:], timesteps.new_zeros(1)]
        )  # trailing 0 = clean
        B = batch_idx.max().item() + 1  # computed once to avoid per-step CPU-GPU sync
        for timestep, prev_timestep in zip(timesteps, prev_timestep_list):
            timestep_batch = timestep.expand(B)  # (B,)

            bp_kwargs = dict(
                noisy_node_pos=x,
                edge_index=edge_index,
                kpt_type_idx=kpt_type_idx,
                batch_idx=batch_idx,
                timestep=timestep_batch,
                image_cond=image_cond,
                global_image_cond=global_image_cond,
            )
            has_cond = image_cond is not None or global_image_cond is not None
            if guidance_scale != 1.0 and has_cond:
                # Classifier-free guidance: blend conditioned and unconditioned predictions.
                cond_delta = self.backward_process(**bp_kwargs)
                uncond_delta = self.backward_process(**{
                    **bp_kwargs,
                    "image_cond": torch.zeros_like(image_cond) if image_cond is not None else None,
                    "global_image_cond": torch.zeros_like(global_image_cond) if global_image_cond is not None else None,
                })
                delta_x = uncond_delta + guidance_scale * (cond_delta - uncond_delta)
            else:
                delta_x = self.backward_process(**bp_kwargs)  # (B*N*K, 2)
            x0_pred = x - delta_x

            # DDIM deterministic update — VE schedule.
            # At prev_timestep=0: sigma_prev=0  →  x = x0_pred  (clean sample). ✓
            timestep_idx     = timestep.long()
            prev_timestep_idx = prev_timestep.long()
            sigma_t    = self._sigmas[timestep_idx]
            sigma_prev = self._sigmas[prev_timestep_idx]
            noise_pred = (x - x0_pred) / sigma_t        # estimated unit Gaussian epsilon
            x = x0_pred + sigma_prev * noise_pred

        return x


@dataclass
class StepOutputs:
    pred_pixel: Tensor       # (B, N, 14, 2)
    pred_canonical: Tensor   # (B, N, 14, 2)
    gt_pixel: Tensor         # (B, N, 14, 2)
    gt_canonical: Tensor     # (B, N, 14, 2)
    timestep: Tensor         # (B,)


def denoise_and_decode(
    model: KeypointDiffusionModel,
    sample: PoseSample,
    graph: PoseGraphBatch,
    timestep: Tensor,        # (B,)
    generator: torch.Generator | None = None,
    noisy_node_pos: Tensor | None = None,  # pre-computed; add_noise is skipped when provided
) -> StepOutputs:
    """Add noise at timestep, run backward_process, decode to pixel space."""
    if noisy_node_pos is None:
        noisy_node_pos, _ = model.add_noise(
            node_pos=graph.node_pos,
            timestep=timestep,
            batch_idx=graph.node_batch,
            generator=generator,
        )
    pred_delta = model.backward_process(
        noisy_node_pos=noisy_node_pos,
        edge_index=graph.edge_index,
        kpt_type_idx=graph.flat_kpt_type_idx,
        batch_idx=graph.node_batch,
        timestep=timestep,
        image_cond=graph.image_cond,
        global_image_cond=graph.global_image_cond,
    )

    # canonical = (noisy - delta); inverse: pixel = (canonical + 1) / 2 * [W, H]
    pred_canonical = (noisy_node_pos - pred_delta).view(graph.B, graph.N, graph.K, 2)  # (B, N, 14, 2)
    pred_pixel = (pred_canonical + 1) / 2 * graph.img_wh[:, None, None, :]             # (B, N, 14, 2)

    gt_canonical = sample.gt_keypoints / graph.img_wh[:, None, None, :] * 2 - 1

    return StepOutputs(
        pred_pixel=pred_pixel,
        pred_canonical=pred_canonical,
        gt_pixel=sample.gt_keypoints,
        gt_canonical=gt_canonical,
        timestep=timestep,
    )


def _log_eval_step(
    module: "PoseDiffusionModule",
    losses: dict[str, Tensor],
    outputs: StepOutputs,
    sample: PoseSample,
    graph: PoseGraphBatch,
    prefix: str,
    on_step: bool,
) -> None:
    log_kwargs = dict(batch_size=graph.B, on_step=on_step, on_epoch=True)
    module.log(f"{prefix}/loss", losses["loss"], prog_bar=True, **log_kwargs)
    for name, value in losses.items():
        if name != "loss":
            module.log(f"{prefix}/{name}", value, **log_kwargs)
    module.log(f"{prefix}/mpjpe_px", compute_mpjpe(outputs.pred_pixel, outputs.gt_pixel, sample.visibility), **log_kwargs)
    module.log(f"{prefix}/pck_20px", compute_pck(outputs.pred_pixel, outputs.gt_pixel, sample.visibility, threshold_px=20.0), **log_kwargs)


def _run_and_log_ap_eval(
    module: "PoseDiffusionModule",
    pred_cache: str,
    metric: "KeypointAPMetric",
    prefix: str,
    epoch: int | None = None,
    log_prog_bar: bool = False,
) -> None:
    for n in (module._ap_step_counts or [None]):
        suffix = f"@{n}" if n is not None else ""
        desc = f"{prefix} AP{suffix}" + (f" [epoch {epoch}]" if epoch is not None else "")
        module._run_ap_eval(pred_cache, metric, desc=desc, num_steps=n)
        results = metric.compute()
        for key, val in results.items():
            module.log(f"{prefix}/{key}{suffix}", val, prog_bar=(log_prog_bar and key == "crowdpose/AP"), sync_dist=True)


def joint_dropout(node_pos: Tensor, can_vis: Tensor, p: float) -> Tensor:
    # Root (index 0) is never dropped.
    B, N, K = can_vis.shape
    drop_mask = torch.zeros_like(can_vis, dtype=torch.bool)
    drop_mask[:, :, 1:] = torch.bernoulli(
        torch.full((B, N, K - 1), p, device=node_pos.device)
    ).bool()
    node_pos[drop_mask.reshape(-1)] = 0.0
    return can_vis.masked_fill(drop_mask, 0.0)


def joint_randomize(node_pos: Tensor, can_vis: Tensor, p: float) -> Tensor:
    # Simulates a badly-localised detector: joint present but wrong position.
    # Visibility is preserved so the loss still penalises the wrong position.
    B, N, K = can_vis.shape
    rand_mask = torch.zeros_like(can_vis, dtype=torch.bool)
    rand_mask[:, :, 1:] = torch.bernoulli(
        torch.full((B, N, K - 1), p, device=node_pos.device)
    ).bool()
    flat_mask = rand_mask.reshape(-1)
    node_pos[flat_mask] = torch.rand_like(node_pos[flat_mask]) * 2.0 - 1.0
    return can_vis  # visibility unchanged — joint is present but wrong


def pose_dropout(node_pos: Tensor, can_vis: Tensor, p: float) -> Tensor:
    B, N, K = can_vis.shape
    person_drop = torch.bernoulli(torch.full((B, N), p, device=node_pos.device)).bool()
    drop_mask = person_drop.unsqueeze(-1).expand(B, N, K)
    node_pos[drop_mask.reshape(-1)] = 0.0
    return can_vis.masked_fill(drop_mask, 0.0)


def keypoint_jitter(keypoints: Tensor, std_px: float) -> Tensor:
    return keypoints + torch.randn_like(keypoints) * std_px


def image_dropout(image_cond: Tensor, p: float) -> None:
    drop = torch.bernoulli(
        torch.full((image_cond.shape[0],), p, device=image_cond.device)
    ).bool()
    image_cond[drop] = 0.0


def compute_joint_loss(
    pred_pixel: Tensor,  # (B, N, K, 2)
    gt_pixel: Tensor,  # (B, N, K, 2)
    visibility: Tensor,  # (B, N, K)  0=absent 1=occluded 2=visible
) -> Tensor:
    visibility_weight = visibility.float()  # (B, N, K)
    per_joint_sq_err = ((pred_pixel - gt_pixel) ** 2).sum(dim=-1)  # (B, N, K)
    total_weight = visibility_weight.sum().clamp(min=1.0)
    return (per_joint_sq_err * visibility_weight).sum() / total_weight


def compute_bone_loss(
    pred_pixel: Tensor,  # (B, N, K, 2)
    gt_pixel: Tensor,  # (B, N, K, 2)
    visibility: Tensor,  # (B, N, K)  0/1/2
    edge_list: list[tuple[int, int]],
) -> Tensor:
    # Edge weight = vis[u] * vis[v]; absent endpoint → 0, both visible → 4.
    visibility_weight = visibility.float()
    loss = gt_pixel.new_zeros(())
    for u, v in edge_list:
        edge_weight = visibility_weight[:, :, u] * visibility_weight[:, :, v]  # (B, N)
        weight_sum = edge_weight.sum().clamp(min=1.0)
        pred_len = (pred_pixel[:, :, u] - pred_pixel[:, :, v]).norm(dim=-1)  # (B, N)
        gt_len = (gt_pixel[:, :, u] - gt_pixel[:, :, v]).norm(dim=-1)  # (B, N)
        loss = loss + ((pred_len - gt_len).pow(2) * edge_weight).sum() / weight_sum
    return loss / len(edge_list)


def _read_img_wh(img_paths: list[str], B: int, device: torch.device) -> Tensor:
    """Return (W, H) per image as a (B, 2) float tensor; fallback to (800, 800)."""
    wh: list[tuple[float, float]] = []
    for i in range(B):
        path = img_paths[i] if i < len(img_paths) else ""
        try:
            h, w = read_image_size(path)
            wh.append((float(w), float(h)))
        except Exception:
            wh.append((800.0, 800.0))
    return torch.tensor(wh, dtype=torch.float32, device=device)


class PoseDiffusionModule(pl.LightningModule):
    """PyTorch-Lightning wrapper for KeypointDiffusionModel."""

    def __init__(
        self,
        model: KeypointDiffusionModel,
        lr: float = 1e-3,
        weight_decay: float = 0.001,
        lr_T_max: int | None = None,
        lr_warmup_epochs: int = 2,
        lr_min_frac: float = 1e-2,
        cfg_guidance_scale: float = 1.0,
        val_ann_file: str | None = None,
        test_ann_file: str | None = None,
        val_pred_cache: str | None = None,
        test_pred_cache: str | None = None,
        num_inference_steps: int | None = None,
        ap_batch_size: int = 8,
        ap_eval_interval: int = 0,
        ap_step_counts: list[int] | None = None,
        start_inference_noise_frac: float = 0.4,
        raw_refiner: bool = False,
        disable_noise: bool = False,
        # DINO conditioning
        dino_model_name: str | None = None,
        dino_neighborhood_radius: int = 1,
        dino_cache_train: str | None = None,
        dino_cache_val: str | None = None,
        dino_cache_test: str | None = None,
        dino_conditioner_type: str = "simple",  # "simple" | "attention"
        dino_inner_dim: int = 256,
        dino_num_heads: int = 4,
        dino_conditioner: "DinoConditioner | None" = None,  # inject directly (e.g. for testing)
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model", "dino_conditioner"])
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_T_max = lr_T_max
        self.lr_warmup_epochs = lr_warmup_epochs
        self.lr_min_frac = lr_min_frac
        self.cfg_guidance_scale = cfg_guidance_scale
        self.num_inference_steps = num_inference_steps
        self._ap_batch_size = ap_batch_size
        self._ap_eval_interval = ap_eval_interval
        self._ap_step_counts = ap_step_counts
        self._start_inference_noise_frac = start_inference_noise_frac
        self._raw_refiner = raw_refiner
        self._disable_noise = disable_noise
        self._val_pred_cache = val_pred_cache
        self._test_pred_cache = test_pred_cache

        self._val_metric = KeypointAPMetric(val_ann_file) if val_ann_file else None
        self._test_metric = KeypointAPMetric(test_ann_file) if test_ann_file else None

        # Optional DINO conditioner (only when model.config.use_image_cond is True)
        self.dino_conditioner: DinoConditioner | None = dino_conditioner
        if self.dino_conditioner is None and dino_model_name is not None and model.config.use_image_cond:
            cache_files = [
                p
                for p in [dino_cache_train, dino_cache_val, dino_cache_test]
                if p is not None
            ]
            if dino_conditioner_type == "attention":
                self.dino_conditioner = AttentionDinoConditioner(
                    model_name=dino_model_name,
                    out_dim=model.config.dim_image_cond,
                    inner_dim=dino_inner_dim,
                    num_heads=dino_num_heads,
                    cache_paths=cache_files or None,
                )
            else:
                self.dino_conditioner = DinoConditioner(
                    model_name=dino_model_name,
                    out_dim=model.config.dim_image_cond,
                    cache_paths=cache_files or None,
                )

        self.global_dino_proj: torch.nn.Linear | None = None
        if model.config.use_global_image_cond:
            if self.dino_conditioner is None:
                raise ValueError(
                    "use_global_image_cond=True requires a DINO conditioner, "
                    "but dino_model_name was not provided (or use_image_cond=False). "
                    "Either pass dino_model_name or set use_global_image_cond=False."
                )
            dino_hidden = self.dino_conditioner.hidden_dim
            self.global_dino_proj = torch.nn.Linear(
                dino_hidden, model.config.dim_global_image_cond
            )


    def prepare_batch(self, batch: dict, augment: bool = False) -> tuple[PoseSample, PoseGraphBatch]:
        sample = prepare_sample(batch)
        if augment:
            self._augment_sample(sample)
        img_wh = _read_img_wh(sample.img_path, sample.B, sample.keypoints.device)
        graph = prepare_pose_graph_batch(sample, img_size=img_wh)

        if augment:
            self._augment_graph(graph)

        image_cond = self.get_local_spatial_condition(sample, graph)
        if image_cond is not None:
            graph.image_cond = image_cond

        global_image_cond = self.get_global_image_cond(sample, graph)
        if global_image_cond is not None:
            graph.global_image_cond = global_image_cond

        if augment:
            self._augment_image_cond(graph)

        return sample, graph

    def _augment_sample(self, sample: PoseSample) -> None:
        cfg = self.model.config
        if cfg.aug_jitter:
            sample.keypoints = keypoint_jitter(sample.keypoints, cfg.p_jitter_px)

    def _augment_graph(self, graph: PoseGraphBatch) -> None:
        cfg = self.model.config

        if cfg.aug_joint_drop:
            graph.canonical_visibility = joint_dropout(graph.node_pos, graph.canonical_visibility, cfg.p_joint_drop)
        if cfg.aug_pose_drop:
            graph.canonical_visibility = pose_dropout(graph.node_pos, graph.canonical_visibility, cfg.p_pose_drop)
        if cfg.aug_joint_rand:
            joint_randomize(graph.node_pos, graph.canonical_visibility, cfg.p_joint_rand)

    def _augment_image_cond(self, graph: PoseGraphBatch) -> None:
        if graph.image_cond is None or not self.model.config.aug_img_drop:
            return
        image_dropout(graph.image_cond, self.model.config.p_img_drop)

    def get_loss(self, sample: PoseSample, graph: PoseGraphBatch, outputs: StepOutputs) -> dict[str, Tensor]:
        # Pixel space keeps loss magnitudes independent of canonical parameterisation.
        cfg = self.model.config
        visibility = sample.visibility  # (B, N, 14)  0=absent 1=occluded 2=visible

        losses: dict[str, Tensor] = {}
        if cfg.use_joint_loss:
            losses["joint_loss"] = (
                compute_joint_loss(outputs.pred_pixel, outputs.gt_pixel, visibility) * cfg.w_joint_loss
            )
        if cfg.use_bone_loss:
            losses["bone_loss"] = (
                compute_bone_loss(outputs.pred_pixel, outputs.gt_pixel, visibility, CROWDPOSE_EDGE_LIST)
                * cfg.w_bone_loss
            )
        if not losses:
            raise ValueError(
                "All losses are disabled — enable at least one loss in KeypointDiffusionConfig"
            )
        losses["loss"] = sum(losses.values())
        return losses

    def get_global_image_cond(self, sample: PoseSample, graph: PoseGraphBatch) -> Tensor | None:  # (B*N*K, dim_global_image_cond)
        if self.global_dino_proj is None:
            return None
        pooled = self.dino_conditioner.fetch_global_feats(sample.image_id, self.device)  # (B, hidden_dim)
        projected = self.global_dino_proj(pooled)   # (B, dim_global_image_cond)
        return projected[graph.node_batch]          # (B*N*K, dim_global_image_cond)

    def get_local_spatial_condition(
        self,
        sample: PoseSample,
        graph: PoseGraphBatch,
        kpt_pixel: Tensor | None = None,  # (B, N, 14, 2) pixel-space; defaults to sample.keypoints
    ) -> Tensor | None:  # (B*N*14, dim_image_cond)
        if self.dino_conditioner is None:
            return None
        image_sizes = [read_image_size(p) for p in sample.img_path]
        img_hw = torch.tensor(image_sizes, dtype=torch.float32, device=self.device)
        kpt = kpt_pixel.to(self.device) if kpt_pixel is not None else sample.keypoints.to(self.device)
        if isinstance(self.dino_conditioner, AttentionDinoConditioner):
            joint_mask = (graph.canonical_visibility == 0).to(self.device)  # (B, N, K)
            feats = self.dino_conditioner(
                kpt, img_hw, image_ids=sample.image_id, joint_mask=joint_mask
            )  # (B, N, K, out_dim)
        else:
            feats = self.dino_conditioner(kpt, img_hw, image_ids=sample.image_id)  # (B, N, K, out_dim)
        return feats.reshape(graph.B * graph.N * graph.K, -1)

    def _ap_step(
        self,
        batch: dict,
        metric: KeypointAPMetric,
        num_steps: int | None = None,
    ) -> None:
        n_passes = num_steps if num_steps is not None else 1
        pred_pixel, pred_scores, image_ids = self._iterative_refine_batch(batch, n_passes=n_passes)
        bbox_scores = batch.get("bbox_scores")  # (B, N) or None
        metric.update(pred_pixel, pred_scores, image_ids, bbox_scores=bbox_scores)

    @torch.no_grad()
    def _iterative_refine_batch(
        self,
        batch: dict,
        n_passes: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run refinement n_passes times, halving noise_frac each pass.

        Pass i uses noise_frac = start_inference_noise_frac / 2^i (0-indexed),
        feeding each pass's pixel output as the next pass's input.
        """
        start_frac = self._start_inference_noise_frac
        current_batch = batch
        for i in range(n_passes):
            noise_frac = start_frac / (2 ** i)
            pred_pixel, pred_scores, image_ids = self._refine_prediction_batch(
                current_batch, noise_frac=noise_frac
            )
            if i < n_passes - 1:
                current_batch = {**batch, "pred_keypoints": pred_pixel.detach()}
        return pred_pixel, pred_scores, image_ids

    @torch.no_grad()
    def _refine_prediction_batch(
        self,
        batch: dict,
        num_steps: int | None = None,
        noise_frac: float | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        pred_kpts = batch["pred_keypoints"].to(self.device)  # (B, N, 14, 2) pixel
        pred_scores = batch["pred_keypoint_scores"].to(self.device)  # (B, N, 14)
        image_ids = batch["image_id"]  # (B,)

        batch["keypoints"] = pred_kpts
        # Detector confidence > 0.1 → treat as visible (matches the threshold used in training data)
        batch["visibility"] = (pred_scores > 0.1).float()

        # Convert predictions to canonical space (root-relative, normalised)
        sample, graph = self.prepare_batch(batch)

        T = self.model.config.num_train_steps
        frac = noise_frac if noise_frac is not None else self._start_inference_noise_frac
        t_start = max(1, round(T * frac))

        if self._raw_refiner:
            # No noise — feed clean detector output directly; single backward pass.
            x_init_noisy = graph.node_pos
            n_steps = 1
        else:
            if self._disable_noise:
                # Skip noise injection — start DDIM from clean detector output.
                x_init_noisy = graph.node_pos
            else:
                # Add a small amount of noise so the model sees in-distribution inputs
                # while keeping predictions close to the detector output.
                noise_timestep = torch.full(
                    (graph.B,), t_start, dtype=torch.long, device=pred_kpts.device
                )
                x_init_noisy, _ = self.model.add_noise(
                    node_pos=graph.node_pos,
                    timestep=noise_timestep,
                    batch_idx=graph.node_batch,
                )  # (B*N*K, 2)

            default_steps = (
                self.num_inference_steps
                if self.num_inference_steps is not None
                else t_start
            )
            n_steps = num_steps if num_steps is not None else default_steps

        nodes_canonical_flat = self.model.ddim_sample(
            edge_index=graph.edge_index,
            kpt_type_idx=graph.flat_kpt_type_idx,
            batch_idx=graph.node_batch,
            num_nodes=graph.num_nodes,
            num_inference_steps=n_steps,
            start_t=t_start,
            device=pred_kpts.device,
            x_init=x_init_noisy,
            image_cond=graph.image_cond,
            global_image_cond=graph.global_image_cond,
            guidance_scale=self.cfg_guidance_scale,
        )  # (B*N*K, 2)

        # Convert back to pixel space using per-image (W, H)
        nodes_canonical = nodes_canonical_flat.view(graph.B, graph.N, graph.K, 2)
        joint_pred_pixel = canonical_to_pixel(nodes_canonical, graph.img_wh)

        return joint_pred_pixel, pred_scores, image_ids

    def _run_ap_eval(
        self,
        pred_cache: str,
        metric: KeypointAPMetric,
        desc: str = "AP eval",
        num_steps: int | None = None,
    ) -> None:
        from tqdm import tqdm
        from src.data import PredictionDataset

        dataset = PredictionDataset.from_path(pred_cache)
        loader = dataset.create_dataloader(
            batch_size=self._ap_batch_size,
            shuffle=False,
            num_workers=0,
        )
        metric.reset()
        for batch in tqdm(loader, desc=desc, leave=False, unit="batch"):
            self._ap_step(batch, metric, num_steps=num_steps)

    def _step(
        self,
        batch: dict,
        stage: str,
        generator: torch.Generator | None = None,
    ) -> tuple[dict[str, Tensor], StepOutputs, PoseSample, PoseGraphBatch]:
        sample, graph = self.prepare_batch(batch, augment=(stage == "train"))

        timestep = torch.randint(
            1,
            self.model.config.num_train_steps + 1,
            (graph.B,),
            device=graph.node_pos.device,
        )  # (B,)  — 1-indexed to match sigma schedule

        # Add noise before deriving conditioning — prevents leaking clean GT positions
        # to the conditioner (DINO patch lookup / CNN crop centre).
        noisy_node_pos, _ = self.model.add_noise(
            node_pos=graph.node_pos,
            timestep=timestep,
            batch_idx=graph.node_batch,
            generator=generator,
        )

        if graph.image_cond is not None:
            # Convert noisy canonical positions back to pixel space.
            noisy_kpt_pixel = (
                (noisy_node_pos.view(graph.B, graph.N, graph.K, 2) + 1) / 2
                * graph.img_wh[:, None, None, :]
            )  # (B, N, 14, 2)
            graph.image_cond = self.get_local_spatial_condition(
                sample, graph, kpt_pixel=noisy_kpt_pixel
            )
            if stage == "train":
                self._augment_image_cond(graph)

        outputs = denoise_and_decode(
            self.model, sample, graph, timestep,
            generator=generator,
            noisy_node_pos=noisy_node_pos,
        )
        losses = self.get_loss(sample, graph, outputs)
        return losses, outputs, sample, graph

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        losses, _, sample, graph = self._step(batch, stage="train")
        log_kwargs = dict(batch_size=graph.B, on_step=True, on_epoch=True)
        self.log("train/loss", losses["loss"], prog_bar=True, **log_kwargs)
        for name, value in losses.items():
            if name != "loss":
                self.log(f"train/{name}", value, **log_kwargs)
        return losses["loss"]

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        gen = torch.Generator(device=self.device).manual_seed(batch_idx)
        losses, outputs, sample, graph = self._step(batch, stage="val", generator=gen)
        _log_eval_step(self, losses, outputs, sample, graph, prefix="val", on_step=False)

    def test_step(self, batch: dict, batch_idx: int) -> None:
        gen = torch.Generator(device=self.device).manual_seed(batch_idx)
        losses, outputs, sample, graph = self._step(batch, stage="test", generator=gen)
        _log_eval_step(self, losses, outputs, sample, graph, prefix="test", on_step=False)

    def on_validation_epoch_end(self) -> None:
        if self._val_metric is None or self._val_pred_cache is None:
            return
        if self.trainer.sanity_checking:
            return
        if self._ap_eval_interval <= 0:
            return
        if (self.current_epoch + 1) % self._ap_eval_interval != 0:
            return
        _run_and_log_ap_eval(self, self._val_pred_cache, self._val_metric, prefix="ap-val", epoch=self.current_epoch, log_prog_bar=True)

    def on_test_epoch_end(self) -> None:
        if self._test_metric is None or self._test_pred_cache is None:
            return
        _run_and_log_ap_eval(self, self._test_pred_cache, self._test_metric, prefix="ap-test")

    def configure_optimizers(self):
        # Weight matrices get weight decay; 1-D tensors (biases, norms, embeddings) do not.
        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or name.endswith(".bias"):
                no_decay.append(param)
            else:
                decay.append(param)
        param_groups = [
            {"params": decay,    "weight_decay": self.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(param_groups, lr=self.lr)

        schedulers: list[torch.optim.lr_scheduler.LRScheduler] = []
        milestones: list[int] = []

        if self.lr_warmup_epochs > 0:
            schedulers.append(
                torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=1e-8,
                    end_factor=1.0,
                    total_iters=self.lr_warmup_epochs,
                )
            )
            milestones.append(self.lr_warmup_epochs)

        if self.lr_T_max is not None:
            cosine_epochs = max(1, self.lr_T_max - self.lr_warmup_epochs)
            schedulers.append(
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=cosine_epochs, eta_min=self.lr * self.lr_min_frac
                )
            )

        if not schedulers:
            return optimizer

        if len(schedulers) == 1:
            scheduler = schedulers[0]
        else:
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=schedulers, milestones=milestones
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

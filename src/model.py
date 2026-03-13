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
from PIL import Image
from torch import Tensor

from src.data import (
    CROWDPOSE_EDGE_LIST,
    CROWDPOSE_JOINT_NAMES,
    prepare_pose_graph_batch,
    read_image_pil,
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

    # Global image conditioning (TODO[5]): pooled DINO as a per-image global signal
    # combined with the timestep embedding before being fed into global_cond_mlp.
    use_global_image_cond: bool = False
    dim_global_image_cond: int = 128  # projected pooled-DINO dim

    # CNN patch conditioning (TODO[3])  — alternative local visual source to DINO
    use_cnn_cond: bool = False
    cnn_patch_size: int = 64  # height and width of the crop fed to the CNN

    # Training augmentations
    aug_joint_drop: bool = True
    p_joint_drop: float = (
        0.1  # probability of dropping each non-root joint independently
    )
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


class DinoConditioner(torch.nn.Module):
    """Per-keypoint local DINO patch features.

    Gathers the (2r+1)×(2r+1) patch neighbourhood for each keypoint and
    mean-pools it into a single projected embedding.  Backbone is frozen;
    only the linear projection is trained.
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        out_dim: int = 128,
        patch_size: int = 16,
        image_size: int = 224,
        neighborhood_radius: int = 1,
        cache_paths: list[str | Path] | None = None,
    ) -> None:
        super().__init__()
        from transformers import AutoConfig, AutoImageProcessor, AutoModel

        self.patch_size = patch_size
        self.image_size = image_size
        self.neighborhood_radius = neighborhood_radius
        self.grid_size = image_size // patch_size  # e.g. 14 for ViT-S/16 @ 224

        _cfg = AutoConfig.from_pretrained(model_name)
        n_registers = getattr(_cfg, "num_register_tokens", 0)
        self.n_prefix_tokens: int = 1 + n_registers
        self.hidden_dim: int = _cfg.hidden_size

        # Only load the full backbone when no caches are provided (live-inference mode).
        # Training always uses precomputed caches, so backbone weights are never needed.
        if cache_paths:
            self.backbone = None
            self.processor = None
        else:
            self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
            self.backbone = AutoModel.from_pretrained(model_name)
            # Freeze the backbone – only the projection is trained
            for param in self.backbone.parameters():
                param.requires_grad_(False)

        self.proj = torch.nn.Linear(self.hidden_dim, out_dim)

        # Precomputed patch cache: {image_id: Tensor(G*G, hidden_dim)}
        # Populated by load_cache(); when set, the backbone is never called.
        self._patch_cache: dict[int, Tensor] | None = None
        if cache_paths:
            self.load_cache(*cache_paths)

    def load_cache(self, *paths: str | Path) -> None:
        """Merge one or more ``{image_id → patch_features}`` cache files into the lookup table."""
        merged: dict = {} if self._patch_cache is None else dict(self._patch_cache)
        for path in paths:
            data = torch.load(Path(path), map_location="cpu", weights_only=False)
            merged.update(data)
        self._patch_cache = merged
        print(
            f"DinoConditioner: patch cache has {len(merged)} images ({len(paths)} file(s) loaded)"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _extract_patch_features(self, pil_images: list[Image.Image]) -> Tensor:
        """Run DINO backbone; return patch features (B, G*G, hidden_dim) with CLS and register tokens dropped."""
        if self.backbone is None:
            raise RuntimeError(
                "DinoConditioner was created in cache-only mode (backbone not loaded). "
                "Cannot run live inference without a backbone."
            )
        inputs = self.processor(images=pil_images, return_tensors="pt")
        inputs = {
            k: v.to(next(self.backbone.parameters()).device) for k, v in inputs.items()
        }
        outputs = self.backbone(**inputs)
        return outputs.last_hidden_state[:, self.n_prefix_tokens:]  # drop CLS + registers  (B, G*G, D)

    def _gather_neighborhood(
        self,
        patch_feats: Tensor,  # (B, G*G, D)
        kpt_pixel: Tensor,    # (B, N, K, 2)  pixel-space (x, y)
        img_hw: Tensor,       # (B, 2)  original (H, W)
    ) -> Tensor:
        """Return the single closest DINO patch for each keypoint → (B, N, K, D)."""
        B, N, K, _ = kpt_pixel.shape
        G, D = self.grid_size, patch_feats.shape[-1]
        device = patch_feats.device

        # Scale pixel coords → DINO input space, then find the containing patch
        img_h = img_hw[:, 0].float().view(B, 1, 1)
        img_w = img_hw[:, 1].float().view(B, 1, 1)
        x_scaled = kpt_pixel[..., 0].to(device) * (self.image_size / img_w)
        y_scaled = kpt_pixel[..., 1].to(device) * (self.image_size / img_h)
        cx = (x_scaled / self.patch_size).long().clamp(0, G - 1)  # (B, N, K) col
        cy = (y_scaled / self.patch_size).long().clamp(0, G - 1)  # (B, N, K) row

        # Index of the single containing patch for every keypoint: (B, N*K)
        patch_idx = (cy * G + cx).reshape(B, N * K)
        gathered = patch_feats.gather(1, patch_idx.unsqueeze(-1).expand(-1, -1, D))
        return gathered.view(B, N, K, D)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        kpt_pixel: Tensor,  # (B, N, K, 2) pixel-space (x, y)
        img_hw: Tensor,  # (B, 2) original image (H, W)
        image_ids: list[int] | None = None,
        pil_images: list[Image.Image] | None = None,
    ) -> Tensor:
        """Return per-keypoint DINO embeddings (B, N, K, out_dim).

        Provide image_ids when using a precomputed cache, or pil_images for live inference.
        """
        if self._patch_cache is not None:
            if image_ids is None:
                raise ValueError(
                    "image_ids must be provided when using a precomputed cache"
                )
            patch_feats = torch.stack([self._patch_cache[iid] for iid in image_ids]).to(
                kpt_pixel.device
            )
        else:
            if pil_images is None:
                raise ValueError("pil_images must be provided when no cache is loaded")
            patch_feats = self._extract_patch_features(pil_images)

        return self.proj(self._gather_neighborhood(patch_feats, kpt_pixel, img_hw))


class PatchCNNConditioner(torch.nn.Module):
    """Per-keypoint patch CNN conditioner (TODO[3]).

    For each joint, extracts a ``patch_size \u00d7 patch_size`` crop from the
    full-resolution image centred on the joint pixel coordinate.  The crop is
    augmented with two extra channels that carry the *relative* spatial
    coordinate (normalised to [-1, 1] within the crop), giving a 5-channel
    input (RGB + coord_x + coord_y).

    CNN architecture (exact from TODO[3]):
        Conv 5 \u2192 32  (3\u00d73, pad 1)
        Conv 32 \u2192 64  (3\u00d73, pad 1, stride 2)
        Conv 64 \u2192 128 (3\u00d73, pad 1, stride 2)
        Conv 128 \u2192 128 (3\u00d73, pad 1, stride 2)
        GlobalAvgPool
        Linear \u2192 out_dim
        LayerNorm

    Args:
        out_dim: Output embedding dimension per keypoint.
        patch_size: Spatial size of the extracted crop (square).
    """

    def __init__(self, out_dim: int = 128, patch_size: int = 64) -> None:
        super().__init__()
        self.patch_size = patch_size

        d = 64
        self.backbone_dim: int = d  # output channels of the CNN backbone (before head)
        self.cnn = torch.nn.Sequential(
            # 5 × P × P  →  32 × (P/2) × (P/2)
            torch.nn.Conv2d(5, d//4, 3, stride=2, padding=1),
            torch.nn.BatchNorm2d(d//4),
            torch.nn.SiLU(),
            # 32 × (P/2) × (P/2)  →  64 × (P/4) × (P/4)
            torch.nn.Conv2d(d//4, d//2, 3, stride=2, padding=1),
            torch.nn.BatchNorm2d(d//2),
            torch.nn.SiLU(),
            # 64 × (P/4) × (P/4)  →  128 × (P/8) × (P/8)
            torch.nn.Conv2d(d//2, d, 3, stride=2, padding=1),
            torch.nn.BatchNorm2d(d),
            torch.nn.SiLU(),
            # → 128 × 1 × 1
            torch.nn.AdaptiveAvgPool2d(1),
        )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(d, out_dim),
            torch.nn.LayerNorm(out_dim),
        )

        # Fixed (2, P, P) spatial coordinate grid in [-1, 1]: channel 0 = x, channel 1 = y.
        # These are appended as channels 4-5 of every extracted patch so the CNN
        # knows where each pixel sits relative to the joint centre.
        P = patch_size
        coords_1d = torch.linspace(-1.0, 1.0, P)
        grid_y, grid_x = torch.meshgrid(coords_1d, coords_1d, indexing="ij")  # (P, P) each
        self.register_buffer("_spatial_grid", torch.stack([grid_x, grid_y], dim=0))  # (2, P, P)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_patches(
        self,
        image: Tensor,     # (3, H, W) float in [0, 1]
        kpt_pixel: Tensor, # (N, K, 2) pixel-space (x, y)
    ) -> Tensor:           # (N*K, 5, P, P)
        """Extract 5-channel (RGB + spatial) patches for all joints of one image.

        Uses F.pad + integer indexing instead of expand + grid_sample to avoid
        creating zero-stride (non-contiguous) tensors that trigger a CUDA
        allocator assertion (!block->expandable_segment_) in PyTorch ≥ 2.1.
        """
        N, K = kpt_pixel.shape[:2]
        P = self.patch_size
        half = P // 2
        M = N * K
        _, H, W = image.shape
        device = image.device

        # Pad image by half on every side so any crop stays in bounds.
        # Use replicate (not reflect) — reflect requires padding < dimension,
        # which fails for images smaller than patch_size.
        padded = F.pad(image.unsqueeze(0), (half, half, half, half), mode="replicate").squeeze(0)
        # (3, H + 2*half, W + 2*half)

        # Integer keypoint coords clamped to original image bounds.
        # These are the top-left corners of the crops in padded space.
        px = kpt_pixel[..., 0].long().clamp(0, W - 1).reshape(M)  # (M,)
        py = kpt_pixel[..., 1].long().clamp(0, H - 1).reshape(M)  # (M,)

        # Build index arrays: row_idxs[m, i] = py[m] + i  (M, P)
        offsets = torch.arange(P, device=device)                    # (P,)
        row_idxs = py.unsqueeze(1) + offsets.unsqueeze(0)           # (M, P)
        col_idxs = px.unsqueeze(1) + offsets.unsqueeze(0)           # (M, P)

        # Gather all patches in one advanced-index call.
        # padded[:, row_idxs[m, i, :], col_idxs[m, :, j]]
        #   row shape (M, P, 1), col shape (M, 1, P) → broadcast (M, P, P)
        #   result shape: (3, M, P, P)
        patches_rgb = padded[:, row_idxs.unsqueeze(2), col_idxs.unsqueeze(1)]
        patches_rgb = patches_rgb.permute(1, 0, 2, 3).contiguous()  # (M, 3, P, P)

        # Pre-allocate output and fill in-place to avoid any torch.cat or repeat
        # allocation — those allocations can hit the CUDA expandable-segment
        # assertion (!block->expandable_segment_) in PyTorch ≥ 2.1.
        out = torch.empty(M, 5, P, P, device=device)
        out[:, :3] = patches_rgb
        out[:, 3:] = self._spatial_grid  # (2, P, P) broadcasts to (M, 2, P, P)
        return out

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        images: list[Tensor],  # B tensors, each (3, H_i, W_i) float in [0, 1]
        kpt_pixel: Tensor,     # (B, N, K, 2) pixel-space (x, y)
    ) -> Tensor:               # (B, N, K, out_dim)
        """Encode per-keypoint CNN embeddings for a batch."""
        B, N, K = kpt_pixel.shape[:3]
        outputs = []
        for b in range(B):
            patches = self._extract_patches(images[b], kpt_pixel[b])  # (N*K, 5, P, P)
            feats = self.cnn(patches).mean(dim=(-2, -1))               # (N*K, 128) global avg pool
            outputs.append(self.head(feats).view(N, K, -1))            # (N, K, out_dim)
        return torch.stack(outputs, dim=0)  # (B, N, K, out_dim)


class FiLMLayer(torch.nn.Module):
    """Feature-wise Linear Modulation: h → (1 + scale) * h + shift.

    Conditions node states on a global vector ``g`` (e.g. the processed
    timestep embedding).  The affine parameters are produced by a single
    linear projection of ``g``.
    """

    def __init__(self, cond_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(cond_dim, 2 * feature_dim)

    def forward(self, h: Tensor, g: Tensor) -> Tensor:
        # h: (M, feature_dim)   g: (M, cond_dim)
        params = self.proj(g)                        # (M, 2 * feature_dim)
        scale, shift = params.chunk(2, dim=-1)
        return (1.0 + scale) * h + shift


class PoseGraphLayer(MessagePassing):
    """One graph denoising layer implementing the 5-step design from the TODO[2] spec:

    5.1  FiLM — modulate node states h with global conditioning g.
    5.2  Q = W_q h,  K = W_k h,  V_state = W_v h
    5.3  For each edge (j→i):  m_ij = MLP_msg([V_state_j, e_ij, v_j])
    5.4  score_ij = dot(q_i, k_j)/√D + w_e^T e_ij;  α = softmax over neighbours
    5.5  Aggregate: agg_i = Σ_j α_ij · m_ij
    5.6  Node update: h ← h + MLP_update([h_i, agg_i, v_i])

    When ``use_message_passing=False`` steps 5.2–5.5 are skipped and the
    update MLP receives only ``[h_i, v_i]``.
    """

    def __init__(
        self,
        dim: int,
        dim_edge: int,
        dim_vis: int,
        dim_global: int,
        num_heads: int = 4,
        use_message_passing: bool = True,
    ) -> None:
        super().__init__(aggr="add")  # weighted sum handled inside message()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        self.use_message_passing = use_message_passing

        if use_message_passing:
            # Step 1: projections for query / key / value-state
            self.W_q = torch.nn.Linear(dim, dim, bias=False)
            self.W_k = torch.nn.Linear(dim, dim, bias=False)
            self.W_v = torch.nn.Linear(dim, dim, bias=False)

            # Step 2: message content MLP — [V_state_j, e_ij, v_j] → message
            msg_in = dim + dim_edge + dim_vis
            self.msg_mlp = torch.nn.Sequential(
                torch.nn.LayerNorm(msg_in),
                torch.nn.Linear(msg_in, dim * 2),
                torch.nn.GELU(),
                torch.nn.Linear(dim * 2, dim),
            )

            # Step 3: edge geometry → per-head attention bias
            self.edge_bias = torch.nn.Linear(dim_edge, num_heads, bias=False)

        # Step 5: node update MLP — input size differs by whether we have agg
        update_in = dim + (dim if use_message_passing else 0) + dim_vis
        self.update_mlp = torch.nn.Sequential(
            torch.nn.LayerNorm(update_in),
            torch.nn.Linear(update_in, dim * 2),
            torch.nn.GELU(),
            torch.nn.Linear(dim * 2, dim),
        )

        # FiLM conditioning: applied once at step 5.1 before attention
        self.film = FiLMLayer(dim_global, dim)

    # ------------------------------------------------------------------
    # MessagePassing hook
    # ------------------------------------------------------------------

    def message(
        self,
        q_i: Tensor,       # (E, dim) — destination query
        k_j: Tensor,       # (E, dim) — source key
        v_state_j: Tensor, # (E, dim) — source value-state
    ) -> Tensor:           # returns (E, dim) attention-weighted messages
        E = q_i.shape[0]
        H, D = self.num_heads, self.head_dim

        e_ij  = self._edge_feats  # (E, dim_edge) — stashed before propagate
        vis_j = self._vis_j       # (E, dim_vis)  — source visual features

        # Step 2: generate message content
        msg = self.msg_mlp(torch.cat([v_state_j, e_ij, vis_j], dim=-1))  # (E, dim)

        # Step 3: compute attention score (head-wise dot + edge geometry bias)
        q_h = q_i.view(E, H, D)
        k_h = k_j.view(E, H, D)
        dot   = (q_h * k_h).sum(-1) / math.sqrt(D)  # (E, H)
        score = dot + self.edge_bias(e_ij)            # (E, H)
        alpha = softmax(score, index=self._edge_dst, num_nodes=self._num_nodes)  # (E, H)

        # Step 4: weight messages by attention
        msg_h    = msg.view(E, H, D)
        weighted = (msg_h * alpha.unsqueeze(-1)).view(E, self.dim)  # (E, dim)
        return weighted

    # ------------------------------------------------------------------
    # Main forward
    # ------------------------------------------------------------------

    def forward(
        self,
        h: Tensor,           # (M, dim)      M = B*N*K
        edge_index: Tensor,  # (2, E)
        e: Tensor,           # (E, dim_edge) per-edge relative-position features
        v: Tensor,           # (M, dim_vis)  per-node local visual features
        g: Tensor,           # (M, dim_global) global conditioning (timestep …)
    ) -> Tensor:
        # 5.1 FiLM: inject global conditioning before attention
        h = self.film(h, g)

        if self.use_message_passing:
            # 5.2 Q / K / V projections
            q       = self.W_q(h)
            k       = self.W_k(h)
            v_state = self.W_v(h)

            # Stash per-edge data for message() — MessagePassing doesn't
            # support arbitrary per-edge tensors natively, so we stash them.
            src, dst = edge_index[0], edge_index[1]
            self._edge_feats = e          # (E, dim_edge)
            self._vis_j      = v[src]     # (E, dim_vis)  — source visual
            self._edge_dst   = dst        # (E,)          — for softmax norm
            self._num_nodes  = h.shape[0] # M

            # 5.3–5.5 generate, score, and aggregate conditioned messages
            agg = self.propagate(edge_index, q=q, k=k, v_state=v_state)  # (M, dim)

            # 5.6 residual node update with aggregated messages + local vis
            h = h + self.update_mlp(torch.cat([h, agg, v], dim=-1))
        else:
            # No graph — update from local visual features only
            h = h + self.update_mlp(torch.cat([h, v], dim=-1))

        return h


class PoseGraphDecoder(torch.nn.Module):
    """Stacked :class:`PoseGraphLayer` decoder.

    Accepts a node state tensor already at ``hidden_dim`` (produced by
    ``KeypointDiffusionModel.backward_process`` as ``h_i = LN(pos_emb + id_emb)``),
    runs ``num_layers`` conditional message-passing layers, then predicts a
    2-D coordinate correction with a final LayerNorm + linear head.
    """

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
            )
            for _ in range(num_layers)
        ])
        self.output_proj = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, out_dim),
        )

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
        num_keypoints: int = 15,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_keypoints = num_keypoints

        # --- Step 1: Node state initialisation ---
        # pos_emb_i = MLP_pos(x_i)  (2 → dim_pos → dim_hidden)
        self.position_embedding = torch.nn.Sequential(
            torch.nn.Linear(2, config.dim_pos),
            torch.nn.GELU(),
            torch.nn.Linear(config.dim_pos, config.dim_hidden),
        )
        # id_emb_i = Embedding(id_i)  (→ dim_hidden)
        self.node_idx_embedding = torch.nn.Embedding(num_keypoints, config.dim_hidden)
        # h_i = LayerNorm(pos_emb_i + id_emb_i)
        self.node_init_norm = torch.nn.LayerNorm(config.dim_hidden)

        # --- Step 3: Global conditioning ---
        self.timestep_embedding = TimestepSinusoidalEmbedding(config.dim_time)
        # MLP_global: [t_emb (, pooled_dino)] → g
        global_in = config.dim_time + (
            config.dim_global_image_cond if config.use_global_image_cond else 0
        )
        self.global_cond_mlp = torch.nn.Sequential(
            torch.nn.Linear(global_in, config.dim_time),
            torch.nn.GELU(),
            torch.nn.Linear(config.dim_time, config.dim_time),
        )

        # --- Step 4: Edge feature encoder: MLP_edge(delta_ij)  (2 → dim_edge) ---
        self.edge_encoder = torch.nn.Sequential(
            torch.nn.Linear(2, config.dim_edge),
            torch.nn.GELU(),
            torch.nn.Linear(config.dim_edge, config.dim_edge),
        )

        # Step 2 visual features (v_i) are injected per-layer via image_cond.
        # Zeros are passed when use_image_cond=False; dim is always dim_image_cond.
        dim_vis = config.dim_image_cond

        self.decoder = PoseGraphDecoder(
            dim_edge=config.dim_edge,
            dim_vis=dim_vis,
            dim_global=config.dim_time,
            hidden_dim=config.dim_hidden,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            use_message_passing=config.use_message_passing,
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

        # Step 1: h_i = LayerNorm(MLP_pos(x_i) + Embedding(id_i))
        pos_emb = self.position_embedding(noisy_node_pos)  # (M, dim_hidden)
        idx_emb = self.node_idx_embedding(kpt_type_idx)    # (M, dim_hidden)
        h = self.node_init_norm(pos_emb + idx_emb)         # (M, dim_hidden)

        # 2. Edge features: e_ij = MLP_edge(x_j − x_i)  (relative joint position)
        src, dst = edge_index[0], edge_index[1]
        delta_ij = noisy_node_pos[src] - noisy_node_pos[dst]  # (E, 2)  x_j − x_i
        e = self.edge_encoder(delta_ij)                        # (E, dim_edge)

        # 3. Local visual features: v_i from image conditioning (zeros if disabled)
        if image_cond is not None:
            v = image_cond  # (M, dim_image_cond) — projected DINO patch features
        else:
            v = torch.zeros(
                M, self.config.dim_image_cond,
                device=noisy_node_pos.device, dtype=noisy_node_pos.dtype,
            )

        # 4. Global conditioning: g = MLP_global([t_emb, pooled_dino?])
        t_emb = self.timestep_embedding(timestep)[batch_idx]  # (M, dim_time)
        if global_image_cond is not None:
            t_emb = torch.cat([t_emb, global_image_cond], dim=-1)  # (M, dim_time + dim_global)
        g = self.global_cond_mlp(t_emb)                           # (M, dim_time)

        # Steps 5–6: stacked graph layers + coordinate prediction head
        return self.decoder(h, edge_index, e, v, g)        # (M, 2)

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

        # Build decreasing timestep schedule: [t_start, ..., 1]
        timesteps = torch.linspace(
            t_start, 1, num_steps, dtype=torch.long, device=device
        )

        # Start from provided initial estimate or pure Gaussian noise
        if x_init is not None:
            x = x_init.to(device)
        else:
            x = torch.randn(num_nodes, 2, device=device)

        prev_timestep_list = torch.cat(
            [timesteps[1:], timesteps.new_zeros(1)]
        )  # trailing 0 = clean
        for timestep, prev_timestep in zip(timesteps, prev_timestep_list):
            timestep_batch = timestep.expand(batch_idx.max().item() + 1)  # (B,)

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


# ---------------------------------------------------------------------------
# Standalone augmentation functions  (pure, stateless — easy to unit-test)
# ---------------------------------------------------------------------------


def joint_dropout(node_pos: Tensor, can_vis: Tensor, p: float) -> Tensor:
    """Zero random non-root joints in-place; return updated visibility (dropped joints → 0)."""
    B, N, K = can_vis.shape
    # Root (index 0) is never dropped; sample only the K-1 remaining joints
    drop_mask = torch.zeros_like(can_vis, dtype=torch.bool)
    drop_mask[:, :, 1:] = torch.bernoulli(
        torch.full((B, N, K - 1), p, device=node_pos.device)
    ).bool()
    node_pos[drop_mask.reshape(-1)] = 0.0
    return can_vis.masked_fill(drop_mask, 0.0)


def joint_randomize(node_pos: Tensor, can_vis: Tensor, p: float) -> Tensor:
    """Replace random non-root joints in-place with uniform positions in [-1, 1]².

    Simulates a badly-localised detector output (present joint, wrong position).
    Visibility is intentionally preserved so the loss still penalises wrong positions.
    Root (index 0) is never touched.
    """
    B, N, K = can_vis.shape
    rand_mask = torch.zeros_like(can_vis, dtype=torch.bool)
    rand_mask[:, :, 1:] = torch.bernoulli(
        torch.full((B, N, K - 1), p, device=node_pos.device)
    ).bool()
    flat_mask = rand_mask.reshape(-1)
    node_pos[flat_mask] = torch.rand_like(node_pos[flat_mask]) * 2.0 - 1.0
    return can_vis  # visibility unchanged — joint is present but wrong


def pose_dropout(node_pos: Tensor, can_vis: Tensor, p: float) -> Tensor:
    """Zero all joints of random persons in-place; return updated visibility (dropped → 0)."""
    B, N, K = can_vis.shape
    person_drop = torch.bernoulli(torch.full((B, N), p, device=node_pos.device)).bool()
    drop_mask = person_drop.unsqueeze(-1).expand(B, N, K)
    node_pos[drop_mask.reshape(-1)] = 0.0
    return can_vis.masked_fill(drop_mask, 0.0)


def keypoint_jitter(keypoints: Tensor, std_px: float) -> Tensor:
    """Return new keypoints with zero-mean Gaussian noise (std=std_px pixels) added."""
    return keypoints + torch.randn_like(keypoints) * std_px


def image_dropout(image_cond: Tensor, p: float) -> None:
    """Zero random rows of image_cond (B*N*K, D) in-place with probability p."""
    drop = torch.bernoulli(
        torch.full((image_cond.shape[0],), p, device=image_cond.device)
    ).bool()
    image_cond[drop] = 0.0


def compute_joint_loss(
    pred_pixel: Tensor,  # (B, N, K, 2)
    gt_pixel: Tensor,  # (B, N, K, 2)
    visibility: Tensor,  # (B, N, K)  0=absent 1=occluded 2=visible
) -> Tensor:
    """Visibility-weighted per-joint MSE (absent→0, occluded→1, visible→2 weight)."""
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
    """Visibility-weighted bone-length MSE averaged over edges.

    Edge weight = vis[u] * vis[v]; absent endpoint → 0, both visible → 4.
    """
    visibility_weight = visibility.float()
    loss = gt_pixel.new_zeros(())
    for u, v in edge_list:
        edge_weight = visibility_weight[:, :, u] * visibility_weight[:, :, v]  # (B, N)
        weight_sum = edge_weight.sum().clamp(min=1.0)
        pred_len = (pred_pixel[:, :, u] - pred_pixel[:, :, v]).norm(dim=-1)  # (B, N)
        gt_len = (gt_pixel[:, :, u] - gt_pixel[:, :, v]).norm(dim=-1)  # (B, N)
        loss = loss + ((pred_len - gt_len).pow(2) * edge_weight).sum() / weight_sum
    return loss / len(edge_list)


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
        inference_noise_frac: float = 0.1,
        # DINO conditioning
        dino_model_name: str | None = None,
        dino_neighborhood_radius: int = 1,
        dino_cache_train: str | None = None,
        dino_cache_val: str | None = None,
        dino_cache_test: str | None = None,
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
        self._inference_noise_frac = inference_noise_frac
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
            self.dino_conditioner = DinoConditioner(
                model_name=dino_model_name,
                out_dim=model.config.dim_image_cond,
                neighborhood_radius=dino_neighborhood_radius,
                cache_paths=cache_files or None,
            )

        # Optional CNN patch conditioner (TODO[3]): local pixel-level evidence per joint.
        self.cnn_conditioner: PatchCNNConditioner | None = None
        if model.config.use_cnn_cond:
            self.cnn_conditioner = PatchCNNConditioner(
                out_dim=model.config.dim_image_cond,
                patch_size=model.config.cnn_patch_size,
            )

        # When both conditioners are active, raw features are projected + fused in
        # get_local_spatial_condition via these three linears.
        self.dino_proj: torch.nn.Linear | None = None
        self.cnn_proj: torch.nn.Linear | None = None
        self.fuse_proj: torch.nn.Linear | None = None
        if self.dino_conditioner is not None and self.cnn_conditioner is not None:
            dino_raw_dim = self.dino_conditioner.hidden_dim
            dim_cond = model.config.dim_image_cond
            self.dino_proj = torch.nn.Linear(dino_raw_dim, dim_cond)
            self.cnn_proj = torch.nn.Linear(self.cnn_conditioner.backbone_dim, dim_cond)
            self.fuse_proj = torch.nn.Linear(2 * dim_cond, dim_cond)

        # Global image conditioning (TODO[5]): project mean-pooled DINO patch tokens
        # to dim_global_image_cond; combined with timestep in global_cond_mlp.
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

    # ------------------------------------------------------------------
    # Batch preparation (graph build + optional DINO conditioning)
    # ------------------------------------------------------------------

    def prepare_batch(self, batch: dict, augment: bool = False) -> dict:
        """Build graph, optionally augment (jitter before conditioning so coordinates match), add image conditioning."""
        prepare_pose_graph_batch(batch)

        if augment:
            self._augment_graph(batch)

        image_cond = self.get_local_spatial_condition(batch)
        if image_cond is not None:
            batch["image_cond"] = image_cond

        global_image_cond = self.get_global_image_cond(batch)
        if global_image_cond is not None:
            batch["global_image_cond"] = global_image_cond

        if augment:
            self._augment_image_cond(batch)

        return batch

    # ------------------------------------------------------------------
    # Augmentation helpers
    # ------------------------------------------------------------------

    def _augment_graph(self, batch: dict) -> None:
        """Joint/pose dropout and pixel jitter (runs before DINO so jittered coords are used)."""
        cfg = self.model.config
        node_pos = batch["node_pos"]  # (B*N*K, 2)  canonical
        can_vis = batch["canonical_visibility"]  # (B, N, K)

        if cfg.aug_joint_drop:
            can_vis = joint_dropout(node_pos, can_vis, cfg.p_joint_drop)
        if cfg.aug_pose_drop:
            can_vis = pose_dropout(node_pos, can_vis, cfg.p_pose_drop)
        if cfg.aug_joint_rand:
            joint_randomize(node_pos, can_vis, cfg.p_joint_rand)
        batch["canonical_visibility"] = can_vis

        if cfg.aug_jitter:
            keypoints = batch["keypoints"]  # (B, N, 14, 2) pixel-space
            batch["gt_keypoints"] = keypoints.clone()
            batch["keypoints"] = keypoint_jitter(keypoints, cfg.p_jitter_px)

    def _augment_image_cond(self, batch: dict) -> None:
        """Zero random DINO node embeddings in-place (runs after DINO conditioning)."""
        image_cond = batch.get("image_cond")
        if image_cond is None or not self.model.config.aug_img_drop:
            return
        image_dropout(image_cond, self.model.config.p_img_drop)

    def get_loss(self, batch: dict, outputs: dict) -> dict[str, Tensor]:
        """Compute weighted losses in canonical (normalised root-relative) space.

        Using canonical coords keeps magnitudes ~O(1) regardless of image scale
        (pixel-space losses were ~1e5 for typical CrowdPose images).
        """
        cfg = self.model.config
        pred_canonical = outputs["pred_canonical"]  # (B, N, 14, 2)
        gt_canonical = outputs["gt_canonical"]       # (B, N, 14, 2)
        visibility = batch["visibility"]              # (B, N, 14)  0=absent 1=occluded 2=visible

        losses: dict[str, Tensor] = {}
        if cfg.use_joint_loss:
            losses["joint_loss"] = (
                compute_joint_loss(pred_canonical, gt_canonical, visibility) * cfg.w_joint_loss
            )
        if cfg.use_bone_loss:
            losses["bone_loss"] = (
                compute_bone_loss(pred_canonical, gt_canonical, visibility, CROWDPOSE_EDGE_LIST)
                * cfg.w_bone_loss
            )
        if not losses:
            raise ValueError(
                "All losses are disabled — enable at least one loss in KeypointDiffusionConfig"
            )
        losses["loss"] = sum(losses.values())
        return losses

    def get_global_image_cond(self, batch: dict) -> Tensor | None:
        """Mean-pool DINO patch tokens per image → (B*N*K, dim_global_image_cond) or None.

        The pooled feature is projected by ``global_dino_proj``, then broadcast
        to every node of the corresponding image via ``batch["batch"]``.
        """
        if self.global_dino_proj is None:
            return None

        dino = self.dino_conditioner
        image_ids = [int(i) for i in batch["image_id"]]

        if dino._patch_cache is None:
            raise RuntimeError(
                "DINO patch cache is not loaded. "
                "Run scripts/precompute_dino.py and set dino.cache_train / dino.cache_val "
                "in your config, or disable use_global_image_cond."
            )
        try:
            patch_feats = torch.stack(
                [dino._patch_cache[iid] for iid in image_ids]
            ).to(self.device)  # (B, G*G, D)
        except KeyError as e:
            raise RuntimeError(
                f"Image id {e} not found in DINO patch cache. "
                f"Re-run scripts/precompute_dino.py for the correct split and update "
                f"dino.cache_train / dino.cache_val in your config."
            ) from e

        pooled = patch_feats.mean(dim=1)                     # (B, D)
        projected = self.global_dino_proj(pooled)            # (B, dim_global_image_cond)
        return projected[batch["batch"]]                     # (B*N*K, dim_global_image_cond)

    def get_local_spatial_condition(self, batch: dict) -> Tensor | None:
        """Compute per-keypoint local visual features → (B*N*15, dim_image_cond) or None.

        DINO only   : dino.proj(dino_feat)
        CNN only    : cnn.head(cnn_feat)
        DINO + CNN  : fuse_proj(cat([dino_proj(dino_feat), cnn_proj(cnn_feat)]))
        """
        from src.data import read_image

        if self.dino_conditioner is None and self.cnn_conditioner is None:
            return None

        keypoints = batch["keypoints"]  # (B, N, 14, 2)
        B, N, K = keypoints.shape[:3]
        kpt = keypoints.to(self.device)
        image_ids = [int(i) for i in batch["image_id"]]

        if self.dino_conditioner is not None:
            dino = self.dino_conditioner
            if dino._patch_cache is None:
                raise RuntimeError(
                    "DINO patch cache is not loaded. "
                    "Run scripts/precompute_dino.py and set dino.cache_train / dino.cache_val "
                    "in your config, or disable use_image_cond."
                )
            try:
                patch_feats = torch.stack(
                    [dino._patch_cache[iid] for iid in image_ids]
                ).to(self.device)
            except KeyError as e:
                raise RuntimeError(
                    f"Image id {e} not found in DINO patch cache. "
                    f"Re-run scripts/precompute_dino.py for the correct split and update "
                    f"dino.cache_train / dino.cache_val in your config."
                ) from e
            image_sizes = [read_image_size(p) for p in batch["img_path"]]
            img_hw = torch.tensor(image_sizes, dtype=torch.float32, device=self.device)
            dino_feat = dino._gather_neighborhood(patch_feats, kpt, img_hw)  # (B, N, K, D_dino)

        if self.cnn_conditioner is not None:
            cnn = self.cnn_conditioner
            # Keep images on CPU — only the extracted patches are moved to GPU.
            images_cpu = [read_image(p).float().div(255.0) for p in batch["img_path"]]
            cnn_feat = torch.stack([
                cnn.cnn(
                    cnn._extract_patches(images_cpu[b], kpt[b].cpu()).to(self.device)
                ).mean(dim=(-2, -1)).view(N, K, -1)
                for b in range(B)
            ])  # (B, N, K, cnn.backbone_dim)

        if self.dino_conditioner is not None and self.cnn_conditioner is not None:
            keypoint_emb = self.fuse_proj(
                torch.cat([self.dino_proj(dino_feat), self.cnn_proj(cnn_feat)], dim=-1)
            )
        elif self.dino_conditioner is not None:
            keypoint_emb = dino.proj(dino_feat)
        else:
            keypoint_emb = cnn.head(cnn_feat)

        j = CROWDPOSE_JOINT_NAMES
        anchor_idx = [j["left_hip"], j["right_hip"], j["neck"]]
        root_emb = keypoint_emb[:, :, anchor_idx, :].mean(dim=2, keepdim=True)  # (B, N, 1, D)
        return torch.cat([root_emb, keypoint_emb], dim=2).reshape(B * N * 15, -1)

    # ------------------------------------------------------------------
    # AP evaluation helpers
    # ------------------------------------------------------------------

    def _ap_step(
        self,
        batch: dict,
        metric: KeypointAPMetric,
        num_steps: int | None = None,
    ) -> None:
        """Refine one batch of detector predictions with DDIM and feed results to *metric*."""
        pred_pixel, pred_scores, image_ids = self._refine_prediction_batch(batch, num_steps=num_steps)
        metric.update(pred_pixel, pred_scores, image_ids)

    def _refine_prediction_batch(
        self,
        batch: dict,
        num_steps: int | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        pred_kpts = batch["pred_keypoints"].to(self.device)  # (B, N, 14, 2) pixel
        pred_scores = batch["pred_keypoint_scores"].to(self.device)  # (B, N, 14)
        image_ids = batch["image_id"]  # (B,)

        B, N = pred_kpts.shape[:2]
        K = 15  # canonical keypoints (14 + virtual root)

        batch["keypoints"] = pred_kpts
        # Detector confidence > 0.1 → treat as visible (matches the threshold used in training data)
        batch["visibility"] = (pred_scores > 0.1).float()

        # Convert predictions to canonical space (root-relative, normalised)
        self.prepare_batch(batch)

        edge_index = batch["edge_index"]  # (2, B*N*E)
        node_type_idx = batch["kpt_type_idx"].reshape(-1)  # (B*N*K,) values in [0, K-1]
        node_batch_idx = batch["batch"]  # (B*N*K,)
        num_nodes = B * N * K

        T = self.model.config.num_train_steps
        t_start = max(1, round(T * self._inference_noise_frac))

        # Add a small amount of noise so the model sees in-distribution inputs
        # while keeping predictions close to the detector output.
        noise_timestep = torch.full(
            (B,), t_start, dtype=torch.long, device=pred_kpts.device
        )
        x_init_noisy, _ = self.model.add_noise(
            node_pos=batch["node_pos"],
            timestep=noise_timestep,
            batch_idx=node_batch_idx,
        )  # (B*N*K, 2)

        default_steps = (
            self.num_inference_steps
            if self.num_inference_steps is not None
            else t_start
        )
        n_steps = num_steps if num_steps is not None else default_steps

        nodes_canonical_flat = self.model.ddim_sample(
            edge_index=edge_index,
            kpt_type_idx=node_type_idx,
            batch_idx=node_batch_idx,
            num_nodes=num_nodes,
            num_inference_steps=n_steps,
            start_t=t_start,
            device=pred_kpts.device,
            x_init=x_init_noisy,
            image_cond=batch.get("image_cond"),
            global_image_cond=batch.get("global_image_cond"),
            guidance_scale=self.cfg_guidance_scale,
        )  # (B*N*K, 2)

        # Convert back to pixel space using the root and scale from the batch
        nodes_canonical = nodes_canonical_flat.view(B, N, K, 2)
        joint_pred_pixel = canonical_to_pixel(
            nodes_canonical, batch["root_pos"], scale=batch["scale"]
        )

        return joint_pred_pixel, pred_scores, image_ids

    def _run_ap_eval(
        self,
        pred_cache: str,
        metric: KeypointAPMetric,
        desc: str = "AP eval",
        num_steps: int | None = None,
    ) -> None:
        """Load a prediction cache, refine every batch, and accumulate into *metric*."""
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
    ) -> Tensor:
        self.prepare_batch(batch, augment=(stage == "train"))

        B, N = batch["keypoints"].shape[:2]
        K = 15  # canonical nodes: 14 joints + 1 virtual root

        node_pos = batch["node_pos"]  # (B*N*K, 2) canonical
        node_type_idx = batch["kpt_type_idx"].reshape(-1)  # (B*N*K,) values in [0, K-1]
        node_batch_idx = batch["batch"]  # (B*N*K,)
        edge_index = batch["edge_index"]  # (2, B*N*E)
        root_pos = batch["root_pos"]  # (B, N, 2) pixel-space root
        scale = batch["scale"]  # (B,) normalisation divisor

        timestep = torch.randint(
            1,
            self.model.config.num_train_steps + 1,
            (B,),
            device=node_pos.device,
            generator=generator,
        )  # (B,)  — 1-indexed to match VP schedule buffers

        noisy_node_pos, _ = self.model.add_noise(
            node_pos=node_pos,
            timestep=timestep,
            batch_idx=node_batch_idx,
            generator=generator,
        )

        pred_delta = self.model.backward_process(
            noisy_node_pos=noisy_node_pos,
            edge_index=edge_index,
            kpt_type_idx=node_type_idx,
            batch_idx=node_batch_idx,
            timestep=timestep,
            image_cond=batch.get("image_cond"),
            global_image_cond=batch.get("global_image_cond"),
        )

        # Drop virtual root (index 0); invert normalisation back to pixel space
        # canonical = (pixel − root) / scale  →  pixel = canonical * scale + root
        joint_pred_canonical = (noisy_node_pos - pred_delta).view(B, N, K, 2)[
            :, :, 1:
        ]  # (B, N, 14, 2)
        joint_pred_pixel = (
            joint_pred_canonical * scale[:, None, None, None] + root_pos[:, :, None]
        )  # (B, N, 14, 2)

        gt_pixel = batch.get("gt_keypoints", batch["keypoints"])  # (B, N, 14, 2)
        gt_canonical = (gt_pixel - root_pos[:, :, None]) / scale[:, None, None, None]  # (B, N, 14, 2)

        outputs = {
            "pred_pixel": joint_pred_pixel,
            "pred_canonical": joint_pred_canonical,
            "gt_pixel": gt_pixel,
            "gt_canonical": gt_canonical,
            "timestep": timestep,
        }

        losses = self.get_loss(batch, outputs)
        return losses, outputs, B

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        losses, _outputs, B = self._step(batch, stage="train")
        log_kwargs = dict(batch_size=B, on_step=True, on_epoch=True)
        self.log("train/loss", losses["loss"], prog_bar=True, **log_kwargs)
        for name, value in losses.items():
            if name != "loss":
                self.log(f"train/{name}", value, **log_kwargs)
        return losses["loss"]

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        gen = torch.Generator(device=self.device).manual_seed(batch_idx)
        losses, outputs, B = self._step(batch, stage="val", generator=gen)
        gt_pixel = outputs["gt_pixel"]
        visibility = batch["visibility"]
        log_kwargs = dict(batch_size=B, on_step=False, on_epoch=True)
        self.log("val/loss", losses["loss"], prog_bar=True, **log_kwargs)
        for name, value in losses.items():
            if name != "loss":
                self.log(f"val/{name}", value, **log_kwargs)
        self.log("val/mpjpe_px", compute_mpjpe(outputs["pred_pixel"], gt_pixel, visibility), **log_kwargs)
        self.log("val/pck_20px", compute_pck(outputs["pred_pixel"], gt_pixel, visibility, threshold_px=20.0), **log_kwargs)

    def test_step(self, batch: dict, batch_idx: int) -> None:
        gen = torch.Generator(device=self.device).manual_seed(batch_idx)
        losses, outputs, B = self._step(batch, stage="test", generator=gen)
        gt_pixel = outputs["gt_pixel"]
        visibility = batch["visibility"]
        log_kwargs = dict(batch_size=B, on_step=False, on_epoch=True)
        self.log("test/loss", losses["loss"], prog_bar=True, **log_kwargs)
        for name, value in losses.items():
            if name != "loss":
                self.log(f"test/{name}", value, **log_kwargs)
        self.log("test/mpjpe_px", compute_mpjpe(outputs["pred_pixel"], gt_pixel, visibility), **log_kwargs)
        self.log("test/pck_20px", compute_pck(outputs["pred_pixel"], gt_pixel, visibility, threshold_px=20.0), **log_kwargs)

    def on_validation_epoch_end(self) -> None:
        if self._val_metric is None or self._val_pred_cache is None:
            return
        if self.trainer.sanity_checking:
            return
        if self._ap_eval_interval <= 0:
            return
        if (self.current_epoch + 1) % self._ap_eval_interval != 0:
            return
        step_counts = self._ap_step_counts
        if step_counts:
            for n in step_counts:
                self._run_ap_eval(
                    self._val_pred_cache,
                    self._val_metric,
                    desc=f"val AP@{n} [epoch {self.current_epoch}]",
                    num_steps=n,
                )
                results = self._val_metric.compute()
                for key, val in results.items():
                    self.log(f"val/{key}@{n}", val, prog_bar=(key == "coco/AP"), sync_dist=True)
        else:
            self._run_ap_eval(
                self._val_pred_cache,
                self._val_metric,
                desc=f"val AP [epoch {self.current_epoch}]",
            )
            results = self._val_metric.compute()
            for key, val in results.items():
                self.log(f"val/{key}", val, prog_bar=(key == "coco/AP"), sync_dist=True)

    def on_test_epoch_end(self) -> None:
        if self._test_metric is None or self._test_pred_cache is None:
            return
        step_counts = self._ap_step_counts
        if step_counts:
            for n in step_counts:
                self._run_ap_eval(
                    self._test_pred_cache,
                    self._test_metric,
                    desc=f"test AP@{n}",
                    num_steps=n,
                )
                results = self._test_metric.compute()
                for key, val in results.items():
                    self.log(f"test/{key}@{n}", val, sync_dist=True)
        else:
            self._run_ap_eval(self._test_pred_cache, self._test_metric)
            results = self._test_metric.compute()
            for key, val in results.items():
                self.log(f"test/{key}", val, sync_dist=True)

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

"""Graph-diffusion pipeline decoder.

A unified diffusion denoiser over (boxes, keypoints) with two sub-stages:

  Stage 1 — attention (QueryProbe-style):
    Takes noisy (boxes, joints_rel) + image tokens and produces rough x₀
    estimates via learned instance + joint attention, following the QueryProbe
    architecture in src/training/backbone_probe.py.

  Stage 2 — GNN refiner:
    Applies skeleton-structured graph message-passing (PoseGraphLayer) over
    Stage 1's joint estimates, sampling visual features at predicted locations.

The diffusion state is the concatenation of:
  * boxes         [B, N, 4]    (cx, cy, w, h) in canonical (−1, 1) space
  * joints_rel    [B, N, K, 2] keypoint offsets relative to box centres

Both receive the same σ-scaled Gaussian noise during training / sampling.

Training protocol (per the plan):
  - Pad GT to N slots with random box / zero-offset joints, then shuffle.
  - Add σ·ε noise (log-normal EDM schedule) to both boxes and joints.
  - Predict clean x₀; optimise with Hungarian-matched loss end-to-end.

Sampling: EDM Heun loop identical to BoxSetDiffusionDecoder.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.base_config import BaseConfig
from src.data.graph import create_graph_from_keypoints
from src.data.skeleton import CANONICAL_EDGE_INDEX
from src.decoders.denoise.graph_network import GraphDenoiseNetwork, GraphDenoiseNetworkConfig
from src.embedders.spatial_attention import make_pe_from_centers
from src.training.diffusion_modules import FourierFeatures, TimeFiLMTransformerBlock


class GraphDiffusionPipelineDecoder(nn.Module):
    """Diffusion denoiser for joint (box, pose) prediction.

    Args:
        num_slots:            N — number of instance slots.
        num_keypoints:        K — keypoints per instance (14 for CrowdPose).
        dim:                  D — hidden dim (must be divisible by 4 and num_heads).
        num_heads:            Attention heads.
        num_trunk_blocks:     Transformer blocks for instance ↔ image attention.
        num_joint_blocks:     Transformer blocks for joint ↔ instance attention.
        time_dim:             Time-embedding dim.
        ffn_mult:             FFN hidden expansion factor.
        image_token_dim:      Channel dim C of backbone image tokens.
        spatial_feat_dim:     Channel dim of backbone spatial feature map for GNN
                              visual conditioning (same backbone output, [B,C,H',W']).
        dim_vis:              Projected visual feature dim fed into the GNN.
        dim_edge:             Edge feature dim for the GNN.
        num_refiner_layers:   Number of PoseGraphLayers in Stage 2.
        fourier_{box,joint,sigma}_bands/scale: Fourier bank hyperparameters.
        sigma_log_offset:     Lower clamp on σ before log (numerical safety).
        pe_scale:             PE scale for image-token sinusoidal encoding.
        dropout:              Dropout rate used in attention / FFN / edge encoder.
    """

    def __init__(
        self,
        num_slots: int = 20,
        num_keypoints: int = 14,
        dim: int = 256,
        num_heads: int = 8,
        num_trunk_blocks: int = 3,
        num_joint_blocks: int = 2,
        time_dim: int = 256,
        ffn_mult: float = 4.0,
        image_token_dim: int = 768,
        spatial_feat_dim: int = 768,
        dim_vis: int = 128,
        dim_edge: int = 64,
        num_refiner_layers: int = 3,
        fourier_box_bands: int = 16,
        fourier_box_scale: float = 1000.0,
        fourier_joint_bands: int = 16,
        fourier_joint_scale: float = 1000.0,
        fourier_sigma_bands: int = 16,
        fourier_sigma_scale: float = 1.0,
        sigma_log_offset: float = 1e-4,
        pe_scale: float = 1000.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if dim % 4 != 0:
            raise ValueError(f"dim must be divisible by 4; got {dim}")
        if dim % num_heads != 0:
            raise ValueError(f"dim must be divisible by num_heads; got {dim} % {num_heads}")
        if num_slots < 1:
            raise ValueError(f"num_slots must be >= 1; got {num_slots}")

        self.num_slots = num_slots
        self.num_keypoints = num_keypoints
        self.dim = dim
        self.time_dim = time_dim
        self.dim_vis = dim_vis
        self.sigma_log_offset = float(sigma_log_offset)
        self.pe_scale = float(pe_scale)

        # ------------------------------------------------------------------
        # Stage 1 — inputs
        # ------------------------------------------------------------------
        # Embed noisy box coordinates (cx, cy, w, h) → D.
        self.box_in = FourierFeatures(
            in_dim=4, out_dim=dim,
            num_bands=fourier_box_bands, scale=fourier_box_scale,
        )
        # Embed noisy joint offsets (dx, dy) → D.
        self.joint_in = FourierFeatures(
            in_dim=2, out_dim=dim,
            num_bands=fourier_joint_bands, scale=fourier_joint_scale,
        )
        # Learned instance-slot queries.
        self.slot_query = nn.Embedding(num_slots, dim)
        # Learned joint-type queries (one per keypoint).
        self.joint_query = nn.Embedding(num_keypoints, dim)
        # Time (log σ) embedding.
        self.time_in = FourierFeatures(
            in_dim=1, out_dim=time_dim,
            num_bands=fourier_sigma_bands, scale=fourier_sigma_scale,
        )
        # Project backbone image tokens to D.
        self.image_proj = nn.Linear(image_token_dim, dim)

        # ------------------------------------------------------------------
        # Stage 1 — trunk: instance slots ↔ image tokens
        # ------------------------------------------------------------------
        self.trunk_blocks = nn.ModuleList([
            TimeFiLMTransformerBlock(
                dim=dim, num_heads=num_heads, time_dim=time_dim,
                ffn_mult=ffn_mult, cross_dim=dim, use_cross_attn=True,
                dropout=dropout,
            )
            for _ in range(num_trunk_blocks)
        ])

        # ------------------------------------------------------------------
        # Stage 1 — heads: objectness + box
        # ------------------------------------------------------------------
        self.obj_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        self.box_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 4))

        # ------------------------------------------------------------------
        # Stage 1 — joint path: joints cross-attend to instance latents
        # ------------------------------------------------------------------
        self.joint_blocks = nn.ModuleList([
            TimeFiLMTransformerBlock(
                dim=dim, num_heads=num_heads, time_dim=time_dim,
                ffn_mult=ffn_mult, cross_dim=dim, use_cross_attn=True,
                dropout=dropout,
            )
            for _ in range(num_joint_blocks)
        ])
        # Within-slot joint self-attention.
        self.joint_self_attn = TimeFiLMTransformerBlock(
            dim=dim, num_heads=num_heads, time_dim=time_dim,
            ffn_mult=ffn_mult, cross_dim=None, use_cross_attn=False,
            dropout=dropout,
        )
        # Joint regression head: per-keypoint (dx, dy) offset from box centre.
        self.joint_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2))

        # ------------------------------------------------------------------
        # Stage 2 — GNN refiner
        # ------------------------------------------------------------------
        # Project backbone spatial channels to dim_vis for node conditioning.
        self.vis_proj = nn.Linear(spatial_feat_dim, dim_vis)
        # Position + type embeddings for GNN nodes.
        self.refiner_pos_emb = nn.Sequential(
            nn.Linear(2, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim),
        )
        self.refiner_type_emb = nn.Embedding(num_keypoints, dim)
        self.refiner_node_norm = nn.LayerNorm(dim)
        # Edge encoder: relative position → dim_edge.
        self.edge_encoder = nn.Sequential(
            nn.Linear(2, dim_edge), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim_edge, dim_edge),
        )
        # Stacked PoseGraphLayers via GraphDenoiseNetwork.
        # dim_global=1: FiLM receives a zero scalar, learning bias-only scale/shift.
        self.refiner = GraphDenoiseNetwork(
            hidden_dim=dim,
            num_layers=num_refiner_layers,
            dim_edge=dim_edge,
            dim_vis=dim_vis,
            dim_global=1,
            num_heads=num_heads,
            out_dim=2,
            dropout_rate=dropout,
            attn_dropout_rate=dropout,
        )

        # Fixed CrowdPose skeleton edge index (registered buffer → moves with .to()).
        self.register_buffer("_edge_index", CANONICAL_EDGE_INDEX.clone())

    # ------------------------------------------------------------------
    # Denoiser forward (one call per diffusion step)
    # ------------------------------------------------------------------

    def forward(
        self,
        noisy_boxes: Tensor,           # [B, N, 4]
        noisy_joints_rel: Tensor,      # [B, N, K, 2]
        sigma: Tensor,                 # [B]
        image_tokens: Tensor,          # [B, T, image_token_dim]
        patch_centers: Tensor,         # [T, 2]  canonical (−1, 1)
        spatial_feat: Tensor,          # [B, C, H', W']
    ) -> dict[str, Tensor]:
        """Predict clean x₀ for boxes and joints.

        Returns:
            pred_boxes       [B, N, 4]    (cx, cy, w, h) canonical x₀.
            pred_joints_rel  [B, N, K, 2] keypoint offsets relative to pred_box_centres.
            obj_logits       [B, N]       raw objectness logits.
        """
        B, N, _ = noisy_boxes.shape
        K = self.num_keypoints
        device = noisy_boxes.device
        dtype = noisy_boxes.dtype

        # ---- time embedding -------------------------------------------------
        log_sigma = torch.log(
            sigma.clamp_min(self.sigma_log_offset).to(dtype)
        ).unsqueeze(-1)                                            # [B, 1]
        time_emb = self.time_in(log_sigma)                        # [B, time_dim]

        # ---- image tokens + PE ----------------------------------------------
        img = self.image_proj(image_tokens)                       # [B, T, D]
        pe = make_pe_from_centers(
            patch_centers.to(dtype), self.dim, scale=self.pe_scale,
        )                                                         # [T, D]
        img = img + pe.unsqueeze(0)                               # [B, T, D]

        # ---- Stage 1: instance path -----------------------------------------
        # Embed noisy boxes, add slot queries.
        slot_tokens = self.box_in(noisy_boxes)                    # [B, N, D]
        slot_tokens = slot_tokens + self.slot_query.weight.unsqueeze(0)

        # Trunk: slots ↔ image cross-attention with time conditioning.
        for blk in self.trunk_blocks:
            slot_tokens = blk(slot_tokens, context=img, time_emb=time_emb)
        inst_latents = slot_tokens                                # [B, N, D]

        # Objectness and box predictions.
        obj_logits  = self.obj_head(inst_latents).squeeze(-1)     # [B, N]
        pred_boxes  = self.box_head(inst_latents)                 # [B, N, 4]

        # ---- Stage 1: joint path --------------------------------------------
        # Embed noisy joint offsets, add joint-type queries (broadcast over N).
        joint_tokens = self.joint_in(noisy_joints_rel)            # [B, N, K, D]
        joint_tokens = joint_tokens + self.joint_query.weight     # broadcasts [K, D]
        joint_tokens = joint_tokens.reshape(B * N, K, self.dim)   # [B*N, K, D]

        # Expand time embedding to per-slot.
        time_per_slot = (
            time_emb.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)
        )                                                         # [B*N, time_dim]

        # Cross-attend joints to instance latents.
        inst_kv = inst_latents.reshape(B * N, 1, self.dim)        # [B*N, 1, D]
        for blk in self.joint_blocks:
            joint_tokens = blk(
                joint_tokens, context=inst_kv, time_emb=time_per_slot,
            )

        # Within-slot joint self-attention (joint interaction per person).
        joint_tokens = self.joint_self_attn(
            joint_tokens, time_emb=time_per_slot,
        )                                                         # [B*N, K, D]

        # Joint regression: predict clean x₀ offsets.
        rough_joints_rel = self.joint_head(joint_tokens)          # [B*N, K, 2]
        rough_joints_rel = rough_joints_rel.reshape(B, N, K, 2)   # [B, N, K, 2]

        # ---- Stage 2: GNN refiner -------------------------------------------
        # Absolute joint positions in canonical space.
        box_centres  = pred_boxes[:, :, :2]                       # [B, N, 2]
        joints_abs   = box_centres.unsqueeze(2) + rough_joints_rel  # [B, N, K, 2]

        M = B * N * K
        graph     = create_graph_from_keypoints(self._edge_index, joints_abs)
        edge_idx  = graph["edge_index"]                           # [2, B*N*E]
        type_flat = graph["kpt_type_idx"].reshape(-1)             # [M]
        node_pos  = joints_abs.reshape(M, 2)                      # [M, 2]

        # Node features: position + joint-type embeddings.
        h = self.refiner_node_norm(
            self.refiner_pos_emb(node_pos)
            + self.refiner_type_emb(type_flat)
        )                                                         # [M, D]

        # Edge features: relative position between connected nodes.
        src, dst = edge_idx[0], edge_idx[1]
        e = self.edge_encoder(node_pos[src] - node_pos[dst])      # [E, dim_edge]

        # Visual conditioning: bilinear-sample from spatial feature map.
        v = self._sample_visual_features(
            spatial_feat, joints_abs, B, N, K, dtype,
        )                                                         # [M, dim_vis]

        # Global conditioning: zeros → FiLM uses learned bias only.
        g = torch.zeros(M, 1, device=device, dtype=dtype)

        delta = self.refiner(h, edge_idx, e, v, g)                # [M, 2]
        pred_joints_rel = rough_joints_rel + delta.reshape(B, N, K, 2)

        return {
            "pred_boxes":      pred_boxes,       # [B, N, 4]
            "pred_joints_rel": pred_joints_rel,  # [B, N, K, 2]
            "obj_logits":      obj_logits,        # [B, N]
        }

    # ------------------------------------------------------------------
    # Visual feature sampling helper
    # ------------------------------------------------------------------

    def _sample_visual_features(
        self,
        spatial_feat: Tensor,   # [B, C, H', W']
        joints_abs: Tensor,     # [B, N, K, 2]  canonical (−1, 1)
        B: int,
        N: int,
        K: int,
        dtype: torch.dtype,
    ) -> Tensor:                # [B*N*K, dim_vis]
        """Bilinearly sample spatial features at each joint location."""
        # grid_sample expects grid in [-1, 1] matching image (x=width, y=height).
        # joints_abs last dim is (x, y) in canonical space — matches convention.
        grid = joints_abs.to(dtype).reshape(B, N * K, 1, 2)       # [B, N*K, 1, 2]
        sampled = F.grid_sample(
            spatial_feat.to(dtype), grid,
            mode="bilinear", align_corners=True, padding_mode="zeros",
        )                                                          # [B, C, N*K, 1]
        v_raw = sampled.squeeze(-1).permute(0, 2, 1)               # [B, N*K, C]
        v = self.vis_proj(v_raw).reshape(B * N * K, self.dim_vis)  # [B*N*K, dim_vis]
        return v

    # ------------------------------------------------------------------
    # Sampling (EDM Heun, x₀-prediction parameterisation)
    # ------------------------------------------------------------------

    @staticmethod
    def edm_sigma_schedule(
        num_steps: int,
        sigma_min: float = 0.002,
        sigma_max: float = 10.0,
        rho: float = 7.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """Karras et al. (2022) σ schedule, trailing zero included.

        Returns ``[num_steps + 1]`` decreasing from ``sigma_max`` to ``0``.
        """
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1; got {num_steps}")
        step_indices = torch.arange(num_steps, device=device, dtype=dtype)
        inv_rho = 1.0 / rho
        sigmas = (
            sigma_max ** inv_rho
            + step_indices / max(num_steps - 1, 1)
            * (sigma_min ** inv_rho - sigma_max ** inv_rho)
        ) ** rho
        return torch.cat([sigmas, sigmas.new_zeros(1)])

    @torch.no_grad()
    def sample(
        self,
        image_tokens: Tensor,          # [B, T, C]
        patch_centers: Tensor,         # [T, 2]
        spatial_feat: Tensor,          # [B, C, H', W']
        *,
        num_inference_steps: int = 1,
        sigma_min: float = 0.002,
        sigma_max: float = 10.0,
        rho: float = 7.0,
        generator: Optional[torch.Generator] = None,
        x_boxes_init: Optional[Tensor] = None,      # [B, N, 4]
        x_joints_init: Optional[Tensor] = None,     # [B, N, K, 2]
    ) -> dict[str, Tensor]:
        """Run EDM Heun sampling over (boxes, joints).

        Single-step is one forward pass at σ_max.  Multi-step uses the Heun
        corrector at every interior step.

        Returns:
            pred_boxes       [B, N, 4]
            pred_joints_rel  [B, N, K, 2]
            obj_logits       [B, N]
        """
        if num_inference_steps < 1:
            raise ValueError(
                f"num_inference_steps must be >= 1; got {num_inference_steps}"
            )
        device = next(self.parameters()).device
        dtype  = next(self.parameters()).dtype
        B = image_tokens.shape[0]
        N, K = self.num_slots, self.num_keypoints

        def _init(x_opt: Optional[Tensor], shape: tuple) -> Tensor:
            if x_opt is not None:
                return x_opt.to(device=device, dtype=dtype)
            return torch.empty(shape, device=device, dtype=dtype).normal_(
                generator=generator
            ) * sigma_max

        x_boxes  = _init(x_boxes_init,  (B, N, 4))
        x_joints = _init(x_joints_init, (B, N, K, 2))

        if num_inference_steps == 1:
            sigma = torch.full((B,), sigma_max, device=device, dtype=dtype)
            return self.forward(
                noisy_boxes=x_boxes, noisy_joints_rel=x_joints, sigma=sigma,
                image_tokens=image_tokens, patch_centers=patch_centers,
                spatial_feat=spatial_feat,
            )

        sigmas = self.edm_sigma_schedule(
            num_inference_steps, sigma_min=sigma_min, sigma_max=sigma_max,
            rho=rho, device=device, dtype=dtype,
        )
        if x_boxes_init is None:
            x_boxes  = x_boxes  * (sigmas[0] / max(sigma_max, 1e-12))
        if x_joints_init is None:
            x_joints = x_joints * (sigmas[0] / max(sigma_max, 1e-12))

        last_pred: dict[str, Tensor] = {}
        for i in range(num_inference_steps):
            sigma_i    = sigmas[i]
            sigma_next = sigmas[i + 1]
            sigma_i_b  = sigma_i.expand(B)

            pred = self.forward(
                noisy_boxes=x_boxes, noisy_joints_rel=x_joints,
                sigma=sigma_i_b,
                image_tokens=image_tokens, patch_centers=patch_centers,
                spatial_feat=spatial_feat,
            )
            last_pred = pred

            x0_boxes  = pred["pred_boxes"]
            x0_joints = pred["pred_joints_rel"]

            # Compute score / derivative for Euler step.
            d_boxes  = (x_boxes  - x0_boxes)  / sigma_i.clamp_min(self.sigma_log_offset)
            d_joints = (x_joints - x0_joints) / sigma_i.clamp_min(self.sigma_log_offset)
            dt = sigma_next - sigma_i
            x_euler_boxes  = x_boxes  + dt * d_boxes
            x_euler_joints = x_joints + dt * d_joints

            if sigma_next > 0:
                sigma_next_b = sigma_next.expand(B)
                pred2 = self.forward(
                    noisy_boxes=x_euler_boxes, noisy_joints_rel=x_euler_joints,
                    sigma=sigma_next_b,
                    image_tokens=image_tokens, patch_centers=patch_centers,
                    spatial_feat=spatial_feat,
                )
                last_pred = pred2
                d2_boxes  = (x_euler_boxes  - pred2["pred_boxes"])  / sigma_next
                d2_joints = (x_euler_joints - pred2["pred_joints_rel"]) / sigma_next
                x_boxes  = x_boxes  + 0.5 * dt * (d_boxes  + d2_boxes)
                x_joints = x_joints + 0.5 * dt * (d_joints + d2_joints)
            else:
                x_boxes  = x_euler_boxes
                x_joints = x_euler_joints

        return last_pred


class GraphDiffusionPipelineDecoderConfig(BaseConfig):
    """Pydantic config for :class:`GraphDiffusionPipelineDecoder`."""

    num_slots: int = 20
    num_keypoints: int = 14
    dim: int = 256
    num_heads: int = 8
    num_trunk_blocks: int = 3
    num_joint_blocks: int = 2
    time_dim: int = 256
    ffn_mult: float = 4.0
    image_token_dim: int = 768
    spatial_feat_dim: int = 768
    dim_vis: int = 128
    dim_edge: int = 64
    num_refiner_layers: int = 3
    fourier_box_bands: int = 16
    fourier_box_scale: float = 1000.0
    fourier_joint_bands: int = 16
    fourier_joint_scale: float = 1000.0
    fourier_sigma_bands: int = 16
    fourier_sigma_scale: float = 1.0
    sigma_log_offset: float = 1e-4
    pe_scale: float = 1000.0
    dropout: float = 0.1

    def build(self) -> GraphDiffusionPipelineDecoder:
        return GraphDiffusionPipelineDecoder(
            num_slots=self.num_slots,
            num_keypoints=self.num_keypoints,
            dim=self.dim,
            num_heads=self.num_heads,
            num_trunk_blocks=self.num_trunk_blocks,
            num_joint_blocks=self.num_joint_blocks,
            time_dim=self.time_dim,
            ffn_mult=self.ffn_mult,
            image_token_dim=self.image_token_dim,
            spatial_feat_dim=self.spatial_feat_dim,
            dim_vis=self.dim_vis,
            dim_edge=self.dim_edge,
            num_refiner_layers=self.num_refiner_layers,
            fourier_box_bands=self.fourier_box_bands,
            fourier_box_scale=self.fourier_box_scale,
            fourier_joint_bands=self.fourier_joint_bands,
            fourier_joint_scale=self.fourier_joint_scale,
            fourier_sigma_bands=self.fourier_sigma_bands,
            fourier_sigma_scale=self.fourier_sigma_scale,
            sigma_log_offset=self.sigma_log_offset,
            pe_scale=self.pe_scale,
            dropout=self.dropout,
        )

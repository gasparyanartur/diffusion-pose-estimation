"""Set-prediction diffusion decoder for object bounding boxes.

The decoder denoises a set of ``N`` boxes (cx, cy, w, h in padded-canvas
canonical space) conditioned on a fixed-size bank of image tokens and a
continuous noise level σ. The forward pass predicts the denoised ``x₀``
boxes plus per-slot objectness logits.

This module implements only the forward path; the EDM sampler lives in a
separate file (added in I5).
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from src.base_config import BaseConfig
from src.embedders.spatial_attention import make_pe_from_centers
from src.training.diffusion_modules import (
    FourierFeatures,
    TimeFiLMFFN,
    TimeFiLMTransformerBlock,
)


class BoxSetDiffusionDecoder(nn.Module):
    """Predict denoised ``x₀`` boxes + objectness for ``N`` learned slots.

    Pipeline
    --------
    1. ``noisy_boxes`` -> per-coord Fourier features -> linear -> box tokens.
    2. Add learned instance-query embeddings (``nn.Embedding(N, dim)``).
    3. σ -> ``log σ`` -> Fourier features -> ``[B, time_dim]`` time embedding.
    4. Image tokens -> ``Linear(C → dim)`` + sinusoidal PE from patch centres.
    5. ``num_blocks`` × ``TimeFiLMTransformerBlock`` shared trunk (self-attn
       over slots, cross-attn over image tokens at ``dim``, FiLM FFN).
    6. Independent projections into two streams:
       - Objectness: ``Linear(dim → obj_dim)`` + ``num_obj_blocks`` ×
         ``TimeFiLMTransformerBlock`` (self-attn + cross-attn to image at
         ``dim`` + FiLM FFN) -> ``Linear(obj_dim, 1)``.
       - Box: ``Linear(dim → box_dim)`` + ``num_box_branch_blocks`` ×
         ``TimeFiLMTransformerBlock`` -> cross-attn over objectness reps
         (``obj_dim``) -> ``num_box_refine_blocks`` ×
         ``TimeFiLMTransformerBlock`` -> ``Linear(box_dim, 4)``.

    Args:
        num_slots: Number of learned instance queries / output slots ``N``.
        dim: Hidden dim ``D``. Must be divisible by 4 for the PE.
        num_heads: Number of attention heads.
        num_blocks: Number of slot ↔ image-token blocks in the shared trunk.
        num_obj_blocks: Number of transformer blocks in the objectness branch.
        num_box_branch_blocks: Number of transformer blocks in the box branch
            before the obj cross-attention.
        num_box_refine_blocks: Number of transformer blocks in the box branch
            after the obj cross-attention.
        obj_dim: Hidden dim for the objectness branch. Projects from ``dim``
            before the first obj block; cross-attn to image tokens uses
            ``dim`` as the KV dim. Defaults to ``256``.
        box_dim: Hidden dim for the box branch. Projects from ``dim`` before
            the first box block; cross-attn to image tokens uses ``dim`` as
            the KV dim; cross-attn to objectness reps uses ``obj_dim``.
            Defaults to ``256``.
        time_dim: Time-embedding dim used by FiLM modules.
        ffn_mult: FFN hidden ratio for all FiLM-FFN modules.
        image_token_dim: ``C`` — channel dim of the input image tokens.
        fourier_box_bands / fourier_box_scale: Fourier bank settings for the
            box-coord input. ``scale=1000`` mirrors :func:`make_pe_from_centers`.
        fourier_sigma_bands / fourier_sigma_scale: Fourier bank settings for
            ``log σ``.
        sigma_log_offset: Lower clamp on σ before taking ``log``.
        pe_scale: Multiplicative scale for the image-token PE (matches the
            backbone-probe pipeline default of 1000).
    """

    def __init__(
        self,
        num_slots: int = 20,
        dim: int = 256,
        num_heads: int = 8,
        num_blocks: int = 3,
        num_obj_blocks: int = 2,
        num_box_branch_blocks: int = 2,
        num_box_refine_blocks: int = 2,
        obj_dim: int = 256,
        box_dim: int = 256,
        time_dim: int = 256,
        ffn_mult: float = 4.0,
        image_token_dim: int = 1280,
        fourier_box_bands: int = 16,
        fourier_box_scale: float = 1000.0,
        fourier_sigma_bands: int = 16,
        fourier_sigma_scale: float = 1.0,
        sigma_log_offset: float = 1e-4,
        pe_scale: float = 1000.0,
    ) -> None:
        super().__init__()
        if dim % 4 != 0:
            raise ValueError(f"dim must be divisible by 4; got {dim}")
        if num_slots < 1:
            raise ValueError(f"num_slots must be >= 1; got {num_slots}")

        self.num_slots = num_slots
        self.dim = dim
        self.obj_dim = obj_dim
        self.box_dim = box_dim
        self.time_dim = time_dim
        self.sigma_log_offset = float(sigma_log_offset)
        self.pe_scale = float(pe_scale)

        # 1. Box-coord embedding.
        self.box_in = FourierFeatures(
            in_dim=4, out_dim=dim,
            num_bands=fourier_box_bands, scale=fourier_box_scale,
        )

        # 2. Learned instance queries.
        self.slot_query = nn.Embedding(num_slots, dim)

        # 3. Time (log σ) embedding.
        self.time_in = FourierFeatures(
            in_dim=1, out_dim=time_dim,
            num_bands=fourier_sigma_bands, scale=fourier_sigma_scale,
        )

        # 4. Image-token projection.
        self.image_proj = nn.Linear(image_token_dim, dim)

        # 5. Slot ↔ image transformer stack.
        self.blocks = nn.ModuleList([
            TimeFiLMTransformerBlock(
                dim=dim, num_heads=num_heads, time_dim=time_dim,
                ffn_mult=ffn_mult, cross_dim=dim, use_cross_attn=True,
            )
            for _ in range(num_blocks)
        ])

        # 6. Trunk → branch projections.
        self.obj_proj = nn.Linear(dim, obj_dim)
        self.box_proj = nn.Linear(dim, box_dim)

        # 6. Objectness branch: num_obj_blocks transformer blocks + head.
        #    Cross-attention KV comes from trunk-dim image tokens.
        self.obj_blocks = nn.ModuleList([
            TimeFiLMTransformerBlock(
                dim=obj_dim, num_heads=num_heads, time_dim=time_dim,
                ffn_mult=ffn_mult, cross_dim=dim, use_cross_attn=True,
            )
            for _ in range(num_obj_blocks)
        ])
        self.obj_head = nn.Linear(obj_dim, 1)

        # 7. Box branch: num_box_branch_blocks blocks -> cross-attn to obj
        #    -> num_box_refine_blocks blocks -> head.
        #    Cross-attention to image uses trunk dim; obj cross uses obj_dim.
        self.box_blocks = nn.ModuleList([
            TimeFiLMTransformerBlock(
                dim=box_dim, num_heads=num_heads, time_dim=time_dim,
                ffn_mult=ffn_mult, cross_dim=dim, use_cross_attn=True,
            )
            for _ in range(num_box_branch_blocks)
        ])
        self.box_obj_cross = TimeFiLMTransformerBlock(
            dim=box_dim, num_heads=num_heads, time_dim=time_dim,
            ffn_mult=ffn_mult, cross_dim=obj_dim, use_cross_attn=True,
        )
        self.box_refine_blocks = nn.ModuleList([
            TimeFiLMTransformerBlock(
                dim=box_dim, num_heads=num_heads, time_dim=time_dim,
                ffn_mult=ffn_mult, cross_dim=dim, use_cross_attn=True,
            )
            for _ in range(num_box_refine_blocks)
        ])
        self.box_head = nn.Linear(box_dim, 4)

    def forward(
        self,
        noisy_boxes: Tensor,                              # [B, N, 4]
        sigma: Tensor,                                    # [B]
        image_tokens: Tensor,                             # [B, T, image_token_dim]
        patch_centers: Tensor,                            # [T, 2] in [-1, 1]
        image_key_padding_mask: Optional[Tensor] = None,  # [B, T] True=pad
    ) -> dict[str, Tensor]:
        if noisy_boxes.shape[1] != self.num_slots:
            raise ValueError(
                f"expected N={self.num_slots} slots, got {noisy_boxes.shape[1]}"
            )

        B = noisy_boxes.shape[0]

        # --- inputs --------------------------------------------------------
        # 1+2: box tokens + slot queries.
        box_tokens = self.box_in(noisy_boxes)                          # [B, N, D]
        x = box_tokens + self.slot_query.weight.unsqueeze(0)           # [B, N, D]

        # 3: time embedding from log σ.
        log_sigma = torch.log(
            sigma.clamp_min(self.sigma_log_offset).to(noisy_boxes.dtype)
        ).unsqueeze(-1)                                                # [B, 1]
        time_emb = self.time_in(log_sigma)                             # [B, time_dim]

        # 4: image tokens + PE.
        img = self.image_proj(image_tokens)                            # [B, T, D]
        pe = make_pe_from_centers(
            patch_centers.to(img.dtype), self.dim, scale=self.pe_scale,
        )                                                              # [T, D]
        img = img + pe.unsqueeze(0)                                    # [B, T, D]

        # 5: slot ↔ image blocks.
        for blk in self.blocks:
            x = blk(
                x, context=img, time_emb=time_emb,
                context_key_padding_mask=image_key_padding_mask,
            )

        # 6: objectness branch — project trunk → obj_dim, then attend to image.
        obj_rep = self.obj_proj(x)                                     # [B, N, obj_dim]
        for blk in self.obj_blocks:
            obj_rep = blk(
                obj_rep, context=img, time_emb=time_emb,
                context_key_padding_mask=image_key_padding_mask,
            )                                                          # [B, N, obj_dim]
        obj_logits = self.obj_head(obj_rep).squeeze(-1)                # [B, N]

        # 7: box branch — independent projection from trunk → box_dim.
        box_rep = self.box_proj(x)                                     # [B, N, box_dim]
        for blk in self.box_blocks:
            box_rep = blk(
                box_rep, context=img, time_emb=time_emb,
                context_key_padding_mask=image_key_padding_mask,
            )                                                          # [B, N, box_dim]
        box_rep = self.box_obj_cross(
            box_rep, context=obj_rep.detach(), time_emb=time_emb,
        )
        for blk in self.box_refine_blocks:
            box_rep = blk(
                box_rep, context=img, time_emb=time_emb,
                context_key_padding_mask=image_key_padding_mask,
            )                                                          # [B, N, box_dim]
        pred_boxes = self.box_head(box_rep)                            # [B, N, 4]

        _ = B  # silence unused warning if B isn't otherwise needed
        return {"pred_boxes": pred_boxes, "obj_logits": obj_logits}

    # ------------------------------------------------------------------
    # Sampling (EDM Heun, x₀-prediction parameterisation)
    # ------------------------------------------------------------------

    @staticmethod
    def edm_sigma_schedule(
        num_steps: int,
        sigma_min: float = 0.002,
        sigma_max: float = 1.0,
        rho: float = 7.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """Karras et al. (2022) σ schedule, with a trailing 0.

        Returns a tensor of length ``num_steps + 1`` decreasing from
        ``sigma_max`` to ``0``. Index ``i`` is σ at step ``i``; the final
        ``0`` represents the clean (denoised) state.
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
        image_tokens: Tensor,                                  # [B, T, C]
        patch_centers: Tensor,                                 # [T, 2]
        *,
        num_inference_steps: int = 1,
        sigma_min: float = 0.002,
        sigma_max: float = 1.0,
        rho: float = 7.0,
        image_key_padding_mask: Optional[Tensor] = None,       # [B, T]
        generator: Optional[torch.Generator] = None,
        x_init: Optional[Tensor] = None,                       # [B, N, 4]
    ) -> dict[str, Tensor]:
        """Run EDM Heun sampling on the box set.

        Returns ``{"pred_boxes": [B, N, 4], "obj_logits": [B, N]}``.

        Single-step (``num_inference_steps == 1``) is one forward pass at
        ``σ_max`` and returns its ``x₀`` prediction directly. Multi-step uses
        the Heun corrector at every interior step.
        """
        if num_inference_steps < 1:
            raise ValueError(
                f"num_inference_steps must be >= 1; got {num_inference_steps}"
            )
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        B = image_tokens.shape[0]

        if x_init is not None:
            x = x_init.to(device=device, dtype=dtype)
        else:
            shape = (B, self.num_slots, 4)
            x = torch.empty(shape, device=device, dtype=dtype).normal_(
                generator=generator
            ) * sigma_max

        if num_inference_steps == 1:
            sigma = torch.full((B,), sigma_max, device=device, dtype=dtype)
            out = self.forward(
                noisy_boxes=x, sigma=sigma,
                image_tokens=image_tokens, patch_centers=patch_centers,
                image_key_padding_mask=image_key_padding_mask,
            )
            return {"pred_boxes": out["pred_boxes"], "obj_logits": out["obj_logits"]}

        sigmas = self.edm_sigma_schedule(
            num_inference_steps, sigma_min=sigma_min, sigma_max=sigma_max,
            rho=rho, device=device, dtype=dtype,
        )
        # Scale the initial noise to match the first σ in the schedule.
        if x_init is None:
            x = x * (sigmas[0] / max(sigma_max, 1e-12))

        last_obj_logits: Tensor | None = None
        for i in range(num_inference_steps):
            sigma_i = sigmas[i]
            sigma_next = sigmas[i + 1]
            sigma_i_batch = sigma_i.expand(B)

            pred = self.forward(
                noisy_boxes=x, sigma=sigma_i_batch,
                image_tokens=image_tokens, patch_centers=patch_centers,
                image_key_padding_mask=image_key_padding_mask,
            )
            x0 = pred["pred_boxes"]
            last_obj_logits = pred["obj_logits"]

            d_cur = (x - x0) / sigma_i.clamp_min(self.sigma_log_offset)
            dt = sigma_next - sigma_i
            x_euler = x + dt * d_cur

            if sigma_next > 0:
                sigma_next_batch = sigma_next.expand(B)
                pred2 = self.forward(
                    noisy_boxes=x_euler, sigma=sigma_next_batch,
                    image_tokens=image_tokens, patch_centers=patch_centers,
                    image_key_padding_mask=image_key_padding_mask,
                )
                x0_next = pred2["pred_boxes"]
                last_obj_logits = pred2["obj_logits"]
                d_next = (x_euler - x0_next) / sigma_next
                x = x + 0.5 * dt * (d_cur + d_next)
            else:
                x = x_euler

        assert last_obj_logits is not None
        return {"pred_boxes": x, "obj_logits": last_obj_logits}


class BoxSetDiffusionDecoderConfig(BaseConfig):
    """Pydantic config for :class:`BoxSetDiffusionDecoder`."""

    num_slots: int = 20
    dim: int = 512
    num_heads: int = 8
    num_blocks: int = 3
    num_obj_blocks: int = 1
    num_box_branch_blocks: int = 1
    num_box_refine_blocks: int = 1
    obj_dim: int = 256
    box_dim: int = 256
    time_dim: int = 256
    ffn_mult: float = 4.0
    image_token_dim: int = 1280
    fourier_box_bands: int = 16
    fourier_box_scale: float = 1000.0
    fourier_sigma_bands: int = 16
    fourier_sigma_scale: float = 100.0
    sigma_log_offset: float = 1e-4
    pe_scale: float = 1000.0

    def build(self) -> BoxSetDiffusionDecoder:
        return BoxSetDiffusionDecoder(
            num_slots=self.num_slots,
            dim=self.dim,
            num_heads=self.num_heads,
            num_blocks=self.num_blocks,
            num_obj_blocks=self.num_obj_blocks,
            num_box_branch_blocks=self.num_box_branch_blocks,
            num_box_refine_blocks=self.num_box_refine_blocks,
            obj_dim=self.obj_dim,
            box_dim=self.box_dim,
            time_dim=self.time_dim,
            ffn_mult=self.ffn_mult,
            image_token_dim=self.image_token_dim,
            fourier_box_bands=self.fourier_box_bands,
            fourier_box_scale=self.fourier_box_scale,
            fourier_sigma_bands=self.fourier_sigma_bands,
            fourier_sigma_scale=self.fourier_sigma_scale,
            sigma_log_offset=self.sigma_log_offset,
            pe_scale=self.pe_scale,
        )

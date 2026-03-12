"""Graph-based keypoint diffusion model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch_geometric as geom
from PIL import Image
from torch import Tensor
from torch_geometric.utils import scatter

from src.data import (
    CROWDPOSE_EDGE_LIST,
    CROWDPOSE_JOINT_NAMES,
    prepare_pose_graph_batch,
    read_image_pil,
    read_image_size,
)
from src.metrics import KeypointAPMetric, canonical_to_pixel


@dataclass
class KeypointDiffusionConfig:
    """Hyperparameters for :class:`KeypointDiffusionModel`."""

    name: str = "keypoint_diffusion"

    # Feature dimensions
    dim_pos: int = 128  # position-embedding output dim
    dim_index: int = 128  # keypoint-index embedding dim
    dim_time: int = 128  # timestep-embedding dim
    dim_bone: int = 128  # bone-vector embedding dim

    # GNN backbone
    dim_hidden: int = 128
    num_layers: int = 4

    # Diffusion schedule
    num_train_steps: int = 100

    # DINO image conditioning
    use_image_cond: bool = False
    dim_image_cond: int = 128  # projected DINO feature dim per node

    # Training augmentations
    aug_joint_drop: bool = True
    p_joint_drop: float = (
        0.1  # probability of dropping each non-root joint independently
    )
    aug_pose_drop: bool = True
    p_pose_drop: float = 0.1  # probability of dropping an entire person
    aug_jitter: bool = True
    p_jitter_px: float = 4.0  # std-dev (pixels) of Gaussian keypoint coordinate noise
    aug_img_drop: bool = True
    p_img_drop: float = 0.1  # fraction of DINO node embeddings zeroed out

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
        model_name: str = "facebook/dinov2-small",
        out_dim: int = 128,
        patch_size: int = 16,
        image_size: int = 224,
        neighborhood_radius: int = 1,
        cache_paths: list[str | Path] | None = None,
    ) -> None:
        super().__init__()
        from transformers import AutoImageProcessor, AutoModel

        self.patch_size = patch_size
        self.image_size = image_size
        self.neighborhood_radius = neighborhood_radius
        self.grid_size = image_size // patch_size  # e.g. 14 for ViT-S/16 @ 224

        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        self.backbone = AutoModel.from_pretrained(model_name)
        # Freeze the backbone – only the projection is trained
        for param in self.backbone.parameters():
            param.requires_grad_(False)

        hidden_dim = self.backbone.config.hidden_size
        self.proj = torch.nn.Linear(hidden_dim, out_dim)

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
        """Run DINO; return patch features (B, G*G, hidden_dim) with CLS token dropped."""
        inputs = self.processor(images=pil_images, return_tensors="pt")
        inputs = {
            k: v.to(next(self.backbone.parameters()).device) for k, v in inputs.items()
        }
        outputs = self.backbone(**inputs)
        return outputs.last_hidden_state[:, 1:]  # drop CLS  (B, G*G, D)

    def _gather_neighborhood(
        self,
        patch_feats: Tensor,  # (B, G*G, D)
        kpt_pixel: Tensor,  # (B, N, K, 2)  pixel-space (x, y)
        img_hw: Tensor,  # (B, 2)  original (H, W)
    ) -> Tensor:
        """Mean-pool the (2r+1)² patch window around each keypoint → (B, N, K, D)."""
        B, N, K, _ = kpt_pixel.shape
        G, D, r = self.grid_size, patch_feats.shape[-1], self.neighborhood_radius
        W = 2 * r + 1
        device = patch_feats.device

        # Scale pixel coords → DINO input space, then find the centre patch per keypoint
        img_h = img_hw[:, 0].float().view(B, 1, 1)
        img_w = img_hw[:, 1].float().view(B, 1, 1)
        x_scaled = kpt_pixel[..., 0].to(device) * (self.image_size / img_w)
        y_scaled = kpt_pixel[..., 1].to(device) * (self.image_size / img_h)
        cx = (x_scaled / self.patch_size).long().clamp(0, G - 1)  # (B, N, K) centre col
        cy = (y_scaled / self.patch_size).long().clamp(0, G - 1)  # (B, N, K) centre row

        # All W² (Δcol, Δrow) offsets in the window
        offsets = torch.arange(-r, r + 1, device=device)
        dcol, drow = torch.meshgrid(offsets, offsets, indexing="ij")
        dcol, drow = dcol.reshape(-1), drow.reshape(-1)  # (W²,)

        # Neighbour patch indices for every keypoint: (B, N, K, W²) → flat (B, N*K*W²)
        cols = (cx.unsqueeze(-1) + dcol).clamp(0, G - 1)
        rows = (cy.unsqueeze(-1) + drow).clamp(0, G - 1)
        patch_idx = (rows * G + cols).reshape(B, N * K * W * W)

        gathered = patch_feats.gather(1, patch_idx.unsqueeze(-1).expand(-1, -1, D))
        return gathered.view(B, N, K, W * W, D).mean(dim=3)

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

        # Per-node input projections
        self.position_embedding = torch.nn.Sequential(
            torch.nn.Linear(2, config.dim_pos),
            torch.nn.LayerNorm(config.dim_pos),
        )
        self.node_idx_embedding = torch.nn.Embedding(num_keypoints, config.dim_index)
        self.timestep_embedding = TimestepSinusoidalEmbedding(config.dim_time)
        self.bone_vector_embedding = torch.nn.Linear(2, config.dim_bone)

        # GNN backbone
        in_channels = (
            config.dim_pos + config.dim_index + config.dim_time + config.dim_bone
        )
        if config.use_image_cond:
            in_channels += config.dim_image_cond
        self.decoder = geom.nn.GAT(
            in_channels=in_channels,
            hidden_channels=config.dim_hidden,
            num_layers=config.num_layers,
            out_channels=2,
        )

        # VP noise schedule; alpha_bars[0]=1 (clean), alpha_bars[T]=most noisy.
        T = config.num_train_steps
        betas = torch.linspace(1e-4, 0.02, T)  # (T,)
        alphas = 1.0 - betas
        ab = torch.cumprod(alphas, dim=0)  # (T,)
        ab = torch.cat([torch.ones(1), ab])  # (T+1,) [0..T]
        self.register_buffer("_sqrt_alpha_bars", ab.sqrt())
        self.register_buffer(
            "_sqrt_one_minus_alpha_bars", (1.0 - ab).clamp(min=0.0).sqrt()
        )

    def add_noise(
        self,
        node_pos: Tensor,
        timestep: Tensor,
        batch_idx: Tensor,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        noise = torch.randn_like(node_pos)
        t_node = timestep[batch_idx]  # broadcast (B,) → (B*N*K,)
        a = self._sqrt_alpha_bars[t_node].unsqueeze(-1)
        b = self._sqrt_one_minus_alpha_bars[t_node].unsqueeze(-1)
        return a * node_pos + b * noise, noise

    def backward_process(
        self,
        noisy_node_pos: Tensor,
        edge_index: Tensor,
        kpt_type_idx: Tensor,
        batch_idx: Tensor,
        timestep: Tensor,
        image_cond: Tensor | None = None,  # (B*N*K, dim_image_cond) or None
        **kwargs,
    ) -> Tensor:
        num_nodes = noisy_node_pos.shape[0]

        # 1. Position embedding
        pos_emb = self.position_embedding(noisy_node_pos)  # (B*N*K, dim_pos)

        # 2. Keypoint-type embedding
        idx_emb = self.node_idx_embedding(kpt_type_idx)  # (B*N*K, dim_index)

        # 3. Timestep embedding broadcast from batch level to node level
        timestep_emb = self.timestep_embedding(timestep)  # (B, dim_time)
        timestep_emb = timestep_emb[batch_idx]  # (B*N*K, dim_time)

        # 4. Bone-vector embedding: embed each edge, then mean-pool back to destination nodes
        edge_src, edge_dst = edge_index[0], edge_index[1]
        bone_vecs = noisy_node_pos[edge_dst] - noisy_node_pos[edge_src]  # (B*N*E, 2)
        bone_emb = scatter(
            self.bone_vector_embedding(bone_vecs),
            edge_dst,
            dim=0,
            dim_size=num_nodes,
            reduce="mean",
        )  # (B*N*K, dim_bone)

        # 5. Concatenate all node features and run GNN
        features = [pos_emb, idx_emb, timestep_emb, bone_emb]
        if image_cond is not None:
            features.append(image_cond)  # (B*N*K, dim_image_cond)
        x = torch.cat(features, dim=-1)  # (B*N*K, in_channels)
        out = self.decoder(x, edge_index)  # (B*N*K, 2)
        return out

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
        image_cond: Tensor | None = None,  # (B*N*K, dim_image_cond) or None
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

            delta_x = self.backward_process(
                noisy_node_pos=x,
                edge_index=edge_index,
                kpt_type_idx=kpt_type_idx,
                batch_idx=batch_idx,
                timestep=timestep_batch,
                image_cond=image_cond,
            )  # (B*N*K, 2)
            x0_pred = x - delta_x

            # DDIM deterministic update — VP schedule.
            # At prev_timestep=0: alpha_prev=1, sigma_prev=0  →  x = x0_pred  (clean sample). ✓
            timestep_idx = timestep.long()
            prev_timestep_idx = prev_timestep.long()
            alpha_t = self._sqrt_alpha_bars[timestep_idx]
            sigma_t = self._sqrt_one_minus_alpha_bars[timestep_idx]
            alpha_prev = self._sqrt_alpha_bars[prev_timestep_idx]
            sigma_prev = self._sqrt_one_minus_alpha_bars[prev_timestep_idx]
            noise_pred = (x - alpha_t * x0_pred) / sigma_t
            x = alpha_prev * x0_pred + sigma_prev * noise_pred

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
        weight_decay: float = 0.0,
        lr_T_max: int | None = None,
        val_ann_file: str | None = None,
        test_ann_file: str | None = None,
        val_pred_cache: str | None = None,
        test_pred_cache: str | None = None,
        num_inference_steps: int | None = None,
        ap_batch_size: int = 8,
        ap_eval_interval: int = 0,
        inference_noise_frac: float = 0.1,
        # DINO conditioning
        dino_model_name: str | None = None,
        dino_neighborhood_radius: int = 1,
        dino_cache_train: str | None = None,
        dino_cache_val: str | None = None,
        dino_cache_test: str | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_T_max = lr_T_max
        self.num_inference_steps = num_inference_steps
        self._ap_batch_size = ap_batch_size
        self._ap_eval_interval = ap_eval_interval
        self._inference_noise_frac = inference_noise_frac
        self._val_pred_cache = val_pred_cache
        self._test_pred_cache = test_pred_cache

        self._val_metric = KeypointAPMetric(val_ann_file) if val_ann_file else None
        self._test_metric = KeypointAPMetric(test_ann_file) if test_ann_file else None

        # Optional DINO conditioner (only when model.config.use_image_cond is True)
        self.dino_conditioner: DinoConditioner | None = None
        if dino_model_name is not None and model.config.use_image_cond:
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

    # ------------------------------------------------------------------
    # Batch preparation (graph build + optional DINO conditioning)
    # ------------------------------------------------------------------

    def prepare_batch(self, batch: dict, augment: bool = False) -> dict:
        """Build graph, optionally augment (jitter before DINO so coordinates match), add DINO conditioning."""
        prepare_pose_graph_batch(batch)

        if augment:
            self._augment_graph(batch)

        if self.dino_conditioner is not None:
            self._add_dino_conditioning(batch)

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
        """Compute weighted losses; active terms and weights are set in KeypointDiffusionConfig."""
        cfg = self.model.config
        pred_pixel = outputs["pred_pixel"]  # (B, N, 14, 2)
        gt_pixel = batch.get("gt_keypoints", batch["keypoints"])  # (B, N, 14, 2)
        visibility = batch["visibility"]  # (B, N, 14)  0=absent 1=occluded 2=visible

        losses: dict[str, Tensor] = {}
        if cfg.use_joint_loss:
            losses["joint_loss"] = (
                compute_joint_loss(pred_pixel, gt_pixel, visibility) * cfg.w_joint_loss
            )
        if cfg.use_bone_loss:
            losses["bone_loss"] = (
                compute_bone_loss(pred_pixel, gt_pixel, visibility, CROWDPOSE_EDGE_LIST)
                * cfg.w_bone_loss
            )
        if not losses:
            raise ValueError(
                "All losses are disabled — enable at least one loss in KeypointDiffusionConfig"
            )
        losses["loss"] = sum(losses.values())
        return losses

    def _add_dino_conditioning(self, batch: dict) -> None:
        """Run DINO and write ``batch["image_cond"]`` (B*N*15, dim_image_cond).

        Root embedding = mean of anchor joints (left_hip, right_hip, neck).
        """
        keypoints = batch["keypoints"]  # (B, N, 14, 2) pixel-space (possibly jittered)
        B, N = keypoints.shape[:2]
        image_ids: list[int] = [int(img_id) for img_id in batch["image_id"]]

        if self.dino_conditioner._patch_cache is not None:
            # Fast path: patch features already cached — only need dimensions to scale keypoints.
            pil_images = None
            image_sizes = [read_image_size(p) for p in batch["img_path"]]
        else:
            # Slow path: load images and run the backbone live (no cache available).

            pil_images = [read_image_pil(p) for p in batch["img_path"]]
            image_sizes = [(img.height, img.width) for img in pil_images]

        img_hw = torch.tensor(image_sizes, dtype=torch.float32, device=self.device)
        keypoint_emb = self.dino_conditioner(
            kpt_pixel=keypoints.to(self.device),
            img_hw=img_hw,
            pil_images=pil_images,
            image_ids=image_ids,
        )

        # Root embedding = mean of three anchor joints
        j = CROWDPOSE_JOINT_NAMES
        anchor_idx = [j["left_hip"], j["right_hip"], j["neck"]]
        root_emb = keypoint_emb[:, :, anchor_idx, :].mean(
            dim=2, keepdim=True
        )  # (B, N, 1, D)

        # Prepend root then flatten: (B, N, 15, D) → (B*N*15, D)
        node_emb = torch.cat([root_emb, keypoint_emb], dim=2)
        batch["image_cond"] = node_emb.reshape(B * N * 15, -1)

    # ------------------------------------------------------------------
    # AP evaluation helpers
    # ------------------------------------------------------------------

    def _ap_step(
        self,
        batch: dict,
        metric: KeypointAPMetric,
    ) -> None:
        """Refine one batch of detector predictions with DDIM and feed results to *metric*."""
        pred_pixel, pred_scores, image_ids = self._refine_prediction_batch(batch)
        metric.update(pred_pixel, pred_scores, image_ids)

    def _refine_prediction_batch(
        self,
        batch: dict,
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

        num_steps = (
            self.num_inference_steps
            if self.num_inference_steps is not None
            else t_start
        )

        nodes_canonical_flat = self.model.ddim_sample(
            edge_index=edge_index,
            kpt_type_idx=node_type_idx,
            batch_idx=node_batch_idx,
            num_nodes=num_nodes,
            num_inference_steps=num_steps,
            start_t=t_start,
            device=pred_kpts.device,
            x_init=x_init_noisy,
            image_cond=batch.get("image_cond"),
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
            self._ap_step(batch, metric)

    def _step(self, batch: dict, stage: str) -> Tensor:
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
        )  # (B,)  — 1-indexed to match VP schedule buffers

        noisy_node_pos, _ = self.model.add_noise(
            node_pos=node_pos,
            timestep=timestep,
            batch_idx=node_batch_idx,
        )

        pred_delta = self.model.backward_process(
            noisy_node_pos=noisy_node_pos,
            edge_index=edge_index,
            kpt_type_idx=node_type_idx,
            batch_idx=node_batch_idx,
            timestep=timestep,
            image_cond=batch.get("image_cond"),
        )

        # Drop virtual root (index 0); invert normalisation back to pixel space
        # canonical = (pixel − root) / scale  →  pixel = canonical * scale + root
        joint_pred_canonical = (noisy_node_pos - pred_delta).view(B, N, K, 2)[
            :, :, 1:
        ]  # (B, N, 14, 2)
        joint_pred_pixel = (
            joint_pred_canonical * scale[:, None, None, None] + root_pos[:, :, None]
        )  # (B, N, 14, 2)

        outputs = {
            "pred_pixel": joint_pred_pixel,
            "pred_canonical": joint_pred_canonical,
            "timestep": timestep,
        }

        losses = self.get_loss(batch, outputs)
        return losses, B

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        losses, B = self._step(batch, stage="train")
        self.log(
            "train_loss",
            losses["loss"],
            batch_size=B,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )
        return losses["loss"]

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        losses, B = self._step(batch, stage="val")
        self.log(
            "val_loss",
            losses["loss"],
            batch_size=B,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

    def test_step(self, batch: dict, batch_idx: int) -> None:
        losses, B = self._step(batch, stage="test")
        self.log(
            "test_loss",
            losses["loss"],
            batch_size=B,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

    def on_validation_epoch_end(self) -> None:
        if self._val_metric is None or self._val_pred_cache is None:
            return
        if self.trainer.sanity_checking:
            return
        if self._ap_eval_interval <= 0:
            return
        if (self.current_epoch + 1) % self._ap_eval_interval != 0:
            return
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
        self._run_ap_eval(self._test_pred_cache, self._test_metric)
        results = self._test_metric.compute()
        for key, val in results.items():
            self.log(f"test/{key}", val, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        if self.lr_T_max is None:
            return optimizer
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.lr_T_max, eta_min=self.lr * 1e-2
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

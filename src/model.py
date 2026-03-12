"""Graph-based keypoint diffusion model."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch_geometric as geom
from PIL import Image
from torch import Tensor
from torch_geometric.utils import scatter

from src.data import CROWDPOSE_JOINT_NAMES, prepare_pose_graph_batch
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
    """Extract per-keypoint local DINO patch features.

    For each keypoint (in pixel space), gathers the surrounding ``(2r+1)×(2r+1)``
    DINO patch features and mean-pools them into a single local embedding.
    The DINO backbone is frozen; only the linear projection is trainable.

    Args:
        model_name:            HuggingFace model ID for a DINOv2/v3 ViT.
        out_dim:               Output dimension after projection.
        patch_size:            ViT patch size in pixels (default 16).
        image_size:            DINO input resolution (default 224).
        neighborhood_radius:   Half-width of the patch neighbourhood (``r=1``
                               gathers a 3×3 window of 9 patches).
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
        """Load (and merge) one or more precomputed cache files.

        Each file is a ``{image_id: Tensor(G*G, hidden_dim)}`` dict produced by
        ``scripts/precompute_dino.py``.  Multiple files (e.g. train + val) are
        merged into one lookup table.  After this call the DINO backbone is
        never invoked — features are looked up by image_id instead.
        """
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
        """Run DINO on a list of PIL images.

        Returns patch features of shape ``(B, G*G, hidden_dim)`` where
        ``G = grid_size``.  The CLS token is dropped.
        """
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
        """Mean-pool the patch neighbourhood for every keypoint.

        Returns ``(B, N, K, hidden_dim)``.
        """
        B, N, K, _ = kpt_pixel.shape
        G = self.grid_size
        D = patch_feats.shape[-1]
        r = self.neighborhood_radius
        device = patch_feats.device

        # Scale keypoint pixel coords → DINO input space [0, image_size)
        img_h = img_hw[:, 0].float().view(B, 1, 1)  # (B, 1, 1)
        img_w = img_hw[:, 1].float().view(B, 1, 1)  # (B, 1, 1)
        x_scaled = kpt_pixel[..., 0].to(device) * (self.image_size / img_w)  # (B, N, K)
        y_scaled = kpt_pixel[..., 1].to(device) * (self.image_size / img_h)  # (B, N, K)

        # Centre patch grid indices (clamped to valid range)
        cx = (x_scaled / self.patch_size).long().clamp(0, G - 1)  # (B, N, K)
        cy = (y_scaled / self.patch_size).long().clamp(0, G - 1)  # (B, N, K)

        # Build 2-D neighbourhood offset grids
        offsets = torch.arange(-r, r + 1, device=device)  # (2r+1,)
        px = (cx.unsqueeze(-1) + offsets).clamp(0, G - 1)  # (B, N, K, 2r+1)
        py = (cy.unsqueeze(-1) + offsets).clamp(0, G - 1)  # (B, N, K, 2r+1)

        # All (px, py) pairs — outer product along the last two dims
        px_grid = px.unsqueeze(-1).expand(*px.shape, 2 * r + 1)  # (B,N,K,2r+1,2r+1)
        py_grid = py.unsqueeze(-2).expand(
            *py.shape[:-1], 2 * r + 1, 2 * r + 1
        )  # same shape

        num_p = (2 * r + 1) ** 2
        flat_idx = (py_grid * G + px_grid).reshape(B, N * K * num_p)  # (B, NK·num_p)

        # Gather from patch_feats (B, G*G, D)
        gathered = patch_feats.gather(
            1, flat_idx.unsqueeze(-1).expand(-1, -1, D)  # (B, NK·num_p, D)
        )
        gathered = gathered.view(B, N, K, num_p, D)
        return gathered.mean(dim=3)  # (B, N, K, D)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        kpt_pixel: Tensor,  # (B, N, K, 2) pixel-space keypoints (x, y)
        img_hw: Tensor,  # (B, 2) original image (H, W)
        image_ids: list[int] | None = None,  # required when using a precomputed cache
        pil_images: list[Image.Image] | None = None,  # required when not using a cache
    ) -> Tensor:
        """Return per-keypoint DINO embeddings of shape ``(B, N, K, out_dim)``.

        Exactly one of *image_ids* (cache path) or *pil_images* (live backbone)
        must be provided.
        """
        if self._patch_cache is not None:
            if image_ids is None:
                raise ValueError(
                    "image_ids must be provided when using a precomputed cache"
                )
            device = kpt_pixel.device
            patch_feats = torch.stack([self._patch_cache[iid] for iid in image_ids]).to(
                device
            )  # (B, G*G, D)
        else:
            if pil_images is None:
                raise ValueError("pil_images must be provided when no cache is loaded")
            patch_feats = self._extract_patch_features(pil_images)  # (B, G*G, D)

        local_feats = self._gather_neighborhood(
            patch_feats, kpt_pixel, img_hw
        )  # (B, N, K, D)
        return self.proj(local_feats)  # (B, N, K, out_dim)


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
        noise = torch.randn_like(node_pos)  # (B*N*K, 2)
        a = self._sqrt_alpha_bars[timestep][batch_idx].unsqueeze(-1)  # (B*N*K, 1)
        b = self._sqrt_one_minus_alpha_bars[timestep][batch_idx].unsqueeze(
            -1
        )  # (B*N*K, 1)
        noisy_node_pos = a * node_pos + b * noise
        return noisy_node_pos, noise

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

        # 4. Bone-vector embedding aggregated back to nodes
        #    Bone vectors are naturally edge features; we mean-pool them per
        #    destination node so they become node features of the right size.
        edge_src = edge_index[0]  # (B*N*E,)
        edge_dst = edge_index[1]  # (B*N*E,)
        bone_vectors = noisy_node_pos[edge_dst] - noisy_node_pos[edge_src]  # (B*N*E, 2)
        bone_emb_edges = self.bone_vector_embedding(bone_vectors)  # (B*N*E, dim_bone)
        bone_emb = scatter(  # (B*N*K, dim_bone)
            bone_emb_edges, edge_dst, dim=0, dim_size=num_nodes, reduce="mean"
        )

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
        S = num_inference_steps if num_inference_steps is not None else t_start
        device = device or edge_index.device

        # Build decreasing timestep schedule: [t_start, ..., 1]
        timesteps = torch.linspace(t_start, 1, S, dtype=torch.long, device=device)

        # Start from provided initial estimate or pure Gaussian noise
        if x_init is not None:
            x = x_init.to(device)
        else:
            x = torch.randn(num_nodes, 2, device=device)

        for i, t_scalar in enumerate(timesteps):
            t_prev = (
                timesteps[i + 1]
                if i + 1 < len(timesteps)
                else torch.zeros(1, device=device)[0]
            )

            t_batch = t_scalar.expand(batch_idx.max().item() + 1)  # (B,)

            delta_x = self.backward_process(
                noisy_node_pos=x,
                edge_index=edge_index,
                kpt_type_idx=kpt_type_idx,
                batch_idx=batch_idx,
                timestep=t_batch,
                image_cond=image_cond,
            )  # (B*N*K, 2)
            x_0_hat = x - delta_x

            # DDIM deterministic update — VP schedule.
            # At t_prev=0: a_prev=1, b_prev=0  →  x = x_0_hat  (clean sample). ✓
            t_idx = t_scalar.long()
            t_prev_idx = t_prev.long()
            a_t = self._sqrt_alpha_bars[t_idx]
            b_t = self._sqrt_one_minus_alpha_bars[t_idx]
            a_prev = self._sqrt_alpha_bars[t_prev_idx]
            b_prev = self._sqrt_one_minus_alpha_bars[t_prev_idx]
            eps_hat = (x - a_t * x_0_hat) / b_t
            x = a_prev * x_0_hat + b_prev * eps_hat

        return x


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

    def prepare_batch(self, batch: dict) -> dict:
        """Build the canonical pose graph and, if enabled, add DINO conditioning.

        Replaces standalone ``prepare_pose_graph_batch`` calls inside the module.
        Updates *batch* in-place and returns it for convenience.

        After this call *batch* contains all fields from
        :func:`~src.data.prepare_pose_graph_batch` plus, when DINO conditioning
        is active, ``image_cond`` of shape ``(B*N*K, dim_image_cond)``.
        """
        prepare_pose_graph_batch(batch)
        if self.dino_conditioner is not None:
            self._add_dino_conditioning(batch)
        return batch

    def _add_dino_conditioning(self, batch: dict) -> None:
        """Load images, run DinoConditioner, and store per-node embeddings in batch.

        The 14 CrowdPose keypoints are embedded individually.  The virtual root
        node's embedding is computed as the mean of its three anchor joints
        (left_hip, right_hip, neck), matching the geometric definition of the
        root used in :func:`~src.data.prepare_pose_graph_batch`.

        Writes ``batch["image_cond"]`` of shape ``(B*N*15, dim_image_cond)``.
        """
        keypoints = batch[
            "keypoints"
        ]  # (B, N, 14, 2) pixel space (unchanged by prepare_batch)
        B, N = keypoints.shape[:2]
        image_ids: list[int] = [int(iid) for iid in batch["image_id"]]

        if self.dino_conditioner._patch_cache is not None:
            # Fast path: look up precomputed patch features, no backbone needed.
            # img_hw is still required by _gather_neighborhood to scale keypoints.
            hw_list = [
                (Image.open(p).height, Image.open(p).width) for p in batch["img_path"]
            ]
            img_hw = torch.tensor(hw_list, dtype=torch.float32, device=self.device)
            kpt_emb = self.dino_conditioner(
                kpt_pixel=keypoints.to(self.device),
                img_hw=img_hw,
                image_ids=image_ids,
            )  # (B, N, 14, dim_image_cond)
        else:
            # Slow path: load PIL images and run the backbone live.
            pil_images: list[Image.Image] = []
            hw_list: list[tuple[int, int]] = []
            for path in batch["img_path"]:
                img = Image.open(path).convert("RGB")
                pil_images.append(img)
                hw_list.append((img.height, img.width))
            img_hw = torch.tensor(hw_list, dtype=torch.float32, device=self.device)
            kpt_emb = self.dino_conditioner(
                kpt_pixel=keypoints.to(self.device),
                img_hw=img_hw,
                pil_images=pil_images,
            )  # (B, N, 14, dim_image_cond)

        # Compute root embedding as mean of anchor joints
        j = CROWDPOSE_JOINT_NAMES
        anchor_idx = [j["left_hip"], j["right_hip"], j["neck"]]
        root_emb = kpt_emb[:, :, anchor_idx, :].mean(
            dim=2, keepdim=True
        )  # (B, N, 1, dim_image_cond)

        # Prepend root → (B, N, 15, dim_image_cond), flatten to (B*N*15, dim_image_cond)
        full_emb = torch.cat([root_emb, kpt_emb], dim=2)  # (B, N, 15, dim_image_cond)
        batch["image_cond"] = full_emb.reshape(
            B * N * 15, -1
        )  # (B*N*K, dim_image_cond)

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

        # Score-based visibility proxy (consistent with training threshold)
        batch["keypoints"] = pred_kpts
        batch["visibility"] = (pred_scores > 0.1).float()  # (B, N, 14)

        # Convert predictions to canonical space (root-relative, normalised)
        self.prepare_batch(batch)

        edge_index = batch["edge_index"]  # (2, B*N*E)
        node_idx = batch["kpt_type_idx"].reshape(-1)  # (B*N*K,) values in [0, K-1]
        batch_index = batch["batch"]  # (B*N*K,)
        num_nodes = B * N * K

        T = self.model.config.num_train_steps
        t_start = max(1, round(T * self._inference_noise_frac))

        # Add noise at the inference starting timestep.  Noise level
        # t_start/T keeps the input in-distribution for the model (which was
        # trained on noisy inputs for every t in [0, T-1]).  Using a small
        # fraction (default 10 %) means only minor noise is added, so an
        # undertrained model still produces coordinates close to the detector
        # predictions → AP never collapses to 0 solely due to inference setup.
        t_vec = torch.full((B,), t_start, dtype=torch.long, device=pred_kpts.device)
        x_init_noisy, _ = self.model.add_noise(
            node_pos=batch["node_pos"],
            timestep=t_vec,
            batch_idx=batch_index,
        )  # (B*N*K, 2)

        S = (
            self.num_inference_steps
            if self.num_inference_steps is not None
            else t_start
        )

        pred_flat = self.model.ddim_sample(
            edge_index=edge_index,
            kpt_type_idx=node_idx,
            batch_idx=batch_index,
            num_nodes=num_nodes,
            num_inference_steps=S,
            start_t=t_start,
            device=pred_kpts.device,
            x_init=x_init_noisy,
            image_cond=batch.get("image_cond"),
        )  # (B*N*K, 2)

        # Convert back to pixel space using the root and scale from the batch
        pred_canonical = pred_flat.view(B, N, K, 2)
        pred_pixel = canonical_to_pixel(
            pred_canonical, batch["root_pos"], scale=batch["scale"]
        )

        return pred_pixel, pred_scores, image_ids

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
        keypoints = batch["keypoints"]  # (B, N, 14, 2) pixel space

        self.prepare_batch(batch)

        B, N = keypoints.shape[:2]
        K = 15  # canonical nodes: 14 joints + 1 virtual root

        node_pos = batch["node_pos"]  # (B*N*K, 2) canonical
        node_index = batch["kpt_type_idx"].reshape(-1)  # (B*N*K,) values in [0, K-1]
        batch_index = batch["batch"]  # (B*N*K,)
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
            batch_idx=batch_index,
        )

        pred_delta = self.model.backward_process(
            noisy_node_pos=noisy_node_pos,
            edge_index=edge_index,
            kpt_type_idx=node_index,
            batch_idx=batch_index,
            timestep=timestep,
            image_cond=batch.get("image_cond"),
        )

        # Predicted clean positions in canonical space; drop the virtual root (index 0)
        pred_x0_flat = noisy_node_pos - pred_delta  # (B*N*K, 2)
        pred_x0_joints = pred_x0_flat.view(B, N, K, 2)[
            :, :, 1:, :
        ]  # (B, N, 14, 2) canonical

        # Invert normalisation back to absolute pixel space:
        #   canonical = (pixel - root) / scale  →  pixel = canonical * scale + root
        pred_pixel = pred_x0_joints * scale.view(B, 1, 1, 1) + root_pos.unsqueeze(
            2
        )  # (B, N, 14, 2)

        joint_loss = torch.nn.functional.mse_loss(pred_pixel, keypoints)
        
        # TODO: Add bone length loss

        loss =  joint_loss
        return loss, B

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        loss, B = self._step(batch, stage="train")
        self.log(
            "train_loss", loss, batch_size=B, prog_bar=True, on_step=True, on_epoch=True
        )
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        loss, B = self._step(batch, stage="val")
        self.log(
            "val_loss", loss, batch_size=B, prog_bar=True, on_step=False, on_epoch=True
        )

    def test_step(self, batch: dict, batch_idx: int) -> None:
        loss, B = self._step(batch, stage="test")
        self.log(
            "test_loss", loss, batch_size=B, prog_bar=True, on_step=False, on_epoch=True
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

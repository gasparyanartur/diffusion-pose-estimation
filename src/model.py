"""Graph-based keypoint diffusion model.

Architecture overview
---------------------
Forward process
    Gaussian noise is added to root-relative, normalised keypoint positions.
    The noise scale is proportional to the normalised timestep t/T.

Backward process (denoising)
    A GNN (GAT) receives per-node features built from four sources:

    * position embedding  – linear projection of the noisy (x, y) position
    * keypoint-index embedding – learned embedding of the joint type (0…K-1)
    * timestep embedding  – sinusoidal embedding broadcast from batch to nodes
    * bone-vector embedding – edge bone vectors aggregated (mean) to nodes

    The GNN predicts a 2-D residual for every node (the denoised position).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytorch_lightning as pl
import torch
import torch_geometric as geom
from torch import Tensor
from torch_geometric.utils import scatter

from src.data import prepare_pose_graph_batch
from src.metrics import KeypointAPMetric, canonical_to_pixel


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class KeypointDiffusionConfig:
    """Hyperparameters for :class:`KeypointDiffusionModel`."""

    name: str = "keypoint_diffusion"

    # Feature dimensions
    dim_pos: int = 128    # position-embedding output dim
    dim_index: int = 128  # keypoint-index embedding dim
    dim_time: int = 128   # timestep-embedding dim
    dim_bone: int = 128   # bone-vector embedding dim

    # GNN backbone
    dim_hidden: int = 128
    num_layers: int = 4

    # Diffusion schedule
    num_train_steps: int = 100


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------


class TimestepSinusoidalEmbedding(torch.nn.Module):
    """Sinusoidal embedding for a scalar integer timestep (batch of B values).

    Produces a (B, dim) float tensor whose columns are alternating cosine /
    sine functions of exponentially spaced frequencies — identical to the
    original DDPM / Transformer positional encoding.

    Args:
        dim: Output embedding dimension (must be even).
        max_period: Controls the minimum frequency (default 10 000).
    """

    def __init__(self, dim: int, max_period: int = 10_000) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim must be even, got {dim}")
        self.dim = dim
        self.max_period = max_period

    def forward(self, timestep: Tensor) -> Tensor:
        """Args:
            timestep: LongTensor of shape (B,).

        Returns:
            FloatTensor of shape (B, dim).
        """
        half_dim = self.dim // 2
        freqs = torch.exp(
            -torch.arange(half_dim, dtype=torch.float32, device=timestep.device)
            * np.log(self.max_period)
            / half_dim
        )  # (half_dim,)
        args = timestep[:, None].float() * freqs[None]   # (B, half_dim)
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, dim)


class KeypointDiffusionModel(torch.nn.Module):
    """Graph diffusion model over pose keypoints.

    Args:
        config: Model hyperparameters.
        num_keypoints: Number of keypoints per pose (default 15, canonical
            CrowdPose + virtual root node produced by
            :func:`src.data.prepare_pose_graph_batch`).
    """

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
        in_channels = config.dim_pos + config.dim_index + config.dim_time + config.dim_bone
        self.decoder = geom.nn.GAT(
            in_channels=in_channels,
            hidden_channels=config.dim_hidden,
            num_layers=config.num_layers,
            out_channels=2,
        )

        # DDPM variance-preserving (VP) noise schedule.
        # Timesteps are 1-indexed: t=1 (smallest noise) to t=T (largest noise).
        # alpha_bars has shape (T+1,) with alpha_bars[0]=1.0 (no noise) so
        # that t_prev=0 in DDIM produces the clean sample automatically.
        T      = config.num_train_steps
        betas  = torch.linspace(1e-4, 0.02, T)                          # (T,)
        alphas = 1.0 - betas
        ab     = torch.cumprod(alphas, dim=0)                           # (T,)
        ab     = torch.cat([torch.ones(1), ab])                         # (T+1,) [0..T]
        self.register_buffer("_sqrt_alpha_bars",           ab.sqrt())
        self.register_buffer("_sqrt_one_minus_alpha_bars", (1.0 - ab).clamp(min=0.0).sqrt())

    # ------------------------------------------------------------------
    # Forward / backward diffusion
    # ------------------------------------------------------------------

    def add_noise(
        self,
        node_pos: Tensor,
        timestep: Tensor,
        batch_idx: Tensor,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        """DDPM variance-preserving (VP) forward process.

        Samples the noisy position at timestep *t* directly from the
        closed-form marginal distribution:

        .. math::

            x_t = \\sqrt{\\bar\\alpha_t}\\,x_0 + \\sqrt{1-\\bar\\alpha_t}\\,\\epsilon,
            \\quad \\epsilon \\sim \\mathcal{N}(0, I)

        Args:
            node_pos:  FloatTensor (B*N*K, 2) – clean node positions in
                       canonical space.
            timestep:  LongTensor  (B,)       – integer diffusion timestep
                       in ``[1, num_train_steps]`` (1-indexed).
            batch_idx: LongTensor  (B*N*K,)   – batch index per node (flat).

        Returns:
            noisy_node_pos: (B*N*K, 2)
            noise:          (B*N*K, 2) – the raw unscaled Gaussian noise
                            ``epsilon`` that was added.
        """
        noise = torch.randn_like(node_pos)                            # (B*N*K, 2)
        a = self._sqrt_alpha_bars[timestep][batch_idx].unsqueeze(-1)  # (B*N*K, 1)
        b = self._sqrt_one_minus_alpha_bars[timestep][batch_idx].unsqueeze(-1)  # (B*N*K, 1)
        noisy_node_pos = a * node_pos + b * noise
        return noisy_node_pos, noise

    def backward_process(
        self,
        noisy_node_pos: Tensor,
        edge_index: Tensor,
        kpt_type_idx: Tensor,
        batch_idx: Tensor,
        timestep: Tensor,
        **kwargs,
    ) -> Tensor:
        """Predict the position correction (delta) for every node.

        The model outputs a residual ``delta_x`` such that
        ``x_0_hat = noisy_node_pos - delta_x`` approximates the clean pose.
        Predicting the correction rather than the absolute position biases the
        model towards small, local adjustments.

        Args:
            noisy_node_pos: FloatTensor (B*N*K, 2)  – noisy node positions.
            edge_index:     LongTensor  (2, B*N*E)  – block-diagonal edge index
                            built from global flat node indices.
            kpt_type_idx:   LongTensor  (B*N*K,)   – keypoint type per node,
                            values in [0, K-1].  Fed into the joint-type embedding.
            batch_idx:      LongTensor  (B*N*K,)   – batch index per node (flat).
            timestep:       LongTensor  (B,)        – diffusion timestep.

        Returns:
            FloatTensor (B*N*K, 2) – predicted correction ``delta_x``.
            Apply ``noisy_node_pos - delta_x`` to obtain the denoised positions.
        """
        num_nodes = noisy_node_pos.shape[0]

        # 1. Position embedding
        pos_emb = self.position_embedding(noisy_node_pos)   # (B*N*K, dim_pos)

        # 2. Keypoint-type embedding
        idx_emb = self.node_idx_embedding(kpt_type_idx)     # (B*N*K, dim_index)

        # 3. Timestep embedding broadcast from batch level to node level
        timestep_emb = self.timestep_embedding(timestep)    # (B, dim_time)
        timestep_emb = timestep_emb[batch_idx]              # (B*N*K, dim_time)

        # 4. Bone-vector embedding aggregated back to nodes
        #    Bone vectors are naturally edge features; we mean-pool them per
        #    destination node so they become node features of the right size.
        edge_src = edge_index[0]                                                    # (B*N*E,)
        edge_dst = edge_index[1]                                                    # (B*N*E,)
        bone_vectors = noisy_node_pos[edge_dst] - noisy_node_pos[edge_src]          # (B*N*E, 2)
        bone_emb_edges = self.bone_vector_embedding(bone_vectors)                   # (B*N*E, dim_bone)
        bone_emb = scatter(                                                         # (B*N*K, dim_bone)
            bone_emb_edges, edge_dst, dim=0, dim_size=num_nodes, reduce="mean"
        )

        # 5. Concatenate all node features and run GNN
        x = torch.cat([pos_emb, idx_emb, timestep_emb, bone_emb], dim=-1)  # (B*N*K, in_channels)
        out = self.decoder(x, edge_index)                                   # (B*N*K, 2)
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
    ) -> Tensor:
        """Run DDIM denoising to a clean pose.

        At each step *t* the model predicts the clean positions ``x_0_hat``
        via ``x_0_hat = x_t - delta_x``.  The DDIM deterministic update rule
        for the DDPM variance-preserving (VP) schedule is:

        .. math::

            \\hat{\\epsilon} = \\frac{x_t - \\sqrt{\\bar\\alpha_t}\\hat{x}_0}
                                    {\\sqrt{1-\\bar\\alpha_t}}

            x_{t-1} = \\sqrt{\\bar\\alpha_{t-1}}\\,\\hat{x}_0
                    + \\sqrt{1-\\bar\\alpha_{t-1}}\\,\\hat{\\epsilon}

        Args:
            edge_index:          LongTensor  (2, B*N*E)  – block-diagonal edges
                                 built from global flat node indices.
            kpt_type_idx:        LongTensor  (B*N*K,)   – keypoint type per node,
                                 values in [0, K-1].
            batch_idx:           LongTensor  (B*N*K,)   – batch index per node.
            num_nodes:           Total number of nodes (B*N*K).
            num_inference_steps: Number of denoising steps.  Defaults to
                                 ``start_t`` when *start_t* is given, else to
                                 ``config.num_train_steps``.
            start_t:             First (highest) timestep in the DDIM schedule.
                                 Defaults to ``config.num_train_steps``.  Setting
                                 this below T limits how many steps are run and
                                 bounds the delta accumulation.
            device:              Target device.  Inferred from *edge_index* when
                                 ``None``.
            x_init:              Optional FloatTensor (B*N*K, 2).  When given
                                 the chain starts here instead of from pure
                                 Gaussian noise.  Use this to refine existing
                                 detector predictions.

        Returns:
            FloatTensor (B*N*K, 2) – predicted clean node positions in canonical
            (root-relative, normalised) space.
        """
        T       = self.config.num_train_steps
        t_start = start_t if start_t is not None else T
        S       = num_inference_steps if num_inference_steps is not None else t_start
        device  = device or edge_index.device

        # Build decreasing timestep schedule: [t_start, ..., 1]
        timesteps = torch.linspace(t_start, 1, S, dtype=torch.long, device=device)

        # Start from provided initial estimate or pure Gaussian noise
        if x_init is not None:
            x = x_init.to(device)
        else:
            x = torch.randn(num_nodes, 2, device=device)

        for i, t_scalar in enumerate(timesteps):
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else torch.zeros(1, device=device)[0]

            t_batch = t_scalar.expand(batch_idx.max().item() + 1)  # (B,)

            delta_x = self.backward_process(
                noisy_node_pos=x,
                edge_index=edge_index,
                kpt_type_idx=kpt_type_idx,
                batch_idx=batch_idx,
                timestep=t_batch,
            )  # (B*N*K, 2)
            x_0_hat = x - delta_x

            # DDIM deterministic update — VP schedule.
            # At t_prev=0: a_prev=1, b_prev=0  →  x = x_0_hat  (clean sample). ✓
            t_idx      = t_scalar.long()
            t_prev_idx = t_prev.long()
            a_t    = self._sqrt_alpha_bars[t_idx]
            b_t    = self._sqrt_one_minus_alpha_bars[t_idx]
            a_prev = self._sqrt_alpha_bars[t_prev_idx]
            b_prev = self._sqrt_one_minus_alpha_bars[t_prev_idx]
            eps_hat = (x - a_t * x_0_hat) / b_t
            x = a_prev * x_0_hat + b_prev * eps_hat

        return x


# ---------------------------------------------------------------------------
# Lightning training module
# ---------------------------------------------------------------------------


class PoseDiffusionModule(pl.LightningModule):
    """PyTorch-Lightning wrapper for :class:`KeypointDiffusionModel`.

    Training loop
    -------------
    For each batch:

    1. Build a canonical pose-graph with :func:`src.data.prepare_pose_graph_batch`.
    2. Sample a random diffusion timestep per image.
    3. Add noise to node positions proportional to the timestep.
    4. Run the backward process to predict the clean positions.
    5. Compute MSE loss between predictions and the clean positions.

    Args:
        model: The underlying :class:`KeypointDiffusionModel`.
        lr: Peak learning rate for Adam (default 1e-3).
        weight_decay: AdamW weight-decay coefficient (default 0).
        lr_T_max: Period for cosine annealing in epochs.  If ``None``,
            no LR schedule is applied.
        val_ann_file: Path to the CrowdPose annotation JSON for OKS scoring.
            Required together with *val_pred_cache* to enable AP evaluation.
        test_ann_file: Same as *val_ann_file* but for the test split.
        val_pred_cache: Path to the prediction pickle produced by
            ``scripts/extract_predictions.py``.  When given together with
            *val_ann_file*, DDIM refinement is run over the cached predictions
            at every validation epoch end and OKS-AP is logged.
        test_pred_cache: Same as *val_pred_cache* but for the test split.
        num_inference_steps: Number of DDIM denoising steps used when computing
            AP.  Defaults to ``int(inference_noise_frac * num_train_steps)``.
        ap_batch_size: Batch size for iterating the prediction cache during AP
            evaluation (default 8).
        ap_eval_interval: Run val AP evaluation every this many epochs.  0
            disables val AP entirely (AP is still computed at test time).
            Default 0.
        inference_noise_frac: Fraction of the total diffusion timesteps to use
            as the DDIM starting noise level.  Noise at level
            ``round(T * inference_noise_frac)`` is added to the canonical
            detector predictions before DDIM; the chain runs from that level
            down to t=1.  Smaller values preserve the detector predictions
            more faithfully (lower floor AP, faster inference).  Larger values
            allow bigger corrections but require a more converged model.
            Default 0.1 (10 % of T, ≈ 80 px noise at T=100, scale=800).
    """

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
        self._val_pred_cache  = val_pred_cache
        self._test_pred_cache = test_pred_cache

        self._val_metric  = KeypointAPMetric(val_ann_file)  if val_ann_file  else None
        self._test_metric = KeypointAPMetric(test_ann_file) if test_ann_file else None

    # ------------------------------------------------------------------
    # AP inference helper
    # ------------------------------------------------------------------

    def _ap_step(
        self,
        batch: dict,
        metric: KeypointAPMetric,
    ) -> None:
        """Refine one batch of detector predictions with DDIM and feed results to *metric*."""
        pred_pixel, pred_scores, image_ids = self._refine_batch(batch)
        metric.update(pred_pixel, pred_scores, image_ids)

    def _refine_batch(
        self,
        batch: dict,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run DDIM refinement on one batch of detector predictions.

        The canonical detector predictions are used as the DDIM starting point
        (no noise is added).  This means an untrained model that predicts
        delta≈0 falls back gracefully to the raw detector output, giving
        AP ≈ detector baseline.  A trained model that learns meaningful
        corrections can push AP above the baseline.

        Adding noise before DDIM was tried but found to be harmful: the full
        noise level (σ=1.0 in canonical space ≈ 800 px) requires the model to
        have fully converged before AP becomes non-zero.  Without noise, AP is
        meaningful even during early training.

        Args:
            batch: dict with keys ``pred_keypoints (B,N,14,2)``,
                   ``pred_keypoint_scores (B,N,14)`` and ``image_id (B,)``.

        Returns:
            pred_pixel:  FloatTensor (B, N, 14, 2) – refined pixel keypoints.
            pred_scores: FloatTensor (B, N, 14)    – original detector scores
                         (scores are not modified by the model).
            image_ids:   Tensor (B,)               – COCO image ids.
        """
        pred_kpts   = batch["pred_keypoints"].to(self.device)         # (B, N, 14, 2) pixel
        pred_scores = batch["pred_keypoint_scores"].to(self.device)   # (B, N, 14)
        image_ids   = batch["image_id"]                               # (B,)

        B, N = pred_kpts.shape[:2]
        K    = 15  # canonical keypoints (14 + virtual root)

        # Score-based visibility proxy (consistent with training threshold)
        visibility = (pred_scores > 0.1).float()  # (B, N, 14)

        # Convert predictions to canonical space (root-relative, normalised)
        graph = prepare_pose_graph_batch(pred_kpts, visibility)

        edge_index  = graph["edge_index"]                 # (2, B*N*E)
        node_idx    = graph["kpt_type_idx"].reshape(-1)   # (B*N*K,) values in [0, K-1]
        batch_index = graph["batch"]                      # (B*N*K,)
        num_nodes   = B * N * K

        T       = self.model.config.num_train_steps
        t_start = max(1, round(T * self._inference_noise_frac))

        # Add noise at the inference starting timestep.  Noise level
        # t_start/T keeps the input in-distribution for the model (which was
        # trained on noisy inputs for every t in [0, T-1]).  Using a small
        # fraction (default 10 %) means only minor noise is added, so an
        # undertrained model still produces coordinates close to the detector
        # predictions → AP never collapses to 0 solely due to inference setup.
        t_vec = torch.full((B,), t_start, dtype=torch.long, device=pred_kpts.device)
        x_init_noisy, _ = self.model.add_noise(
            node_pos=graph["node_pos"],
            timestep=t_vec,
            batch_idx=batch_index,
        )  # (B*N*K, 2)

        S = self.num_inference_steps if self.num_inference_steps is not None else t_start

        pred_flat = self.model.ddim_sample(
            edge_index=edge_index,
            kpt_type_idx=node_idx,
            batch_idx=batch_index,
            num_nodes=num_nodes,
            num_inference_steps=S,
            start_t=t_start,
            device=pred_kpts.device,
            x_init=x_init_noisy,
        )  # (B*N*K, 2)

        # Convert back to pixel space using the root and scale from the graph
        pred_canonical = pred_flat.view(B, N, K, 2)
        root_pos   = graph["root_pos"]
        pred_pixel = canonical_to_pixel(pred_canonical, root_pos, scale=graph["scale"])

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
        loader  = dataset.create_dataloader(
            batch_size=self._ap_batch_size,
            shuffle=False,
            num_workers=0,
        )
        metric.reset()
        for batch in tqdm(loader, desc=desc, leave=False, unit="batch"):
            self._ap_step(batch, metric)

    # ------------------------------------------------------------------
    # Shared step
    # ------------------------------------------------------------------

    def _step(self, batch: dict, stage: str) -> Tensor:
        keypoints = batch["keypoints"]   # (B, N, 14, 2) pixel space
        visibility = batch["visibility"] # (B, N, 14)

        pose_graph_batch = prepare_pose_graph_batch(keypoints, visibility)

        B, N = keypoints.shape[:2]
        K = 15  # canonical nodes: 14 joints + 1 virtual root

        node_pos    = pose_graph_batch["node_pos"]                   # (B*N*K, 2) canonical
        node_index  = pose_graph_batch["kpt_type_idx"].reshape(-1)   # (B*N*K,) values in [0, K-1]
        batch_index = pose_graph_batch["batch"]                      # (B*N*K,)
        edge_index  = pose_graph_batch["edge_index"]                 # (2, B*N*E)
        root_pos    = pose_graph_batch["root_pos"]                   # (B, N, 2) pixel-space root
        scale       = pose_graph_batch["scale"]                      # (B,) normalisation divisor

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
        )

        # Predicted clean positions in canonical space; drop the virtual root (index 0)
        pred_x0_flat = noisy_node_pos - pred_delta                          # (B*N*K, 2)
        pred_x0_joints = pred_x0_flat.view(B, N, K, 2)[:, :, 1:, :]       # (B, N, 14, 2) canonical

        # Invert normalisation back to absolute pixel space:
        #   canonical = (pixel - root) / scale  →  pixel = canonical * scale + root
        pred_pixel = pred_x0_joints * scale.view(B, 1, 1, 1) + root_pos.unsqueeze(2)  # (B, N, 14, 2)

        loss = torch.nn.functional.mse_loss(pred_pixel, keypoints)
        return loss, B

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        loss, B = self._step(batch, stage="train")
        self.log("train_loss", loss, batch_size=B, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        loss, B = self._step(batch, stage="val")
        self.log("val_loss", loss, batch_size=B, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch: dict, batch_idx: int) -> None:
        loss, B = self._step(batch, stage="test")
        self.log("test_loss", loss, batch_size=B, prog_bar=True, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        if self._val_metric is None or self._val_pred_cache is None:
            return
        if self.trainer.sanity_checking:
            return
        if self._ap_eval_interval <= 0:
            return
        if (self.current_epoch + 1) % self._ap_eval_interval != 0:
            return
        self._run_ap_eval(self._val_pred_cache, self._val_metric, desc=f"val AP [epoch {self.current_epoch}]")
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

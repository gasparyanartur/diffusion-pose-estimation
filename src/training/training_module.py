"""PyTorch-Lightning wrapper around :class:`PoseEstimator`.

Owns: optimisers, schedulers, training loop, evaluation, augmentation hyper-
parameters, and the conversion between pixel/canonical coordinate systems.
The model itself (extractor + embedder + decoder) is :class:`PoseEstimator`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytorch_lightning as pl
import torch
from pydantic import BaseModel
from torch import Tensor

from src.data import (
    CROWDPOSE_EDGE_LIST,
    PoseGraphBatch,
    PoseSample,
    PredictionDataset,
    prepare_pose_graph_batch,
    prepare_sample,
    read_image_size,
)
from src.decoders.diffusion.base import BaseDiffusionDecoder
from src.metrics import KeypointAPMetric, canonical_to_pixel, compute_mpjpe, compute_pck
from src.pose_estimator import PoseEstimator
from src.training.augmentations import (
    image_dropout,
    joint_dropout,
    joint_randomize,
    keypoint_jitter,
    pose_dropout,
)
from src.training.losses import compute_bone_loss, compute_joint_loss


@dataclass
class StepOutputs:
    pred_pixel: Tensor
    pred_canonical: Tensor
    gt_pixel: Tensor
    gt_canonical: Tensor
    timestep: Tensor


class AugmentationConfig(BaseModel):
    aug_jitter: bool = True
    p_jitter_px: float = 4.0
    aug_joint_drop: bool = True
    p_joint_drop: float = 0.1
    aug_pose_drop: bool = True
    p_pose_drop: float = 0.1
    aug_joint_rand: bool = True
    p_joint_rand: float = 0.1
    aug_img_drop: bool = True
    p_img_drop: float = 0.1


class LossConfig(BaseModel):
    use_joint_loss: bool = True
    w_joint_loss: float = 1.0
    use_bone_loss: bool = True
    w_bone_loss: float = 1.0


def _read_img_wh(img_paths: list[str], B: int, device: torch.device) -> Tensor:
    wh: list[tuple[float, float]] = []
    for i in range(B):
        path = img_paths[i] if i < len(img_paths) else ""
        try:
            h, w = read_image_size(path)
            wh.append((float(w), float(h)))
        except Exception:
            wh.append((800.0, 800.0))
    return torch.tensor(wh, dtype=torch.float32, device=device)


class PoseEstimatorTrainingModule(pl.LightningModule):
    """Lightning wrapper for :class:`PoseEstimator` with diffusion-style training."""

    def __init__(
        self,
        pose_estimator: PoseEstimator,
        augmentations: AugmentationConfig | None = None,
        losses: LossConfig | None = None,
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
    ) -> None:
        super().__init__()
        self.pose_estimator = pose_estimator
        self.aug_cfg = augmentations or AugmentationConfig()
        self.loss_cfg = losses or LossConfig()
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

        decoder = self.pose_estimator.decoder
        if not isinstance(decoder, BaseDiffusionDecoder):
            raise TypeError(
                "PoseEstimatorTrainingModule currently expects a diffusion decoder; "
                f"got {type(decoder).__name__}"
            )
        self._decoder: BaseDiffusionDecoder = decoder
        
        self.save_hyperparameters(ignore=["pose_estimator"])


    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _canonical_to_pixel(self, canonical: Tensor, img_wh: Tensor) -> Tensor:
        return (canonical + 1) / 2 * img_wh[:, None, None, :]

    def prepare_batch(
        self, batch: dict, augment: bool = False
    ) -> tuple[PoseSample, PoseGraphBatch]:
        sample = prepare_sample(batch)
        if augment and self.aug_cfg.aug_jitter:
            sample.keypoints = keypoint_jitter(sample.keypoints, self.aug_cfg.p_jitter_px)
        img_wh = _read_img_wh(sample.img_path, sample.B, sample.keypoints.device)
        graph = prepare_pose_graph_batch(sample, img_size=img_wh)
        if augment:
            self._augment_graph(graph)
        return sample, graph

    def _augment_graph(self, graph: PoseGraphBatch) -> None:
        cfg = self.aug_cfg
        if cfg.aug_joint_drop:
            graph.canonical_visibility = joint_dropout(
                graph.node_pos, graph.canonical_visibility, cfg.p_joint_drop
            )
        if cfg.aug_pose_drop:
            graph.canonical_visibility = pose_dropout(
                graph.node_pos, graph.canonical_visibility, cfg.p_pose_drop
            )
        if cfg.aug_joint_rand:
            joint_randomize(graph.node_pos, graph.canonical_visibility, cfg.p_joint_rand)

    def _maybe_drop_image_cond(self, embeddings: dict[str, Tensor]) -> None:
        node_feats = embeddings.get("node_feats")
        if node_feats is None or not self.aug_cfg.aug_img_drop:
            return
        flat = node_feats.reshape(-1, node_feats.shape[-1])
        image_dropout(flat, self.aug_cfg.p_img_drop)

    def _encode_at(
        self,
        sample: PoseSample,
        graph: PoseGraphBatch,
        kpt_pixel: Tensor,
    ) -> dict[str, Tensor]:
        image_sizes = [read_image_size(p) for p in sample.img_path]
        img_hw = torch.tensor(image_sizes, dtype=torch.float32, device=self.device)
        joint_mask = (graph.canonical_visibility == 0).to(self.device)
        return self.pose_estimator.encode(
            graph=graph,
            image_ids=sample.image_id,
            kpt_pixel=kpt_pixel.to(self.device),
            img_hw=img_hw,
            joint_mask=joint_mask,
        )

    def get_loss(
        self, sample: PoseSample, graph: PoseGraphBatch, outputs: StepOutputs
    ) -> dict[str, Tensor]:
        cfg = self.loss_cfg
        visibility = sample.visibility
        losses: dict[str, Tensor] = {}
        if cfg.use_joint_loss:
            losses["joint_loss"] = (
                compute_joint_loss(outputs.pred_pixel, outputs.gt_pixel, visibility)
                * cfg.w_joint_loss
            )
        if cfg.use_bone_loss:
            losses["bone_loss"] = (
                compute_bone_loss(
                    outputs.pred_pixel, outputs.gt_pixel, visibility, CROWDPOSE_EDGE_LIST
                )
                * cfg.w_bone_loss
            )
        if not losses:
            raise ValueError("All losses disabled; enable at least one in LossConfig")
        losses["loss"] = sum(losses.values())
        return losses

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------
    def _step(
        self,
        batch: dict,
        stage: str,
        generator: torch.Generator | None = None,
    ) -> tuple[dict[str, Tensor], StepOutputs, PoseSample, PoseGraphBatch]:
        sample, graph = self.prepare_batch(batch, augment=(stage == "train"))

        timestep = torch.randint(
            1,
            self._decoder.num_train_steps + 1,
            (graph.B,),
            device=graph.node_pos.device,
        )
        noisy_node_pos, _ = self._decoder.add_noise(
            node_pos=graph.node_pos,
            timestep=timestep,
            batch_idx=graph.node_batch,
            generator=generator,
        )

        # Sample features at the noisy (not GT) keypoint locations.
        noisy_kpt_pixel = self._canonical_to_pixel(
            noisy_node_pos.view(graph.B, graph.N, graph.K, 2), graph.img_wh
        )
        embeddings = self._encode_at(sample, graph, noisy_kpt_pixel)
        if stage == "train":
            self._maybe_drop_image_cond(embeddings)

        delta = self._decoder.predict_delta(noisy_node_pos, graph, embeddings, timestep)
        pred_canonical = (noisy_node_pos - delta).view(graph.B, graph.N, graph.K, 2)
        pred_pixel = self._canonical_to_pixel(pred_canonical, graph.img_wh)
        gt_canonical = sample.gt_keypoints / graph.img_wh[:, None, None, :] * 2 - 1

        outputs = StepOutputs(
            pred_pixel=pred_pixel,
            pred_canonical=pred_canonical,
            gt_pixel=sample.gt_keypoints,
            gt_canonical=gt_canonical,
            timestep=timestep,
        )
        losses = self.get_loss(sample, graph, outputs)
        return losses, outputs, sample, graph

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        losses, _, _, graph = self._step(batch, stage="train")
        log_kwargs = dict(batch_size=graph.B, on_step=True, on_epoch=True)
        self.log("train/loss", losses["loss"], prog_bar=True, **log_kwargs)
        for name, value in losses.items():
            if name != "loss":
                self.log(f"train/{name}", value, **log_kwargs)
        return losses["loss"]

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        gen = torch.Generator(device=self.device).manual_seed(batch_idx)
        losses, outputs, sample, graph = self._step(batch, stage="val", generator=gen)
        self._log_eval(losses, outputs, sample, graph, "val")

    def test_step(self, batch: dict, batch_idx: int) -> None:
        gen = torch.Generator(device=self.device).manual_seed(batch_idx)
        losses, outputs, sample, graph = self._step(batch, stage="test", generator=gen)
        self._log_eval(losses, outputs, sample, graph, "test")

    def _log_eval(
        self,
        losses: dict[str, Tensor],
        outputs: StepOutputs,
        sample: PoseSample,
        graph: PoseGraphBatch,
        prefix: str,
    ) -> None:
        log_kwargs = dict(batch_size=graph.B, on_step=False, on_epoch=True)
        self.log(f"{prefix}/loss", losses["loss"], prog_bar=True, **log_kwargs)
        for name, value in losses.items():
            if name != "loss":
                self.log(f"{prefix}/{name}", value, **log_kwargs)
        self.log(
            f"{prefix}/mpjpe_px",
            compute_mpjpe(outputs.pred_pixel, outputs.gt_pixel, sample.visibility),
            **log_kwargs,
        )
        self.log(
            f"{prefix}/pck_20px",
            compute_pck(
                outputs.pred_pixel, outputs.gt_pixel, sample.visibility, threshold_px=20.0
            ),
            **log_kwargs,
        )

    # ------------------------------------------------------------------
    # AP eval (full DDIM sampling)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _refine_prediction_batch(
        self,
        batch: dict,
        num_steps: int | None = None,
        noise_frac: float | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        pred_kpts = batch["pred_keypoints"].to(self.device)
        pred_scores = batch["pred_keypoint_scores"].to(self.device)
        image_ids = batch["image_id"]

        batch["keypoints"] = pred_kpts
        batch["visibility"] = (pred_scores > 0.1).float()
        sample, graph = self.prepare_batch(batch)

        T = self._decoder.num_train_steps
        frac = noise_frac if noise_frac is not None else self._start_inference_noise_frac
        t_start = max(1, round(T * frac))

        if self._raw_refiner:
            x_init = graph.node_pos
            n_steps = 1
        else:
            if self._disable_noise:
                x_init = graph.node_pos
            else:
                noise_t = torch.full(
                    (graph.B,), t_start, dtype=torch.long, device=pred_kpts.device
                )
                x_init, _ = self._decoder.add_noise(
                    node_pos=graph.node_pos,
                    timestep=noise_t,
                    batch_idx=graph.node_batch,
                )
            default_steps = (
                self.num_inference_steps if self.num_inference_steps is not None else t_start
            )
            n_steps = num_steps if num_steps is not None else default_steps

        noisy_kpt_pixel = self._canonical_to_pixel(
            x_init.view(graph.B, graph.N, graph.K, 2), graph.img_wh
        )
        embeddings = self._encode_at(sample, graph, noisy_kpt_pixel)

        nodes_canonical_flat = self._decoder.ddim_sample(
            graph=graph,
            embeddings=embeddings,
            num_nodes=graph.num_nodes,
            num_inference_steps=n_steps,
            start_t=t_start,
            device=pred_kpts.device,
            x_init=x_init,
            guidance_scale=self.cfg_guidance_scale,
        )
        nodes_canonical = nodes_canonical_flat.view(graph.B, graph.N, graph.K, 2)
        return canonical_to_pixel(nodes_canonical, graph.img_wh), pred_scores, image_ids

    @torch.no_grad()
    def _iterative_refine_batch(
        self, batch: dict, n_passes: int
    ) -> tuple[Tensor, Tensor, Tensor]:
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

    def _ap_step(
        self, batch: dict, metric: KeypointAPMetric, num_steps: int | None = None
    ) -> None:
        n_passes = num_steps if num_steps is not None else 1
        pred_pixel, pred_scores, image_ids = self._iterative_refine_batch(batch, n_passes=n_passes)
        bbox_scores = batch.get("bbox_scores")
        metric.update(pred_pixel, pred_scores, image_ids, bbox_scores=bbox_scores)

    def _run_ap_eval(
        self,
        pred_cache: str,
        metric: KeypointAPMetric,
        desc: str = "AP eval",
        num_steps: int | None = None,
    ) -> None:
        from tqdm import tqdm

        dataset = PredictionDataset.from_path(pred_cache)
        loader = dataset.create_dataloader(
            batch_size=self._ap_batch_size, shuffle=False, num_workers=0
        )
        metric.reset()
        for batch in tqdm(loader, desc=desc, leave=False, unit="batch"):
            self._ap_step(batch, metric, num_steps=num_steps)

    def _run_and_log_ap(
        self,
        pred_cache: str,
        metric: KeypointAPMetric,
        prefix: str,
        epoch: int | None = None,
        log_prog_bar: bool = False,
    ) -> None:
        for n in (self._ap_step_counts or [None]):
            suffix = f"@{n}" if n is not None else ""
            desc = f"{prefix} AP{suffix}" + (f" [epoch {epoch}]" if epoch is not None else "")
            self._run_ap_eval(pred_cache, metric, desc=desc, num_steps=n)
            for key, val in metric.compute().items():
                self.log(
                    f"{prefix}/{key}{suffix}",
                    val,
                    prog_bar=(log_prog_bar and key == "crowdpose/AP"),
                    sync_dist=True,
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
        self._run_and_log_ap(
            self._val_pred_cache,
            self._val_metric,
            prefix="ap-val",
            epoch=self.current_epoch,
            log_prog_bar=True,
        )

    def on_test_epoch_end(self) -> None:
        if self._test_metric is None or self._test_pred_cache is None:
            return
        self._run_and_log_ap(self._test_pred_cache, self._test_metric, prefix="ap-test")

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or name.endswith(".bias"):
                no_decay.append(param)
            else:
                decay.append(param)
        param_groups = [
            {"params": decay, "weight_decay": self.weight_decay},
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

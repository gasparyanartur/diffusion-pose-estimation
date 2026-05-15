"""Polymorphic :class:`TrainingModuleConfig` that builds a Lightning module.

The training config owns optimiser, schedule, augmentations, losses, AP-eval
settings, AND trainer-level fields (max_epochs, grad_clip, etc). The data
batch size lives on :class:`DataConfig` because it is a property of the data
loader; the training script reads it from the data module when needed.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import pytorch_lightning as pl
from pydantic import Field

from src.base_config import BaseConfig
from src.training.backbone_probe import BackboneProbeModule, ProbeHead
from src.training.losses import SetPredictionLossWeights
from src.training.set_prediction_module import SetPredictionTrainingModule
from src.training.training_module import (
    AugmentationConfig,
    LossConfig,
    PoseEstimatorTrainingModule,
)

if TYPE_CHECKING:
    from src.data.data_module import PoseDataModule
    from src.features.base import BaseFeatureExtractor
    from src.pose_estimator import PoseEstimator


class TrainingModuleConfig(BaseConfig):
    """Base polymorphic training-module config. Override ``build()``."""

    @abstractmethod
    def build(
        self,
        pose_estimator: "PoseEstimator",
        data: "PoseDataModule",
    ) -> pl.LightningModule:
        ...


class DiffusionTrainingModuleConfig(TrainingModuleConfig):
    """Builds :class:`PoseEstimatorTrainingModule` with all training params."""

    # optimiser / schedule
    lr: float = 1e-3
    weight_decay: float = 1e-4
    cosine_lr: bool = True
    lr_warmup_epochs: int = 0
    lr_min_frac: float = 1e-1
    cfg_guidance_scale: float = 1.0

    # AP / inference
    num_inference_steps: int | None = 5
    ap_eval_interval: int = 0
    ap_step_counts: list[int] | None = None
    start_inference_noise_frac: float = 0.4
    raw_refiner: bool = False
    disable_noise: bool = False

    # trainer-level
    max_epochs: int = 200
    seed: int = 42
    grad_clip: float = 1.0
    early_stop_patience: int = 0
    log_every_n_steps: int = 50
    compile: bool = False
    running_loss_window: int = 5
    resume: str | None = None

    augmentations: AugmentationConfig = Field(default_factory=AugmentationConfig)
    losses: LossConfig = Field(default_factory=LossConfig)

    def build(
        self,
        pose_estimator: "PoseEstimator",
        data: "PoseDataModule",
    ) -> PoseEstimatorTrainingModule:
        return PoseEstimatorTrainingModule(
            pose_estimator=pose_estimator,
            augmentations=self.augmentations,
            losses=self.losses,
            lr=self.lr,
            weight_decay=self.weight_decay,
            lr_T_max=self.max_epochs if self.cosine_lr else None,
            lr_warmup_epochs=self.lr_warmup_epochs,
            lr_min_frac=self.lr_min_frac,
            cfg_guidance_scale=self.cfg_guidance_scale,
            val_ann_file=str(data.val_ann_path),
            test_ann_file=str(data.test_ann_path),
            val_pred_cache=data.val_pred_cache_resolved,
            test_pred_cache=data.test_pred_cache_resolved,
            num_inference_steps=self.num_inference_steps,
            ap_batch_size=data.ap_batch_size,
            ap_eval_interval=self.ap_eval_interval,
            ap_step_counts=list(self.ap_step_counts) if self.ap_step_counts else None,
            start_inference_noise_frac=self.start_inference_noise_frac,
            raw_refiner=self.raw_refiner,
            disable_noise=self.disable_noise,
        )


class SetPredictionLossWeightsConfig(BaseConfig):
    """Pydantic mirror of :class:`SetPredictionLossWeights`."""

    oks: float = 1.0
    obj: float = 1.0
    vis: float = 0.5
    rank: float = 1.0
    matcher_lambda_oks: float = 1.0
    matcher_lambda_obj: float = 0.1

    def to_dataclass(self) -> SetPredictionLossWeights:
        return SetPredictionLossWeights(
            oks=self.oks,
            obj=self.obj,
            vis=self.vis,
            rank=self.rank,
            matcher_lambda_oks=self.matcher_lambda_oks,
            matcher_lambda_obj=self.matcher_lambda_obj,
        )


class SetPredictionTrainingModuleConfig(TrainingModuleConfig):
    """Builds :class:`SetPredictionTrainingModule`."""

    lr: float = 1e-3
    weight_decay: float = 1e-4
    cosine_lr: bool = True
    lr_warmup_epochs: int = 0
    lr_min_frac: float = 1e-1

    max_epochs: int = 50
    seed: int = 42
    grad_clip: float = 5.0
    early_stop_patience: int = 0
    log_every_n_steps: int = 50
    compile: bool = False
    resume: str | None = None
    gt_sanity_check: bool = False

    weights: SetPredictionLossWeightsConfig = Field(
        default_factory=SetPredictionLossWeightsConfig,
    )

    def build(
        self,
        pose_estimator: "PoseEstimator",
        data: "PoseDataModule",
    ) -> SetPredictionTrainingModule:
        return SetPredictionTrainingModule(
            pose_estimator=pose_estimator,
            weights=self.weights.to_dataclass(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            lr_T_max=self.max_epochs if self.cosine_lr else None,
            lr_warmup_epochs=self.lr_warmup_epochs,
            lr_min_frac=self.lr_min_frac,
            val_ann_file=str(data.val_ann_path),
            test_ann_file=str(data.test_ann_path),
            num_keypoints=data.num_keypoints,
            grad_clip=self.grad_clip,
            gt_sanity_check=self.gt_sanity_check,
            max_persons=getattr(data, "max_persons", None),
        )


class BackboneProbeConfig(BaseConfig):
    """Builds :class:`BackboneProbeModule`.

    The probe head and slot decoder are constructed internally from the
    parameters below; ``backbone`` is passed in from the calling script.
    """

    # optimiser / schedule
    lr: float = 1e-3
    weight_decay: float = 1e-3
    cosine_lr: bool = True
    lr_warmup_epochs: int = 0
    lr_min_frac: float = 1e-1

    # trainer
    max_epochs: int = 50
    seed: int = 42
    grad_clip: float = 5.0
    log_every_n_steps: int = 50
    resume: str | None = None
    gt_sanity_check: bool = False

    # probe head
    feature_dim: int = 256
    proj_dim: int = 768
    conv_dim: int = 96
    target_size: int = 16
    num_conv_layers: int = 5
    num_attn_layers: int = 2
    num_heads: int = 4
    ffn_mult: int = 4
    conv_dropout: float = 0.1
    attn_dropout: float = 0.3
    ffn_dropout: float = 0.5
    feature_key: str = "spatial"

    # slot decoder
    hidden_dim: int = 256
    num_slots: int = 20
    proj_dropout: float = 0.2

    # augmentation
    noise_std: float = 0.05
    brightness: float = 0.3
    contrast: float = 0.3
    saturation: float = 0.2

    # evaluation
    val_ap_every_n_epochs: int = 10

    # loss
    weights: SetPredictionLossWeightsConfig = Field(
        default_factory=SetPredictionLossWeightsConfig,
    )

    def build(
        self,
        backbone: "BaseFeatureExtractor",
        data: "PoseDataModule",
    ) -> BackboneProbeModule:
        probe_head = ProbeHead(
            in_channels=backbone.out_channels,
            feature_dim=self.feature_dim,
            proj_dim=self.proj_dim,
            conv_dim=self.conv_dim,
            target_size=self.target_size,
            num_conv_layers=self.num_conv_layers,
            num_attn_layers=self.num_attn_layers,
            num_heads=self.num_heads,
            ffn_mult=self.ffn_mult,
            conv_dropout=self.conv_dropout,
            attn_dropout=self.attn_dropout,
            ffn_dropout=self.ffn_dropout,
            num_slots=self.num_slots,
            num_keypoints=data.num_keypoints,
            hidden_dim=self.hidden_dim,
            proj_dropout=self.proj_dropout,
            feature_key=self.feature_key,
        )
        return BackboneProbeModule(
            backbone=backbone,
            probe_head=probe_head,
            weights=self.weights.to_dataclass(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            lr_T_max=self.max_epochs if self.cosine_lr else None,
            lr_warmup_epochs=self.lr_warmup_epochs,
            lr_min_frac=self.lr_min_frac,
            val_ann_file=str(data.val_ann_path),
            test_ann_file=str(data.test_ann_path),
            num_keypoints=data.num_keypoints,
            grad_clip=self.grad_clip,
            val_ap_every_n_epochs=self.val_ap_every_n_epochs,
            gt_sanity_check=self.gt_sanity_check,
            max_persons=getattr(data, "max_persons", None),
            noise_std=self.noise_std,
            brightness=self.brightness,
            contrast=self.contrast,
            saturation=self.saturation,
        )


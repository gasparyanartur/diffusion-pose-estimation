"""Tensor-native fixed-slot pose-estimation pipeline."""

from __future__ import annotations

import pytorch_lightning as pl
import torch
from pydantic import Field
from torch import Tensor, nn

from src.configs.base_config import BaseConfig
from src.training.losses import (
    SetPredictionLoss,
    SetPredictionLossWeights,
    compute_oks,
    hungarian_match,
)
from src.eval.metrics import CROWDPOSE_SIGMAS, KeypointAPMetric, canonical_to_pixel
from src.models.features.base import (
    BaseFeatureExtractor,
    BaseFeatureExtractorConfig,
)
from src.models.common.positional import make_2d_sinusoidal_pe
from src.training.optim import build_adamw_with_schedule


class _SpatialAttentionPool(nn.Module):
    def __init__(self, in_channels: int, feature_dim: int, num_heads: int) -> None:
        super().__init__()
        if feature_dim % 4:
            raise ValueError(f"feature_dim must be divisible by 4; got {feature_dim}")
        if feature_dim % num_heads:
            raise ValueError("feature_dim must be divisible by num_heads")
        self.project = nn.Conv2d(in_channels, feature_dim, kernel_size=1)
        self.attention = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(feature_dim)
        self.feature_dim = feature_dim

    def forward(self, spatial_features: Tensor) -> Tensor:
        batch_size, _, height, width = spatial_features.shape
        tokens = self.project(spatial_features).flatten(2).permute(0, 2, 1)
        positional_encoding = make_2d_sinusoidal_pe(
            height, width, self.feature_dim, tokens.device, tokens.dtype
        )
        tokens = tokens + positional_encoding.unsqueeze(0)
        attended_tokens, _ = self.attention(tokens, tokens, tokens)
        return self.norm(tokens + attended_tokens).mean(dim=1).reshape(batch_size, -1)


class _FixedSlotPredictionHead(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_slots: int,
        num_keypoints: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.num_keypoints = num_keypoints
        output_dim = num_slots * (1 + num_keypoints * 3)
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, pooled_features: Tensor) -> dict[str, Tensor]:
        batch_size = pooled_features.shape[0]
        num_slots = self.num_slots
        num_keypoints = self.num_keypoints
        output = self.network(pooled_features)
        scores_logits = output[:, :num_slots]
        keypoint_start = num_slots
        keypoint_end = keypoint_start + num_slots * num_keypoints * 2
        keypoint_logits = output[:, keypoint_start:keypoint_end].view(
            batch_size, num_slots, num_keypoints, 2
        )
        vis_logits = output[:, keypoint_end:].view(batch_size, num_slots, num_keypoints)
        return {
            "keypoints": torch.sigmoid(keypoint_logits) * 2.0 - 1.0,
            "scores": torch.sigmoid(scores_logits),
            "scores_logits": scores_logits,
            "keypoint_scores": torch.sigmoid(vis_logits),
            "vis_logits": vis_logits,
        }


class BaselinePosePipeline(pl.LightningModule):
    """End-to-end fixed-slot baseline pose estimator."""

    def __init__(
        self,
        *,
        backbone: BaseFeatureExtractor,
        spatial_channels: int,
        weights: SetPredictionLossWeights | None = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        lr_T_max: int | None = None,
        lr_warmup_epochs: int = 0,
        lr_min_frac: float = 1e-1,
        val_ann_file: str | None = None,
        test_ann_file: str | None = None,
        max_persons: int | None = None,
        feature_dim: int = 256,
        num_heads: int = 8,
        hidden_dim: int = 512,
        num_slots: int = 20,
        num_keypoints: int = 14,
        mlp_dropout: float = 0.0,
        feature_key: str = "spatial",
        gt_sanity_check: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.feature_key = feature_key
        self.pool = _SpatialAttentionPool(spatial_channels, feature_dim, num_heads)
        self.head = _FixedSlotPredictionHead(
            feature_dim, hidden_dim, num_slots, num_keypoints, mlp_dropout
        )
        self.weights = weights or SetPredictionLossWeights()
        self._loss = SetPredictionLoss(
            self.weights, torch.from_numpy(CROWDPOSE_SIGMAS).float()
        )
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_T_max = lr_T_max
        self.lr_warmup_epochs = lr_warmup_epochs
        self.lr_min_frac = lr_min_frac
        self.gt_sanity_check = gt_sanity_check
        self._val_metric = (
            KeypointAPMetric(val_ann_file, max_persons=max_persons) if val_ann_file else None
        )
        self._test_metric = (
            KeypointAPMetric(test_ann_file, max_persons=max_persons) if test_ann_file else None
        )
        self.save_hyperparameters(ignore=["backbone", "weights"])

    @property
    def num_slots(self) -> int:
        return self.head.num_slots

    @property
    def num_keypoints(self) -> int:
        return self.head.num_keypoints

    def forward(self, images: Tensor) -> dict[str, Tensor]:
        features = self.backbone(image=images)
        spatial_features = features[self.feature_key]
        if not isinstance(spatial_features, Tensor) or spatial_features.ndim != 4:
            raise ValueError(
                f"baseline backbones must return features['{self.feature_key}'] as [B, C, H, W]"
            )
        return self.head(self.pool(spatial_features))

    def _make_ground_truth_prediction(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        keypoints = batch["gt_keypoints"].to(self.device)
        visible = batch["gt_visible"].to(self.device)
        person_mask = batch["person_mask"].to(self.device)
        batch_size, max_people, num_keypoints, _ = keypoints.shape
        prediction_keypoints = torch.zeros(
            batch_size, self.num_slots, num_keypoints, 2, device=self.device
        )
        vis_logits = torch.full(
            (batch_size, self.num_slots, num_keypoints), -10.0, device=self.device
        )
        scores_logits = torch.full((batch_size, self.num_slots), -10.0, device=self.device)
        filled_slots = min(self.num_slots, max_people)
        prediction_keypoints[:, :filled_slots] = keypoints[:, :filled_slots]
        vis_logits[:, :filled_slots] = visible[:, :filled_slots] * 20.0 - 10.0
        scores_logits[:, :filled_slots] = person_mask[:, :filled_slots] * 20.0 - 10.0
        return {
            "keypoints": prediction_keypoints,
            "scores": torch.sigmoid(scores_logits),
            "scores_logits": scores_logits,
            "keypoint_scores": torch.sigmoid(vis_logits),
            "vis_logits": vis_logits,
        }

    def _predict_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.gt_sanity_check:
            return self._make_ground_truth_prediction(batch)
        return self(batch["image"].to(self.device))

    def _compute_loss(self, predictions: dict[str, Tensor], batch: dict[str, Tensor]) -> dict[str, Tensor]:
        return self._loss(predictions, {
            "gt_keypoints": batch["gt_keypoints"],
            "gt_labeled": batch["gt_labeled"],
            "gt_visible": batch["gt_visible"],
            "person_scale": batch["person_scale"],
            "person_mask": batch["person_mask"],
        })

    def _log_losses(
        self, prefix: str, losses: dict[str, Tensor], batch_size: int, on_step: bool
    ) -> None:
        self.log(
            f"{prefix}/loss", losses["total"], prog_bar=True, batch_size=batch_size,
            on_step=on_step, on_epoch=True,
        )
        for name in (
            "oks_loss", "objectness_loss", "visibility_loss", "ranking_loss", "mean_oks",
        ):
            self.log(
                f"{prefix}/{name}", losses[name], batch_size=batch_size,
                on_step=on_step, on_epoch=True,
            )

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        del batch_idx
        predictions = self._predict_batch(batch)
        losses = self._compute_loss(predictions, batch)
        self._log_losses("train", losses, batch["image"].shape[0], True)
        return losses["total"]

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        del batch_idx
        predictions = self._predict_batch(batch)
        losses = self._compute_loss(predictions, batch)
        self._log_losses("val", losses, batch["image"].shape[0], False)

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        del batch_idx
        predictions = self._predict_batch(batch)
        losses = self._compute_loss(predictions, batch)
        self._log_losses("test", losses, batch["image"].shape[0], False)
        if self._test_metric is not None:
            self._update_ap_metric(self._test_metric, predictions, batch)

    def _update_ap_metric(
        self, metric: KeypointAPMetric, predictions: dict[str, Tensor], batch: dict[str, Tensor]
    ) -> None:
        image_wh = batch["img_hw"].to(self.device)[:, [1, 0]]
        metric.update(
            pred_kpts=canonical_to_pixel(predictions["keypoints"], image_wh).float(),
            pred_scores=predictions["keypoint_scores"].float(),
            image_ids=batch["image_id"].to(self.device),
            bbox_scores=predictions["scores"].float(),
        )

    def on_test_epoch_end(self) -> None:
        if self._test_metric is None:
            return
        for name, value in self._test_metric.compute().items():
            self.log(f"test/{name}", value, sync_dist=True)
        self._test_metric.reset()

    @torch.no_grad()
    def matched_oks(self, predictions: dict[str, Tensor], batch: dict[str, Tensor]) -> Tensor:
        values: list[Tensor] = []
        for index in range(predictions["keypoints"].shape[0]):
            people = int(batch["person_mask"][index].sum().item())
            if people == 0:
                continue
            oks = compute_oks(
                predictions["keypoints"][index],
                batch["gt_keypoints"][index, :people].to(self.device),
                batch["gt_labeled"][index, :people].to(self.device),
                batch["person_scale"][index, :people].to(self.device),
                self._loss.sigmas,
            )
            slot_indices, person_indices = hungarian_match(
                oks,
                predictions["scores_logits"][index],
                lambda_oks=self.weights.matcher_lambda_oks,
                lambda_obj=self.weights.matcher_lambda_obj,
            )
            if slot_indices.numel():
                values.append(oks[slot_indices, person_indices])
        if not values:
            return torch.zeros((), device=self.device)
        return torch.cat(values).mean()

    def configure_optimizers(self):
        return build_adamw_with_schedule(
            self,
            lr=self.lr,
            weight_decay=self.weight_decay,
            lr_T_max=self.lr_T_max,
            lr_warmup_epochs=self.lr_warmup_epochs,
            lr_min_frac=self.lr_min_frac,
        )


class BaselinePoseLossWeightsConfig(BaseConfig):
    oks: float = 1.0
    obj: float = 1.0
    vis: float = 0.5
    rank: float = 1.0
    matcher_lambda_oks: float = 1.0
    matcher_lambda_obj: float = 0.1

    def to_dataclass(self) -> SetPredictionLossWeights:
        return SetPredictionLossWeights(**self.model_dump())


class BaselinePosePipelineConfig(BaseConfig):
    backbone: BaseFeatureExtractorConfig
    lr: float = 1e-3
    weight_decay: float = 1e-4
    cosine_lr: bool = True
    lr_warmup_epochs: int = 0
    lr_min_frac: float = 1e-1
    feature_dim: int = 256
    num_heads: int = 8
    hidden_dim: int = 512
    num_slots: int = 20
    num_keypoints: int = 14
    mlp_dropout: float = 0.0
    feature_key: str = "spatial"
    gt_sanity_check: bool = False
    weights: BaselinePoseLossWeightsConfig = Field(default_factory=BaselinePoseLossWeightsConfig)

    def build(
        self, *, val_ann_file: str | None = None, test_ann_file: str | None = None,
        max_persons: int | None = None,
        max_epochs: int,
    ) -> BaselinePosePipeline:
        backbone = self.backbone.build()
        try:
            spatial_channels = backbone.out_channels
        except AttributeError as exc:
            raise ValueError("baseline backbones must expose `out_channels`") from exc
        return BaselinePosePipeline(
            backbone=backbone,
            spatial_channels=spatial_channels,
            weights=self.weights.to_dataclass(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            lr_T_max=max_epochs if self.cosine_lr else None,
            lr_warmup_epochs=self.lr_warmup_epochs,
            lr_min_frac=self.lr_min_frac,
            val_ann_file=val_ann_file,
            test_ann_file=test_ann_file,
            max_persons=max_persons,
            feature_dim=self.feature_dim,
            num_heads=self.num_heads,
            hidden_dim=self.hidden_dim,
            num_slots=self.num_slots,
            num_keypoints=self.num_keypoints,
            mlp_dropout=self.mlp_dropout,
            feature_key=self.feature_key,
            gt_sanity_check=self.gt_sanity_check,
        )

"""Tensor-native graph-pose diffusion pipeline."""

from __future__ import annotations

import math
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pydantic import Field
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

from src.configs.base_config import BaseConfig
from src.data.augmentations import add_gaussian_noise, color_jitter
from src.data.graph import create_graph_from_keypoints
from src.data.skeleton import CANONICAL_EDGE_INDEX
from src.training.losses import GraphPipelineLoss, GraphPipelineLossWeights
from src.eval.metrics import KeypointAPMetric, canonical_to_pixel
from src.models.common.attention import FourierFeatures, TimeFiLMTransformerBlock
from src.models.common.conditioning import FiLMLayer
from src.models.features.base import (
    BaseFeatureExtractor,
    BaseFeatureExtractorConfig,
)
from src.models.common.positional import make_pe_from_centers
from src.training.optim import build_adamw_with_schedule


def _boxes_pixel_to_canonical(boxes_xywh: Tensor, img_hw: Tensor) -> Tensor:
    """Convert pixel xywh boxes to canonical cxcywh boxes."""
    height = img_hw[:, 0:1]
    width = img_hw[:, 1:2]
    x, y, box_width, box_height = boxes_xywh.unbind(-1)
    return torch.stack(
        [
            (x + box_width / 2) / width * 2 - 1,
            (y + box_height / 2) / height * 2 - 1,
            box_width / width * 2,
            box_height / height * 2,
        ],
        dim=-1,
    )


def _normalize_batch(batch: dict[str, Tensor]) -> dict[str, Tensor]:
    """Provide canonical graph-pose targets from the set-format data batch."""
    normalized = dict(batch)
    if "gt_joints_canon" not in normalized:
        normalized["gt_joints_canon"] = normalized["gt_keypoints"]
    if "gt_visibility" not in normalized:
        normalized["gt_visibility"] = normalized["gt_labeled"]
    if "gt_boxes_pixel" in normalized and "img_hw" in normalized:
        normalized["gt_boxes_canon"] = _boxes_pixel_to_canonical(
            normalized["gt_boxes_pixel"].float(), normalized["img_hw"].float()
        )
    return normalized


def _image_condition(
    backbone: BaseFeatureExtractor, images: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    features = backbone(image=images)
    spatial_features = features["spatial"]
    if spatial_features.ndim != 4:
        raise ValueError("graph-pose backbones must return a spatial feature map")
    batch_size, channels, height, width = spatial_features.shape
    image_tokens = spatial_features.permute(0, 2, 3, 1).reshape(
        batch_size, height * width, channels
    )
    patch_centers = features["patch_centers"].reshape(-1, 2)
    if patch_centers.shape[0] != image_tokens.shape[1]:
        raise ValueError(
            f"patch_centers length {patch_centers.shape[0]} != token count "
            f"{image_tokens.shape[1]}"
        )
    return image_tokens, patch_centers, spatial_features


class _PoseGraphLayer(MessagePassing):
    def __init__(self, dim: int, dim_edge: int, dim_vis: int, dim_global: int, num_heads: int, dropout_rate: float, attn_dropout_rate: float) -> None:
        super().__init__(aggr="add")
        if dim % num_heads:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_dropout = attn_dropout_rate
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        message_dim = dim + dim_edge + dim_vis
        self.message_mlp = nn.Sequential(
            nn.LayerNorm(message_dim), nn.Linear(message_dim, dim * 2), nn.GELU(),
            nn.Dropout(dropout_rate), nn.Linear(dim * 2, dim),
        )
        self.edge_bias = nn.Linear(dim_edge, num_heads, bias=False)
        update_dim = dim * 2 + dim_vis
        self.update_mlp = nn.Sequential(
            nn.LayerNorm(update_dim), nn.Linear(update_dim, dim * 2), nn.GELU(),
            nn.Dropout(dropout_rate), nn.Linear(dim * 2, dim),
        )
        self.film = FiLMLayer(dim_global, dim)

    def message(self, q_i: Tensor, k_j: Tensor, v_state_j: Tensor, edge_feat: Tensor, vis_j: Tensor, edge_dst: Tensor) -> Tensor:
        edge_count = q_i.shape[0]
        message = self.message_mlp(torch.cat([v_state_j, edge_feat, vis_j], dim=-1))
        query = q_i.view(edge_count, self.num_heads, self.head_dim)
        key = k_j.view(edge_count, self.num_heads, self.head_dim)
        score = (query * key).sum(-1) / math.sqrt(self.head_dim) + self.edge_bias(edge_feat)
        weights = softmax(score, index=edge_dst, num_nodes=self._num_nodes)
        weights = F.dropout(weights, p=self.attn_dropout, training=self.training)
        return (message.view(edge_count, self.num_heads, self.head_dim) * weights.unsqueeze(-1)).view(edge_count, self.dim)

    def forward(self, features: Tensor, edge_index: Tensor, edge_features: Tensor, visibility: Tensor, condition: Tensor) -> Tensor:
        features = self.film(features, condition)
        _, destination = edge_index
        self._num_nodes = int(features.shape[0])
        aggregate = self.propagate(
            edge_index, q=self.query(features), k=self.key(features),
            v_state=self.value(features), edge_feat=edge_features,
            vis=visibility, edge_dst=destination,
        )
        return features + self.update_mlp(torch.cat([features, aggregate, visibility], dim=-1))


class _GraphPoseRefiner(nn.Module):
    def __init__(self, dim: int, num_layers: int, dim_edge: int, dim_vis: int, num_heads: int, dropout_rate: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            _PoseGraphLayer(dim, dim_edge, dim_vis, 1, num_heads, dropout_rate, dropout_rate)
            for _ in range(num_layers)
        )
        self.output_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2))
        nn.init.zeros_(self.output_proj[1].weight)
        nn.init.zeros_(self.output_proj[1].bias)

    def forward(self, features: Tensor, edge_index: Tensor, edge_features: Tensor, visibility: Tensor, condition: Tensor) -> Tensor:
        for layer in self.layers:
            features = layer(features, edge_index, edge_features, visibility, condition)
        return self.output_proj(features)


class _GraphPoseDenoiser(nn.Module):
    """Joint box-and-pose denoiser with attention and skeleton refinement."""

    def __init__(
        self,
        *,
        num_slots: int,
        num_keypoints: int,
        dim: int,
        num_heads: int,
        num_trunk_blocks: int,
        num_joint_blocks: int,
        time_dim: int,
        ffn_mult: float,
        image_token_dim: int,
        spatial_feat_dim: int,
        dim_vis: int,
        dim_edge: int,
        num_refiner_layers: int,
        fourier_box_bands: int,
        fourier_box_scale: float,
        fourier_joint_bands: int,
        fourier_joint_scale: float,
        fourier_sigma_bands: int,
        fourier_sigma_scale: float,
        sigma_log_offset: float,
        pe_scale: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if dim % 4 or dim % num_heads:
            raise ValueError("dim must be divisible by 4 and num_heads")
        if num_slots < 1:
            raise ValueError("num_slots must be at least one")
        self.num_slots = num_slots
        self.num_keypoints = num_keypoints
        self.dim = dim
        self.dim_vis = dim_vis
        self.sigma_log_offset = float(sigma_log_offset)
        self.pe_scale = float(pe_scale)
        self.box_in = FourierFeatures(4, dim, fourier_box_bands, fourier_box_scale)
        self.joint_in = FourierFeatures(2, dim, fourier_joint_bands, fourier_joint_scale)
        self.slot_query = nn.Embedding(num_slots, dim)
        self.joint_query = nn.Embedding(num_keypoints, dim)
        self.time_in = FourierFeatures(1, time_dim, fourier_sigma_bands, fourier_sigma_scale)
        self.image_proj = nn.Linear(image_token_dim, dim)
        self.trunk_blocks = nn.ModuleList(
            TimeFiLMTransformerBlock(
                dim=dim, num_heads=num_heads, time_dim=time_dim,
                ffn_mult=ffn_mult, cross_dim=dim, use_cross_attn=True,
                dropout=dropout,
            )
            for _ in range(num_trunk_blocks)
        )
        self.obj_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        self.box_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 4))
        self.joint_blocks = nn.ModuleList(
            TimeFiLMTransformerBlock(
                dim=dim, num_heads=num_heads, time_dim=time_dim,
                ffn_mult=ffn_mult, cross_dim=dim, use_cross_attn=True,
                dropout=dropout,
            )
            for _ in range(num_joint_blocks)
        )
        self.joint_self_attn = TimeFiLMTransformerBlock(
            dim=dim, num_heads=num_heads, time_dim=time_dim,
            ffn_mult=ffn_mult, use_cross_attn=False, dropout=dropout,
        )
        self.joint_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2))
        self.vis_proj = nn.Linear(spatial_feat_dim, dim_vis)
        self.refiner_pos_emb = nn.Sequential(
            nn.Linear(2, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim)
        )
        self.refiner_type_emb = nn.Embedding(num_keypoints, dim)
        self.refiner_node_norm = nn.LayerNorm(dim)
        self.edge_encoder = nn.Sequential(
            nn.Linear(2, dim_edge), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim_edge, dim_edge)
        )
        self.refiner = _GraphPoseRefiner(dim, num_refiner_layers, dim_edge, dim_vis, num_heads, dropout)
        self.register_buffer("_edge_index", CANONICAL_EDGE_INDEX.clone())

    def forward(
        self,
        noisy_boxes: Tensor,
        noisy_joints_rel: Tensor,
        sigma: Tensor,
        image_tokens: Tensor,
        patch_centers: Tensor,
        spatial_features: Tensor,
    ) -> dict[str, Tensor]:
        batch_size, num_slots, _ = noisy_boxes.shape
        if num_slots != self.num_slots:
            raise ValueError(f"expected N={self.num_slots} slots, got {num_slots}")
        if noisy_joints_rel.shape[2] != self.num_keypoints:
            raise ValueError(f"expected K={self.num_keypoints} joints")
        dtype = noisy_boxes.dtype
        time_embedding = self.time_in(
            torch.log(sigma.clamp_min(self.sigma_log_offset).to(dtype)).unsqueeze(-1)
        )
        image_context = self.image_proj(image_tokens)
        image_context = image_context + make_pe_from_centers(
            patch_centers.to(dtype), self.dim, scale=self.pe_scale
        ).unsqueeze(0)
        slot_tokens = self.box_in(noisy_boxes) + self.slot_query.weight.unsqueeze(0)
        for block in self.trunk_blocks:
            slot_tokens = block(slot_tokens, context=image_context, time_emb=time_embedding)
        pred_boxes = self.box_head(slot_tokens)
        obj_logits = self.obj_head(slot_tokens).squeeze(-1)
        joint_tokens = self.joint_in(noisy_joints_rel) + self.joint_query.weight
        joint_tokens = joint_tokens.reshape(batch_size * self.num_slots, self.num_keypoints, self.dim)
        time_per_slot = time_embedding.unsqueeze(1).expand(-1, self.num_slots, -1).reshape(
            batch_size * self.num_slots, -1
        )
        instance_context = slot_tokens.reshape(batch_size * self.num_slots, 1, self.dim)
        for block in self.joint_blocks:
            joint_tokens = block(joint_tokens, context=instance_context, time_emb=time_per_slot)
        rough_joints_rel = self.joint_head(
            self.joint_self_attn(joint_tokens, time_emb=time_per_slot)
        ).reshape(batch_size, self.num_slots, self.num_keypoints, 2)
        joints_abs = pred_boxes[:, :, :2].unsqueeze(2) + rough_joints_rel
        graph = create_graph_from_keypoints(self._edge_index, joints_abs)
        node_pos = joints_abs.reshape(-1, 2)
        node_types = graph["kpt_type_idx"].reshape(-1)
        node_features = self.refiner_node_norm(
            self.refiner_pos_emb(node_pos) + self.refiner_type_emb(node_types)
        )
        edge_index = graph["edge_index"]
        source, destination = edge_index
        edge_features = self.edge_encoder(node_pos[source] - node_pos[destination])
        visual_features = self._sample_visual_features(spatial_features, joints_abs, dtype)
        global_features = torch.zeros(node_pos.shape[0], 1, device=node_pos.device, dtype=dtype)
        joint_delta = self.refiner(
            node_features, edge_index, edge_features, visual_features, global_features
        )
        return {
            "pred_boxes": pred_boxes,
            "pred_joints_rel": rough_joints_rel + joint_delta.reshape_as(rough_joints_rel),
            "obj_logits": obj_logits,
        }

    def _sample_visual_features(
        self, spatial_features: Tensor, joints_abs: Tensor, dtype: torch.dtype
    ) -> Tensor:
        batch_size, num_slots, num_keypoints, _ = joints_abs.shape
        grid = joints_abs.to(dtype).reshape(batch_size, num_slots * num_keypoints, 1, 2)
        sampled = F.grid_sample(
            spatial_features.to(dtype), grid, mode="bilinear", align_corners=True, padding_mode="zeros"
        )
        sampled = sampled.squeeze(-1).permute(0, 2, 1)
        return self.vis_proj(sampled).reshape(batch_size * num_slots * num_keypoints, self.dim_vis)


class GraphPosePipeline(pl.LightningModule):
    """End-to-end tensor-native Lightning pipeline for graph-pose diffusion."""

    def __init__(
        self,
        *,
        backbone: BaseFeatureExtractor,
        weights: GraphPipelineLossWeights | None = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        lr_T_max: int | None = None,
        lr_warmup_epochs: int = 0,
        lr_min_frac: float = 1e-2,
        sigma_min: float = 0.002,
        sigma_max: float = 10.0,
        rho: float = 7.0,
        sigma_p_mean: float = -1.2,
        sigma_p_std: float = 1.2,
        num_inference_steps: int = 5,
        val_ap_every_n_epochs: int = 10,
        val_ann_file: str | None = None,
        test_ann_file: str | None = None,
        noise_std: float = 0.0,
        brightness: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
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
        self.backbone = backbone
        self.denoiser = _GraphPoseDenoiser(
            num_slots=num_slots, num_keypoints=num_keypoints, dim=dim, num_heads=num_heads,
            num_trunk_blocks=num_trunk_blocks, num_joint_blocks=num_joint_blocks,
            time_dim=time_dim, ffn_mult=ffn_mult, image_token_dim=image_token_dim,
            spatial_feat_dim=spatial_feat_dim, dim_vis=dim_vis, dim_edge=dim_edge,
            num_refiner_layers=num_refiner_layers, fourier_box_bands=fourier_box_bands,
            fourier_box_scale=fourier_box_scale, fourier_joint_bands=fourier_joint_bands,
            fourier_joint_scale=fourier_joint_scale, fourier_sigma_bands=fourier_sigma_bands,
            fourier_sigma_scale=fourier_sigma_scale, sigma_log_offset=sigma_log_offset,
            pe_scale=pe_scale, dropout=dropout,
        )
        self._loss = GraphPipelineLoss(weights)
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_T_max = lr_T_max
        self.lr_warmup_epochs = lr_warmup_epochs
        self.lr_min_frac = lr_min_frac
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.rho = float(rho)
        self.sigma_p_mean = float(sigma_p_mean)
        self.sigma_p_std = float(sigma_p_std)
        self.num_inference_steps = int(num_inference_steps)
        self.val_ap_every_n_epochs = int(val_ap_every_n_epochs)
        self.noise_std = float(noise_std)
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.saturation = float(saturation)
        self._val_ap = KeypointAPMetric(ann_file=val_ann_file) if val_ann_file else None
        self._val_ap1 = KeypointAPMetric(ann_file=val_ann_file) if val_ann_file else None
        self._test_ap = KeypointAPMetric(ann_file=test_ann_file) if test_ann_file else None
        self._test_ap1 = KeypointAPMetric(ann_file=test_ann_file) if test_ann_file else None
        self._log_loss_names = (
            "box_l1", "box_giou", "box_l2", "kpt_l2", "obj_loss", "ranking_loss", "matched"
        )
        self.save_hyperparameters(ignore=["backbone", "weights"])

    @property
    def num_slots(self) -> int:
        return self.denoiser.num_slots

    @property
    def num_keypoints(self) -> int:
        return self.denoiser.num_keypoints

    def forward(
        self, images: Tensor, noisy_boxes: Tensor, noisy_joints_rel: Tensor, sigma: Tensor
    ) -> dict[str, Tensor]:
        """Predict clean canonical boxes, relative joints, and objectness logits."""
        image_tokens, patch_centers, spatial_features = _image_condition(self.backbone, images)
        return self.denoiser(
            noisy_boxes, noisy_joints_rel, sigma, image_tokens, patch_centers, spatial_features
        )

    @staticmethod
    def edm_sigma_schedule(
        num_steps: int, sigma_min: float = 0.002, sigma_max: float = 10.0,
        rho: float = 7.0, device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1; got {num_steps}")
        step_indices = torch.arange(num_steps, device=device, dtype=dtype)
        inverse_rho = 1.0 / rho
        sigmas = (
            sigma_max ** inverse_rho + step_indices / max(num_steps - 1, 1)
            * (sigma_min ** inverse_rho - sigma_max ** inverse_rho)
        ) ** rho
        return torch.cat([sigmas, sigmas.new_zeros(1)])

    @torch.no_grad()
    def sample(
        self, images: Tensor, *, num_inference_steps: int | None = None,
        generator: Optional[torch.Generator] = None, x_boxes_init: Optional[Tensor] = None,
        x_joints_init: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        """Run EDM Heun sampling over canonical boxes and relative joints."""
        steps = self.num_inference_steps if num_inference_steps is None else num_inference_steps
        if steps < 1:
            raise ValueError(f"num_inference_steps must be >= 1; got {steps}")
        image_tokens, patch_centers, spatial_features = _image_condition(self.backbone, images)
        batch_size = images.shape[0]
        dtype = image_tokens.dtype
        def initial_state(value: Optional[Tensor], shape: tuple[int, ...]) -> Tensor:
            if value is not None:
                return value.to(images.device, dtype)
            return torch.empty(shape, device=images.device, dtype=dtype).normal_(generator=generator) * self.sigma_max
        boxes = initial_state(x_boxes_init, (batch_size, self.num_slots, 4))
        joints = initial_state(x_joints_init, (batch_size, self.num_slots, self.num_keypoints, 2))
        def denoise(current_boxes: Tensor, current_joints: Tensor, current_sigma: Tensor) -> dict[str, Tensor]:
            return self.denoiser(current_boxes, current_joints, current_sigma, image_tokens, patch_centers, spatial_features)
        if steps == 1:
            return denoise(boxes, joints, torch.full((batch_size,), self.sigma_max, device=images.device, dtype=dtype))
        sigmas = self.edm_sigma_schedule(steps, self.sigma_min, self.sigma_max, self.rho, images.device, dtype)
        if x_boxes_init is None:
            boxes = boxes * (sigmas[0] / max(self.sigma_max, 1e-12))
        if x_joints_init is None:
            joints = joints * (sigmas[0] / max(self.sigma_max, 1e-12))
        final: dict[str, Tensor] = {}
        for index in range(steps):
            sigma = sigmas[index]
            sigma_next = sigmas[index + 1]
            prediction = denoise(boxes, joints, sigma.expand(batch_size))
            final = prediction
            box_derivative = (boxes - prediction["pred_boxes"]) / sigma.clamp_min(self.denoiser.sigma_log_offset)
            joint_derivative = (joints - prediction["pred_joints_rel"]) / sigma.clamp_min(self.denoiser.sigma_log_offset)
            delta = sigma_next - sigma
            euler_boxes = boxes + delta * box_derivative
            euler_joints = joints + delta * joint_derivative
            if sigma_next > 0:
                correction = denoise(euler_boxes, euler_joints, sigma_next.expand(batch_size))
                final = correction
                boxes = boxes + delta * 0.5 * (box_derivative + (euler_boxes - correction["pred_boxes"]) / sigma_next)
                joints = joints + delta * 0.5 * (joint_derivative + (euler_joints - correction["pred_joints_rel"]) / sigma_next)
            else:
                boxes, joints = euler_boxes, euler_joints
        return final

    def _sample_sigma(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        return (self.sigma_p_mean + self.sigma_p_std * torch.randn(batch_size, device=device, dtype=dtype)).exp().clamp(self.sigma_min, self.sigma_max)

    def _build_noisy_input(self, batch: dict[str, Tensor], sigma: Tensor) -> tuple[Tensor, Tensor]:
        gt_boxes = batch["gt_boxes_canon"]
        gt_joints = batch["gt_joints_canon"]
        labels = batch["gt_box_labeled"]
        batch_size, max_people, _ = gt_boxes.shape
        boxes = torch.stack([
            torch.rand(batch_size, self.num_slots, device=gt_boxes.device, dtype=gt_boxes.dtype) * 2 - 1,
            torch.rand(batch_size, self.num_slots, device=gt_boxes.device, dtype=gt_boxes.dtype) * 2 - 1,
            torch.rand(batch_size, self.num_slots, device=gt_boxes.device, dtype=gt_boxes.dtype) * 0.45 + 0.05,
            torch.rand(batch_size, self.num_slots, device=gt_boxes.device, dtype=gt_boxes.dtype) * 0.45 + 0.05,
        ], dim=-1)
        joints = torch.zeros(batch_size, self.num_slots, self.num_keypoints, 2, device=gt_boxes.device, dtype=gt_boxes.dtype)
        relative_joints = gt_joints - gt_boxes[:, :, :2].unsqueeze(2)
        for index in range(batch_size):
            count = min(int((labels[index] > 0).sum().item()), self.num_slots, max_people)
            boxes[index, :count] = gt_boxes[index, :count]
            joints[index, :count] = relative_joints[index, :count]
            permutation = torch.randperm(self.num_slots, device=gt_boxes.device)
            boxes[index], joints[index] = boxes[index, permutation], joints[index, permutation]
        return (
            boxes + sigma.view(batch_size, 1, 1).to(boxes.dtype) * torch.randn_like(boxes),
            joints + sigma.view(batch_size, 1, 1, 1).to(joints.dtype) * torch.randn_like(joints),
        )

    def _losses_for_batch(self, batch: dict[str, Tensor], images: Tensor) -> tuple[dict[str, Tensor], Tensor]:
        sigma = self._sample_sigma(images.shape[0], images.device, images.dtype)
        noisy_boxes, noisy_joints = self._build_noisy_input(batch, sigma)
        predictions = self(images, noisy_boxes, noisy_joints, sigma)
        return self._loss(predictions, {
            "gt_boxes_canon": batch["gt_boxes_canon"], "gt_joints_canon": batch["gt_joints_canon"],
            "gt_box_labeled": batch["gt_box_labeled"], "gt_visibility": batch["gt_visibility"],
        }), sigma

    def _log_losses(self, prefix: str, losses: dict[str, Tensor], batch_size: int, on_step: bool) -> None:
        self.log(f"{prefix}/loss", losses["total"], prog_bar=True, batch_size=batch_size, on_step=on_step, on_epoch=True)
        for name in self._log_loss_names:
            self.log(f"{prefix}/{name}", losses[name], batch_size=batch_size, on_step=on_step, on_epoch=True)

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        del batch_idx
        batch = _normalize_batch(batch)
        images = batch["image"]
        if self.noise_std:
            images = add_gaussian_noise(images, self.noise_std)
        if self.brightness or self.contrast or self.saturation:
            images = color_jitter(images, self.brightness, self.contrast, self.saturation)
        losses, sigma = self._losses_for_batch(batch, images)
        self._log_losses("train", losses, images.shape[0], True)
        self.log("debug/sigma_mean", sigma.mean(), batch_size=images.shape[0], on_step=True, on_epoch=False)
        return losses["total"]

    def _ap_epoch_active(self) -> bool:
        return ((self.current_epoch + 1) % self.val_ap_every_n_epochs == 0 or (
            self._trainer is not None and self.current_epoch + 1 == self.trainer.max_epochs
        ))

    def _update_ap(self, metric: KeypointAPMetric, predictions: dict[str, Tensor], batch: dict[str, Tensor]) -> None:
        boxes = predictions["pred_boxes"].detach().float()
        keypoints = boxes[:, :, :2].unsqueeze(2) + predictions["pred_joints_rel"].detach().float()
        image_wh = batch["img_hw"].to(boxes.device).float()[:, [1, 0]]
        scores = torch.sigmoid(predictions["obj_logits"]).detach().float()
        metric.update(canonical_to_pixel(keypoints, image_wh), scores.unsqueeze(-1).expand_as(keypoints[..., 0]), batch["image_id"].to(boxes.device))

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        del batch_idx
        batch = _normalize_batch(batch)
        losses, _ = self._losses_for_batch(batch, batch["image"])
        self._log_losses("val", losses, batch["image"].shape[0], False)
        if self._ap_epoch_active() and self._val_ap is not None:
            self._update_ap(self._val_ap, self.sample(batch["image"]), batch)
        if self._ap_epoch_active() and self._val_ap1 is not None:
            self._update_ap(self._val_ap1, self.sample(batch["image"], num_inference_steps=1), batch)

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        del batch_idx
        batch = _normalize_batch(batch)
        losses, _ = self._losses_for_batch(batch, batch["image"])
        self._log_losses("test", losses, batch["image"].shape[0], False)
        if self._test_ap is not None:
            self._update_ap(self._test_ap, self.sample(batch["image"]), batch)
        if self._test_ap1 is not None:
            self._update_ap(self._test_ap1, self.sample(batch["image"], num_inference_steps=1), batch)

    def _log_and_reset_metric(self, metric: KeypointAPMetric | None, prefix: str) -> None:
        if metric is not None:
            for name, value in metric.compute().items():
                self.log(f"{prefix}/{name}", value, sync_dist=True)
            metric.reset()

    def on_validation_epoch_end(self) -> None:
        if self._ap_epoch_active():
            self._log_and_reset_metric(self._val_ap, f"ap-val/{self.num_inference_steps}-step")
            self._log_and_reset_metric(self._val_ap1, "ap-val/1-step")

    def on_test_epoch_end(self) -> None:
        self._log_and_reset_metric(self._test_ap, f"ap-test/{self.num_inference_steps}-step")
        self._log_and_reset_metric(self._test_ap1, "ap-test/1-step")

    def configure_optimizers(self):
        return build_adamw_with_schedule(self, lr=self.lr, weight_decay=self.weight_decay, lr_T_max=self.lr_T_max, lr_warmup_epochs=self.lr_warmup_epochs, lr_min_frac=self.lr_min_frac)


class GraphPoseLossWeightsConfig(BaseConfig):
    box_l1: float = 5.0
    box_giou: float = 2.0
    box_l2: float = 1.0
    kpt_l2: float = 5.0
    obj_bce: float = 1.0
    rank: float = 1.0
    matcher_lambda_l1: float = 5.0
    matcher_lambda_giou: float = 2.0
    matcher_lambda_kpt: float = 1.0
    matcher_lambda_obj: float = 0.1

    def to_dataclass(self) -> GraphPipelineLossWeights:
        return GraphPipelineLossWeights(**self.model_dump())


class GraphPosePipelineConfig(BaseConfig):
    backbone: BaseFeatureExtractorConfig
    lr: float = 2e-4
    weight_decay: float = 1e-4
    cosine_lr: bool = True
    lr_warmup_epochs: int = 1
    lr_min_frac: float = 1e-2
    sigma_min: float = 0.002
    sigma_max: float = 10.0
    rho: float = 7.0
    sigma_p_mean: float = -1.2
    sigma_p_std: float = 1.2
    num_inference_steps: int = 5
    val_ap_every_n_epochs: int = 10
    noise_std: float = 0.02
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.1
    num_slots: int = 20
    num_keypoints: int = 14
    dim: int = 256
    num_heads: int = 8
    num_trunk_blocks: int = 3
    num_joint_blocks: int = 2
    time_dim: int = 256
    ffn_mult: float = 4.0
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
    weights: GraphPoseLossWeightsConfig = Field(default_factory=GraphPoseLossWeightsConfig)

    def build(
        self,
        *,
        val_ann_file: str | None = None,
        test_ann_file: str | None = None,
        max_epochs: int,
    ) -> GraphPosePipeline:
        backbone = self.backbone.build()
        try:
            backbone_channels = backbone.out_channels
        except AttributeError as exc:
            raise ValueError("graph-pose backbones must expose `out_channels`") from exc
        return GraphPosePipeline(
            backbone=backbone, weights=self.weights.to_dataclass(), lr=self.lr,
            weight_decay=self.weight_decay, lr_T_max=max_epochs if self.cosine_lr else None,
            lr_warmup_epochs=self.lr_warmup_epochs, lr_min_frac=self.lr_min_frac,
            sigma_min=self.sigma_min, sigma_max=self.sigma_max, rho=self.rho,
            sigma_p_mean=self.sigma_p_mean, sigma_p_std=self.sigma_p_std,
            num_inference_steps=self.num_inference_steps, val_ap_every_n_epochs=self.val_ap_every_n_epochs,
            val_ann_file=val_ann_file, test_ann_file=test_ann_file, noise_std=self.noise_std,
            brightness=self.brightness, contrast=self.contrast, saturation=self.saturation,
            num_slots=self.num_slots, num_keypoints=self.num_keypoints, dim=self.dim,
            num_heads=self.num_heads, num_trunk_blocks=self.num_trunk_blocks,
            num_joint_blocks=self.num_joint_blocks, time_dim=self.time_dim, ffn_mult=self.ffn_mult,
            image_token_dim=backbone_channels, spatial_feat_dim=backbone_channels, dim_vis=self.dim_vis,
            dim_edge=self.dim_edge, num_refiner_layers=self.num_refiner_layers,
            fourier_box_bands=self.fourier_box_bands, fourier_box_scale=self.fourier_box_scale,
            fourier_joint_bands=self.fourier_joint_bands, fourier_joint_scale=self.fourier_joint_scale,
            fourier_sigma_bands=self.fourier_sigma_bands, fourier_sigma_scale=self.fourier_sigma_scale,
            sigma_log_offset=self.sigma_log_offset, pe_scale=self.pe_scale, dropout=self.dropout,
        )

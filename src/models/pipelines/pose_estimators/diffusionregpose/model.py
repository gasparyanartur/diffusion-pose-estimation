"""Checkpoint-shaped DiffusionRegPose model assembly."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .backbone import DiffusionRegPoseResNetBackbone, PositionEmbeddingSineHW
from .diffusion import DiffusionRegPoseDiffusion
from .transformer import (
    DeformableTransformerDecoder,
    DeformableTransformerDecoderLayer,
    DeformableTransformerEncoder,
    DeformableTransformerEncoderLayer,
    DiffusionRegPosePredictionHeads,
    DiffusionRegPoseTimeConditioning,
    MLP,
    build_input_projections,
    inverse_sigmoid,
)


class _BackboneJoiner(nn.Sequential):
    def __init__(self, backbone: nn.Module, position_embedding: nn.Module) -> None:
        super().__init__(backbone, position_embedding)

    def forward(self, image: Tensor, padding_mask: Tensor) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        features = self[0](image)
        masks = [
            torch.nn.functional.interpolate(
                padding_mask[:, None].float(), size=feature.shape[-2:]
            ).to(torch.bool)[:, 0]
            for feature in features
        ]
        positions = [self[1](feature, mask).to(feature.dtype) for feature, mask in zip(features, masks)]
        return features, masks, positions


@dataclass
class TransformerInputs:
    memory: Tensor
    position: Tensor
    padding_mask: Tensor
    spatial_shapes: Tensor
    level_start_index: Tensor
    valid_ratios: Tensor


def gen_encoder_output_proposals(
    memory: Tensor,
    padding_mask: Tensor,
    spatial_shapes: Tensor,
    learned_wh: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    proposals = []
    current = 0
    for level, (height, width) in enumerate(spatial_shapes.tolist()):
        level_mask = padding_mask[:, current:current + height * width].view(
            memory.shape[0], height, width, 1
        )
        valid_height = (~level_mask[:, :, 0, 0]).sum(1)
        valid_width = (~level_mask[:, 0, :, 0]).sum(1)
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, height - 1, height, device=memory.device),
            torch.linspace(0, width - 1, width, device=memory.device),
            indexing="ij",
        )
        grid = torch.stack((grid_x, grid_y), -1)
        scale = torch.stack((valid_width, valid_height), 1)[:, None, None]
        grid = (grid[None] + 0.5) / scale
        if learned_wh is None:
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** level)
        else:
            wh = torch.ones_like(grid) * learned_wh.sigmoid() * (2.0 ** level)
        proposals.append(torch.cat((grid, wh), -1).view(memory.shape[0], -1, 4))
        current += height * width
    output_proposals = torch.cat(proposals, 1)
    valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
    output_proposals = torch.log(output_proposals / (1 - output_proposals))
    output_proposals = output_proposals.masked_fill(padding_mask.unsqueeze(-1), float("inf"))
    output_proposals = output_proposals.masked_fill(~valid, float("inf"))
    output_memory = memory.masked_fill(padding_mask.unsqueeze(-1), 0.0)
    output_memory = output_memory.masked_fill(~valid, 0.0)
    return output_memory, output_proposals


class DiffusionRegPoseTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_encoder_layers: int,
        num_decoder_layers: int,
        hidden_dim: int,
        num_heads: int,
        num_feature_levels: int,
        num_body_points: int,
        num_box_decoder_layers: int,
        num_group: int,
        num_classes: int,
        bbox_embed: nn.ModuleList,
        class_embed: nn.ModuleList,
        pose_embed: nn.ModuleList,
        pose_hw_embed: nn.ModuleList,
    ) -> None:
        super().__init__()
        encoder_layer = DeformableTransformerEncoderLayer(
            d_model=hidden_dim,
            d_ffn=2048,
            dropout=0.0,
            num_levels=num_feature_levels,
            num_heads=num_heads,
            num_points=4,
        )
        decoder_layer = DeformableTransformerDecoderLayer(
            d_model=hidden_dim,
            d_ffn=2048,
            dropout=0.0,
            num_levels=num_feature_levels,
            num_heads=num_heads,
            num_points=4,
        )
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)
        self.decoder = DeformableTransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            d_model=hidden_dim,
            num_feature_levels=num_feature_levels,
            num_body_points=num_body_points,
            num_box_decoder_layers=num_box_decoder_layers,
            num_group=num_group,
        )
        self.level_embed = nn.Parameter(torch.empty(num_feature_levels, hidden_dim))
        nn.init.normal_(self.level_embed)
        self.enc_output = nn.Linear(hidden_dim, hidden_dim)
        self.enc_output_norm = nn.LayerNorm(hidden_dim)
        self.enc_out_bbox_embed = copy.deepcopy(bbox_embed[0])
        self.enc_out_class_embed = copy.deepcopy(class_embed[0])
        self.time_mlp = DiffusionRegPoseTimeConditioning(hidden_dim).time_mlp
        self.block_time_mlp = DiffusionRegPoseTimeConditioning(hidden_dim).block_time_mlp
        self.decoder.bbox_embed = bbox_embed
        self.decoder.class_embed = class_embed
        self.decoder.pose_embed = pose_embed
        self.decoder.pose_hw_embed = pose_hw_embed


class DiffusionRegPoseModel(DiffusionRegPoseDiffusion):
    """Assemble the checkpoint-compatible backbone and transformer components."""

    def __init__(
        self,
        *,
        backbone_pretrained: bool = True,
        timesteps: int = 1000,
        sampling_timesteps: int = 1,
        num_queries: int = 900,
        num_proposals: int = 100,
        num_classes: int = 2,
        num_feature_levels: int = 4,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_body_points: int = 14,
        num_box_decoder_layers: int = 2,
        num_group: int = 100,
    ) -> None:
        super().__init__(
            timesteps=timesteps,
            sampling_timesteps=sampling_timesteps,
            scale=6.0,
            ddim_sampling_eta=0.0,
        )
        self.num_queries = num_queries
        self.num_proposals = num_proposals
        self.num_feature_levels = num_feature_levels
        self.label_enc = nn.Embedding(101, hidden_dim)
        self.backbone = _BackboneJoiner(
            DiffusionRegPoseResNetBackbone(pretrained=backbone_pretrained),
            PositionEmbeddingSineHW(hidden_dim // 2, temperature_h=20, temperature_w=20),
        )
        self.input_proj = build_input_projections((512, 1024, 2048), hidden_dim=hidden_dim, num_levels=num_feature_levels)
        heads = DiffusionRegPosePredictionHeads(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_decoder_layers=num_decoder_layers,
            num_box_decoder_layers=num_box_decoder_layers,
            num_body_points=num_body_points,
        )
        self.bbox_embed = heads.bbox_embed
        self.class_embed = heads.class_embed
        self.pose_embed = heads.pose_embed
        self.pose_hw_embed = heads.pose_hw_embed
        self.transformer = DiffusionRegPoseTransformer(
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_feature_levels=num_feature_levels,
            num_body_points=num_body_points,
            num_box_decoder_layers=num_box_decoder_layers,
            num_group=num_group,
            num_classes=num_classes,
            bbox_embed=self.bbox_embed,
            class_embed=self.class_embed,
            pose_embed=self.pose_embed,
            pose_hw_embed=self.pose_hw_embed,
        )

    def prepare_transformer_inputs(self, images: Tensor, padding_mask: Tensor) -> TransformerInputs:
        features, masks, positions = self.backbone(images, padding_mask)
        projected = [projection(feature) for projection, feature in zip(self.input_proj, features)]
        while len(projected) < self.num_feature_levels:
            source = features[-1] if len(projected) == len(features) else projected[-1]
            projected.append(self.input_proj[len(projected)](source))
            masks.append(torch.nn.functional.interpolate(padding_mask[:, None].float(), size=projected[-1].shape[-2:]).to(torch.bool)[:, 0])
            positions.append(self.backbone[1](projected[-1], masks[-1]).to(projected[-1].dtype))
        flattened, flattened_pos, flattened_masks, shapes = [], [], [], []
        for feature, position, mask in zip(projected, positions, masks):
            _, _, height, width = feature.shape
            shapes.append((height, width))
            flattened.append(feature.flatten(2).transpose(1, 2))
            flattened_pos.append(position.flatten(2).transpose(1, 2))
            flattened_masks.append(mask.flatten(1))
        spatial_shapes = torch.as_tensor(shapes, dtype=torch.long, device=images.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(1), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([
            torch.stack((((~mask[:, 0]).sum(1).float() / mask.shape[2]), ((~mask[:, :, 0]).sum(1).float() / mask.shape[1])), -1)
            for mask in masks
        ], 1)
        return TransformerInputs(
            memory=torch.cat(flattened, 1),
            position=torch.cat(flattened_pos, 1),
            padding_mask=torch.cat(flattened_masks, 1),
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
        )

    def forward(
        self, images: Tensor, diffusion_keypoints: Tensor, timesteps: Tensor
    ) -> dict[str, Tensor | list[dict[str, Tensor]]]:
        inputs = self.prepare_transformer_inputs(
            images, torch.zeros(images.shape[0], *images.shape[2:], dtype=torch.bool, device=images.device)
        )
        memory = self.encode_features(inputs, timesteps)
        query_features, reference_boxes, _, _ = self.select_two_stage_queries(memory, inputs)
        decoder_features, decoder_references = self.transformer.decoder(
            query_features.transpose(0, 1),
            memory,
            refpoints_unsigmoid=reference_boxes.transpose(0, 1),
            level_start_index=inputs.level_start_index,
            spatial_shapes=inputs.spatial_shapes,
            valid_ratios=inputs.valid_ratios,
            memory_key_padding_mask=inputs.padding_mask,
            pos=inputs.position,
            diffusion_keypoints=((diffusion_keypoints / self.scale) + 1.0) / 2.0,
        )
        decoder_outputs = []
        for features, references, box_head, class_head in zip(
            decoder_features,
            decoder_references[1:],
            self.bbox_embed,
            self.class_embed,
        ):
            if features.shape[1] > self.num_queries:
                stride = self.transformer.decoder.num_body_points + 1
                box_indices = torch.arange(0, features.shape[1], stride, device=features.device)
                point_indices = torch.arange(features.shape[1], device=features.device)
                point_indices = point_indices[point_indices.remainder(stride) != 0]
                box_features = features.index_select(1, box_indices)
                box_references = references.index_select(1, box_indices)
                point_features = features.index_select(1, point_indices)
                point_references = references.index_select(1, point_indices)
                boxes = (box_head(box_features) + inverse_sigmoid(box_references)).sigmoid()
                pose_head = self.pose_embed[len(decoder_outputs) - self.transformer.decoder.num_box_decoder_layers]
                keypoint_xy = (pose_head(point_features) + inverse_sigmoid(point_references[..., :2])).sigmoid()
                visibility = torch.ones_like(keypoint_xy[..., :1]).sigmoid()
                point_count = self.transformer.decoder.num_body_points
                keypoint_xy = keypoint_xy.view(features.shape[0], -1, point_count, 2)
                visibility = visibility.view(features.shape[0], -1, point_count, 1)
                keypoints = torch.cat(
                    (keypoint_xy.flatten(2), visibility.flatten(2)), dim=-1
                )
                logits = class_head(box_features)
            else:
                boxes = (box_head(features) + inverse_sigmoid(references)).sigmoid()
                logits = class_head(features)
                keypoints = torch.zeros(
                    features.shape[0], features.shape[1], self.transformer.decoder.num_body_points * 3,
                    device=features.device, dtype=features.dtype
                )
            decoder_outputs.append({"pred_logits": logits, "pred_boxes": boxes, "pred_keypoints": keypoints})
        return {**decoder_outputs[-1], "aux_outputs": decoder_outputs}

    @torch.no_grad()
    def sample_keypoints(
        self,
        images: Tensor,
        *,
        num_proposals: int | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, Tensor | list[dict[str, Tensor]]]:
        """Sample diffusion keypoints and return the final model predictions."""
        proposal_count = self.num_proposals if num_proposals is None else num_proposals
        if proposal_count < 1:
            raise ValueError(f"num_proposals must be positive; got {proposal_count}")
        if proposal_count != self.transformer.decoder.num_group:
            raise ValueError(
                "num_proposals must equal the configured decoder group count for query expansion"
            )

        state = torch.randn(
            images.shape[0],
            proposal_count,
            self.transformer.decoder.num_body_points * 2,
            device=images.device,
            generator=generator,
        )
        final_output: dict[str, Tensor | list[dict[str, Tensor]]] | None = None
        for time, time_next in self.ddim_timestep_pairs():
            timestep = torch.full(
                (images.shape[0],), time, dtype=torch.long, device=images.device
            )
            final_output = self(images, state, timestep)
            predicted_keypoints = final_output["pred_keypoints"]
            if not isinstance(predicted_keypoints, Tensor):
                raise RuntimeError("model forward returned an invalid keypoint prediction")
            keypoint_coordinate_count = self.transformer.decoder.num_body_points * 2
            x_start = (predicted_keypoints[..., :keypoint_coordinate_count] * 2.0 - 1.0) * self.scale
            x_start = x_start.clamp(min=-self.scale, max=self.scale)
            predicted_noise = self.predict_noise_from_start(state, timestep, x_start).to(state.dtype)
            step_noise = torch.zeros_like(state) if self.ddim_sampling_eta == 0.0 else None
            state = self.ddim_step(x_start, predicted_noise, time, time_next, step_noise)
        if final_output is None:
            raise RuntimeError("diffusion schedule produced no timestep pairs")
        final_output["diffusion_keypoints"] = state
        return final_output

    def encode_features(self, inputs: TransformerInputs, timesteps: Tensor | None = None) -> Tensor:
        position = inputs.position
        if timesteps is not None:
            time_embedding = self.transformer.time_mlp(timesteps)
            scale_shift = self.transformer.block_time_mlp(time_embedding).unsqueeze(1)
            position = position * (scale_shift + 1.0) + scale_shift
        return self.transformer.encoder(
            inputs.memory,
            position + self._level_position_offsets(inputs),
            inputs.spatial_shapes,
            inputs.level_start_index,
            inputs.valid_ratios,
            inputs.padding_mask,
        )

    def select_two_stage_queries(
        self, memory: Tensor, inputs: TransformerInputs, num_queries: int | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        query_count = self.num_queries if num_queries is None else num_queries
        output_memory, output_proposals = gen_encoder_output_proposals(
            memory, inputs.padding_mask, inputs.spatial_shapes
        )
        output_memory = self.transformer.enc_output_norm(self.transformer.enc_output(output_memory))
        class_logits = self.transformer.enc_out_class_embed(output_memory)
        box_logits = self.transformer.enc_out_bbox_embed(output_memory) + output_proposals
        topk = torch.topk(class_logits.max(-1).values, query_count, dim=1).indices
        gather_boxes = topk.unsqueeze(-1).expand(-1, -1, 4)
        gather_features = topk.unsqueeze(-1).expand(-1, -1, output_memory.shape[-1])
        return (
            torch.gather(output_memory, 1, gather_features).detach(),
            torch.gather(box_logits, 1, gather_boxes).detach(),
            torch.gather(class_logits, 1, gather_boxes[..., :class_logits.shape[-1]]),
            torch.gather(output_proposals, 1, gather_boxes).sigmoid(),
        )

    def _level_position_offsets(self, inputs: TransformerInputs) -> Tensor:
        offsets = []
        for level, shape in enumerate(inputs.spatial_shapes.tolist()):
            offsets.append(self.transformer.level_embed[level].view(1, 1, -1).expand(inputs.memory.shape[0], shape[0] * shape[1], -1))
        return torch.cat(offsets, 1)

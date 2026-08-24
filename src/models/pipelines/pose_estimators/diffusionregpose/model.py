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
        dn_number: int = 100,
        dn_box_noise_scale: float = 0.4,
        dn_label_noise_ratio: float = 0.5,
        freeze_backbone_stem_and_layer1: bool = True,
    ) -> None:
        if num_classes <= 1:
            raise ValueError(
                "num_classes must include CrowdPose person label 1, "
                f"got {num_classes}"
            )
        super().__init__(
            timesteps=timesteps,
            sampling_timesteps=sampling_timesteps,
            scale=6.0,
            ddim_sampling_eta=0.0,
        )
        self.num_queries = num_queries
        self.num_proposals = num_proposals
        self.num_classes = num_classes
        if dn_number < 0:
            raise ValueError("dn_number must be non-negative")
        self.num_feature_levels = num_feature_levels
        self.label_enc = nn.Embedding(101, hidden_dim)
        self.backbone = _BackboneJoiner(
            DiffusionRegPoseResNetBackbone(
                pretrained=backbone_pretrained,
                freeze_stem_and_layer1=freeze_backbone_stem_and_layer1,
            ),
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
        self.transformer.decoder.num_queries = num_queries
        self.transformer.decoder.num_dn = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio

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

    def inference_attention_mask(self, batch_size: int, device: torch.device) -> Tensor:
        """Build the evaluation mask for each expanded proposal group.

        Each group contains one box query followed by its body-point queries.
        A box query may attend to every box query, while body-point queries
        may only attend within their own group.
        """
        group_size = self.transformer.decoder.num_body_points + 1
        query_count = self.transformer.decoder.num_group * group_size
        mask = torch.zeros(
            batch_size,
            self.transformer.decoder.num_heads,
            query_count,
            query_count,
            dtype=torch.bool,
            device=device,
        )
        for group_index in range(self.transformer.decoder.num_group):
            start = group_index * group_size
            end = start + group_size
            mask[:, :, start:end, :start] = True
            mask[:, :, start:end, end:] = True
            box_indices = torch.arange(query_count, device=device)
            box_indices = box_indices[box_indices.remainder(group_size) == 0]
            mask[:, :, start, box_indices] = False
        return mask.flatten(0, 1)

    def training_attention_mask(
        self,
        targets: list[dict[str, Tensor]],
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        """Build the external group-isolated mask including DN queries."""
        group_size = self.transformer.decoder.num_body_points + 1
        normal_count = self.transformer.decoder.num_group * group_size
        dn_count = self.transformer.decoder.num_dn
        total_count = dn_count + normal_count
        mask = torch.zeros(
            batch_size,
            self.transformer.decoder.num_heads,
            total_count,
            total_count,
            dtype=torch.bool,
            device=device,
        )
        mask[:, :, dn_count:, :dn_count] = True
        normal_box_indices = dn_count + torch.arange(
            0, normal_count, group_size, device=device
        )
        for group_index in range(self.transformer.decoder.num_group):
            start = dn_count + group_index * group_size
            end = start + group_size
            mask[:, :, start:end, dn_count:start] = True
            mask[:, :, start:end, end:] = True
            mask[:, :, start, normal_box_indices] = False
        for batch_index, target in enumerate(targets):
            target_count = target["labels"].numel()
            if target_count == 0:
                continue
            for query_index in range(dn_count):
                start = (query_index // target_count) * target_count
                end = min(start + target_count, dn_count)
                mask[batch_index, :, query_index, :start] = True
                mask[batch_index, :, query_index, end:dn_count] = True
        return mask.flatten(0, 1)

    def training_box_attention_mask(
        self,
        targets: list[dict[str, Tensor]],
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        """Build the external mask used before keypoint-query expansion."""
        dn_count = self.transformer.decoder.num_dn
        normal_count = self.num_queries
        total_count = dn_count + normal_count
        mask = torch.zeros(
            batch_size,
            self.transformer.decoder.num_heads,
            total_count,
            total_count,
            dtype=torch.bool,
            device=device,
        )
        mask[:, :, dn_count:, :dn_count] = True
        for batch_index, target in enumerate(targets):
            target_count = target["labels"].numel()
            if target_count == 0:
                continue
            for query_index in range(dn_count):
                start = (query_index // target_count) * target_count
                end = min(start + target_count, dn_count)
                mask[batch_index, :, query_index, :start] = True
                mask[batch_index, :, query_index, end:dn_count] = True
        return mask.flatten(0, 1)

    def _prepare_denoising_queries(
        self,
        targets: list[dict[str, Tensor]],
        device: torch.device,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        """Create 100 repeated, noised GT queries used by training."""
        dn_count = self.transformer.decoder.num_dn
        labels, boxes = [], []
        for target in targets:
            target_labels, target_boxes = target["labels"], target["boxes"]
            if target_labels.numel():
                repeat_count = dn_count // target_labels.numel()
                indices = torch.cat(
                    (
                        torch.arange(target_labels.numel(), device=device).repeat(repeat_count),
                        torch.randint(
                            target_labels.numel(), (dn_count % target_labels.numel(),), device=device
                        ),
                    )
                )
                labels.append(target_labels.index_select(0, indices))
                boxes.append(target_boxes.index_select(0, indices))
            else:
                labels.append(torch.full((dn_count,), self.num_classes, dtype=torch.long, device=device))
                boxes.append(torch.rand(dn_count, 4, device=device))
        known_labels = torch.stack(labels)
        known_boxes = torch.stack(boxes)
        noisy_labels = known_labels.clone()
        label_noise = torch.rand_like(noisy_labels.float()) < self.dn_label_noise_ratio
        noisy_labels[label_noise] = torch.randint(
            0, 100, (int(label_noise.sum().item()),), device=device
        )
        noisy_boxes = known_boxes.clone()
        box_difference = torch.zeros_like(noisy_boxes)
        box_difference[..., :2] = noisy_boxes[..., 2:] / 2
        box_difference[..., 2:] = noisy_boxes[..., 2:]
        noisy_boxes = (
            noisy_boxes
            + (torch.rand_like(noisy_boxes) * 2 - 1) * box_difference * self.dn_box_noise_scale
        ).clamp(0, 1)
        return (
            self.label_enc(noisy_labels),
            inverse_sigmoid(noisy_boxes),
            {"labels": known_labels, "boxes": known_boxes},
        )

    def forward(
        self,
        images: Tensor,
        diffusion_keypoints: Tensor,
        timesteps: Tensor,
        targets: list[dict[str, Tensor]] | None = None,
        padding_mask: Tensor | None = None,
    ) -> dict[str, Tensor | list[dict[str, Tensor]]]:
        if padding_mask is None:
            padding_mask = torch.zeros(
                images.shape[0], *images.shape[2:], dtype=torch.bool, device=images.device
            )
        inputs = self.prepare_transformer_inputs(
            images, padding_mask,
        )
        memory = self.encode_features(inputs, timesteps)
        (
            query_features,
            reference_boxes,
            intermediate_logits,
            intermediate_boxes,
        ) = self.select_two_stage_queries(memory, inputs)
        denoising = self.training and targets is not None
        dn_state: dict[str, Tensor] | None = None
        if denoising:
            dn_features, dn_references, dn_state = self._prepare_denoising_queries(
                targets, images.device
            )
            query_features = torch.cat((dn_features, query_features), dim=1)
            reference_boxes = torch.cat((dn_references, reference_boxes), dim=1)
        decoder_features, decoder_references = self.transformer.decoder(
            query_features.transpose(0, 1),
            memory,
            refpoints_unsigmoid=reference_boxes.transpose(0, 1),
            level_start_index=inputs.level_start_index,
            spatial_shapes=inputs.spatial_shapes,
            valid_ratios=inputs.valid_ratios,
            memory_key_padding_mask=inputs.padding_mask,
            pos=inputs.position,
            # The evaluation sampler supplies this mask as its second
            # attention-mask argument; the decoder applies it only after
            # diffusion keypoint queries are expanded.
            tgt_mask=(
                self.training_box_attention_mask(targets, images.shape[0], images.device)
                if denoising else None
            ),
            tgt_mask2=(
                self.training_attention_mask(targets, images.shape[0], images.device)
                if denoising else self.inference_attention_mask(images.shape[0], images.device)
            ),
            diffusion_keypoints=((diffusion_keypoints / self.scale) + 1.0) / 2.0,
        )
        decoder_outputs = []
        denoising_outputs = []
        dn_count = self.transformer.decoder.num_dn if denoising else 0
        for features, references, box_head, class_head in zip(
            decoder_features,
            decoder_references[:-1],
            self.bbox_embed,
            self.class_embed,
        ):
            dn_features = features[:, :dn_count]
            dn_references = references[:, :dn_count]
            normal_features = features[:, dn_count:]
            normal_references = references[:, dn_count:]
            if dn_count:
                denoising_outputs.append({
                    "pred_logits": class_head(dn_features),
                    "pred_boxes": (box_head(dn_features) + inverse_sigmoid(dn_references)).sigmoid(),
                })
            if normal_features.shape[1] > self.num_queries:
                stride = self.transformer.decoder.num_body_points + 1
                box_indices = torch.arange(0, normal_features.shape[1], stride, device=features.device)
                point_indices = torch.arange(normal_features.shape[1], device=features.device)
                point_indices = point_indices[point_indices.remainder(stride) != 0]
                box_features = normal_features.index_select(1, box_indices)
                box_references = normal_references.index_select(1, box_indices)
                point_features = normal_features.index_select(1, point_indices)
                point_references = normal_references.index_select(1, point_indices)
                boxes = (box_head(box_features) + inverse_sigmoid(box_references)).sigmoid()
                pose_head = self.pose_embed[len(decoder_outputs) - self.transformer.decoder.num_box_decoder_layers]
                keypoint_xy = (pose_head(point_features) + inverse_sigmoid(point_references[..., :2])).sigmoid()
                visibility = torch.ones_like(keypoint_xy[..., :1]).sigmoid()
                point_count = self.transformer.decoder.num_body_points
                keypoint_xy = keypoint_xy.view(features.shape[0], -1, point_count, 2)
                visibility = visibility.view(features.shape[0], -1, point_count, 1)
                interleaved_keypoints = torch.cat((keypoint_xy, visibility), dim=-1)
                interleaved_keypoints = interleaved_keypoints.flatten(2)
                keypoints = torch.zeros_like(interleaved_keypoints)
                keypoints[..., : point_count * 2 : 2] = interleaved_keypoints[..., 0::3]
                keypoints[..., 1 : point_count * 2 : 2] = interleaved_keypoints[..., 1::3]
                keypoints[..., point_count * 2 :] = interleaved_keypoints[..., 2::3]
                logits = class_head(box_features)
            else:
                boxes = (box_head(normal_features) + inverse_sigmoid(normal_references)).sigmoid()
                logits = class_head(normal_features)
                keypoints = torch.zeros(
                    normal_features.shape[0], normal_features.shape[1],
                    self.transformer.decoder.num_body_points * 3,
                    device=normal_features.device, dtype=normal_features.dtype
                )
            decoder_outputs.append({"pred_logits": logits, "pred_boxes": boxes, "pred_keypoints": keypoints})
        # The final decoder prediction is returned separately. Auxiliary
        # losses are applied only to the preceding decoder outputs.
        result = {**decoder_outputs[-1], "aux_outputs": decoder_outputs[:-1]}
        if self.training:
            result["interm_outputs"] = {
                "pred_logits": intermediate_logits,
                "pred_boxes": intermediate_boxes,
                "pred_keypoints": torch.zeros(
                    intermediate_logits.shape[0],
                    intermediate_logits.shape[1],
                    self.transformer.decoder.num_body_points * 3,
                    dtype=intermediate_logits.dtype,
                    device=intermediate_logits.device,
                ),
            }
        if dn_state is not None:
            result["dn_class_input"] = dn_state["labels"]
            result["dn_bbox_input"] = dn_state["boxes"]
            result["dn_class_pred"] = denoising_outputs[-1]["pred_logits"]
            result["dn_bbox_pred"] = denoising_outputs[-1]["pred_boxes"]
            result["dn_aux_outputs"] = denoising_outputs[:-1]
        return result

    @torch.no_grad()
    def sample_keypoints(
        self,
        images: Tensor,
        *,
        padding_mask: Tensor | None = None,
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
            final_output = self(
                images,
                state,
                timestep,
                padding_mask=padding_mask,
            )
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
        scores = class_logits.max(-1).values
        available_count = scores.shape[1]
        if available_count == 0:
            raise RuntimeError("encoder produced no valid tokens for two-stage query selection")
        selected_count = min(query_count, available_count)
        topk = torch.topk(scores, selected_count, dim=1).indices
        if selected_count < query_count:
            repeat_count = (query_count + selected_count - 1) // selected_count
            topk = topk.repeat(1, repeat_count)[:, :query_count]
        gather_boxes = topk.unsqueeze(-1).expand(-1, -1, 4)
        gather_features = topk.unsqueeze(-1).expand(-1, -1, output_memory.shape[-1])
        return (
            torch.gather(output_memory, 1, gather_features).detach(),
            torch.gather(box_logits, 1, gather_boxes).detach(),
            torch.gather(class_logits, 1, gather_boxes[..., :class_logits.shape[-1]]),
            torch.gather(box_logits, 1, gather_boxes).sigmoid(),
        )

    def _level_position_offsets(self, inputs: TransformerInputs) -> Tensor:
        offsets = []
        for level, shape in enumerate(inputs.spatial_shapes.tolist()):
            offsets.append(self.transformer.level_embed[level].view(1, 1, -1).expand(inputs.memory.shape[0], shape[0] * shape[1], -1))
        return torch.cat(offsets, 1)

"""Structured person tokens with center/size flow geometry."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.models.pipelines.pose_estimators.diffusionregpose.transformer import (
    DeformableTransformerDecoderLayer, MLP,
)
from .model import PoseFlowHypothesisStage, PoseFlowV5Model
from .state import fourier_encode_coordinates, state_reference_gate


class PoseFlowV11DecoderLayer(DeformableTransformerDecoderLayer):
    def __init__(self, d_model=256, d_ffn=2048, dropout=0.1,
                 num_levels=4, num_heads=8, num_points=4):
        super().__init__(d_model, d_ffn, dropout, num_levels, num_heads, num_points)
        self.exist_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.exist_norm = nn.LayerNorm(d_model)
        self.exist_dropout = nn.Dropout(dropout)
        self.last_existence_update: Tensor | None = None

    def _within_person_attention(self, tokens: Tensor) -> Tensor:
        batch_size, person_count, token_count, width = tokens.shape
        queries = tokens.reshape(batch_size * person_count, token_count, width).transpose(0, 1)
        update = self.self_attn(queries, queries, queries, need_weights=False)[0]
        updated = self.norm2(queries + self.dropout2(update))
        return updated.transpose(0, 1).reshape_as(tokens)

    def forward(self, tokens: Tensor, dn_tokens: Tensor | None, *, references: Tensor,
                memory: Tensor, spatial_shapes: Tensor, level_start_index: Tensor,
                padding_mask: Tensor | None = None, existence_mask: Tensor | None = None,
                collect_diagnostics: bool = False) -> tuple[Tensor, Tensor | None]:
        batch_size, person_count, token_count, width = tokens.shape
        dn_count = 0 if dn_tokens is None else dn_tokens.shape[1]
        clean_flat = tokens.flatten(1, 2)
        flat = clean_flat if dn_tokens is None else torch.cat((dn_tokens.flatten(1, 2), clean_flat), dim=1)
        crossed = self.cross_attn(flat, references.contiguous(), memory,
                                  spatial_shapes, level_start_index, padding_mask)
        flat = self.norm1(flat + self.dropout1(crossed))
        tokens = self._within_person_attention(flat[:, dn_count * 3:].reshape_as(tokens))
        if dn_tokens is not None:
            dn_tokens = self._within_person_attention(flat[:, :dn_count * 3].reshape_as(dn_tokens))
        existence = tokens[:, :, 0]
        if dn_tokens is not None:
            existence = torch.cat((dn_tokens[:, :, 0], existence), dim=1)
        queries = existence.transpose(0, 1)
        update = self.exist_attn(queries, queries, queries, attn_mask=existence_mask, need_weights=False)[0]
        updated = self.exist_norm(queries + self.exist_dropout(update)).transpose(0, 1)
        self.last_existence_update = (
            (updated[:, dn_count:] - existence[:, dn_count:]).detach().float().square().mean().sqrt()
            if collect_diagnostics else None
        )
        tokens = torch.cat((updated[:, dn_count:, None], tokens[:, :, 1:]), dim=2)
        if dn_tokens is not None:
            dn_tokens = torch.cat((updated[:, :dn_count, None], dn_tokens[:, :, 1:]), dim=2)
        flat = tokens.flatten(1, 2)
        if dn_tokens is not None:
            flat = torch.cat((dn_tokens.flatten(1, 2), flat), dim=1)
        flat = self.norm3(flat + self.dropout4(self.dropout3(self.linear2(F.relu(self.linear1(flat))))))
        tokens = flat[:, dn_count * 3:].reshape(batch_size, person_count, token_count, width)
        if dn_tokens is not None:
            dn_tokens = flat[:, :dn_count * 3].reshape_as(dn_tokens)
        return tokens, dn_tokens


class PoseFlowV11Model(PoseFlowV5Model):
    def __init__(self, *args, sampling_joint_conditioning: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampling_joint_conditioning = sampling_joint_conditioning
        decoder = self.transformer.decoder
        width = decoder.d_model
        self.hypothesis_stage = PoseFlowHypothesisStage(
            num_queries=self.num_queries, state_width=4, hidden_dim=width,
            num_heads=decoder.num_heads, fourier_bands=self.hypothesis_stage.fourier_bands,
        )
        # Empty legacy heads retain the checked V5 initialization interface.
        self.initial_coordinate_head = nn.Identity()
        self.initial_class_head = nn.Identity()
        self.v5_box_heads = nn.ModuleList()
        self.v4_class_heads = nn.ModuleList()
        self.v4_pose_heads = nn.ModuleList()
        self.v4_pose_hw_heads = nn.ModuleList()
        self.role_embeddings = nn.Embedding(3 + decoder.num_body_points, width)
        self.joint_conditioning = MLP(4 * self.hypothesis_stage.fourier_bands, width, width, 2)
        self.dn_box_conditioning = MLP(4, width, width, 2)
        self.point_head = MLP(width, width, 2, 3)
        self.size_head = MLP(width, width, 2, 3)
        self.existence_head = MLP(width, width, self.num_classes, 3)
        nn.init.constant_(self.existence_head.layers[-1].bias, -math.log(99))
        layers = []
        for original in decoder.layers:
            layer = PoseFlowV11DecoderLayer(
                d_model=width, d_ffn=original.linear1.out_features,
                dropout=original.dropout1.p, num_levels=original.cross_attn.num_levels,
                num_heads=decoder.num_heads, num_points=original.cross_attn.num_points,
            )
            layer.load_state_dict(original.state_dict(), strict=False)
            layers.append(layer)
        decoder.layers = nn.ModuleList(layers)
        self._last_token_diagnostics: list[dict[str, Tensor]] = []

    def predict_coarse(self, latent: Tensor) -> dict[str, Tensor]:
        return {
            'pred_logits': self.existence_head(latent),
            'pred_boxes': torch.cat((self.point_head(latent).sigmoid(), self.size_head(latent).sigmoid()), dim=-1),
            'poseflow_full_pose_loss': torch.zeros((), dtype=torch.bool, device=latent.device),
        }

    def expand_tokens(self, latent: Tensor, person_state: Tensor, times: Tensor,
                      selected: Tensor, *, apply_joint_conditioning: bool = True) -> Tensor:
        selected_latent = latent.gather(1, selected[..., None].expand(-1, -1, latent.shape[-1]))
        tokens = selected_latent[:, :, None] + self.role_embeddings.weight[None, None]
        if not apply_joint_conditioning:
            return tokens
        joint_state = person_state[..., 4:].gather(
            1, selected[..., None].expand(-1, -1, person_state.shape[-1] - 4),
        ).reshape(*selected.shape, -1, 2) / self.scale
        if not self.flow_state_conditioning:
            joint_state = torch.zeros_like(joint_state)
        conditioning = self.joint_conditioning(fourier_encode_coordinates(
            joint_state, num_bands=self.hypothesis_stage.fourier_bands,
        )) * state_reference_gate(times)[:, None, None, None]
        return torch.cat((tokens[:, :, :3], tokens[:, :, 3:] + conditioning), dim=2)

    def predict_tokens(self, tokens: Tensor) -> tuple[dict[str, Tensor], Tensor]:
        center = self.point_head(tokens[:, :, 1]).sigmoid()
        size = self.size_head(tokens[:, :, 2]).sigmoid()
        boxes = torch.cat((center, size), dim=-1)
        points = self.point_head(tokens[:, :, 3:]).sigmoid()
        reference_xy = torch.cat((center[:, :, None].expand(-1, -1, 3, -1), points), dim=2)
        references = torch.cat((reference_xy, size[:, :, None].expand_as(reference_xy)), dim=-1)
        prediction = {'pred_logits': self.existence_head(tokens[:, :, 0]), 'pred_boxes': boxes}
        if tokens.shape[2] > 3:
            prediction['pred_keypoints'] = self._keypoints_from_xy(points)
            prediction['poseflow_full_pose_loss'] = torch.ones((), dtype=torch.bool, device=tokens.device)
        return prediction, references

    def existence_attention_mask(self, targets, batch_size: int, device) -> Tensor:
        if targets is None:
            return torch.zeros(self.num_proposals, self.num_proposals, dtype=torch.bool, device=device)
        count = self.transformer.decoder.num_dn + self.num_proposals
        return self.training_box_attention_mask(targets, batch_size, device)[:, :count, :count]

    def _prepare_dn_tokens(self, targets, device):
        features, references, state = self._prepare_denoising_queries(targets, device)
        conditioning = self.dn_box_conditioning(references.sigmoid())
        tokens = features[:, :, None] + self.role_embeddings.weight[None, None, :3] + conditioning[:, :, None]
        return tokens, state

    def _token_summary(self, before: Tensor, after: Tensor, references: Tensor,
                       layer: PoseFlowV11DecoderLayer) -> dict[str, Tensor]:
        result = {}
        for name, subset in [('existence', slice(0, 1)), ('center', slice(1, 2)),
                             ('size', slice(2, 3)), ('joints', slice(3, None))]:
            result[f'{name}_rms'] = after[:, :, subset].detach().float().square().mean().sqrt()
            result[f'{name}_update_rms'] = (after[:, :, subset] - before[:, :, subset]).detach().float().square().mean().sqrt()
        geometry = references.detach().float()
        result['width_mean'] = geometry[:, :, 0, 2].mean()
        result['height_mean'] = geometry[:, :, 0, 3].mean()
        result['tiny_box_fraction'] = (geometry[:, :, 0, 2:].amin(-1) < .01).float().mean()
        result['reference_saturation_fraction'] = ((geometry[..., :2] < .01) | (geometry[..., :2] > .99)).float().mean()
        result['cross_person_existence_update_rms'] = layer.last_existence_update
        return result

    def last_token_diagnostics(self) -> list[dict[str, Tensor]]:
        return self._last_token_diagnostics

    def _run_token_refiners(self, tokens: Tensor, dn_tokens: Tensor | None, *,
                           inputs, memory: Tensor, mask: Tensor,
                           collect_diagnostics: bool):
        predictions, dn_predictions = [], []
        _, references = self.predict_tokens(tokens)
        self._last_token_diagnostics = []
        dn_references = None
        if dn_tokens is not None:
            _, dn_references = self.predict_tokens(dn_tokens)
        for layer in self.transformer.decoder.layers:
            flat_references = references.flatten(1, 2)
            if dn_tokens is not None:
                flat_references = torch.cat((dn_references.flatten(1, 2), flat_references), dim=1)
            reference_input = flat_references[:, :, None] * torch.cat((inputs.valid_ratios, inputs.valid_ratios), dim=-1)[:, None]
            before = tokens
            tokens, dn_tokens = layer(
                tokens, dn_tokens, references=reference_input, memory=memory,
                spatial_shapes=inputs.spatial_shapes, level_start_index=inputs.level_start_index,
                padding_mask=inputs.padding_mask, existence_mask=mask,
                collect_diagnostics=collect_diagnostics,
            )
            normalized = self.transformer.decoder.norm(tokens)
            prediction, references = self.predict_tokens(normalized)
            predictions.append(prediction)
            if collect_diagnostics:
                self._last_token_diagnostics.append(self._token_summary(before, tokens, references, layer))
            if dn_tokens is not None:
                dn_prediction, dn_references = self.predict_tokens(self.transformer.decoder.norm(dn_tokens))
                dn_predictions.append(dn_prediction)
        return predictions, dn_predictions

    def forward(self, images: Tensor, diffusion_keypoints: Tensor, timesteps: Tensor,
                targets=None, padding_mask=None, *, person_state: Tensor,
                return_selection_metadata=False, collect_diagnostics=False,
                apply_joint_conditioning: bool = True):
        del diffusion_keypoints
        self._validate_person_state(person_state, images.shape[0])
        if padding_mask is None:
            padding_mask = torch.zeros(images.shape[0], *images.shape[2:], dtype=torch.bool, device=images.device)
        inputs = self.prepare_transformer_inputs(images, padding_mask)
        memory = self.encode_features(inputs, timesteps)
        time_embedding = self.transformer.block_time_mlp(self.transformer.time_mlp(timesteps))
        box_state = person_state[..., :4] / self.scale
        if not self.flow_state_conditioning:
            box_state = torch.zeros_like(box_state)
        latent = self.hypothesis_stage(
            memory, inputs.position + self._level_position_offsets(inputs), inputs.padding_mask,
            box_state, timesteps, time_embedding, collect_diagnostics=collect_diagnostics,
        )
        coarse = self.predict_coarse(latent)
        selected = coarse['pred_logits'].amax(-1).topk(self.num_proposals, dim=1).indices
        tokens = self.expand_tokens(latent, person_state, timesteps, selected,
                                    apply_joint_conditioning=apply_joint_conditioning)
        dn_tokens, dn_state = None, None
        if self.training and targets is not None and self.transformer.decoder.num_dn:
            dn_tokens, dn_state = self._prepare_dn_tokens(targets, images.device)
        mask = self.existence_attention_mask(targets if dn_tokens is not None else None, images.shape[0], images.device)
        decoder_memory = self.decoder_feature_memory(inputs, memory)
        predictions, dn_predictions = self._run_token_refiners(
            tokens, dn_tokens, inputs=inputs, memory=decoder_memory, mask=mask,
            collect_diagnostics=collect_diagnostics,
        )
        self._last_v4_diagnostics = self.hypothesis_stage.last_diagnostics() if collect_diagnostics else None
        result = {**predictions[-1], 'aux_outputs': [coarse, *predictions[:-1]]}
        if return_selection_metadata:
            result['selected_person_indices'] = selected
        if dn_state is not None:
            result.update(dn_class_input=dn_state['labels'], dn_bbox_input=dn_state['boxes'],
                          dn_class_pred=dn_predictions[-1]['pred_logits'], dn_bbox_pred=dn_predictions[-1]['pred_boxes'],
                          dn_aux_outputs=[None, *dn_predictions[:-1]])
        return result

    def sample_keypoints(self, images: Tensor, *, padding_mask=None, num_proposals=None,
                         generator=None, collect_step_diagnostics=False):
        from .training_v11 import flow_sample_poseflow_v11_states
        return flow_sample_poseflow_v11_states(
            self, images, padding_mask=padding_mask, num_proposals=num_proposals,
            sampling_steps=self.flow_sampling_steps, ensemble=self.flow_ensemble,
            generator=generator, collect_step_diagnostics=collect_step_diagnostics,
        )

"""PoseFlow model identity and flow-sampling boundary."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

from src.data.skeleton import CROWDPOSE_EDGE_LIST
from src.models.pipelines.pose_estimators.diffusionregpose.model import (
    DiffusionRegPoseModel,
)
from src.models.pipelines.pose_estimators.diffusionregpose.backbone import (
    PositionEmbeddingSineHW,
)
from src.models.pipelines.pose_estimators.diffusionregpose.transformer import (
    MLP,
    MultiScaleDeformableAttention,
    inverse_sigmoid,
    sine_embedding_for_position,
)
from src.models.pipelines.pose_estimators.diffusionregpose.model import TransformerInputs

from .state import (
    blend_box_references,
    corners_to_cxcywh,
    cxcywh_to_corners,
    fourier_encode_coordinates,
    state_reference_gate,
    unpack_person_state,
)


class PoseHypothesisCodebook(nn.Module):
    """Shared pose hypotheses read by image-conditioned decoder queries."""

    def __init__(
        self,
        *,
        codebook_size: int,
        hidden_dim: int,
        num_heads: int,
        initial_gate: float = 0.0,
    ) -> None:
        super().__init__()
        if not math.isfinite(initial_gate):
            raise ValueError("initial_gate must be finite")
        self.embeddings = nn.Parameter(torch.empty(codebook_size, hidden_dim))
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=0.0)
        self.residual_gate = nn.Parameter(torch.tensor(initial_gate))
        self._diagnostic_sums: dict[str, Tensor] | None = None
        nn.init.xavier_uniform_(self.embeddings)

    def reset_diagnostics(self) -> None:
        self._diagnostic_sums = None

    def consume_diagnostics(self) -> dict[str, Tensor] | None:
        if self._diagnostic_sums is None:
            return None
        sums = self._diagnostic_sums
        self._diagnostic_sums = None
        query_rms = (sums["query_squared_sum"] / sums["element_count"]).sqrt()
        attended_rms = (sums["attended_squared_sum"] / sums["element_count"]).sqrt()
        residual_rms = self.residual_gate.detach().abs() * attended_rms
        diagnostics = {
            "gate": self.residual_gate.detach(),
            "query_rms": query_rms,
            "attended_rms": attended_rms,
            "residual_rms": residual_rms,
            "residual_to_query_rms": residual_rms / query_rms.clamp_min(torch.finfo(query_rms.dtype).eps),
        }
        if "attention_entropy_sum" in sums:
            attention_entropy = sums["attention_entropy_sum"] / sums["attention_count"]
            codebook_size = self.embeddings.shape[0]
            normalized_entropy = (
                attention_entropy / math.log(codebook_size)
                if codebook_size > 1
                else torch.zeros_like(attention_entropy)
            )
            diagnostics["normalized_attention_entropy"] = normalized_entropy
            diagnostics["effective_slot_count"] = attention_entropy.exp().clamp(
                max=float(codebook_size)
            )
        return diagnostics

    def _collect_diagnostics(
        self,
        clean_features: Tensor,
        attended_features: Tensor,
        attention_weights: Tensor | None,
    ) -> None:
        if self._diagnostic_sums is None:
            self._diagnostic_sums = {}
        sums = self._diagnostic_sums
        element_count = clean_features.new_tensor(clean_features.numel())
        for name, value in {
            "query_squared_sum": clean_features.detach().square().sum(),
            "attended_squared_sum": attended_features.detach().square().sum(),
            "element_count": element_count,
        }.items():
            sums[name] = sums.get(name, torch.zeros_like(value)) + value
        if attention_weights is None:
            return
        entropy = -(
            attention_weights.detach()
            * attention_weights.detach().clamp_min(torch.finfo(attention_weights.dtype).eps).log()
        ).sum(dim=-1)
        entropy_sum = entropy.sum()
        attention_count = entropy.new_tensor(entropy.numel())
        sums["attention_entropy_sum"] = sums.get(
            "attention_entropy_sum", torch.zeros_like(entropy_sum)
        ) + entropy_sum
        sums["attention_count"] = sums.get(
            "attention_count", torch.zeros_like(attention_count)
        ) + attention_count

    def forward(
        self,
        query_features: Tensor,
        *,
        denoising_query_count: int = 0,
        collect_diagnostics: bool = False,
        collect_attention: bool = False,
    ) -> Tensor:
        if not 0 <= denoising_query_count <= query_features.shape[0]:
            raise ValueError("denoising_query_count must index the query sequence")
        denoising_features = query_features[:denoising_query_count]
        clean_features = query_features[denoising_query_count:]
        if clean_features.shape[0] == 0:
            return query_features
        codebook = self.embeddings[:, None].expand(-1, clean_features.shape[1], -1)
        attended, attention_weights = self.attention(
            clean_features,
            codebook,
            codebook,
            need_weights=collect_attention,
        )
        if collect_diagnostics:
            self._collect_diagnostics(clean_features, attended, attention_weights)
        conditioned_clean_features = clean_features + self.residual_gate * attended
        return torch.cat((denoising_features, conditioned_clean_features), dim=0)


class PoseFlowHypothesisStage(nn.Module):
    """Build one independent person hypothesis per flow call from image tokens."""

    def __init__(
        self,
        *,
        num_queries: int,
        state_width: int,
        hidden_dim: int,
        num_heads: int,
        fourier_bands: int = 8,
    ) -> None:
        super().__init__()
        if num_queries < 1:
            raise ValueError("num_queries must be positive")
        if state_width < 1:
            raise ValueError("state_width must be positive")
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if fourier_bands < 1:
            raise ValueError("fourier_bands must be positive")
        self.num_queries = num_queries
        self.state_width = state_width
        self.hidden_dim = hidden_dim
        self.fourier_bands = fourier_bands
        self.slot_embeddings = nn.Embedding(num_queries, hidden_dim)
        self.coordinate_projection = MLP(
            state_width * 2 * fourier_bands, hidden_dim, hidden_dim, 2
        )
        self.time_projection = nn.Linear(hidden_dim, hidden_dim)
        self.query_projection = MLP(hidden_dim * 3, hidden_dim, hidden_dim, 2)
        self.image_token_projection = nn.Linear(hidden_dim, hidden_dim)
        self.image_position_projection = nn.Linear(hidden_dim, hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=0.0, batch_first=True
        )
        self.update_projection = MLP(hidden_dim * 2, hidden_dim, hidden_dim, 2)
        self._last_diagnostics: dict[str, Tensor] | None = None

    @staticmethod
    def _diagnostic_rms(value: Tensor) -> Tensor:
        """Return a detached finite scalar without retaining an activation graph."""
        return torch.nan_to_num(value.detach().float()).square().mean().sqrt()

    def last_diagnostics(self) -> dict[str, Tensor] | None:
        """Return the most recent opt-in hypothesis-stage diagnostics."""
        return self._last_diagnostics

    def forward(
        self,
        image_tokens: Tensor,
        image_positions: Tensor,
        image_padding_mask: Tensor | None,
        person_state: Tensor,
        timesteps: Tensor,
        timestep_embedding: Tensor,
        *,
        collect_diagnostics: bool = False,
    ) -> Tensor:
        """Return `[B, N, D]` hypotheses, independently for this flow call."""
        batch_size = person_state.shape[0]
        if person_state.shape != (batch_size, self.num_queries, self.state_width):
            raise ValueError(
                "person_state must have shape "
                f"({batch_size}, {self.num_queries}, {self.state_width})"
            )
        if image_tokens.ndim != 3 or image_tokens.shape[0] != batch_size:
            raise ValueError("image_tokens must have matching [B, L, D] shape")
        if image_tokens.shape[-1] != self.hidden_dim or image_positions.shape != image_tokens.shape:
            raise ValueError("image token and position dimensions must match hidden_dim")
        if image_padding_mask is not None and image_padding_mask.shape != image_tokens.shape[:2]:
            raise ValueError("image_padding_mask must have shape [B, L]")
        if timesteps.shape != (batch_size,):
            raise ValueError("timesteps must have shape [B]")
        if timestep_embedding.shape != (batch_size, self.hidden_dim):
            raise ValueError("timestep_embedding must have shape [B, D]")

        gate = state_reference_gate(timesteps)
        coordinate_features = self.coordinate_projection(
            fourier_encode_coordinates(person_state, num_bands=self.fourier_bands)
        )
        slot_features = self.slot_embeddings.weight[None].expand(batch_size, -1, -1)
        time_features = self.time_projection(timestep_embedding)[:, None].expand(
            -1, self.num_queries, -1
        )
        pre_attention = self.query_projection(
            torch.cat(
                (
                    slot_features,
                    gate[:, None, None] * coordinate_features,
                    time_features,
                ),
                dim=-1,
            )
        )
        token_features = self.image_token_projection(image_tokens) + self.image_position_projection(
            image_positions
        )
        image_update = self.cross_attention(
            pre_attention,
            token_features,
            token_features,
            key_padding_mask=image_padding_mask,
            need_weights=False,
        )[0]
        hypothesis = self.update_projection(
            torch.cat(
                (pre_attention, (1.0 - gate[:, None, None]) * image_update), dim=-1
            )
        )
        if collect_diagnostics:
            gated_coordinate_features = gate[:, None, None] * coordinate_features
            self._last_diagnostics = {
                "flow_time_mean": timesteps.detach().float().mean(),
                "coordinate_features_rms": self._diagnostic_rms(coordinate_features),
                "coordinate_gated_rms": self._diagnostic_rms(gated_coordinate_features),
                "image_tokens_rms": self._diagnostic_rms(token_features),
                "image_update_rms": self._diagnostic_rms(image_update),
                "pre_attention_rms": self._diagnostic_rms(pre_attention),
                "hypothesis_rms": self._diagnostic_rms(hypothesis),
            }
        else:
            self._last_diagnostics = None
        return hypothesis


class PoseFlowModel(DiffusionRegPoseModel):
    """ED-Pose architecture whose proposal state is sampled as endpoint flow."""

    def __init__(
        self,
        *args,
        flow_sampling_steps: int = 1,
        flow_renewal: bool = False,
        flow_renewal_score_threshold: float = 0.5,
        flow_ensemble: bool = False,
        pose_codebook_size: int = 0,
        pose_codebook_layers: tuple[int, ...] | None = None,
        pose_codebook_initial_gate: float = 0.0,
        **kwargs,
    ) -> None:
        if flow_sampling_steps < 1:
            raise ValueError("flow_sampling_steps must be positive")
        if pose_codebook_size < 0:
            raise ValueError("pose_codebook_size must be non-negative")
        if not math.isfinite(pose_codebook_initial_gate):
            raise ValueError("pose_codebook_initial_gate must be finite")
        super().__init__(*args, **kwargs)
        self.flow_sampling_steps = flow_sampling_steps
        self.flow_renewal = flow_renewal
        self.flow_renewal_score_threshold = flow_renewal_score_threshold
        self.flow_ensemble = flow_ensemble
        first_pose_layer = self.transformer.decoder.num_box_decoder_layers
        if pose_codebook_layers is None:
            pose_codebook_layers = (first_pose_layer,) if pose_codebook_size else ()
        if pose_codebook_size == 0 and pose_codebook_layers:
            raise ValueError("pose_codebook_layers requires a positive pose_codebook_size")
        if len(set(pose_codebook_layers)) != len(pose_codebook_layers):
            raise ValueError("pose_codebook_layers must not contain duplicates")
        for layer_index in pose_codebook_layers:
            if not 0 <= layer_index < self.transformer.decoder.num_layers:
                raise ValueError("pose_codebook_layers must select decoder layers")
        self.pose_codebook_layers = pose_codebook_layers
        self.pose_codebook = (
            PoseHypothesisCodebook(
                codebook_size=pose_codebook_size,
                hidden_dim=self.transformer.decoder.d_model,
                num_heads=self.transformer.decoder.num_heads,
                initial_gate=pose_codebook_initial_gate,
            )
            if pose_codebook_layers
            else None
        )
        self._last_codebook_diagnostics: dict[str, Tensor] | None = None

    def post_image_attention_by_layer(self) -> dict[int, nn.Module] | None:
        if self.pose_codebook is None:
            return None
        return {
            layer_index: self.pose_codebook
            for layer_index in self.pose_codebook_layers
        }

    def forward(
        self,
        *args,
        collect_codebook_diagnostics: bool = False,
        collect_codebook_attention: bool = False,
        **kwargs,
    ) -> dict[str, Tensor | list[dict[str, Tensor]]]:
        if collect_codebook_diagnostics and self.pose_codebook is not None:
            self.pose_codebook.reset_diagnostics()
        outputs = super().forward(
            *args,
            collect_post_image_attention_diagnostics=collect_codebook_diagnostics,
            collect_post_image_attention_weights=collect_codebook_attention,
            **kwargs,
        )
        self._last_codebook_diagnostics = (
            self.pose_codebook.consume_diagnostics()
            if collect_codebook_diagnostics and self.pose_codebook is not None
            else None
        )
        return outputs

    def last_codebook_diagnostics(self) -> dict[str, Tensor] | None:
        return self._last_codebook_diagnostics

    def sample_keypoints(
        self,
        images: Tensor,
        *,
        padding_mask: Tensor | None = None,
        num_proposals: int | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, Tensor | list[dict[str, Tensor]]]:
        """Sample through PoseFlow's fixed-source Euler solver."""
        from .training import flow_sample_keypoints

        return flow_sample_keypoints(
            self,
            images,
            padding_mask=padding_mask,
            num_proposals=num_proposals,
            sampling_steps=self.flow_sampling_steps,
            renewal=self.flow_renewal,
            renewal_score_threshold=self.flow_renewal_score_threshold,
            ensemble=self.flow_ensemble,
            generator=generator,
        )


class PoseFlowV3Model(PoseFlowModel):
    """PoseFlow with a joint two-corner box and keypoint transport state."""

    def __init__(self, *args, flow_state_conditioning: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        state_width = 4 + 2 * self.transformer.decoder.num_body_points
        hidden_dim = self.transformer.decoder.d_model
        self.person_state_embedding = MLP(state_width, hidden_dim, hidden_dim, 2)
        self.flow_state_conditioning = flow_state_conditioning

    def _validate_person_state(self, person_state: Tensor, batch_size: int) -> None:
        expected_shape = (
            batch_size,
            self.num_queries,
            4 + 2 * self.transformer.decoder.num_body_points,
        )
        if person_state.shape != expected_shape:
            raise ValueError(f"person_state must have shape {expected_shape}")

    def _combine_clean_query_pool(
        self,
        image_features: Tensor,
        image_references: Tensor,
        *,
        active_query_features: Tensor | None,
        active_image_references: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        if active_query_features is None and active_image_references is None:
            return image_features, image_references
        if active_query_features is None or active_image_references is None:
            raise ValueError("active query features and image references must be provided together")
        active_count = active_query_features.shape[1]
        if active_count > self.num_queries:
            raise ValueError("active query count must not exceed num_queries")
        if active_query_features.shape != (
            image_features.shape[0], active_count, image_features.shape[-1]
        ):
            raise ValueError("active query features must match image feature dimensions")
        if active_image_references.shape != (image_features.shape[0], active_count, 4):
            raise ValueError("active image references must have one cxcywh box per active query")
        fresh_count = self.num_queries - active_count
        return (
            torch.cat((active_query_features, image_features[:, :fresh_count]), dim=1),
            torch.cat((active_image_references, image_references[:, :fresh_count]), dim=1),
        )

    def forward(
        self,
        images: Tensor,
        diffusion_keypoints: Tensor,
        timesteps: Tensor,
        targets: list[dict[str, Tensor]] | None = None,
        padding_mask: Tensor | None = None,
        *,
        person_state: Tensor,
        active_query_features: Tensor | None = None,
        active_image_references: Tensor | None = None,
        return_selection_metadata: bool = False,
    ) -> dict[str, Tensor | list[dict[str, Tensor]]]:
        """Decode a 200-slot pool with fixed image/state geometry blending."""
        del diffusion_keypoints
        if padding_mask is None:
            padding_mask = torch.zeros(
                images.shape[0], *images.shape[2:], dtype=torch.bool, device=images.device
            )
        self._validate_person_state(person_state, images.shape[0])
        if timesteps.shape != (images.shape[0],):
            raise ValueError("timesteps must contain one continuous value per image")

        inputs = self.prepare_transformer_inputs(images, padding_mask)
        memory = self.encode_features(inputs, timesteps)
        (
            image_query_features,
            image_reference_logits,
            intermediate_logits,
            intermediate_boxes,
        ) = self.select_two_stage_queries(memory, inputs)
        query_features, image_references = self._combine_clean_query_pool(
            image_query_features,
            image_reference_logits.sigmoid(),
            active_query_features=active_query_features,
            active_image_references=active_image_references,
        )
        state_corners, state_keypoints = unpack_person_state(
            person_state, self.transformer.decoder.num_body_points
        )
        state_references = corners_to_cxcywh(((state_corners / self.scale) + 1.0) / 2.0)
        state_keypoints = ((state_keypoints / self.scale) + 1.0) / 2.0
        gate = state_reference_gate(timesteps)
        if self.flow_state_conditioning:
            query_features = query_features + gate[:, None, None] * self.person_state_embedding(
                person_state / self.scale
            )
            initial_clean_references = blend_box_references(
                image_references, state_references, timesteps
            )
            point_gate = gate
        else:
            initial_clean_references = image_references
            point_gate = torch.zeros_like(gate)

        denoising = self.training and targets is not None and self.transformer.decoder.num_dn > 0
        dn_state: dict[str, Tensor] | None = None
        if denoising:
            dn_features, dn_references, dn_state = self._prepare_denoising_queries(
                targets, images.device
            )
            query_features = torch.cat((dn_features, query_features), dim=1)
            initial_references = torch.cat((dn_references.sigmoid(), initial_clean_references), dim=1)
            reference_logits = torch.cat((dn_references, inverse_sigmoid(image_references)), dim=1)
        else:
            initial_references = initial_clean_references
            reference_logits = inverse_sigmoid(image_references)
        denoising_query_count = self.transformer.decoder.num_dn if denoising else 0
        decoder_result = self.transformer.decoder(
            query_features.transpose(0, 1),
            memory,
            refpoints_unsigmoid=reference_logits.transpose(0, 1),
            initial_reference_points=initial_references.transpose(0, 1),
            level_start_index=inputs.level_start_index,
            spatial_shapes=inputs.spatial_shapes,
            valid_ratios=inputs.valid_ratios,
            memory_key_padding_mask=inputs.padding_mask,
            pos=inputs.position,
            tgt_mask=(
                self.training_box_attention_mask(targets, images.shape[0], images.device)
                if denoising else None
            ),
            tgt_mask2=(
                self.training_attention_mask(targets, images.shape[0], images.device)
                if denoising else self.inference_attention_mask(images.shape[0], images.device)
            ),
            diffusion_keypoints=state_keypoints,
            state_keypoint_gate=point_gate,
            return_selection_metadata=return_selection_metadata,
            post_image_attention_by_layer=self.post_image_attention_by_layer(),
            denoising_query_count=denoising_query_count,
        )
        if return_selection_metadata:
            decoder_features, decoder_references, selection_metadata = decoder_result
        else:
            decoder_features, decoder_references = decoder_result
            selection_metadata = None

        decoder_outputs: list[dict[str, Tensor]] = []
        denoising_outputs: list[dict[str, Tensor]] = []
        for features, references, box_head, class_head in zip(
            decoder_features, decoder_references[:-1], self.bbox_embed, self.class_embed,
        ):
            dn_features = features[:, :denoising_query_count]
            dn_references = references[:, :denoising_query_count]
            normal_features = features[:, denoising_query_count:]
            normal_references = references[:, denoising_query_count:]
            if denoising_query_count:
                denoising_outputs.append({
                    "pred_logits": class_head(dn_features),
                    "pred_boxes": (
                        box_head(dn_features) + inverse_sigmoid(dn_references)
                    ).sigmoid(),
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
                pose_head = self.pose_embed[
                    len(decoder_outputs) - self.transformer.decoder.num_box_decoder_layers
                ]
                keypoint_xy = (
                    pose_head(point_features) + inverse_sigmoid(point_references[..., :2])
                ).sigmoid()
                visibility = torch.ones_like(keypoint_xy[..., :1]).sigmoid()
                point_count = self.transformer.decoder.num_body_points
                keypoint_xy = keypoint_xy.view(features.shape[0], -1, point_count, 2)
                visibility = visibility.view(features.shape[0], -1, point_count, 1)
                interleaved = torch.cat((keypoint_xy, visibility), dim=-1).flatten(2)
                keypoints = torch.zeros_like(interleaved)
                keypoints[..., : point_count * 2 : 2] = interleaved[..., 0::3]
                keypoints[..., 1 : point_count * 2 : 2] = interleaved[..., 1::3]
                keypoints[..., point_count * 2 :] = interleaved[..., 2::3]
                logits = class_head(box_features)
            else:
                boxes = (box_head(normal_features) + inverse_sigmoid(normal_references)).sigmoid()
                logits = class_head(normal_features)
                keypoints = torch.zeros(
                    normal_features.shape[0], normal_features.shape[1],
                    self.transformer.decoder.num_body_points * 3,
                    device=normal_features.device, dtype=normal_features.dtype,
                )
            decoder_outputs.append({
                "pred_logits": logits, "pred_boxes": boxes, "pred_keypoints": keypoints,
            })
        result: dict[str, Tensor | list[dict[str, Tensor]]] = {
            **decoder_outputs[-1], "aux_outputs": decoder_outputs[:-1],
        }
        if selection_metadata is not None:
            result.update(selection_metadata)
        if self.training:
            result["interm_outputs"] = {
                "pred_logits": intermediate_logits,
                "pred_boxes": intermediate_boxes,
                "pred_keypoints": torch.zeros(
                    intermediate_logits.shape[0], intermediate_logits.shape[1],
                    self.transformer.decoder.num_body_points * 3,
                    dtype=intermediate_logits.dtype, device=intermediate_logits.device,
                ),
            }
        if dn_state is not None:
            result["dn_class_input"] = dn_state["labels"]
            result["dn_bbox_input"] = dn_state["boxes"]
            result["dn_class_pred"] = denoising_outputs[-1]["pred_logits"]
            result["dn_bbox_pred"] = denoising_outputs[-1]["pred_boxes"]
            result["dn_aux_outputs"] = denoising_outputs[:-1]
        return result

    def sample_keypoints(
        self,
        images: Tensor,
        *,
        padding_mask: Tensor | None = None,
        num_proposals: int | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, Tensor | list[dict[str, Tensor]]]:
        """Sample the recurrent 50 active plus 150 fresh person-state pool."""
        from .training import flow_sample_person_states

        return flow_sample_person_states(
            self,
            images,
            padding_mask=padding_mask,
            num_proposals=num_proposals,
            sampling_steps=self.flow_sampling_steps,
            ensemble=self.flow_ensemble,
            generator=generator,
        )


class PoseFlowV4Model(PoseFlowModel):
    """Coordinate-flow PoseFlow with an independent image-token hypothesis stage."""

    def __init__(
        self,
        *args,
        flow_state_conditioning: bool = True,
        fourier_bands: int = 8,
        corner_order_loss_weight: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if self.num_proposals >= self.num_queries:
            raise ValueError("PoseFlow V4 requires fewer active than total queries")
        if fourier_bands < 1:
            raise ValueError("fourier_bands must be positive")
        if corner_order_loss_weight < 0.0:
            raise ValueError("corner_order_loss_weight must be non-negative")
        hidden_dim = self.transformer.decoder.d_model
        point_count = self.transformer.decoder.num_body_points
        state_width = 4 + 2 * point_count
        self.hypothesis_stage = PoseFlowHypothesisStage(
            num_queries=self.num_queries,
            state_width=state_width,
            hidden_dim=hidden_dim,
            num_heads=self.transformer.decoder.num_heads,
            fourier_bands=fourier_bands,
        )
        self.initial_coordinate_head = MLP(hidden_dim, hidden_dim, state_width, 3)
        self.initial_class_head = nn.Linear(hidden_dim, self.num_classes)
        nn.init.constant_(self.initial_class_head.bias, -math.log((1 - 0.01) / 0.01))
        decoder_layers = self.transformer.decoder.num_layers
        self.v4_box_heads = nn.ModuleList(
            MLP(hidden_dim, hidden_dim, 4, 3) for _ in range(decoder_layers)
        )
        self.v4_class_heads = nn.ModuleList(
            nn.Linear(hidden_dim, self.num_classes) for _ in range(decoder_layers)
        )
        self.v4_pose_heads = nn.ModuleList(
            MLP(hidden_dim, hidden_dim, 2, 3) for _ in range(decoder_layers)
        )
        self.v4_pose_hw_heads = nn.ModuleList(
            MLP(hidden_dim, hidden_dim, 2, 3) for _ in range(decoder_layers)
        )
        for class_head in self.v4_class_heads:
            nn.init.constant_(class_head.bias, -math.log((1 - 0.01) / 0.01))
        self.flow_state_conditioning = flow_state_conditioning
        self.corner_order_loss_weight = corner_order_loss_weight
        self._last_v4_diagnostics: dict[str, Tensor] | None = None

    def last_v4_diagnostics(self) -> dict[str, Tensor] | None:
        """Return detached diagnostics from the most recent opt-in V4 forward."""
        return self._last_v4_diagnostics

    def _validate_person_state(self, person_state: Tensor, batch_size: int) -> None:
        expected_shape = (
            batch_size,
            self.num_queries,
            4 + 2 * self.transformer.decoder.num_body_points,
        )
        if person_state.shape != expected_shape:
            raise ValueError(f"person_state must have shape {expected_shape}")

    @staticmethod
    def _corners_to_valid_boxes(corners: Tensor) -> tuple[Tensor, Tensor]:
        corner_a, corner_b = corners[..., :2], corners[..., 2:]
        displacement = corner_b - corner_a
        boxes = torch.cat(((corner_a + corner_b) / 2.0, displacement.abs()), dim=-1)
        order_penalty = (-displacement).relu().sum(dim=-1)
        return boxes, order_penalty

    def _keypoints_from_xy(self, keypoint_xy: Tensor) -> Tensor:
        batch_size, query_count, point_count, _ = keypoint_xy.shape
        visibility = torch.ones_like(keypoint_xy[..., :1]).sigmoid()
        interleaved = torch.cat((keypoint_xy, visibility), dim=-1).flatten(2)
        keypoints = torch.zeros_like(interleaved)
        keypoints[..., : point_count * 2 : 2] = interleaved[..., 0::3]
        keypoints[..., 1 : point_count * 2 : 2] = interleaved[..., 1::3]
        keypoints[..., point_count * 2 :] = interleaved[..., 2::3]
        return keypoints

    def _initial_prediction(self, hypothesis: Tensor) -> tuple[dict[str, Tensor], Tensor, Tensor]:
        point_count = self.transformer.decoder.num_body_points
        coordinates = self.initial_coordinate_head(hypothesis).sigmoid()
        corners, keypoint_xy = coordinates[..., :4], coordinates[..., 4:].view(
            hypothesis.shape[0], self.num_queries, point_count, 2
        )
        boxes, order_penalty = self._corners_to_valid_boxes(corners)
        return (
            {
                "pred_logits": self.initial_class_head(hypothesis),
                "pred_boxes": boxes,
                "pred_keypoints": self._keypoints_from_xy(keypoint_xy),
                "poseflow_full_pose_loss": torch.ones((), dtype=torch.bool, device=hypothesis.device),
                "corner_order_penalty": order_penalty.mean(),
            },
            corners,
            keypoint_xy,
        )

    def _expand_hypotheses(
        self, hypothesis: Tensor, corners: Tensor, keypoint_xy: Tensor, initial: dict[str, Tensor]
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        group_count = self.num_proposals
        scores = initial["pred_logits"].max(-1).values
        selected = torch.topk(scores, group_count, dim=1).indices
        feature_index = selected[..., None].expand(-1, -1, hypothesis.shape[-1])
        corner_index = selected[..., None].expand(-1, -1, 4)
        point_index = selected[..., None, None].expand(
            -1, -1, self.transformer.decoder.num_body_points, 2
        )
        box_features = torch.gather(hypothesis, 1, feature_index)
        selected_corners = torch.gather(corners, 1, corner_index)
        selected_keypoints = torch.gather(keypoint_xy, 1, point_index)
        selected_boxes, _ = self._corners_to_valid_boxes(selected_corners)
        point_features = box_features[:, :, None] + self.transformer.decoder.keypoint_embed.weight[
            None, None, : self.transformer.decoder.num_body_points
        ]
        point_wh = (
            self.transformer.decoder.hw.weight[: self.transformer.decoder.num_body_points]
            .sigmoid()[None, None]
            * selected_boxes[:, :, None, 2:]
        )
        point_references = torch.cat((selected_keypoints, point_wh), dim=-1)
        expanded_features = torch.cat((box_features[:, :, None], point_features), dim=2).flatten(1, 2)
        expanded_references = torch.cat(
            (selected_boxes[:, :, None], point_references), dim=2
        ).flatten(1, 2)
        return expanded_features, expanded_references, {
            "selected_person_indices": selected,
            "selected_person_corners": selected_corners,
        }

    def _predict_expanded(
        self, layer_index: int, features: Tensor, references: Tensor, *, dn_count: int
    ) -> tuple[dict[str, Tensor], Tensor, dict[str, Tensor] | None]:
        point_count = self.transformer.decoder.num_body_points
        stride = point_count + 1
        dn_features, normal_features = features[:, :dn_count], features[:, dn_count:]
        dn_references, normal_references = references[:, :dn_count], references[:, dn_count:]
        box_features = normal_features[:, ::stride]
        box_references = normal_references[:, ::stride]
        point_features = normal_features.reshape(normal_features.shape[0], -1, stride, normal_features.shape[-1])[:, :, 1:]
        point_references = normal_references.reshape(normal_references.shape[0], -1, stride, 4)[:, :, 1:]
        boxes = (
            self.v4_box_heads[layer_index](box_features) + inverse_sigmoid(box_references)
        ).sigmoid()
        keypoint_xy = (
            self.v4_pose_heads[layer_index](point_features)
            + inverse_sigmoid(point_references[..., :2])
        ).sigmoid()
        point_wh = (
            self.v4_pose_hw_heads[layer_index](point_features)
            + inverse_sigmoid(point_references[..., 2:])
        ).sigmoid()
        normal_updated_references = torch.cat(
            (
                boxes[:, :, None],
                torch.cat((keypoint_xy, point_wh), dim=-1),
            ),
            dim=2,
        ).flatten(1, 2)
        dn_prediction = None
        if dn_count:
            dn_boxes = (
                self.v4_box_heads[layer_index](dn_features) + inverse_sigmoid(dn_references)
            ).sigmoid()
            dn_prediction = {
                "pred_logits": self.v4_class_heads[layer_index](dn_features),
                "pred_boxes": dn_boxes,
            }
            updated_references = torch.cat((dn_boxes, normal_updated_references), dim=1)
        else:
            updated_references = normal_updated_references
        return {
            "pred_logits": self.v4_class_heads[layer_index](box_features),
            "pred_boxes": boxes,
            "pred_keypoints": self._keypoints_from_xy(keypoint_xy),
            "poseflow_full_pose_loss": torch.ones((), dtype=torch.bool, device=features.device),
        }, updated_references, dn_prediction

    def _refiner_attention_mask(
        self, mask: Tensor, layer_index: int, dn_count: int, dtype: torch.dtype
    ) -> Tensor:
        return mask

    def _run_joint_refiners(
        self,
        features: Tensor,
        references: Tensor,
        *,
        memory: Tensor,
        inputs,
        targets: list[dict[str, Tensor]] | None,
        dn_count: int,
    ) -> tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]:
        output = features.transpose(0, 1)
        references_t = references.transpose(0, 1)
        mask = (
            self.training_attention_mask(targets, features.shape[0], features.device)
            if dn_count
            else self.inference_attention_mask(features.shape[0], features.device)
        )
        predictions: list[dict[str, Tensor]] = []
        dn_predictions: list[dict[str, Tensor]] = []
        post_image_attention_by_layer = self.post_image_attention_by_layer()
        decoder_context = getattr(inputs, "decoder_context", None)
        decoder_valid_ratios = (
            inputs.valid_ratios
            if inputs.decoder_valid_ratios is None
            else inputs.decoder_valid_ratios
        )
        decoder_spatial_shapes = (
            inputs.spatial_shapes
            if inputs.decoder_spatial_shapes is None
            else inputs.decoder_spatial_shapes
        )
        decoder_level_start_index = (
            inputs.level_start_index
            if inputs.decoder_level_start_index is None
            else inputs.decoder_level_start_index
        )
        decoder_padding_mask = (
            inputs.padding_mask
            if inputs.decoder_padding_mask is None
            else inputs.decoder_padding_mask
        )
        for layer_index, layer in enumerate(self.transformer.decoder.layers):
            reference_input = references_t[:, :, None] * torch.cat(
                (decoder_valid_ratios, decoder_valid_ratios), dim=-1
            )[None]
            query_position = self.transformer.decoder.ref_point_head(
                sine_embedding_for_position(
                    reference_input[:, :, 0, :],
                    num_pos_feats=self.transformer.decoder.d_model // 2,
                )
            )
            output = layer(
                output,
                tgt_query_pos=query_position,
                tgt_reference_points=reference_input,
                memory=memory,
                memory_level_start_index=decoder_level_start_index,
                memory_spatial_shapes=decoder_spatial_shapes,
                memory_key_padding_mask=decoder_padding_mask,
                self_attn_mask=self._refiner_attention_mask(
                    mask, layer_index, dn_count, output.dtype
                ),
                post_image_attention=(
                    None if post_image_attention_by_layer is None
                    else post_image_attention_by_layer.get(layer_index)
                ),
                post_image_attention_context=decoder_context,
                raw_reference_points=references_t,
            )
            normal_features = self.transformer.decoder.norm(output).transpose(0, 1)
            prediction, updated_references, dn_prediction = self._predict_expanded(
                layer_index,
                normal_features,
                references_t.transpose(0, 1),
                dn_count=dn_count,
            )
            predictions.append(prediction)
            if dn_prediction is not None:
                dn_predictions.append(dn_prediction)
            references_t = updated_references.detach().transpose(0, 1)
        return predictions, dn_predictions

    def forward(
        self,
        images: Tensor,
        diffusion_keypoints: Tensor,
        timesteps: Tensor,
        targets: list[dict[str, Tensor]] | None = None,
        padding_mask: Tensor | None = None,
        *,
        person_state: Tensor,
        return_selection_metadata: bool = False,
        collect_diagnostics: bool = False,
    ) -> dict[str, Tensor | list[dict[str, Tensor]]]:
        del diffusion_keypoints
        if padding_mask is None:
            padding_mask = torch.zeros(
                images.shape[0], *images.shape[2:], dtype=torch.bool, device=images.device
            )
        self._validate_person_state(person_state, images.shape[0])
        inputs = self.prepare_transformer_inputs(images, padding_mask)
        memory = self.encode_features(inputs, timesteps)
        timestep_embedding = self.transformer.block_time_mlp(
            self.transformer.time_mlp(timesteps)
        )
        stage_state = person_state / self.scale if self.flow_state_conditioning else torch.zeros_like(person_state)
        hypothesis = self.hypothesis_stage(
            memory,
            inputs.position + self._level_position_offsets(inputs),
            inputs.padding_mask,
            stage_state,
            timesteps,
            timestep_embedding,
            collect_diagnostics=collect_diagnostics,
        )
        initial, corners, keypoint_xy = self._initial_prediction(hypothesis)
        features, references, metadata = self._expand_hypotheses(
            hypothesis, corners, keypoint_xy, initial
        )
        dn_state: dict[str, Tensor] | None = None
        dn_count = 0
        if self.training and targets is not None and self.transformer.decoder.num_dn:
            dn_features, dn_references, dn_state = self._prepare_denoising_queries(targets, images.device)
            dn_count = dn_features.shape[1]
            features = torch.cat((dn_features, features), dim=1)
            references = torch.cat((dn_references.sigmoid(), references), dim=1)
        refiners, dn_refiners = self._run_joint_refiners(
            features,
            references,
            memory=self.decoder_feature_memory(inputs, memory),
            inputs=inputs,
            targets=targets,
            dn_count=dn_count,
        )
        if collect_diagnostics:
            stage_diagnostics = self.hypothesis_stage.last_diagnostics()
            if stage_diagnostics is None:
                raise RuntimeError("PoseFlow V4 diagnostics were not collected")
            selected = metadata["selected_person_indices"]
            selected_box_index = selected[..., None].expand(-1, -1, 4)
            point_count = self.transformer.decoder.num_body_points
            selected_keypoint_index = selected[..., None].expand(-1, -1, point_count * 2)
            initial_boxes = torch.gather(initial["pred_boxes"], 1, selected_box_index)
            initial_keypoints = torch.gather(
                initial["pred_keypoints"][..., : point_count * 2],
                1,
                selected_keypoint_index,
            )
            final_prediction = refiners[-1]
            self._last_v4_diagnostics = {
                **stage_diagnostics,
                "selected_initial_score_mean": (
                    torch.gather(
                        initial["pred_logits"].sigmoid().amax(dim=-1), 1, selected
                    )
                    .detach()
                    .float()
                    .mean()
                ),
                "hypothesis_to_final_box_l1": (
                    (initial_boxes - final_prediction["pred_boxes"]).detach().float().abs().mean()
                ),
                "hypothesis_to_final_pose_l1": (
                    (
                        initial_keypoints
                        - final_prediction["pred_keypoints"][..., : point_count * 2]
                    )
                    .detach()
                    .float()
                    .abs()
                    .mean()
                ),
            }
        else:
            self._last_v4_diagnostics = None
        result: dict[str, Tensor | list[dict[str, Tensor]]] = {
            **refiners[-1],
            "aux_outputs": [initial, *refiners[:-1]],
            "corner_order_penalty": initial["corner_order_penalty"],
        }
        if return_selection_metadata:
            result.update(metadata)
        if dn_state is not None:
            result["dn_class_input"] = dn_state["labels"]
            result["dn_bbox_input"] = dn_state["boxes"]
            result["dn_class_pred"] = dn_refiners[-1]["pred_logits"]
            result["dn_bbox_pred"] = dn_refiners[-1]["pred_boxes"]
            result["dn_aux_outputs"] = [None, *dn_refiners[:-1]]
        return result

    def sample_keypoints(
        self,
        images: Tensor,
        *,
        padding_mask: Tensor | None = None,
        num_proposals: int | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, Tensor | list[dict[str, Tensor]]]:
        from .training import flow_sample_poseflow_v4_states

        return flow_sample_poseflow_v4_states(
            self,
            images,
            padding_mask=padding_mask,
            num_proposals=num_proposals,
            sampling_steps=self.flow_sampling_steps,
            ensemble=self.flow_ensemble,
            generator=generator,
        )


class PoseFlowV5Model(PoseFlowV4Model):
    """V4 hypotheses with augmented sources and explicit center supervision."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        hidden_dim = self.transformer.decoder.d_model
        state_width = 4 + 2 * self.transformer.decoder.num_body_points
        self.initial_coordinate_head = MLP(hidden_dim, hidden_dim, state_width + 2, 3)
        del self.v4_box_heads
        self.v5_box_heads = nn.ModuleList(
            MLP(hidden_dim, hidden_dim, 6, 3)
            for _ in range(self.transformer.decoder.num_layers)
        )

    def _initial_prediction(self, hypothesis: Tensor) -> tuple[dict[str, Tensor], Tensor, Tensor]:
        point_count = self.transformer.decoder.num_body_points
        coordinates = self.initial_coordinate_head(hypothesis).sigmoid()
        corners = coordinates[..., :4]
        center = coordinates[..., 4:6]
        keypoint_xy = coordinates[..., 6:].view(
            hypothesis.shape[0], self.num_queries, point_count, 2
        )
        boxes, order_penalty = self._corners_to_valid_boxes(corners)
        return (
            {
                "pred_logits": self.initial_class_head(hypothesis),
                "pred_boxes": boxes,
                "pred_keypoints": self._keypoints_from_xy(keypoint_xy),
                "pred_box_points": torch.cat((corners, center), dim=-1),
                "poseflow_full_pose_loss": torch.ones((), dtype=torch.bool, device=hypothesis.device),
                "corner_order_penalty": order_penalty.mean(),
            },
            corners,
            keypoint_xy,
        )

    def _predict_expanded(
        self, layer_index: int, features: Tensor, references: Tensor, *, dn_count: int
    ) -> tuple[dict[str, Tensor], Tensor, dict[str, Tensor] | None]:
        point_count = self.transformer.decoder.num_body_points
        stride = point_count + 1
        dn_features, normal_features = features[:, :dn_count], features[:, dn_count:]
        dn_references, normal_references = references[:, :dn_count], references[:, dn_count:]
        box_features = normal_features[:, ::stride]
        box_references = normal_references[:, ::stride]
        point_features = normal_features.reshape(
            normal_features.shape[0], -1, stride, normal_features.shape[-1]
        )[:, :, 1:]
        point_references = normal_references.reshape(
            normal_references.shape[0], -1, stride, 4
        )[:, :, 1:]
        reference_corners = cxcywh_to_corners(box_references)
        reference_points = torch.cat((reference_corners, box_references[..., :2]), dim=-1)
        box_points = (
            self.v5_box_heads[layer_index](box_features) + inverse_sigmoid(reference_points)
        ).sigmoid()
        corners, center = box_points[..., :4], box_points[..., 4:]
        boxes, _ = self._corners_to_valid_boxes(corners)
        keypoint_xy = (
            self.v4_pose_heads[layer_index](point_features)
            + inverse_sigmoid(point_references[..., :2])
        ).sigmoid()
        point_wh = (
            self.v4_pose_hw_heads[layer_index](point_features)
            + inverse_sigmoid(point_references[..., 2:])
        ).sigmoid()
        normal_updated_references = torch.cat(
            (boxes[:, :, None], torch.cat((keypoint_xy, point_wh), dim=-1)), dim=2
        ).flatten(1, 2)
        dn_prediction = None
        if dn_count:
            dn_reference_corners = cxcywh_to_corners(dn_references)
            dn_reference_points = torch.cat((dn_reference_corners, dn_references[..., :2]), dim=-1)
            dn_points = (
                self.v5_box_heads[layer_index](dn_features) + inverse_sigmoid(dn_reference_points)
            ).sigmoid()
            dn_boxes, _ = self._corners_to_valid_boxes(dn_points[..., :4])
            dn_prediction = {
                "pred_logits": self.v4_class_heads[layer_index](dn_features),
                "pred_boxes": dn_boxes,
            }
            updated_references = torch.cat((dn_boxes, normal_updated_references), dim=1)
        else:
            updated_references = normal_updated_references
        return {
            "pred_logits": self.v4_class_heads[layer_index](box_features),
            "pred_boxes": boxes,
            "pred_keypoints": self._keypoints_from_xy(keypoint_xy),
            "pred_box_points": torch.cat((corners, center), dim=-1),
            "poseflow_full_pose_loss": torch.ones((), dtype=torch.bool, device=features.device),
        }, updated_references, dn_prediction

    def sample_keypoints(
        self,
        images: Tensor,
        *,
        padding_mask: Tensor | None = None,
        num_proposals: int | None = None,
        generator: torch.Generator | None = None,
        collect_step_diagnostics: bool = False,
    ) -> dict[str, Tensor | list[dict[str, Tensor]]]:
        from .training import flow_sample_poseflow_v5_states

        return flow_sample_poseflow_v5_states(
            self,
            images,
            padding_mask=padding_mask,
            num_proposals=num_proposals,
            sampling_steps=self.flow_sampling_steps,
            ensemble=self.flow_ensemble,
            generator=generator,
            collect_step_diagnostics=collect_step_diagnostics,
        )


class PoseFlowV10Model(PoseFlowV5Model):
    """Whole-person latent refinement with one shared absolute geometry readout."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        hidden_dim = self.transformer.decoder.d_model
        self.dn_box_conditioning = MLP(4, hidden_dim, hidden_dim, 2)
        # Empty legacy containers preserve the initialization contract without parameters.
        self.v5_box_heads = nn.ModuleList()
        self.v4_class_heads = nn.ModuleList()
        self.v4_pose_heads = nn.ModuleList()
        self.v4_pose_hw_heads = nn.ModuleList()

    def _initial_prediction(self, hypothesis: Tensor):
        coordinates = self.initial_coordinate_head(hypothesis).sigmoid()
        corners, center = coordinates[..., :4], coordinates[..., 4:6]
        keypoint_xy = coordinates[..., 6:].reshape(
            *hypothesis.shape[:2], self.transformer.decoder.num_body_points, 2
        )
        boxes, order_penalty = self._corners_to_valid_boxes(corners)
        return {
            "pred_logits": self.initial_class_head(hypothesis),
            "pred_boxes": boxes,
            "pred_keypoints": self._keypoints_from_xy(keypoint_xy),
            "pred_box_points": torch.cat((corners, center), dim=-1),
            "poseflow_full_pose_loss": torch.ones((), dtype=torch.bool, device=hypothesis.device),
            "corner_order_penalty": order_penalty.mean(),
        }, corners, keypoint_xy

    def _expand_hypotheses(self, hypothesis, corners, keypoint_xy, initial):
        selected = initial["pred_logits"].max(-1).values.topk(self.num_proposals, dim=1).indices
        features = hypothesis.gather(1, selected[..., None].expand(-1, -1, hypothesis.shape[-1]))
        selected_corners = corners.gather(1, selected[..., None].expand(-1, -1, 4))
        references, _ = self._corners_to_valid_boxes(selected_corners)
        return features, references, {
            "selected_person_indices": selected,
            "selected_person_corners": selected_corners,
        }

    def _prepare_denoising_queries(self, targets, device):
        features, references, state = super()._prepare_denoising_queries(targets, device)
        features = features + self.dn_box_conditioning(references.sigmoid())
        return features, references, state

    def inference_attention_mask(self, batch_size, device):
        return torch.zeros(
            batch_size * self.transformer.decoder.num_heads,
            self.num_proposals, self.num_proposals, dtype=torch.bool, device=device,
        )

    def training_attention_mask(self, targets, batch_size, device):
        mask = self.training_box_attention_mask(targets, batch_size, device)
        count = self.transformer.decoder.num_dn + self.num_proposals
        return mask[:, :count, :count]

    def _predict_expanded(self, layer_index, features, references, *, dn_count):
        prediction, _, _ = self._initial_prediction(features)
        updated_references = prediction["pred_boxes"]
        dn_prediction = None
        if dn_count:
            dn_prediction = {
                "pred_logits": prediction["pred_logits"][:, :dn_count],
                "pred_boxes": updated_references[:, :dn_count],
            }
        clean_prediction = {
            name: value[:, dn_count:] if value.ndim > 0 else value
            for name, value in prediction.items()
        }
        return clean_prediction, updated_references, dn_prediction

    def _run_joint_refiners(
        self,
        features: Tensor,
        references: Tensor,
        *,
        memory: Tensor,
        inputs,
        targets: list[dict[str, Tensor]] | None,
        dn_count: int,
    ) -> tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]:
        output = features.transpose(0, 1)
        references_t = references.transpose(0, 1)
        mask = (
            self.training_attention_mask(targets, features.shape[0], features.device)
            if dn_count
            else self.inference_attention_mask(features.shape[0], features.device)
        )
        predictions: list[dict[str, Tensor]] = []
        dn_predictions: list[dict[str, Tensor]] = []
        post_image_attention_by_layer = self.post_image_attention_by_layer()
        decoder_context = getattr(inputs, "decoder_context", None)
        decoder_valid_ratios = (
            inputs.valid_ratios
            if inputs.decoder_valid_ratios is None
            else inputs.decoder_valid_ratios
        )
        decoder_spatial_shapes = (
            inputs.spatial_shapes
            if inputs.decoder_spatial_shapes is None
            else inputs.decoder_spatial_shapes
        )
        decoder_level_start_index = (
            inputs.level_start_index
            if inputs.decoder_level_start_index is None
            else inputs.decoder_level_start_index
        )
        decoder_padding_mask = (
            inputs.padding_mask
            if inputs.decoder_padding_mask is None
            else inputs.decoder_padding_mask
        )
        for layer_index, layer in enumerate(self.transformer.decoder.layers):
            reference_input = references_t[:, :, None] * torch.cat(
                (decoder_valid_ratios, decoder_valid_ratios), dim=-1
            )[None]
            query_position = self.transformer.decoder.ref_point_head(
                sine_embedding_for_position(
                    reference_input[:, :, 0, :],
                    num_pos_feats=self.transformer.decoder.d_model // 2,
                )
            )
            output = layer(
                output,
                tgt_query_pos=query_position,
                tgt_reference_points=reference_input,
                memory=memory,
                memory_level_start_index=decoder_level_start_index,
                memory_spatial_shapes=decoder_spatial_shapes,
                memory_key_padding_mask=decoder_padding_mask,
                self_attn_mask=self._refiner_attention_mask(
                    mask, layer_index, dn_count, output.dtype
                ),
                post_image_attention=(
                    None if post_image_attention_by_layer is None
                    else post_image_attention_by_layer.get(layer_index)
                ),
                post_image_attention_context=decoder_context,
                raw_reference_points=references_t,
            )
            normal_features = self.transformer.decoder.norm(output).transpose(0, 1)
            prediction, updated_references, dn_prediction = self._predict_expanded(
                layer_index,
                normal_features,
                references_t.transpose(0, 1),
                dn_count=dn_count,
            )
            predictions.append(prediction)
            if dn_prediction is not None:
                dn_predictions.append(dn_prediction)
            references_t = updated_references.transpose(0, 1)
        return predictions, dn_predictions


class PoseFlowV9Model(PoseFlowV5Model):
    """V5 with a learned nonneighbor bias in expanded joint attention."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        decoder = self.transformer.decoder
        if decoder.num_body_points != 14:
            raise ValueError("PoseFlow V9 requires the 14-joint CrowdPose skeleton")
        self.off_edge_attention_bias = nn.Parameter(
            torch.full((decoder.num_layers,), -5.0)
        )

        adjacency = torch.eye(decoder.num_body_points, dtype=torch.bool)
        for source, destination in CROWDPOSE_EDGE_LIST:
            adjacency[source, destination] = True
            adjacency[destination, source] = True
        group_size = decoder.num_body_points + 1
        query_count = decoder.num_group * group_size
        positions = torch.arange(query_count)
        group_ids = positions // group_size
        joint_ids = (positions % group_size - 1).clamp_min(0)
        joint_query = positions % group_size != 0
        off_edge_pairs = (
            (group_ids[:, None] == group_ids[None, :])
            & joint_query[:, None]
            & joint_query[None, :]
            & ~adjacency[joint_ids[:, None], joint_ids[None, :]]
        )
        self.register_buffer("_off_edge_pairs", off_edge_pairs, persistent=False)

    def _refiner_attention_mask(
        self, mask: Tensor, layer_index: int, dn_count: int, dtype: torch.dtype
    ) -> Tensor:
        off_edge_pairs = F.pad(
            self._off_edge_pairs, (dn_count, 0, dn_count, 0), value=False
        )
        if mask.shape[-2:] != off_edge_pairs.shape:
            raise ValueError("V9 attention mask does not match expanded query layout")
        hard_mask = torch.zeros_like(mask, dtype=dtype).masked_fill(mask, float("-inf"))
        return hard_mask + self.off_edge_attention_bias[layer_index].to(dtype) * off_edge_pairs.to(dtype)


@dataclass
class PoseFlowDINOContext:
    features: tuple[Tensor, ...]
    masks: tuple[Tensor, ...]
    valid_ratios: Tensor
    spatial_shapes: Tensor


class PoseFlowDINOv3ConvNeXtBackbone(nn.Module):
    _STAGE_CHANNELS = {1: 96, 2: 192, 3: 384, 4: 768}
    _CHANNELS = (192, 384, 768)
    _HIDDEN_STATE_INDICES = (2, 3, 4)

    def __init__(
        self,
        *,
        model_name: str,
        frozen: bool = True,
        model: nn.Module | None = None,
        hidden_state_indices: tuple[int, ...] = _HIDDEN_STATE_INDICES,
        expected_channels: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        if not hidden_state_indices or tuple(sorted(set(hidden_state_indices))) != hidden_state_indices:
            raise ValueError("DINO hidden-state indices must be non-empty, unique, and increasing")
        if expected_channels is None:
            try:
                expected_channels = tuple(self._STAGE_CHANNELS[index] for index in hidden_state_indices)
            except KeyError as exc:
                raise ValueError(f"Unsupported DINO ConvNeXt stage index: {exc.args[0]}") from exc
        if len(expected_channels) != len(hidden_state_indices):
            raise ValueError("DINO hidden-state indices and expected channels must have equal length")
        if model is None:
            from transformers import AutoModel

            model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.model = model
        self.frozen = frozen
        self.hidden_state_indices = hidden_state_indices
        self.expected_channels = expected_channels
        if frozen:
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)
            self.model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.model.eval()
        return self

    def forward(self, images: Tensor) -> tuple[Tensor, ...]:
        if self.frozen:
            with torch.no_grad():
                outputs = self.model(images)
        else:
            outputs = self.model(images)
        hidden_states = outputs.hidden_states
        if hidden_states is None or len(hidden_states) <= max(self.hidden_state_indices):
            raise RuntimeError("DINOv3 ConvNeXt did not return the expected hidden states")
        features = tuple(hidden_states[index] for index in self.hidden_state_indices)
        for feature, channels in zip(features, self.expected_channels):
            if feature.ndim != 4 or feature.shape[1] != channels:
                raise RuntimeError(
                    f"DINOv3 stage feature must have {channels} channels; got {tuple(feature.shape)}"
                )
        return features


class PoseFlowDINODecoderFusion(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        num_scales: int = 3,
        num_points: int = 4,
    ) -> None:
        super().__init__()
        if num_scales < 1:
            raise ValueError("DINO fusion requires at least one feature scale")
        self.num_scales = num_scales
        self.stage_attention = nn.ModuleList(
            MultiScaleDeformableAttention(
                d_model=hidden_dim,
                num_levels=1,
                num_heads=num_heads,
                num_points=num_points,
            )
            for _ in range(num_scales)
        )
        self.joint_projection = MLP(hidden_dim * (num_scales + 1), hidden_dim, hidden_dim, 2)
        self.residual_gate = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        query_features: Tensor,
        *,
        query_position: Tensor,
        reference_points: Tensor,
        decoder_context: PoseFlowDINOContext,
        denoising_query_count: int = 0,
        collect_diagnostics: bool = False,
        collect_attention: bool = False,
    ) -> Tensor:
        del denoising_query_count, collect_diagnostics, collect_attention
        if len(decoder_context.features) != self.num_scales or len(decoder_context.masks) != self.num_scales:
            raise RuntimeError("DINO decoder context scale count does not match its fusion module")
        query = query_features.transpose(0, 1)
        positioned_query = (query_features + query_position).transpose(0, 1)
        batch_reference_points = reference_points.transpose(0, 1)
        scale_readouts = []
        for scale_index, (attention, feature, mask) in enumerate(zip(
            self.stage_attention,
            decoder_context.features,
            decoder_context.masks,
        )):
            spatial_shapes = decoder_context.spatial_shapes[scale_index:scale_index + 1]
            level_start_index = spatial_shapes.new_zeros(1)
            valid_ratio = decoder_context.valid_ratios[:, scale_index:scale_index + 1]
            scale_reference_points = batch_reference_points[:, :, None, :] * torch.cat(
                (valid_ratio, valid_ratio), dim=-1
            )[:, None]
            readout = attention(
                positioned_query,
                scale_reference_points,
                feature.flatten(2).transpose(1, 2),
                spatial_shapes,
                level_start_index,
                mask.flatten(1),
            )
            if readout.shape[1:] != query.shape[1:]:
                raise RuntimeError("DINO scale readout does not match decoder query dimensions")
            scale_readouts.append(readout)
        joint_features = self.joint_projection(
            torch.cat((query, *scale_readouts), dim=-1)
        )
        return (query + self.residual_gate * joint_features).transpose(0, 1)


class PoseFlowV6Model(PoseFlowV5Model):
    """V5 flow model with per-layer, multi-scale DINO decoder fusion."""

    def __init__(
        self,
        *args,
        dino_model_name: str = "facebook/dinov3-convnext-small-pretrain-lvd1689m",
        dino_frozen: bool = True,
        dino_backbone: nn.Module | None = None,
        dino_hidden_state_indices: tuple[int, ...] = (2, 3, 4),
        dino_feature_channels: tuple[int, ...] = (192, 384, 768),
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        hidden_dim = self.transformer.decoder.d_model
        num_heads = self.transformer.decoder.num_heads
        self.dino_backbone = PoseFlowDINOv3ConvNeXtBackbone(
            model_name=dino_model_name,
            frozen=dino_frozen,
            model=dino_backbone,
            hidden_state_indices=dino_hidden_state_indices,
            expected_channels=dino_feature_channels,
        )
        self.dino_input_projections = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(channels, hidden_dim, kernel_size=1),
                nn.GroupNorm(32, hidden_dim),
            )
            for channels in dino_feature_channels
        )
        self.dino_decoder_fusion = nn.ModuleList(
            PoseFlowDINODecoderFusion(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_scales=len(dino_feature_channels),
            )
            for _ in range(self.transformer.decoder.num_layers)
        )

    @staticmethod
    def _valid_ratio(mask: Tensor) -> Tensor:
        valid_height = (~mask[:, :, 0]).sum(1).float() / mask.shape[1]
        valid_width = (~mask[:, 0, :]).sum(1).float() / mask.shape[2]
        return torch.stack((valid_width, valid_height), dim=-1)

    def prepare_transformer_inputs(self, images: Tensor, padding_mask: Tensor) -> TransformerInputs:
        inputs = super().prepare_transformer_inputs(images, padding_mask)
        raw_features = self.dino_backbone(images)
        projected_features = tuple(
            projection(feature)
            for projection, feature in zip(self.dino_input_projections, raw_features)
        )
        masks = tuple(
            F.interpolate(
                padding_mask[:, None].float(), size=feature.shape[-2:], mode="nearest"
            )[:, 0].to(torch.bool)
            for feature in projected_features
        )
        spatial_shapes = torch.as_tensor(
            [feature.shape[-2:] for feature in projected_features],
            dtype=torch.long,
            device=images.device,
        )
        valid_ratios = torch.stack([self._valid_ratio(mask) for mask in masks], dim=1)
        inputs.decoder_context = PoseFlowDINOContext(
            features=projected_features,
            masks=masks,
            valid_ratios=valid_ratios,
            spatial_shapes=spatial_shapes,
        )
        return inputs

    def post_image_attention_by_layer(self) -> dict[int, nn.Module]:
        return dict(enumerate(self.dino_decoder_fusion))


class PoseFlowV7Model(PoseFlowV6Model):
    """V6 flow model with Tiny DINOv3 stages 2 and 4."""

    def __init__(
        self,
        *args,
        dino_model_name: str = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
        dino_frozen: bool = True,
        dino_backbone: nn.Module | None = None,
        dino_hidden_state_indices: tuple[int, ...] = (2, 4),
        dino_feature_channels: tuple[int, ...] = (192, 768),
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            dino_model_name=dino_model_name,
            dino_frozen=dino_frozen,
            dino_backbone=dino_backbone,
            dino_hidden_state_indices=dino_hidden_state_indices,
            dino_feature_channels=dino_feature_channels,
            **kwargs,
        )


class PoseFlowV8Model(PoseFlowV5Model):
    """DINO-only feature pyramid with independent coarse and fine adapters."""

    def __init__(
        self,
        *args,
        dino_model_name: str = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
        dino_frozen: bool = True,
        dino_backbone: nn.Module | None = None,
        dino_hidden_state_indices: tuple[int, ...] = (1, 2, 3, 4),
        dino_feature_channels: tuple[int, ...] = (96, 192, 384, 768),
        dino_coarse_hidden_state_indices: tuple[int, ...] = (2, 3, 4),
        **kwargs,
    ) -> None:
        kwargs["backbone_pretrained"] = False
        kwargs.setdefault("num_encoder_feature_levels", 3)
        super().__init__(*args, **kwargs)
        if self.num_feature_levels != 4:
            raise ValueError("PoseFlow V8 requires all four native DINO feature levels")
        if dino_hidden_state_indices != (1, 2, 3, 4):
            raise ValueError("PoseFlow V8 requires all four native DINO stage outputs")
        if len(dino_feature_channels) != 4:
            raise ValueError("PoseFlow V8 requires exactly four DINO stage outputs")
        if dino_coarse_hidden_state_indices != (2, 3, 4):
            raise ValueError("PoseFlow V8 coarse features must use DINO stages 2, 3, and 4")
        if kwargs.get("num_encoder_feature_levels", 3) != 3:
            raise ValueError("PoseFlow V8 requires a three-level coarse encoder")

        self.position_embedding = PositionEmbeddingSineHW(
            self.transformer.decoder.d_model // 2,
            temperature_h=20,
            temperature_w=20,
        )
        self.dino_backbone = PoseFlowDINOv3ConvNeXtBackbone(
            model_name=dino_model_name,
            frozen=dino_frozen,
            model=dino_backbone,
            hidden_state_indices=dino_hidden_state_indices,
            expected_channels=dino_feature_channels,
        )
        hidden_dim = self.transformer.decoder.d_model
        channel_by_stage = dict(zip(dino_hidden_state_indices, dino_feature_channels, strict=True))
        self.dino_coarse_input_projections = self._build_dino_input_projections(
            tuple(channel_by_stage[index] for index in dino_coarse_hidden_state_indices),
            hidden_dim,
        )
        self.dino_fine_input_projections = self._build_dino_input_projections(
            dino_feature_channels, hidden_dim
        )

        # V8 replaces, rather than augments, the legacy ResNet feature path.
        del self.backbone
        del self.input_proj

    @staticmethod
    def _build_dino_input_projections(
        channels: tuple[int, ...], hidden_dim: int
    ) -> nn.ModuleList:
        projections = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                nn.GroupNorm(32, hidden_dim),
            )
            for in_channels in channels
        )
        for projection in projections:
            nn.init.xavier_uniform_(projection[0].weight)
            nn.init.zeros_(projection[0].bias)
        return projections

    @staticmethod
    def _flatten_feature_levels(
        features: tuple[Tensor, ...],
    ) -> Tensor:
        return torch.cat(
            [feature.flatten(2).transpose(1, 2) for feature in features], dim=1
        )

    @staticmethod
    def _feature_valid_ratio(mask: Tensor) -> Tensor:
        valid_height = (~mask[:, :, 0]).sum(1).float() / mask.shape[1]
        valid_width = (~mask[:, 0, :]).sum(1).float() / mask.shape[2]
        return torch.stack((valid_width, valid_height), dim=-1)

    def prepare_transformer_inputs(
        self, images: Tensor, padding_mask: Tensor
    ) -> TransformerInputs:
        dino_features = self.dino_backbone(images)
        feature_by_stage = dict(
            zip(self.dino_backbone.hidden_state_indices, dino_features, strict=True)
        )
        coarse_sources = tuple(feature_by_stage[index] for index in (2, 3, 4))
        coarse_features = tuple(
            projection(feature)
            for projection, feature in zip(
                self.dino_coarse_input_projections, coarse_sources, strict=True
            )
        )
        fine_features = tuple(
            projection(feature)
            for projection, feature in zip(
                self.dino_fine_input_projections, dino_features, strict=True
            )
        )
        coarse_masks = tuple(
            F.interpolate(
                padding_mask[:, None].float(),
                size=feature.shape[-2:],
                mode="nearest",
            )[:, 0].to(torch.bool)
            for feature in coarse_features
        )
        positions = tuple(
            self.position_embedding(feature, mask).to(feature.dtype)
            for feature, mask in zip(coarse_features, coarse_masks, strict=True)
        )
        coarse_spatial_shapes = torch.as_tensor(
            [feature.shape[-2:] for feature in coarse_features],
            dtype=torch.long,
            device=images.device,
        )
        coarse_padding_mask = torch.cat([mask.flatten(1) for mask in coarse_masks], dim=1)
        coarse_valid_ratios = torch.stack(
            [self._feature_valid_ratio(mask) for mask in coarse_masks], dim=1
        )
        coarse_level_start_index = torch.cat(
            (
                coarse_spatial_shapes.new_zeros(1),
                coarse_spatial_shapes.prod(1).cumsum(0)[:-1],
            )
        )
        fine_masks = tuple(
            F.interpolate(
                padding_mask[:, None].float(),
                size=feature.shape[-2:],
                mode="nearest",
            )[:, 0].to(torch.bool)
            for feature in fine_features
        )
        fine_spatial_shapes = torch.as_tensor(
            [feature.shape[-2:] for feature in fine_features],
            dtype=torch.long,
            device=images.device,
        )
        fine_padding_mask = torch.cat([mask.flatten(1) for mask in fine_masks], dim=1)
        fine_valid_ratios = torch.stack(
            [self._feature_valid_ratio(mask) for mask in fine_masks], dim=1
        )
        fine_level_start_index = torch.cat(
            (
                fine_spatial_shapes.new_zeros(1),
                fine_spatial_shapes.prod(1).cumsum(0)[:-1],
            )
        )
        return TransformerInputs(
            memory=self._flatten_feature_levels(coarse_features),
            position=self._flatten_feature_levels(positions),
            padding_mask=coarse_padding_mask,
            spatial_shapes=coarse_spatial_shapes,
            level_start_index=coarse_level_start_index,
            valid_ratios=coarse_valid_ratios,
            decoder_memory=self._flatten_feature_levels(fine_features),
            decoder_padding_mask=fine_padding_mask,
            decoder_spatial_shapes=fine_spatial_shapes,
            decoder_level_start_index=fine_level_start_index,
            decoder_valid_ratios=fine_valid_ratios,
        )

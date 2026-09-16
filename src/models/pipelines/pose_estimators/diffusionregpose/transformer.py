"""Deformable transformer components kept together for checkpoint parity."""

from __future__ import annotations

import copy
import math
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd import Function


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


_native_msda = None
_native_msda_error: Exception | None = None
_NATIVE_MSDA_MAX_IM2COL_STEP = 64


def _select_im2col_step(batch_size: int, max_step: int = _NATIVE_MSDA_MAX_IM2COL_STEP) -> int:
    """Select the largest upstream-compatible chunk that divides the batch."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_step <= 0:
        raise ValueError("max_step must be positive")
    for step in range(min(batch_size, max_step), 0, -1):
        if batch_size % step == 0:
            return step
    raise RuntimeError("every positive batch size must have a divisor")


def _load_native_msda():
    global _native_msda, _native_msda_error
    if _native_msda is not None or _native_msda_error is not None:
        return _native_msda
    try:
        from torch.utils.cpp_extension import load

        source_root = Path(__file__).with_name("third_party") / "dinov3_msda"
        _native_msda = load(
            name="pose_graph_diffusionregpose_msda_dinov3",
            sources=[
                str(source_root / "binding.cpp"),
                str(source_root / "ms_deform_attn_cuda.cu"),
            ],
            extra_include_paths=[str(source_root)],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    except Exception as error:  # pragma: no cover - depends on local CUDA toolchain
        _native_msda_error = error
        warnings.warn(
            f"native CUDA deformable attention is unavailable; using the PyTorch fallback: {error}",
            RuntimeWarning,
            stacklevel=2,
        )
    return _native_msda


class _NativeMSDAFunction(Function):
    """Autograd bridge for the vendored DINOv3 attention operator."""

    @staticmethod
    def forward(ctx, value, spatial_shapes, level_start_index, locations, weights, im2col_step):
        native_msda = _load_native_msda()
        if native_msda is None or not hasattr(native_msda, "backward"):
            raise RuntimeError("the local CUDA attention operator lacks backward support")
        ctx.im2col_step = im2col_step
        ctx.save_for_backward(value, spatial_shapes, level_start_index, locations, weights)
        return native_msda.forward(
            value, spatial_shapes, level_start_index, locations, weights, im2col_step
        )

    @staticmethod
    def backward(ctx, grad_output):
        value, spatial_shapes, level_start_index, locations, weights = ctx.saved_tensors
        native_msda = _load_native_msda()
        gradients = native_msda.backward(
            value,
            spatial_shapes,
            level_start_index,
            locations,
            weights,
            grad_output.contiguous(),
            ctx.im2col_step,
        )
        return gradients[0], None, None, gradients[1], gradients[2], None


class MultiScaleDeformableAttention(nn.Module):
    def __init__(self, d_model=256, num_levels=4, num_heads=8, num_points=4, use_4d_normalizer=False):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if not _is_power_of_two(d_model // num_heads):
            warnings.warn("attention head dimension is not a power of two", stacklevel=2)
        self.d_model, self.num_levels, self.num_heads, self.num_points = d_model, num_levels, num_heads, num_points
        self.use_4d_normalizer = use_4d_normalizer
        self.use_native_cuda_kernel = True
        self.sampling_offsets = nn.Linear(d_model, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(d_model, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        angles = torch.arange(self.num_heads, dtype=torch.float32) * (
            2 * math.pi / self.num_heads
        )
        grid = torch.stack((angles.cos(), angles.sin()), dim=-1)
        grid = (grid / grid.abs().amax(-1, keepdim=True)).view(
            self.num_heads, 1, 1, 2
        ).repeat(1, self.num_levels, self.num_points, 1)
        for point in range(self.num_points):
            grid[:, :, point] *= point + 1
        self.sampling_offsets.bias.data.copy_(grid.reshape(-1))
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        batch_size, query_length, _ = query.shape
        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], 0.0)
        value = value.view(batch_size, -1, self.num_heads, self.d_model // self.num_heads)
        offsets = self.sampling_offsets(query).view(batch_size, query_length, self.num_heads, self.num_levels, self.num_points, 2)
        weights = self.attention_weights(query).view(batch_size, query_length, self.num_heads, self.num_levels * self.num_points).softmax(-1)
        weights = weights.view(batch_size, query_length, self.num_heads, self.num_levels, self.num_points)
        if reference_points.shape[-1] == 2:
            normalizer = torch.stack((input_spatial_shapes[:, 1], input_spatial_shapes[:, 0]), -1)
            locations = reference_points[:, :, None, :, None] + offsets / normalizer[None, None, None, :, None]
        elif reference_points.shape[-1] == 4:
            locations = reference_points[:, :, None, :, None, :2] + offsets / self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError("reference_points must have last dimension 2 or 4")
        if self.use_native_cuda_kernel and query.is_cuda:
            native_msda = _load_native_msda()
            can_use_native = native_msda is not None and (
                hasattr(native_msda, "backward") or not torch.is_grad_enabled()
            )
            if can_use_native:
                native_dtype = value.dtype
                if native_dtype != torch.float32 or locations.dtype != native_dtype or weights.dtype != native_dtype:
                    native_dtype = torch.float32
                native_value = value.to(native_dtype)
                native_locations = locations.to(native_dtype)
                native_weights = weights.to(native_dtype)
                native_inputs = (
                    native_value.contiguous(),
                    input_spatial_shapes.contiguous(),
                    input_level_start_index.contiguous(),
                    native_locations.contiguous(),
                    native_weights.contiguous(),
                )
                im2col_step = _select_im2col_step(batch_size)
                if torch.is_grad_enabled():
                    native_output = _NativeMSDAFunction.apply(*native_inputs, im2col_step)
                else:
                    native_output = native_msda.forward(*native_inputs, im2col_step)
                if native_output.dtype != value.dtype:
                    native_output = native_output.to(value.dtype)
                return self.output_proj(native_output)
        head_dim = self.d_model // self.num_heads
        heights, widths = input_spatial_shapes[:, 0], input_spatial_shapes[:, 1]
        max_height, max_width = int(heights.max()), int(widths.max())
        max_area = max_height * max_width
        positions = torch.arange(max_area, device=query.device)
        rows = positions // max_width
        columns = positions % max_width
        level_indices = input_level_start_index[:, None] + rows[None, :] * widths[:, None] + columns[None, :]
        valid_positions = (rows[None, :] < heights[:, None]) & (columns[None, :] < widths[:, None])
        safe_indices = level_indices.masked_fill(~valid_positions, 0)
        level_value = value[:, safe_indices.reshape(-1)]
        level_value = level_value.view(batch_size, self.num_levels, max_area, self.num_heads, head_dim)
        level_value = level_value * valid_positions[None, :, :, None, None]
        level_value = level_value.view(batch_size, self.num_levels, max_height, max_width, self.num_heads, head_dim)
        level_value = level_value.permute(0, 1, 4, 5, 2, 3).reshape(
            batch_size * self.num_levels * self.num_heads, head_dim, max_height, max_width
        )
        level_locations = locations.permute(0, 3, 2, 1, 4, 5).reshape(
            batch_size * self.num_levels * self.num_heads, query_length * self.num_points, 1, 2
        )
        spatial_scale = torch.stack((widths, heights), -1).to(level_locations)
        spatial_scale = spatial_scale[None, :, None, :].expand(batch_size, self.num_levels, self.num_heads, 2).reshape(-1, 2)
        max_spatial = level_locations.new_tensor((max_width, max_height))
        level_locations = level_locations * spatial_scale[:, None, None] / max_spatial * 2 - 1
        sampled = F.grid_sample(
            level_value,
            level_locations,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).reshape(batch_size, self.num_levels, self.num_heads, head_dim, query_length, self.num_points)
        sampled = sampled.permute(0, 4, 2, 1, 5, 3)
        output = (sampled * weights[..., None]).sum(dim=(3, 4))
        return self.output_proj(output.reshape(batch_size, query_length, self.d_model))


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=2048, dropout=0.1, num_levels=4, num_heads=8, num_points=4):
        super().__init__()
        self.self_attn = MultiScaleDeformableAttention(d_model, num_levels, num_heads, num_points)
        self.dropout1, self.norm1 = nn.Dropout(dropout), nn.LayerNorm(d_model)
        self.linear1, self.activation = nn.Linear(d_model, d_ffn), nn.ReLU()
        self.dropout2, self.linear2 = nn.Dropout(dropout), nn.Linear(d_ffn, d_model)
        self.dropout3, self.norm2 = nn.Dropout(dropout), nn.LayerNorm(d_model)

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, key_padding_mask=None):
        attended = self.self_attn(src + pos, reference_points, src, spatial_shapes, level_start_index, key_padding_mask)
        src = self.norm1(src + self.dropout1(attended))
        return self.norm2(src + self.dropout3(self.linear2(self.dropout2(self.activation(self.linear1(src))))))


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(copy.deepcopy(encoder_layer) for _ in range(num_layers))
        self.norm = None
        self.num_layers = num_layers
        self.d_model = encoder_layer.linear2.out_features

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        points = []
        for level, (height, width) in enumerate(spatial_shapes.tolist()):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, height - 0.5, height, device=device), torch.linspace(0.5, width - 0.5, width, device=device), indexing="ij")
            points.append(torch.stack((ref_x.reshape(-1) / (valid_ratios[:, None, level, 0] * width), ref_y.reshape(-1) / (valid_ratios[:, None, level, 1] * height)), -1))
        return torch.cat(points, 1)[:, :, None] * valid_ratios[:, None]

    def forward(self, src, pos, spatial_shapes, level_start_index, valid_ratios, key_padding_mask):
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, src.device)
        output = src
        for layer in self.layers:
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, key_padding_mask)
        return output if self.norm is None else self.norm(output)


def build_input_projections(backbone_channels, *, hidden_dim=256, num_levels=4):
    if num_levels < len(backbone_channels):
        raise ValueError("num_levels cannot be smaller than the backbone feature count")
    projections = [nn.Sequential(nn.Conv2d(channels, hidden_dim, 1), nn.GroupNorm(32, hidden_dim)) for channels in backbone_channels]
    channels = backbone_channels[-1]
    for _ in range(num_levels - len(backbone_channels)):
        projections.append(nn.Sequential(nn.Conv2d(channels, hidden_dim, 3, stride=2, padding=1), nn.GroupNorm(32, hidden_dim)))
    return nn.ModuleList(projections)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        hidden = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(source, target) for source, target in zip([input_dim] + hidden, hidden + [output_dim]))

    def forward(self, inputs):
        for index, layer in enumerate(self.layers):
            inputs = F.relu(layer(inputs)) if index < len(self.layers) - 1 else layer(inputs)
        return inputs


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.dimension = dimension

    def forward(self, timesteps):
        half = self.dimension // 2
        frequencies = torch.exp(torch.arange(half, device=timesteps.device) * -(math.log(10000) / (half - 1)))
        values = timesteps[:, None] * frequencies[None]
        return torch.cat((values.sin(), values.cos()), -1)


class DiffusionRegPoseTimeConditioning(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        time_dim = d_model * 4
        self.time_mlp = nn.Sequential(SinusoidalPositionEmbeddings(d_model), nn.Linear(d_model, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim))
        self.block_time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, d_model))

    def forward(self, timesteps):
        time_embedding = self.time_mlp(timesteps)
        return time_embedding, self.block_time_mlp(time_embedding)


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=2048, dropout=0.1, num_levels=4, num_heads=8, num_points=4):
        super().__init__()
        self.cross_attn = MultiScaleDeformableAttention(d_model, num_levels, num_heads, num_points)
        self.dropout1, self.norm1 = nn.Dropout(dropout), nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.dropout2, self.norm2 = nn.Dropout(dropout), nn.LayerNorm(d_model)
        self.linear1, self.dropout3 = nn.Linear(d_model, d_ffn), nn.Dropout(dropout)
        self.linear2, self.dropout4, self.norm3 = nn.Linear(d_ffn, d_model), nn.Dropout(dropout), nn.LayerNorm(d_model)

    def forward(self, tgt, *, tgt_query_pos, tgt_reference_points, memory, memory_level_start_index, memory_spatial_shapes, memory_key_padding_mask=None, self_attn_mask=None, post_image_attention=None, denoising_query_count=0, collect_post_image_attention_diagnostics=False, collect_post_image_attention_weights=False):
        attended = self.self_attn(tgt + tgt_query_pos, tgt + tgt_query_pos, tgt, attn_mask=self_attn_mask)[0]
        tgt = self.norm2(tgt + self.dropout2(attended))
        crossed = self.cross_attn((tgt + tgt_query_pos).transpose(0, 1), tgt_reference_points.transpose(0, 1).contiguous(), memory, memory_spatial_shapes, memory_level_start_index, memory_key_padding_mask).transpose(0, 1)
        tgt = self.norm1(tgt + self.dropout1(crossed))
        if post_image_attention is not None:
            tgt = post_image_attention(
                tgt,
                denoising_query_count=denoising_query_count,
                collect_diagnostics=collect_post_image_attention_diagnostics,
                collect_attention=collect_post_image_attention_weights,
            )
        return self.norm3(tgt + self.dropout4(self.dropout3(self.linear2(F.relu(self.linear1(tgt))))))


def inverse_sigmoid(values, eps=1e-3):
    values = values.clamp(0, 1)
    return torch.log(values.clamp_min(eps) / (1 - values).clamp_min(eps))


def sine_embedding_for_position(position, *, num_pos_feats=128):
    if position.shape[-1] not in (2, 4):
        raise ValueError("position must have last dimension 2 or 4")
    if num_pos_feats <= 0 or num_pos_feats % 2:
        raise ValueError("num_pos_feats must be a positive even integer")
    frequencies = 10000 ** (
        2
        * (torch.arange(num_pos_feats, device=position.device) // 2)
        / num_pos_feats
    )
    embeddings = []
    coordinate_indices = [1, 0] if position.shape[-1] == 2 else [1, 0, 2, 3]
    for index in coordinate_indices:
        coordinate = position[..., index]
        values = coordinate[..., None] * (2 * math.pi) / frequencies
        embeddings.append(torch.stack((values[..., 0::2].sin(), values[..., 1::2].cos()), -1).flatten(-2))
    return torch.cat(embeddings, -1)


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, *, d_model=256, query_dim=4, num_feature_levels=4, num_body_points=14, num_box_decoder_layers=2, num_group=100):
        super().__init__()
        self.layers = nn.ModuleList(copy.deepcopy(decoder_layer) for _ in range(num_layers))
        self.num_layers, self.norm, self.return_intermediate = num_layers, nn.LayerNorm(d_model), True
        self.query_dim, self.num_feature_levels = query_dim, num_feature_levels
        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)
        self.query_scale = self.bbox_embed = self.class_embed = self.pose_embed = self.pose_hw_embed = None
        self.expansion_pose_embed = None
        self.num_box_decoder_layers, self.d_model = num_box_decoder_layers, d_model
        self.num_heads = decoder_layer.self_attn.num_heads
        self.deformable_decoder, self.num_group, self.num_body_points, self.num_dn = True, num_group, num_body_points, 100
        self.emb_Q, self.emb_K, self.emb_V = nn.Linear(2, d_model), nn.Linear(2, d_model), nn.Linear(2, 2)
        self.emb_SQ, self.emb_SK, self.emb_SV = nn.Linear(2, d_model), nn.Linear(2, d_model), nn.Linear(2, 2)
        self.hw, self.keypoint_embed = nn.Embedding(num_body_points, 2), nn.Embedding(num_body_points, d_model)
        self.kpt_index = [i for i in range(num_group * (num_body_points + 1)) if i % (num_body_points + 1) != 0]

    def forward(self, tgt, memory, *, refpoints_unsigmoid, level_start_index, spatial_shapes, valid_ratios, memory_key_padding_mask=None, pos=None, tgt_mask=None, tgt_mask2=None, diffusion_keypoints=None, initial_reference_points=None, state_keypoint_gate=None, return_selection_metadata=False, post_image_attention_by_layer=None, denoising_query_count=0, collect_post_image_attention_diagnostics=False, collect_post_image_attention_weights=False):
        if initial_reference_points is not None and initial_reference_points.shape != refpoints_unsigmoid.shape:
            raise ValueError("initial_reference_points must match refpoints_unsigmoid")
        output = tgt
        reference_points = (
            refpoints_unsigmoid.sigmoid()
            if initial_reference_points is None
            else initial_reference_points
        )
        intermediate, references = [], [reference_points]
        selection_metadata: dict[str, Tensor] | None = None
        for layer_id, layer in enumerate(self.layers):
            reference_points_input = reference_points[:, :, None] * torch.cat((valid_ratios, valid_ratios), -1)[None]
            query_pos = self.ref_point_head(sine_embedding_for_position(
                reference_points_input[:, :, 0, :],
                num_pos_feats=self.d_model // 2,
            ))
            layer_mask = tgt_mask if layer_id < self.num_box_decoder_layers else tgt_mask2
            post_image_attention = None
            if post_image_attention_by_layer is not None:
                post_image_attention = post_image_attention_by_layer.get(layer_id)
            output = layer(output, tgt_query_pos=query_pos, tgt_reference_points=reference_points_input, memory=memory, memory_level_start_index=level_start_index, memory_spatial_shapes=spatial_shapes, memory_key_padding_mask=memory_key_padding_mask, self_attn_mask=layer_mask, post_image_attention=post_image_attention, denoising_query_count=denoising_query_count, collect_post_image_attention_diagnostics=collect_post_image_attention_diagnostics, collect_post_image_attention_weights=collect_post_image_attention_weights)
            intermediate.append(self.norm(output))

            if self.bbox_embed is not None and layer_id < self.num_box_decoder_layers:
                reference_points = (
                    self.bbox_embed[layer_id](output) + inverse_sigmoid(reference_points)
                ).sigmoid()

            if layer_id == self.num_box_decoder_layers - 1 and diffusion_keypoints is not None:
                output, reference_points, selection_metadata = self._expand_keypoint_queries(
                    output,
                    reference_points,
                    diffusion_keypoints,
                    state_keypoint_gate=state_keypoint_gate,
                )
            elif layer_id >= self.num_box_decoder_layers and self.bbox_embed is not None:
                output, reference_points = self._update_expanded_queries(
                    layer_id, output, reference_points
                )
            references.append(reference_points)
            reference_points = reference_points.detach()

        decoder_outputs = [item.transpose(0, 1) for item in intermediate]
        decoder_references = [item.transpose(0, 1) for item in references]
        if return_selection_metadata:
            if selection_metadata is None:
                raise RuntimeError("decoder did not expand person queries")
            return decoder_outputs, decoder_references, selection_metadata
        return decoder_outputs, decoder_references

    def _expand_keypoint_queries(
        self,
        output: Tensor,
        reference_points: Tensor,
        diffusion_keypoints: Tensor,
        *,
        state_keypoint_gate: Tensor | None,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        if self.class_embed is None or self.pose_embed is None:
            raise RuntimeError("prediction heads must be attached before query expansion")
        batch_size = output.shape[1]
        dn_count = self.num_dn if output.shape[0] > self.num_queries else 0
        dn_output, dn_references = output[:dn_count], reference_points[:dn_count]
        normal_output = output[dn_count:]
        normal_references = reference_points[dn_count:]
        group_count = min(self.num_group, diffusion_keypoints.shape[1], normal_output.shape[0])
        class_scores = self.class_embed[self.num_box_decoder_layers - 1](normal_output).max(-1).values
        selected = torch.topk(class_scores, group_count, dim=0).indices
        gather_features = selected[..., None].expand(group_count, batch_size, self.d_model)
        gather_references = selected[..., None].expand(group_count, batch_size, 4)
        box_features = torch.gather(normal_output, 0, gather_features)
        box_references = torch.gather(normal_references, 0, gather_references)

        point_features = box_features[:, None] + self.keypoint_embed.weight[: self.num_body_points, None, :]
        expansion_pose_embed = self.expansion_pose_embed or self.pose_embed[0]
        point_delta = expansion_pose_embed(point_features)[..., :2]
        point_xy = (inverse_sigmoid(box_references[:, None, :, :2]) + point_delta).sigmoid()
        point_wh = self.hw.weight[: self.num_body_points, None, :].sigmoid() * box_references[:, None, :, 2:]
        point_references = torch.cat((point_xy, point_wh), dim=-1)

        if state_keypoint_gate is None:
            diffusion_xy = diffusion_keypoints[:, :group_count].reshape(batch_size, group_count, self.num_body_points, 2)
            diffusion_xy = diffusion_xy.permute(1, 0, 2, 3).reshape(group_count, batch_size, self.num_body_points, 2)
            point_reference_xy = self._refine_diffusion_keypoints(
                diffusion_xy, point_references[..., :2]
            )
        else:
            if state_keypoint_gate.shape != (batch_size,):
                raise ValueError("state_keypoint_gate must contain one value per batch item")
            state_indices = selected.transpose(0, 1)
            state_index = state_indices[..., None].expand(
                batch_size, group_count, diffusion_keypoints.shape[-1]
            )
            selected_state = torch.gather(diffusion_keypoints, 1, state_index)
            selected_xy = selected_state.reshape(
                batch_size, group_count, self.num_body_points, 2
            ).permute(1, 2, 0, 3)
            gate = state_keypoint_gate[None, None, :, None]
            point_reference_xy = (
                (1.0 - gate) * point_references[..., :2] + gate * selected_xy
            )
        point_references = torch.cat((point_reference_xy, point_references[..., 2:]), dim=-1)

        expanded_output = torch.cat((box_features[:, None], point_features), dim=1)
        expanded_references = torch.cat((box_references[:, None], point_references), dim=1)
        expanded_output = expanded_output.flatten(0, 1)
        expanded_references = expanded_references.flatten(0, 1)
        metadata = {
            "selected_person_indices": selected.transpose(0, 1),
            "selected_person_features": box_features.transpose(0, 1),
            "selected_person_references": box_references.transpose(0, 1),
        }
        if dn_count:
            return (
                torch.cat((dn_output, expanded_output)),
                torch.cat((dn_references, expanded_references)),
                metadata,
            )
        return expanded_output, expanded_references, metadata

    def _refine_diffusion_keypoints(self, diffusion_xy: Tensor, reference_xy: Tensor) -> Tensor:
        group_count, batch_size, point_count, _ = diffusion_xy.shape
        diffusion_xy = diffusion_xy.reshape(batch_size, group_count * point_count, 2)
        self_query = self.emb_SQ(diffusion_xy)
        self_key = self.emb_SK(diffusion_xy)
        self_value = self.emb_SV(diffusion_xy)
        self_attention = (self_query * (self_query.shape[1] ** 0.5)) @ self_key.transpose(-2, -1)
        diffusion_xy = self_attention.softmax(dim=-1) @ self_value

        reference_xy = reference_xy.permute(2, 0, 1, 3).reshape(batch_size, group_count * point_count, 2)
        query = self.emb_Q(diffusion_xy + reference_xy)
        key = self.emb_K(reference_xy)
        value = self.emb_V(reference_xy)
        attention = (query * (query.shape[1] ** 0.5)) @ key.transpose(-2, -1)
        refined = attention.softmax(dim=-1) @ value
        return refined.view(batch_size, group_count, point_count, 2).permute(1, 2, 0, 3)

    def _update_expanded_queries(
        self, layer_id: int, output: Tensor, reference_points: Tensor
    ) -> tuple[Tensor, Tensor]:
        stride = self.num_body_points + 1
        dn_count = self.num_dn if self.training else 0
        normal_start = dn_count
        normal_length = output.shape[0] - normal_start
        box_indices = normal_start + torch.arange(0, normal_length, stride, device=output.device)
        point_indices = normal_start + torch.arange(normal_length, device=output.device)
        point_indices = point_indices[(point_indices - normal_start).remainder(stride) != 0]
        box_delta = self.bbox_embed[layer_id](output.index_select(0, box_indices))
        point_output = output.index_select(0, point_indices)
        point_delta = self.pose_embed[layer_id - self.num_box_decoder_layers](point_output)
        point_hw_delta = self.pose_hw_embed[layer_id - self.num_box_decoder_layers](point_output)
        updated = reference_points.clone()
        updated[box_indices] = (
            box_delta + inverse_sigmoid(reference_points.index_select(0, box_indices))
        ).sigmoid()
        point_reference = reference_points.index_select(0, point_indices)
        updated[point_indices] = torch.cat(
            (
                point_delta[..., :2] + inverse_sigmoid(point_reference[..., :2]),
                point_hw_delta + inverse_sigmoid(point_reference[..., 2:]),
            ),
            dim=-1,
        ).sigmoid()
        if dn_count:
            updated[:dn_count] = (
                self.bbox_embed[layer_id](output[:dn_count])
                + inverse_sigmoid(reference_points[:dn_count])
            ).sigmoid()
        return output, updated


class DiffusionRegPosePredictionHeads(nn.Module):
    def __init__(self, *, hidden_dim, num_classes, num_decoder_layers, num_box_decoder_layers, num_body_points, share_class=False, share_boxes=False, share_pose=False, cls_no_bias=False, separate_expansion_pose_head=False):
        super().__init__()
        class_head = nn.Linear(hidden_dim, num_classes, bias=not cls_no_bias)
        if not cls_no_bias:
            nn.init.constant_(class_head.bias, -math.log((1 - 0.01) / 0.01))
        box_head, pose_head, pose_hw_head = MLP(hidden_dim, hidden_dim, 4, 3), MLP(hidden_dim, hidden_dim, 2, 3), MLP(hidden_dim, hidden_dim, 2, 3)
        for head in (box_head, pose_head):
            nn.init.constant_(head.layers[-1].weight, 0.0)
            nn.init.constant_(head.layers[-1].bias, 0.0)
        self.bbox_embed = nn.ModuleList([box_head if share_boxes else copy.deepcopy(box_head) for _ in range(num_decoder_layers)])
        self.class_embed = nn.ModuleList([class_head if share_class else copy.deepcopy(class_head) for _ in range(num_decoder_layers)])
        pose_count = num_decoder_layers - num_box_decoder_layers + (1 if num_body_points == 17 else 0)
        self.pose_embed = nn.ModuleList([pose_head if share_pose else copy.deepcopy(pose_head) for _ in range(pose_count)])
        self.pose_hw_embed = nn.ModuleList(pose_hw_head for _ in range(pose_count))
        self.expansion_pose_embed = copy.deepcopy(pose_head) if separate_expansion_pose_head else None

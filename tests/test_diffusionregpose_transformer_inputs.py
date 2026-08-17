from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from src.models.pipelines.pose_estimators.diffusionregpose.transformer import (
    MultiScaleDeformableAttention,
)
from src.models.pipelines.pose_estimators.diffusionregpose.transformer import (
    DeformableTransformerEncoder,
    DeformableTransformerEncoderLayer,
    build_input_projections,
)
from src.models.pipelines.pose_estimators.diffusionregpose.backbone import (
    PositionEmbeddingSineHW,
)


class LoopFreeMultiScaleDeformableAttention(MultiScaleDeformableAttention):
    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        input_flatten: torch.Tensor,
        input_spatial_shapes: torch.Tensor,
        input_level_start_index: torch.Tensor,
        input_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, query_length, _ = query.shape
        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], 0.0)
        head_dim = self.d_model // self.num_heads
        value = value.view(batch_size, -1, self.num_heads, head_dim)
        offsets = self.sampling_offsets(query).view(
            batch_size, query_length, self.num_heads, self.num_levels, self.num_points, 2
        )
        weights = self.attention_weights(query).view(
            batch_size, query_length, self.num_heads, self.num_levels * self.num_points
        ).softmax(-1).view(batch_size, query_length, self.num_heads, self.num_levels, self.num_points)
        if reference_points.shape[-1] == 2:
            normalizer = torch.stack((input_spatial_shapes[:, 1], input_spatial_shapes[:, 0]), dim=-1)
            locations = reference_points[:, :, None, :, None, :] + offsets / normalizer[None, None, None, :, None]
        elif reference_points.shape[-1] == 4:
            locations = reference_points[:, :, None, :, None, :2] + offsets / self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError("reference_points must have last dimension 2 or 4")

        heights = input_spatial_shapes[:, 0]
        widths = input_spatial_shapes[:, 1]
        max_height = int(heights.max())
        max_width = int(widths.max())
        max_area = max_height * max_width
        positions = torch.arange(max_area, device=query.device)
        rows = positions // max_width
        columns = positions % max_width
        level_indices = input_level_start_index[:, None] + rows[None, :] * widths[:, None] + columns[None, :]
        valid_positions = (rows[None, :] < heights[:, None]) & (columns[None, :] < widths[:, None])
        safe_indices = level_indices.masked_fill(~valid_positions, 0)
        level_value = value[:, safe_indices.reshape(-1)]
        level_value = level_value.view(
            batch_size, self.num_levels, max_area, self.num_heads, head_dim
        )
        level_value = level_value * valid_positions[None, :, :, None, None]
        level_value = level_value.view(
            batch_size,
            self.num_levels,
            max_height,
            max_width,
            self.num_heads,
            head_dim,
        ).permute(0, 1, 4, 5, 2, 3)
        level_value = level_value.reshape(
            batch_size * self.num_levels * self.num_heads,
            head_dim,
            max_height,
            max_width,
        )

        level_locations = locations.permute(0, 3, 2, 1, 4, 5).reshape(
            batch_size * self.num_levels * self.num_heads,
            query_length * self.num_points,
            1,
            2,
        )
        spatial_scale = torch.stack((widths, heights), dim=-1).to(level_locations)
        spatial_scale = spatial_scale[None, :, None, :].expand(
            batch_size, self.num_levels, self.num_heads, 2
        ).reshape(-1, 2)
        max_spatial = level_locations.new_tensor((max_width, max_height))
        level_locations = (
            level_locations * spatial_scale[:, None, None]
            / max_spatial
            * 2.0
            - 1.0
        )
        sampled = F.grid_sample(
            level_value,
            level_locations,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).reshape(
            batch_size,
            self.num_levels,
            self.num_heads,
            head_dim,
            query_length,
            self.num_points,
        )
        sampled = sampled.permute(0, 4, 2, 1, 5, 3)
        output = (sampled * weights[..., None]).sum(dim=(3, 4))
        return self.output_proj(output.reshape(batch_size, query_length, self.d_model))


def test_input_projections_create_reference_feature_levels() -> None:
    projections = build_input_projections((512, 1024, 2048), hidden_dim=256, num_levels=4)
    outputs = [projection(torch.rand(2, channels, size, size)) for projection, channels, size in zip(
        projections,
        (512, 1024, 2048, 2048),
        (16, 8, 4, 2),
    )]

    assert [output.shape for output in outputs] == [
        (2, 256, 16, 16),
        (2, 256, 8, 8),
        (2, 256, 4, 4),
        (2, 256, 1, 1),
    ]


def test_sine_position_embedding_respects_padding_mask() -> None:
    position_embedding = PositionEmbeddingSineHW(
        num_pos_feats=128,
        temperature_h=20,
        temperature_w=20,
        normalize=True,
    )
    image = torch.zeros(1, 256, 4, 5)
    mask = torch.zeros(1, 4, 5, dtype=torch.bool)
    mask[:, -1, -1] = True

    output = position_embedding(image, mask)

    assert output.shape == (1, 256, 4, 5)
    assert torch.isfinite(output).all()


def test_deformable_attention_has_reference_parameter_shapes() -> None:
    attention = MultiScaleDeformableAttention(
        d_model=256,
        num_levels=4,
        num_heads=8,
        num_points=4,
    )

    assert attention.sampling_offsets.weight.shape == (256, 256)
    assert attention.sampling_offsets.bias.shape == (256,)
    assert attention.attention_weights.weight.shape == (128, 256)
    assert attention.value_proj.weight.shape == (256, 256)
    assert attention.output_proj.weight.shape == (256, 256)


@pytest.mark.parametrize("reference_dimension", [2, 4])
@pytest.mark.parametrize("use_padding_mask", [False, True])
def test_loop_free_deformable_attention_matches_reference(
    reference_dimension: int,
    use_padding_mask: bool,
) -> None:
    torch.manual_seed(7)
    attention = MultiScaleDeformableAttention(
        d_model=16,
        num_levels=3,
        num_heads=4,
        num_points=2,
    )
    optimized_attention = LoopFreeMultiScaleDeformableAttention(
        d_model=16,
        num_levels=3,
        num_heads=4,
        num_points=2,
    )
    optimized_attention.load_state_dict(attention.state_dict())

    batch_size = 2
    query_length = 5
    spatial_shapes = torch.tensor([[2, 3], [1, 4], [3, 2]])
    level_lengths = spatial_shapes.prod(dim=1)
    level_start_index = torch.cat((torch.zeros(1, dtype=torch.long), level_lengths.cumsum(0)[:-1]))
    query = torch.randn(batch_size, query_length, 16)
    input_flatten = torch.randn(batch_size, int(level_lengths.sum()), 16)
    reference_points = torch.rand(batch_size, query_length, 3, reference_dimension)
    if reference_dimension == 4:
        reference_points[..., 2:] = reference_points[..., 2:] * 0.75 + 0.25
    padding_mask = None
    if use_padding_mask:
        padding_mask = torch.zeros(input_flatten.shape[:2], dtype=torch.bool)
        padding_mask[:, 2] = True
        padding_mask[:, -1] = True

    expected = attention(
        query,
        reference_points,
        input_flatten,
        spatial_shapes,
        level_start_index,
        padding_mask,
    )
    actual = optimized_attention(
        query,
        reference_points,
        input_flatten,
        spatial_shapes,
        level_start_index,
        padding_mask,
    )

    assert torch.equal(actual, expected)


def test_encoder_parameter_shapes_match_available_checkpoint() -> None:
    checkpoint_path = Path(
        "external/DiffusionRegPose/logs/"
        "crowdpose_r50_effective_batch8_40epoch/checkpoint.pth"
    )
    if not checkpoint_path.exists():
        pytest.skip("trained DiffusionRegPose checkpoint is not available")

    encoder = DeformableTransformerEncoder(
        DeformableTransformerEncoderLayer(
            d_model=256,
            d_ffn=2048,
            dropout=0.0,
            num_levels=4,
            num_heads=8,
            num_points=4,
        ),
        num_layers=6,
    )
    checkpoint = torch.load(checkpoint_path, map_location="meta", weights_only=False)
    checkpoint_state = {
        key.removeprefix("transformer.encoder."): value
        for key, value in checkpoint["model"].items()
        if key.startswith("transformer.encoder.")
    }
    native_state = encoder.state_dict()

    assert set(checkpoint_state) == set(native_state)
    for key, value in checkpoint_state.items():
        assert tuple(value.shape) == tuple(native_state[key].shape), key

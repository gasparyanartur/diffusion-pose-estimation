from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.models.pipelines.pose_estimators.diffusionregpose.transformer import (
    DeformableTransformerDecoder,
    DeformableTransformerDecoderLayer,
    sine_embedding_for_position,
)


def _make_decoder() -> DeformableTransformerDecoder:
    layer = DeformableTransformerDecoderLayer(
        d_model=256,
        d_ffn=2048,
        dropout=0.0,
        num_levels=4,
        num_heads=8,
        num_points=4,
    )
    return DeformableTransformerDecoder(
        layer,
        num_layers=6,
        d_model=256,
        query_dim=4,
        num_feature_levels=4,
        num_body_points=14,
        num_group=100,
        num_box_decoder_layers=2,
    )


def test_decoder_parameter_shapes_match_available_checkpoint() -> None:
    checkpoint_path = Path(
        "external/DiffusionRegPose/logs/"
        "crowdpose_r50_effective_batch8_40epoch/checkpoint.pth"
    )
    if not checkpoint_path.exists():
        pytest.skip("trained DiffusionRegPose checkpoint is not available")

    checkpoint = torch.load(checkpoint_path, map_location="meta", weights_only=False)
    checkpoint_state = {
        key.removeprefix("transformer.decoder."): value
        for key, value in checkpoint["model"].items()
        if key.startswith("transformer.decoder.")
        and not key.removeprefix("transformer.decoder.").startswith(
            ("bbox_embed.", "class_embed.", "pose_embed.", "pose_hw_embed.")
        )
    }
    native_state = _make_decoder().state_dict()

    assert set(checkpoint_state) == set(native_state)
    for key, value in checkpoint_state.items():
        assert tuple(value.shape) == tuple(native_state[key].shape), key


def test_decoder_layer_forward_preserves_query_shape() -> None:
    layer = DeformableTransformerDecoderLayer(
        d_model=32,
        d_ffn=64,
        dropout=0.0,
        num_levels=2,
        num_heads=4,
        num_points=2,
    )
    query = torch.randn(5, 2, 32)
    memory = torch.randn(2, 20, 32)
    reference_points = torch.rand(5, 2, 2, 4)
    spatial_shapes = torch.tensor([[2, 4], [2, 6]])
    level_start_index = torch.tensor([0, 8])
    output = layer(
        query,
        tgt_query_pos=torch.randn_like(query),
        tgt_reference_points=reference_points,
        memory=memory,
        memory_level_start_index=level_start_index,
        memory_spatial_shapes=spatial_shapes,
    )

    assert output.shape == query.shape
    assert torch.isfinite(output).all()


def test_decoder_sine_embedding_uses_reference_coordinate_order() -> None:
    position = torch.tensor([[[0.25, 0.5, 0.75, 0.125]]])
    embedding = sine_embedding_for_position(position)
    expected_first_y = torch.sin(torch.tensor(0.5 * 2 * torch.pi))
    expected_first_x = torch.sin(torch.tensor(0.25 * 2 * torch.pi))

    assert torch.allclose(embedding[0, 0, 0], expected_first_y)
    assert torch.allclose(embedding[0, 0, 128], expected_first_x)

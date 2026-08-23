from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from src.models.pipelines.pose_estimators.diffusionregpose.transformer import (
    DiffusionRegPoseTimeConditioning,
)
from src.models.pipelines.pose_estimators.diffusionregpose.transformer import (
    DiffusionRegPosePredictionHeads,
)


def test_time_conditioning_has_reference_parameter_shapes() -> None:
    conditioning = DiffusionRegPoseTimeConditioning(d_model=256)
    state = conditioning.state_dict()

    assert state["time_mlp.1.weight"].shape == (1024, 256)
    assert state["time_mlp.3.weight"].shape == (1024, 1024)
    assert state["block_time_mlp.1.weight"].shape == (256, 1024)


def test_prediction_heads_have_reference_layer_counts() -> None:
    heads = DiffusionRegPosePredictionHeads(
        hidden_dim=256,
        num_classes=2,
        num_decoder_layers=6,
        num_box_decoder_layers=2,
        num_body_points=14,
    )

    assert len(heads.bbox_embed) == 6
    assert len(heads.class_embed) == 6
    assert len(heads.pose_embed) == 4
    assert len(heads.pose_hw_embed) == 4
    assert heads.class_embed[0].out_features == 2
    assert heads.pose_embed[0].layers[-1].out_features == 2


def test_pose_hw_heads_share_one_parameterized_module() -> None:
    heads = DiffusionRegPosePredictionHeads(
        hidden_dim=32,
        num_classes=2,
        num_decoder_layers=6,
        num_box_decoder_layers=2,
        num_body_points=14,
    )

    assert heads.pose_hw_embed[0] is heads.pose_hw_embed[1]
    assert heads.pose_hw_embed[0].layers[0].weight is heads.pose_hw_embed[-1].layers[0].weight


def test_prediction_head_shapes_match_available_checkpoint() -> None:
    checkpoint_path = Path(
        "external/DiffusionRegPose/logs/"
        "crowdpose_r50_effective_batch8_40epoch/checkpoint.pth"
    )
    if not checkpoint_path.exists():
        pytest.skip("trained DiffusionRegPose checkpoint is not available")

    checkpoint = torch.load(checkpoint_path, map_location="meta", weights_only=False)
    checkpoint_state = {
        key: value
        for key, value in checkpoint["model"].items()
        if key.startswith(("bbox_embed.", "class_embed.", "pose_embed.", "pose_hw_embed."))
    }
    native_state = nn.ModuleDict({
        "bbox_embed": DiffusionRegPosePredictionHeads(
            hidden_dim=256,
            num_classes=2,
            num_decoder_layers=6,
            num_box_decoder_layers=2,
            num_body_points=14,
        ).bbox_embed,
    }).state_dict()
    heads = DiffusionRegPosePredictionHeads(
        hidden_dim=256,
        num_classes=2,
        num_decoder_layers=6,
        num_box_decoder_layers=2,
        num_body_points=14,
    )
    native_state = heads.state_dict()

    assert set(checkpoint_state) == set(native_state)
    for key, value in checkpoint_state.items():
        assert tuple(value.shape) == tuple(native_state[key].shape), key


def test_time_conditioning_forward_returns_expected_shapes() -> None:
    conditioning = DiffusionRegPoseTimeConditioning(d_model=32)
    time_embedding, block_embedding = conditioning(torch.tensor([0, 3], dtype=torch.long))

    assert time_embedding.shape == (2, 128)
    assert block_embedding.shape == (2, 32)
    assert torch.isfinite(time_embedding).all()
    assert torch.isfinite(block_embedding).all()

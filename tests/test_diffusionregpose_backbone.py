from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.models.pipelines.pose_estimators.diffusionregpose.backbone import (
    DiffusionRegPoseResNetBackbone,
    FrozenBatchNorm2d,
)


def test_frozen_batch_norm_uses_stored_statistics() -> None:
    norm = FrozenBatchNorm2d(2)
    norm.weight.copy_(torch.tensor([2.0, 3.0]))
    norm.bias.copy_(torch.tensor([0.5, -1.0]))
    norm.running_mean.copy_(torch.tensor([1.0, 2.0]))
    norm.running_var.copy_(torch.tensor([4.0, 9.0]))

    output = norm(torch.tensor([[[[3.0]], [[5.0]]]]))

    expected = torch.tensor([[[[2.5]], [[2.0]]]])
    torch.testing.assert_close(output, expected)


def test_resnet_backbone_returns_three_ordered_feature_levels() -> None:
    backbone = DiffusionRegPoseResNetBackbone(
        pretrained=False,
        return_interm_indices=(1, 2, 3),
    )

    features = backbone(torch.rand(2, 3, 128, 128))

    assert [feature.shape[:2] for feature in features] == [
        (2, 512),
        (2, 1024),
        (2, 2048),
    ]
    assert [feature.shape[-2:] for feature in features] == [(16, 16), (8, 8), (4, 4)]


def test_resnet_backbone_state_keys_match_external_backbone_body() -> None:
    backbone = DiffusionRegPoseResNetBackbone(pretrained=False)
    keys = set(backbone.state_dict())

    assert "body.conv1.weight" in keys
    assert "body.layer1.0.conv1.weight" in keys
    assert "body.layer4.2.conv3.weight" in keys
    assert "body.layer1.0.bn1.running_mean" in keys


def test_backbone_key_shapes_match_available_checkpoint() -> None:
    checkpoint_path = Path(
        "external/DiffusionRegPose/logs/"
        "crowdpose_r50_effective_batch8_40epoch/checkpoint.pth"
    )
    if not checkpoint_path.exists():
        pytest.skip("trained DiffusionRegPose checkpoint is not available")

    checkpoint = torch.load(checkpoint_path, map_location="meta", weights_only=False)
    checkpoint_state = {
        key.removeprefix("backbone.0."): value
        for key, value in checkpoint["model"].items()
        if key.startswith("backbone.0.")
    }
    native_state = DiffusionRegPoseResNetBackbone(pretrained=False).state_dict()

    assert set(checkpoint_state) == set(native_state)
    for key, value in checkpoint_state.items():
        assert tuple(value.shape) == tuple(native_state[key].shape), key

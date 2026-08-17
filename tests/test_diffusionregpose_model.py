from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.models.pipelines.pose_estimators.diffusionregpose.model import (
    DiffusionRegPoseModel,
)


def _make_model() -> DiffusionRegPoseModel:
    return DiffusionRegPoseModel(
        backbone_pretrained=False,
        timesteps=1000,
        sampling_timesteps=1,
        num_queries=900,
        num_proposals=100,
        num_classes=2,
        num_feature_levels=4,
        num_encoder_layers=6,
        num_decoder_layers=6,
        hidden_dim=256,
        num_heads=8,
        num_body_points=14,
        num_box_decoder_layers=2,
    )


def test_model_state_shapes_match_available_checkpoint() -> None:
    checkpoint_path = Path(
        "external/DiffusionRegPose/logs/"
        "crowdpose_r50_effective_batch8_40epoch/checkpoint.pth"
    )
    if not checkpoint_path.exists():
        pytest.skip("trained DiffusionRegPose checkpoint is not available")

    checkpoint = torch.load(checkpoint_path, map_location="meta", weights_only=False)
    checkpoint_state = checkpoint["model"]
    native_state = _make_model().state_dict()

    assert set(checkpoint_state) == set(native_state)
    for key, value in checkpoint_state.items():
        assert tuple(value.shape) == tuple(native_state[key].shape), key


def test_available_checkpoint_loads_strictly_and_runs_expanded_forward() -> None:
    checkpoint_path = Path(
        "external/DiffusionRegPose/logs/"
        "crowdpose_r50_effective_batch8_40epoch/checkpoint.pth"
    )
    if not checkpoint_path.exists():
        pytest.skip("trained DiffusionRegPose checkpoint is not available")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = DiffusionRegPoseModel(
        backbone_pretrained=False,
        sampling_timesteps=1,
        num_queries=32,
        num_proposals=8,
        num_group=8,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    with torch.no_grad():
        output = model(
            torch.rand(1, 3, 128, 128),
            torch.rand(1, 8, 14 * 2),
            torch.zeros(1, dtype=torch.long),
        )

    assert torch.isfinite(output["pred_logits"]).all()
    assert torch.isfinite(output["pred_boxes"]).all()
    assert torch.isfinite(output["pred_keypoints"]).all()


def test_model_feature_boundary_returns_flattened_multi_scale_inputs() -> None:
    model = _make_model()
    images = torch.rand(1, 3, 128, 128)
    padding_mask = torch.zeros(1, 128, 128, dtype=torch.bool)

    flattened = model.prepare_transformer_inputs(images, padding_mask)

    assert flattened.memory.shape == (1, 340, 256)
    assert flattened.position.shape == (1, 340, 256)
    assert flattened.padding_mask.shape == (1, 340)
    assert flattened.spatial_shapes.tolist() == [[16, 16], [8, 8], [4, 4], [2, 2]]
    assert torch.isfinite(flattened.memory).all()


def test_two_stage_selection_returns_encoder_queries_and_boxes() -> None:
    model = DiffusionRegPoseModel(
        backbone_pretrained=False,
        sampling_timesteps=1,
        num_queries=32,
        num_proposals=100,
    )
    images = torch.rand(1, 3, 128, 128)
    inputs = model.prepare_transformer_inputs(images, torch.zeros(1, 128, 128, dtype=torch.bool))
    memory = model.encode_features(inputs)

    query_features, reference_boxes, class_logits, initial_boxes = model.select_two_stage_queries(
        memory, inputs
    )

    assert query_features.shape == (1, 32, 256)
    assert reference_boxes.shape == (1, 32, 4)
    assert class_logits.shape == (1, 32, 2)
    assert initial_boxes.shape == (1, 32, 4)
    assert torch.isfinite(query_features).all()


def test_forward_decodes_iterative_boxes_and_auxiliary_outputs() -> None:
    model = DiffusionRegPoseModel(
        backbone_pretrained=False,
        sampling_timesteps=1,
        num_queries=32,
        num_proposals=8,
        num_group=8,
    )
    images = torch.rand(1, 3, 128, 128)
    diffusion_keypoints = torch.rand(1, 8, 14 * 2)
    timesteps = torch.zeros(1, dtype=torch.long)

    output = model(images, diffusion_keypoints, timesteps)

    assert output["pred_boxes"].shape == (1, 8, 4)
    assert output["pred_logits"].shape == (1, 8, 2)
    assert output["pred_keypoints"].shape == (1, 8, 14 * 3)
    assert len(output["aux_outputs"]) == 6
    assert all(output["aux_outputs"][index]["pred_boxes"].shape == (1, 32, 4) for index in (0, 1))
    assert all(aux["pred_boxes"].shape == (1, 8, 4) for aux in output["aux_outputs"][2:])
    assert torch.isfinite(output["pred_boxes"]).all()


def test_forward_uses_timestep_conditioning() -> None:
    model = DiffusionRegPoseModel(
        backbone_pretrained=False,
        sampling_timesteps=1,
        num_queries=8,
        num_proposals=8,
    )
    images = torch.rand(1, 3, 128, 128)
    inputs = model.prepare_transformer_inputs(images, torch.zeros(1, 128, 128, dtype=torch.bool))

    early_t, early_block = model.transformer.time_mlp(torch.tensor([0])), model.transformer.block_time_mlp(
        model.transformer.time_mlp(torch.tensor([0]))
    )
    late_t, late_block = model.transformer.time_mlp(torch.tensor([999])), model.transformer.block_time_mlp(
        model.transformer.time_mlp(torch.tensor([999]))
    )
    early = model.encode_features(inputs, torch.tensor([0]))
    late = model.encode_features(inputs, torch.tensor([999]))

    assert early.shape == late.shape == (1, 340, 256)
    assert not torch.allclose(early_t, late_t)
    assert not torch.allclose(early_block, late_block)


def test_sample_keypoints_runs_reference_timestep_schedule_deterministically() -> None:
    model = DiffusionRegPoseModel(
        backbone_pretrained=False,
        timesteps=8,
        sampling_timesteps=2,
        num_queries=8,
        num_proposals=2,
        num_group=2,
    )
    model.eval()
    images = torch.rand(1, 3, 64, 64)

    first = model.sample_keypoints(images, generator=torch.Generator().manual_seed(7))
    second = model.sample_keypoints(images, generator=torch.Generator().manual_seed(7))

    assert first["pred_boxes"].shape == (1, 2, 4)
    assert first["pred_keypoints"].shape == (1, 2, 14 * 3)
    assert first["diffusion_keypoints"].shape == (1, 2, 14 * 2)
    torch.testing.assert_close(first["pred_boxes"], second["pred_boxes"])
    torch.testing.assert_close(first["diffusion_keypoints"], second["diffusion_keypoints"])

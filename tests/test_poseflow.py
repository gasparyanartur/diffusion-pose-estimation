from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.models.pipelines.pose_estimators.diffusionregpose.training import (
    DiffusionRegPoseOTAMatcher,
)
from src.models.pipelines.pose_estimators.poseflow import (
    PoseFlowModel,
    PoseFlowV3Model,
    PoseFlowV4Model,
    PoseFlowV5Model,
    PoseFlowPipelineConfig,
    PoseFlowV3PipelineConfig,
    PoseFlowV4PipelineConfig,
    PoseFlowV5PipelineConfig,
    PoseHypothesisCodebook,
    build_poseflow_targets,
    build_poseflow_v3_targets,
    build_poseflow_v5_targets,
    flow_sample_keypoints,
    flow_sample_person_states,
)
from src.models.pipelines.pose_estimators.poseflow.model import PoseFlowHypothesisStage
from src.models.pipelines.pose_estimators.poseflow.state import (
    blend_box_references,
    corners_to_cxcywh,
    cxcywh_to_corners,
    euler_update_person_state,
    fourier_encode_coordinates,
    gather_person_slot_fields,
    interpolate_person_state,
    merge_person_slot_fields,
    pack_person_state,
    state_reference_gate,
    unpack_person_state,
)


def _batch() -> dict[str, torch.Tensor]:
    return {
        "gt_keypoints": torch.tensor(
            [[[[0.0, 0.0], [0.5, -0.5]], [[0.2, 0.4], [-0.5, 0.5]]]]
        ),
        "gt_labeled": torch.tensor([[[True, True], [True, False]]]),
        "gt_boxes_canon": torch.tensor([[[0.5, 0.5, 0.2, 0.4], [0.4, 0.6, 0.3, 0.2]]]),
    }


def test_poseflow_uses_ota_by_default() -> None:
    config = PoseFlowPipelineConfig()

    assert config.matching_strategy == "ota"
    assert config.ota_candidate_topk == 5
    assert config.flow_padding_policy == "source_prior"


def test_poseflow_config_builds_ota_matcher() -> None:
    pipeline = PoseFlowPipelineConfig(
        backbone_pretrained=False,
        num_queries=4,
        num_proposals=2,
        num_group=2,
        num_encoder_layers=1,
        num_decoder_layers=3,
        num_feature_levels=3,
        hidden_dim=32,
        feedforward_dim=64,
        num_heads=8,
        num_body_points=2,
        num_box_decoder_layers=1,
    ).build(max_epochs=1)

    assert isinstance(pipeline.criterion.matcher, DiffusionRegPoseOTAMatcher)
    assert isinstance(pipeline.model, PoseFlowModel)
    assert pipeline.flow_sampling_steps == 1


def test_poseflow_targets_interpolate_source_and_endpoint() -> None:
    model = SimpleNamespace(scale=6.0)
    batch = _batch()
    decoder_state, source, times, targets = build_poseflow_targets(
        batch,
        model,
        num_proposals=3,
        num_body_points=2,
        padding_policy="source_prior",
        noise_seed=7,
    )

    assert decoder_state.shape == source.shape == (1, 3, 4)
    assert times.shape == (1,)
    assert times.dtype.is_floating_point
    assert len(targets) == 1
    assert targets[0]["labels"].tolist() == [1, 1]
    assert torch.all((decoder_state >= 0.0) & (decoder_state <= 1.0))


def test_v3_person_state_round_trips_normal_and_crossed_boxes_without_projection() -> None:
    boxes = torch.tensor([[[0.5, 0.4, 0.2, 0.6]]])
    corners = cxcywh_to_corners(boxes)
    torch.testing.assert_close(corners, torch.tensor([[[0.4, 0.1, 0.6, 0.7]]]))
    torch.testing.assert_close(corners_to_cxcywh(corners), boxes)

    crossed_corners = torch.tensor([[[1.2, 0.8, -0.4, -0.1]]])
    crossed_box = corners_to_cxcywh(crossed_corners)
    torch.testing.assert_close(crossed_box, torch.tensor([[[0.4, 0.35, -1.6, -0.9]]]))
    assert torch.isfinite(crossed_box).all()

    keypoints = torch.tensor([[[0.1, 0.2, 0.3, 0.4]]])
    state = pack_person_state(crossed_corners, keypoints)
    unpacked_corners, unpacked_keypoints = unpack_person_state(state, num_body_points=2)
    torch.testing.assert_close(unpacked_corners, crossed_corners)
    torch.testing.assert_close(unpacked_keypoints, keypoints)


def test_v3_state_interpolation_euler_and_gate_oracle() -> None:
    source = torch.zeros(2, 1, 8)
    endpoint = torch.ones_like(source)
    times = torch.tensor([0.0, 1.0])
    interpolated = interpolate_person_state(source, endpoint, times)
    torch.testing.assert_close(interpolated[:, 0, 0], times)
    torch.testing.assert_close(
        euler_update_person_state(source, endpoint, source, sampling_steps=2),
        torch.full_like(source, 0.5),
    )
    image = torch.zeros(2, 1, 4)
    state = torch.ones_like(image)
    torch.testing.assert_close(state_reference_gate(times), torch.tensor([0.0, 0.5]))
    torch.testing.assert_close(
        blend_box_references(image, state, times)[:, 0, 0], torch.tensor([0.0, 0.5])
    )


def test_v4_fourier_coordinate_encoding_is_layout_preserving_and_finite() -> None:
    coordinates = torch.tensor([[[0.0, 0.5], [1.5, -2.0]]])

    encoded = fourier_encode_coordinates(coordinates, num_bands=3)

    assert encoded.shape == (1, 2, 12)
    assert torch.isfinite(encoded).all()
    assert not torch.equal(encoded[:, 0], encoded[:, 1])


def test_v4_hypothesis_stage_is_image_conditioned_and_gates_coordinate_prior() -> None:
    stage = PoseFlowHypothesisStage(
        num_queries=4,
        state_width=8,
        hidden_dim=32,
        num_heads=8,
        fourier_bands=3,
    ).eval()
    image_tokens = torch.randn(2, 7, 32)
    image_positions = torch.randn(2, 7, 32)
    image_padding_mask = torch.zeros(2, 7, dtype=torch.bool)
    time_embedding = torch.randn(2, 32)
    state_a = torch.zeros(2, 4, 8)
    state_b = torch.ones(2, 4, 8)

    with torch.no_grad():
        at_zero_a = stage(
            image_tokens, image_positions, image_padding_mask, state_a,
            torch.zeros(2), time_embedding,
        )
        at_zero_b = stage(
            image_tokens, image_positions, image_padding_mask, state_b,
            torch.zeros(2), time_embedding,
        )
        at_one_a = stage(
            image_tokens, image_positions, image_padding_mask, state_a,
            torch.ones(2), time_embedding,
        )
        at_one_b = stage(
            image_tokens, image_positions, image_padding_mask, state_b,
            torch.ones(2), time_embedding,
        )

    assert at_zero_a.shape == (2, 4, 32)
    torch.testing.assert_close(at_zero_a, at_zero_b)
    assert not torch.equal(at_one_a, at_one_b)
    assert not torch.equal(at_zero_a[:, 0], at_zero_a[:, 1])


def test_v4_hypothesis_diagnostics_are_detached_and_preserve_disabled_outputs() -> None:
    stage = PoseFlowHypothesisStage(
        num_queries=4,
        state_width=8,
        hidden_dim=32,
        num_heads=8,
        fourier_bands=3,
    ).eval()
    image_tokens = torch.randn(2, 7, 32)
    image_positions = torch.randn(2, 7, 32)
    image_padding_mask = torch.zeros(2, 7, dtype=torch.bool)
    person_state = torch.tensor(
        [[[1.5] * 8] * 4, [[-2.0] * 8] * 4], dtype=torch.float32
    )
    times = torch.tensor([0.0, 1.0])
    time_embedding = torch.randn(2, 32)

    with torch.no_grad():
        without_diagnostics = stage(
            image_tokens, image_positions, image_padding_mask, person_state, times, time_embedding,
        )
        with_diagnostics = stage(
            image_tokens,
            image_positions,
            image_padding_mask,
            person_state,
            times,
            time_embedding,
            collect_diagnostics=True,
        )

    torch.testing.assert_close(without_diagnostics, with_diagnostics, rtol=0.0, atol=0.0)
    diagnostics = stage.last_diagnostics()
    assert diagnostics is not None
    assert diagnostics["coordinate_gated_rms"].item() > 0.0
    assert diagnostics["flow_time_mean"].item() == 0.5
    assert all(value.ndim == 0 and not value.requires_grad and torch.isfinite(value) for value in diagnostics.values())

    with torch.no_grad():
        stage(
            image_tokens,
            image_positions,
            image_padding_mask,
            person_state,
            torch.zeros(2),
            time_embedding,
            collect_diagnostics=True,
        )
    endpoint_diagnostics = stage.last_diagnostics()
    assert endpoint_diagnostics is not None
    assert endpoint_diagnostics["coordinate_gated_rms"].item() == 0.0


def test_v3_slot_pool_gathers_paired_fields_atomically() -> None:
    active = {
        "state": torch.tensor([[[10.0], [20.0]]]),
        "source": torch.tensor([[[110.0], [120.0]]]),
        "features": torch.tensor([[[210.0], [220.0]]]),
    }
    fresh = {
        "state": torch.tensor([[[30.0], [40.0], [50.0]]]),
        "source": torch.tensor([[[130.0], [140.0], [150.0]]]),
        "features": torch.tensor([[[230.0], [240.0], [250.0]]]),
    }
    pool = merge_person_slot_fields(active, fresh)
    selected = gather_person_slot_fields(pool, torch.tensor([[4, 1]], dtype=torch.long))

    assert pool["state"].shape == (1, 5, 1)
    assert selected["state"].flatten().tolist() == [50.0, 20.0]
    assert selected["source"].flatten().tolist() == [150.0, 120.0]
    assert selected["features"].flatten().tolist() == [250.0, 220.0]


def test_v3_model_selects_person_state_and_carries_box_decoder_features() -> None:
    model = PoseFlowV3Model(
        backbone_pretrained=False,
        num_queries=4,
        num_proposals=2,
        num_group=2,
        num_encoder_layers=1,
        num_decoder_layers=3,
        num_feature_levels=3,
        hidden_dim=32,
        feedforward_dim=64,
        num_heads=8,
        num_body_points=2,
        num_box_decoder_layers=1,
        dn_number=0,
    ).eval()
    person_state = torch.randn(1, 4, 8)

    with torch.no_grad():
        output = model(
            torch.randn(1, 3, 64, 64),
            torch.zeros(1, 4, 4),
            torch.tensor([1.0]),
            person_state=person_state,
            return_selection_metadata=True,
        )

    assert output["pred_boxes"].shape == (1, 2, 4)
    assert output["pred_keypoints"].shape == (1, 2, 6)
    assert output["selected_person_features"].shape == (1, 2, 32)
    assert output["selected_person_references"].shape == (1, 2, 4)
    assert output["selected_person_indices"].shape == (1, 2)
    assert output["selected_person_indices"].max() < 4


def test_v4_model_predicts_pose_at_hypothesis_and_every_refiner_layer() -> None:
    model = PoseFlowV4Model(
        backbone_pretrained=False,
        num_queries=4,
        num_proposals=2,
        num_group=2,
        num_encoder_layers=1,
        num_decoder_layers=3,
        num_feature_levels=3,
        hidden_dim=32,
        feedforward_dim=64,
        num_heads=8,
        num_body_points=2,
        num_box_decoder_layers=1,
        dn_number=0,
    ).eval()

    with torch.no_grad():
        output = model(
            torch.randn(1, 3, 64, 64),
            torch.zeros(1, 4, 4),
            torch.tensor([0.5]),
            person_state=torch.randn(1, 4, 8),
            return_selection_metadata=True,
        )

    assert output["pred_boxes"].shape == (1, 2, 4)
    assert output["pred_keypoints"].shape == (1, 2, 6)
    assert output["selected_person_indices"].shape == (1, 2)
    assert len(output["aux_outputs"]) == 3
    assert output["aux_outputs"][0]["pred_keypoints"].shape == (1, 4, 6)
    assert all(auxiliary["pred_keypoints"].shape == (1, 2, 6) for auxiliary in output["aux_outputs"][1:])
    assert torch.isfinite(output["corner_order_penalty"])


def test_v4_model_diagnostics_track_selected_hypothesis_refinement() -> None:
    model = PoseFlowV4Model(
        backbone_pretrained=False,
        num_queries=4,
        num_proposals=2,
        num_group=2,
        num_encoder_layers=1,
        num_decoder_layers=3,
        num_feature_levels=3,
        hidden_dim=32,
        feedforward_dim=64,
        num_heads=8,
        num_body_points=2,
        num_box_decoder_layers=1,
        dn_number=0,
    ).eval()
    image = torch.randn(1, 3, 64, 64)
    state = torch.randn(1, 4, 8)

    with torch.no_grad():
        without_diagnostics = model(
            image, state[..., 4:], torch.tensor([0.5]), person_state=state,
        )
        with_diagnostics = model(
            image,
            state[..., 4:],
            torch.tensor([0.5]),
            person_state=state,
            collect_diagnostics=True,
        )

    torch.testing.assert_close(
        without_diagnostics["pred_boxes"], with_diagnostics["pred_boxes"], rtol=0.0, atol=0.0,
    )
    diagnostics = model.last_v4_diagnostics()
    assert diagnostics is not None
    assert {"selected_initial_score_mean", "hypothesis_to_final_box_l1", "hypothesis_to_final_pose_l1"} <= diagnostics.keys()
    assert all(value.ndim == 0 and not value.requires_grad and torch.isfinite(value) for value in diagnostics.values())


def test_v3_model_supports_a_multi_image_joint_reference_gate() -> None:
    model = PoseFlowV3Model(
        backbone_pretrained=False,
        num_queries=4,
        num_proposals=2,
        num_group=2,
        num_encoder_layers=1,
        num_decoder_layers=3,
        num_feature_levels=3,
        hidden_dim=32,
        feedforward_dim=64,
        num_heads=8,
        num_body_points=2,
        num_box_decoder_layers=1,
        dn_number=0,
    ).eval()

    with torch.no_grad():
        output = model(
            torch.randn(2, 3, 64, 64),
            torch.zeros(2, 4, 4),
            torch.tensor([0.25, 0.75]),
            person_state=torch.randn(2, 4, 8),
        )

    assert output["pred_keypoints"].shape == (2, 2, 6)


def test_v3_control_is_state_independent_and_enabled_gate_changes_outputs() -> None:
    model = PoseFlowV3Model(
        backbone_pretrained=False,
        num_queries=4,
        num_proposals=2,
        num_group=2,
        num_encoder_layers=1,
        num_decoder_layers=3,
        num_feature_levels=3,
        hidden_dim=32,
        feedforward_dim=64,
        num_heads=8,
        num_body_points=2,
        num_box_decoder_layers=1,
        dn_number=0,
    ).eval()
    image = torch.randn(1, 3, 64, 64)
    state_a = torch.zeros(1, 4, 8)
    state_b = torch.ones(1, 4, 8)
    time = torch.tensor([1.0])

    with torch.no_grad():
        model.flow_state_conditioning = False
        control_a = model(image, state_a[..., 4:], time, person_state=state_a)
        control_b = model(image, state_b[..., 4:], time, person_state=state_b)
        torch.testing.assert_close(control_a["pred_boxes"], control_b["pred_boxes"])

        model.flow_state_conditioning = True
        conditioned_a = model(image, state_a[..., 4:], time, person_state=state_a)
        conditioned_b = model(image, state_b[..., 4:], time, person_state=state_b)

    assert not torch.equal(conditioned_a["pred_boxes"], conditioned_b["pred_boxes"])


def test_v4_config_defaults_to_ota_and_builds_a_matched_state_control() -> None:
    config = PoseFlowV4PipelineConfig(
        backbone_pretrained=False,
        num_queries=4,
        num_proposals=2,
        num_group=2,
        num_encoder_layers=1,
        num_decoder_layers=3,
        num_feature_levels=3,
        hidden_dim=32,
        feedforward_dim=64,
        num_heads=8,
        num_body_points=2,
        num_box_decoder_layers=1,
        dn_number=0,
        fourier_bands=3,
    )

    pipeline = config.build(max_epochs=1)

    assert isinstance(pipeline.model, PoseFlowV4Model)
    assert isinstance(pipeline.criterion.matcher, DiffusionRegPoseOTAMatcher)


def test_v4_training_loss_supervises_hypothesis_and_joint_refiners() -> None:
    pipeline = PoseFlowV4PipelineConfig(
        backbone_pretrained=False,
        num_queries=4,
        num_proposals=2,
        num_group=2,
        num_encoder_layers=1,
        num_decoder_layers=3,
        num_feature_levels=3,
        hidden_dim=32,
        feedforward_dim=64,
        num_heads=8,
        num_body_points=2,
        num_box_decoder_layers=1,
        dn_number=0,
        fourier_bands=3,
    ).build(max_epochs=1)
    batch = _batch() | {"image": torch.randn(1, 3, 64, 64)}

    losses = pipeline._losses_for_batch(batch)

    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert pipeline.model.hypothesis_stage.coordinate_projection.layers[0].weight.grad is not None
    assert "loss_keypoints_0" in losses
    assert "loss_oks_0" in losses
    assert "loss_keypoints_1" in losses
    assert "loss_keypoints_2" in losses
    assert "loss_corner_order" in losses


def test_v4_pipeline_logs_activity_and_named_stage_diagnostics() -> None:
    pipeline = PoseFlowV4PipelineConfig(
        backbone_pretrained=False,
        num_queries=4,
        num_proposals=2,
        num_group=2,
        num_encoder_layers=1,
        num_decoder_layers=3,
        num_feature_levels=3,
        hidden_dim=32,
        feedforward_dim=64,
        num_heads=8,
        num_body_points=2,
        num_box_decoder_layers=1,
        dn_number=0,
    ).build(max_epochs=1)
    logged_names: list[str] = []

    def record(name: str, *args: object, **kwargs: object) -> None:
        del args, kwargs
        logged_names.append(name)

    pipeline.log = record  # type: ignore[method-assign]
    pipeline._latest_v4_diagnostics = {
        "coordinate_gated_rms": torch.tensor(0.25),
        "hypothesis_to_final_box_l1": torch.tensor(0.1),
    }
    pipeline._log_codebook_diagnostics("train", batch_size=1)
    pipeline._log_losses(
        "train",
        {
            "total": torch.tensor(1.0),
            "loss_ce_0": torch.tensor(1.0),
            "loss_keypoints_1": torch.tensor(1.0),
            "loss_oks_2": torch.tensor(1.0),
            "loss_bbox": torch.tensor(1.0),
        },
        batch_size=1,
    )

    assert "debug/train/coordinate_gated_rms" in logged_names
    assert "debug/train/hypothesis/loss_ce" in logged_names
    assert "debug/train/refiner_0/loss_keypoints" in logged_names
    assert "debug/train/refiner_1/loss_oks" in logged_names
    assert "debug/train/final_refiner/loss_bbox" in logged_names
    assert not any("poseflow_v" in name for name in logged_names if name.startswith("debug/"))


def test_v5_pipeline_uses_common_diagnostic_tags() -> None:
    pipeline = PoseFlowV5PipelineConfig(
        backbone_pretrained=False,
        num_queries=4,
        num_proposals=2,
        num_group=2,
        num_encoder_layers=1,
        num_decoder_layers=3,
        num_feature_levels=3,
        hidden_dim=32,
        feedforward_dim=64,
        num_heads=8,
        num_body_points=2,
        num_box_decoder_layers=1,
        dn_number=0,
    ).build(max_epochs=1)
    logged_names: list[str] = []

    def record(name: str, *args: object, **kwargs: object) -> None:
        del args, kwargs
        logged_names.append(name)

    pipeline.log = record  # type: ignore[method-assign]
    pipeline._latest_v4_diagnostics = {"coordinate_gated_rms": torch.tensor(0.25)}
    pipeline._log_codebook_diagnostics("val", batch_size=1)
    pipeline._log_losses(
        "val",
        {
            "total": torch.tensor(1.0),
            "loss_center_direct_0": torch.tensor(1.0),
            "loss_center_midpoint": torch.tensor(1.0),
        },
        batch_size=1,
    )

    assert "debug/val/coordinate_gated_rms" in logged_names
    assert "debug/val/hypothesis/loss_center_direct" in logged_names
    assert "debug/val/final_refiner/loss_center_midpoint" in logged_names
    assert not any("poseflow_v" in name for name in logged_names if name.startswith("debug/"))


def test_v4_sampler_recomputes_hypotheses_with_50_plus_fresh_coordinate_pool() -> None:
    model = PoseFlowV4Model(
        backbone_pretrained=False,
        num_queries=4,
        num_proposals=2,
        num_group=2,
        num_encoder_layers=1,
        num_decoder_layers=3,
        num_feature_levels=3,
        hidden_dim=32,
        feedforward_dim=64,
        num_heads=8,
        num_body_points=2,
        num_box_decoder_layers=1,
        dn_number=0,
        flow_sampling_steps=3,
    ).eval()

    with torch.no_grad():
        output = model.sample_keypoints(torch.randn(1, 3, 64, 64))

    assert output["pred_boxes"].shape == (1, 2, 4)
    assert output["flow_person_state"].shape == (1, 2, 8)
    assert output["flow_person_source"].shape == (1, 2, 8)
    assert torch.isfinite(output["flow_person_state"]).all()


def test_v4_control_zeros_only_the_coordinate_prior_path() -> None:
    model = PoseFlowV4Model(
        backbone_pretrained=False,
        num_queries=4,
        num_proposals=2,
        num_group=2,
        num_encoder_layers=1,
        num_decoder_layers=3,
        num_feature_levels=3,
        hidden_dim=32,
        feedforward_dim=64,
        num_heads=8,
        num_body_points=2,
        num_box_decoder_layers=1,
        dn_number=0,
        flow_state_conditioning=False,
    ).eval()
    image = torch.randn(1, 3, 64, 64)

    with torch.no_grad():
        output_a = model(image, torch.zeros(1, 4, 4), torch.ones(1), person_state=torch.zeros(1, 4, 8))
        output_b = model(image, torch.zeros(1, 4, 4), torch.ones(1), person_state=torch.ones(1, 4, 8))

    torch.testing.assert_close(output_a["pred_logits"], output_b["pred_logits"])
    torch.testing.assert_close(output_a["pred_boxes"], output_b["pred_boxes"])


def test_v3_zero_denoising_queries_skip_denoising_bookkeeping() -> None:
    model = PoseFlowV3Model(
        backbone_pretrained=False,
        num_queries=4,
        num_proposals=2,
        num_group=2,
        num_encoder_layers=1,
        num_decoder_layers=3,
        num_feature_levels=3,
        hidden_dim=32,
        feedforward_dim=64,
        num_heads=8,
        num_body_points=2,
        num_box_decoder_layers=1,
        dn_number=0,
    ).train()
    targets = [{
        "labels": torch.tensor([1]),
        "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
    }]

    output = model(
        torch.randn(1, 3, 64, 64),
        torch.zeros(1, 4, 4),
        torch.tensor([0.5]),
        targets,
        person_state=torch.zeros(1, 4, 8),
    )

    assert "dn_class_pred" not in output


def test_v3_target_builder_keeps_source_prior_padding_paired() -> None:
    model = SimpleNamespace(scale=6.0)
    state, source, times, targets = build_poseflow_v3_targets(
        _batch(),
        model,
        num_queries=3,
        num_body_points=2,
        padding_policy="source_prior",
        noise_seed=7,
    )

    endpoint = source + (state - source) / times[:, None, None]
    assert state.shape == source.shape == (1, 3, 8)
    torch.testing.assert_close(endpoint[:, 2], source[:, 2])
    assert targets[0]["boxes"].shape == (2, 4)


def test_v5_target_builder_mixes_augmented_and_interpolated_person_sources() -> None:
    model = SimpleNamespace(scale=6.0)
    state, source, times, targets = build_poseflow_v5_targets(
        _batch(),
        model,
        num_queries=4,
        num_body_points=2,
        padding_policy="source_prior",
        noise_seed=7,
    )

    endpoint = source + (state - source) / times[:, None, None]
    assert state.shape == source.shape == (1, 4, 8)
    assert torch.isfinite(source).all()
    torch.testing.assert_close(endpoint[:, 2:], source[:, 2:])
    assert targets[0]["boxes"].shape == (2, 4)


def test_v5_box_outputs_supervise_center_and_corner_midpoint() -> None:
    pipeline = PoseFlowV5PipelineConfig(
        backbone_pretrained=False,
        num_queries=4,
        num_proposals=2,
        num_group=2,
        num_encoder_layers=1,
        num_decoder_layers=3,
        num_feature_levels=3,
        hidden_dim=32,
        feedforward_dim=64,
        num_heads=8,
        num_body_points=2,
        num_box_decoder_layers=1,
        dn_number=0,
    ).build(max_epochs=1)
    batch = _batch() | {"image": torch.randn(1, 3, 64, 64)}

    losses = pipeline._losses_for_batch(batch)

    assert isinstance(pipeline.model, PoseFlowV5Model)
    assert torch.isfinite(losses["total"])
    assert "loss_center_direct" in losses
    assert "loss_center_midpoint" in losses
    losses["total"].backward()
    assert pipeline.model.initial_coordinate_head.layers[-1].weight.grad is not None


def test_v5_target_builder_uses_augmented_resamples_for_one_person_images() -> None:
    model = SimpleNamespace(scale=6.0)
    batch = _batch()
    batch["gt_labeled"][:, 1] = False
    batch["person_mask"] = torch.tensor([[True, False]])
    state, source, times, _ = build_poseflow_v5_targets(
        batch,
        model,
        num_queries=4,
        num_body_points=2,
        padding_policy="source_prior",
        noise_seed=7,
    )

    endpoint = source + (state - source) / times[:, None, None]
    assert torch.isfinite(source).all()
    torch.testing.assert_close(endpoint[:, 1:], source[:, 1:])


def test_v5_benchmark_resolves_its_distinct_schedule() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "conf"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        config = compose(config_name="benchmark_poseflow_v5")

    assert config.pipeline._target_.endswith("PoseFlowV5PipelineConfig")
    assert config.pipeline.scheduler_type == "two_cycle_cosine"
    assert config.pipeline.lr_cycle_restart_epoch == 40
    assert config.pipeline.center_loss_factor == pytest.approx(0.1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_v5_cuda_forward_backward_and_three_step_sampler_are_finite() -> None:
    pipeline = PoseFlowV5PipelineConfig(
        backbone_pretrained=False,
        num_queries=4,
        num_proposals=2,
        num_group=2,
        num_encoder_layers=1,
        num_decoder_layers=3,
        num_feature_levels=3,
        hidden_dim=32,
        feedforward_dim=64,
        num_heads=8,
        num_body_points=2,
        num_box_decoder_layers=1,
        dn_number=0,
    ).build(max_epochs=3).cuda()
    batch = {name: value.cuda() for name, value in _batch().items()} | {
        "image": torch.randn(1, 3, 64, 64, device="cuda"),
    }

    losses = pipeline._losses_for_batch(batch)
    losses["total"].backward()
    with torch.no_grad():
        sampled = pipeline.model.sample_keypoints(batch["image"])

    assert torch.isfinite(losses["total"])
    assert torch.isfinite(sampled["flow_person_state"]).all()


class _V3EndpointOracle(torch.nn.Module):
    scale = 6.0
    num_queries = 5
    num_proposals = 2
    transformer = SimpleNamespace(decoder=SimpleNamespace(num_group=2, num_body_points=2))

    def __init__(self) -> None:
        super().__init__()
        self.states: list[torch.Tensor] = []

    def forward(
        self,
        images,
        diffusion_keypoints,
        times,
        padding_mask=None,
        *,
        person_state,
        active_query_features=None,
        active_image_references=None,
        return_selection_metadata=False,
    ):
        del diffusion_keypoints, times, padding_mask
        assert return_selection_metadata
        self.states.append(person_state.clone())
        assert (active_query_features is None) == (len(self.states) == 1)
        assert (active_image_references is None) == (len(self.states) == 1)
        return {
            "pred_logits": torch.zeros(images.shape[0], 2, 1),
            "pred_boxes": torch.full((images.shape[0], 2, 4), 0.5),
            "pred_keypoints": torch.full((images.shape[0], 2, 6), 0.5),
            "selected_person_indices": torch.tensor([[4, 1]], dtype=torch.long),
            "selected_person_features": torch.ones(images.shape[0], 2, 3),
            "selected_person_references": torch.full((images.shape[0], 2, 4), 0.5),
        }


def test_v3_sampler_keeps_selected_source_paired_through_50_plus_150_refill() -> None:
    oracle = _V3EndpointOracle()
    output = flow_sample_person_states(
        oracle,
        torch.ones(1, 3, 8, 8),
        sampling_steps=2,
        generator=torch.Generator().manual_seed(4),
    )

    assert oracle.states[0].shape == (1, 5, 8)
    assert oracle.states[1].shape == (1, 5, 8)
    torch.testing.assert_close(output["flow_person_source"][:, 0], oracle.states[1][:, 4])
    torch.testing.assert_close(output["flow_person_source"][:, 1], oracle.states[0][:, 1])
    assert output["flow_person_state"].shape == (1, 2, 8)


def test_v3_config_uses_the_matched_200_slot_50_active_pool() -> None:
    config = PoseFlowV3PipelineConfig(num_queries=200, num_proposals=50, num_group=50)

    assert config.matching_strategy == "ota"
    assert config.num_queries == 200
    assert config.flow_state_conditioning


class _EndpointOracle(torch.nn.Module):
    scale = 6.0
    num_proposals = 2
    transformer = SimpleNamespace(decoder=SimpleNamespace(num_group=2, num_body_points=2))

    def forward(self, images, state, times, padding_mask=None):
        del state, times, padding_mask
        return {
            "pred_keypoints": torch.full(
                (images.shape[0], 2, 4), 0.5, device=images.device, dtype=images.dtype
            ),
            "pred_logits": torch.zeros(images.shape[0], 2, 1, device=images.device),
            "pred_boxes": torch.zeros(images.shape[0], 2, 4, device=images.device),
        }


def test_flow_sampler_reaches_oracle_endpoint_with_fixed_source() -> None:
    output = flow_sample_keypoints(
        _EndpointOracle(),
        torch.ones(1, 3, 8, 8),
        sampling_steps=3,
        generator=torch.Generator().manual_seed(4),
    )

    torch.testing.assert_close(output["flow_keypoints"], torch.zeros(1, 2, 4))


class _RenewalOracle(_EndpointOracle):
    def __init__(self) -> None:
        super().__init__()
        self.states: list[torch.Tensor] = []

    def forward(self, images, state, times, padding_mask=None):
        self.states.append(state.clone())
        output = super().forward(images, state, times, padding_mask)
        output["pred_logits"][:, 0, 0] = 8.0
        output["pred_logits"][:, 1, 0] = -8.0
        return output


def test_flow_renewal_restarts_weak_slot_with_paired_source() -> None:
    oracle = _RenewalOracle()

    output = flow_sample_keypoints(
        oracle,
        torch.ones(1, 3, 8, 8),
        sampling_steps=2,
        renewal=True,
        renewal_score_threshold=0.5,
        generator=torch.Generator().manual_seed(4),
    )

    assert output["flow_renewed_slots"].tolist() == [[1]]
    torch.testing.assert_close(oracle.states[1][:, 1], output["flow_source"][:, 1])
    assert not torch.equal(oracle.states[1][:, 0], output["flow_source"][:, 0])


def test_flow_does_not_renew_after_final_decoder_call() -> None:
    output = flow_sample_keypoints(
        _RenewalOracle(),
        torch.ones(1, 3, 8, 8),
        sampling_steps=1,
        renewal=True,
        renewal_score_threshold=0.5,
        generator=torch.Generator().manual_seed(4),
    )

    assert output["flow_renewed_slots"].shape == (0, 1)


def test_flow_without_renewal_reports_zero_counts_for_each_step() -> None:
    output = flow_sample_keypoints(
        _EndpointOracle(), torch.ones(1, 3, 8, 8), sampling_steps=3
    )

    assert output["flow_renewed_slots"].tolist() == [[0], [0]]


def test_flow_ensemble_preserves_ordered_step_predictions() -> None:
    output = flow_sample_keypoints(
        _EndpointOracle(),
        torch.ones(1, 3, 8, 8),
        sampling_steps=3,
        ensemble=True,
        generator=torch.Generator().manual_seed(4),
    )

    assert output["pred_logits"].shape[1] == 2
    assert output["pred_boxes"].shape[1] == 2
    assert output["pred_keypoints"].shape[1] == 2
    step_predictions = output["flow_step_predictions"]
    assert isinstance(step_predictions, list)
    assert len(step_predictions) == 3
    assert all(prediction["pred_logits"].shape[1] == 2 for prediction in step_predictions)


def test_flow_rejects_invalid_renewal_threshold() -> None:
    with pytest.raises(ValueError, match="renewal_score_threshold"):
        flow_sample_keypoints(
            _EndpointOracle(),
            torch.ones(1, 3, 8, 8),
            sampling_steps=2,
            renewal=True,
            renewal_score_threshold=1.1,
        )


def test_poseflow_config_rejects_invalid_renewal_threshold() -> None:
    with pytest.raises(ValueError, match="flow_renewal_score_threshold"):
        PoseFlowPipelineConfig(flow_renewal_score_threshold=-0.1)


def test_poseflow_hydra_config_has_distinct_identity_and_ota_default() -> None:
    config_dir = str((__file__.replace("/tests/test_poseflow.py", "/conf")))
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        poseflow = OmegaConf.to_container(compose(config_name="benchmark_poseflow"), resolve=True)
        control = OmegaConf.to_container(
            compose(
                config_name=(
                    "benchmark_diffusionregpose_development_compatible_256_keep_layers_ota"
                )
            ),
            resolve=True,
        )

    assert poseflow["pipeline"]["matching_strategy"] == "ota"
    assert poseflow["run_name"] == "poseflow_training"
    assert poseflow["pipeline"]["num_encoder_layers"] == 3
    assert poseflow["pipeline"]["num_decoder_layers"] == 3
    assert poseflow["pipeline"]["num_box_decoder_layers"] == 1
    assert poseflow["pipeline"]["pretrained_keep_encoder_layers"] == [2, 4, 6]
    assert poseflow["pipeline"]["pretrained_keep_decoder_layers"] == [2, 4, 6]
    for section in ("data", "training", "hardware"):
        assert poseflow[section] == control[section]


def test_poseflow_c128_codebook_hydra_config_changes_only_codebook_fields() -> None:
    config_dir = str((__file__.replace("/tests/test_poseflow.py", "/conf")))
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        poseflow = OmegaConf.to_container(compose(config_name="benchmark_poseflow"), resolve=True)
        codebook = OmegaConf.to_container(
            compose(config_name="benchmark_poseflow_codebook_c128_all_layers"), resolve=True
        )

    assert codebook["pipeline"]["pose_codebook_size"] == 128
    assert codebook["pipeline"]["pose_codebook_layers"] == [0, 1, 2]
    assert codebook["pipeline"]["pose_codebook_initial_gate"] == 0.1
    for section in ("data", "training", "hardware"):
        assert codebook[section] == poseflow[section]
    for name, value in poseflow["pipeline"].items():
        if name not in {
            "pose_codebook_size",
            "pose_codebook_layers",
            "pose_codebook_initial_gate",
        }:
            assert codebook["pipeline"][name] == value


def test_poseflow_rejects_zero_sampling_steps() -> None:
    with pytest.raises(ValueError, match="flow_sampling_steps"):
        PoseFlowPipelineConfig(flow_sampling_steps=0)


def test_pose_hypothesis_codebook_zero_gate_preserves_queries() -> None:
    codebook = PoseHypothesisCodebook(codebook_size=3, hidden_dim=8, num_heads=2)
    queries = torch.randn(5, 2, 8, requires_grad=True)

    output = codebook(queries)

    assert torch.equal(output, queries)
    output.square().mean().backward()
    assert codebook.residual_gate.grad is not None
    assert torch.isfinite(codebook.residual_gate.grad)
    assert codebook.residual_gate.grad.item() != 0.0


def test_pose_hypothesis_codebook_leaves_denoising_queries_unchanged() -> None:
    codebook = PoseHypothesisCodebook(codebook_size=3, hidden_dim=8, num_heads=2)
    codebook.residual_gate.data.fill_(1.0)
    queries = torch.randn(5, 2, 8)

    conditioned = codebook(queries, denoising_query_count=2)

    assert torch.equal(conditioned[:2], queries[:2])
    assert not torch.equal(conditioned[2:], queries[2:])


def test_pose_hypothesis_codebook_has_finite_gradients_after_gate_opens() -> None:
    codebook = PoseHypothesisCodebook(codebook_size=3, hidden_dim=8, num_heads=2)
    codebook.residual_gate.data.fill_(1.0)
    queries = torch.randn(5, 2, 8, requires_grad=True)

    codebook(queries).square().mean().backward()

    assert torch.isfinite(codebook.embeddings.grad).all()
    assert torch.isfinite(codebook.attention.in_proj_weight.grad).all()
    assert torch.isfinite(queries.grad).all()


def test_pose_hypothesis_codebook_nonzero_initial_gate_activates_branch() -> None:
    codebook = PoseHypothesisCodebook(
        codebook_size=3,
        hidden_dim=8,
        num_heads=2,
        initial_gate=0.1,
    )
    queries = torch.randn(5, 2, 8, requires_grad=True)

    conditioned = codebook(queries, denoising_query_count=2)
    conditioned.square().mean().backward()

    assert codebook.residual_gate.item() == pytest.approx(0.1)
    assert torch.equal(conditioned[:2], queries[:2])
    assert not torch.equal(conditioned[2:], queries[2:])
    assert torch.isfinite(codebook.embeddings.grad).all()
    assert codebook.embeddings.grad.abs().sum().item() > 0.0
    assert torch.isfinite(codebook.attention.in_proj_weight.grad).all()
    assert codebook.attention.in_proj_weight.grad.abs().sum().item() > 0.0


def test_pose_hypothesis_codebook_reports_clean_query_diagnostics() -> None:
    codebook = PoseHypothesisCodebook(
        codebook_size=3,
        hidden_dim=8,
        num_heads=2,
        initial_gate=0.1,
    )
    queries = torch.randn(5, 2, 8)

    conditioned = codebook(
        queries,
        denoising_query_count=2,
        collect_diagnostics=True,
        collect_attention=True,
    )
    diagnostics = codebook.consume_diagnostics()

    assert torch.equal(conditioned[:2], queries[:2])
    assert diagnostics is not None
    assert diagnostics["gate"].item() == pytest.approx(0.1)
    assert diagnostics["query_rms"].item() > 0.0
    assert diagnostics["attended_rms"].item() > 0.0
    assert diagnostics["residual_rms"].item() > 0.0
    assert diagnostics["residual_to_query_rms"].item() > 0.0
    assert 0.0 <= diagnostics["normalized_attention_entropy"].item() <= 1.0
    assert 1.0 <= diagnostics["effective_slot_count"].item() <= 3.0


def test_pose_hypothesis_codebook_zero_gate_has_zero_residual_diagnostic() -> None:
    codebook = PoseHypothesisCodebook(codebook_size=3, hidden_dim=8, num_heads=2)
    queries = torch.randn(5, 2, 8)

    codebook(queries, collect_diagnostics=True)
    diagnostics = codebook.consume_diagnostics()

    assert diagnostics is not None
    assert diagnostics["residual_rms"].item() == 0.0
    assert diagnostics["residual_to_query_rms"].item() == 0.0
    assert "normalized_attention_entropy" not in diagnostics


def test_pose_hypothesis_codebook_diagnostics_handle_one_slot_and_no_clean_queries() -> None:
    codebook = PoseHypothesisCodebook(codebook_size=1, hidden_dim=8, num_heads=2)
    queries = torch.randn(2, 2, 8)

    codebook(queries, collect_diagnostics=True, collect_attention=True)
    diagnostics = codebook.consume_diagnostics()

    assert diagnostics is not None
    assert diagnostics["normalized_attention_entropy"].item() == 0.0
    assert diagnostics["effective_slot_count"].item() == 1.0
    assert torch.equal(
        codebook(queries, denoising_query_count=queries.shape[0], collect_diagnostics=True),
        queries,
    )
    assert codebook.consume_diagnostics() is None


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"pose_codebook_size": -1}, "pose_codebook_size"),
        ({"pose_codebook_layers": (1,)}, "pose_codebook_layers"),
        ({"pose_codebook_size": 2, "pose_codebook_layers": (3,)}, "decoder"),
        ({"pose_codebook_size": 2, "pose_codebook_layers": (1, 1)}, "duplicates"),
        ({"pose_codebook_initial_gate": float("nan")}, "initial_gate"),
    ],
)
def test_poseflow_rejects_invalid_codebook_configuration(settings, message) -> None:
    with pytest.raises(ValueError, match=message):
        PoseFlowPipelineConfig(
            num_decoder_layers=3,
            num_box_decoder_layers=1,
            **settings,
        )


def test_poseflow_codebook_defaults_to_first_pose_decoder_layer() -> None:
    model = PoseFlowModel(
        backbone_pretrained=False,
        num_queries=4,
        num_proposals=2,
        num_group=2,
        num_encoder_layers=1,
        num_decoder_layers=3,
        num_feature_levels=3,
        hidden_dim=32,
        feedforward_dim=64,
        num_heads=8,
        num_body_points=2,
        num_box_decoder_layers=1,
        pose_codebook_size=2,
    )

    assert model.pose_codebook_layers == (1,)
    assert set(model.post_image_attention_by_layer()) == {1}


def test_poseflow_reuses_one_codebook_at_every_configured_decoder_layer() -> None:
    model = PoseFlowModel(
        backbone_pretrained=False,
        num_queries=4,
        num_proposals=2,
        num_group=2,
        num_encoder_layers=1,
        num_decoder_layers=3,
        num_feature_levels=3,
        hidden_dim=32,
        feedforward_dim=64,
        num_heads=8,
        num_body_points=2,
        num_box_decoder_layers=1,
        pose_codebook_size=2,
        pose_codebook_layers=(0, 1, 2),
        pose_codebook_initial_gate=0.1,
    )

    post_image_attention = model.post_image_attention_by_layer()

    assert post_image_attention is not None
    assert set(post_image_attention) == {0, 1, 2}
    assert post_image_attention[0] is post_image_attention[1] is post_image_attention[2]
    assert model.pose_codebook.residual_gate.item() == pytest.approx(0.1)


def test_poseflow_codebook_build_preserves_existing_initialization() -> None:
    pipeline = PoseFlowPipelineConfig(
        backbone_pretrained=False,
        num_queries=4,
        num_proposals=2,
        num_group=2,
        num_encoder_layers=1,
        num_decoder_layers=3,
        num_feature_levels=3,
        hidden_dim=32,
        feedforward_dim=64,
        num_heads=8,
        num_body_points=2,
        num_box_decoder_layers=1,
        pose_codebook_size=2,
    ).build(max_epochs=1)

    assert pipeline.model.pose_codebook is not None

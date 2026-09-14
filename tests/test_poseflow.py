from __future__ import annotations

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
    PoseFlowPipelineConfig,
    build_poseflow_targets,
    flow_sample_keypoints,
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


class _EndpointOracle(torch.nn.Module):
    scale = 6.0
    num_proposals = 2
    transformer = SimpleNamespace(decoder=SimpleNamespace(num_group=2, num_body_points=2))

    def forward(self, images, state, times, padding_mask=None):
        del state, times, padding_mask
        return {
            "pred_keypoints": torch.full(
                (images.shape[0], 2, 4), 0.5, device=images.device, dtype=images.dtype
            )
        }


def test_flow_sampler_reaches_oracle_endpoint_with_fixed_source() -> None:
    output = flow_sample_keypoints(
        _EndpointOracle(),
        torch.ones(1, 3, 8, 8),
        sampling_steps=3,
        generator=torch.Generator().manual_seed(4),
    )

    torch.testing.assert_close(output["flow_keypoints"], torch.zeros(1, 2, 4))


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


def test_poseflow_rejects_zero_sampling_steps() -> None:
    with pytest.raises(ValueError, match="flow_sampling_steps"):
        PoseFlowPipelineConfig(flow_sampling_steps=0)

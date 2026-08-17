from __future__ import annotations

import torch

from src.models.pipelines.pose_estimators.diffusionregpose.training import (
    DiffusionRegPoseCriterion,
    DiffusionRegPoseMatcher,
    DiffusionRegPosePipelineConfig,
    build_diffusionregpose_targets,
)


def _batch() -> dict[str, torch.Tensor]:
    return {
        "gt_keypoints": torch.tensor(
            [[[[0.0, 0.0], [0.1, 0.1]]]], dtype=torch.float32
        ),
        "gt_labeled": torch.ones(1, 1, 2),
        "gt_visible": torch.ones(1, 1, 2),
        "gt_boxes_pixel": torch.tensor([[[10.0, 20.0, 20.0, 40.0]]]),
        "gt_boxes_canon": torch.tensor([[[0.0, 0.0, 0.2, 0.4]]]),
        "gt_box_labeled": torch.ones(1, 1),
        "img_hw": torch.tensor([[100.0, 100.0]]),
    }


def test_build_targets_preserves_reference_layout_and_zero_timestep() -> None:
    batch = _batch()
    diffusion = type("Diffusion", (), {
        "scale": 6.0,
        "q_sample": lambda self, x_start, timesteps, noise: x_start,
    })()

    diffusion_keypoints, timesteps, targets = build_diffusionregpose_targets(
        batch, diffusion, num_proposals=3, num_body_points=2,
        noise=torch.zeros(1, 3, 4),
    )

    assert diffusion_keypoints.shape == (1, 3, 4)
    assert timesteps.tolist() == [0]
    assert targets[0]["boxes"].shape == (1, 4)
    assert targets[0]["keypoints"].shape == (1, 6)
    torch.testing.assert_close(
        targets[0]["keypoints"], torch.tensor([[0.5, 0.5, 0.55, 0.55, 1.0, 1.0]])
    )


def test_matcher_and_criterion_handle_empty_target_image() -> None:
    matcher = DiffusionRegPoseMatcher(num_body_points=2)
    criterion = DiffusionRegPoseCriterion(
        num_classes=2, matcher=matcher, num_body_points=2,
        num_box_decoder_layers=1,
    )
    outputs = {
        "pred_logits": torch.zeros(1, 3, 2, requires_grad=True),
        "pred_boxes": torch.full((1, 3, 4), 0.5, requires_grad=True),
        "pred_keypoints": torch.full((1, 3, 6), 0.5, requires_grad=True),
    }
    targets = [{
        "labels": torch.zeros(0, dtype=torch.long),
        "boxes": torch.zeros(0, 4),
        "keypoints": torch.zeros(0, 6),
        "area": torch.zeros(0),
    }]

    losses = criterion(outputs, targets)

    assert torch.isfinite(losses["total"])
    losses["total"].backward()


def test_reduced_pipeline_builds_finite_training_loss() -> None:
    pipeline = DiffusionRegPosePipelineConfig(
        backbone_pretrained=False,
        num_queries=4,
        num_proposals=2,
        num_group=2,
        num_encoder_layers=1,
        num_decoder_layers=3,
    ).build(max_epochs=1)
    batch = {
        "image": torch.rand(1, 3, 128, 128),
        "gt_keypoints": torch.zeros(1, 1, 14, 2),
        "gt_labeled": torch.ones(1, 1, 14),
        "gt_visible": torch.ones(1, 1, 14),
        "gt_boxes_canon": torch.tensor([[[0.0, 0.0, 0.4, 0.6]]]),
        "gt_boxes_pixel": torch.tensor([[[38.0, 26.0, 51.0, 77.0]]]),
        "gt_box_labeled": torch.ones(1, 1),
        "img_hw": torch.tensor([[128.0, 128.0]]),
    }

    losses = pipeline._losses_for_batch(batch)

    assert torch.isfinite(losses["total"])
    losses["total"].backward()

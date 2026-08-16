"""Contract tests for the canonical tensor-native box-diffusion pipeline."""

from __future__ import annotations

import pytest
import torch

from src.models.features.cnn import CNNFeatureExtractor, CNNFeatureExtractorConfig
from src.models.pipelines.object_detectors.box_diffusion import (
    BoxDiffusionPipeline,
    BoxDiffusionPipelineConfig,
    _image_tokens_and_centers,
)


N_SLOTS = 4
FEATURE_CHANNELS = 16


def _make_pipeline(**overrides) -> BoxDiffusionPipeline:
    defaults = dict(
        backbone=CNNFeatureExtractor(out_channels=FEATURE_CHANNELS),
        num_slots=N_SLOTS,
        dim=32,
        num_heads=4,
        num_blocks=2,
        time_dim=32,
        image_token_dim=FEATURE_CHANNELS,
        fourier_box_bands=4,
        fourier_sigma_bands=4,
        num_inference_steps=3,
    )
    defaults.update(overrides)
    return BoxDiffusionPipeline(**defaults)


def _fake_batch(batch_size: int = 2, max_people: int = 3) -> dict:
    boxes = torch.zeros(batch_size, max_people, 4)
    labeled = torch.zeros(batch_size, max_people)
    for index in range(batch_size):
        boxes[index, : index + 1] = torch.tensor([0.0, 0.0, 0.4, 0.4])
        labeled[index, : index + 1] = 1.0
    return {
        "image": torch.rand(batch_size, 3, 32, 32),
        "gt_boxes_canon": boxes,
        "gt_box_labeled": labeled,
        "img_hw": torch.full((batch_size, 2), 32.0),
        "image_id": torch.arange(1, batch_size + 1),
    }


def test_pipeline_config_builds_complete_model():
    config = BoxDiffusionPipelineConfig(
        backbone=CNNFeatureExtractorConfig(out_channels=FEATURE_CHANNELS),
        num_slots=N_SLOTS,
        dim=32,
        num_heads=4,
        num_blocks=1,
        time_dim=32,
        obj_dim=32,
        box_dim=32,
        fourier_box_bands=4,
        fourier_sigma_bands=4,
    )
    pipeline = config.build()
    assert isinstance(pipeline, BoxDiffusionPipeline)
    assert isinstance(pipeline.backbone, CNNFeatureExtractor)
    assert pipeline.num_slots == N_SLOTS


def test_image_tokens_accept_spatial_maps_and_token_sequences():
    spatial_tokens, centers = _image_tokens_and_centers(
        {"spatial": torch.randn(2, FEATURE_CHANNELS, 4, 5), "patch_centers": torch.randn(4, 5, 2)},
        "spatial",
    )
    sequence_tokens, sequence_centers = _image_tokens_and_centers(
        {"spatial": torch.randn(2, 7, FEATURE_CHANNELS), "patch_centers": torch.randn(7, 2)},
        "spatial",
    )
    assert spatial_tokens.shape == (2, 20, FEATURE_CHANNELS)
    assert centers.shape == (20, 2)
    assert sequence_tokens.shape == (2, 7, FEATURE_CHANNELS)
    assert sequence_centers.shape == (7, 2)


def test_forward_returns_finite_canonical_box_and_objectness_tensors():
    pipeline = _make_pipeline()
    images = torch.rand(2, 3, 32, 32)
    noisy_boxes = torch.randn(2, N_SLOTS, 4)
    output = pipeline(images, noisy_boxes, torch.tensor([0.1, 0.5]))
    assert set(output) == {"pred_boxes", "obj_logits"}
    assert output["pred_boxes"].shape == noisy_boxes.shape
    assert output["obj_logits"].shape == noisy_boxes.shape[:2]
    assert torch.isfinite(output["pred_boxes"]).all()
    assert torch.isfinite(output["obj_logits"]).all()


def test_forward_rejects_wrong_slot_count():
    pipeline = _make_pipeline()
    with pytest.raises(ValueError, match="expected N"):
        pipeline(torch.rand(1, 3, 32, 32), torch.randn(1, 2, 4), torch.tensor([0.1]))


def test_edm_schedule_has_expected_endpoints():
    schedule = BoxDiffusionPipeline.edm_sigma_schedule(8, sigma_min=0.01, sigma_max=1.0)
    assert schedule.shape == (9,)
    assert torch.isclose(schedule[0], torch.tensor(1.0))
    assert torch.isclose(schedule[-2], torch.tensor(0.01))
    assert schedule[-1].item() == 0.0


def test_sample_returns_finite_outputs_for_single_and_multi_step_sampling():
    pipeline = _make_pipeline()
    images = torch.rand(2, 3, 32, 32)
    for steps in (1, 3):
        output = pipeline.sample(images, num_inference_steps=steps)
        assert output["pred_boxes"].shape == (2, N_SLOTS, 4)
        assert output["obj_logits"].shape == (2, N_SLOTS)
        assert torch.isfinite(output["pred_boxes"]).all()


def test_sample_rejects_zero_inference_steps():
    with pytest.raises(ValueError, match="num_inference_steps"):
        _make_pipeline().sample(torch.rand(1, 3, 32, 32), num_inference_steps=0)


def test_sample_is_deterministic_with_generator_and_initial_boxes():
    pipeline = _make_pipeline()
    images = torch.rand(2, 3, 32, 32)
    generator_one = torch.Generator().manual_seed(123)
    generator_two = torch.Generator().manual_seed(123)
    first = pipeline.sample(images, generator=generator_one)
    second = pipeline.sample(images, generator=generator_two)
    assert torch.allclose(first["pred_boxes"], second["pred_boxes"])
    fixed = pipeline.sample(images, x_init=torch.zeros(2, N_SLOTS, 4))
    assert fixed["pred_boxes"].shape == (2, N_SLOTS, 4)


def test_training_step_returns_finite_differentiable_loss():
    pipeline = _make_pipeline()
    loss = pipeline.training_step(_fake_batch(), batch_idx=0)
    assert torch.isfinite(loss)
    assert loss.requires_grad


def test_training_step_supports_augmentations_and_disabled_self_conditioning():
    pipeline = _make_pipeline(
        self_cond=False,
        noise_std=0.05,
        brightness=0.1,
        contrast=0.1,
        saturation=0.1,
        hflip_p=1.0,
    )
    assert torch.isfinite(pipeline.training_step(_fake_batch(), batch_idx=0))


def test_configure_optimizers_returns_scheduler_configuration():
    output = _make_pipeline(lr_T_max=5).configure_optimizers()
    assert isinstance(output, dict)
    assert "optimizer" in output
    assert "lr_scheduler" in output

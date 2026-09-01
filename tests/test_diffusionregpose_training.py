from __future__ import annotations

import random

from pathlib import Path
from types import SimpleNamespace

import torch
import pytest
from hydra import compose, initialize_config_dir
from src.data import DataConfig
from src.data.data_module import (
    CrowdPoseDataModule,
    _boxes_to_image_canonical_batch,
    _external_eval_transform,
    _external_train_transform,
    _normalize_imagenet_image,
    _pad_variable_images,
)
from src.training.optim import build_adamw_with_schedule

from src.models.pipelines.pose_estimators.diffusionregpose.training import (
    CROWDPOSE_PERSON_LABEL,
    DiffusionRegPoseCriterion,
    DiffusionRegPoseMatcher,
    DiffusionRegPosePipeline,
    DiffusionRegPosePipelineConfig,
    _generalized_box_iou,
    build_diffusionregpose_targets,
    diffusionregpose_ap_epoch_active,
    update_diffusionregpose_ap_metric,
    update_diffusionregpose_bbox_ap_metric,
)
from src.models.pipelines.pose_estimators.diffusionregpose.model import DiffusionRegPoseModel


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


def test_diffusionregpose_giou_matches_reference_epsilon_formula() -> None:
    boxes1 = torch.tensor([[0.01, 0.02, 0.08, 0.11]], requires_grad=True)
    boxes2 = torch.tensor([[0.03, 0.01, 0.09, 0.10]])

    actual = _generalized_box_iou(boxes1, boxes2)
    intersection = torch.tensor(0.05 * 0.08)
    union = torch.tensor(0.07 * 0.09 + 0.06 * 0.09) - intersection
    enclosing_area = torch.tensor(0.08 * 0.10)
    expected = intersection / (union + 1e-6) - (
        enclosing_area - union
    ) / (enclosing_area + 1e-6)

    assert torch.allclose(actual, expected.unsqueeze(0), atol=1e-7)
    actual.sum().backward()
    assert boxes1.grad is not None
    assert torch.isfinite(boxes1.grad).all()


def test_diffusionregpose_config_accepts_data_subset_override() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "conf"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        config = compose(
            config_name="benchmark_diffusionregpose",
            overrides=["data.subset_frac=0.001"],
        )

    assert config.data.subset_frac == 0.001
    assert config.data.batch_size == 1
    assert config.training.accumulate_grad_batches == 8
    assert config.training.max_epochs == 80
    assert config.training.grad_clip == 0.1
    assert config.training.limit_val_batches == 1.0
    assert config.training.checkpoint.save_last_every_epoch is True
    assert config.pipeline.val_ap_every_n_epochs == 10
    assert config.pipeline.pretrained_checkpoint.endswith(
        "external/DiffusionRegPose/pretrain_models/edpose_r50_crowdpose.pth"
    )


def test_diffusionregpose_ap_cadence_uses_completed_epoch_numbers() -> None:
    assert diffusionregpose_ap_epoch_active(9, 40, 10)
    assert not diffusionregpose_ap_epoch_active(10, 40, 10)
    assert diffusionregpose_ap_epoch_active(39, 40, 10)
    assert diffusionregpose_ap_epoch_active(2, 3, 10)


def test_diffusionregpose_ap_update_filters_person_predictions_and_scales_pixels() -> None:
    class Metric:
        def update(self, pred_kpts, pred_scores, image_ids, bbox_scores=None):
            self.pred_kpts = pred_kpts
            self.pred_scores = pred_scores
            self.image_ids = image_ids
            self.bbox_scores = bbox_scores

    keypoints = torch.full((1, 2, 42), 0.5)
    keypoints[..., :14] = torch.linspace(0.2, 0.8, 14)
    keypoints[..., 14:28] = torch.linspace(0.2, 0.8, 14)
    outputs = {
        "pred_logits": torch.tensor([[[0.0, 10.0], [0.0, -10.0]]]),
        "pred_boxes": torch.full((1, 2, 4), 0.5),
        "pred_keypoints": keypoints,
    }
    batch = {
        "image": torch.zeros(1, 3, 100, 100),
        "img_hw": torch.tensor([[100.0, 100.0]]),
        "image_id": torch.tensor([7]),
    }
    metric = Metric()

    update_diffusionregpose_ap_metric(metric, outputs, batch)

    assert metric.pred_kpts.shape == (1, 1, 14, 3)
    coordinate_values = torch.cat((
        torch.linspace(20.0, 80.0, 14),
        torch.linspace(20.0, 80.0, 14),
    ))
    expected_coordinates = torch.stack((
        coordinate_values[::2], coordinate_values[1::2],
    ), dim=-1).view(1, 1, 14, 2)
    assert torch.allclose(metric.pred_kpts[..., :2], expected_coordinates)
    assert torch.allclose(metric.pred_kpts[..., 2], torch.full((1, 1, 14), 0.5))
    assert metric.pred_scores.shape == (1, 1, 14)
    assert metric.image_ids.tolist() == [7]
    assert metric.bbox_scores.shape == (1, 1)


def test_diffusionregpose_ap_uses_original_and_resized_dimensions_separately(monkeypatch) -> None:
    captured = {}

    def fake_postprocess(outputs, target_sizes, *, num_body_points, nms_target_sizes):
        del outputs, num_body_points
        captured["target_sizes"] = target_sizes
        captured["nms_target_sizes"] = nms_target_sizes
        return [{
            "labels": torch.tensor([1]),
            "keypoints": torch.zeros(1, 42),
            "scores": torch.ones(1),
        }]

    class Metric:
        def update(self, *args, **kwargs):
            del args, kwargs

    monkeypatch.setattr(
        "src.models.pipelines.pose_estimators.diffusionregpose.training.postprocess_crowdpose",
        fake_postprocess,
    )
    from src.models.pipelines.pose_estimators.diffusionregpose.training import (
        update_diffusionregpose_ap_metric,
    )

    update_diffusionregpose_ap_metric(
        Metric(),
        {"pred_logits": torch.zeros(1, 1, 2)},
        {
            "image": torch.zeros(1, 3, 80, 120),
            "original_img_hw": torch.tensor([[200.0, 300.0]]),
            "img_hw": torch.tensor([[80.0, 120.0]]),
            "image_id": torch.tensor([7]),
        },
    )

    assert torch.equal(captured["target_sizes"], torch.tensor([[200.0, 300.0]]))
    assert torch.equal(captured["nms_target_sizes"], torch.tensor([[80.0, 120.0]]))


def test_diffusionregpose_bbox_ap_uses_shared_pose_nms_results() -> None:
    class Metric:
        def update(self, boxes, scores, labels, image_ids):
            self.boxes = boxes
            self.scores = scores
            self.labels = labels
            self.image_ids = image_ids

    metric = Metric()
    update_diffusionregpose_bbox_ap_metric(
        metric,
        [{
            "labels": torch.tensor([1, 0]),
            "boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0], [0.0, 0.0, 1.0, 1.0]]),
            "scores": torch.tensor([0.9, 0.8]),
            "keypoints": torch.zeros(2, 42),
        }],
        {
            "image": torch.zeros(1, 3, 80, 120),
            "original_img_hw": torch.tensor([[200.0, 300.0]]),
            "img_hw": torch.tensor([[80.0, 120.0]]),
            "image_id": torch.tensor([7]),
        },
    )

    assert metric.boxes.tolist() == [[[10.0, 20.0, 30.0, 40.0]]]
    torch.testing.assert_close(metric.scores, torch.tensor([[0.9]]))
    assert metric.labels.tolist() == [[1]]
    assert metric.image_ids.tolist() == [7]


def test_reduced_pipeline_runs_validation_ap_on_configured_cadence(monkeypatch) -> None:
    class Metric:
        def __init__(self, ann_file):
            self.ann_file = ann_file
            self.update_count = 0
            self.reset_count = 0

        def update(self, *args, **kwargs):
            del args, kwargs
            self.update_count += 1

        def compute(self):
            return {"crowdpose/AP": 0.25}

        def reset(self):
            self.reset_count += 1

    class BBoxMetric(Metric):
        def update(self, boxes, scores, labels, image_ids):
            super().update(boxes, scores, labels, image_ids)
            self.box_counts = getattr(self, "box_counts", []) + [boxes.shape[1]]

        def compute(self):
            return {"crowdpose/bbox/AP": 0.5}

    monkeypatch.setattr(
        "src.models.pipelines.pose_estimators.diffusionregpose.training.DiffusionRegPoseAPMetric",
        Metric,
    )
    monkeypatch.setattr(
        "src.models.pipelines.pose_estimators.diffusionregpose.training.DiffusionRegPoseBBoxAPMetric",
        BBoxMetric,
    )
    postprocess_calls = []

    def fake_postprocess(outputs, target_sizes, *, num_body_points, nms_target_sizes):
        del outputs, target_sizes, num_body_points, nms_target_sizes
        postprocess_calls.append(None)
        return [{
            "labels": torch.tensor([1]),
            "boxes": torch.full((1, 4), 0.5),
            "scores": torch.ones(1),
            "keypoints": torch.zeros(1, 42),
        }]

    monkeypatch.setattr(
        "src.models.pipelines.pose_estimators.diffusionregpose.training.postprocess_crowdpose",
        fake_postprocess,
    )
    pipeline = DiffusionRegPosePipelineConfig(
        backbone_pretrained=False,
        num_queries=4,
        num_proposals=2,
        num_group=2,
        num_feature_levels=3,
        num_encoder_layers=1,
        num_decoder_layers=3,
        val_ap_every_n_epochs=2,
    ).build(max_epochs=2, val_ann_file="validation.json")
    pipeline._trainer = SimpleNamespace(
        current_epoch=0, max_epochs=2, sanity_checking=False,
    )
    pipeline._losses_for_batch = lambda batch: {"total": torch.zeros(())}
    pipeline._log_losses = lambda *args, **kwargs: None
    logged = {}
    pipeline.log = lambda name, value, **kwargs: logged.setdefault(name, kwargs)
    keypoints = torch.full((1, 2, 42), 0.5)
    keypoints[..., :14] = torch.linspace(0.2, 0.8, 14)
    keypoints[..., 14:28] = torch.linspace(0.2, 0.8, 14)
    outputs = {
        "pred_logits": torch.tensor([[[0.0, 10.0], [0.0, -10.0]]]),
        "pred_boxes": torch.full((1, 2, 4), 0.5),
        "pred_keypoints": keypoints,
    }
    seen_padding_masks = []

    def sample_keypoints(images, *, padding_mask=None):
        del images
        seen_padding_masks.append(padding_mask)
        return outputs

    pipeline.model.sample_keypoints = sample_keypoints
    batch = {
        "image": torch.zeros(1, 3, 100, 100),
        "img_hw": torch.tensor([[100.0, 100.0]]),
        "padding_mask": torch.zeros(1, 100, 100, dtype=torch.bool),
        "image_id": torch.tensor([7]),
    }

    pipeline._trainer.sanity_checking = True
    pipeline._trainer.current_epoch = 1
    pipeline.validation_step(batch, 0)
    assert pipeline._val_ap.update_count == 0

    pipeline._trainer.sanity_checking = False
    pipeline._trainer.current_epoch = 1
    pipeline.validation_step(batch, 0)
    pipeline.on_validation_epoch_end()

    assert pipeline._val_ap.update_count == 1
    assert pipeline._val_ap.reset_count == 1
    assert pipeline._val_bbox_ap.update_count == 1
    assert pipeline._val_bbox_ap.reset_count == 1
    assert pipeline._val_bbox_ap.box_counts == [1]
    assert logged["ap-val/crowdpose/AP"]["prog_bar"] is True
    assert logged["ap-val/crowdpose/bbox/AP"]["prog_bar"] is False
    assert seen_padding_masks == [batch["padding_mask"]]
    assert len(postprocess_calls) == 1


def test_diffusionregpose_debug_config_is_small_and_deterministic() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "conf"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        config = compose(config_name="benchmark_diffusionregpose_debug")

    assert config.data.subset_frac == 0.0032
    assert config.data.batch_size == 1
    assert config.pipeline.backbone_pretrained is True
    assert config.pipeline.hidden_dim == 256
    assert config.pipeline.num_queries == 100
    assert config.pipeline.num_proposals == 10
    assert config.pipeline.num_group == 10
    assert config.pipeline.val_ap_every_n_epochs == 10
    assert config.training.max_epochs == 100
    assert config.training.test_after_fit is False


def test_diffusionregpose_fast_config_keeps_full_checkpoint_structure() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "conf"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        config = compose(config_name="benchmark_diffusionregpose_fast")

    assert config.pipeline.num_queries == 900
    assert config.pipeline.num_proposals == 100
    assert config.pipeline.num_box_decoder_layers == 2
    assert config.pipeline.num_feature_levels == 4
    assert config.data.subset_frac == 0.0032
    assert config.data.train_resize_scales == [256]


def test_diffusionregpose_lowres_config_uses_full_data_and_128_pixel_eval() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "conf"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        config = compose(config_name="benchmark_diffusionregpose_lowres")

    assert config.data.subset_frac == 1.0
    assert config.data.train_resize_scales == [128]
    assert config.data.train_max_size == 192
    assert config.data.eval_resize_scale == 128
    assert config.data.eval_max_size == 192
    assert config.data.batch_size == 8
    data_config = DataConfig.from_hydra_config(config.data)
    data_module = data_config.build()
    assert data_module.eval_resize_scale == 128
    assert data_module.eval_max_size == 192
    assert config.pipeline.num_queries == 900
    assert config.pipeline.val_ap_every_n_epochs == 10
    assert config.pipeline.lr_decay_milestones == [31, 61]
    assert config.training.max_epochs == 80
    assert config.training.accumulate_grad_batches == 1
    assert config.training.test_after_fit is False


def test_external_eval_transform_uses_configured_resize() -> None:
    transformed = _external_eval_transform(
        torch.ones(3, 100, 200),
        torch.zeros(1, 14, 2),
        torch.ones(1, 14),
        torch.ones(1, 14),
        torch.zeros(1, 4),
        torch.ones(1),
        short_side=128,
        max_size=192,
    )

    assert transformed[0].shape[-2:] == (96, 192)


def test_external_images_are_normalized_before_variable_size_padding() -> None:
    first = torch.full((3, 2, 3), 0.5)
    second = torch.full((3, 3, 2), 0.25)

    padded, padding_mask = _pad_variable_images([
        _normalize_imagenet_image(first),
        _normalize_imagenet_image(second),
    ])

    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    expected_first = (first - mean) / std
    expected_second = (second - mean) / std
    torch.testing.assert_close(padded[0, :, :2, :3], expected_first)
    torch.testing.assert_close(padded[1, :, :3, :2], expected_second)
    assert torch.equal(padded.masked_select(padding_mask[:, None]).unique(), torch.zeros(1))


def test_external_crop_keeps_diffusion_keypoint_rows_after_clean_target_filter(monkeypatch) -> None:
    monkeypatch.setattr("src.data.data_module.random.random", lambda: 1.0)
    monkeypatch.setattr("src.data.data_module.random.choice", lambda values: values[0])
    monkeypatch.setattr("src.data.data_module.random.randint", lambda lower, upper: lower)
    image = torch.zeros(3, 128, 128)
    keypoints = torch.tensor([
        [[-0.75, -0.75], [-0.5, -0.5]],
        [[0.75, 0.75], [0.5, 0.5]],
    ])
    labeled = torch.ones(2, 2)
    visible = torch.ones(2, 2)
    boxes = torch.tensor([[0.0, 0.0, 20.0, 20.0], [90.0, 90.0, 20.0, 20.0]])
    box_labeled = torch.ones(2)

    transformed = _external_train_transform(
        image,
        keypoints,
        labeled,
        visible,
        boxes,
        box_labeled,
        diffusion_keypoints_canon=keypoints,
        scales=(128,),
        max_size=192,
        scales2_resize=(128,),
        scales2_crop=(64, 64),
    )
    _, clean_keypoints, diffusion_keypoints, clean_labeled, _, _, _, _, _ = transformed

    assert clean_labeled[0].all()
    assert not clean_labeled[1].any()
    assert diffusion_keypoints.shape[0] == 2
    torch.testing.assert_close(diffusion_keypoints[1], torch.ones_like(clean_keypoints[1]))


def test_external_train_transform_can_disable_random_augmentation() -> None:
    image = torch.arange(3 * 40 * 60, dtype=torch.float32).reshape(3, 40, 60)
    keypoints = torch.tensor([[[0.0, 0.0], [0.5, -0.5]]])
    labeled = torch.ones(1, 2)
    visible = torch.ones(1, 2)
    boxes = torch.tensor([[10.0, 10.0, 20.0, 10.0]])
    box_labeled = torch.ones(1)

    first = _external_train_transform(
        image, keypoints, labeled, visible, boxes, box_labeled,
        scales=(32,), max_size=64, scales2_resize=(16,), scales2_crop=(8, 16),
        random_flip=False, random_crop=False,
    )
    second = _external_train_transform(
        image, keypoints, labeled, visible, boxes, box_labeled,
        scales=(32,), max_size=64, scales2_resize=(16,), scales2_crop=(8, 16),
        random_flip=False, random_crop=False,
    )
    for first_value, second_value in zip(first, second):
        torch.testing.assert_close(first_value, second_value)


def test_external_train_transform_samples_resize_scale_when_crop_is_disabled() -> None:
    class ChooseLast:
        def random(self) -> float:
            return 0.0

        def choice(self, values):
            return values[-1]

    image = torch.zeros(3, 40, 60)
    keypoints = torch.zeros(1, 2, 2)
    labeled = torch.ones(1, 2)
    visible = torch.ones(1, 2)
    boxes = torch.tensor([[10.0, 10.0, 20.0, 10.0]])
    box_labeled = torch.ones(1)

    transformed = _external_train_transform(
        image, keypoints, labeled, visible, boxes, box_labeled,
        scales=(32, 48), max_size=128, scales2_resize=(16,), scales2_crop=(8, 16),
        random_flip=False, random_crop=False, rng=ChooseLast(),
    )

    assert transformed[0].shape[-2:] == (48, 72)


def test_external_train_transform_replays_seeded_augmentation() -> None:
    image = torch.zeros(3, 80, 100)
    keypoints = torch.tensor([[[0.0, 0.0], [0.5, -0.5]]])
    labeled = torch.ones(1, 2)
    visible = torch.ones(1, 2)
    boxes = torch.tensor([[10.0, 10.0, 40.0, 30.0]])
    box_labeled = torch.ones(1)
    kwargs = dict(
        scales=(64,), max_size=96, scales2_resize=(64,), scales2_crop=(32, 64),
        random_flip=True, random_crop=True,
    )
    first = _external_train_transform(
        image, keypoints, labeled, visible, boxes, box_labeled,
        rng=random.Random(17), **kwargs,
    )
    second = _external_train_transform(
        image, keypoints, labeled, visible, boxes, box_labeled,
        rng=random.Random(17), **kwargs,
    )
    for first_value, second_value in zip(first, second):
        torch.testing.assert_close(first_value, second_value)


def test_external_crop_updates_person_mask_for_removed_people(monkeypatch) -> None:
    module = CrowdPoseDataModule(
        train_ann=Path("train.json"), val_ann=Path("val.json"),
        test_ann=Path("test.json"), img_dir=Path("images"),
        val_pred_cache=None, test_pred_cache=None, batch_size=1,
        num_workers=0, ap_batch_size=1, target_format="set",
        external_train_transforms=True,
    )
    module.trainer = SimpleNamespace(training=True)

    def fake_transform(*args, **kwargs):
        del kwargs
        image, keypoints, labeled, visible, boxes, box_labeled = args[:6]
        return (
            image, keypoints, keypoints.clone(), labeled, visible, boxes,
            torch.tensor([1.0, 0.0]), image.new_tensor([100.0, 100.0]),
            torch.ones(2),
        )

    monkeypatch.setattr("src.data.data_module._external_train_transform", fake_transform)
    batch = {
        "image": [torch.zeros(3, 100, 100)],
        "gt_keypoints": torch.zeros(1, 2, 14, 2),
        "gt_labeled": torch.ones(1, 2, 14),
        "gt_visible": torch.ones(1, 2, 14),
        "person_scale": torch.ones(1, 2),
        "person_mask": torch.ones(1, 2),
        "gt_boxes_pixel": torch.ones(1, 2, 4),
        "gt_box_labeled": torch.ones(1, 2),
        "img_hw": torch.tensor([[100.0, 100.0]]),
    }

    transformed = module.on_after_batch_transfer(batch, 0)

    assert transformed["person_mask"].tolist() == [[1.0, 0.0]]


def test_diffusion_targets_use_unfiltered_diffusion_keypoint_stream() -> None:
    batch = _batch()
    batch["gt_keypoints"] = torch.tensor([[[[-0.5, -0.5]], [[0.5, 0.5]]]])
    batch["gt_diffusion_keypoints"] = torch.tensor([[[[-0.5, -0.5]], [[0.5, 0.5]]]])
    batch["gt_labeled"] = torch.tensor([[[1.0], [0.0]]])
    batch["gt_visible"] = torch.tensor([[[1.0], [0.0]]])
    batch["person_mask"] = torch.tensor([[1.0, 1.0]])
    batch["gt_boxes_canon"] = torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]]])
    diffusion = type("Diffusion", (), {
        "scale": 6.0,
        "q_sample": lambda self, x_start, timesteps, noise: x_start,
    })()

    diffusion_inputs, _, targets = build_diffusionregpose_targets(
        batch,
        diffusion,
        num_proposals=2,
        num_body_points=1,
        noise=torch.zeros(1, 2, 2),
    )

    assert targets[0]["labels"].numel() == 1
    torch.testing.assert_close(
        diffusion_inputs,
        torch.tensor([[[-3.0, -3.0], [3.0, 3.0]]]),
    )


def test_diffusion_targets_reuse_seeded_noise() -> None:
    batch = _batch()
    diffusion = type("Diffusion", (), {
        "scale": 6.0,
        "q_sample": lambda self, x_start, timesteps, noise: x_start + noise,
    })()

    first, first_timesteps, _ = build_diffusionregpose_targets(
        batch, diffusion, num_proposals=3, num_body_points=2, noise_seed=17,
    )
    second, second_timesteps, _ = build_diffusionregpose_targets(
        batch, diffusion, num_proposals=3, num_body_points=2, noise_seed=17,
    )
    torch.testing.assert_close(first, second)
    assert torch.equal(first_timesteps, second_timesteps)


def test_external_compatible_model_has_reference_trainable_parameter_count() -> None:
    model = DiffusionRegPoseModel(backbone_pretrained=False)

    assert sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) == 49_304_924


def test_diffusionregpose_targets_use_crowdpose_person_label() -> None:
    batch = _batch()
    diffusion = type("Diffusion", (), {
        "scale": 6.0,
        "q_sample": lambda self, x_start, timesteps, noise: x_start,
    })()

    _, _, targets = build_diffusionregpose_targets(
        batch, diffusion, num_proposals=2, num_body_points=2,
        noise=torch.zeros(1, 2, 4),
    )

    assert targets[0]["labels"].tolist() == [CROWDPOSE_PERSON_LABEL]


def test_diffusionregpose_rejects_missing_person_class() -> None:
    try:
        DiffusionRegPosePipelineConfig(num_classes=1)
    except ValueError as error:
        assert "person label" in str(error)
    else:
        raise AssertionError("num_classes=1 must not omit the person class")


def test_diffusionregpose_uses_paper_loss_and_schedule_defaults() -> None:
    config = DiffusionRegPosePipelineConfig()

    assert config.weights.keypoints == 10.0
    assert config.lr == 2e-4
    assert config.lr_warmup_epochs == 0
    assert config.scheduler_type == "multistep"
    assert config.lr_decay_milestones == (31, 61)
    assert config.dn_number == 100
    assert config.dn_box_noise_scale == 0.4
    assert config.dn_label_noise_ratio == 0.5


def test_multistep_scheduler_decays_at_paper_milestone() -> None:
    schedule = build_adamw_with_schedule(
        torch.nn.Linear(2, 2),
        lr=2e-4,
        weight_decay=1e-4,
        scheduler_type="multistep",
        lr_decay_milestones=(31, 61),
    )
    optimizer = schedule["optimizer"]
    scheduler = schedule["lr_scheduler"]["scheduler"]

    assert optimizer.param_groups[0]["lr"] == 2e-4
    for _ in range(31):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == 2e-5

    for _ in range(35):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(2e-6)


def test_diffusionregpose_criterion_scores_person_class_column() -> None:
    criterion = DiffusionRegPoseCriterion(
        num_classes=2,
        num_body_points=2,
        num_box_decoder_layers=1,
    )
    targets = [{
        "labels": torch.tensor([CROWDPOSE_PERSON_LABEL]),
        "boxes": torch.tensor([[0.0, 0.0, 0.2, 0.4]]),
        "keypoints": torch.tensor([[0.0, 0.0, 0.1, 0.1, 1.0, 1.0]]),
        "area": torch.tensor([0.08]),
    }]

    def classification_loss(logits: torch.Tensor) -> torch.Tensor:
        outputs = {
            "pred_logits": logits,
            "pred_boxes": torch.tensor([[[0.0, 0.0, 0.2, 0.4]]]),
            "pred_keypoints": torch.tensor([[[0.0, 0.0, 0.1, 0.1, 1.0, 1.0]]]),
        }
        return criterion(outputs, targets)["loss_ce"]

    person_logit_high = classification_loss(torch.tensor([[[0.0, 8.0]]]))
    background_logit_high = classification_loss(torch.tensor([[[8.0, 0.0]]]))

    assert person_logit_high < background_logit_high


def test_diffusionregpose_classification_loss_has_one_query_scaling() -> None:
    criterion = DiffusionRegPoseCriterion(
        num_classes=2,
        num_body_points=2,
        num_box_decoder_layers=1,
    )
    logits = torch.zeros(1, 3, 2)
    outputs = {
        "pred_logits": logits,
        "pred_boxes": torch.tensor([[[0.0, 0.0, 0.2, 0.4]] * 3]),
        "pred_keypoints": torch.tensor([[[0.0, 0.0, 0.1, 0.1, 1.0, 1.0]] * 3]),
    }
    targets = [{
        "labels": torch.tensor([CROWDPOSE_PERSON_LABEL]),
        "boxes": torch.tensor([[0.0, 0.0, 0.2, 0.4]]),
        "keypoints": torch.tensor([[0.0, 0.0, 0.1, 0.1, 1.0, 1.0]]),
        "area": torch.tensor([0.08]),
    }]

    probability = logits.sigmoid()
    target_classes = torch.zeros_like(logits)
    target_classes[0, 0, CROWDPOSE_PERSON_LABEL] = 1.0
    cross_entropy = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, target_classes, reduction="none"
    )
    p_t = probability * target_classes + (1 - probability) * (1 - target_classes)
    alpha_t = 0.25 * target_classes + 0.75 * (1 - target_classes)
    expected = (alpha_t * cross_entropy * (1 - p_t).pow(2)).mean(1).sum() * 3

    torch.testing.assert_close(criterion(outputs, targets)["loss_ce"], expected)


def test_diffusionregpose_classification_diagnostics_split_clean_focal_loss() -> None:
    criterion = DiffusionRegPoseCriterion(
        num_body_points=2,
        num_box_decoder_layers=1,
    )
    outputs = {
        "pred_logits": torch.tensor([[[0.0, 3.0], [0.0, -2.0]]]),
        "pred_boxes": torch.tensor([[[0.0, 0.0, 0.2, 0.4], [0.5, 0.5, 0.2, 0.4]]]),
        "pred_keypoints": torch.tensor([[[0.0, 0.0, 0.1, 0.1, 1.0, 1.0]] * 2]),
    }
    targets = [{
        "labels": torch.tensor([CROWDPOSE_PERSON_LABEL]),
        "boxes": torch.tensor([[0.0, 0.0, 0.2, 0.4]]),
        "keypoints": torch.tensor([[0.0, 0.0, 0.1, 0.1, 1.0, 1.0]]),
        "area": torch.tensor([0.08]),
    }]

    losses = criterion(outputs, targets)

    torch.testing.assert_close(
        losses["loss_ce_positive"] + losses["loss_ce_negative"],
        losses["loss_ce"],
    )
    assert losses["matched_queries"].item() == 1
    assert losses["person_score_matched"] > losses["person_score_unmatched"]


def test_diffusionregpose_logs_classification_diagnostics_under_debug_namespace() -> None:
    logged = {}
    module = SimpleNamespace(
        log=lambda name, value, **kwargs: logged.setdefault(name, (value, kwargs)),
    )
    losses = {
        "total": torch.tensor(2.0),
        "loss_ce": torch.tensor(1.0),
        "loss_ce_positive": torch.tensor(0.25),
        "loss_ce_negative": torch.tensor(0.75),
        "person_score_matched": torch.tensor(0.8),
        "person_score_unmatched": torch.tensor(0.2),
        "matched_queries": torch.tensor(3.0),
    }

    DiffusionRegPosePipeline._log_losses(module, "train", losses, batch_size=2)

    assert "train/loss" in logged
    assert "train/loss_ce" in logged
    assert "debug/train/loss_ce_positive" in logged
    assert "debug/train/loss_ce_negative" in logged
    assert "debug/train/person_score_matched" in logged
    assert "debug/train/person_score_unmatched" in logged
    assert "debug/train/matched_queries" in logged


def test_diffusionregpose_denoising_classification_uses_external_sentinel_label() -> None:
    criterion = DiffusionRegPoseCriterion(num_body_points=2, num_box_decoder_layers=1)
    logits = torch.tensor([[[0.1, -0.2], [0.3, 0.4], [-0.5, 0.6]]])
    labels = torch.tensor([[CROWDPOSE_PERSON_LABEL, 2, 0]])
    outputs = {
        "dn_class_pred": logits,
        "dn_class_input": labels,
        "dn_bbox_pred": torch.full((1, 3, 4), 0.5),
        "dn_bbox_input": torch.full((1, 3, 4), 0.5),
    }

    actual = criterion._denoising_losses(outputs)["dn_loss_ce"]
    # The external criterion creates C+1 classes then drops its C/no-object
    # sentinel. Label 2 therefore becomes an all-zero binary target.
    targets = torch.zeros(1, 3, 3)
    targets.scatter_(2, labels.unsqueeze(-1), 1)
    targets = targets[..., :-1]
    probabilities = logits.sigmoid()
    cross_entropy = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none",
    )
    p_t = probabilities * targets + (1 - probabilities) * (1 - targets)
    alpha_t = 0.25 * targets + 0.75 * (1 - targets)
    expected = (
        (alpha_t * cross_entropy * (1 - p_t).pow(2)).mean(1).sum()
        / labels.shape[1] * logits.shape[1]
    )

    torch.testing.assert_close(actual, expected)


def test_diffusionregpose_auxiliary_losses_use_reference_weights() -> None:
    criterion = DiffusionRegPoseCriterion(
        num_classes=2,
        num_body_points=2,
        num_box_decoder_layers=0,
    )
    output = {
        "pred_logits": torch.tensor([[[0.0, 1.0]]]),
        "pred_boxes": torch.tensor([[[0.2, 0.3, 0.4, 0.5]]]),
        "pred_keypoints": torch.tensor([[[0.2, 0.3, 0.4, 0.5, 1.0, 1.0]]]),
    }
    targets = [{
        "labels": torch.tensor([CROWDPOSE_PERSON_LABEL]),
        "boxes": torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
        "keypoints": torch.tensor([[0.1, 0.2, 0.3, 0.4, 1.0, 1.0]]),
        "area": torch.tensor([0.12]),
    }]
    final_values = criterion._loss_for_outputs(output, targets)
    auxiliary_values = criterion._loss_for_outputs(output, targets)
    losses = criterion({**output, "aux_outputs": [output]}, targets)
    expected = final_values["total"] + criterion._weighted_clean_loss(auxiliary_values)

    torch.testing.assert_close(losses["total"], expected)


def test_diffusionregpose_intermediate_encoder_losses_use_reference_weights() -> None:
    criterion = DiffusionRegPoseCriterion(
        num_classes=2,
        num_body_points=2,
        num_box_decoder_layers=0,
    )
    output = {
        "pred_logits": torch.tensor([[[0.0, 1.0]]]),
        "pred_boxes": torch.tensor([[[0.2, 0.3, 0.4, 0.5]]]),
        "pred_keypoints": torch.tensor([[[0.2, 0.3, 0.4, 0.5, 1.0, 1.0]]]),
    }
    intermediate = {
        "pred_logits": torch.tensor([[[0.3, 0.7]]]),
        "pred_boxes": torch.tensor([[[0.4, 0.3, 0.2, 0.5]]]),
        "pred_keypoints": torch.zeros(1, 1, 6),
    }
    targets = [{
        "labels": torch.tensor([CROWDPOSE_PERSON_LABEL]),
        "boxes": torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
        "keypoints": torch.tensor([[0.1, 0.2, 0.3, 0.4, 1.0, 1.0]]),
        "area": torch.tensor([0.12]),
    }]

    final_values = criterion._loss_for_outputs(output, targets)
    intermediate_values = criterion._loss_for_outputs(
        intermediate, targets, include_keypoints=False,
    )
    losses = criterion({**output, "interm_outputs": intermediate}, targets)
    expected = final_values["total"] + (
        criterion.weights.cls * intermediate_values["loss_ce"]
        + criterion.weights.bbox * intermediate_values["loss_bbox"]
        + criterion.weights.giou * intermediate_values["loss_giou"]
    )

    torch.testing.assert_close(losses["total"], expected)
    torch.testing.assert_close(losses["loss_ce_interm"], intermediate_values["loss_ce"])
    torch.testing.assert_close(losses["loss_bbox_interm"], intermediate_values["loss_bbox"])
    torch.testing.assert_close(losses["loss_giou_interm"], intermediate_values["loss_giou"])


def test_build_targets_preserves_reference_layout_visibility_and_zero_timestep() -> None:
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


def test_build_targets_compacts_cropped_rows_and_supervises_visibility_one() -> None:
    batch = _batch()
    batch["gt_keypoints"] = torch.tensor(
        [[
            [[-0.9, -0.9], [-0.9, -0.9]],
            [[0.2, -0.2], [0.4, 0.0]],
            [[0.8, 0.8], [0.8, 0.8]],
        ]]
    )
    batch["gt_labeled"] = torch.tensor([[[0, 0], [1, 1], [0, 0]]])
    batch["gt_visible"] = torch.zeros(1, 3, 2)
    batch["gt_boxes_canon"] = torch.tensor([[[0.01, 0.01, 0.01, 0.01], [0.2, 0.3, 0.4, 0.5], [0.9, 0.9, 0.01, 0.01]]])
    diffusion = type("Diffusion", (), {
        "scale": 6.0,
        "q_sample": lambda self, x_start, timesteps, noise: x_start,
    })()

    _, _, targets = build_diffusionregpose_targets(
        batch, diffusion, num_proposals=2, num_body_points=2,
        noise=torch.zeros(1, 2, 4),
    )

    torch.testing.assert_close(targets[0]["boxes"], batch["gt_boxes_canon"][:, 1])
    torch.testing.assert_close(
        targets[0]["keypoints"],
        torch.tensor([[0.6, 0.4, 0.7, 0.5, 1.0, 1.0]]),
    )


def test_diffusionregpose_box_targets_use_sigmoid_coordinate_range() -> None:
    boxes = torch.tensor([[[0.0, 0.0, 20.0, 40.0]]])
    img_hw = torch.tensor([[100.0, 200.0]])

    result = _boxes_to_image_canonical_batch(boxes, img_hw)

    torch.testing.assert_close(result, torch.tensor([[[0.05, 0.2, 0.1, 0.4]]]))


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


def test_denoising_queries_and_losses_are_present_during_training() -> None:
    model = DiffusionRegPoseModel(
        backbone_pretrained=False,
        num_queries=4,
        num_proposals=2,
        num_group=2,
        num_feature_levels=3,
        num_encoder_layers=1,
        num_decoder_layers=3,
        num_body_points=2,
        num_box_decoder_layers=1,
        dn_number=4,
    )
    model.train()
    targets = [{
        "labels": torch.tensor([CROWDPOSE_PERSON_LABEL]),
        "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.3]]),
        "keypoints": torch.tensor([[0.5, 0.5, 0.5, 0.5, 1.0, 1.0]]),
        "area": torch.tensor([0.06]),
    }]
    outputs = model(
        torch.rand(1, 3, 64, 64),
        torch.zeros(1, 2, 4),
        torch.zeros(1, dtype=torch.long),
        targets,
    )
    losses = DiffusionRegPoseCriterion(
        num_body_points=2, num_box_decoder_layers=1,
    )(outputs, targets)

    assert outputs["dn_class_pred"].shape == (1, 4, 2)
    assert len(outputs["dn_aux_outputs"]) == 2
    for name in ("dn_loss_ce", "dn_loss_bbox", "dn_loss_giou"):
        assert torch.isfinite(losses[name])
    losses["total"].backward()


def test_optimizer_can_match_external_two_group_weight_decay() -> None:
    module = torch.nn.Module()
    module.main = torch.nn.Linear(2, 2)
    module.model = torch.nn.Module()
    module.model.backbone = torch.nn.Linear(2, 2)

    schedule = build_adamw_with_schedule(
        module,
        lr=2e-4,
        lr_backbone=1e-4,
        weight_decay=1e-4,
        scheduler_type="multistep",
        lr_decay_milestones=(31,),
        split_weight_decay=False,
    )

    groups = schedule["optimizer"].param_groups
    assert len(groups) == 2
    assert {group["lr"] for group in groups} == {1e-4, 2e-4}
    assert all(group["weight_decay"] == 1e-4 for group in groups)


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

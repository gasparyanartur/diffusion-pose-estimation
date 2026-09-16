from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.evaluation.evaluate_diffusionregpose import (
    build_evaluation_model,
    extract_model_state_dict,
)

from src.models.pipelines.pose_estimators.diffusionregpose.evaluation import (
    DiffusionRegPoseAPMetric,
    DiffusionRegPoseBBoxAPMetric,
    crowdpose_result_records,
    evaluate_crowdpose_results,
    postprocess_crowdpose,
    postprocess_crowdpose_step_ensemble,
    resize_crowdpose_eval_image,
)
from src.models.pipelines.pose_estimators.diffusionregpose import DiffusionRegPosePipelineConfig


def test_postprocess_crowdpose_selects_top_class_scores_and_scales_coordinates() -> None:
    outputs = {
        "pred_logits": torch.tensor([[[0.0, 4.0], [5.0, -2.0]]]),
        "pred_boxes": torch.tensor([[[0.5, 0.5, 0.5, 0.25], [0.25, 0.5, 0.2, 0.4]]]),
        "pred_keypoints": torch.tensor(
            [[[0.1, 0.2] * 14 + [0.3] * 14, [0.4, 0.5] * 14 + [0.7] * 14]]
        ),
    }

    result = postprocess_crowdpose(
        outputs, torch.tensor([[100.0, 200.0]]), pose_nms=False
    )[0]

    assert result["scores"].shape == (2,)
    assert result["labels"].tolist() == [0, 1]
    torch.testing.assert_close(result["boxes"][0], torch.tensor([30.0, 30.0, 70.0, 70.0]))
    torch.testing.assert_close(result["keypoints"][0, :3], torch.tensor([80.0, 50.0, 0.7]))
    torch.testing.assert_close(result["keypoints"][0, 3:6], torch.tensor([80.0, 50.0, 0.7]))


def test_postprocess_crowdpose_rejects_inconsistent_shapes() -> None:
    outputs = {
        "pred_logits": torch.zeros(1, 2, 2),
        "pred_boxes": torch.zeros(1, 1, 4),
        "pred_keypoints": torch.zeros(1, 2, 42),
    }

    try:
        postprocess_crowdpose(outputs, torch.tensor([[10.0, 10.0]]))
    except ValueError as error:
        assert "agree" in str(error)
    else:
        raise AssertionError("inconsistent output shapes should raise ValueError")


def test_cross_step_ensemble_selects_candidates_per_step_before_pose_nms() -> None:
    def step(logits: list[list[float]], offset: float) -> dict[str, torch.Tensor]:
        coordinates = torch.tensor(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.2, 0.3, 0.4, 0.5, 0.6,
             0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7],
            dtype=torch.float32,
        )
        keypoints = torch.stack((coordinates + offset, coordinates + offset)).clamp(0.0, 1.0)
        keypoints = torch.cat((keypoints, torch.ones(2, 14)), dim=-1)[None]
        return {
            "pred_logits": torch.tensor([logits]),
            "pred_boxes": torch.tensor(
                [[[0.2, 0.2, 0.1, 0.1], [0.8, 0.8, 0.1, 0.1]]]
            ),
            "pred_keypoints": keypoints,
        }

    # A global top-4 over the concatenated raw logits would retain only step 1.
    # Per-step top-2 must preserve both candidates from step 2 before final NMS.
    results = postprocess_crowdpose_step_ensemble(
        [step([[8.0, 7.0], [6.0, 5.0]], 0.0), step([[1.0, 0.5], [0.0, -0.5]], 0.45)],
        torch.tensor([[100.0, 100.0]]),
        pose_nms=True,
    )

    assert results[0]["scores"].shape == (2,)
    assert (results[0]["scores"] < 0.8).any()


def test_crowdpose_preprocessing_and_result_records_preserve_contract() -> None:
    image = torch.ones(3, 100, 200)
    resized = resize_crowdpose_eval_image(image, shortest_side=80, max_size=120)

    assert resized.shape == (3, 60, 120)
    records = crowdpose_result_records(
        42,
        [{
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([1]),
            "boxes": torch.zeros(1, 4),
            "keypoints": torch.zeros(1, 42),
        }],
    )

    assert records[0]["image_id"] == 42
    assert records[0]["category_id"] == 1
    assert records[0]["bbox"] == [0.0] * 4
    assert records[0]["keypoints"] == [0.0] * 42
    assert records[0]["score"] == torch.tensor(0.9).item()


def test_crowdpose_resize_uses_reference_integer_aspect_ratio() -> None:
    image = torch.ones(3, 100, 333)

    resized = resize_crowdpose_eval_image(image, shortest_side=80, max_size=1333)

    assert resized.shape == (3, 80, 266)

    image = torch.ones(3, 640, 963)
    resized = resize_crowdpose_eval_image(image)

    assert resized.shape == (3, 800, 1203)

    image = torch.ones(3, 284, 500)
    resized = resize_crowdpose_eval_image(image)

    assert resized.shape == (3, 757, 1332)


def test_crowdpose_resize_normalizes_non_rgb_images() -> None:
    grayscale = torch.ones(1, 100, 200)
    rgba = torch.ones(4, 100, 200)

    assert resize_crowdpose_eval_image(grayscale).shape[0] == 3
    assert resize_crowdpose_eval_image(rgba).shape[0] == 3


def test_diffusionregpose_metric_matches_reference_evaluator(tmp_path: Path) -> None:
    annotation_path = Path("tests/fixtures/mini_crowdpose_val.json")
    annotation_data = json.loads(annotation_path.read_text())
    records = []
    metric = DiffusionRegPoseAPMetric(str(annotation_path))

    by_image: dict[int, list[dict]] = {}
    for annotation in annotation_data["annotations"]:
        by_image.setdefault(annotation["image_id"], []).append(annotation)

    for image_id, annotations in by_image.items():
        raw = np.asarray(
            [annotation["keypoints"] for annotation in annotations],
            dtype=np.float32,
        ).reshape(len(annotations), 14, 3)
        keypoints = torch.from_numpy(raw).unsqueeze(0)
        keypoint_scores = keypoints[..., 2]
        bbox_scores = torch.ones(1, len(annotations))
        metric.update(
            keypoints,
            keypoint_scores,
            torch.tensor([image_id]),
            bbox_scores=bbox_scores,
        )
        records.extend(
            {
                "image_id": image_id,
                "category_id": 1,
                "bbox": annotation["bbox"],
                "keypoints": annotation["keypoints"],
                "score": 1.0,
            }
            for annotation in annotations
        )

    result_path = tmp_path / "reference.json"
    result_path.write_text(json.dumps(records))
    expected = evaluate_crowdpose_results(
        result_path,
        annotation_path,
        image_ids=list(by_image),
    )
    actual = metric.compute()

    assert actual["crowdpose/AP"] == pytest.approx(expected["AP"], abs=1e-12)
    assert actual["crowdpose/AP(E)"] == pytest.approx(expected["AP_E"], abs=1e-12)
    assert actual["crowdpose/AP(M)"] == pytest.approx(expected["AP_M"], abs=1e-12)
    assert actual["crowdpose/AP(H)"] == pytest.approx(expected["AP_H"], abs=1e-12)
    assert expected["bbox/AP"] == pytest.approx(1.0, abs=1e-12)


def test_diffusionregpose_bbox_metric_filters_labels_and_matches_perfect_boxes() -> None:
    annotation_path = Path("tests/fixtures/mini_crowdpose_val.json")
    annotation_data = json.loads(annotation_path.read_text())
    metric = DiffusionRegPoseBBoxAPMetric(annotation_path)

    for annotation in annotation_data["annotations"]:
        x, y, width, height = annotation["bbox"]
        metric.update(
            torch.tensor([[[x, y, x + width, y + height], [0.0, 0.0, 1.0, 1.0]]]),
            torch.tensor([[1.0, 1.0]]),
            torch.tensor([[1, 0]]),
            torch.tensor([annotation["image_id"]]),
        )

    assert all(record["category_id"] == 1 for record in metric._records)
    assert metric._records[0]["bbox"] == pytest.approx(annotation_data["annotations"][0]["bbox"])
    actual = metric.compute()

    assert actual["crowdpose/bbox/AP"] == pytest.approx(1.0, abs=1e-12)
    assert actual["crowdpose/bbox/AP .5"] == pytest.approx(1.0, abs=1e-12)
    assert actual["crowdpose/bbox/AP .75"] == pytest.approx(1.0, abs=1e-12)


def test_evaluation_accepts_external_and_lightning_checkpoint_layouts() -> None:
    external = {"model": {"module.weight": torch.ones(1)}}
    lightning = {"state_dict": {"model.weight": torch.ones(1), "criterion.x": torch.zeros(1)}}

    assert set(extract_model_state_dict(external)) == {"weight"}
    assert set(extract_model_state_dict(lightning)) == {"weight"}


def test_selected_layer_evaluation_build_preserves_expansion_head_without_initializer() -> None:
    config = DiffusionRegPosePipelineConfig(
        backbone_pretrained=False,
        pretrained_checkpoint="unused-edpose-initializer.pth",
        pretrained_keep_encoder_layers=(2, 4, 6),
        pretrained_keep_decoder_layers=(2, 4, 6),
        num_queries=8,
        num_proposals=2,
        num_group=2,
        num_encoder_layers=3,
        num_decoder_layers=3,
        num_box_decoder_layers=1,
    )

    model = build_evaluation_model(config, sampling_timesteps=1)

    assert model.expansion_pose_embed is not None
    assert model.pose_embed[0] is not model.expansion_pose_embed


def test_debug_pipeline_can_reconstruct_its_small_model() -> None:
    config = DiffusionRegPosePipelineConfig(
        backbone_pretrained=False,
        hidden_dim=256,
        num_heads=8,
        num_queries=100,
        num_proposals=10,
        num_group=10,
        num_feature_levels=3,
        num_encoder_layers=2,
        num_decoder_layers=3,
        num_box_decoder_layers=1,
    )

    model = config.build(max_epochs=1).model

    assert model.num_queries == 100
    assert model.num_proposals == 10
    assert model.transformer.decoder.num_group == 10

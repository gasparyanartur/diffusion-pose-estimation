from __future__ import annotations

import torch

from src.models.pipelines.pose_estimators.diffusionregpose.evaluation import (
    crowdpose_result_records,
    postprocess_crowdpose,
    resize_crowdpose_eval_image,
)


def test_postprocess_crowdpose_selects_top_class_scores_and_scales_coordinates() -> None:
    outputs = {
        "pred_logits": torch.tensor([[[0.0, 4.0], [5.0, -2.0]]]),
        "pred_boxes": torch.tensor([[[0.5, 0.5, 0.5, 0.25], [0.25, 0.5, 0.2, 0.4]]]),
        "pred_keypoints": torch.tensor(
            [[[0.1, 0.2] * 14 + [0.3] * 14, [0.4, 0.5] * 14 + [0.7] * 14]]
        ),
    }

    result = postprocess_crowdpose(outputs, torch.tensor([[100.0, 200.0]]))[0]

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
    assert records[0]["keypoints"] == [0.0] * 42
    assert records[0]["score"] == torch.tensor(0.9).item()


def test_crowdpose_resize_uses_reference_integer_aspect_ratio() -> None:
    image = torch.ones(3, 100, 333)

    resized = resize_crowdpose_eval_image(image, shortest_side=80, max_size=1333)

    assert resized.shape == (3, 80, 266)


def test_crowdpose_resize_normalizes_non_rgb_images() -> None:
    grayscale = torch.ones(1, 100, 200)
    rgba = torch.ones(4, 100, 200)

    assert resize_crowdpose_eval_image(grayscale).shape[0] == 3
    assert resize_crowdpose_eval_image(rgba).shape[0] == 3

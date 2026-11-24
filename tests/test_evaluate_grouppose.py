import hashlib
import json

import pytest
import torch

from scripts.evaluation.evaluate_grouppose import native_records, select_images
from src.models.pipelines.pose_estimators.diffusionregpose.evaluation import (
    evaluate_crowdpose_results,
    resize_crowdpose_eval_image,
)


def test_subset_requires_matching_annotations_and_unique_ids(tmp_path):
    annotations = tmp_path / "annotations.json"
    annotations.write_text(json.dumps({"images": [{"id": 7}, {"id": 3}, {"id": 5}]}))
    manifest = tmp_path / "subset.json"
    digest = hashlib.sha256(annotations.read_bytes()).hexdigest()
    manifest.write_text(json.dumps({"annotation_sha256": digest, "image_ids": [5, 3]}))
    assert [image["id"] for image in select_images(annotations, manifest)] == [3, 5]
    assert [image["id"] for image in select_images(annotations, None)] == [3, 5, 7]

    for ids in ([5, 5], [8], []):
        manifest.write_text(json.dumps({"annotation_sha256": digest, "image_ids": ids}))
        with pytest.raises(ValueError):
            select_images(annotations, manifest)

    manifest.write_text(json.dumps({"annotation_sha256": "stale", "image_ids": [5]}))
    with pytest.raises(ValueError):
        select_images(annotations, manifest)


def test_ledger_preserves_native_joint_order_and_supports_empty_images():
    prediction = {
        "scores": torch.tensor([0.75, 0.5]),
        "labels": torch.tensor([1, 0]),
        "keypoints": torch.arange(84).reshape(2, 42),
    }
    records = native_records(7, prediction)
    assert records == [
        {"image_id": 7, "category_id": 1, "score": 0.75, "keypoints": list(range(42))},
        {"image_id": 7, "category_id": 0, "score": 0.5, "keypoints": list(range(42, 84))},
    ]
    assert native_records(7, {key: value[:0] for key, value in prediction.items()}) == []


def test_reduced_resize_matches_reference_integer_rounding():
    image = torch.zeros((3, 300, 600))
    resized = resize_crowdpose_eval_image(image, shortest_side=256, max_size=427)
    assert resized.shape == (3, 214, 428)


def test_pose_only_metrics_include_thresholds_without_derived_boxes(tmp_path):
    annotations = tmp_path / "annotations.json"
    joints = [100, 100, 2] * 14
    annotations.write_text(json.dumps({
        "images": [{"id": 7, "width": 256, "height": 256, "crowdIndex": 0.1}],
        "categories": [{"id": 1, "name": "person"}],
        "annotations": [{
            "id": 1, "image_id": 7, "category_id": 1, "iscrowd": 0,
            "area": 10000, "bbox": [50, 50, 100, 100],
            "keypoints": joints, "num_keypoints": 14,
        }],
    }))
    results = tmp_path / "predictions.json"
    results.write_text(json.dumps([
        {"image_id": 7, "category_id": 1, "score": 0.9, "keypoints": joints},
    ]))
    metrics = evaluate_crowdpose_results(
        results, annotations, include_pose_thresholds=True, pose_only=True
    )
    assert metrics["AP"] == pytest.approx(1)
    assert metrics["AP50"] == pytest.approx(1)
    assert metrics["AP75"] == pytest.approx(1)
    assert not any(name.startswith("bbox/") for name in metrics)

    results.write_text("[]")
    metrics = evaluate_crowdpose_results(
        results, annotations, include_pose_thresholds=True, pose_only=True
    )
    assert metrics["AP"] == 0


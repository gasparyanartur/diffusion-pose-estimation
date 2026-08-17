"""Reference-compatible DiffusionRegPose output conversion for CrowdPose."""

from __future__ import annotations

import json
import math
import builtins
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.ops import batched_nms
from torchvision.transforms import InterpolationMode
from torch import Tensor


def box_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    center_x, center_y, width, height = boxes.unbind(-1)
    return torch.stack(
        (
            center_x - width / 2.0,
            center_y - height / 2.0,
            center_x + width / 2.0,
            center_y + height / 2.0,
        ),
        dim=-1,
    )


@torch.no_grad()
def postprocess_crowdpose(
    outputs: dict[str, Tensor],
    target_sizes: Tensor,
    *,
    num_body_points: int = 14,
) -> list[dict[str, Tensor]]:
    """Convert normalized model predictions into CrowdPose result records."""
    logits = outputs["pred_logits"]
    boxes = outputs["pred_boxes"]
    keypoints = outputs["pred_keypoints"]
    if logits.ndim != 3 or boxes.ndim != 3 or keypoints.ndim != 3:
        raise ValueError("model outputs must have batch, query, and channel dimensions")
    if logits.shape[:2] != boxes.shape[:2] or logits.shape[:2] != keypoints.shape[:2]:
        raise ValueError("prediction tensors must agree on batch and query dimensions")
    if target_sizes.shape != (logits.shape[0], 2):
        raise ValueError("target_sizes must have shape [batch, 2] containing height and width")
    if keypoints.shape[-1] != num_body_points * 3:
        raise ValueError("keypoint channel count does not match num_body_points")

    batch_size, query_count, class_count = logits.shape
    select_count = query_count
    probabilities = logits.sigmoid()
    scores, flattened_indices = torch.topk(
        probabilities.flatten(1), select_count, dim=1
    )
    query_indices = flattened_indices // class_count
    labels = flattened_indices % class_count

    boxes = box_cxcywh_to_xyxy(boxes)
    boxes = torch.gather(boxes, 1, query_indices[..., None].expand(-1, -1, 4))
    height, width = target_sizes.unbind(1)
    box_scale = torch.stack((width, height, width, height), dim=1)
    boxes = boxes * box_scale[:, None]

    selected_keypoints = torch.gather(
        keypoints,
        1,
        query_indices[..., None].expand(-1, -1, num_body_points * 3),
    )
    coordinates = selected_keypoints[..., : num_body_points * 2]
    visibility = selected_keypoints[..., num_body_points * 2 :]
    coordinate_scale = torch.stack((width, height), dim=1).repeat(1, num_body_points)
    coordinates = coordinates * coordinate_scale[:, None]
    serialized_keypoints = torch.zeros_like(selected_keypoints)
    serialized_keypoints[..., 0::3] = coordinates[..., 0::2]
    serialized_keypoints[..., 1::3] = coordinates[..., 1::2]
    serialized_keypoints[..., 2::3] = visibility

    return [
        {
            "scores": image_scores,
            "labels": image_labels,
            "boxes": image_boxes,
            "keypoints": image_keypoints,
        }
        for image_scores, image_labels, image_boxes, image_keypoints in zip(
            scores, labels, boxes, serialized_keypoints
        )
    ]


def resize_crowdpose_eval_image(
    image: Tensor,
    *,
    shortest_side: int = 800,
    max_size: int = 1333,
) -> Tensor:
    """Apply the reference test resize and ImageNet normalization."""
    if image.ndim != 3 or image.shape[0] not in (1, 3, 4):
        raise ValueError("image must have shape [1, 3, or 4, height, width]")
    if image.shape[0] == 1:
        image = image.expand(3, -1, -1)
    elif image.shape[0] == 4:
        image = image[:3]
    _, height, width = image.shape
    if width <= height:
        resized_width = shortest_side
        resized_height = int(shortest_side * height / width)
    else:
        resized_height = shortest_side
        resized_width = int(shortest_side * width / height)
    if max(resized_height, resized_width) > max_size:
        if width <= height:
            resized_width = int(round(max_size * width / height))
            resized_height = max_size
        else:
            resized_height = int(round(max_size * height / width))
            resized_width = max_size
    image = TF.resize(
        image,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    return TF.normalize(
        image,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )


def crowdpose_result_records(
    image_id: int,
    results: list[dict[str, Tensor]],
) -> list[dict]:
    """Convert one-image post-processing output into JSON result records."""
    if len(results) != 1:
        raise ValueError("crowdpose_result_records expects one image result")
    result = results[0]
    keep = batched_nms(
        result["boxes"],
        result["scores"],
        result["labels"],
        iou_threshold=1.0,
    )
    records = []
    for label, score, keypoints in zip(
        result["labels"][keep], result["scores"][keep], result["keypoints"][keep]
    ):
        records.append(
            {
                "image_id": int(image_id),
                "category_id": int(label.item()),
                "keypoints": keypoints.detach().cpu().tolist(),
                "score": float(score.item()),
            }
        )
    return records


def write_crowdpose_results(records: list[dict], output_path: Path | str) -> None:
    """Write CrowdPose result records as a JSON array."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records))


def evaluate_crowdpose_results(result_path: Path | str, ground_truth_path: Path | str) -> dict[str, float]:
    """Evaluate a CrowdPose result JSON using crowdposetools."""
    try:
        from crowdposetools.coco import COCO
        from crowdposetools.cocoeval import COCOeval
    except ImportError as error:
        raise RuntimeError("crowdposetools is required for CrowdPose metrics") from error

    ground_truth = COCO(str(ground_truth_path))
    detections = json.loads(Path(result_path).read_text())
    # crowdposetools still checks the Python 2 ``unicode`` name in loadRes.
    had_unicode = hasattr(builtins, "unicode")
    had_numpy_float = hasattr(np, "float")
    if not had_unicode:
        builtins.unicode = str  # type: ignore[attr-defined]
    if not had_numpy_float:
        np.float = float  # type: ignore[attr-defined]
    try:
        detected = COCO.loadRes(ground_truth, detections)
        evaluator = COCOeval(ground_truth, detected, "keypoints")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        difficulty = evaluator.get_type_result(first=0.2, second=0.8)
        metrics = {
            "AP": float(evaluator.stats[0]),
            "AP_E": float(difficulty[0]),
            "AP_M": float(difficulty[1]),
            "AP_H": float(difficulty[2]),
        }
    finally:
        if not had_unicode:
            del builtins.unicode  # type: ignore[attr-defined]
        if not had_numpy_float:
            del np.float  # type: ignore[attr-defined]
    return metrics

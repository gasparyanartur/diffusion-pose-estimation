"""Reference-compatible DiffusionRegPose output conversion for CrowdPose."""

from __future__ import annotations

import json
import math
import builtins
import contextlib
import io
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.ops import batched_nms
from torchvision.transforms import InterpolationMode
from torch import Tensor

CROWDPOSE_POSE_NMS_SCORE_THRESHOLD = 0.05


class DiffusionRegPoseAPMetric:
    """Accumulate predictions with the external CrowdPose AP contract.

    The external evaluator uses ``crowdposetools.COCOeval`` with the
    ``keypoints`` IoU type.  Keeping the result records here and evaluating
    them in one call preserves that contract during Lightning validation.
    """

    def __init__(self, ann_file: str | Path) -> None:
        self._ann_file = str(ann_file)
        self._valid_img_ids = self._load_valid_image_ids()
        self._records: list[dict] = []

    def _load_valid_image_ids(self) -> frozenset[int]:
        from crowdposetools.coco import COCO

        return frozenset(COCO(self._ann_file).getImgIds())

    def reset(self) -> None:
        self._records.clear()

    def update(
        self,
        pred_kpts: Tensor,  # (B, N, 14, 2 or 3), pixel space
        pred_scores: Tensor,  # (B, N, 14), used as visibility for 2-channel kpts
        image_ids: Tensor,  # (B,)
        bbox_scores: Tensor | None = None,  # (B, N), final detection scores
    ) -> None:
        if pred_kpts.ndim != 4 or pred_kpts.shape[-1] not in (2, 3):
            raise ValueError("pred_kpts must have shape [B, N, K, 2 or 3]")
        if pred_scores.shape != pred_kpts.shape[:3]:
            raise ValueError("pred_scores must have shape [B, N, K]")
        if image_ids.shape != (pred_kpts.shape[0],):
            raise ValueError("image_ids must have shape [B]")
        if bbox_scores is not None and bbox_scores.shape != pred_kpts.shape[:2]:
            raise ValueError("bbox_scores must have shape [B, N]")

        keypoints_np = pred_kpts.detach().cpu().numpy()
        scores_np = pred_scores.detach().cpu().numpy()
        image_ids_np = image_ids.detach().cpu().numpy()
        bbox_scores_np = (
            bbox_scores.detach().cpu().numpy() if bbox_scores is not None else None
        )
        for batch_index, image_id_value in enumerate(image_ids_np.tolist()):
            image_id = int(image_id_value)
            if image_id not in self._valid_img_ids:
                continue
            for person_index in range(keypoints_np.shape[1]):
                coordinates = keypoints_np[batch_index, person_index]
                if coordinates.shape[-1] == 2:
                    visibility = scores_np[batch_index, person_index, :, None]
                    coordinates = np.concatenate((coordinates, visibility), axis=-1)
                score = (
                    float(bbox_scores_np[batch_index, person_index])
                    if bbox_scores_np is not None
                    else float(scores_np[batch_index, person_index].mean())
                )
                self._records.append({
                    "image_id": image_id,
                    "category_id": 1,
                    "keypoints": coordinates.reshape(-1).tolist(),
                    "score": score,
                })

    def compute(self) -> dict[str, float]:
        if not self._records:
            return {}
        from crowdposetools.coco import COCO
        from crowdposetools.cocoeval import COCOeval

        had_unicode = hasattr(builtins, "unicode")
        had_numpy_float = hasattr(np, "float")
        if not had_unicode:
            builtins.unicode = str  # type: ignore[attr-defined]
        if not had_numpy_float:
            np.float = float  # type: ignore[attr-defined]
        try:
            ground_truth = COCO(self._ann_file)
            image_ids = {record["image_id"] for record in self._records}
            ground_truth.dataset["images"] = [
                image for image in ground_truth.dataset["images"]
                if image["id"] in image_ids
            ]
            ground_truth.dataset["annotations"] = [
                annotation for annotation in ground_truth.dataset["annotations"]
                if annotation["image_id"] in image_ids
            ]
            ground_truth.createIndex()
            detected = COCO.loadRes(ground_truth, self._records)
            evaluator = COCOeval(ground_truth, detected, "keypoints")
            with contextlib.redirect_stdout(io.StringIO()):
                evaluator.evaluate()
                evaluator.accumulate()
                evaluator.summarize()
            difficulty = evaluator.get_type_result(first=0.2, second=0.8)
            stats = evaluator.stats
            return {
                "crowdpose/AP": float(stats[0]),
                "crowdpose/AP .5": float(stats[1]),
                "crowdpose/AP .75": float(stats[2]),
                "crowdpose/AR": float(stats[5]),
                "crowdpose/AR .5": float(stats[6]),
                "crowdpose/AR .75": float(stats[7]),
                "crowdpose/AP(E)": float(difficulty[0]),
                "crowdpose/AP(M)": float(difficulty[1]),
                "crowdpose/AP(H)": float(difficulty[2]),
            }
        finally:
            if not had_unicode:
                del builtins.unicode  # type: ignore[attr-defined]
            if not had_numpy_float:
                del np.float  # type: ignore[attr-defined]


class DiffusionRegPoseBBoxAPMetric:
    """Accumulate pixel-space person boxes for CrowdPose bbox AP."""

    def __init__(self, ann_file: str | Path) -> None:
        self._ann_file = str(ann_file)
        self._valid_img_ids = self._load_valid_image_ids()
        self._records: list[dict] = []

    def _load_valid_image_ids(self) -> frozenset[int]:
        from crowdposetools.coco import COCO

        return frozenset(COCO(self._ann_file).getImgIds())

    def reset(self) -> None:
        self._records.clear()

    def update(
        self,
        boxes: Tensor,  # (B, N, 4), xyxy pixel space
        scores: Tensor,  # (B, N)
        labels: Tensor,  # (B, N), CrowdPose category ids
        image_ids: Tensor,  # (B,)
    ) -> None:
        if boxes.ndim != 3 or boxes.shape[-1] != 4:
            raise ValueError("boxes must have shape [B, N, 4] in xyxy format")
        if scores.shape != boxes.shape[:2]:
            raise ValueError("scores must have shape [B, N]")
        if labels.shape != boxes.shape[:2]:
            raise ValueError("labels must have shape [B, N]")
        if image_ids.shape != (boxes.shape[0],):
            raise ValueError("image_ids must have shape [B]")

        boxes_np = boxes.detach().cpu().numpy()
        scores_np = scores.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        for batch_index, image_id_value in enumerate(image_ids.detach().cpu().tolist()):
            image_id = int(image_id_value)
            if image_id not in self._valid_img_ids:
                continue
            for person_index, label in enumerate(labels_np[batch_index].tolist()):
                if int(label) != 1:
                    continue
                xmin, ymin, xmax, ymax = boxes_np[batch_index, person_index]
                self._records.append({
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [
                        float(xmin),
                        float(ymin),
                        float(xmax - xmin),
                        float(ymax - ymin),
                    ],
                    "score": float(scores_np[batch_index, person_index]),
                })

    def compute(self) -> dict[str, float]:
        if not self._records:
            return {}
        from crowdposetools.coco import COCO
        from crowdposetools.cocoeval import COCOeval

        had_unicode = hasattr(builtins, "unicode")
        had_numpy_float = hasattr(np, "float")
        if not had_unicode:
            builtins.unicode = str  # type: ignore[attr-defined]
        if not had_numpy_float:
            np.float = float  # type: ignore[attr-defined]
        try:
            ground_truth = COCO(self._ann_file)
            image_ids = {record["image_id"] for record in self._records}
            ground_truth.dataset["images"] = [
                image for image in ground_truth.dataset["images"]
                if image["id"] in image_ids
            ]
            ground_truth.dataset["annotations"] = [
                annotation for annotation in ground_truth.dataset["annotations"]
                if annotation["image_id"] in image_ids
            ]
            ground_truth.createIndex()
            detected = COCO.loadRes(ground_truth, self._records)
            evaluator = COCOeval(ground_truth, detected, "bbox")
            with contextlib.redirect_stdout(io.StringIO()):
                evaluator.evaluate()
                evaluator.accumulate()
                evaluator.summarize()
            stats = evaluator.stats
            return {
                "crowdpose/bbox/AP": float(stats[0]),
                "crowdpose/bbox/AP .5": float(stats[1]),
                "crowdpose/bbox/AP .75": float(stats[2]),
                "crowdpose/bbox/AR": float(stats[8]),
            }
        finally:
            if not had_unicode:
                del builtins.unicode  # type: ignore[attr-defined]
            if not had_numpy_float:
                del np.float  # type: ignore[attr-defined]


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


def _crowdpose_pose_nms_keep(keypoints: Tensor, scores: Tensor, num_body_points: int) -> Tensor:
    """Reproduce the reference CrowdPose keypoint-overlap suppression."""
    coordinates = torch.stack((keypoints[:, :num_body_points], keypoints[:, num_body_points : num_body_points * 2]), dim=-1)
    pose_area = (
        coordinates.amax(dim=1) - coordinates.amin(dim=1)
    ).square().sum(dim=-1)
    pairwise_distance = (coordinates[:, None] - coordinates[None]).square().sum(-1).sqrt()
    threshold = 0.04 * pose_area[:, None, None].sqrt()
    nearby_joint_count = (pairwise_distance < threshold).sum(-1)
    suppress = (nearby_joint_count > 10) & (
        scores[None] > CROWDPOSE_POSE_NMS_SCORE_THRESHOLD
    )

    ignored: set[int] = set()
    keep: list[int] = []
    for index in range(suppress.shape[0]):
        if index in ignored:
            continue
        candidates = suppress[index].nonzero(as_tuple=False).flatten().tolist()
        if candidates:
            best = candidates[int(scores[candidates].argmax().item())]
        else:
            best = 0
        if best in ignored:
            continue
        keep.append(best)
        ignored.update(set(candidates) - set(ignored))
    return torch.tensor(keep, dtype=torch.long, device=keypoints.device)


def _select_crowdpose_candidates(
    outputs: dict[str, Tensor],
    target_sizes: Tensor,
    *,
    num_body_points: int,
    nms_target_sizes: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Select one class-expanded candidate per query before pose suppression."""
    logits = outputs["pred_logits"]
    boxes = outputs["pred_boxes"]
    keypoints = outputs["pred_keypoints"]
    if logits.ndim != 3 or boxes.ndim != 3 or keypoints.ndim != 3:
        raise ValueError("model outputs must have batch, query, and channel dimensions")
    if logits.shape[:2] != boxes.shape[:2] or logits.shape[:2] != keypoints.shape[:2]:
        raise ValueError("prediction tensors must agree on batch and query dimensions")
    if target_sizes.shape != (logits.shape[0], 2):
        if nms_target_sizes is not None and nms_target_sizes.shape != (logits.shape[0], 2):
            raise ValueError("nms_target_sizes must have shape [batch, 2]")
        raise ValueError("target_sizes must have shape [batch, 2] containing height and width")
    if keypoints.shape[-1] != num_body_points * 3:
        raise ValueError("keypoint channel count does not match num_body_points")

    _, query_count, class_count = logits.shape
    selected = [
        image_probabilities.flatten(0, 1).topk(query_count, sorted=False)
        for image_probabilities in logits.sigmoid()
    ]
    scores = torch.stack([item.values for item in selected])
    flattened_indices = torch.stack([item.indices for item in selected])
    query_indices = flattened_indices // class_count
    labels = flattened_indices % class_count

    raw_boxes = box_cxcywh_to_xyxy(boxes)
    selected_boxes = torch.gather(raw_boxes, 1, query_indices[..., None].expand(-1, -1, 4))
    height, width = target_sizes.unbind(1)
    box_scale = torch.stack((width, height, width, height), dim=1)
    selected_boxes = selected_boxes * box_scale[:, None]

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

    if nms_target_sizes is None:
        nms_target_sizes = target_sizes
    nms_height, nms_width = nms_target_sizes.unbind(1)
    nms_coordinate_scale = torch.cat(
        (
            nms_width[:, None].expand(-1, num_body_points),
            nms_height[:, None].expand(-1, num_body_points),
        ),
        dim=1,
    )
    nms_coordinates = selected_keypoints[..., : num_body_points * 2] * nms_coordinate_scale[:, None]
    nms_keypoints = torch.cat((nms_coordinates, visibility), dim=-1)
    return scores, labels, selected_boxes, serialized_keypoints, nms_keypoints, query_indices


@torch.no_grad()
def postprocess_crowdpose(
    outputs: dict[str, Tensor],
    target_sizes: Tensor,
    *,
    num_body_points: int = 14,
    pose_nms: bool = True,
    nms_target_sizes: Tensor | None = None,
) -> list[dict[str, Tensor]]:
    """Convert normalized model predictions into CrowdPose result records."""
    logits = outputs["pred_logits"]
    scores, labels, boxes, serialized_keypoints, nms_keypoints, query_indices = _select_crowdpose_candidates(
        outputs,
        target_sizes,
        num_body_points=num_body_points,
        nms_target_sizes=nms_target_sizes,
    )
    class_count = logits.shape[-1]
    if pose_nms:
        results = []
        for batch_index, (image_scores, image_labels, image_boxes, image_keypoints, nms_image_keypoints) in enumerate(
            zip(scores, labels, boxes, serialized_keypoints, nms_keypoints)
        ):
            keep = _crowdpose_pose_nms_keep(nms_image_keypoints, image_scores, num_body_points)
            kept_queries = query_indices[batch_index].index_select(0, keep)
            kept_logits = logits[batch_index].index_select(0, kept_queries)
            kept_boxes = image_boxes.index_select(0, keep)
            kept_keypoints = image_keypoints.index_select(0, keep)
            second_count = kept_logits.shape[0]
            second_scores, second_indices = torch.topk(
                kept_logits.sigmoid().flatten(), second_count, sorted=True
            )
            second_labels = second_indices.remainder(class_count)
            second_queries = second_indices.div(class_count, rounding_mode="floor")
            final_boxes = kept_boxes.index_select(0, second_queries)
            final_serialized = kept_keypoints.index_select(0, second_queries)
            results.append(
                {
                    "scores": second_scores,
                    "labels": second_labels,
                    "boxes": final_boxes,
                    "keypoints": final_serialized,
                }
            )
        return results

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


@torch.no_grad()
def postprocess_crowdpose_step_ensemble(
    step_outputs: list[dict[str, Tensor]],
    target_sizes: Tensor,
    *,
    num_body_points: int = 14,
    pose_nms: bool = True,
    nms_target_sizes: Tensor | None = None,
) -> list[dict[str, Tensor]]:
    """Select candidates per flow step, then suppress once across all steps."""
    if not step_outputs:
        raise ValueError("step_outputs must contain at least one flow prediction")
    per_step = [
        _select_crowdpose_candidates(
            outputs,
            target_sizes,
            num_body_points=num_body_points,
            nms_target_sizes=nms_target_sizes,
        )
        for outputs in step_outputs
    ]
    batch_size = target_sizes.shape[0]
    if any(candidates[0].shape[0] != batch_size for candidates in per_step):
        raise ValueError("step prediction batch sizes must match target_sizes")

    results = []
    for batch_index in range(batch_size):
        scores, labels, boxes, keypoints, nms_keypoints = (
            torch.cat([candidates[field][batch_index] for candidates in per_step], dim=0)
            for field in range(5)
        )
        keep = (
            _crowdpose_pose_nms_keep(nms_keypoints, scores, num_body_points)
            if pose_nms
            else torch.arange(scores.shape[0], device=scores.device)
        )
        kept_scores = scores.index_select(0, keep)
        order = kept_scores.argsort(descending=True)
        keep = keep.index_select(0, order)
        results.append({
            "scores": scores.index_select(0, keep),
            "labels": labels.index_select(0, keep),
            "boxes": boxes.index_select(0, keep),
            "keypoints": keypoints.index_select(0, keep),
        })
    return results


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
        resized_short_side = shortest_side
        if height / width * resized_short_side > max_size:
            resized_short_side = int(round(max_size * width / height))
        resized_width = resized_short_side
        resized_height = int(resized_short_side * height / width)
    else:
        resized_short_side = shortest_side
        if width / height * resized_short_side > max_size:
            resized_short_side = int(round(max_size * height / width))
        resized_height = resized_short_side
        resized_width = int(resized_short_side * width / height)
    # The evaluation transform receives a PIL image in the original data
    # pipeline.  Keeping the resize in PIL space avoids a different sampling
    # kernel from torchvision's tensor path.
    image = TF.to_tensor(
        TF.resize(
            TF.to_pil_image(image),
            [resized_height, resized_width],
            interpolation=InterpolationMode.BILINEAR,
        )
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
    for label, score, box, keypoints in zip(
        result["labels"][keep],
        result["scores"][keep],
        result["boxes"][keep],
        result["keypoints"][keep],
    ):
        xmin, ymin, xmax, ymax = box.detach().cpu().tolist()
        records.append(
            {
                "image_id": int(image_id),
                "category_id": int(label.item()),
                "bbox": [
                    float(xmin),
                    float(ymin),
                    float(xmax - xmin),
                    float(ymax - ymin),
                ],
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


def evaluate_crowdpose_results(
    result_path: Path | str,
    ground_truth_path: Path | str,
    *,
    image_ids: list[int] | None = None,
) -> dict[str, float]:
    """Evaluate a CrowdPose result JSON using crowdposetools."""
    try:
        from crowdposetools.coco import COCO
        from crowdposetools.cocoeval import COCOeval
    except ImportError as error:
        raise RuntimeError("crowdposetools is required for CrowdPose metrics") from error

    ground_truth = COCO(str(ground_truth_path))
    if image_ids is not None:
        selected_ids = set(image_ids)
        ground_truth.dataset["images"] = [
            image for image in ground_truth.dataset["images"]
            if image["id"] in selected_ids
        ]
        ground_truth.dataset["annotations"] = [
            annotation for annotation in ground_truth.dataset["annotations"]
            if annotation["image_id"] in selected_ids
        ]
        ground_truth.createIndex()
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
        metrics: dict[str, float] = {
            "AP": float(evaluator.stats[0]),
            "AP_E": float(difficulty[0]),
            "AP_M": float(difficulty[1]),
            "AP_H": float(difficulty[2]),
        }
        if detections and all("bbox" in detection for detection in detections):
            bbox_detected = COCO.loadRes(ground_truth, detections)
            bbox_evaluator = COCOeval(ground_truth, bbox_detected, "bbox")
            bbox_evaluator.evaluate()
            bbox_evaluator.accumulate()
            bbox_evaluator.summarize()
            metrics.update(
                {
                    "bbox/AP": float(bbox_evaluator.stats[0]),
                    "bbox/AP50": float(bbox_evaluator.stats[1]),
                    "bbox/AP75": float(bbox_evaluator.stats[2]),
                    "bbox/AR": float(bbox_evaluator.stats[8]),
                }
            )
    finally:
        if not had_unicode:
            del builtins.unicode  # type: ignore[attr-defined]
        if not had_numpy_float:
            del np.float  # type: ignore[attr-defined]
    return metrics

"""Keypoint AP metric wrapping MMPose's CocoMetric.

Handles the coordinate-space conversion between the model's internal
representation and what CocoMetric expects:

  canonical (image-normalised)  →  pixel space  →  CocoMetric

Canonical space
    The model operates on image-normalised poses produced by
    :func:`src.data.prepare_pose_graph_batch`:

    * canonical = pixel / [W, H] * 2 - 1  (in-frame joints are in (-1, 1)).
    * Inverse: pixel = (canonical + 1) / 2 * [W, H].

    Shape: (B, N, 14, 2)

Pixel space
    Absolute (x, y) image coordinates in pixels.
    Shape: (B, N, 14, 2)
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from src.data import CROWDPOSE_JOINT_NAMES

# ---------------------------------------------------------------------------
# CrowdPose metadata (mirrors external/mmpose/configs/_base_/datasets/crowdpose.py)
# ---------------------------------------------------------------------------

CROWDPOSE_SIGMAS: np.ndarray = np.array([
    0.079, 0.079,   # shoulders
    0.072, 0.072,   # elbows
    0.062, 0.062,   # wrists
    0.107, 0.107,   # hips
    0.087, 0.087,   # knees
    0.089, 0.089,   # ankles
    0.079, 0.079,   # head, neck
], dtype=np.float64)

CROWDPOSE_DATASET_META: dict = {
    "num_keypoints": 14,
    "sigmas": CROWDPOSE_SIGMAS,
    "CLASSES": ["person"],
    "dataset_name": "crowdpose",
}


def canonical_to_pixel(
    pred_canonical: Tensor,  # (B, N, 14, 2)
    img_wh: Tensor,          # (B, 2) [W, H] in pixels
) -> Tensor:                 # (B, N, 14, 2)
    """Invert canonical-space normalisation: canonical = pixel / [W, H] * 2 - 1."""
    return (pred_canonical + 1) / 2 * img_wh[:, None, None, :]  # (B, N, 14, 2)


class KeypointAPMetric:
    """OKS-based AP / AR metric for CrowdPose, backed by MMPose's CocoMetric.

    Accumulates predictions batch-by-batch during an epoch; call
    :meth:`compute` at the end to get the final AP/AR dict.

    Args:
        ann_file: Path to the CrowdPose annotation JSON (train/val/test).

    Example::

        metric = KeypointAPMetric("data/crowdpose/annotations/crowdpose_val.json")
        for batch in loader:
            # pred_kpts: (B, N, 14, 2) pixel space
            # pred_scores: (B, N, 14)
            metric.update(pred_kpts, pred_scores, batch["image_id"])
        results = metric.compute()   # {"crowdpose/AP": ..., "crowdpose/AP .5": ..., ...}
        metric.reset()
    """

    def __init__(self, ann_file: str | None) -> None:
        self._ann_file = ann_file
        self._metric = self._build_metric()
        self._valid_img_ids: frozenset[int] = frozenset(self._metric.coco.getImgIds())
        self._n_images = 0

    def _build_metric(self):
        from mmpose.evaluation.metrics import CocoMetric
        m = CocoMetric(
            ann_file=self._ann_file,
            use_area=False,
            iou_type="keypoints_crowd",
            prefix="crowdpose",
        )
        m.dataset_meta = CROWDPOSE_DATASET_META
        return m

    def reset(self) -> None:
        """Clear accumulated results; ready for a new epoch."""
        self._metric = self._build_metric()
        self._n_images = 0
        # valid_img_ids is constant for the lifetime of this metric — no need to rebuild

    @staticmethod
    def build_samples_for_image(
        img_id: int,
        kpts: np.ndarray,          # (N, 14, 2)
        scores: np.ndarray,        # (N, 14)
        ann_ids: list,
        bbox_scores: np.ndarray | None = None,  # (N,) detector bbox confidence
    ) -> list:
        """Format predictions for one image into one CocoMetric sample per person.

        ``CocoMetric._sort_and_unique_bboxes`` deduplicates by the ``'id'``
        field.  If all N people share the same id (as in the naive
        one-dict-per-image approach) N-1 predictions are silently deleted and
        recall collapses to near zero.  This method emits one dict per person
        each with a distinct id.

        Args:
            img_id:      COCO image id.
            kpts:        Float array (N, 14, 2) – pixel keypoints for the image.
            scores:      Float array (N, 14)    – per-keypoint confidence scores.
            ann_ids:     List of GT annotation ids for this image.  Used as
                         sample ids when available; synthetic fallback otherwise.
            bbox_scores: Float array (N,) – detector bbox confidence scores.
                         When provided they are written to ``pred_instances`` so
                         CocoMetric's ``score_mode='bbox_keypoint'`` uses real
                         detector confidence.  When None, CocoMetric falls back
                         to the ones stored in ``gt_instances``.

        Returns:
            List of N data-sample dicts ready for ``CocoMetric.process``.
        """
        N = kpts.shape[0]
        samples = []
        for n in range(N):
            # Re-use the GT annotation id when available; fall back to a
            # deterministic unique id derived from (img_id, n).
            sample_id = int(ann_ids[n]) if n < len(ann_ids) else img_id * 10000 + n
            pred_instances: dict = {
                "keypoints":       kpts[n : n + 1],    # (1, 14, 2)
                "keypoint_scores": scores[n : n + 1],  # (1, 14)
            }
            if bbox_scores is not None:
                pred_instances["bbox_scores"] = np.array(
                    [bbox_scores[n]], dtype=np.float32
                )  # (1,)
            samples.append({
                "id":          sample_id,
                "img_id":      img_id,
                "category_id": 1,
                "pred_instances": pred_instances,
                "gt_instances": {
                    "bbox_scores": np.ones(1, dtype=np.float32),
                },
            })
        return samples

    def update(
        self,
        pred_kpts: Tensor,    # (B, N, 14, 2) pixel space
        pred_scores: Tensor,  # (B, N, 14)
        image_ids: Tensor,    # (B,)
        bbox_scores: Tensor | None = None,  # (B, N) detector bbox confidence
    ) -> None:
        """Accumulate one batch of predictions.

        All tensors may still be on GPU; they are moved to CPU here.
        """
        kpts_np        = pred_kpts.detach().cpu().numpy()     # (B, N, 14, 2)
        scores_np      = pred_scores.detach().cpu().numpy()   # (B, N, 14)
        bbox_scores_np = (
            bbox_scores.detach().cpu().numpy()                # (B, N)
            if bbox_scores is not None else None
        )
        ids_list = image_ids.tolist()

        B = kpts_np.shape[0]
        data_samples = []
        for b in range(B):
            img_id = int(ids_list[b])
            if img_id not in self._valid_img_ids:
                continue  # prediction belongs to a different split – skip
            ann_ids = list(self._metric.coco.getAnnIds(imgIds=[img_id]))
            b_bbox_scores = bbox_scores_np[b] if bbox_scores_np is not None else None
            data_samples.extend(
                self.build_samples_for_image(
                    img_id, kpts_np[b], scores_np[b], ann_ids, b_bbox_scores
                )
            )

        self._metric.process({}, data_samples)
        self._n_images += len(data_samples)  # counts persons, matching len(self._metric.results)

    def compute(self) -> dict[str, float]:
        """Return AP / AR metrics, or empty dict if nothing accumulated."""
        if self._n_images == 0:
            return {}
        return self._metric.evaluate(self._n_images)


def compute_mpjpe(
    pred_pixel: Tensor,  # (B, N, K, 2)
    gt_pixel: Tensor,    # (B, N, K, 2)
    visibility: Tensor,  # (B, N, K)  0=absent 1=occluded 2=visible
) -> Tensor:
    """Mean per-joint position error in pixels, averaged over visible joints."""
    visible = visibility > 0
    per_joint_err = (pred_pixel - gt_pixel).norm(dim=-1)  # (B, N, K)
    if not visible.any():
        return pred_pixel.new_zeros(())
    return per_joint_err[visible].mean()


def compute_pck(
    pred_pixel: Tensor,  # (B, N, K, 2)
    gt_pixel: Tensor,    # (B, N, K, 2)
    visibility: Tensor,  # (B, N, K)  0=absent 1=occluded 2=visible
    threshold_px: float = 20.0,
) -> Tensor:
    """Fraction of visible joints predicted within threshold_px pixels of GT."""
    visible = visibility > 0
    per_joint_err = (pred_pixel - gt_pixel).norm(dim=-1)
    if not visible.any():
        return pred_pixel.new_zeros(())
    return (per_joint_err[visible] < threshold_px).float().mean()

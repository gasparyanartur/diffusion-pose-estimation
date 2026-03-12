"""Keypoint AP metric wrapping MMPose's CocoMetric.

Handles the two coordinate-space conversions that sit between the model's
internal representation and what CocoMetric expects:

  canonical (root-relative, normalised)  →  pixel space  →  CocoMetric

Canonical space
    The model operates on root-relative, normalised poses produced by
    :func:`src.data.prepare_pose_graph_batch`:

    * Root node (index 0) is at (0, 0).
    * Remaining 14 joints are relative to the root.
    * All coordinates are divided by ``scale`` (default 800 px).

    Shape: (B, N, 15, 2)

Pixel space
    Absolute (x, y) image coordinates in pixels.
    Shape: (B, N, 14, 2)  — virtual root node is dropped.
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


def compute_gt_root_pos(keypoints: Tensor, visibility: Tensor) -> Tensor:
    """Compute the root anchor per (batch, person) from CrowdPose keypoints.

    Mirrors the root-computation logic in :func:`src.data.prepare_pose_graph_batch`
    so that the inverse transform uses exactly the same anchor.

    Args:
        keypoints:  FloatTensor (B, N, 14, 2) – pixel-space GT keypoints.
        visibility: FloatTensor (B, N, 14)    – 0=absent / 1=occluded / 2=visible.

    Returns:
        FloatTensor (B, N, 2) – root position per (batch, person) in pixel space.
    """
    j = CROWDPOSE_JOINT_NAMES
    anchor_idx = torch.tensor(
        [j["left_hip"], j["right_hip"], j["neck"]], device=keypoints.device
    )

    anchor_vis  = visibility[:, :, anchor_idx]           # (B, N, 3)
    anchor_mask = anchor_vis > 0                         # (B, N, 3)
    has_anchor  = anchor_mask.any(dim=2, keepdim=True)   # (B, N, 1)

    anchor_pos = keypoints[:, :, anchor_idx, :]                                  # (B, N, 3, 2)
    anchor_w   = anchor_mask.float().unsqueeze(-1)                               # (B, N, 3, 1)
    root_from_anchors = (anchor_pos * anchor_w).sum(2) / anchor_w.sum(2).clamp(min=1)  # (B, N, 2)

    all_mask      = (visibility > 0).float().unsqueeze(-1)                       # (B, N, 14, 1)
    root_from_all = (keypoints * all_mask).sum(2) / all_mask.sum(2).clamp(min=1) # (B, N, 2)

    return torch.where(has_anchor, root_from_anchors, root_from_all)             # (B, N, 2)


def canonical_to_pixel(
    pred_canonical: Tensor,
    root_pos: Tensor,
    scale: "float | Tensor" = 800.0,
) -> Tensor:
    """Invert the canonical-space normalisation and recover pixel keypoints.

    Args:
        pred_canonical: FloatTensor (B, N, 15, 2) – model output in canonical
                        space (root-relative, divided by *scale*).
        root_pos:       FloatTensor (B, N, 2) – root position in pixel space.
        scale:          Normalisation divisor used in prepare_pose_graph_batch.
                        Can be a scalar float (applied uniformly) or a
                        FloatTensor of shape (B,) for per-image scales.  Must
                        match the scale used when building the canonical graph.

    Returns:
        FloatTensor (B, N, 14, 2) – pixel-space keypoints (virtual root at
        index 0 is dropped; indices 1–14 become 0–13).
    """
    if isinstance(scale, Tensor):
        scale = scale.view(scale.shape[0], 1, 1, 1)  # (B, 1, 1, 1) for broadcasting
    # Undo normalisation and shift back to pixel origin
    pixel = pred_canonical * scale + root_pos.unsqueeze(2)  # (B, N, 15, 2)
    # Drop the virtual root node (index 0)
    return pixel[:, :, 1:, :]  # (B, N, 14, 2)


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
        results = metric.compute()   # {"coco/AP": ..., "coco/AP .5": ..., ...}
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
            iou_type="keypoints",
            score_mode="keypoint",
            nms_mode="none",
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
        kpts: np.ndarray,    # (N, 14, 2)
        scores: np.ndarray,  # (N, 14)
        ann_ids: list,
    ) -> list:
        """Format predictions for one image into one CocoMetric sample per person.

        ``CocoMetric._sort_and_unique_bboxes`` deduplicates by the ``'id'``
        field.  If all N people share the same id (as in the naive
        one-dict-per-image approach) N-1 predictions are silently deleted and
        recall collapses to near zero.  This method emits one dict per person
        each with a distinct id.

        Args:
            img_id:  COCO image id.
            kpts:    Float array (N, 14, 2) – pixel keypoints for the image.
            scores:  Float array (N, 14)    – per-keypoint confidence scores.
            ann_ids: List of GT annotation ids for this image.  Used as
                     sample ids when available; synthetic fallback otherwise.

        Returns:
            List of N data-sample dicts ready for ``CocoMetric.process``.
        """
        N = kpts.shape[0]
        samples = []
        for n in range(N):
            # Re-use the GT annotation id when available; fall back to a
            # deterministic unique id derived from (img_id, n).
            sample_id = int(ann_ids[n]) if n < len(ann_ids) else img_id * 10000 + n
            samples.append({
                "id":          sample_id,
                "img_id":      img_id,
                "category_id": 1,
                "pred_instances": {
                    "keypoints":       kpts[n : n + 1],    # (1, 14, 2)
                    "keypoint_scores": scores[n : n + 1],  # (1, 14)
                },
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
    ) -> None:
        """Accumulate one batch of predictions.

        All tensors may still be on GPU; they are moved to CPU here.
        """
        kpts_np   = pred_kpts.detach().cpu().numpy()    # (B, N, 14, 2)
        scores_np = pred_scores.detach().cpu().numpy()  # (B, N, 14)
        ids_list  = image_ids.tolist()

        B = kpts_np.shape[0]
        data_samples = []
        for b in range(B):
            img_id = int(ids_list[b])
            if img_id not in self._valid_img_ids:
                continue  # prediction belongs to a different split – skip
            ann_ids = list(self._metric.coco.getAnnIds(imgIds=[img_id]))
            data_samples.extend(
                self.build_samples_for_image(img_id, kpts_np[b], scores_np[b], ann_ids)
            )

        self._metric.process({}, data_samples)
        self._n_images += len(data_samples)  # counts persons, matching len(self._metric.results)

    def compute(self) -> dict[str, float]:
        """Return AP / AR metrics, or empty dict if nothing accumulated."""
        if self._n_images == 0:
            return {}
        return self._metric.evaluate(self._n_images)

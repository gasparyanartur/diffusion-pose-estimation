"""Loss functions for keypoint training.

Two families:
  * Refinement losses (joint/bone): per-(B,N,K) tensors with paired predictions.
  * Set-prediction losses (OKS-based Hungarian matching): nn.Module components
    composed by SetPredictionLoss.

Set-prediction inputs use canonical (-1, 1) coordinates. See ``docs/project.md``.

Output dict keys from SetPredictionLoss.forward:
    total              weighted sum of all component losses
    oks_loss           1 - mean OKS over matched slots
    objectness_loss    BCE with dynamic pos_weight over all slots
    visibility_loss    BCE over matched slots and labeled joints
    ranking_loss       pairwise RankNet loss on objectness logits
    mean_oks           mean OKS over matched slots (detached diagnostic)
    matched            total matched count across the batch
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_linear_assignment import assignment_to_indices, batch_linear_assignment


# ---------------------------------------------------------------------------
# Refinement losses (free functions — no state)
# ---------------------------------------------------------------------------


def compute_joint_loss(
    pred_pixel: Tensor,   # (B, N, K, 2)
    gt_pixel: Tensor,     # (B, N, K, 2)
    visibility: Tensor,   # (B, N, K)  0=absent, 1=occluded, 2=visible
) -> Tensor:
    """Visibility-weighted squared joint error in pixel space."""
    visibility_weight = visibility.float()
    per_joint_sq_err = ((pred_pixel - gt_pixel) ** 2).sum(dim=-1)
    total_weight = visibility_weight.sum().clamp(min=1.0)
    return (per_joint_sq_err * visibility_weight).sum() / total_weight


def compute_bone_loss(
    pred_pixel: Tensor,
    gt_pixel: Tensor,
    visibility: Tensor,
    edge_list: list[tuple[int, int]],
) -> Tensor:
    """Per-edge squared length error, weighted by endpoint visibility product."""
    visibility_weight = visibility.float()
    loss = gt_pixel.new_zeros(())
    for u, v in edge_list:
        edge_weight = visibility_weight[:, :, u] * visibility_weight[:, :, v]
        weight_sum = edge_weight.sum().clamp(min=1.0)
        pred_len = (pred_pixel[:, :, u] - pred_pixel[:, :, v]).norm(dim=-1)
        gt_len = (gt_pixel[:, :, u] - gt_pixel[:, :, v]).norm(dim=-1)
        loss = loss + ((pred_len - gt_len).pow(2) * edge_weight).sum() / weight_sum
    return loss / len(edge_list)


# ---------------------------------------------------------------------------
# Set-prediction utilities (free functions — pure math, no state)
# ---------------------------------------------------------------------------


def compute_oks(
    pred_keypoints: Tensor,   # [N, K, 2] canonical
    gt_keypoints: Tensor,     # [P, K, 2] canonical
    gt_labeled: Tensor,       # [P, K]    {0, 1}
    person_scale: Tensor,     # [P]       canonical-space scale
    sigmas: Tensor,           # [K]
    eps: float = 1e-6,
) -> Tensor:                  # [N, P]
    """OKS matrix between every predicted slot and every GT person."""
    N = pred_keypoints.shape[0]
    P = gt_keypoints.shape[0]
    if P == 0 or N == 0:
        return pred_keypoints.new_zeros(N, P)

    diff = pred_keypoints[:, None, :, :] - gt_keypoints[None, :, :, :]   # [N,P,K,2]
    d2 = (diff ** 2).sum(dim=-1)                                         # [N,P,K]

    s = person_scale.clamp_min(eps)[None, :, None]                       # [1,P,1]
    k = sigmas.to(pred_keypoints.device, pred_keypoints.dtype)[None, None, :]
    e = torch.exp(-d2 / (2.0 * (s ** 2) * (k ** 2) + eps))               # [N,P,K]

    valid = gt_labeled[None, :, :].to(pred_keypoints.dtype)              # [1,P,K]
    num = (e * valid).sum(dim=-1)
    den = valid.sum(dim=-1).clamp_min(eps)
    return num / den


def hungarian_match(
    oks: Tensor,               # [N, P]
    objectness_logits: Tensor, # [N]
    lambda_oks: float = 1.0,
    lambda_obj: float = 0.1,
) -> tuple[Tensor, Tensor]:
    """Min-cost bipartite assignment of slots to GT people.

    Returns ``(slot_idx, gt_idx)`` of length ``min(N, P)``.
    """
    N, P = oks.shape
    device = oks.device
    if N == 0 or P == 0:
        empty = torch.zeros(0, dtype=torch.long, device=device)
        return empty, empty
    obj_cost = -F.logsigmoid(objectness_logits)[:, None].expand(N, P)
    cost = lambda_oks * (1.0 - oks) + lambda_obj * obj_cost
    assignment = batch_linear_assignment(cost.detach().unsqueeze(0))  # [1, N, P]
    row, col = assignment_to_indices(assignment)                       # [1, M], [1, M]
    return row[0], col[0]


# ---------------------------------------------------------------------------
# Loss weight config
# ---------------------------------------------------------------------------


@dataclass
class SetPredictionLossWeights:
    """Multipliers for the components of ``SetPredictionLoss``."""

    oks: float = 1.0
    obj: float = 1.0
    vis: float = 0.5
    rank: float = 1.0
    matcher_lambda_oks: float = 1.0
    matcher_lambda_obj: float = 0.1


# ---------------------------------------------------------------------------
# Per-component loss modules
# ---------------------------------------------------------------------------


class OKSLoss(nn.Module):
    """1 - mean OKS over matched slots."""

    def forward(self, matched_oks: Tensor, matched_count: int) -> Tensor:
        if matched_count == 0:
            return matched_oks.new_zeros(())
        return (1.0 - matched_oks).mean()


class ObjectnessLoss(nn.Module):
    """BCE with dynamic pos_weight to compensate slot imbalance.

    pos_weight = (N - P) / max(P, 1), capped at 20.
    """

    _max_pos_weight: float = 20.0

    def forward(
        self,
        pred_objectness: Tensor,   # [N]
        objectness_target: Tensor, # [N]
        matched_count: int,
    ) -> Tensor:
        N = pred_objectness.shape[0]
        pos_weight_val = min(
            (N - matched_count) / max(matched_count, 1),
            self._max_pos_weight,
        )
        pos_weight = pred_objectness.new_tensor([pos_weight_val])
        return F.binary_cross_entropy_with_logits(
            pred_objectness, objectness_target, pos_weight=pos_weight,
        )


class VisibilityLoss(nn.Module):
    """BCE over matched slots and labeled joints only."""

    def forward(
        self,
        pred_vis_logits: Tensor,  # [M, K]
        gt_visible: Tensor,       # [M, K]  {0, 1}
        gt_labeled: Tensor,       # [M, K]  {0, 1}
    ) -> Tensor:
        labeled_count = gt_labeled.sum()
        if labeled_count == 0:
            return pred_vis_logits.new_zeros(())
        bce = F.binary_cross_entropy_with_logits(
            pred_vis_logits, gt_visible.float(), reduction="none",
        )
        return (bce * gt_labeled).sum() / labeled_count


class RankingLoss(nn.Module):
    """Pairwise RankNet loss: matched slots should outscore unmatched slots.

    For each (matched slot i, unmatched slot j) pair:
        loss_ij = -log sigmoid(logit_i - logit_j)
    """

    def forward(
        self,
        pred_objectness: Tensor,  # [N]
        slot_idx: Tensor,         # [M] indices of matched slots
        unmatched_idx: Tensor,    # [U] indices of unmatched slots
    ) -> Tensor:
        if slot_idx.numel() == 0 or unmatched_idx.numel() == 0:
            return pred_objectness.new_zeros(())
        logit_matched   = pred_objectness[slot_idx]    # [M]
        logit_unmatched = pred_objectness[unmatched_idx]  # [U]
        diff = logit_matched[:, None] - logit_unmatched[None, :]  # [M, U]
        return -F.logsigmoid(diff).mean()


# ---------------------------------------------------------------------------
# Composed SetPredictionLoss
# ---------------------------------------------------------------------------


class SetPredictionLoss(nn.Module):
    """Full set-prediction loss: matching + per-component losses.

    Registers ``sigmas`` as a buffer so it moves with ``.to(device)``.

    Args:
        weights: loss component multipliers.
        sigmas:  per-keypoint OKS sigma values, shape ``[K]``.

    forward inputs
    --------------
    pred:
        scores_logits  [B, N]
        keypoints      [B, N, K, 2]  canonical (-1, 1)
        vis_logits     [B, N, K]
    targets:
        gt_keypoints   [B, P_max, K, 2]  canonical
        gt_labeled     [B, P_max, K]
        gt_visible     [B, P_max, K]
        person_scale   [B, P_max]
        person_mask    [B, P_max]

    forward outputs
    ---------------
    total            weighted sum
    oks_loss         1 - mean OKS over matched slots
    objectness_loss  BCE with dynamic pos_weight
    visibility_loss  BCE over matched labeled joints
    ranking_loss     pairwise RankNet on objectness logits
    mean_oks         mean OKS (detached diagnostic)
    matched          total matched count across batch
    """

    def __init__(self, weights: SetPredictionLossWeights, sigmas: Tensor) -> None:
        super().__init__()
        self.weights = weights
        self.register_buffer("sigmas", sigmas)
        self._oks_loss        = OKSLoss()
        self._objectness_loss = ObjectnessLoss()
        self._visibility_loss = VisibilityLoss()
        self._ranking_loss    = RankingLoss()

    def forward(
        self,
        pred: dict[str, Tensor],
        targets: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        obj_logits = pred["scores_logits"]
        pred_kp    = pred["keypoints"]
        pred_vis   = pred["vis_logits"]
        B          = obj_logits.shape[0]
        device     = obj_logits.device

        component_keys = ("oks_loss", "objectness_loss", "visibility_loss", "ranking_loss", "mean_oks")
        accum: dict[str, list[Tensor]] = {k: [] for k in component_keys}
        matched_total = 0

        for b in range(B):
            P = int(targets["person_mask"][b].sum().item())
            per = self._forward_single(
                obj_logits[b],
                pred_kp[b],
                pred_vis[b],
                targets["gt_keypoints"][b, :P].to(device),
                targets["gt_labeled"][b, :P].to(device),
                targets["gt_visible"][b, :P].to(device),
                targets["person_scale"][b, :P].to(device),
            )
            for k in component_keys:
                accum[k].append(per[k])
            matched_total += per["matched_count"]

        components = {k: torch.stack(accum[k]).mean() for k in component_keys}
        w = self.weights
        total = (
            w.oks  * components["oks_loss"]
            + w.obj  * components["objectness_loss"]
            + w.vis  * components["visibility_loss"]
            + w.rank * components["ranking_loss"]
        )
        return {
            "total":            total,
            "oks_loss":         components["oks_loss"],
            "objectness_loss":  components["objectness_loss"],
            "visibility_loss":  components["visibility_loss"],
            "ranking_loss":     components["ranking_loss"],
            "mean_oks":         components["mean_oks"],
            "matched":          torch.tensor(float(matched_total), device=device),
        }

    def _forward_single(
        self,
        pred_objectness: Tensor,  # [N]
        pred_keypoints: Tensor,   # [N, K, 2]
        pred_vis_logits: Tensor,  # [N, K]
        gt_keypoints: Tensor,     # [P, K, 2]
        gt_labeled: Tensor,       # [P, K]
        gt_visible: Tensor,       # [P, K]
        person_scale: Tensor,     # [P]
    ) -> dict[str, Tensor | int]:
        N = pred_objectness.shape[0]
        device = pred_objectness.device

        # Drop persons with no labeled joints.
        if gt_keypoints.shape[0] > 0:
            has_label = gt_labeled.sum(dim=-1) > 0
            gt_keypoints = gt_keypoints[has_label]
            gt_labeled   = gt_labeled[has_label]
            gt_visible   = gt_visible[has_label]
            person_scale = person_scale[has_label]
        P = gt_keypoints.shape[0]

        objectness_target = torch.zeros(N, device=device)
        matched_oks_vals  = pred_objectness.new_zeros(())
        matched_count     = 0
        slot_idx          = torch.zeros(0, dtype=torch.long, device=device)
        unmatched_idx     = torch.arange(N, device=device)

        if P > 0:
            oks_matrix = compute_oks(
                pred_keypoints, gt_keypoints, gt_labeled, person_scale, self.sigmas,
            )
            slot_idx, gt_idx = hungarian_match(
                oks_matrix, pred_objectness,
                lambda_oks=self.weights.matcher_lambda_oks,
                lambda_obj=self.weights.matcher_lambda_obj,
            )
            if slot_idx.numel() > 0:
                objectness_target[slot_idx] = 1.0
                matched_oks_vals = oks_matrix[slot_idx, gt_idx]
                matched_count    = slot_idx.numel()

                matched_mask = torch.zeros(N, dtype=torch.bool, device=device)
                matched_mask[slot_idx] = True
                unmatched_idx = torch.arange(N, device=device)[~matched_mask]

        mean_oks = matched_oks_vals.mean().detach() if matched_count > 0 else pred_objectness.new_zeros(())

        vis_loss = pred_objectness.new_zeros(())
        if matched_count > 0:
            vis_loss = self._visibility_loss(
                pred_vis_logits[slot_idx],
                gt_visible[gt_idx],
                gt_labeled[gt_idx],
            )

        return {
            "oks_loss":         self._oks_loss(matched_oks_vals, matched_count),
            "objectness_loss":  self._objectness_loss(pred_objectness, objectness_target, matched_count),
            "visibility_loss":  vis_loss,
            "ranking_loss":     self._ranking_loss(pred_objectness, slot_idx, unmatched_idx),
            "mean_oks":         mean_oks,
            "matched_count":    matched_count,
        }


# ---------------------------------------------------------------------------
# Box set-prediction utilities (free functions — pure math, no state)
# ---------------------------------------------------------------------------


def box_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    """Convert ``(cx, cy, w, h)`` → ``(x1, y1, x2, y2)``.

    Negative widths/heights are not corrected here; the caller is expected to
    feed sensible boxes (matcher and GIoU clamp where necessary).
    """
    cx, cy, w, h = boxes.unbind(dim=-1)
    half_w = w * 0.5
    half_h = h * 0.5
    return torch.stack(
        [cx - half_w, cy - half_h, cx + half_w, cy + half_h],
        dim=-1,
    )


def pairwise_giou(
    boxes_a: Tensor,   # [A, 4] xyxy
    boxes_b: Tensor,   # [B, 4] xyxy
    eps: float = 1e-7,
) -> Tensor:           # [A, B]
    """Pairwise generalised IoU between two sets of xyxy boxes.

    GIoU ∈ [-1, 1].  Identical boxes → 1, disjoint boxes → < 1 (and can be
    negative when the enclosing box is much larger than the union).
    """
    A = boxes_a.shape[0]
    B = boxes_b.shape[0]
    if A == 0 or B == 0:
        return boxes_a.new_zeros(A, B)

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]).clamp_min(0) * (boxes_a[:, 3] - boxes_a[:, 1]).clamp_min(0)
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]).clamp_min(0) * (boxes_b[:, 3] - boxes_b[:, 1]).clamp_min(0)

    inter_lt = torch.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])  # [A, B, 2]
    inter_rb = torch.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])  # [A, B, 2]
    inter_wh = (inter_rb - inter_lt).clamp_min(0)
    inter = inter_wh[..., 0] * inter_wh[..., 1]                            # [A, B]

    union = area_a[:, None] + area_b[None, :] - inter
    iou = inter / union.clamp_min(eps)

    encl_lt = torch.minimum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    encl_rb = torch.maximum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    encl_wh = (encl_rb - encl_lt).clamp_min(0)
    encl_area = encl_wh[..., 0] * encl_wh[..., 1]

    giou = iou - (encl_area - union) / encl_area.clamp_min(eps)
    return giou


def box_matching_cost(
    pred_boxes_cxcywh: Tensor,  # [N, 4]
    gt_boxes_cxcywh: Tensor,    # [P, 4]
    obj_logits: Tensor,         # [N]
    lambda_l1: float = 1.0,
    lambda_giou: float = 1.0,
    lambda_obj: float = 0.1,
) -> Tensor:                    # [N, P]
    """Combined L1 + GIoU + objectness cost for Hungarian matching."""
    N = pred_boxes_cxcywh.shape[0]
    P = gt_boxes_cxcywh.shape[0]
    if N == 0 or P == 0:
        return pred_boxes_cxcywh.new_zeros(N, P)

    l1_cost = (pred_boxes_cxcywh[:, None, :] - gt_boxes_cxcywh[None, :, :]).abs().sum(dim=-1)
    giou = pairwise_giou(
        box_cxcywh_to_xyxy(pred_boxes_cxcywh),
        box_cxcywh_to_xyxy(gt_boxes_cxcywh),
    )
    giou_cost = -giou
    obj_cost = -F.logsigmoid(obj_logits)[:, None].expand(N, P)
    return lambda_l1 * l1_cost + lambda_giou * giou_cost + lambda_obj * obj_cost


def hungarian_box_match(cost: Tensor) -> tuple[Tensor, Tensor]:
    """Min-cost bipartite assignment for a single image.

    Returns ``(slot_idx, gt_idx)`` of length ``min(N, P)``. Cost tensor is
    detached internally so this is safe to call inside a training step.
    """
    N, P = cost.shape
    device = cost.device
    if N == 0 or P == 0:
        empty = torch.zeros(0, dtype=torch.long, device=device)
        return empty, empty
    assignment = batch_linear_assignment(cost.detach().unsqueeze(0))   # [1, N, P]
    row, col = assignment_to_indices(assignment)
    return row[0], col[0]


# ---------------------------------------------------------------------------
# Box set-prediction loss
# ---------------------------------------------------------------------------


@dataclass
class SetBoxDiffusionLossWeights:
    """Multipliers for the components of :class:`SetBoxDiffusionLoss`."""

    l1: float = 5.0
    giou: float = 2.0
    diff_l2: float = 1.0
    obj_bce: float = 1.0
    matcher_lambda_l1: float = 5.0
    matcher_lambda_giou: float = 2.0
    matcher_lambda_obj: float = 0.1


class SetBoxDiffusionLoss(nn.Module):
    """Hungarian-matched diffusion loss for the box set decoder.

    Matched (slot, gt) pairs receive L1 + GIoU + L2 (the diffusion ``x₀``
    regression term). All slots receive a BCE objectness loss with dynamic
    ``pos_weight`` (same policy as :class:`ObjectnessLoss`).

    forward inputs
    --------------
    pred:
        pred_boxes  [B, N, 4]  cxcywh in padded-canvas canonical space
        obj_logits  [B, N]
    targets:
        gt_boxes_canon  [B, P_max, 4]  cxcywh canonical
        gt_box_labeled  [B, P_max]     {0, 1} — padded GTs have label 0

    forward outputs
    ---------------
    total          weighted sum
    l1_loss        mean L1 over matched slots
    giou_loss      mean (1 - GIoU) over matched slots
    diff_l2_loss   mean MSE over matched slots (x₀ regression)
    obj_loss       BCE over all slots with dynamic pos_weight
    matched        total matched count across batch
    """

    _max_pos_weight: float = 20.0

    def __init__(self, weights: SetBoxDiffusionLossWeights | None = None) -> None:
        super().__init__()
        self.weights = weights if weights is not None else SetBoxDiffusionLossWeights()

    def forward(
        self,
        pred: dict[str, Tensor],
        targets: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        pred_boxes = pred["pred_boxes"]   # [B, N, 4]
        obj_logits = pred["obj_logits"]   # [B, N]
        gt_boxes   = targets["gt_boxes_canon"].to(pred_boxes.device)
        gt_labeled = targets["gt_box_labeled"].to(pred_boxes.device)
        B          = pred_boxes.shape[0]

        component_keys = ("l1_loss", "giou_loss", "diff_l2_loss", "obj_loss")
        accum: dict[str, list[Tensor]] = {k: [] for k in component_keys}
        matched_total = 0
        for b in range(B):
            per = self._forward_single(
                pred_boxes[b], obj_logits[b],
                gt_boxes[b], gt_labeled[b],
            )
            for k in component_keys:
                accum[k].append(per[k])
            matched_total += per["matched_count"]

        components = {k: torch.stack(accum[k]).mean() for k in component_keys}
        w = self.weights
        total = (
            w.l1      * components["l1_loss"]
            + w.giou    * components["giou_loss"]
            + w.diff_l2 * components["diff_l2_loss"]
            + w.obj_bce * components["obj_loss"]
        )
        return {
            "total":         total,
            "l1_loss":       components["l1_loss"],
            "giou_loss":     components["giou_loss"],
            "diff_l2_loss":  components["diff_l2_loss"],
            "obj_loss":      components["obj_loss"],
            "matched":       torch.tensor(float(matched_total), device=pred_boxes.device),
        }

    def _forward_single(
        self,
        pred_boxes: Tensor,   # [N, 4]
        obj_logits: Tensor,   # [N]
        gt_boxes: Tensor,     # [P_max, 4]
        gt_labeled: Tensor,   # [P_max]
    ) -> dict[str, Tensor | int]:
        N = pred_boxes.shape[0]
        device = pred_boxes.device

        keep = gt_labeled > 0
        gt_boxes_kept = gt_boxes[keep]                  # [P, 4]
        P = gt_boxes_kept.shape[0]

        zero = pred_boxes.new_zeros(())
        objectness_target = torch.zeros(N, device=device)

        if P == 0:
            obj_loss = self._objectness_bce(obj_logits, objectness_target, matched_count=0)
            return {
                "l1_loss":       zero,
                "giou_loss":     zero,
                "diff_l2_loss":  zero,
                "obj_loss":      obj_loss,
                "matched_count": 0,
            }

        cost = box_matching_cost(
            pred_boxes, gt_boxes_kept, obj_logits,
            lambda_l1=self.weights.matcher_lambda_l1,
            lambda_giou=self.weights.matcher_lambda_giou,
            lambda_obj=self.weights.matcher_lambda_obj,
        )
        slot_idx, gt_idx = hungarian_box_match(cost)
        matched_count = slot_idx.numel()
        objectness_target[slot_idx] = 1.0

        matched_pred = pred_boxes[slot_idx]              # [M, 4]
        matched_gt   = gt_boxes_kept[gt_idx]             # [M, 4]

        l1_loss = (matched_pred - matched_gt).abs().sum(dim=-1).mean()
        giou_mat = pairwise_giou(
            box_cxcywh_to_xyxy(matched_pred),
            box_cxcywh_to_xyxy(matched_gt),
        )
        giou_loss = (1.0 - giou_mat.diagonal()).mean()
        diff_l2_loss = ((matched_pred - matched_gt) ** 2).sum(dim=-1).mean()
        obj_loss = self._objectness_bce(obj_logits, objectness_target, matched_count)

        return {
            "l1_loss":       l1_loss,
            "giou_loss":     giou_loss,
            "diff_l2_loss":  diff_l2_loss,
            "obj_loss":      obj_loss,
            "matched_count": matched_count,
        }

    def _objectness_bce(
        self,
        obj_logits: Tensor,
        target: Tensor,
        matched_count: int,
    ) -> Tensor:
        N = obj_logits.shape[0]
        pos_weight_val = min(
            (N - matched_count) / max(matched_count, 1),
            self._max_pos_weight,
        )
        pos_weight = obj_logits.new_tensor([pos_weight_val])
        return F.binary_cross_entropy_with_logits(
            obj_logits, target, pos_weight=pos_weight,
        )

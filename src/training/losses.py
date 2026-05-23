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

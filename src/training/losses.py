"""Loss functions for keypoint training.

Two families:
  * Refinement losses (joint/bone): per-(B,N,K) tensors with paired predictions.
  * Set-prediction losses (OKS-based Hungarian matching): per-image, no pairing.

Set-prediction inputs use canonical (-1, 1) coordinates. See `docs/assumptions.md`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import Tensor


# ---------------------------------------------------------------------------
# Refinement losses
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
# Set-prediction losses (Hungarian matching + OKS)
# ---------------------------------------------------------------------------


@dataclass
class SetPredictionLossWeights:
    """Multipliers for the components of `set_prediction_loss`."""

    l1: float = 5.0
    oks: float = 1.0
    obj: float = 1.0
    vis: float = 0.5
    matcher_lambda_oks: float = 1.0
    matcher_lambda_obj: float = 0.1


def compute_oks(
    pred_keypoints: Tensor,   # [N, K, 2] canonical
    gt_keypoints: Tensor,     # [P, K, 2] canonical
    gt_labeled: Tensor,       # [P, K]    {0, 1}
    person_scale: Tensor,     # [P]       canonical-space scale
    sigmas: Tensor,           # [K]
    eps: float = 1e-6,
) -> Tensor:
    """OKS matrix [N, P] between every predicted slot and every GT person.

    All inputs share the same coordinate space; sigmas are unitless multipliers
    applied to person_scale.
    """
    N, _, _ = pred_keypoints.shape
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
    oks: Tensor,                     # [N, P]
    objectness_logits: Tensor,       # [N]
    lambda_oks: float = 1.0,
    lambda_obj: float = 0.1,
) -> tuple[Tensor, Tensor]:
    """Min-cost bipartite assignment of slots to GT people.

    Cost = lambda_oks * (1 - oks) + lambda_obj * (-log_sigmoid(objectness)).
    Returns ``(slot_idx, gt_idx)`` of length ``min(N, P)``.
    """
    N, P = oks.shape
    device = oks.device
    if N == 0 or P == 0:
        empty = torch.zeros(0, dtype=torch.long, device=device)
        return empty, empty
    obj_cost = -F.logsigmoid(objectness_logits)[:, None].expand(N, P)
    cost = lambda_oks * (1.0 - oks) + lambda_obj * obj_cost
    row, col = linear_sum_assignment(cost.detach().cpu().numpy())
    return (
        torch.as_tensor(row, dtype=torch.long, device=device),
        torch.as_tensor(col, dtype=torch.long, device=device),
    )


def set_prediction_loss_per_image(
    pred_objectness: Tensor,    # [N]
    pred_keypoints: Tensor,     # [N, K, 2]  canonical
    pred_vis_logits: Tensor,    # [N, K]
    gt_keypoints: Tensor,       # [P, K, 2]  canonical
    gt_labeled: Tensor,         # [P, K]     {0, 1}
    gt_visible: Tensor,         # [P, K]     {0, 1}
    person_scale: Tensor,       # [P]
    sigmas: Tensor,             # [K]
    weights: SetPredictionLossWeights,
) -> dict[str, Tensor]:
    """Set-prediction loss for one image.

    Persons with no labeled joints are dropped before matching.
    """
    N = pred_objectness.shape[0]
    device = pred_objectness.device

    if gt_keypoints.shape[0] > 0:
        person_has_label = gt_labeled.sum(dim=-1) > 0
        gt_keypoints = gt_keypoints[person_has_label]
        gt_labeled = gt_labeled[person_has_label]
        gt_visible = gt_visible[person_has_label]
        person_scale = person_scale[person_has_label]
    P = gt_keypoints.shape[0]

    objectness_target = torch.zeros(N, device=device)
    matched_oks = pred_objectness.new_zeros(())
    l1_loss = pred_objectness.new_zeros(())
    vis_loss = pred_objectness.new_zeros(())
    matched_count = 0

    if P > 0:
        oks = compute_oks(
            pred_keypoints, gt_keypoints, gt_labeled, person_scale, sigmas,
        )
        slot_idx, gt_idx = hungarian_match(
            oks, pred_objectness,
            lambda_oks=weights.matcher_lambda_oks,
            lambda_obj=weights.matcher_lambda_obj,
        )
        if slot_idx.numel() > 0:
            objectness_target[slot_idx] = 1.0
            matched_oks = oks[slot_idx, gt_idx]
            matched_count = slot_idx.numel()

            matched_kp = pred_keypoints[slot_idx]
            matched_gt_kp = gt_keypoints[gt_idx]
            matched_labeled = gt_labeled[gt_idx]
            labeled_count = matched_labeled.sum()
            if labeled_count > 0:
                l1 = (matched_kp - matched_gt_kp).abs().sum(dim=-1)  # [M, K]
                l1_loss = (l1 * matched_labeled).sum() / labeled_count

                matched_vis_logits = pred_vis_logits[slot_idx]
                matched_vis_target = gt_visible[gt_idx]
                vis_bce = F.binary_cross_entropy_with_logits(
                    matched_vis_logits, matched_vis_target.float(),
                    reduction="none",
                )
                vis_loss = (vis_bce * matched_labeled).sum() / labeled_count

    obj_loss = F.binary_cross_entropy_with_logits(pred_objectness, objectness_target)
    oks_loss = (
        (1.0 - matched_oks).mean() if matched_count > 0
        else pred_objectness.new_zeros(())
    )
    mean_oks = (
        matched_oks.mean().detach() if matched_count > 0
        else pred_objectness.new_zeros(())
    )

    return {
        "oks": oks_loss,
        "l1": l1_loss,
        "obj": obj_loss,
        "vis": vis_loss,
        "mean_oks": mean_oks,
        "matched_count": torch.tensor(float(matched_count), device=device),
    }


def set_prediction_loss(
    pred: dict[str, Tensor],
    targets: dict[str, Tensor],
    sigmas: Tensor,
    weights: SetPredictionLossWeights,
) -> dict[str, Tensor]:
    """Batched set-prediction loss.

    ``pred`` (from a set-prediction decoder):
        - ``scores_logits``: [B, N]
        - ``keypoints``:     [B, N, K, 2] canonical
        - ``vis_logits``:    [B, N, K]

    ``targets`` (from set-format collate):
        - ``gt_keypoints``:  [B, P_max, K, 2] canonical
        - ``gt_labeled``:    [B, P_max, K]
        - ``gt_visible``:    [B, P_max, K]
        - ``person_scale``:  [B, P_max]
        - ``person_mask``:   [B, P_max]
    """
    obj_logits = pred["scores_logits"]
    pred_kp = pred["keypoints"]
    pred_vis = pred["vis_logits"]
    B = obj_logits.shape[0]
    device = obj_logits.device

    accum_keys = ("oks", "l1", "obj", "vis", "mean_oks")
    accum: dict[str, list[Tensor]] = {k: [] for k in accum_keys}
    matched_total = 0
    for b in range(B):
        P = int(targets["person_mask"][b].sum().item())
        per = set_prediction_loss_per_image(
            obj_logits[b],
            pred_kp[b],
            pred_vis[b],
            targets["gt_keypoints"][b, :P].to(device),
            targets["gt_labeled"][b, :P].to(device),
            targets["gt_visible"][b, :P].to(device),
            targets["person_scale"][b, :P].to(device),
            sigmas,
            weights,
        )
        for k in accum_keys:
            accum[k].append(per[k])
        matched_total += int(per["matched_count"].item())

    components = {k: torch.stack(accum[k]).mean() for k in accum_keys}
    total = (
        weights.l1 * components["l1"]
        + weights.oks * components["oks"]
        + weights.obj * components["obj"]
        + weights.vis * components["vis"]
    )
    return {
        "total": total,
        "oks": components["oks"],
        "l1": components["l1"],
        "obj": components["obj"],
        "vis": components["vis"],
        "mean_oks": components["mean_oks"],
        "matched": torch.tensor(float(matched_total), device=device),
    }

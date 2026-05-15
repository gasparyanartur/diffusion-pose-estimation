"""Diagnostic metrics for backbone probe validation steps."""

from __future__ import annotations

import torch
from torch import Tensor

from src.training.losses import compute_oks, hungarian_match


def _scalar(value: float) -> Tensor:
    return torch.tensor(value, device="cpu")


def _f1(tp: float, fp: float, fn: float) -> float:
    return 2 * tp / max(2 * tp + fp + fn, 1e-8)


def _match_image(
    pred_keypoints: Tensor,  # [N, K, 2]
    pred_logits: Tensor,     # [N]
    gt_keypoints: Tensor,    # [P, K, 2]
    gt_labeled: Tensor,      # [P, K]
    person_scale: Tensor,    # [P]
    sigmas: Tensor,
    lambda_oks: float,
    lambda_obj: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Returns (slot_idx, gt_idx, oks_matrix [N, P])."""
    oks_matrix = compute_oks(pred_keypoints, gt_keypoints, gt_labeled, person_scale, sigmas)
    slot_idx, gt_idx = hungarian_match(
        oks_matrix, pred_logits, lambda_oks=lambda_oks, lambda_obj=lambda_obj,
    )
    return slot_idx, gt_idx, oks_matrix


def _count_missed(
    slot_idx: Tensor,  # [M]
    gt_idx: Tensor,    # [M]
    scores: Tensor,    # [N]
    num_gt: int,
    threshold: float,
) -> int:
    matched_above = {
        g for s, g in zip(slot_idx.tolist(), gt_idx.tolist())
        if scores[s].item() >= threshold
    }
    return num_gt - len(matched_above)


def _has_duplicate(oks_matrix: Tensor, threshold: float) -> bool:  # [N, P]
    for g in range(oks_matrix.shape[1]):
        if (oks_matrix[:, g] >= threshold).sum().item() >= 2:
            return True
    return False


def _vis_stats(
    pred_vis_logits: Tensor,  # [M, K]
    gt_visible: Tensor,       # [M, K]
    gt_labeled: Tensor,       # [M, K]
) -> tuple[float, float, float, float, float, float]:
    """Returns (correct, count, tp, fp, tn, fn) over labeled joints."""
    mask = gt_labeled.bool()
    if not mask.any():
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    pred_bin = (torch.sigmoid(pred_vis_logits[mask]) > 0.5).float()
    target = gt_visible[mask].float()

    correct = (pred_bin == target).sum().item()
    count = mask.sum().item()
    tp = (pred_bin * target).sum().item()
    fp = (pred_bin * (1 - target)).sum().item()
    fn = ((1 - pred_bin) * target).sum().item()
    tn = count - tp - fp - fn
    return correct, count, tp, fp, tn, fn


def _accumulate_joint_oks(
    slot_idx: Tensor,    # [M]
    gt_idx: Tensor,      # [M]
    gt_labeled: Tensor,  # [P, K]
    oks_matrix: Tensor,  # [N, P]
    oks_sum: Tensor,     # [K]
    oks_count: Tensor,   # [K]
) -> None:
    matched_oks = oks_matrix[slot_idx, gt_idx]  # [M]
    labeled = gt_labeled[gt_idx]                # [M, K]
    for j in range(oks_sum.shape[0]):
        joint_mask = labeled[:, j].bool()       # [M]
        if joint_mask.any():
            oks_sum[j] += matched_oks[joint_mask].sum()
            oks_count[j] += joint_mask.sum().float()


def compute_probe_diagnostics(
    pred: dict[str, Tensor],
    batch: dict,
    sigmas: Tensor,
    *,
    objectness_threshold: float = 0.5,
    oks_duplicate_threshold: float = 0.5,
    matcher_lambda_oks: float = 1.0,
    matcher_lambda_obj: float = 0.1,
) -> dict[str, Tensor]:
    device = pred["scores_logits"].device
    B = pred["scores_logits"].shape[0]
    K = pred["keypoints"].shape[2]

    missed_total = 0
    gt_total = 0
    dup_images = 0
    vis_correct = vis_count = 0.0
    vis_tp = vis_fp = vis_tn = vis_fn = 0.0
    joint_oks_sum = torch.zeros(K, device=device)
    joint_oks_count = torch.zeros(K, device=device)

    for b in range(B):
        num_gt = int(batch["person_mask"][b].sum().item())
        if num_gt == 0:
            continue
        gt_total += num_gt

        gt_kp        = batch["gt_keypoints"][b, :num_gt].to(device)  # [P, K, 2]
        gt_labeled   = batch["gt_labeled"][b, :num_gt].to(device)    # [P, K]
        gt_visible   = batch["gt_visible"][b, :num_gt].to(device)    # [P, K]
        person_scale = batch["person_scale"][b, :num_gt].to(device)  # [P]
        pred_logits  = pred["scores_logits"][b]                       # [N]

        slot_idx, gt_idx, oks_matrix = _match_image(
            pred["keypoints"][b], pred_logits, gt_kp, gt_labeled,
            person_scale, sigmas, matcher_lambda_oks, matcher_lambda_obj,
        )

        missed_total += _count_missed(
            slot_idx, gt_idx, torch.sigmoid(pred_logits), num_gt, objectness_threshold,
        )
        if _has_duplicate(oks_matrix, oks_duplicate_threshold):
            dup_images += 1

        if slot_idx.numel() > 0:
            correct, count, tp, fp, tn, fn = _vis_stats(
                pred["vis_logits"][b][slot_idx], gt_visible[gt_idx], gt_labeled[gt_idx],
            )
            vis_correct += correct
            vis_count += count
            vis_tp += tp
            vis_fp += fp
            vis_tn += tn
            vis_fn += fn

            _accumulate_joint_oks(
                slot_idx, gt_idx, gt_labeled, oks_matrix, joint_oks_sum, joint_oks_count,
            )

    out: dict[str, Tensor] = {
        "missed_person_rate":    _scalar(missed_total / max(gt_total, 1)),
        "duplicate_person_rate": _scalar(dup_images / B),
        "vis_accuracy":          _scalar(vis_correct / max(vis_count, 1)),
        "vis_tp":                _scalar(vis_tp),
        "vis_fp":                _scalar(vis_fp),
        "vis_tn":                _scalar(vis_tn),
        "vis_fn":                _scalar(vis_fn),
    }
    for j in range(K):
        count = joint_oks_count[j].item()
        out[f"per_joint_oks_{j:02d}"] = _scalar(
            joint_oks_sum[j].item() / count if count > 0 else 0.0
        )
    return out

"""Set-prediction evaluation helpers (non-AP metrics).

Reports diagnostic metrics that complement the standard COCO AP/AR:
mean OKS, missed-person rate, duplicate-person rate, visibility F1,
and per-joint mean OKS.

For COCO AP/AR use `src.metrics.KeypointAPMetric` instead.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from src.metrics import CROWDPOSE_SIGMAS
from src.training.losses import compute_oks, hungarian_match


@torch.no_grad()
def evaluate_full(
    forward_fn,
    loader: DataLoader,
    device: torch.device,
    *,
    num_keypoints: int = 14,
    sigmas: Tensor | None = None,
    score_threshold: float = 0.5,
    oks_threshold: float = 0.5,
    matcher_lambda_oks: float = 1.0,
    matcher_lambda_obj: float = 0.1,
) -> dict[str, float | list[float] | int]:
    """Run the model over ``loader`` and aggregate diagnostic set-prediction metrics.

    ``forward_fn(batch) -> dict`` must return decoder output keys
    ``keypoints [B,N,K,2]`` (canonical), ``scores_logits [B,N]``,
    ``vis_logits [B,N,K]``.

    The loader must yield set-format batches: ``gt_keypoints``,
    ``gt_labeled``, ``gt_visible``, ``person_scale``, ``person_mask``.
    """
    K = num_keypoints
    if sigmas is None:
        sigmas = torch.from_numpy(CROWDPOSE_SIGMAS).float().to(device)
    else:
        sigmas = sigmas.to(device)

    matched_oks_all: list[float] = []
    per_joint_oks_sum = torch.zeros(K)
    per_joint_oks_n = torch.zeros(K)
    total_gt = 0
    matched_gt = 0
    total_active_slots = 0
    duplicate_active_slots = 0
    vis_correct = vis_total = 0
    vis_tp = vis_fp = vis_fn = 0

    for batch in loader:
        pred = forward_fn(batch)
        scores = torch.sigmoid(pred["scores_logits"])
        vis_prob = torch.sigmoid(pred["vis_logits"])
        keypoints = pred["keypoints"]
        B = keypoints.shape[0]

        for b in range(B):
            P = int(batch["person_mask"][b].sum().item())
            gt_kp = batch["gt_keypoints"][b, :P].to(device)
            gt_lab = batch["gt_labeled"][b, :P].to(device)
            gt_vis = batch["gt_visible"][b, :P].to(device)
            person_scale = batch["person_scale"][b, :P].to(device)

            active = scores[b] >= score_threshold
            total_active_slots += int(active.sum().item())
            total_gt += P

            if P == 0:
                continue

            oks = compute_oks(keypoints[b], gt_kp, gt_lab, person_scale, sigmas)
            slot_idx, gt_idx = hungarian_match(
                oks, pred["scores_logits"][b],
                lambda_oks=matcher_lambda_oks,
                lambda_obj=matcher_lambda_obj,
            )
            if slot_idx.numel() == 0:
                continue
            matched_oks = oks[slot_idx, gt_idx]
            keep = matched_oks >= oks_threshold
            matched_gt += int(keep.sum().item())
            matched_oks_all.extend(matched_oks[keep].tolist())

            if keep.any():
                kept_slot = slot_idx[keep]
                kept_gt = gt_idx[keep]
                pk = keypoints[b][kept_slot]
                gk = gt_kp[kept_gt]
                lab = gt_lab[kept_gt]
                sk = person_scale[kept_gt].clamp_min(1e-6)[:, None]
                sd = sigmas[None, :]
                d2 = ((pk - gk) ** 2).sum(dim=-1)
                e = torch.exp(-d2 / (2.0 * (sk ** 2) * (sd ** 2) + 1e-6))
                contrib = e * lab
                per_joint_oks_sum += contrib.sum(dim=0).cpu()
                per_joint_oks_n += lab.sum(dim=0).cpu()

                vp = vis_prob[b][kept_slot]
                vt = gt_vis[kept_gt]
                vmask = lab.bool()
                if vmask.any():
                    pred_bin = (vp >= 0.5).float()
                    vis_correct += int(((pred_bin == vt).float() * vmask).sum().item())
                    vis_total += int(vmask.sum().item())
                    vis_tp += int(((pred_bin == 1) & (vt == 1) & vmask).sum().item())
                    vis_fp += int(((pred_bin == 1) & (vt == 0) & vmask).sum().item())
                    vis_fn += int(((pred_bin == 0) & (vt == 1) & vmask).sum().item())

            if active.any():
                best_gt_per_slot = oks.argmax(dim=1)
                best_oks_per_slot = oks.max(dim=1).values
                act_idx = active.nonzero(as_tuple=True)[0]
                act_best_gt = best_gt_per_slot[act_idx]
                act_best_oks = best_oks_per_slot[act_idx]
                qualifying = act_best_oks >= oks_threshold
                act_best_gt_q = act_best_gt[qualifying]
                if act_best_gt_q.numel() > 0:
                    n_active_q = int(act_best_gt_q.numel())
                    n_unique = int(torch.unique(act_best_gt_q).numel())
                    duplicate_active_slots += (n_active_q - n_unique)

    mean_oks = float(np.mean(matched_oks_all)) if matched_oks_all else 0.0
    missed_rate = 1.0 - (matched_gt / total_gt) if total_gt > 0 else 0.0
    duplicate_rate = duplicate_active_slots / max(total_active_slots, 1)
    per_joint_oks = (per_joint_oks_sum / per_joint_oks_n.clamp_min(1.0)).tolist()
    vis_acc = vis_correct / max(vis_total, 1)
    vis_precision = vis_tp / max(vis_tp + vis_fp, 1)
    vis_recall = vis_tp / max(vis_tp + vis_fn, 1)
    vis_f1 = 2 * vis_precision * vis_recall / max(vis_precision + vis_recall, 1e-9)

    return {
        "mean_oks": mean_oks,
        "missed_person_rate": missed_rate,
        "duplicate_person_rate": duplicate_rate,
        "visibility_accuracy": vis_acc,
        "visibility_f1": vis_f1,
        "per_joint_mean_oks": per_joint_oks,
        "n_matched_gt": matched_gt,
        "n_total_gt": total_gt,
    }

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from scripts.evaluation.evaluate_poseflow import (
    apply_flow_sampling_steps,
    seed_flow_sampling,
    summarize_step_trajectory,
)


def test_flow_sampling_step_override_is_evaluation_local() -> None:
    model = SimpleNamespace(flow_sampling_steps=1)

    assert apply_flow_sampling_steps(model, 3) == 3
    assert model.flow_sampling_steps == 3


def test_flow_sampling_step_override_rejects_zero() -> None:
    model = SimpleNamespace(flow_sampling_steps=1)

    with pytest.raises(ValueError, match="positive"):
        apply_flow_sampling_steps(model, 0)


def test_flow_seed_is_deterministic_per_image_and_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    seeds: list[int] = []
    monkeypatch.setattr("torch.manual_seed", seeds.append)

    seed_flow_sampling(flow_seed=17, image_id=42)
    seed_flow_sampling(flow_seed=17, image_id=42)
    seed_flow_sampling(flow_seed=17, image_id=9)
    seed_flow_sampling(flow_seed=None, image_id=42)

    assert seeds == [59, 59, 26]


def test_step_trajectory_pairs_reselected_slots_and_counts_fresh_sources() -> None:
    def step(ids: list[int], fresh: list[bool], score: float, box: float) -> dict:
        return {
            "time": torch.tensor([0.0]),
            "selected_slot_ids": torch.tensor([ids]),
            "selected_from_fresh": torch.tensor([fresh]),
            "selected_state": torch.ones(1, 2, 8),
            "selected_source": torch.zeros(1, 2, 8),
            "pred_logits": torch.full((1, 2, 1), score),
            "pred_boxes": torch.full((1, 2, 4), box),
            "pred_keypoints": torch.full((1, 2, 6), box),
        }

    trajectory = summarize_step_trajectory([
        step([2, 1], [True, True], 0.0, 0.2),
        step([1, 4], [False, True], 1.0, 0.3),
    ])

    assert trajectory[1]["paired_count"] == 1
    assert trajectory[1]["selected_from_fresh"] == 1
    assert trajectory[1]["selected_from_carried"] == 1
    assert trajectory[1]["paired_box_l1"] == pytest.approx(0.1)
    assert trajectory[1]["paired_pose_l1"] == pytest.approx(0.1)

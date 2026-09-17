from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.evaluation.evaluate_poseflow import apply_flow_sampling_steps, seed_flow_sampling


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

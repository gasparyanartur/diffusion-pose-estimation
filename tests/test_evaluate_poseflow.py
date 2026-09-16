from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.evaluation.evaluate_poseflow import apply_flow_sampling_steps


def test_flow_sampling_step_override_is_evaluation_local() -> None:
    model = SimpleNamespace(flow_sampling_steps=1)

    assert apply_flow_sampling_steps(model, 3) == 3
    assert model.flow_sampling_steps == 3


def test_flow_sampling_step_override_rejects_zero() -> None:
    model = SimpleNamespace(flow_sampling_steps=1)

    with pytest.raises(ValueError, match="positive"):
        apply_flow_sampling_steps(model, 0)

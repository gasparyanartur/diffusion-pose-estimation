from __future__ import annotations

import torch
from types import SimpleNamespace

from src.eval.poseflow_diagnostics import (
    ablate_poseflow_path,
    capture_v9_attention,
    compare_feature_predictions,
    fixed_final_pairs,
    pairwise_pose_oks,
    stage_quality_counts,
    selected_stage_outputs,
)


def test_selected_stage_outputs_keeps_query_identity_across_refiners() -> None:
    def output(count: int, marker: float) -> dict[str, torch.Tensor]:
        return {
            "pred_logits": torch.full((1, count, 1), marker),
            "pred_boxes": torch.full((1, count, 4), marker),
            "pred_keypoints": torch.full((1, count, 6), marker),
        }

    initial = output(4, 0.0)
    initial["pred_logits"][0, :, 0] = torch.arange(4)
    sample = {
        **output(2, 3.0),
        "aux_outputs": [initial, output(2, 1.0), output(2, 2.0)],
        "selected_person_indices": torch.tensor([[3, 1]]),
    }

    stages = selected_stage_outputs(sample)

    assert [name for name, _ in stages] == [
        "hypothesis_all", "hypothesis_selected", "refiner_0", "refiner_1", "refiner_2",
    ]
    assert stages[1][1]["pred_logits"][0, :, 0].tolist() == [3.0, 1.0]


def test_pairwise_pose_oks_uses_visible_joints_and_returns_oracle_coverage() -> None:
    prediction = torch.tensor([[[0.5, 0.5], [0.0, 0.0]]])
    target = torch.tensor([[[50.0, 50.0], [90.0, 90.0]]])
    visibility = torch.tensor([[2.0, 0.0]])

    oks = pairwise_pose_oks(
        prediction, target, visibility,
        area=torch.tensor([1000.0]),
        image_size=(100, 100),
        sigmas=torch.tensor([0.079, 0.079]),
    )

    torch.testing.assert_close(oks, torch.ones(1, 1))


def test_stage_quality_reuses_final_person_assignment() -> None:
    final_oks = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    initial_oks = torch.tensor([[0.9, 0.1], [0.2, 0.8]])

    pairs = fixed_final_pairs(final_oks)
    counts = stage_quality_counts(initial_oks, fixed_pairs=pairs)

    assert counts["target_count"] == 2
    assert counts["covered_oks_75"] == 2
    assert counts["paired_count"] == 2
    assert counts["paired_oks_sum"] == torch.tensor(0.3).item()


def test_stage_quality_handles_images_without_labeled_people() -> None:
    oks = pairwise_pose_oks(
        torch.zeros(2, 2, 2), torch.zeros(0, 2, 2), torch.zeros(0, 2),
        area=torch.zeros(0), image_size=(100, 100), sigmas=torch.ones(2),
    )

    assert oks.shape == (2, 0)
    assert stage_quality_counts(oks)["target_count"] == 0
    assert fixed_final_pairs(oks)[0].numel() == 0


def test_feature_ablation_restores_original_forward() -> None:
    projection = torch.nn.Linear(2, 2, bias=False)
    model = SimpleNamespace(hypothesis_stage=SimpleNamespace(coordinate_projection=projection))
    value = torch.ones(1, 2)
    original = projection(value)

    with ablate_poseflow_path(model, "hypothesis_coordinate"):
        torch.testing.assert_close(projection(value), torch.zeros_like(original))

    torch.testing.assert_close(projection(value), original)


def test_feature_comparison_aligns_final_predictions_by_initial_slot() -> None:
    initial = {
        "pred_logits": torch.zeros(1, 2, 1),
        "pred_boxes": torch.full((1, 2, 4), 0.5),
        "pred_keypoints": torch.full((1, 2, 6), 0.5),
    }
    baseline = {
        **initial,
        "aux_outputs": [initial],
        "selected_person_indices": torch.tensor([[0, 1]]),
    }
    changed = {
        **{name: value[:, [1, 0]].clone() for name, value in initial.items()},
        "aux_outputs": [initial],
        "selected_person_indices": torch.tensor([[1, 0]]),
    }
    sample = {
        "img_hw": (100, 100),
        "keypoints": torch.tensor([[[50.0, 50.0], [50.0, 50.0]]]),
        "visibility": torch.ones(1, 2),
        "gt_boxes_pixel": torch.tensor([[40.0, 40.0, 20.0, 20.0]]),
    }

    comparison = compare_feature_predictions(
        baseline, changed, sample, torch.tensor([0.079, 0.079]),
    )

    assert comparison["shared_selected_count"] == 2
    assert comparison["final_shared_box_l1"] == 0.0
    assert comparison["final_shared_pose_l1"] == 0.0


def test_v9_attention_probe_measures_bias_effect_without_changing_logits() -> None:
    attention = torch.nn.MultiheadAttention(4, 2, dropout=0.0).eval()
    off_edge = torch.zeros(3, 3, dtype=torch.bool)
    off_edge[1, 2] = off_edge[2, 1] = True
    model = SimpleNamespace(
        transformer=SimpleNamespace(decoder=SimpleNamespace(
            num_group=1, num_body_points=2,
            layers=[SimpleNamespace(self_attn=attention)],
        )),
        _off_edge_pairs=off_edge,
        off_edge_attention_bias=torch.tensor([-5.0]),
    )
    queries = torch.randn(3, 1, 4)

    with capture_v9_attention(model) as biased:
        attention(queries, queries, queries, attn_mask=-5.0 * off_edge.float())
    with capture_v9_attention(model) as unbiased:
        attention(queries, queries, queries, attn_mask=torch.zeros(3, 3))

    assert len(biased) == len(unbiased) == 1
    assert biased[0]["nonedge_attention_mass"] < unbiased[0]["nonedge_attention_mass"]
    assert biased[0]["raw_logit_std"] == unbiased[0]["raw_logit_std"]

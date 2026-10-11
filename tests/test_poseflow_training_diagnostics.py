import torch

from src.eval.poseflow_training_diagnostics import (
    summarize_feature_intervention, summarize_validation_solution,
)


def test_solution_summary_tracks_selection_refinement_and_nms() -> None:
    # One labeled person, with the good hypothesis selected and refined.
    person = torch.tensor([0.5, 0.5]).repeat(14)
    far = torch.tensor([0.0, 0.0]).repeat(14)
    def stage(points, scores):
        return {
            "pred_keypoints": torch.cat((points, torch.ones(points.shape[0], 14)), -1)[None],
            "pred_boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]] * points.shape[0])[None],
            "pred_logits": torch.tensor(scores)[None, :, None],
        }

    initial = stage(torch.stack((far, person)), [0.0, 2.0])
    final = stage(person[None], [3.0])
    output = {
        **final,
        "aux_outputs": [initial],
        "selected_person_indices": torch.tensor([[1]]),
        "flow_step_diagnostics": [{
            **final,
            "selected_from_fresh": torch.tensor([[True]]),
            "selected_slot_ids": torch.tensor([[1]]),
            "selected_state": torch.zeros(1, 1, 32),
            "selected_source": torch.zeros(1, 1, 32),
        }],
    }
    batch = {
        "image": torch.zeros(1, 3, 100, 100),
        "gt_keypoints": torch.zeros(1, 1, 14, 2),
        "gt_labeled": torch.ones(1, 1, 14),
        "gt_boxes_pixel": torch.tensor([[[40.0, 40.0, 20.0, 20.0]]]),
        "person_mask": torch.ones(1, 1),
        "img_hw": torch.tensor([[100.0, 100.0]]),
        "original_img_hw": torch.tensor([[100.0, 100.0]]),
    }
    kept = [{"keypoints": torch.stack((person[0::2] * 100, person[1::2] * 100,
                                        torch.ones(14)), -1).reshape(1, -1),
             "scores": torch.tensor([0.95])}]

    summary = summarize_validation_solution(batch, output, kept, torch.full((14,), 0.1))

    assert summary["gt_count"] == 1
    assert summary["stage/hypothesis_all/covered_oks50"] == 1
    assert summary["stage/hypothesis_selected/covered_oks50"] == 1
    assert summary["stage/refiner_0/paired_oks"] == 1
    assert summary["postprocess/candidates_before_nms"] == 1
    assert summary["postprocess/candidates_after_nms"] == 1
    assert summary["flow/step_0/fresh_fraction"] == 1
    assert summary["stage/refiner_0/paired_box_iou"] == 1
    assert summary["postprocess/matched_score_count"] == 1
    feature = summarize_feature_intervention(batch, output, output, torch.full((14,), 0.1))
    assert feature["shared_fraction"] == 1
    assert feature["final_oracle_oks_change_per_gt"] == 0
    assert feature["refiner_0/paired_oks_change"] == 0


def test_solution_summary_handles_empty_gt() -> None:
    points = torch.zeros(1, 1, 42)
    prediction = {"pred_keypoints": points, "pred_logits": torch.zeros(1, 1, 1),
                  "pred_boxes": torch.zeros(1, 1, 4)}
    output = {**prediction, "aux_outputs": [prediction],
              "selected_person_indices": torch.tensor([[0]])}
    batch = {"image": torch.zeros(1, 3, 100, 100),
             "gt_keypoints": torch.zeros(1, 1, 14, 2),
             "gt_labeled": torch.zeros(1, 1, 14),
             "gt_boxes_pixel": torch.zeros(1, 1, 4),
             "person_mask": torch.zeros(1, 1),
             "img_hw": torch.tensor([[100.0, 100.0]])}
    kept = [{"keypoints": torch.zeros(0, 42), "scores": torch.zeros(0)}]

    summary = summarize_validation_solution(batch, output, kept, torch.full((14,), 0.1))

    assert summary["gt_count"] == 0
    assert summary["stage/hypothesis_all/covered_oks50"] == 0
    assert summary["postprocess/candidates_after_nms"] == 0


def test_v10_solution_collection_preserves_predictions_rng_and_scope(monkeypatch):
    from src.models.pipelines.pose_estimators.poseflow import PoseFlowV10PipelineConfig
    from src.eval import poseflow_diagnostics
    pipeline = PoseFlowV10PipelineConfig(
        backbone_pretrained=False, num_queries=4, num_proposals=2,
        num_group=2, num_encoder_layers=1, num_decoder_layers=2,
        num_feature_levels=3, hidden_dim=32, feedforward_dim=64,
        num_heads=8, num_body_points=14, num_box_decoder_layers=1,
        dn_number=0, flow_sampling_steps=2, solution_feature_diagnostics=True,
    ).build(max_epochs=2).eval()
    image = torch.randn(1, 3, 64, 64)
    generator = torch.Generator().manual_seed(42)
    rng_before = torch.random.get_rng_state()
    baseline = pipeline.model.sample_keypoints(image, generator=generator)
    expected_generator = generator.get_state()
    expected_rng = torch.random.get_rng_state()
    torch.random.set_rng_state(rng_before)
    generator.manual_seed(42)
    def forbidden_attention(*args, **kwargs):
        raise AssertionError("V10 must not collect V9 skeleton attention")
    monkeypatch.setattr(poseflow_diagnostics, "capture_v9_attention", forbidden_attention)
    output = pipeline._validation_sample(image, None, generator, collect_diagnostics=True)
    for name in ("pred_logits", "pred_boxes", "pred_keypoints"):
        torch.testing.assert_close(output[name], baseline[name], rtol=0, atol=0)
    assert torch.equal(generator.get_state(), expected_generator)
    assert torch.equal(torch.random.get_rng_state(), expected_rng)
    assert len(output["flow_step_diagnostics"]) == 2
    assert set(pipeline._validation_feature_outputs) == set(poseflow_diagnostics.FEATURE_PATHS)
    assert pipeline._validation_attention_rows == []
    assert len(pipeline._validation_latent_rows) == 4
    assert all(row["person_count"] == 2 for row in pipeline._validation_latent_rows)
    assert all(not module._forward_hooks for module in pipeline.model.modules())
    batch = {
        "image": image, "gt_keypoints": torch.zeros(1, 1, 14, 2),
        "gt_labeled": torch.ones(1, 1, 14),
        "gt_boxes_pixel": torch.tensor([[[24., 24., 16., 16.]]]),
        "person_mask": torch.ones(1, 1), "img_hw": torch.tensor([[64., 64.]]),
    }
    results = [{"keypoints": torch.zeros(0, 42), "scores": torch.zeros(0)}]
    logged = {}
    monkeypatch.setattr(pipeline, "log", lambda name, value, **kwargs: logged.update({name: value}))
    pipeline._log_validation_solution(batch, output, results, batch_index=0)
    prefix = "debug/solution/val_first_batch/"
    assert logged[prefix + "gt_count"] == 1
    assert prefix + "stage/hypothesis_all/covered_oks50" in logged
    assert prefix + "stage/refiner_1/paired_oks" in logged
    assert prefix + "flow/step_1/fresh_fraction" in logged
    assert prefix + "postprocess/candidates_after_nms" in logged
    assert any(name.startswith(prefix + "feature/") for name in logged)
    assert not any(name.startswith(prefix + "attention/") for name in logged)
    assert prefix + "latent/step_1/layer_1/update_input_rms_ratio" in logged
    assert all(torch.isfinite(torch.tensor(value)) for value in logged.values())
    logged.clear()
    pipeline._log_validation_solution(batch, output, results, batch_index=1)
    assert not logged
    output = pipeline._validation_sample(image, None, generator, collect_diagnostics=False)
    assert "flow_step_diagnostics" not in output
    assert not pipeline._validation_feature_outputs
    assert not pipeline._validation_latent_rows
    pipeline.model.flow_ensemble = True
    pipeline._log_validation_solution(batch, output, results, batch_index=0)
    assert logged == {prefix + "ensemble_skipped": 1.0}

import torch

from src.models.pipelines.pose_estimators.poseflow.quick_eval_v50 import evaluate_v50_batches, tensorboard_metrics
from tests.test_poseflow_v43_training import batch
from tests.test_poseflow_v50 import model


def test_image_only_evaluation_and_distinct_diagnostics(tmp_path):
    network = model()
    network.selector.eos_bias.data.fill_(1e6)
    sample = batch((2,))
    sample['original_img_hw'] = sample['img_hw']
    def evaluator(records, annotations, ids):
        assert records == [] and ids == [0]
        return dict(pose_AP=0., bbox_AP=0.)
    report = evaluate_v50_batches(network, [sample], tmp_path, 'unused', [0], metric_evaluator=evaluator)
    assert report['count_mae'] == 2 and report['mean_generation_steps'] == 7
    assert report['refinement']['teacher_forced/rank_01/birth_coordinate_mse'] >= 0
    assert report['refinement']['teacher_forced/rank_01/final_clean_input_coordinate_mse'] < 1e-12
    assert 'quick-test/proposal_attention_residual/rows_7/pose_AP' in tensorboard_metrics(report)
    network.selector.eos_bias.data.fill_(-1e6)
    evaluator = lambda *args: dict(pose_AP=0., bbox_AP=0.)
    first = evaluate_v50_batches(network, [sample], tmp_path/'first', 'unused', [0], metric_evaluator=evaluator)
    second = evaluate_v50_batches(network, [sample], tmp_path/'second', 'unused', [0], metric_evaluator=evaluator)
    assert first['refinement'] == second['refinement'] and first['diagnostics'] == second['diagnostics']
    assert first['capacity_stop_fraction'] == 1

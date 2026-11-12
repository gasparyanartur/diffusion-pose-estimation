import torch

from src.models.pipelines.pose_estimators.poseflow.quick_eval_v47 import evaluate_v47_batches, tensorboard_metrics
from tests.test_poseflow_v43_training import batch
from tests.test_poseflow_v47 import model


def test_image_only_evaluator_empty_predictions_and_diagnostic_labels(tmp_path):
    network = model()
    for head in network.eos_heads:
        head.weight.data.zero_()
        head.bias.data.fill_(10.)
    sample = batch((2,))
    sample['original_img_hw'] = sample['img_hw']
    def metrics(records, annotations, ids):
        assert records == [] and ids == [0]
        return dict(pose_AP=0., bbox_AP=0.)
    report = evaluate_v47_batches(network, [sample], tmp_path, 'unused', [0], metric_evaluator=metrics)
    assert report['count_mae'] == 2 and report['exact_count_images'] == 0
    assert report['refinement']['teacher_forced/rank_01/birth_coordinate_mse'] >= 0
    assert 'generated/rank_01/final_coordinate_mse' not in report['refinement']
    assert 'latent' not in str(report['refinement'])
    assert 'quick-test/ordered_coordinate/rows_8/pose_AP' in tensorboard_metrics(report)

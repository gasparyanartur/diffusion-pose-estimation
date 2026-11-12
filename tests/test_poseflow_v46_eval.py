import json

import torch

from src.models.pipelines.pose_estimators.poseflow.quick_eval_v46 import evaluate_v46_batches, tensorboard_metrics
from tests.test_poseflow_v46 import model
from tests.test_poseflow_v43_training import batch


def test_native_evaluator_separates_diagnostics_and_excludes_padding(tmp_path, monkeypatch):
    network = model()
    for head in network.eos_heads:
        head.weight.data.zero_()
        head.bias.data.fill_(8.)
    sample = batch((2,))
    sample['original_img_hw'] = sample['img_hw']
    def metrics(records, annotations, image_ids):
        assert records == [] and image_ids == [0]
        return dict(pose_AP=0., bbox_AP=0.)
    report = evaluate_v46_batches(network, [sample], tmp_path, 'unused.json', [0], metric_evaluator=metrics)
    assert report['count_mae'] == 2 and report['rows'] == 8
    assert report['refinement']['teacher_forced/rank_01/birth_latent_mse'] >= 0
    assert 'generated/rank_01/birth_latent_mse' not in report['refinement']
    assert report['refinement_counts']['teacher_forced/rank_01/birth_latent_mse'] == 1
    tags = tensorboard_metrics(report)
    assert 'quick-test/ordered_sequence/rows_8/pose_AP' in tags
    assert not any('noise_var' in key or 'progressive' in key for key in tags)
    assert json.loads((tmp_path/'metrics.json').read_text())['split'] == 'val'


def test_ddim_evaluator_labels_sampler_and_records_rank_one_trajectory(tmp_path):
    network = model()
    for head in network.eos_heads:
        head.weight.data.zero_()
        head.bias.data.fill_(-8.)
    sample = batch((2,))
    sample['original_img_hw'] = sample['img_hw']
    report = evaluate_v46_batches(network, [sample], tmp_path, 'unused.json', [0],
        sampler='ddim', collect_trajectory=True,
        metric_evaluator=lambda *_: dict(pose_AP=0., bbox_AP=0.))
    assert report['evaluation_sampler'] == 'ddim' and 'DDIM eta0' in report['sampler']
    assert len(report['trajectory']) == 16
    assert report['trajectory_counts']['trajectory/row_01/rank_01/latent_mse'] == 1
    assert abs(report['trajectory']['trajectory/row_08/rank_01/latent_mse']-
               report['refinement']['generated/rank_01/final_latent_mse']) < 1e-6
    tags = tensorboard_metrics(report)
    assert 'quick-test/ordered_sequence/ddim_eta0/rows_8/pose_AP' in tags
    assert 'quick-test/ordered_sequence/rows_8/pose_AP' not in tags

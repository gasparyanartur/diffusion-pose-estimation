import json

import pytest
import torch

from src.models.pipelines.pose_estimators.poseflow.quick_eval_v50 import (
    birth_geometry_output, evaluate_v50_batches,
)
from src.models.pipelines.pose_estimators.poseflow.quick_eval_v51 import birth_tensorboard_metrics, tensorboard_metrics
from tests.test_poseflow_v43_training import batch
from tests.test_poseflow_v50 import model
from scripts.training.evaluate_poseflow_v51_birth import verify_final_parity


def test_birth_snapshot_uses_only_second_layer_births_and_keeps_scores():
    network = model()
    coordinates = torch.zeros(2, network.capacity, 32)
    counts = torch.tensor([2, 0])
    scores = torch.rand(2, network.capacity)
    sizes = torch.tensor([[80., 40.], [40., 80.]])
    output = network._geometry_output(coordinates, counts, scores, sizes)
    output['proposal_indices'] = torch.full((2, network.capacity), -1)
    output['trace'] = []
    for rank in range(2):
        for layer in (1, 2):
            output['trace'].append(dict(layer=layer, birth=torch.tensor([True, False]),
                prefix_counts=torch.tensor([rank, 0]),
                pred_coordinates=torch.full_like(coordinates, rank+layer)))
    output['trace'].append(dict(layer=2, birth=torch.tensor([False, False]),
        prefix_counts=counts, pred_coordinates=torch.full_like(coordinates, 99.)))
    result = birth_geometry_output(network, output, sizes)
    assert torch.equal(result['pred_coordinates'][0, :2, 0], torch.tensor([2., 3.]))
    assert not result['pred_coordinates'][1].any()
    assert torch.equal(result['scores'], output['scores'])
    assert torch.equal(result['proposal_indices'], output['proposal_indices'])
    assert not output['pred_coordinates'].any()
    output['trace'].append(output['trace'][1])
    with pytest.raises(ValueError, match='birth'):
        birth_geometry_output(network, output, sizes)


@pytest.mark.parametrize('empty', [False, True])
def test_birth_ap_is_opt_in_and_logged_under_quick_test(tmp_path, empty):
    network = model()
    network.selector.eos_bias.data.fill_(1e6 if empty else -1e6)
    sample = batch((2,))
    sample['original_img_hw'] = sample['img_hw']
    calls = []

    def evaluator(records, annotations, ids):
        calls.append(records)
        assert ids == [0]
        return dict(pose_AP=.1*len(calls), bbox_AP=.2*len(calls))

    report = evaluate_v50_batches(network, [sample], tmp_path, 'unused', [0],
        metric_evaluator=evaluator, collect_birth=True)
    assert len(calls) == 2
    assert report['metrics']['pose_AP'] == .1
    assert report['birth_metrics']['pose_AP'] == .2
    assert json.loads((tmp_path/'birth_predictions.json').read_text()) == calls[1]
    if empty:
        assert calls == [[], []]
    metrics = tensorboard_metrics(report)
    assert metrics['quick-test/pose_AP'] == .1
    assert birth_tensorboard_metrics(report)['quick-test/pose_AP'] == .2
    assert all('gradient_balanced_residual' not in tag for tag in metrics)
    legacy = evaluate_v50_batches(network, [sample], tmp_path/'legacy', 'unused', [0],
        metric_evaluator=evaluator)
    assert len(calls) == 3 and 'birth_metrics' not in legacy
    assert json.loads((tmp_path/'predictions.json').read_text()) == calls[2]


def test_checkpoint_evaluation_rejects_unpaired_final_results():
    original = dict(image_ids=[1, 2], seed=42, resize=[256, 427],
        detections=3, metrics=dict(pose_AP=.2, bbox_AP=.4))
    assert verify_final_parity(original, original)['passed']
    with pytest.raises(ValueError, match='parity'):
        verify_final_parity(dict(original, metrics=dict(pose_AP=.21, bbox_AP=.4)), original)
    with pytest.raises(ValueError, match='parity'):
        verify_final_parity(dict(original, image_ids=[2, 1]), original)

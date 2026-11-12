from unittest.mock import patch

import pytest
import torch

from tests.test_poseflow_v45_task_pose import batch, settings
from src.models.pipelines.pose_estimators.poseflow.training_v45_ordered import (
    OrderedTaskPoseConfig, ordered_people,
)


def config():
    values = settings(1.).model_dump(exclude={'matching_strategy', 'target_assignment'})
    return OrderedTaskPoseConfig(**values)


def test_order_centers_centroid_size_and_stable_duplicates():
    target = dict(boxes=torch.tensor([[.7,.1,.2,.2],[.1,.8,.2,.2],
        [.1,.2,.2,.2],[.1,.2,.4,.4],[.1,.2,.4,.4]]),
        keypoints=torch.tensor([[.5,.5,1.]]).repeat(5,1))
    assert ordered_people(target).tolist() == [3,4,2,1,0]
    target['keypoints'][2,0] = .1
    assert ordered_people(target).tolist() == [2,3,4,1,0]


@pytest.mark.parametrize('empty,partial', [(False,False),(True,False),(False,True)])
def test_no_hungarian_rank_sources_gradients_and_metadata(empty,partial):
    pipeline = config().build(max_epochs=80)
    value = batch(empty,partial)
    with patch('src.models.pipelines.pose_estimators.diffusionregpose.training.linear_sum_assignment',
               side_effect=AssertionError('Hungarian forbidden')):
        losses = pipeline._losses_for_batch(value)
    assert torch.isfinite(losses['total'])
    losses['total'].backward()
    assert pipeline.contract()['assignment'] == 'ordered_rank_512'
    assert all(p.grad is None for p in pipeline.teacher.parameters())
    assert pipeline.model.rank_features.weight.grad is not None
    assert pipeline.model.image_confidence_head.weight.grad.abs().sum() > 0
    if not empty:
        assert pipeline.model.rank_references.weight.grad.abs().sum() > 0
        assert pipeline.model.pose_codec.reader[-2].weight.grad.abs().sum() > 0


@pytest.mark.parametrize('steps',[1,4,8])
def test_gt_free_fixed_rank_context_and_restore(steps):
    pipeline = config().build(max_epochs=80)
    model = pipeline.model.eval()
    images = torch.randn(1,3,64,80)
    with torch.no_grad(), patch.object(model,'prepare_context',wraps=model.prepare_context) as prepare:
        model.sampling_timesteps = steps
        prediction = model.sample_keypoints(images,generator=torch.Generator().manual_seed(42))
    assert prepare.call_count == 1
    assert prediction['reference_indices'].tolist() == [[0,1]]
    assert not prediction['pred_boxes'].requires_grad
    assert torch.equal(prediction['pred_boxes'],prediction['reference_boxes'])
    restored = config().build(max_epochs=80)
    restored.load_state_dict(pipeline.state_dict(),strict=True)
    record = {}
    pipeline.on_save_checkpoint(record)
    restored.on_load_checkpoint(record)
    record['v45_task_pose']['assignment'] = 'wrong'
    with pytest.raises(ValueError):
        restored.on_load_checkpoint(record)


def test_rank_assignment_covers_targets_and_rejects_overflow():
    pipeline = config().build(max_epochs=80)
    target = dict(labels=torch.ones(2,dtype=torch.long),
        boxes=torch.tensor([[.8,.5,.1,.2],[.1,.5,.1,.2]]),
        keypoints=torch.tensor([[.5,.5,1.],[.5,.5,1.]]))
    output = dict(pred_logits=torch.zeros(1,2,2),pred_boxes=torch.rand(1,2,4))
    matches = pipeline.criterion.matcher(output,[target])
    assert matches[0][0].tolist() == [0,1]
    assert matches[0][1].tolist() == [1,0]
    output['pred_logits'] = output['pred_logits'][:,:1]
    with pytest.raises(ValueError):
        pipeline.criterion.matcher(output,[target])

from copy import deepcopy
import json

import pytest
import torch

from src.models.pipelines.pose_estimators.poseflow.alignment_v43 import evaluate_alignment_batches
from src.models.pipelines.pose_estimators.poseflow.training_v43 import PoseFlowV43Pipeline, PoseFlowV43PipelineConfig
from tests.test_poseflow_v43 import model
from tests.test_poseflow_v43_training import batch, pipeline


def alignment_pipeline():
    original = pipeline()
    config = original.config.model_copy(update=dict(training_task='image_alignment', posterior_mode='mean',
        center_min_side_px=1., scheduler_type='cosine'))
    return PoseFlowV43Pipeline(model=original.model, config=config, max_epochs=10).eval()


def test_image_only_contract_decoder_gradients_and_determinism(monkeypatch):
    network = model()
    image = torch.randn(1, 3, 128, 128, requires_grad=True)
    image_wh = torch.tensor([[128., 101.]])
    def forbidden(*args, **kwargs):
        raise AssertionError('Image-only inference cannot encode target geometry')
    monkeypatch.setattr(network.codec.model, 'encode', forbidden)
    rng = torch.get_rng_state().clone()
    first = network.predict_image(image, image_wh=image_wh)
    second = network.predict_image(image, image_wh=image_wh)
    assert torch.equal(torch.get_rng_state(), rng)
    assert torch.equal(first['pred_latents'], second['pred_latents'])
    expected = network.codec.decode_latents(first['pred_latents'], image_wh[:, None].expand(-1, 6, -1))
    assert torch.equal(first['keypoints_px'], expected['keypoints_px'])
    first['keypoints_px'].square().mean().backward()
    assert image.grad.abs().sum() > 0
    assert network.latent_output.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in network.codec.parameters())


def test_alignment_loss_uses_targets_only_for_supervision_and_no_random_draws(monkeypatch):
    module = alignment_pipeline()
    observed = []
    original = module.model.predict_image_context
    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        observed.append(result['pred_latents'].detach().clone())
        return result
    monkeypatch.setattr(module.model, 'predict_image_context', capture)
    data = batch((1,))
    states = module.training_rng_state.clone(), module.posterior_rng_state.clone()
    first = module._losses_for_batch(data)
    altered = deepcopy(data)
    altered['gt_keypoints'] += .1
    second = module._losses_for_batch(altered)
    assert torch.equal(observed[0], observed[1])
    assert first['loss_latent'] != second['loss_latent']
    assert torch.equal(module.training_rng_state, states[0])
    assert torch.equal(module.posterior_rng_state, states[1])
    first['total'].backward()
    assert module.model.latent_output.weight.grad.abs().sum() > 0
    assert module.model.input_proj[0][0].weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in module.model.codec.parameters())
    empty = module._losses_for_batch(batch((0,)))
    assert torch.isfinite(empty['total']) and empty['geometry/total'] == 0


def test_task_checkpoint_semantics_and_frozen_restore():
    original = alignment_pipeline()
    record = {}
    original.on_save_checkpoint(record)
    assert record['v43_training_task'] == 'image_alignment'
    with pytest.raises(ValueError, match='task'):
        pipeline().on_load_checkpoint(record)
    with pytest.raises(ValueError, match='task'):
        original.on_load_checkpoint({})
    restored = alignment_pipeline()
    restored.on_load_checkpoint(record)
    restored.load_state_dict(deepcopy(original.state_dict()), strict=True)
    data = batch((1,))
    assert torch.equal(original.model.predict_image(data['image'], image_wh=data['img_hw'].flip(-1))['pred_latents'],
        restored.model.predict_image(data['image'], image_wh=data['img_hw'].flip(-1))['pred_latents'])
    with pytest.raises(ValueError, match='alignment'):
        PoseFlowV43PipelineConfig(training_task='image_alignment', prediction_type='v', posterior_mode='mean')


def test_alignment_evaluation_coverage_pair_identity_and_no_rng(tmp_path):
    module = alignment_pipeline()
    batches = [batch((1,)) for _ in range(3)]
    ids = [101, 202, 303]
    for item, image_id in zip(batches, ids):
        item['image_id'] = torch.tensor([image_id])
    state = deepcopy(module.model.state_dict())
    rng = torch.get_rng_state().clone()
    result = evaluate_alignment_batches(module.model, batches, tmp_path, 'unused.json', ids,
        matcher=module.matcher, metric_evaluator=lambda records, annotations, image_ids: dict(records=len(records)))
    assert result['image_ids'] == ids
    assert result['paired_people'] == 3
    assert all(pair['image_id'] != pair['donor_image_id'] for pair in result['conditioning_pairs'])
    assert set(result['conditions']) == {'correct_image', 'wrong_image'}
    assert result['reference_boxes_scores_unchanged']
    assert all(torch.isfinite(torch.tensor(row['paired']['paired_OKS_mean'])) for row in result['conditions'].values())
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(value, module.model.state_dict()[name]) for name, value in state.items())


def test_alignment_training_split_must_exclude_validation(tmp_path):
    from scripts.training.train_poseflow_v43_alignment import alignment_splits
    train, val = tmp_path/'train.json', tmp_path/'val.json'
    train.write_text(json.dumps(dict(images=[dict(id=1)])))
    val.write_text(json.dumps(dict(images=[dict(id=2)])))
    assert alignment_splits(train, val)['overlap'] == 0
    val.write_text(json.dumps(dict(images=[dict(id=1)])))
    with pytest.raises(ValueError, match='disjoint'):
        alignment_splits(train, val)

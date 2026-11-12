from copy import deepcopy

import pytest
import torch

from src.models.pipelines.pose_estimators.poseflow.model_v44 import PoseFlowV44Model
from src.models.pipelines.pose_estimators.poseflow.model_v45 import PoseFlowV45Model
from src.models.pipelines.pose_estimators.poseflow.training_v45 import (
    PoseFlowV45Pipeline, PoseFlowV45PipelineConfig, progressive_weights,
)
from src.models.pipelines.pose_autoencoder.model import geometry_loss, geometry_loss_per_person
from tests.test_poseflow_v44 import arguments
from tests.test_poseflow_v43_codec import small_codec
from tests.test_poseflow_v43_training import batch


def model():
    return PoseFlowV45Model(codec=small_codec(), **arguments()).eval()


def pipeline():
    config = PoseFlowV45PipelineConfig(native=arguments(), codec_width=32, latent_dim=8,
        pretrained_checkpoint=None, quick_eval_enabled=False)
    return PoseFlowV45Pipeline(config=config, model=model(), max_epochs=2).eval()


def test_context_parity_and_zero_predict_token():
    original = PoseFlowV44Model(codec=small_codec(), **arguments()).eval()
    network = model()
    common = {name: value for name, value in original.state_dict().items() if name in network.state_dict()}
    network.load_state_dict(common, strict=False)
    images = torch.randn(2, 3, 128, 96)
    old, new = original.prepare_context(images), network.prepare_context(images)
    for name in ('features', 'boxes', 'indices', 'existence_logits', 'memory'):
        torch.testing.assert_close(old[name], new[name], rtol=0, atol=0)
    assert torch.count_nonzero(network.predict_token) == 0
    assert not hasattr(network, 'latent_time') and not hasattr(network, 'latent_input')


def test_prefix_growth_shared_layers_and_deterministic_image_only_inference(monkeypatch):
    network = model()
    images, sizes = torch.randn(1, 3, 128, 96), torch.tensor([[96., 128.]])
    counts = []
    hooks = [layer.register_forward_pre_hook(lambda module, args: counts.append(len(args[0])))
        for layer in network.transformer.decoder.layers[1:]]
    monkeypatch.setattr(network.codec, 'encode_posterior', lambda *_: pytest.fail('Inference used GT'))
    output = network.sample_keypoints(images, image_wh=sizes, collect_trace=True)
    for hook in hooks:
        hook.remove()
    assert counts == [value for stage in range(1, 7) for value in (stage, stage)]
    assert [(row['stage'], row['layer']) for row in output['trace']] == [
        (stage, layer) for stage in range(1, 7) for layer in (1, 2)]
    assert output['pred_latents'].shape == (1, 6, 8)
    for row in output['trace']:
        assert torch.equal(row['reference_indices'], output['reference_indices'][:, :row['stage']])
    again = network.sample_keypoints(images, image_wh=sizes)
    assert 'trace' not in again
    torch.testing.assert_close(output['pred_latents'], again['pred_latents'], rtol=0, atol=0)


def test_no_future_leakage_and_later_active_person_changes_earlier_predictions():
    network = model()
    context = network.prepare_context(torch.randn(1, 3, 128, 96))
    sizes = torch.tensor([[96., 128.]])
    original = network.predict(context, sizes, collect_trace=True)
    changed = dict(context, features=context['features'].clone())
    changed['features'][:, 2] += torch.linspace(-10, 10, 32)
    perturbed = network.predict(changed, sizes, collect_trace=True)
    for a, b in zip(original['trace'][:4], perturbed['trace'][:4]):
        torch.testing.assert_close(a['pred_latents'], b['pred_latents'], rtol=0, atol=0)
    assert not torch.allclose(original['trace'][5]['pred_latents'][:, 0],
                              perturbed['trace'][5]['pred_latents'][:, 0])


def test_every_person_has_equal_accumulated_weight():
    accumulated = torch.zeros(50)
    for stage in range(1, 51):
        for _ in range(2):
            accumulated[:stage] += progressive_weights(50)[:stage]
    torch.testing.assert_close(accumulated, torch.ones_like(accumulated))


@pytest.mark.parametrize('counts', [(2, 1), (0,)])
def test_fixed_assignment_losses_gradients_and_frozen_codec(counts):
    module = pipeline()
    sample = batch(counts)
    sample['gt_labeled'] = sample['gt_labeled'].clone()
    sample['gt_labeled'][..., 3] = False
    before = deepcopy(module.model.codec.state_dict())
    matches, contexts = [], []
    hook = module.slot_matcher.register_forward_hook(lambda *_: matches.append(1))
    prepare = module.model.prepare_context
    def context(*args, **kwargs):
        contexts.append(1)
        return prepare(*args, **kwargs)
    module.model.prepare_context = context
    losses = module._losses_for_batch(sample)
    hook.remove()
    assert len(matches) == len(contexts) == 1
    assert torch.isfinite(losses['total'])
    assert losses['debug/positive_slots'] == sum(counts)
    losses['total'].backward()
    assert module.model.predict_token.grad.abs().sum() > 0
    assert all(head.weight.grad.abs().sum() > 0 for head in module.model.latent_heads)
    assert all(value.grad is None for value in module.model.codec.parameters())
    for name, value in before.items():
        assert torch.equal(value, module.model.codec.state_dict()[name])


def test_checkpointed_stage_gradient_parity():
    plain = model().train()
    checked = deepcopy(plain)
    plain.activation_checkpointing = False
    checked.activation_checkpointing = True
    image, sizes = torch.randn(1, 3, 128, 96), torch.tensor([[96., 128.]])
    predictions = []
    for network in (plain, checked):
        torch.manual_seed(81)
        output = network.sample_keypoints(image, image_wh=sizes)
        output['keypoints_px'].square().mean().backward()
        predictions.append(output['pred_latents'])
    torch.testing.assert_close(*predictions, rtol=0, atol=0)
    for (name, value), (_, other) in zip(plain.named_parameters(), checked.named_parameters()):
        assert (value.grad is None) == (other.grad is None), name
        if value.grad is not None:
            torch.testing.assert_close(value.grad, other.grad, rtol=2e-5, atol=2e-5, msg=name)


def test_geometry_per_person_keeps_mean_api_exact():
    target = dict(keypoints_px=torch.randn(3, 14, 2), boxes_cxcywh_px=torch.tensor([[20., 30., 10., 15.]]).repeat(3, 1),
        joint_mask=torch.rand(3, 14) > .2)
    predicted = dict(keypoints_px=target['keypoints_px'] + .1,
        boxes_cxcywh_px=target['boxes_cxcywh_px'] + .2)
    per_person = geometry_loss_per_person(predicted, target)
    reduced = geometry_loss(predicted, target)
    for name, value in per_person.items():
        torch.testing.assert_close(value.mean(), reduced[name], rtol=0, atol=0)


def test_strict_contract_and_capacity():
    module = pipeline()
    record = {}
    module.on_save_checkpoint(record)
    restored = pipeline()
    restored.on_load_checkpoint(record)
    restored.load_state_dict(module.state_dict(), strict=True)
    incompatible = deepcopy(record)
    incompatible['v45_contract']['attention'] = 'causal'
    with pytest.raises(ValueError, match='contract'):
        restored.on_load_checkpoint(incompatible)
    module.model.num_proposals = 1
    with pytest.raises(ValueError, match='capacity'):
        module._losses_for_batch(batch((2,)))


def test_native_evaluator_and_tensorboard_tags(tmp_path):
    from src.models.pipelines.pose_estimators.poseflow.quick_eval_v45 import evaluate_v45_batches, tensorboard_metrics
    network = model()
    sample = batch((2,))
    sample['original_img_hw'] = sample['img_hw']
    seen = []
    def metrics(records, annotations, ids):
        seen.append((records, ids))
        return dict(pose_AP=1.5, bbox_AP=2.5)
    report = evaluate_v45_batches(network, [sample], tmp_path, 'unused.json', [0], metric_evaluator=metrics)
    assert len(seen) == 1 and report['images'] == 1
    tags = tensorboard_metrics(report)
    assert tags['quick-test/progressive/stages_6/pose_AP'] == 1.5
    assert tags['quick-test/progressive/stages_6/bbox_AP'] == 2.5
    assert any('birth' in name for name in tags)
    assert not any('noise' in name or 'steps_' in name for name in tags)


def test_carried_states_are_not_reset_or_reinitialized():
    network = model()
    captures = []
    def stage(features, references, memory, inputs):
        captures.append(features.clone())
        return features+1, features+2
    network._stage = stage
    context = dict(features=torch.zeros(1, 6, 32), boxes=torch.zeros(1, 6, 4), memory=None, inputs=None)
    list(network.latent_predictions(context))
    for stage_index, states in enumerate(captures):
        for person in range(stage_index+1):
            torch.testing.assert_close(states[person], torch.full_like(states[person], 2*(stage_index-person)))


def test_normalized_losses_match_single_prediction_reference():
    from src.models.pipelines.pose_estimators.poseflow.latent_v43 import assign_latent_sources, geometry_targets
    module = pipeline()
    sample = batch((2, 1))
    geometry, targets = geometry_targets(sample)
    context = module.model.prepare_context(sample['image'])
    matches = module.slot_matcher(dict(pred_boxes=context['boxes'], pred_logits=context['pred_logits']), targets, include_keypoints=False)
    sources = assign_latent_sources(module.model.codec, geometry, targets, matches,
        slots=6, generator=torch.Generator(), posterior_mode='mean')
    positive = sources['positive']
    predicted = sources['latents'] + torch.where(positive[..., None], 1., 2.)
    module.model.prepare_context = lambda *args, **kwargs: context
    module.model.latent_predictions = lambda context: (
        (stage, layer, predicted[:, :stage]) for stage in range(1, 7) for layer in (1, 2))
    losses = module._losses_for_batch(sample)
    torch.testing.assert_close(losses['latent_positive'], torch.tensor(1.))
    torch.testing.assert_close(losses['latent_background'], torch.tensor(4.))
    sizes = sample['img_hw'].flip(-1)[:, None].expand(-1, 6, -1)[positive]
    decoded = module.model.codec.decode_latents(predicted[positive], sizes)
    rows = sources['target_indices'][positive]
    expected = geometry_loss(decoded, {name: value[rows] for name, value in geometry.items()})['loss']
    torch.testing.assert_close(losses['geometry'], expected)


def test_fresh_config_build_without_assets_returns_v45():
    config = PoseFlowV45PipelineConfig(native=arguments(), codec_width=32, latent_dim=8,
        pretrained_checkpoint=None, quick_eval_enabled=False)
    restored = config.build(max_epochs=80, load_pretrained_checkpoint=False)
    assert isinstance(restored, PoseFlowV45Pipeline)
    assert isinstance(restored.model, PoseFlowV45Model)
    assert restored.model.activation_checkpointing

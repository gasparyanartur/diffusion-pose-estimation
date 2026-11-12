from copy import deepcopy
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from tests.test_poseflow_v35 import batch


def pipeline(slots=6, **overrides):
    from src.models.pipelines.pose_estimators.poseflow.training_v41 import PoseFlowV41PipelineConfig
    options = dict(backbone_pretrained=False, pretrained_checkpoint=None, hidden_dim=32,
        num_heads=4, feedforward_dim=64, num_queries=slots, num_proposals=slots,
        num_group=slots, grid_rows=1, grid_columns=slots, reference_candidates=max(12, slots),
        reference_encoder_source_layers=(2,), quick_eval_enabled=False)
    return PoseFlowV41PipelineConfig(**dict(options, **overrides)).build(max_epochs=80)


def test_recipe_and_coordinate_forward_noise_are_independent_of_embeddings():
    from src.models.pipelines.pose_estimators.poseflow.coordinates_v41 import build_coordinate_targets
    with initialize_config_dir(config_dir=str(Path('conf').resolve()), version_base=None):
        config = compose(config_name='benchmark_poseflow_v41')
    assert config.training.max_epochs == 80 and config.training.seed == 42
    assert config.pipeline.matching_strategy == 'hungarian' and config.pipeline.coordinate_scale == 6
    module = pipeline().eval()
    model = module.model
    data = batch()
    targets = build_coordinate_targets(data, slots=6, scale=6., generator=torch.Generator().manual_seed(2))
    assert targets['positive'].sum(-1).tolist() == [2, 1]
    torch.testing.assert_close(targets['source_coordinates'][targets['positive']],
                               6*(2*targets['boxes'][targets['positive']]-1))
    assert not targets['source_coordinates'].requires_grad
    times = torch.tensor([0, 999])
    noise = torch.randn(2, 6, 4)
    clean, predicted, noisy, returned_times = model.training_outputs(data['image'], targets['boxes'],
        source_coordinates=targets['source_coordinates'], timesteps=times, noise=noise)
    expected = model.sqrt_alphas_cumprod[times, None, None]*targets['source_coordinates']
    expected += model.sqrt_one_minus_alphas_cumprod[times, None, None]*noise
    torch.testing.assert_close(noisy, expected.float())
    assert torch.equal(returned_times, times) and noisy.shape == (2, 6, 4)
    assert clean['pred_embeddings'].shape == predicted['pred_embeddings'].shape == (2, 6, 32)
    assert 'pred_latents' not in predicted and hasattr(model, 'propose')
    sampled = torch.cat([module._draw_training_inputs(data)[1] for _ in range(16)])
    assert sampled.min() < 333 and sampled.max() >= 666 and sampled.unique().numel() > 2


def test_output_hungarian_assignment_controls_geometry_and_live_embedding_targets():
    from src.models.pipelines.pose_estimators.poseflow.coordinates_v41 import (
        assign_coordinate_targets, build_coordinate_targets, matched_embedding_loss,
    )
    module = pipeline(slots=3)
    data = batch((2,))
    targets = build_coordinate_targets(data, slots=3, scale=6., generator=torch.Generator().manual_seed(3))
    embeddings = torch.randn(1, 3, 32, requires_grad=True)
    boxes = torch.stack((data['gt_boxes_canon'][0, 1], torch.ones(4)*.5,
                         data['gt_boxes_canon'][0, 0]))[None].requires_grad_()
    logits = torch.tensor([[4., -4., 4.]], requires_grad=True)
    output = dict(pred_boxes=boxes, existence_logits=logits,
                  pred_logits=torch.stack((logits-20, logits), -1))
    assigned, gathered, indices = assign_coordinate_targets(output, embeddings, targets, module.matcher)
    assert indices.tolist() == [[1, -1, 0]]
    assert assigned['positive'].tolist() == [[True, False, True]]
    torch.testing.assert_close(assigned['boxes'][0, [0, 2]], boxes.detach()[0, [0, 2]])
    prediction = torch.randn_like(embeddings, requires_grad=True)
    loss = matched_embedding_loss(prediction, gathered, assigned['positive'])
    loss.backward()
    assert embeddings.grad[0, :2].abs().sum() > 0 and embeddings.grad[0, 2].abs().sum() == 0
    assert prediction.grad[0, 1].abs().sum() == 0
    targets['source_coordinates'].normal_(100, 50)
    other = assign_coordinate_targets(output, embeddings, targets, module.matcher)
    assert torch.equal(indices, other[2])


@pytest.mark.parametrize('loss_name', ['loss_jepa', 'denoised/total', 'regularization/total'])
def test_shared_encoder_receives_independent_prediction_and_regularizer_gradients(loss_name):
    module = pipeline().eval()
    captured = []
    handle = module.model.clean_norm.register_forward_hook(lambda _, args, output: captured.append(output))
    losses = module._losses_for_batch(batch(), training=False)
    handle.remove()
    assert len(captured) == 2
    for codes in captured:
        codes.retain_grad()
    losses[loss_name].backward()
    if loss_name != 'denoised/total':
        assert captured[0].grad is not None and captured[0].grad.abs().sum() > 0
    if loss_name != 'regularization/total':
        assert captured[1].grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.model.clean_layers.parameters())


def test_masks_empty_single_invalid_full_budget_and_objective():
    module = pipeline().eval()
    for counts in ((0,), (1,), (2,), (0, 2)):
        losses = module._losses_for_batch(batch(counts), training=False)
        expected = losses['proposal/total']+.5*losses['clean/total']+.5*losses['denoised/total']
        expected += 10*losses['loss_jepa']+.05*losses['regularization/sigreg']
        torch.testing.assert_close(losses['total'], expected)
        assert all(torch.isfinite(value) for value in losses.values())
        assert module._diagnostics['sigreg/count'] == sum(counts)
        if not sum(counts):
            assert losses['loss_jepa'] == losses['denoised/loss_bbox'] == losses['clean/loss_bbox'] == 0
            losses['total'].backward()
            assert module.model.existence_head.bias.grad.abs().sum() > 0
            module.zero_grad(set_to_none=True)
    data = batch((2,))
    data['gt_boxes_canon'][0, 0, 0] = float('nan')
    data['gt_box_labeled'][0, 1] = False
    assert torch.isfinite(module._losses_for_batch(data)['total'])
    assert module._diagnostics['sigreg/count'] == 0
    full = pipeline(slots=50).eval()
    data = batch((1,))
    data['gt_boxes_canon'] = torch.rand(1, 50, 4)*.6+.1
    data['person_mask'] = data['gt_box_labeled'] = torch.ones(1, 50, dtype=torch.bool)
    assert torch.isfinite(full._losses_for_batch(data)['total'])
    assert full._diagnostics['sigreg/count'] == 50
    with pytest.raises(ValueError, match='budget'):
        pipeline(slots=1)._losses_for_batch(batch((2,)))


def test_batch_isolation_permutation_and_coordinate_clipping():
    model = pipeline().eval().model
    images = batch()['image']
    coordinates = torch.randn(2, 6, 4)*10
    times = torch.tensor([999, 120])
    with torch.no_grad():
        together = model(images, coordinates, times)['pred_boxes']
        independent = torch.cat([model(images[i:i+1], coordinates[i:i+1], times[i:i+1])['pred_boxes'] for i in range(2)])
        torch.testing.assert_close(together, independent, atol=2e-6, rtol=2e-6)
        order = torch.tensor([4, 0, 2, 5, 1, 3])
        context = model.encode_context(images)
        references = model.select_references(model.propose(context))
        permuted_references = {k:v[:, order] for k,v in references.items()}
        permuted = model.denoise(coordinates[:, order], times, context['image'],
                                 references=permuted_references)['pred_boxes']
        torch.testing.assert_close(permuted, together[:, order], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(model.coordinates_to_boxes(torch.tensor([-100., 0., 100.])),
                               torch.tensor([0., .5, 1.]))


def test_gaussian_coordinate_sampling_caches_image_and_reencodes_each_step(monkeypatch):
    model = pipeline().eval().model
    calls, states, views, proposal_calls, references = [], [], [], [], []
    encode_image, denoise, encode_coordinates = model.encode_context, model.denoise, model.encode_coordinates
    propose = model.propose
    def proposals(*args, **kwargs):
        proposal_calls.append(1)
        return propose(*args, **kwargs)
    def image(*args, **kwargs):
        calls.append(1)
        return encode_image(*args, **kwargs)
    def predict(state, *args, **kwargs):
        states.append(state.clone())
        references.append(kwargs['references'])
        return denoise(state, *args, **kwargs)
    def encode(boxes, *args, **kwargs):
        views.append(boxes.clone())
        return encode_coordinates(boxes, *args, **kwargs)
    monkeypatch.setattr(model, 'encode_context', image)
    monkeypatch.setattr(model, 'propose', proposals)
    monkeypatch.setattr(model, 'denoise', predict)
    monkeypatch.setattr(model, 'encode_coordinates', encode)
    model.flow_sampling_steps = 4
    output = model.sample_boxes(batch((1,))['image'], generator=torch.Generator().manual_seed(8),
                                collect_step_diagnostics=True)
    assert len(calls) == len(proposal_calls) == 1 and len(states) == len(views) == 4
    assert all(item is references[0] for item in references)
    expected = torch.randn(1, 6, 4, generator=torch.Generator().manual_seed(8))
    torch.testing.assert_close(states[0], expected, atol=0, rtol=0)
    assert all(state.shape == (1, 6, 4) for state in states)
    assert not torch.equal(views[0], views[1])
    assert output['coordinate_state'].shape == (1, 6, 4) and 'latent_state' not in output
    assert len(output['coordinate_step_diagnostics']) == 4
    torch.testing.assert_close(output['coordinate_state'], model.boxes_to_coordinates(output['pred_boxes']))
    time, next_time = model.ddim_timestep_pairs()[0]
    clean = model.boxes_to_coordinates(output['pred_boxes'])
    noise = model.predict_noise_from_start(states[0], torch.tensor([time]), clean).float()
    expected = model.alphas_cumprod[next_time].sqrt()*clean+(1-model.alphas_cumprod[next_time]).sqrt()*noise
    torch.testing.assert_close(model.ddim_step(clean, noise, time, next_time), expected.float())


def test_strict_optimizer_scheduler_rng_restore_and_validation_independence():
    torch.manual_seed(3)
    module = pipeline().eval()
    data = batch()
    before = {name: getattr(module, name).clone() for name in ('training_rng_state', 'sigreg_rng_state')}
    first = module._losses_for_batch(data, training=False)
    second = module._losses_for_batch(data, training=False)
    torch.testing.assert_close(first['total'], second['total'], atol=0, rtol=0)
    assert all(torch.equal(value, getattr(module, name)) for name, value in before.items())
    configured = module.configure_optimizers()
    optimizer = configured['optimizer']
    module._losses_for_batch(data)['total'].backward()
    optimizer.step()
    restored = pipeline().eval()
    restored.load_state_dict(deepcopy(module.state_dict()), strict=True)
    other = restored.configure_optimizers()
    other['optimizer'].load_state_dict(deepcopy(optimizer.state_dict()))
    scheduler = configured['lr_scheduler']['scheduler']
    scheduler.step()
    other['lr_scheduler']['scheduler'].load_state_dict(deepcopy(scheduler.state_dict()))
    for owner, opt in ((module, optimizer), (restored, other['optimizer'])):
        opt.zero_grad()
        owner._losses_for_batch(data)['total'].backward()
        opt.step()
    for name, value in module.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], atol=0, rtol=0)
    assert scheduler.state_dict() == other['lr_scheduler']['scheduler'].state_dict()
    with torch.no_grad():
        for steps in (1, 4):
            module.model.flow_sampling_steps = restored.model.flow_sampling_steps = steps
            left = module.model.sample_boxes(data['image'], generator=torch.Generator().manual_seed(9))
            right = restored.model.sample_boxes(data['image'], generator=torch.Generator().manual_seed(9))
            torch.testing.assert_close(left['pred_boxes'], right['pred_boxes'], atol=0, rtol=0)


def test_initialization_mapping_retains_full_reference_front_end():
    model = pipeline().model
    assert model.pretrained_source_name('proposal_encoder.layers.0.norm1.weight', (2,)) == 'transformer.encoder.layers.1.norm1.weight'
    assert model.pretrained_source_name('denoiser_layers.0.modulation.weight', (2,)) is None
    assert hasattr(model, 'proposal_box_head') and hasattr(model, 'reference_box_embedding')


def test_proposal_and_final_assignments_are_separate_and_references_remain_live(monkeypatch):
    module = pipeline().eval()
    for layer in module.model.denoiser_layers:
        torch.nn.init.normal_(layer.modulation.weight, std=.02)
    captured, calls = {}, []
    select = module.model.select_references
    def selected(candidates):
        refs = select(candidates)
        refs['features'].retain_grad()
        refs['boxes'].retain_grad()
        expected = candidates['pred_logits'].amax(-1).argsort(descending=True, stable=True)[:, :6]
        assert torch.equal(refs['indices'], expected)
        captured.update(refs)
        return refs
    monkeypatch.setattr(module.model, 'select_references', selected)
    handle = module.matcher.register_forward_hook(lambda owner,args,result:calls.append(args[0]['pred_boxes'].shape[1]))
    losses = module._losses_for_batch(batch())
    handle.remove()
    assert calls == [12, 6]
    losses['loss_jepa'].backward()
    assert captured['features'].grad.abs().sum() > 0 and captured['boxes'].grad.abs().sum() > 0


def test_final_assignment_can_reverse_proposal_identity_for_jepa(monkeypatch):
    module = pipeline(slots=2).eval()
    data = batch((2,))
    embeddings = torch.randn(1, 2, 32, requires_grad=True)
    logits = torch.full((1, 2), 4.)
    clean = dict(pred_boxes=data['gt_boxes_canon'], pred_embeddings=embeddings,
        existence_logits=logits, pred_logits=torch.stack((logits-20, logits), -1))
    predicted = {name:value.flip(1) for name,value in clean.items()}
    candidates = dict(clean, features=embeddings)
    monkeypatch.setattr(module.model, 'encode_context', lambda *args, **kwargs:None)
    monkeypatch.setattr(module.model, 'propose', lambda context:candidates)
    def outputs(*args, **kwargs):
        return clean, predicted, kwargs['source_coordinates'], kwargs['timesteps']
    monkeypatch.setattr(module.model, 'training_outputs', outputs)
    losses = module._losses_for_batch(data, training=False)
    assert losses['loss_jepa'] == 0
    assert losses['proposal/loss_bbox'] == losses['denoised/loss_bbox'] == 0


def test_sigreg_skip_and_background_mask_do_not_advance_projection_rng():
    module = pipeline().eval()
    for count in (0, 1):
        original = module.sigreg_rng_state.clone()
        losses = module._losses_for_batch(batch((count,)))
        assert losses['regularization/total'] == 0
        assert torch.equal(original, module.sigreg_rng_state)
    disabled = pipeline(sigreg_weight=0).eval()
    original = disabled.sigreg_rng_state.clone()
    assert disabled._losses_for_batch(batch())['regularization/total'] == 0
    assert torch.equal(original, disabled.sigreg_rng_state)


def test_checkpoint_hooks_restore_global_augmentation_and_torch_rng():
    import random
    import numpy as np
    module = pipeline()
    checkpoint = {}
    module.on_save_checkpoint(checkpoint)
    expected = (random.random(), np.random.rand(), torch.randn(3))
    random.random(), np.random.rand(), torch.randn(10)
    module.on_load_checkpoint(checkpoint)
    assert random.random() == expected[0] and np.random.rand() == expected[1]
    torch.testing.assert_close(torch.randn(3), expected[2], atol=0, rtol=0)

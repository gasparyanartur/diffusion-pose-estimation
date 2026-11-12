from copy import deepcopy
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from tests.test_poseflow_v35 import batch
from tests.test_poseflow_v40 import pipeline as baseline_pipeline


def pipeline(**overrides):
    from src.models.pipelines.pose_estimators.poseflow import PoseFlowV40SigRegPipelineConfig
    values = dict(backbone_pretrained=False, pretrained_checkpoint=None,
        hidden_dim=32, num_heads=4, feedforward_dim=64, num_queries=6,
        num_proposals=6, num_group=6, grid_rows=1, grid_columns=6,
        reference_candidates=12, reference_encoder_source_layers=(2,),
        quick_eval_enabled=False)
    return PoseFlowV40SigRegPipelineConfig(**(values | overrides)).build(max_epochs=2)


def test_recipe_shared_initial_tensors_and_intentional_normalization_change():
    with initialize_config_dir(config_dir=str(Path('conf').resolve()), version_base=None):
        config = compose(config_name='benchmark_poseflow_v40_sigreg')
    assert config.run_name == 'poseflow_v40_sigreg'
    assert config.pipeline.sigreg_weight == .05
    assert config.training.max_epochs == 80 and config.training.seed == 42
    torch.manual_seed(42)
    baseline = baseline_pipeline()
    state = torch.get_rng_state().clone()
    torch.manual_seed(42)
    module = pipeline()
    assert torch.equal(state, torch.get_rng_state())
    assert isinstance(module.model.clean_norm, torch.nn.Identity)
    assert not hasattr(module, 'teacher')
    for name, value in baseline.model.state_dict().items():
        torch.testing.assert_close(value, module.model.state_dict()[name], atol=0, rtol=0)


def test_sigreg_matches_complex_characteristic_function_reference_and_distribution_shape():
    from src.models.pipelines.pose_estimators.poseflow.latent_v40_sigreg import sigreg_penalty
    generator = torch.Generator().manual_seed(7)
    codes = torch.randn(512, 32, generator=generator)
    directions = torch.randn(32, 64, generator=generator)
    directions /= directions.norm(dim=0)
    frequencies = torch.linspace(0, 3, 17)
    empirical = torch.exp(1j*(codes @ directions)[..., None]*frequencies).mean(0)
    target = torch.exp(-frequencies.square()/2)
    reference = (2*torch.trapezoid((empirical-target).abs().square()*target,
                                  frequencies, dim=-1)*len(codes)).mean()
    actual = sigreg_penalty(codes, directions)
    torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-6)
    assert actual < sigreg_penalty(torch.zeros_like(codes), directions)
    assert actual < sigreg_penalty(codes+2, directions)
    assert actual < sigreg_penalty(codes*3, directions)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        assert sigreg_penalty(codes, directions).dtype == torch.float32


def test_regularizer_masks_background_skips_small_counts_and_is_disabled_at_zero_weight():
    module = pipeline().eval()
    codes = torch.randn(2, 6, 32, requires_grad=True)
    positive = torch.zeros(2, 6, dtype=torch.bool)
    positive[:, 0] = True
    original_rng = module.sigreg_rng_state.clone()
    result = module._clean_regularization(dict(pred_latents=codes), dict(positive=positive),
                                         training=False, batch_idx=0)
    result['total'].backward()
    assert codes.grad[positive].abs().sum() > 0 and codes.grad[~positive].abs().sum() == 0
    altered = codes.detach().clone()
    altered[~positive] = float('nan')
    other = module._clean_regularization(dict(pred_latents=altered), dict(positive=positive),
                                        training=False, batch_idx=0)
    torch.testing.assert_close(result['sigreg'], other['sigreg'], atol=0, rtol=0)
    assert torch.equal(original_rng, module.sigreg_rng_state)
    for count in (0, 1):
        positive.zero_()
        positive.view(-1)[:count] = True
        result = module._clean_regularization(dict(pred_latents=codes), dict(positive=positive),
                                             training=True, batch_idx=0)
        assert result['total'].requires_grad and result['total'] == 0
        assert module._diagnostics['sigreg/skipped'] == 1
        assert torch.equal(original_rng, module.sigreg_rng_state)
    disabled = pipeline(sigreg_weight=0)
    positive[:, 0] = True
    state = disabled.sigreg_rng_state.clone()
    result = disabled._clean_regularization(dict(pred_latents=codes), dict(positive=positive),
                                           training=True, batch_idx=0)
    assert result['total'] == 0 and torch.equal(state, disabled.sigreg_rng_state)


@pytest.mark.parametrize('loss_name', ['loss_latent', 'denoised/total', 'regularization/total'])
def test_independent_recovered_and_regularizer_gradients_reach_clean_encoder(loss_name):
    module = pipeline().eval()
    codes, noisy_sources = [], []
    handle = module.model.clean_norm.register_forward_hook(lambda _, args, output: codes.append(output))
    original_q_sample = module.model.q_sample
    def q_sample(source, times, noise=None):
        noisy_sources.append(source)
        return original_q_sample(source, times, noise)
    module.model.q_sample = q_sample
    losses = module._losses_for_batch(batch(), training=False)
    handle.remove()
    assert noisy_sources[0].requires_grad
    codes[0].retain_grad()
    losses[loss_name].backward()
    assert codes[0].grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.model.clean_layers.parameters())
    from src.models.pipelines.pose_estimators.poseflow.latent_fixed_source import positive_latent_mse
    target = torch.randn(1, 2, 32, requires_grad=True)
    predicted = torch.randn_like(target, requires_grad=True)
    positive_latent_mse(predicted, target, torch.ones(1, 2, dtype=torch.bool)).backward()
    assert target.grad is None


@pytest.mark.parametrize('detach_target', [True, False])
def test_objective_rng_validation_and_strict_next_update_restore(detach_target):
    module = pipeline(detach_latent_target=detach_target).eval()
    data = batch()
    state = module.sigreg_rng_state.clone()
    loss = module._losses_for_batch(data, training=False)
    expected = (loss['proposal/total']+.5*loss['clean/total']+.5*loss['denoised/total']
                +10*loss['loss_latent']+.05*loss['regularization/sigreg'])
    torch.testing.assert_close(loss['total'], expected)
    assert torch.equal(state, module.sigreg_rng_state)
    other = module._losses_for_batch(data, training=False)
    torch.testing.assert_close(loss['regularization/total'], other['regularization/total'], atol=0, rtol=0)
    optimizer = module.configure_optimizers()['optimizer']
    module._losses_for_batch(data)['total'].backward()
    optimizer.step()
    assert not torch.equal(state, module.sigreg_rng_state)
    restored = pipeline(detach_latent_target=detach_target).eval()
    restored.load_state_dict(deepcopy(module.state_dict()), strict=True)
    restored_optimizer = restored.configure_optimizers()['optimizer']
    restored_optimizer.load_state_dict(deepcopy(optimizer.state_dict()))
    for owner, opt in ((module, optimizer), (restored, restored_optimizer)):
        opt.zero_grad()
        owner._losses_for_batch(data)['total'].backward()
        opt.step()
    for name, value in module.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], atol=0, rtol=0)
    state = module.sigreg_rng_state.clone()
    with torch.no_grad():
        for steps in (1, 4):
            module.model.flow_sampling_steps = restored.model.flow_sampling_steps = steps
            left = module.model.sample_boxes(data['image'], generator=torch.Generator().manual_seed(8))
            right = restored.model.sample_boxes(data['image'], generator=torch.Generator().manual_seed(8))
            torch.testing.assert_close(left['pred_boxes'], right['pred_boxes'], atol=0, rtol=0)
    assert torch.equal(state, module.sigreg_rng_state)


def test_empty_single_invalid_and_full_fifty_training_and_cached_references():
    module = pipeline(detach_latent_target=False).eval()
    for counts in ((0,), (1,), (2,), (0, 2)):
        losses = module._losses_for_batch(batch(counts))
        assert all(torch.isfinite(value) for value in losses.values())
        assert module._diagnostics['sigreg/count'] == sum(counts)
    data = batch((2,))
    data['gt_boxes_canon'][0, 0, 0] = float('nan')
    assert torch.isfinite(module._losses_for_batch(data)['total'])
    assert module._diagnostics['sigreg/count'] == 1
    full = pipeline(num_queries=50, num_proposals=50, num_group=50,
                    grid_columns=50, reference_candidates=60, detach_latent_target=False).eval()
    data = batch((1,))
    data['gt_boxes_canon'] = torch.rand(1, 50, 4)*.6+.1
    data['person_mask'] = data['gt_box_labeled'] = torch.ones(1, 50, dtype=torch.bool)
    assert torch.isfinite(full._losses_for_batch(data)['total'])
    assert full._diagnostics['sigreg/count'] == 50
    calls, references = [], []
    propose, denoise = module.model.propose, module.model.denoise
    def counted(*args, **kwargs):
        calls.append(1)
        return propose(*args, **kwargs)
    def cached(*args, **kwargs):
        references.append(kwargs['references'])
        return denoise(*args, **kwargs)
    module.model.propose, module.model.denoise = counted, cached
    with torch.no_grad():
        module.model.flow_sampling_steps = 4
        module.model.sample_boxes(batch()['image'])
    assert len(calls) == 1 and len(references) == 4
    assert all(item is references[0] for item in references)


def test_new_recipe_enables_target_gradients_but_missing_historical_flag_preserves_detach():
    with initialize_config_dir(config_dir=str(Path('conf').resolve()), version_base=None):
        config = compose(config_name='benchmark_poseflow_v40_sigreg')
    assert config.pipeline.detach_latent_target is False
    assert pipeline().v26_config.detach_latent_target is True


@pytest.mark.parametrize('detach_target', [True, False])
def test_target_gradient_mask_and_exact_forward_loss_parity(detach_target):
    module = pipeline(detach_latent_target=detach_target)
    clean = torch.randn(2, 6, 32, requires_grad=True)
    recovered = torch.randn_like(clean, requires_grad=True)
    positive = torch.zeros(2, 6, dtype=torch.bool)
    positive[:, 0] = True
    target = clean.detach()
    loss = module._latent_reconstruction_loss(dict(pred_latents=clean),
        dict(pred_latents=recovered), target, dict(positive=positive))
    from src.models.pipelines.pose_estimators.poseflow.latent_fixed_source import positive_latent_mse
    torch.testing.assert_close(loss, positive_latent_mse(recovered, clean, positive), atol=0, rtol=0)
    loss.backward()
    assert recovered.grad[positive].abs().sum() > 0
    assert recovered.grad[~positive].abs().sum() == 0
    if detach_target:
        assert clean.grad is None
    else:
        assert clean.grad[positive].abs().sum() > 0 and clean.grad[~positive].abs().sum() == 0


def test_live_target_removes_expansion_gradient_for_contracting_predictor():
    for detach_target, expected_sign in [(True, -1), (False, 1)]:
        module = pipeline(detach_latent_target=detach_target)
        clean = torch.randn(1, 6, 32, requires_grad=True)
        recovered = .5*clean
        positive = torch.ones(1, 6, dtype=torch.bool)
        loss = module._latent_reconstruction_loss(dict(pred_latents=clean),
            dict(pred_latents=recovered), clean.detach(), dict(positive=positive))
        gradient = torch.autograd.grad(loss, clean)[0]
        assert expected_sign*(gradient*clean).sum() > 0


def test_batch_loss_uses_live_target_without_changing_forward_values(monkeypatch):
    torch.manual_seed(42)
    historical = pipeline(detach_latent_target=True).eval()
    torch.manual_seed(42)
    current = pipeline(detach_latent_target=False).eval()
    data = batch()
    left = historical._losses_for_batch(data, training=False)
    right = current._losses_for_batch(data, training=False)
    for name in left:
        torch.testing.assert_close(left[name], right[name], atol=0, rtol=0)
    original = current._latent_reconstruction_loss
    captured = {}
    def isolate_target(clean, recovered, latent_target, targets):
        captured['clean'] = clean['pred_latents']
        captured['clean'].retain_grad()
        independent_prediction = dict(recovered, pred_latents=recovered['pred_latents'].detach().requires_grad_())
        return original(clean, independent_prediction, latent_target, targets)
    monkeypatch.setattr(current, '_latent_reconstruction_loss', isolate_target)
    current._losses_for_batch(data, training=False)['loss_latent'].backward()
    assert captured['clean'].grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in current.model.clean_layers.parameters())

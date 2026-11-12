from copy import deepcopy
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from tests.test_poseflow_v35 import batch
from tests.test_poseflow_v37 import pipeline as v37_pipeline


def pipeline():
    from src.models.pipelines.pose_estimators.poseflow.training_v38 import PoseFlowV38PipelineConfig
    return PoseFlowV38PipelineConfig(backbone_pretrained=False, pretrained_checkpoint=None,
        hidden_dim=32, num_heads=4, feedforward_dim=64, num_queries=6, num_proposals=6,
        num_group=6, grid_rows=2, grid_columns=3, quick_eval_enabled=False).build(max_epochs=2)


def outputs(module, data, *, posterior_noise=None):
    targets, times, noise = module._draw_training_inputs(data, training=False)
    if posterior_noise is None:
        posterior_noise = module._draw_posterior_noise(noise, training=False)
    result = module.model.training_outputs(data['image'], targets['input_pixels'],
        timesteps=times, noise=noise, posterior_noise=posterior_noise)
    return targets, result


def test_recipe():
    with initialize_config_dir(config_dir=str(Path('conf').resolve()), version_base=None):
        config = compose(config_name='benchmark_poseflow_v38')
    assert config.pipeline.latent_loss_weight == 100
    assert config.pipeline.kl_loss_weight == .1 and config.pipeline.alignment_loss_weight == 0
    assert config.pipeline.posterior_seed == 1042
    assert config.pipeline.posterior_logvar_init == -4
    assert (config.pipeline.posterior_logvar_min, config.pipeline.posterior_logvar_max) == (-10, 2)
    assert config.pipeline.denoised_task_loss_weight == .5
    assert config.pipeline.num_proposals == 50 and config.training.max_epochs == 80


def test_analytical_kl_and_gradients():
    from src.models.pipelines.pose_estimators.poseflow.training_v38 import prior_kl
    mean = torch.tensor([[[0., 2.], [1., -1.]]], requires_grad=True)
    logvar = torch.tensor([[[0., -4.], [2., -10.]]], requires_grad=True)
    expected = .5*(mean.square()+logvar.exp()-1-logvar)
    actual = prior_kl(mean, logvar)
    torch.testing.assert_close(actual, expected.mean())
    actual.backward()
    torch.testing.assert_close(mean.grad, mean.detach()/mean.numel())
    torch.testing.assert_close(logvar.grad, .5*(logvar.detach().exp()-1)/mean.numel())
    assert prior_kl(torch.zeros(2, 3, 4), torch.zeros(2, 3, 4)) == 0


def test_shared_initialization_global_rng_and_original_draws():
    torch.manual_seed(42); baseline = v37_pipeline().eval()
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(42); module = pipeline().eval()
    torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
    assert not hasattr(module, 'alignment') and not hasattr(module, 'matcher')
    for key, value in baseline.model.state_dict().items():
        torch.testing.assert_close(value, module.model.state_dict()[key], rtol=0, atol=0)
    torch.testing.assert_close(module.model.posterior_mean.weight, torch.eye(32), rtol=0, atol=0)
    assert module.model.posterior_logvar.weight.count_nonzero() == 0
    data = batch()
    for _ in range(2):
        left = baseline._draw_training_inputs(data)
        module._draw_posterior_noise(left[2])
        right = module._draw_training_inputs(data)
        for key in left[0]:
            torch.testing.assert_close(left[0][key], right[0][key], rtol=0, atol=0)
        for a, b in zip(left[1:], right[1:]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    targets, times, noise = baseline._draw_training_inputs(data, training=False)
    with torch.no_grad():
        expected = baseline.model.training_outputs(data['image'], targets['input_pixels'], timesteps=times, noise=noise)
        actual = module.model.training_outputs(data['image'], targets['input_pixels'], timesteps=times,
            noise=noise, posterior_noise=torch.zeros_like(noise))
    for key in expected[0]:
        torch.testing.assert_close(expected[0][key], actual[0][key], rtol=0, atol=0)
    for key in expected[1]:
        torch.testing.assert_close(expected[1][key], actual[1][key], rtol=0, atol=0)


def test_posterior_noise_isolation_clamping_and_inference_parity():
    from src.models.pipelines.pose_estimators.poseflow.latent_v37 import assign_latent_targets
    torch.manual_seed(42); baseline = v37_pipeline().eval()
    torch.manual_seed(42); module = pipeline().eval()
    data = batch()
    with torch.no_grad():
        targets, first = outputs(module, data, posterior_noise=torch.zeros(2, 6, 32))
        _, second = outputs(module, data, posterior_noise=torch.ones(2, 6, 32))
        assert not torch.equal(first[0]['sampled_pred_boxes'], second[0]['sampled_pred_boxes'])
        for key in ('pred_latents', 'pred_boxes', 'existence_logits', 'posterior_logvar'):
            torch.testing.assert_close(first[0][key], second[0][key], rtol=0, atol=0)
        torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)
        for key in first[1]:
            torch.testing.assert_close(first[1][key], second[1][key], rtol=0, atol=0)
        left = assign_latent_targets(first[0], first[1], targets)
        right = assign_latent_targets(second[0], second[1], targets)
        torch.testing.assert_close(left[1], right[1], rtol=0, atol=0)
        torch.testing.assert_close(left[2]['selected_slots'], right[2]['selected_slots'], rtol=0, atol=0)
        for extreme, boundary in ((-1e6, -10), (1e6, 2)):
            module.model.posterior_logvar.bias.fill_(extreme)
            _, result = outputs(module, data)
            assert (result[0]['posterior_logvar'] == boundary).all()
            assert (result[0]['posterior_logvar'].exp() > 0).all()
            assert all(torch.isfinite(value).all() for value in result[0].values())
        def forbidden(*args):
            raise AssertionError('Gaussian inference executed a posterior head')
        hooks = [head.register_forward_hook(forbidden) for head in
            (module.model.posterior_mean, module.model.posterior_logvar)]
        try:
            for steps in (1, 4):
                baseline.model.flow_sampling_steps = module.model.flow_sampling_steps = steps
                left = baseline.model.sample_boxes(data['image'], generator=torch.Generator().manual_seed(8))
                right = module.model.sample_boxes(data['image'], generator=torch.Generator().manual_seed(8))
                for key in left:
                    if isinstance(left[key], torch.Tensor):
                        torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
        finally:
            for hook in hooks:
                hook.remove()


def test_gradient_routes_and_exact_objective():
    module = pipeline().eval()
    losses = module._losses_for_batch(batch())
    assert 'loss_alignment' not in losses
    torch.testing.assert_close(losses['total'], .5*losses['clean/total']+.5*losses['denoised/total']
        +100*losses['loss_latent']+.1*losses['loss_kl'])
    (losses['loss_latent']+losses['denoised/total']).backward(retain_graph=True)
    for name in ('posterior_mean', 'posterior_logvar', 'clean_layers', 'input_mlp'):
        assert all(p.grad is None for p in getattr(module.model, name).parameters())
    assert module.model.denoiser_projection.weight.grad.abs().sum() > 0
    module.zero_grad(); losses['clean/total'].backward(retain_graph=True)
    for name in ('posterior_mean', 'posterior_logvar', 'clean_layers'):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in getattr(module.model, name).parameters())
    module.zero_grad(); losses['loss_kl'].backward()
    assert module.model.posterior_mean.weight.grad.abs().sum() > 0
    assert module.model.posterior_logvar.bias.grad.abs().sum() > 0
    optimizer = module.configure_optimizers()['optimizer']
    members = {id(p) for group in optimizer.param_groups for p in group['params']}
    assert all(id(p) in members for p in module.model.posterior_mean.parameters())
    assert all(id(p) in members for p in module.model.posterior_logvar.parameters())


@pytest.mark.parametrize('counts', [(2, 1), (0, 0)])
def test_fake_existence_kl_and_real_only_geometry(counts):
    from src.models.pipelines.pose_estimators.poseflow.latent_v35 import box_task_loss
    from src.models.pipelines.pose_estimators.poseflow.training_v38 import prior_kl
    module = pipeline().eval()
    targets, (clean, recovered, _, _) = outputs(module, batch(counts))
    positive = targets['positive']
    assert (~positive).any()
    boxes = clean['sampled_pred_boxes']; boxes.retain_grad()
    logits = clean['sampled_existence_logits']; logits.retain_grad()
    losses = box_task_loss(dict(pred_boxes=boxes, existence_logits=logits), targets)
    losses['total'].backward(retain_graph=True)
    assert boxes.grad[~positive].count_nonzero() == 0
    assert logits.grad[~positive].abs().sum() > 0
    clean['pred_latents'].retain_grad(); clean['posterior_logvar'].retain_grad()
    prior_kl(clean['pred_latents'], clean['posterior_logvar']).backward()
    assert clean['pred_latents'].grad[~positive].abs().sum() > 0
    assert clean['posterior_logvar'].grad[~positive].abs().sum() > 0
    assert all(torch.isfinite(value) for value in module._losses_for_batch(batch(counts)).values())
    if counts[0]:
        assert positive.sum() > targets['unique_people'].sum()


def test_strict_restore_optimizer_and_all_rng_continuation(tmp_path):
    from omegaconf import OmegaConf
    from scripts.evaluation.evaluate_poseflow import load_poseflow_model
    module = pipeline().eval()
    optimizer = module.configure_optimizers()['optimizer']
    module._losses_for_batch(batch())['total'].backward(); optimizer.step()
    restored = pipeline().eval(); restored.load_state_dict(deepcopy(module.state_dict()), strict=True)
    other = restored.configure_optimizers()['optimizer']; other.load_state_dict(deepcopy(optimizer.state_dict()))
    for head in ('posterior_mean', 'posterior_logvar'):
        for a, b in zip(getattr(module.model, head).parameters(), getattr(restored.model, head).parameters()):
            for key in ('step', 'exp_avg', 'exp_avg_sq'):
                torch.testing.assert_close(optimizer.state[a][key], other.state[b][key], rtol=0, atol=0)
    data = batch()
    left = module._draw_training_inputs(data); right = restored._draw_training_inputs(data)
    for key in left[0]:
        torch.testing.assert_close(left[0][key], right[0][key], rtol=0, atol=0)
    for a, b in zip(left[1:], right[1:]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(module._draw_posterior_noise(left[2]), restored._draw_posterior_noise(right[2]), rtol=0, atol=0)
    state = module.posterior_rng_state.clone()
    a = module._draw_posterior_noise(left[2], training=False, batch_idx=3)
    b = module._draw_posterior_noise(left[2], training=False, batch_idx=3)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(state, module.posterior_rng_state, rtol=0, atol=0)
    checkpoint = tmp_path/'last.ckpt'; config = tmp_path/'config.yaml'
    torch.save(dict(state_dict=module.state_dict()), checkpoint)
    OmegaConf.save(dict(pipeline=dict(_target_='src.models.pipelines.pose_estimators.poseflow.PoseFlowV38PipelineConfig',
        **module.v26_config.model_dump(mode='json'))), config)
    loaded, _ = load_poseflow_model(checkpoint, config)
    for name, value in loaded.state_dict().items():
        torch.testing.assert_close(value, module.model.state_dict()[name], rtol=0, atol=0)


def test_bbox_noise_metrics_use_positive_assignment_counts():
    from src.models.pipelines.pose_estimators.poseflow.training_v38 import bbox_noise_metrics
    boxes = torch.zeros(4, 3, 4)
    targets = dict(boxes=torch.ones_like(boxes), positive=torch.tensor([
        [True, False, False], [True, True, True], [False, False, False], [True, True, False]]))
    boxes[1] = .5
    result = bbox_noise_metrics(boxes, targets, torch.tensor([0, 100, 500, 999]), 1000)
    assert set(result) == {'low', 'high'}
    assert result['low'][1] == 4 and result['high'][1] == 2
    torch.testing.assert_close(result['low'][0], torch.tensor((4+3*2)/4))
    module = pipeline()
    module._diagnostics = {}; module._bbox_noise_metrics = result
    calls = []
    module._log_losses = lambda *args: None
    module.log = lambda name, value, **kwargs: calls.append((name, kwargs['batch_size']))
    module._log_v26('val', {}, 4)
    assert calls == [('debug/v38/val/denoised_bbox_l1/low', 4), ('debug/v38/val/denoised_bbox_l1/high', 2)]

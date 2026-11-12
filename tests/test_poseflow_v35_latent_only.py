from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from tests.test_poseflow_v35 import batch, pipeline


def test_spatial_latent_only_monitoring_and_gradient_routes():
    module = pipeline(True)
    module.v26_config.denoised_task_loss_weight = 0.
    observed = {}
    original = module.model.training_outputs

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        observed['clean'] = result[0]['pred_latents']
        observed['clean'].retain_grad()
        assert not result[2].requires_grad
        return result

    module.model.training_outputs = capture
    losses = module._losses_for_batch(batch())
    assert not losses['denoised/total'].requires_grad
    torch.testing.assert_close(losses['total'], .5*losses['clean/total']+10*losses['loss_latent'], atol=0, rtol=0)
    denoiser = list(module.model.denoiser_layers.parameters())+list(module.model.denoiser_projection.parameters())
    actual = torch.autograd.grad(losses['total'], denoiser, retain_graph=True)
    expected = torch.autograd.grad(10*losses['loss_latent'], denoiser, retain_graph=True)
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    losses['loss_latent'].backward(retain_graph=True)
    assert observed['clean'].grad is None
    for component in (module.model.clean_layers, module.model.box_head, module.model.existence_head):
        assert all(p.grad is None for p in component.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.model.denoiser_layers.parameters())
    module.zero_grad(set_to_none=True)
    losses['total'].backward()
    for component in (module.model.clean_layers, module.model.box_head, module.model.existence_head, module.model.backbone):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in component.parameters())


def test_latent_only_configuration_changes_only_coefficient_and_namespace():
    with initialize_config_dir(config_dir=str(Path('conf').resolve()), version_base=None):
        baseline = OmegaConf.to_container(compose(config_name='benchmark_poseflow_v35_spatial'), resolve=True)
        actual = OmegaConf.to_container(compose(config_name='benchmark_poseflow_v35_spatial_latent_only'), resolve=True)
    baseline['pipeline']['denoised_task_loss_weight'] = 0.
    baseline.update(run_name='poseflow_v35_spatial_latent_only', out_dir='results/poseflow_v35_spatial_latent_only')
    assert actual == baseline


def test_default_supervised_path_has_identical_forward_monitoring():
    supervised = pipeline(True)
    latent_only = pipeline(True)
    latent_only.load_state_dict(supervised.state_dict(), strict=True)
    latent_only.v26_config.denoised_task_loss_weight = 0.
    data = batch()
    baseline = supervised._losses_for_batch(data)
    actual = latent_only._losses_for_batch(data)
    for key in baseline:
        if key != 'total':
            torch.testing.assert_close(baseline[key], actual[key], atol=0, rtol=0)
    assert baseline['denoised/total'].requires_grad

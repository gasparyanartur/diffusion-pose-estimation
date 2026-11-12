from copy import deepcopy
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from tests.test_poseflow_v35 import batch


def pipeline(detach_source):
    from src.models.pipelines.pose_estimators.poseflow import PoseFlowV39PipelineConfig
    return PoseFlowV39PipelineConfig(backbone_pretrained=False, pretrained_checkpoint=None,
        hidden_dim=32, num_heads=4, feedforward_dim=64, num_queries=6, num_proposals=6,
        num_group=6, grid_rows=2, grid_columns=3, quick_eval_enabled=False,
        detach_corruption_source=detach_source).build(max_epochs=2)


def test_recipe_changes_only_corruption_boundary_and_namespace():
    with initialize_config_dir(config_dir=str(Path('conf').resolve()), version_base=None):
        baseline = OmegaConf.to_container(compose(config_name='benchmark_poseflow_v39'), resolve=True)
        current = OmegaConf.to_container(compose(config_name='benchmark_poseflow_v39_denoiser_grad'), resolve=True)
    assert current.pop('run_name') == 'poseflow_v39_denoiser_grad'
    assert current.pop('out_dir') == 'results/poseflow_v39_denoiser_grad'
    baseline.pop('run_name')
    baseline.pop('out_dir')
    assert current['pipeline'].pop('detach_corruption_source') is False
    assert current == baseline
    assert pipeline(True).v26_config.detach_corruption_source is True


def test_forward_rng_initialization_and_inference_are_exactly_unchanged():
    torch.manual_seed(42)
    baseline = pipeline(True).eval()
    rng = torch.get_rng_state().clone()
    torch.manual_seed(42)
    live = pipeline(False).eval()
    assert torch.equal(rng, torch.get_rng_state())
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(value, live.state_dict()[name], atol=0, rtol=0)
    data = batch()
    left = baseline._losses_for_batch(data, training=False)
    right = live._losses_for_batch(data, training=False)
    for name in left:
        torch.testing.assert_close(left[name], right[name], atol=0, rtol=0)
    targets, times, noise = baseline._draw_training_inputs(data, training=False)
    detached = baseline.model.training_outputs(data['image'], targets['input_pixels'], timesteps=times, noise=noise)
    attached = live.model.training_outputs(data['image'], targets['input_pixels'], timesteps=times, noise=noise,
                                         detach_corruption_source=False)
    assert not detached[2].requires_grad and attached[2].requires_grad
    for index in (0, 1):
        for name in detached[index]:
            torch.testing.assert_close(detached[index][name], attached[index][name], atol=0, rtol=0)
    torch.testing.assert_close(detached[2], attached[2], atol=0, rtol=0)
    with torch.no_grad():
        for steps in (1, 4):
            baseline.model.flow_sampling_steps = live.model.flow_sampling_steps = steps
            left = baseline.model.sample_boxes(data['image'], generator=torch.Generator().manual_seed(5))
            right = live.model.sample_boxes(data['image'], generator=torch.Generator().manual_seed(5))
            for name in ('pred_boxes', 'existence_logits', 'pred_latents'):
                torch.testing.assert_close(left[name], right[name], atol=0, rtol=0)


@pytest.mark.parametrize('loss_name', ['loss_latent', 'denoised/total'])
def test_each_recovered_loss_trains_clean_encoder_only_through_live_source(loss_name):
    data = batch()
    for detach_source in (True, False):
        module = pipeline(detach_source).eval()
        clean_codes = []
        handle = module.model.clean_norm.register_forward_hook(lambda _, args, output: clean_codes.append(output))
        losses = module._losses_for_batch(data)
        clean_codes[0].retain_grad()
        losses[loss_name].backward()
        if detach_source:
            assert clean_codes[0].grad is None
            assert all(p.grad is None for p in module.model.clean_layers.parameters())
            assert all(p.grad is None for p in module.model.input_mlp.parameters())
        else:
            assert clean_codes[0].grad.abs().sum() > 0
            for component in (module.model.clean_layers, module.model.input_mlp):
                assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in component.parameters())
        handle.remove()


def test_corruption_jacobian_and_latent_target_stop_gradient():
    from src.models.pipelines.pose_estimators.poseflow.latent_fixed_source import positive_latent_mse
    module = pipeline(False)
    source = torch.randn(2, 6, 32, requires_grad=True)
    times = torch.tensor([0, 999])
    noisy = module.model.q_sample(source, times, torch.zeros_like(source))
    derivative = torch.autograd.grad(noisy.sum(), source)[0]
    expected = module.model.sqrt_alphas_cumprod[times].float()[:, None, None].expand_as(source)
    torch.testing.assert_close(derivative, expected, atol=0, rtol=0)
    prediction = torch.randn_like(source, requires_grad=True)
    target = source.detach().clone().requires_grad_()
    positive = torch.zeros(2, 6, dtype=torch.bool)
    positive[:, 0] = True
    positive_latent_mse(prediction, target, positive).backward()
    assert target.grad is None
    assert prediction.grad[~positive].abs().sum() == 0


def test_live_source_empty_cases_and_strict_restore_with_identical_next_draws():
    module = pipeline(False).eval()
    for counts in ((0,), (2, 1)):
        losses = module._losses_for_batch(batch(counts))
        losses['total'].backward()
        assert all(torch.isfinite(value) for value in losses.values())
        if counts == (0,):
            assert losses['loss_latent'] == 0
        module.zero_grad(set_to_none=True)
    restored = pipeline(False).eval()
    restored.load_state_dict(deepcopy(module.state_dict()), strict=True)
    data = batch()
    left, right = module._draw_training_inputs(data), restored._draw_training_inputs(data)
    for name in left[0]:
        torch.testing.assert_close(left[0][name], right[0][name], atol=0, rtol=0)
    torch.testing.assert_close(left[1], right[1], atol=0, rtol=0)
    torch.testing.assert_close(left[2], right[2], atol=0, rtol=0)

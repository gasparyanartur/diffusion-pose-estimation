from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from tests.test_poseflow_v35 import batch


def pipeline(condition=False, **kwargs):
    from src.models.pipelines.pose_estimators.poseflow.training_v36 import PoseFlowV36PipelineConfig
    return PoseFlowV36PipelineConfig(backbone_pretrained=False, pretrained_checkpoint=None,
        hidden_dim=32, num_heads=4, feedforward_dim=64, num_queries=6, num_proposals=6,
        num_group=6, grid_rows=2, grid_columns=3, spatial_queries=True, spatial_cost_weight=1.,
        clean_existence_input=condition, quick_eval_enabled=False, **kwargs).build(max_epochs=2)


@pytest.mark.parametrize('condition', [False, True])
def test_clean_independent_encoder_and_gradient_routes(condition):
    module = pipeline(condition)
    model = module.model
    assert not hasattr(model, 'clean_layers')
    data = batch()
    targets, times, noise = module._draw_training_inputs(data)
    args = dict(existence=targets['positive']) if condition else {}
    clean = model.encode_clean(targets['input_pixels'], **args)
    expected = model.clean_norm(model.input_mlp(torch.cat((targets['input_pixels']/100.,
        targets['positive'][..., None].float()), -1) if condition else targets['input_pixels']/100.))
    torch.testing.assert_close(clean['pred_latents'], expected, atol=0, rtol=0)
    changed = targets['input_pixels'].clone(); changed[:, 1:] += 20
    torch.testing.assert_close(model.encode_clean(changed, **args)['pred_latents'][:, 0],
                               clean['pred_latents'][:, 0], atol=0, rtol=0)
    clean['pred_boxes'].sum().backward()
    assert all(p.grad is None for p in model.backbone.parameters())
    module.zero_grad(set_to_none=True)
    losses = module._losses_for_batch(data)
    assert not losses['denoised/total'].requires_grad
    torch.testing.assert_close(losses['total'], .5*losses['clean/total']+10*losses['loss_latent'], atol=0, rtol=0)
    losses['loss_latent'].backward(retain_graph=True)
    for component in (model.input_mlp, model.box_head, model.existence_head):
        assert all(p.grad is None for p in component.parameters())
    for component in (model.backbone, model.denoiser_layers, model.query_projection):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in component.parameters())
    module.zero_grad(set_to_none=True)
    losses['total'].backward()
    for component in (model.input_mlp, model.box_head, model.existence_head):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in component.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_c_boundary_and_shared_initialization():
    torch.manual_seed(42); box = pipeline().model
    torch.manual_seed(42); labeled = pipeline(True).model
    for name, value in box.state_dict().items():
        other = labeled.state_dict()[name]
        torch.testing.assert_close(value, other[:, :4] if name == 'input_mlp.0.weight' else other, atol=0, rtol=0)
    pixels = torch.ones(1, 6, 4)*40
    with pytest.raises(ValueError, match='existence'):
        labeled.encode_clean(pixels)
    with pytest.raises(ValueError, match='existence'):
        labeled.encode_clean(pixels, existence=torch.ones(1, 6)*2)
    with pytest.raises(ValueError, match='existence'):
        labeled.encode_clean(pixels, existence=torch.ones(1, 5))
    positive = labeled.encode_clean(pixels, existence=torch.ones(1, 6))
    negative = labeled.encode_clean(pixels, existence=torch.zeros(1, 6))
    assert not torch.equal(positive['pred_latents'], negative['pred_latents'])
    assert box.input_mlp[0].in_features == 4 and labeled.input_mlp[0].in_features == 5


@pytest.mark.parametrize('condition', [False, True])
def test_restore_next_draws_and_gaussian_inference(condition):
    module = pipeline(condition).eval()
    data = batch()
    module._draw_training_inputs(data)
    restored = pipeline(condition).eval()
    restored.load_state_dict(module.state_dict(), strict=True)
    for left, right in zip(module._draw_training_inputs(data), restored._draw_training_inputs(data)):
        if isinstance(left, dict):
            for name in left:
                torch.testing.assert_close(left[name], right[name], atol=0, rtol=0)
        else:
            torch.testing.assert_close(left, right, atol=0, rtol=0)
    with torch.no_grad():
        expected = module.model.sample_boxes(data['image'], generator=torch.Generator().manual_seed(5))
        actual = restored.model.sample_boxes(data['image'], generator=torch.Generator().manual_seed(5))
    for name in ('pred_boxes', 'existence_logits', 'pred_latents'):
        torch.testing.assert_close(expected[name], actual[name], atol=0, rtol=0)


def test_configs():
    with initialize_config_dir(config_dir=str(Path('conf').resolve()), version_base=None):
        configs = [OmegaConf.to_container(compose(config_name=f'benchmark_poseflow_v36_{name}'), resolve=True)
                   for name in ('box', 'box_c')]
    for config, condition in zip(configs, (False, True)):
        assert config['pipeline']['clean_existence_input'] == condition
        assert config['pipeline']['denoised_task_loss_weight'] == 0
        assert config['pipeline']['latent_loss_weight'] == 10
        assert config['pipeline']['spatial_queries']

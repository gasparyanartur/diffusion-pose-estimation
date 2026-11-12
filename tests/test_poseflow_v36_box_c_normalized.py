from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from tests.test_poseflow_v35 import batch
from tests.test_poseflow_v36 import pipeline


def test_normalized_box_c_configuration_is_single_input_change():
    with initialize_config_dir(config_dir=str(Path('conf').resolve()), version_base=None):
        baseline = OmegaConf.to_container(compose(config_name='benchmark_poseflow_v36_box_c'), resolve=True)
        actual = OmegaConf.to_container(compose(config_name='benchmark_poseflow_v36_box_c_normalized'), resolve=True)
    baseline['pipeline']['clean_box_input_format'] = 'normalized'
    baseline.update(run_name='poseflow_v36_box_c_normalized', out_dir='results/poseflow_v36_box_c_normalized')
    assert actual == baseline


def test_normalized_box_c_input_detach_initialization_and_restore():
    torch.manual_seed(42); baseline = pipeline(True)
    torch.manual_seed(42); module = pipeline(True, clean_box_input_format='normalized')
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(value, module.state_dict()[name], atol=0, rtol=0)
    data = batch()
    targets, times, noise = module._draw_training_inputs(data)
    observed = []
    handle = module.model.input_mlp.register_forward_pre_hook(lambda model, args: observed.append(args[0].detach()))
    clean, recovered, noisy, _ = module._training_outputs(data, targets, times, noise)
    handle.remove()
    expected = torch.cat((targets['input_boxes'], targets['positive'][..., None].float()), -1)
    torch.testing.assert_close(observed[0], expected, atol=0, rtol=0)
    assert not noisy.requires_grad
    losses = module._losses_for_batch(data)
    losses['loss_latent'].backward()
    for component in (module.model.input_mlp, module.model.box_head, module.model.existence_head):
        assert all(p.grad is None for p in component.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.model.denoiser_layers.parameters())
    restored = module.v26_config.build(max_epochs=2)
    restored.load_state_dict(module.state_dict(), strict=True)
    expected = module.model.encode_clean(targets['input_boxes'], existence=targets['positive'])
    actual = restored.model.encode_clean(targets['input_boxes'], existence=targets['positive'])
    for key in expected:
        torch.testing.assert_close(expected[key], actual[key], atol=0, rtol=0)

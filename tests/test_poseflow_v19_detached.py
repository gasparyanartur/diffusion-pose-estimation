import copy

import torch
import pytest

from tests.test_poseflow_v19 import configuration, source_model
from src.models.pipelines.pose_estimators.poseflow.model_v19 import PoseFlowV19Model


def test_detached_geometry_preserves_values_and_cuts_only_geometry_handoff():
    torch.manual_seed(42)
    baseline = PoseFlowV19Model(source_model()).eval()
    for layer in baseline.layers:
        torch.nn.init.normal_(layer.box_head.layers[-1].weight, std=.01)
        torch.nn.init.normal_(layer.joint_head.layers[-1].weight, std=.01)
    detached = copy.deepcopy(baseline)
    detached.detach_decoder_geometry = True
    image = torch.randn(1, 3, 64, 64)
    state = torch.randn(1, 3, 8, requires_grad=True)
    times = torch.ones(1)
    expected = baseline.denoise(state, times, baseline.encode_image(image))
    baseline_gradient = torch.autograd.grad(expected['pred_coordinates'].sum(),
        expected['aux_outputs'][0]['pred_coordinates'], retain_graph=True)[0]
    assert baseline_gradient.abs().sum() > 0
    references, features = [], []
    def capture_inputs(module, args):
        references.append(args[2])
        features.append(args[0])

    detached.layers[1].register_forward_pre_hook(capture_inputs)
    actual = detached.denoise(state, times, detached.encode_image(image))
    torch.testing.assert_close(actual['pred_coordinates'], expected['pred_coordinates'], atol=0, rtol=0)
    torch.testing.assert_close(actual['hypothesis_memory'], expected['hypothesis_memory'], atol=0, rtol=0)
    assert not references[0].requires_grad
    assert features[0].requires_grad
    earlier = actual['aux_outputs'][0]['pred_coordinates']
    gradient = torch.autograd.grad(actual['pred_coordinates'].sum(), earlier,
                                   retain_graph=True, allow_unused=True)[0]
    assert gradient is None
    assert torch.autograd.grad(earlier.sum(), detached.layers[0].box_head.layers[-1].bias,
                               retain_graph=True)[0].abs().sum() > 0
    feature_gradient = torch.autograd.grad(actual['pred_coordinates'].square().sum(),
                                         features[0], retain_graph=True)[0]
    assert torch.isfinite(feature_gradient).all() and feature_gradient.abs().sum() > 0
    for steps in [1, 3]:
        expected_sample = baseline.sample_keypoints(image, initial_latents=state.detach(), sampling_steps=steps)
        actual_sample = detached.sample_keypoints(image, initial_latents=state.detach(), sampling_steps=steps)
        torch.testing.assert_close(actual_sample['pred_coordinates'], expected_sample['pred_coordinates'],
                                   atol=0, rtol=0)


@pytest.mark.parametrize('config_name,flag', [
    ('benchmark_poseflow_v19_detached', 'detach_decoder_geometry'),
    ('benchmark_poseflow_v19_sampling_detach', 'detach_sampling_references'),
])
def test_detached_config_and_cross_call_gradients(tmp_path, config_name, flag):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from pathlib import Path
    from scripts.evaluation.evaluate_poseflow import load_poseflow_model
    from src.models.pipelines.pose_estimators.poseflow import PoseFlowPipelineConfig
    from src.models.pipelines.pose_estimators.poseflow.latent_v12 import ddim_update

    with initialize_config_dir(config_dir=str(Path('conf').resolve()), version_base=None):
        resolved = compose(config_name=config_name)
    config = PoseFlowPipelineConfig.from_hydra_config(resolved.pipeline)
    assert getattr(config, flag)
    pipeline = configuration(**{flag: True}).build(max_epochs=2).eval()
    assert getattr(pipeline.model, flag)
    for layer in pipeline.model.layers:
        torch.nn.init.normal_(layer.box_head.layers[-1].weight, std=.01)
        torch.nn.init.normal_(layer.joint_head.layers[-1].weight, std=.01)
    image = torch.randn(1, 3, 64, 64)
    encoded = pipeline.model.encode_image(image)
    state = torch.randn(1, 3, 8)
    first = pipeline.model.denoise(state, torch.ones(1), encoded)
    next_state = ddim_update(state, first['pred_coordinates'], torch.ones(1), torch.full((1,), .5))
    second = pipeline.model.denoise(next_state, torch.full((1,), .5), encoded,
                                    hypothesis_memory=first['hypothesis_memory'])
    gradients = torch.autograd.grad(second['pred_coordinates'].square().sum(),
        (first['pred_coordinates'], first['hypothesis_memory']))
    assert all(torch.isfinite(gradient).all() and gradient.abs().sum() > 0 for gradient in gradients)
    torch.save({'state_dict': pipeline.state_dict()}, tmp_path/'model.ckpt')
    OmegaConf.save(OmegaConf.create({'pipeline': pipeline.v12_config.model_dump() | {
        '_target_': 'src.models.pipelines.pose_estimators.poseflow.PoseFlowV19PipelineConfig'}}),
        tmp_path/'config.yaml')
    restored, _ = load_poseflow_model(tmp_path/'model.ckpt', tmp_path/'config.yaml')
    assert getattr(restored, flag)
    restored.eval()
    actual = restored.denoise(state, torch.ones(1), restored.encode_image(image))
    torch.testing.assert_close(actual['pred_coordinates'], first['pred_coordinates'])


def test_sampling_stop_preserves_coordinate_residual_gradient():
    torch.manual_seed(42)
    baseline = PoseFlowV19Model(source_model()).eval()
    for layer in baseline.layers:
        torch.nn.init.normal_(layer.box_head.layers[-1].weight, std=.01)
        torch.nn.init.normal_(layer.joint_head.layers[-1].weight, std=.01)
    model = copy.deepcopy(baseline)
    model.detach_sampling_references = True
    references = []
    for layer in model.layers:
        layer.register_forward_pre_hook(lambda module, args: references.append(args[2]))
    image = torch.randn(1, 3, 64, 64)
    state = torch.randn(1, 3, 8, requires_grad=True)
    expected = baseline.denoise(state, torch.ones(1), baseline.encode_image(image))
    actual = model.denoise(state, torch.ones(1), model.encode_image(image))
    for a, b in zip(actual['aux_outputs']+[actual], expected['aux_outputs']+[expected]):
        torch.testing.assert_close(a['pred_coordinates'], b['pred_coordinates'], atol=0, rtol=0)
    assert references[0].requires_grad and not references[1].requires_grad
    gradient = torch.autograd.grad(actual['pred_coordinates'].sum(),
        actual['aux_outputs'][0]['pred_coordinates'])[0]
    torch.testing.assert_close(gradient, torch.ones_like(gradient), atol=0, rtol=0)


def test_gradient_boundary_flags_are_exclusive():
    with pytest.raises(ValueError, match='mutually exclusive'):
        configuration(detach_decoder_geometry=True, detach_sampling_references=True)

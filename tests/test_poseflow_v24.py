from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from src.models.pipelines.pose_estimators.poseflow.training_v24 import PoseFlowV24PipelineConfig
from tests.test_poseflow import _batch


@pytest.fixture(autouse=True)
def preserve_rng():
    with torch.random.fork_rng(devices=[]):
        yield


def configuration(**overrides):
    return PoseFlowV24PipelineConfig(**(dict(
        backbone_pretrained=False, pretrained_checkpoint=None,
        num_queries=5, num_proposals=3, num_group=3, num_body_points=2,
        hidden_dim=32, feedforward_dim=64, num_heads=4, num_feature_levels=2,
        num_encoder_layers=1, num_decoder_layers=3, num_box_decoder_layers=1,
        dn_number=2, quick_eval_enabled=False,
    ) | overrides))


def matched_models():
    torch.manual_seed(42)
    control = configuration(h_enabled=False)._build_poseflow_model()
    control_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(42)
    treatment = configuration(h_enabled=True)._build_poseflow_model()
    assert torch.equal(control_rng, torch.random.get_rng_state())
    for name, value in control.state_dict().items():
        assert torch.equal(value, treatment.state_dict()[name]), name
    return control, treatment


@pytest.mark.parametrize('training', [False, True])
def test_closed_gate_exact_prediction_and_rng_parity_with_dn(training):
    control, treatment = matched_models()
    control.train(training); treatment.train(training)
    image, source, times = torch.randn(1, 3, 64, 64), torch.randn(1, 3, 4), torch.tensor([.3])
    pipeline = configuration().build(max_epochs=2)
    from src.models.pipelines.pose_estimators.poseflow.training import build_poseflow_targets
    _, _, _, targets = build_poseflow_targets(_batch(), control, num_proposals=3,
        num_body_points=2, padding_policy='source_prior', noise_seed=42)
    torch.manual_seed(91)
    expected = control(image, source, times, targets if training else None)
    expected_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(91)
    actual = treatment(image, source, times, targets if training else None)
    assert torch.equal(expected_rng, torch.random.get_rng_state())
    for key in ('pred_logits', 'pred_boxes', 'pred_keypoints'):
        torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
    if training:
        first = pipeline.criterion(expected, targets)['total']
        second = pipeline.criterion(actual, targets)['total']
        torch.testing.assert_close(first, second, atol=0, rtol=0)
        first.backward(); second.backward()
        for name, parameter in control.named_parameters():
            other = dict(treatment.named_parameters())[name]
            if parameter.grad is None:
                assert other.grad is None, name
            else:
                torch.testing.assert_close(parameter.grad, other.grad, atol=0, rtol=0)
        assert treatment.h_branch.gates.grad.abs().sum() > 0


def test_open_gate_image_gradients_and_dn_bypass():
    pipeline = configuration(h_enabled=True).build(max_epochs=2).train()
    model = pipeline.model
    with torch.no_grad(): model.h_branch.gates.fill_(.1)
    image = torch.randn(1, 3, 64, 64)
    pipeline._losses_for_batch(_batch() | {'image':image})['total'].backward()
    for part in (model.h_branch.image_attention, model.h_branch.write_attention,
                 model.h_branch.read_attention, model.h_branch.feedforward):
        grads = [p.grad for p in part.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum() for g in grads) > 0
    from src.models.pipelines.pose_estimators.poseflow.model_v24 import HypothesisContext
    branch = model.h_branch
    geometry = torch.randn(11, 1, 32)
    context = HypothesisContext(torch.randn(1, 7, 32), torch.randn(1, 7, 32),
                                torch.zeros(1, 7, dtype=torch.bool))
    actual = branch(geometry, layer_index=0, denoising_query_count=2, decoder_context=context)
    torch.testing.assert_close(actual[:2], geometry[:2], atol=0, rtol=0)
    changed = geometry.clone(); changed[:2] += 100
    context.hypotheses = None
    other = branch(changed, layer_index=0, denoising_query_count=2, decoder_context=context)
    torch.testing.assert_close(actual[2:], other[2:], atol=0, rtol=0)
    context.hypotheses = None
    changed_group = geometry.clone(); changed_group[5:8] += 10
    other = branch(changed_group, layer_index=0, denoising_query_count=2, decoder_context=context)
    torch.testing.assert_close(actual[2:5], other[2:5], atol=0, rtol=0)
    context.hypotheses = None
    context.memory = context.memory + torch.randn_like(context.memory)
    image_changed = branch(geometry, layer_index=0, denoising_query_count=2, decoder_context=context)
    assert not torch.allclose(actual[2:], image_changed[2:])
    with pytest.raises(ValueError, match='valid image'):
        bad = HypothesisContext(context.memory, context.position, torch.ones_like(context.padding_mask))
        branch(geometry, layer_index=0, denoising_query_count=2, decoder_context=bad)


@pytest.mark.parametrize('batch_size', [2, 4])
def test_coordinate_isolation_and_open_branch_batch_equivalence(batch_size):
    _, model = matched_models()
    with torch.no_grad(): model.h_branch.gates.fill_(.1)
    decoder = model.transformer.decoder
    noisy, refs = torch.randn(3, batch_size, 2, 2), torch.rand(3, 2, batch_size, 2)
    combined = decoder._refine_diffusion_keypoints(noisy, refs)
    independent = torch.cat([decoder._refine_diffusion_keypoints(noisy[:, i:i+1], refs[:, :, i:i+1])
                             for i in range(batch_size)], dim=2)
    torch.testing.assert_close(combined, independent, atol=1e-6, rtol=1e-6)
    noisy[:, 1:] += 10
    changed = decoder._refine_diffusion_keypoints(noisy, refs)
    torch.testing.assert_close(combined[:, :, :1], changed[:, :, :1], atol=0, rtol=0)
    model.eval()
    images, states = torch.randn(batch_size, 3, 64, 64), torch.randn(batch_size, 3, 4)
    times = torch.linspace(.1, .8, batch_size)
    with torch.no_grad():
        together = model(images, states, times)
        separate = [model(images[i:i+1], states[i:i+1], times[i:i+1]) for i in range(batch_size)]
    for key in ('pred_logits', 'pred_boxes', 'pred_keypoints'):
        torch.testing.assert_close(together[key], torch.cat([o[key] for o in separate]), atol=3e-5, rtol=1e-5)


@pytest.mark.parametrize('enabled', [False, True])
def test_strict_restore_and_native_one_call(tmp_path, enabled):
    from scripts.evaluation.evaluate_poseflow import load_poseflow_model
    config = configuration(h_enabled=enabled)
    pipeline = config.build(max_epochs=2).eval()
    torch.save({'state_dict': pipeline.state_dict()}, tmp_path/'model.ckpt')
    OmegaConf.save(OmegaConf.create({'pipeline':config.model_dump() | {
        '_target_':'src.models.pipelines.pose_estimators.poseflow.PoseFlowV24PipelineConfig'}}), tmp_path/'config.yaml')
    restored, _ = load_poseflow_model(tmp_path/'model.ckpt', tmp_path/'config.yaml')
    restored.eval(); image = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        expected = pipeline.model.sample_keypoints(image, generator=torch.Generator().manual_seed(42))
        actual = restored.sample_keypoints(image, generator=torch.Generator().manual_seed(42))
    for key in ('pred_logits', 'pred_boxes', 'pred_keypoints'):
        torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
    assert bool(restored.h_branch is not None) == enabled
    with pytest.raises(ValueError, match='one call'):
        configuration(flow_sampling_steps=2)


def test_matched_recipe_and_old_default():
    from hydra import compose, initialize_config_dir
    from src.models.pipelines.pose_estimators.poseflow.training import PoseFlowPipelineConfig
    with initialize_config_dir(config_dir=str(Path('conf').resolve()), version_base=None):
        a = compose(config_name='benchmark_poseflow_v24_control')
        b = compose(config_name='benchmark_poseflow_v24_h')
    pa, pb = OmegaConf.to_container(a.pipeline), OmegaConf.to_container(b.pipeline)
    assert pa.pop('h_enabled') is False and pb.pop('h_enabled') is True
    assert pa == pb
    assert a.training == b.training and a.data == b.data
    assert a.pipeline.num_queries == 200 and a.pipeline.num_proposals == 50
    old = PoseFlowPipelineConfig(**{k:v for k,v in configuration().model_dump().items()
        if k in PoseFlowPipelineConfig.model_fields})._build_poseflow_model()
    assert old.transformer.decoder.coordinate_batch_layout == 'legacy'


def test_b1_corrected_control_matches_historical_forward():
    control, _ = matched_models()
    control.eval()
    image, state, times = torch.randn(1, 3, 64, 64), torch.randn(1, 3, 4), torch.tensor([.2])
    with torch.no_grad():
        corrected = control(image, state, times)
        control.transformer.decoder.coordinate_batch_layout = 'legacy'
        historical = control(image, state, times)
    for name in ('pred_logits', 'pred_boxes', 'pred_keypoints'):
        torch.testing.assert_close(corrected[name], historical[name], atol=0, rtol=0)

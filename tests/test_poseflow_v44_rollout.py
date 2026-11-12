from copy import deepcopy

import pytest
import torch

from src.models.pipelines.pose_estimators.poseflow.rollout_v44 import (
    PoseFlowV44RolloutConfig, PoseFlowV44RolloutPipeline, warmstart_rollout,
)
from tests.test_poseflow_v44 import arguments, model, pipeline
from tests.test_poseflow_v43_training import batch


def rollout_pipeline(mode='rollout', probability=1.):
    config = PoseFlowV44RolloutConfig(native=arguments(), codec_width=32, latent_dim=8,
        pretrained_checkpoint=None, quick_eval_enabled=False, state_policy=mode,
        trajectory_probability=probability)
    return PoseFlowV44RolloutPipeline(config=config, model=model(), max_epochs=2).eval()


def test_legacy_input_helper_preserves_exact_state_and_rng():
    module = pipeline()
    clean = torch.randn(2, 6, 8)
    generator = torch.Generator().manual_seed(42)
    reference = torch.Generator().manual_seed(42)
    times = torch.randint(1000, (2,), generator=reference)
    noise = torch.randn(clean.shape, generator=reference)
    state, actual_times, actual_noise, debug = module._diffusion_inputs(clean, generator, None, None, training=True)
    torch.testing.assert_close(actual_times, times, rtol=0, atol=0)
    torch.testing.assert_close(actual_noise, noise, rtol=0, atol=0)
    torch.testing.assert_close(state, module.model.q_sample(clean, times, noise), rtol=0, atol=0)
    assert torch.equal(generator.get_state(), reference.get_state()) and debug == {}


def test_rollout_uses_detached_fixed_context_and_no_clean_injection(monkeypatch):
    module = rollout_pipeline()
    module.train()
    clean = torch.randn(1, 6, 8)
    context = {'identity': object()}
    calls = []
    def denoise(state, times, actual_context, sizes):
        calls.append((int(times[0]), id(actual_context), torch.is_grad_enabled(), module.model.training))
        return {'pred_latents': state * .25}
    monkeypatch.setattr(module.model, 'denoise', denoise)
    modes = [child.training for child in module.model.modules()]
    global_rng = torch.get_rng_state().clone()
    generator = torch.Generator().manual_seed(42)
    state, times, noise, debug = module._diffusion_inputs(clean, generator, context, None, training=True)
    next_rng = generator.get_state()
    alternate = module._diffusion_inputs(clean + 100, torch.Generator().manual_seed(42), context, None, training=True)
    torch.testing.assert_close(state, alternate[0], rtol=0, atol=0)
    assert not state.requires_grad and int(times[0]) in (749, 499, 249)
    assert all(not enabled and not training and key == id(context) for _, key, enabled, training in calls)
    assert modes == [child.training for child in module.model.modules()]
    assert torch.equal(global_rng, torch.get_rng_state())
    expected = noise.clone()
    visited = [999, 749, 499, 249]
    for time, next_time in zip(visited, visited[1:]):
        timestep = torch.tensor([time])
        predicted = expected*.25
        epsilon = module.model.predict_noise_from_start(expected, timestep, predicted)
        expected = module.model.ddim_step(predicted, epsilon, time, next_time, noise=torch.zeros_like(expected))
        if next_time == int(times[0]):
            break
    torch.testing.assert_close(state, expected, rtol=0, atol=0)
    control = rollout_pipeline('forward')
    control.model.load_state_dict(module.model.state_dict(), strict=True)
    control_generator = torch.Generator().manual_seed(42)
    forward, forward_times, forward_noise, _ = control._diffusion_inputs(clean, control_generator, context, None, training=True)
    assert torch.equal(times, forward_times) and torch.equal(noise, forward_noise)
    assert torch.equal(next_rng, control_generator.get_state())
    torch.testing.assert_close(forward, control.model.q_sample(clean, times, noise), rtol=0, atol=0)
    assert debug['trajectory_batch'] == 1 and debug['rollout_batch'] == 1


@pytest.mark.parametrize('people', [(2,), (0,)])
def test_rollout_loss_single_assignment_and_final_gradients(monkeypatch, people):
    module = rollout_pipeline()
    counters = {'frontend': 0, 'matcher': 0}
    contexts, modes = [], []
    prepare, denoise = module.model.prepare_context, module.model.denoise
    def prepare_once(*args, **kwargs):
        counters['frontend'] += 1
        return prepare(*args, **kwargs)
    def traced_denoise(state, times, context, sizes):
        contexts.append(id(context))
        modes.append(torch.is_grad_enabled())
        return denoise(state, times, context, sizes)
    monkeypatch.setattr(module.model, 'prepare_context', prepare_once)
    monkeypatch.setattr(module.model, 'denoise', traced_denoise)
    def matched(*_):
        counters['matcher'] += 1
    hook = module.slot_matcher.register_forward_hook(matched)
    codec = deepcopy(module.model.codec.state_dict())
    loss = module._losses_for_batch(batch(people))['total']
    hook.remove()
    assert torch.isfinite(loss) and counters == {'frontend': 1, 'matcher': 1}
    assert len(set(contexts)) == 1 and modes[-1] and not any(modes[:-1])
    loss.backward()
    assert module.model.latent_heads[-1].weight.grad.abs().sum() > 0
    assert module.model.latent_heads[0].weight.grad.abs().sum() > 0
    assert module.model.latent_input.layers[0].weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in module.model.codec.parameters())
    assert all(torch.equal(value, module.model.codec.state_dict()[name]) for name, value in codec.items())


def test_warmstart_and_checkpoint_policy_contract():
    source = pipeline()
    record = dict(state_dict=deepcopy(source.state_dict()), hyper_parameters=deepcopy(dict(source.hparams)))
    source.on_save_checkpoint(record)
    module = warmstart_rollout(record, state_policy='rollout', max_epochs=2)
    assert module.config.prediction_type == 'x0' and module.config.trajectory_probability == .5
    assert all(torch.equal(value, module.state_dict()[name]) for name, value in record['state_dict'].items())
    saved = {}
    module.on_save_checkpoint(saved)
    module.on_load_checkpoint(saved)
    saved['v44_training_state_policy']['state_policy'] = 'forward'
    with pytest.raises(ValueError, match='policy'):
        module.on_load_checkpoint(saved)
    with pytest.raises(ValueError, match='x0'):
        PoseFlowV44RolloutConfig(prediction_type='v')


def test_rollout_summary_requires_ap_recovery_and_gap_gates(tmp_path):
    import json
    from scripts.training.run_poseflow_v44_rollout import summarize_rollout
    for arm in ('forward', 'rollout'):
        root = tmp_path/arm
        root.mkdir()
        candidate = arm == 'rollout'
        report = dict(checkpoint='chosen.ckpt', checkpoint_sha256=arm, epoch=1, continuation_steps=2500,
            conditions={f'noise_var_100/steps_{steps}': dict(metrics=dict(pose_AP=.3+.02*candidate, bbox_AP=.5))
                for steps in (1,4)}, small_noise={f'noise_{level}/output_paired_OKS_mean': .9
                for level in ('1','5')},
            last_checkpoint_trajectory=dict(OKS_gap=.1-.02*candidate,
                branches={'generated': {'paired_OKS_mean': .6+.02*candidate}}))
        (root/'final_results.json').write_text(json.dumps(report))
        (root/'first_batches.json').write_text('[{"image_ids":[1]}]')
        for epoch in range(2):
            directory = root/'quick_test'/f'epoch{epoch:03d}'
            directory.mkdir(parents=True)
            (directory/'metrics.json').write_text(json.dumps({'conditions':report['conditions']}))
    result = summarize_rollout(tmp_path)
    assert result['candidate_passes_all_gates'] and not result['automatic_promotion']
    report_path = tmp_path/'rollout/final_results.json'
    report = json.loads(report_path.read_text())
    report['small_noise']['noise_5/output_paired_OKS_mean'] = .89
    report_path.write_text(json.dumps(report))
    assert not summarize_rollout(tmp_path)['candidate_passes_all_gates']

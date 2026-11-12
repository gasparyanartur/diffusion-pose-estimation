from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from hydra import compose, initialize_config_dir

from src.models.pipelines.pose_estimators.diffusionregpose.training import build_diffusionregpose_targets
from tests.test_diffusionregpose_training import _batch
from tests.test_poseflow_v42 import config


def repeated_batch(count):
    return {name: value.repeat(count, *([1] * (value.ndim - 1)))
            for name, value in _batch().items()}


def test_explicit_mixed_timesteps_corrupt_coordinates_with_the_requested_schedule():
    model = config().build(max_epochs=80).model
    batch = repeated_batch(3)
    timesteps = torch.tensor([0, 500, 999])
    noise = torch.full((3, 1, 4), .25)
    coordinates, actual_times, targets = build_diffusionregpose_targets(
        batch, model, num_proposals=1, num_body_points=2,
        timesteps=timesteps, noise=noise,
    )
    clean = batch['gt_keypoints'].flatten(2) * model.scale
    expected = model.q_sample(clean, timesteps, noise).to(clean).clamp(-model.scale, model.scale)
    assert torch.equal(actual_times, timesteps)
    assert torch.allclose(coordinates, expected, atol=1e-6, rtol=1e-6)
    assert not torch.equal(coordinates[0], coordinates[-1])
    assert len(targets) == 3


def test_omitted_timesteps_preserve_exact_legacy_corruption_and_rng():
    model = config().build(max_epochs=80).model
    torch.manual_seed(29)
    legacy = build_diffusionregpose_targets(_batch(), model, num_proposals=2, num_body_points=2)
    legacy_rng = torch.get_rng_state()
    torch.manual_seed(29)
    explicit = build_diffusionregpose_targets(
        _batch(), model, num_proposals=2, num_body_points=2, timesteps=torch.zeros(1, dtype=torch.long),
    )
    assert torch.equal(legacy_rng, torch.get_rng_state())
    assert all(torch.equal(left, right) for left, right in zip(legacy[:2], explicit[:2]))
    assert all(torch.equal(legacy[2][0][key], explicit[2][0][key]) for key in legacy[2][0])


@pytest.mark.parametrize('timesteps', [
    torch.tensor([0.]), torch.tensor([[0]]), torch.tensor([0, 1]),
    torch.tensor([-1]), torch.tensor([1000]), torch.empty(1, dtype=torch.long, device='meta'),
])
def test_invalid_explicit_timesteps_are_rejected(timesteps):
    model = config().build(max_epochs=80).model
    with pytest.raises((ValueError, TypeError), match='[Tt]imestep'):
        build_diffusionregpose_targets(
            _batch(), model, num_proposals=2, num_body_points=2, timesteps=timesteps,
        )


@pytest.mark.parametrize('empty', [False, True])
def test_uniform_training_forwards_actual_random_timesteps_and_uses_output_matching(empty):
    pipeline = config(training_timestep_policy='uniform', pose_readout_mode='box_center_residual').build(max_epochs=80)
    batch = repeated_batch(4)
    batch['image'] = torch.randn(4, 3, 64, 64)
    if empty:
        batch['gt_labeled'].zero_()
    observed = []
    hook = pipeline.model.register_forward_pre_hook(lambda module, args: observed.append(args[2].detach().clone()))
    with patch.object(pipeline.model, 'q_sample', wraps=pipeline.model.q_sample) as corrupt:
        with patch.object(pipeline.criterion, 'forward', wraps=pipeline.criterion.forward) as criterion:
            losses = pipeline._losses_for_batch(batch)
    hook.remove()
    times = corrupt.call_args.args[1]
    assert torch.equal(observed[0], times)
    assert times.shape == (4,) and times.dtype == torch.long
    assert (times > 0).any() and times.unique().numel() > 1
    assert ((times >= 0) & (times < 1000)).all()
    assert criterion.call_args.kwargs == {}
    assert torch.isfinite(losses['total'])
    assert losses['diffusion/timestep_mean'] == times.float().mean()
    assert pipeline.hparams['training_timestep_policy'] == 'uniform'
    losses['total'].backward()
    assert pipeline.model.pose_encoder.layers[0].weight.grad is not None


def test_uniform_config_is_fresh_box_center_output_hungarian_and_full_schedule():
    with initialize_config_dir(version_base=None, config_dir=str(Path(__file__).resolve().parents[1] / 'conf')):
        preset = compose(config_name='benchmark_poseflow_v42_uniform_t')
    assert preset.pipeline.training_timestep_policy == 'uniform'
    assert preset.pipeline.pose_readout_mode == 'box_center_residual'
    assert preset.pipeline.matching_strategy == 'hungarian'
    assert 'target_assignment' not in preset.pipeline
    assert preset.training.max_epochs == 80
    assert preset.pipeline.quick_eval_steps == [1, 4, 8]
    assert 'uniform_t' in preset.run_name and 'uniform_t' in preset.out_dir


def test_historical_config_and_initial_model_weights_are_preserved():
    torch.manual_seed(42)
    legacy = config().build(max_epochs=80)
    torch.manual_seed(42)
    uniform = config(training_timestep_policy='uniform').build(max_epochs=80)
    assert legacy.v42_config.training_timestep_policy == 'timestep_zero'
    assert all(torch.equal(value, uniform.state_dict()[name]) for name, value in legacy.state_dict().items())
    uniform.load_state_dict(legacy.state_dict(), strict=True)

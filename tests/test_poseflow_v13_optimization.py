from types import SimpleNamespace

import pytest
import torch

from tests.test_poseflow import _batch
from tests.test_poseflow_v13 import _config
from src.models.pipelines.pose_estimators.poseflow.optimization_v13 import (
    gradient_norms, reconstruction_weight,
)


def test_reconstruction_schedule_endpoints_midpoint_and_resume_epoch():
    assert reconstruction_weight(1., .01, 41, 0) == 1.
    assert reconstruction_weight(1., .01, 41, 20) == pytest.approx(.1)
    assert reconstruction_weight(1., .01, 41, 40) == .01
    assert reconstruction_weight(1., .01, 41, 79) == .01
    assert reconstruction_weight(1., 1., 40, 20) == 1.
    with pytest.raises(ValueError):
        _config(reconstruction_weight_end=0)
    with pytest.raises(ValueError):
        _config(reconstruction_decay_epochs=1)


def test_weighted_objective_changes_codec_gradient_without_changing_raw_losses():
    baseline = _config().build(max_epochs=80)
    scheduled = _config(reconstruction_weight_start=1., reconstruction_weight_end=.01,
                        reconstruction_decay_epochs=40).build(max_epochs=80)
    scheduled.load_state_dict(baseline.state_dict(), strict=True)
    scheduled._trainer = SimpleNamespace(current_epoch=39)
    batch = _batch() | {'image': torch.randn(1, 3, 64, 64)}
    torch.manual_seed(42)
    before = baseline._losses_for_batch(batch)
    torch.manual_seed(42)
    after = scheduled._losses_for_batch(batch)
    torch.testing.assert_close(before['loss_reconstruction'], after['loss_reconstruction'], rtol=0, atol=0)
    torch.testing.assert_close(after['total'], after['loss_latent_mean'] + after['loss_prediction']
                               + .01 * after['loss_reconstruction'])
    torch.testing.assert_close(before['total'], before['loss_latent_mean'] + before['loss_prediction']
                               + before['loss_reconstruction'], rtol=0, atol=0)
    before['total'].backward()
    after['total'].backward()
    torch.testing.assert_close(scheduled.model.codec.joint_encoder.layers[0].weight.grad,
                               .01 * baseline.model.codec.joint_encoder.layers[0].weight.grad)


def test_gradient_norms_and_objective_probes_do_not_populate_or_replace_gradients():
    parameter = torch.nn.Parameter(torch.tensor([1., 2.]))
    parameter.grad = torch.tensor([7., 8.])
    expected = parameter.grad.clone()
    norms = gradient_norms([('model.codec.point_head.weight', parameter)], [torch.tensor([3., 4.])])
    assert norms['global'] == norms['geometry_readouts'] == 5
    torch.testing.assert_close(parameter.grad, expected)
    pipeline = _config(optimization_diagnostics_enabled=True).build(max_epochs=2)
    pipeline._trainer = SimpleNamespace(current_epoch=0, is_global_zero=True)
    pipeline._log_optimization_metrics = lambda metrics: None
    losses = pipeline._losses_for_batch(_batch() | {'image': torch.randn(1, 3, 64, 64)})
    pipeline._log_objective_gradients(losses)
    assert all(parameter.grad is None for parameter in pipeline.parameters())
    losses['total'].backward()
    assert pipeline.model.codec.joint_encoder.layers[0].weight.grad.abs().sum() > 0


def test_optimizer_hook_reports_accumulated_norm_before_clipping_and_is_bounded():
    pipeline = _config(optimization_diagnostics_enabled=True).build(max_epochs=2)
    trainer = SimpleNamespace(current_epoch=0, global_step=1, is_global_zero=True, gradient_clip_val=2.)
    pipeline._trainer = trainer
    parameter = next(parameter for parameter in pipeline.parameters() if parameter.requires_grad)
    parameter.grad = torch.ones_like(parameter)
    expected = parameter.grad.clone()
    logged = {}
    pipeline._log_optimization_metrics = lambda metrics: logged.update(metrics)
    optimizer = torch.optim.AdamW([parameter], lr=1e-4)
    pipeline.on_before_optimizer_step(optimizer)
    norm = expected.norm()
    torch.testing.assert_close(logged['debug/optimization/preclip/global'], norm)
    torch.testing.assert_close(logged['debug/optimization/clip_factor'], (2 / (norm + 1e-6)).clamp(max=1))
    torch.testing.assert_close(parameter.grad, expected)
    logged.clear()
    trainer.global_step = 2
    pipeline.on_before_optimizer_step(optimizer)
    assert not logged
    trainer.global_step = 100
    pipeline.on_before_optimizer_step(optimizer)
    assert logged


def test_experiment_config_is_isolated_and_uses_approved_recipe():
    from pathlib import Path
    from hydra import compose, initialize_config_dir
    from src.models.pipelines.pose_estimators.poseflow import PoseFlowPipelineConfig
    with initialize_config_dir(config_dir=str(Path('conf').resolve()), version_base=None):
        config = compose(config_name='benchmark_poseflow_v13_reconstruction_decay')
    parsed = PoseFlowPipelineConfig.from_hydra_config(config.pipeline)
    assert (parsed.reconstruction_weight_start, parsed.reconstruction_weight_end,
            parsed.reconstruction_decay_epochs) == (1., .01, 40)
    assert parsed.optimization_diagnostics_enabled
    assert config.training.grad_clip == 2.
    assert config.training.max_epochs == 80 and config.training.resume is None
    assert config.out_dir == 'results/poseflow_v13_reconstruction_decay'


def test_sparse_optimization_metrics_reach_all_loggers_without_training_interval():
    pipeline = _config().build(max_epochs=2)
    captured = []
    logger = SimpleNamespace(log_metrics=lambda values, step: captured.append((values, step)))
    pipeline._trainer = SimpleNamespace(global_step=3, loggers=[logger, logger])
    pipeline._log_optimization_metrics({'debug/optimization/probe': torch.tensor(2.)})
    assert captured == [({'debug/optimization/probe': 2.}, 3)] * 2

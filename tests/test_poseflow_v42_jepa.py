from unittest.mock import patch
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

from tests.test_poseflow_v42_image_confidence import image_config
from tests.test_poseflow_v42_uniform_t import repeated_batch
from src.models.pipelines.pose_estimators.poseflow.training_v42_jepa import PoseFlowV42JEPAPipelineConfig


def jepa_config():
    return PoseFlowV42JEPAPipelineConfig(**image_config().model_dump())


def test_initial_model_and_rng_match_parent_teacher_is_not_optimized():
    torch.manual_seed(42)
    parent = image_config().build(max_epochs=80)
    rng = torch.get_rng_state()
    torch.manual_seed(42)
    pipeline = jepa_config().build(max_epochs=80)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(v, pipeline.model.state_dict()[k]) for k, v in parent.model.state_dict().items())
    pipeline.train()
    assert not pipeline.teacher.training
    optimized = {id(p) for g in pipeline.configure_optimizers()['optimizer'].param_groups for p in g['params']}
    assert all(not p.requires_grad and id(p) not in optimized for p in pipeline.teacher.parameters())


def test_losses_follow_pairing_and_native_reader_with_intended_gradients():
    pipeline = jepa_config().build(max_epochs=80)
    batch = repeated_batch(3)
    batch['image'] = torch.randn(3, 3, 64, 64)
    with patch.object(pipeline.criterion, 'forward', wraps=pipeline.criterion.forward) as criterion:
        with patch.object(pipeline.teacher, 'forward', wraps=pipeline.teacher.forward) as teacher:
            losses = pipeline._losses_for_batch(batch)
    outputs, targets = criterion.call_args.args
    expected = torch.cat([targets[b]['keypoints'][people, :4]
                          for b, (_, people) in enumerate(outputs['pose_target_matches'])])
    assert torch.equal(teacher.call_args.args[0], expected)
    assert losses['jepa/positive_count'] == len(expected)
    assert losses['jepa/loss'] > 0 and losses['jepa/clean_geometry'] > 0
    assert torch.allclose(losses['total'], losses['jepa/baseline_total'] + losses['jepa/loss'] + .5 * losses['jepa/clean_geometry'])
    losses['total'].backward()
    for module in (pipeline.model.pose_encoder, pipeline.model.pose_readers, pipeline.model.person_attention):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    assert all(p.grad is None for p in pipeline.teacher.parameters())
    assert pipeline.ema_updates == 0


def test_empty_targets_and_ema_strict_restore():
    pipeline = jepa_config().build(max_epochs=80)
    batch = repeated_batch(2)
    batch['image'] = torch.randn(2, 3, 64, 64)
    batch['gt_labeled'].zero_()
    losses = pipeline._losses_for_batch(batch)
    assert losses['jepa/loss'] == losses['jepa/clean_geometry'] == 0
    assert torch.isfinite(losses['total'])
    losses['total'].backward()
    old = next(pipeline.teacher.parameters()).clone()
    with torch.no_grad():
        next(pipeline.model.pose_encoder.parameters()).add_(1)
    pipeline.update_ema()
    assert pipeline.ema_updates == 1
    assert torch.allclose(next(pipeline.teacher.parameters()), old + .001, atol=1e-7)
    restored = jepa_config().build(max_epochs=80)
    restored.load_state_dict(pipeline.state_dict(), strict=True)
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in pipeline.state_dict().items())


def test_optional_references_preserve_outputs_and_sampling_without_teacher():
    torch.manual_seed(42)
    parent = image_config().build(max_epochs=80).model.eval()
    torch.manual_seed(42)
    model = jepa_config().build(max_epochs=80).model.eval()
    image = torch.randn(1, 3, 64, 64)
    state = torch.randn(1, 2, 4)
    with torch.no_grad():
        a = parent(image, state, torch.tensor([500]))
        b = model(image, state, torch.tensor([500]))
        assert 'pose_reference_boxes' not in a
        assert torch.equal(model.read_pose(b['pose_features'], b['pose_reference_boxes'], layer_index=1), b['pred_keypoints'][..., :4])
        for key in ('pred_keypoints', 'pred_boxes', 'pred_logits', 'pose_features'):
            assert torch.equal(a[key], b[key])
        model.sample_keypoints(image, generator=torch.Generator().manual_seed(42))


def test_preset_dispatch_and_optimizer_step_ema_cadence():
    from src.models.pipelines.pose_estimators.diffusionregpose.training import DiffusionRegPosePipeline, DiffusionRegPosePipelineConfig
    with initialize_config_dir(version_base=None, config_dir=str(Path(__file__).resolve().parents[1] / 'conf')):
        preset = compose(config_name='benchmark_poseflow_v42_jepa')
    settings = DiffusionRegPosePipelineConfig.from_hydra_config(preset.pipeline)
    assert isinstance(settings, PoseFlowV42JEPAPipelineConfig)
    assert preset.training.max_epochs == 80 and settings.quick_eval_steps == (1, 4, 8)
    pipeline = jepa_config().build(max_epochs=80)
    with patch.object(DiffusionRegPosePipeline, 'optimizer_step') as step:
        pipeline.optimizer_step(0, 0, None)
        step.assert_called_once()
    assert pipeline.ema_updates == 1
    pipeline.eval()
    assert pipeline.ema_updates == 1 and not pipeline.teacher.training


def test_checkpoint_hooks_restore_random_draws():
    import random
    import numpy as np
    pipeline = jepa_config().build(max_epochs=80)
    checkpoint = {}
    pipeline.on_save_checkpoint(checkpoint)
    expected = (random.random(), np.random.rand(), torch.rand(3))
    pipeline.on_load_checkpoint(checkpoint)
    actual = (random.random(), np.random.rand(), torch.rand(3))
    assert expected[:2] == actual[:2] and torch.equal(expected[2], actual[2])

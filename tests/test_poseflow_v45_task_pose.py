from unittest.mock import patch

import pytest
import torch

from src.models.pipelines.pose_estimators.poseflow.training_v45_task_pose import PoseFlowV45TaskPosePipelineConfig
from tests.test_poseflow_v42_image_confidence import image_config
from tests.test_poseflow_v42_uniform_t import repeated_batch


def settings(weight=0.):
    values = image_config().model_dump()
    values['pose_readout_mode'] = 'absolute'
    return PoseFlowV45TaskPosePipelineConfig(**values, jepa_weight=weight)


def batch(empty=False, partial=False):
    value = repeated_batch(2)
    value['image'] = torch.randn(2, 3, 64, 80)
    if empty:
        value['gt_labeled'].zero_()
    elif partial:
        value['gt_labeled'][..., 0] = False
    return value


def test_arm_initialization_rng_codec_and_identity_residual():
    torch.manual_seed(42)
    control = settings().build(max_epochs=80)
    rng = torch.get_rng_state()
    torch.manual_seed(42)
    jepa = settings(1.).build(max_epochs=80)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(v, jepa.state_dict()[k]) for k, v in control.state_dict().items())
    model = control.model.eval()
    xy = torch.rand(3, 4)
    code = model.pose_codec.encode(xy)
    assert code.shape == (3, 32)
    torch.testing.assert_close(code.mean(-1), torch.zeros(3), atol=1e-6, rtol=0)
    assert model.pose_codec.decode(code).shape == xy.shape
    context = model.prepare_context(torch.randn(1, 3, 64, 80))
    state = torch.randn(1, 2, 4)
    output = model.denoise_coordinates(state, torch.tensor([500]), context)
    encoded = model.pose_codec.encode((state / 6 + 1) / 2)
    torch.testing.assert_close(output['pred_latents'], model.pose_codec.norm(encoded))
    torch.testing.assert_close(output['pred_keypoints'][..., :4], model.pose_codec.decode(output['pred_latents']))
    assert not any('pose_readers' in k for k in model.state_dict())


@pytest.mark.parametrize('steps', [1, 4, 8])
def test_context_cached_and_exact_fixed_scores_boxes_ids(steps):
    model = settings().build(max_epochs=80).model.eval()
    seen = []
    original = model.denoise_coordinates
    def capture(state, times, context):
        output = original(state, times, context)
        seen.append((output['pred_logits'].clone(), output['pred_boxes'].clone(), context['indices'].clone(), state.clone()))
        return output
    with patch.object(model, 'prepare_context', wraps=model.prepare_context) as frontend:
        with patch.object(model, 'denoise_coordinates', side_effect=capture):
            model.sampling_timesteps = steps
            result = model.sample_keypoints(torch.randn(1, 3, 64, 80), generator=torch.Generator().manual_seed(42))
    assert frontend.call_count == 1 and len(seen) == steps
    assert all(torch.equal(a, b) for row in seen for a, b in zip(seen[0][:3], row[:3]))
    assert result['diffusion_keypoints'].shape == (1, 2, 4)
    assert all(attention.num_points == 12 for attention in model.person_attention)
    assert not result['pred_boxes'].requires_grad


@pytest.mark.parametrize('empty,partial', [(False, False), (True, False), (False, True)])
def test_complete_only_losses_pairing_gradient_masks_and_teacher(empty, partial):
    pipeline = settings(1.).build(max_epochs=80)
    with patch.object(pipeline.criterion.matcher, 'forward', wraps=pipeline.criterion.matcher.forward) as matcher:
        losses = pipeline._losses_for_batch(batch(empty, partial))
    assert torch.isfinite(losses['total'])
    assert losses['task_pose/complete_count'] == (0 if empty or partial else 2)
    if empty or partial:
        assert losses['task_pose/jepa'] == losses['task_pose/clean'] == 0
    losses['total'].backward()
    assert all(p.grad is None for p in pipeline.teacher.parameters())
    assert pipeline.ema_updates == 0
    for module in (pipeline.model.pose_codec, pipeline.model.latent_residual):
        if not empty:
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    # Selected source pairing, native 200-candidate box loss and encoder proposal loss.
    assert matcher.call_count == 3


def test_only_jepa_coefficient_changes_objective_and_native_geometry_formula():
    from src.models.pipelines.pose_estimators.poseflow.training_v45_task_pose import pose_geometry_loss
    control, jepa = settings().build(max_epochs=80), settings(1.).build(max_epochs=80)
    jepa.load_state_dict(control.state_dict(), strict=True)
    value = batch()
    torch.manual_seed(71)
    first = control._losses_for_batch(value)
    rng = torch.get_rng_state()
    torch.manual_seed(71)
    second = jepa._losses_for_batch(value)
    assert torch.equal(rng, torch.get_rng_state())
    torch.testing.assert_close(second['total'] - first['total'], second['task_pose/jepa'])
    torch.testing.assert_close(first['task_pose/clean'], second['task_pose/clean'])
    targets = [{'labels': torch.tensor([1]), 'boxes': torch.tensor([[.5, .5, .4, .4]]),
                'area': torch.tensor([.16]), 'keypoints': torch.tensor([[.2, .3, .6, .7, 1., 1.]])}]
    outputs = dict(pred_logits=torch.zeros(1, 1, 2), pred_boxes=targets[0]['boxes'][None],
                   pred_keypoints=torch.tensor([[[.3, .4, .7, .8, 1., 1.]]]))
    criterion = control.criterion
    losses = criterion._loss_for_outputs(outputs, targets, matches=[(torch.tensor([0]), torch.tensor([0]))])
    expected = criterion.weights.keypoints * losses['loss_keypoints'] + criterion.weights.oks * losses['loss_oks']
    actual = pose_geometry_loss(outputs['pred_keypoints'][0, :, :4], targets[0]['keypoints'],
        targets[0]['area'], criterion.matcher.sigmas, criterion.weights)
    torch.testing.assert_close(actual, expected)


def test_capacity_rejected_and_presets_dispatch():
    from pathlib import Path
    from hydra import compose, initialize_config_dir
    from src.models.pipelines.pose_estimators.diffusionregpose.training import DiffusionRegPosePipelineConfig
    with initialize_config_dir(version_base=None, config_dir=str(Path(__file__).resolve().parents[1] / 'conf')):
        for arm, weight in (('control', 0.), ('jepa', 1.)):
            config = compose(config_name=f'benchmark_poseflow_v45_task_pose_{arm}')
            parsed = DiffusionRegPosePipelineConfig.from_hydra_config(config.pipeline)
            assert isinstance(parsed, PoseFlowV45TaskPosePipelineConfig) and parsed.jepa_weight == weight
            assert config.training.max_epochs == 80 and config.training.accumulate_grad_batches == 2
    model = settings().build(max_epochs=80).model.eval()
    with pytest.raises(ValueError):
        model.sample_keypoints(torch.randn(1, 3, 64, 80), num_proposals=3)
    pipeline = settings().build(max_epochs=80)
    value = batch()
    value['gt_labeled'] = value['gt_labeled'].repeat(1, 3, 1)
    with patch.object(pipeline.model, 'prepare_context', side_effect=AssertionError('frontend must not run')):
        with pytest.raises(ValueError, match='capacity'):
            pipeline._losses_for_batch(value)


def test_clean_and_jepa_gradient_routes_are_separate():
    pipeline = settings(1.).build(max_epochs=80)
    losses = pipeline._losses_for_batch(batch())
    losses['task_pose/clean'].backward(retain_graph=True)
    for module in (pipeline.model.pose_codec.encoder, pipeline.model.pose_codec.reader):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    pipeline.zero_grad(set_to_none=True)
    losses['task_pose/jepa'].backward()
    assert pipeline.model.latent_residual[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in pipeline.model.pose_codec.reader.parameters())
    assert all(p.grad is None for p in pipeline.teacher.parameters())


def test_teacher_optimizer_exclusion_ema_rng_and_contract_restore():
    pipeline = settings().build(max_epochs=80)
    pipeline.train()
    assert not pipeline.teacher.training
    ids = {id(p) for group in pipeline.configure_optimizers()['optimizer'].param_groups for p in group['params']}
    assert all(id(p) not in ids and not p.requires_grad for p in pipeline.teacher.parameters())
    before = next(pipeline.teacher.parameters()).clone()
    with torch.no_grad():
        next(pipeline.model.pose_codec.encoder.parameters()).add_(1)
    pipeline.update_ema()
    torch.testing.assert_close(next(pipeline.teacher.parameters()), before + .001)
    checkpoint = {}
    pipeline.on_save_checkpoint(checkpoint)
    expected = torch.rand(3)
    restored = settings().build(max_epochs=80)
    restored.on_load_checkpoint(checkpoint)
    assert torch.equal(expected, torch.rand(3))
    restored.load_state_dict(pipeline.state_dict(), strict=True)
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in pipeline.state_dict().items())
    with pytest.raises(ValueError):
        settings(1.).build(max_epochs=80).on_load_checkpoint(checkpoint)

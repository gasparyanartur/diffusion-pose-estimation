import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from pathlib import Path

from src.models.pipelines.pose_estimators.diffusionregpose.training import (
    DiffusionRegPoseCriterion, DiffusionRegPoseMatcher,
    build_diffusionregpose_targets,
)
from tests.test_diffusionregpose_training import _batch
from tests.test_poseflow_v42 import config


def targets():
    xy = torch.tensor([[0.2, 0.2, 0.25, 0.25], [0.8, 0.8, 0.75, 0.75]])
    return [dict(labels=torch.ones(2, dtype=torch.long),
                 boxes=torch.tensor([[0.2, 0.2, 0.2, 0.2], [0.8, 0.8, 0.2, 0.2]]),
                 keypoints=torch.cat((xy, torch.ones(2, 2)), -1),
                 area=torch.full((2,), 0.04))]


def test_noisy_input_matching_swaps_targets_without_gradients_or_rng_draws():
    from src.models.pipelines.pose_estimators.poseflow.training_v42_matched_target import match_noisy_pose_targets

    target = targets()
    matcher = DiffusionRegPoseMatcher(num_body_points=2)
    noisy = ((target[0]['keypoints'][:, :4].flip(0)[None] * 2 - 1) * 6).requires_grad_()
    rng = torch.get_rng_state().clone()
    matches = match_noisy_pose_targets(noisy, target, scale=6, matcher=matcher)
    assert matches[0][0].tolist() == [0, 1]
    assert matches[0][1].tolist() == [1, 0]
    assert torch.equal(rng, torch.get_rng_state())
    assert noisy.grad is None
    assert not matches[0][1].requires_grad


def test_input_cost_uses_masked_pose_geometry_even_for_zero_xy():
    from src.models.pipelines.pose_estimators.poseflow.training_v42_matched_target import noisy_pose_matching_costs

    target = targets()
    target[0]['keypoints'][0, 4:] = torch.tensor([1., 0.])
    matcher = DiffusionRegPoseMatcher(num_body_points=2)
    # Native -scale is normalized zero, which must not trigger dummy-pose box costs.
    noisy = torch.full((1, 3, 4), -6., requires_grad=True)
    costs = noisy_pose_matching_costs(noisy, target, scale=6, matcher=matcher)
    xy = torch.zeros(3, 2, 2)
    clean = target[0]['keypoints'][:, :4].reshape(2, 2, 2)
    visible = target[0]['keypoints'][:, 4:]
    distance = (xy[:, None] - clean[None]).square().sum(-1)
    oks = (-distance / (target[0]['area'][None, :, None]
                       * (2 * matcher.sigmas).square()[None, None] * 2)).exp()
    oks = (oks * visible[None]).sum(-1) / (visible.sum(-1)[None] + 1e-6)
    l1 = ((xy[:, None] - clean[None]).abs() * visible[None, :, :, None]).sum((-1, -2))
    expected = 10 * l1 + 4 * (1 - oks.clamp_min(1e-6))
    assert torch.equal(costs[0], expected)
    assert not costs[0].requires_grad
    target[0]['keypoints'][0, 2:4] = 1000
    changed = noisy_pose_matching_costs(noisy, target, scale=6, matcher=matcher)
    assert torch.equal(costs[0][:, 0], changed[0][:, 0])


def test_empty_surplus_capacity_and_batch_isolation():
    from src.models.pipelines.pose_estimators.poseflow.training_v42_matched_target import match_noisy_pose_targets

    target = targets()[0]
    empty = {name: value[:0] for name, value in target.items()}
    matcher = DiffusionRegPoseMatcher(num_body_points=2)
    noisy = torch.tensor([[[0., 0., 0., 0.]], [[6., 6., 6., 6.]]])
    result = match_noisy_pose_targets(noisy, [empty, target], scale=6, matcher=matcher)
    assert result[0][0].numel() == result[0][1].numel() == 0
    assert result[1][0].tolist() == [0] and result[1][1].tolist() == [1]
    reversed_result = match_noisy_pose_targets(noisy.flip(0), [target, empty], scale=6, matcher=matcher)
    assert torch.equal(result[1][1], reversed_result[0][1])
    surplus = match_noisy_pose_targets(torch.zeros(1, 4, 4), [target], scale=6, matcher=matcher)
    assert len(surplus[0][0]) == 2 and len(surplus[0][1].unique()) == 2


def test_explicit_aux_matches_share_targets_and_preserve_default_losses():
    criterion = DiffusionRegPoseCriterion(num_body_points=2, num_box_decoder_layers=1)
    target = targets()
    output = dict(pred_logits=torch.zeros(1, 2, 2, requires_grad=True),
                  pred_boxes=target[0]['boxes'][None].clone(),
                  pred_keypoints=target[0]['keypoints'][None].clone().requires_grad_())
    box_aux = {name: value.detach().clone() for name, value in output.items()}
    box_aux['pred_keypoints'].zero_()
    output['aux_outputs'] = [box_aux, {name: value for name, value in output.items() if name != 'aux_outputs'}]
    fixed = [(torch.tensor([0, 1]), torch.tensor([1, 0]))]
    legacy = criterion(output, target)
    actual = criterion(output, target, final_matches=fixed, auxiliary_matches=[None, fixed])
    assert actual['loss_keypoints'] > 0
    assert torch.equal(actual['loss_keypoints'], actual['loss_keypoints_1'])
    assert torch.equal(legacy['loss_bbox_0'], actual['loss_bbox_0'])
    assert torch.equal(legacy['loss_ce_0'], actual['loss_ce_0'])
    assert all(torch.equal(value, criterion(output, target, auxiliary_matches=None)[name])
               for name, value in legacy.items())
    actual['total'].backward()
    assert output['pred_keypoints'].grad.abs().sum() > 0
    with pytest.raises(ValueError, match='auxiliary'):
        criterion(output, target, auxiliary_matches=[])


@pytest.mark.parametrize('empty', [False, True])
def test_variant_native_loss_and_assignment_routing(empty, monkeypatch):
    from src.models.pipelines.pose_estimators.poseflow.training_v42_matched_target import (
        PoseFlowV42MatchedTargetPipelineConfig,
    )

    settings = PoseFlowV42MatchedTargetPipelineConfig(**config(pose_readout_mode='box_center_residual').model_dump())
    pipeline = settings.build(max_epochs=80)
    batch = _batch()
    batch['image'] = torch.randn(1, 3, 64, 64)
    # A cropped-out person is retained for diffusion, never as a supervised target.
    batch['gt_diffusion_keypoints'] = torch.cat((batch['gt_keypoints'], -batch['gt_keypoints']), 1)
    batch['diffusion_person_mask'] = torch.ones(1, 2, dtype=torch.bool)
    if empty:
        batch['gt_labeled'].zero_()
    original = pipeline.criterion.forward
    captured = []

    def record(outputs, target, **kwargs):
        captured.append((target, kwargs))
        return original(outputs, target, **kwargs)

    monkeypatch.setattr(pipeline.criterion, 'forward', record)
    losses = pipeline._losses_for_batch(batch)
    target, kwargs = captured[0]
    assert target[0]['labels'].numel() == (0 if empty else 1)
    assert kwargs['auxiliary_matches'][0] is None
    assert kwargs['auxiliary_matches'][1] is kwargs['final_matches']
    assert all(not value.requires_grad for name, value in losses.items() if name.startswith('input_match/'))
    assert torch.isfinite(losses['total']) and 'dn_loss_bbox' in losses
    losses['total'].backward()
    if not empty:
        assert pipeline.model.pose_encoder.layers[0].weight.grad.abs().sum() > 0
        assert pipeline.model.pose_readers[-1].layers[-1].weight.grad.abs().sum() > 0
    restored = settings.build(max_epochs=80, load_pretrained_checkpoint=False)
    restored.load_state_dict(pipeline.state_dict(), strict=True)


def test_variant_config_initialization_noising_and_inference_match_v42():
    from src.models.pipelines.pose_estimators.poseflow.training_v42_matched_target import PoseFlowV42MatchedTargetPipelineConfig

    with initialize_config_dir(version_base=None, config_dir=str(Path(__file__).resolve().parents[1] / 'conf')):
        base = compose(config_name='benchmark_poseflow_v42')
        variant = compose(config_name='benchmark_poseflow_v42_matched_target')
    assert OmegaConf.to_container(base.data) == OmegaConf.to_container(variant.data)
    assert OmegaConf.to_container(base.training) == OmegaConf.to_container(variant.training)
    assert variant.run_name == 'v42-matched-target' and variant.training.max_epochs == 80
    assert variant.pipeline.target_assignment == 'noisy_input_hungarian'
    torch.manual_seed(42)
    baseline = config(pose_readout_mode='box_center_residual').build(max_epochs=80)
    baseline_rng = torch.get_rng_state().clone()
    torch.manual_seed(42)
    settings = PoseFlowV42MatchedTargetPipelineConfig(**config(pose_readout_mode='box_center_residual').model_dump())
    candidate = settings.build(max_epochs=80)
    assert torch.equal(baseline_rng, torch.get_rng_state())
    assert all(torch.equal(value, candidate.state_dict()[name]) for name, value in baseline.state_dict().items())
    assert candidate.hparams['target_assignment'] == 'noisy_input_hungarian'
    first = build_diffusionregpose_targets(_batch(), baseline.model, num_proposals=2, num_body_points=2, noise_seed=42)
    second = build_diffusionregpose_targets(_batch(), candidate.model, num_proposals=2, num_body_points=2, noise_seed=42)
    assert torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])
    baseline.eval(); candidate.eval()
    image = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        for steps in (1, 4, 8):
            baseline.model.sampling_timesteps = candidate.model.sampling_timesteps = steps
            reference = baseline.model.sample_keypoints(image, generator=torch.Generator().manual_seed(12))
            actual = candidate.model.sample_keypoints(image, generator=torch.Generator().manual_seed(12))
            for name in ('pred_logits', 'pred_boxes', 'pred_keypoints'):
                assert torch.equal(reference[name], actual[name])

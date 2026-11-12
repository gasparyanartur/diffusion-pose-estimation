from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from hydra import compose, initialize_config_dir

from src.models.pipelines.pose_estimators.diffusionregpose.training import (
    DiffusionRegPoseMatcher, build_diffusionregpose_clean_targets,
)
from src.models.pipelines.pose_estimators.poseflow.training_v42_paired_proposal import (
    PoseFlowV42PairedProposalPipelineConfig, build_proposal_paired_pose_inputs,
)
from tests.test_diffusionregpose_training import _batch
from tests.test_poseflow_v42 import config
from tests.test_poseflow_v42_matched_target import targets
from tests.test_poseflow_v42_uniform_t import repeated_batch


def paired_config():
    values = config(pose_readout_mode='box_center_residual', training_timestep_policy='uniform').model_dump()
    return PoseFlowV42PairedProposalPipelineConfig(**values)


def test_swapped_proposals_pair_clean_source_before_known_noise_and_keep_background():
    model = paired_config().build(max_epochs=80).model
    target = targets()[0]
    logits = torch.zeros(3, 3, 2, requires_grad=True)
    boxes = torch.cat((target['boxes'].flip(0), torch.tensor([[.5, .5, .05, .05]])))[None].repeat(3, 1, 1)
    times = torch.tensor([0, 500, 999])
    noise = torch.full((3, 3, 4), .25)
    captured = []
    original = model.q_sample

    def record(clean, actual_times, noise):
        captured.append(clean.clone())
        return original(clean, actual_times, noise)

    rng = torch.get_rng_state().clone()
    with patch.object(model, 'q_sample', side_effect=record):
        noisy, matches = build_proposal_paired_pose_inputs(
            logits, boxes, times, [target] * 3, diffusion=model,
            matcher=DiffusionRegPoseMatcher(num_body_points=2), noise=noise, noise_seed=42,
        )
    assert torch.equal(rng, torch.get_rng_state())
    for queries, people in matches:
        assert queries.tolist() == [0, 1] and people.tolist() == [1, 0]
    expected_clean = (target['keypoints'][:, :4].flip(0) * 2 - 1) * model.scale
    assert torch.equal(captured[0][:, :2], expected_clean[None].expand(3, -1, -1))
    expected = original(captured[0], times, noise).to(noisy).clamp(-model.scale, model.scale)
    assert torch.equal(noisy, expected) and not noisy.requires_grad
    assert not torch.equal(noisy[0], noisy[-1])
    assert captured[0][:, 2].abs().sum() > 0


def test_clean_targets_exclude_cropped_people_and_capacity_is_explicit():
    batch = repeated_batch(2)
    batch['gt_labeled'][1].zero_()
    batch['gt_diffusion_keypoints'] = torch.full((2, 9, 2, 2), 100.)
    target = build_diffusionregpose_clean_targets(batch, num_body_points=2)
    assert [len(row['labels']) for row in target] == [1, 0]
    assert torch.equal(target[0]['keypoints'][:, :4], ((batch['gt_keypoints'][0] + 1) / 2).flatten(1))
    model = paired_config().build(max_epochs=80).model
    noisy, matches = build_proposal_paired_pose_inputs(
        torch.zeros(2, 2, 2), torch.full((2, 2, 4), .2), torch.tensor([0, 999]), target,
        diffusion=model, matcher=DiffusionRegPoseMatcher(num_body_points=2), noise_seed=7,
    )
    assert noisy.shape == (2, 2, 4) and len(matches[1][0]) == 0
    with pytest.raises(ValueError, match='capacity'):
        build_proposal_paired_pose_inputs(
            torch.zeros(1, 1, 2), torch.zeros(1, 1, 4), torch.tensor([0]), targets(),
            diffusion=model, matcher=DiffusionRegPoseMatcher(num_body_points=2),
        )


@pytest.mark.parametrize('empty', [False, True])
@pytest.mark.parametrize('training', [False, True])
def test_actual_model_time_corruption_pairing_and_loss_routes(empty, training):
    pipeline = paired_config().build(max_epochs=80)
    pipeline.train(training)
    batch = repeated_batch(3)
    batch['image'] = torch.randn(3, 3, 64, 64)
    batch['gt_diffusion_keypoints'] = torch.full((3, 4, 2, 2), 50.)
    if empty:
        batch['gt_labeled'].zero_()
    observed = []
    hook = pipeline.model.register_forward_pre_hook(lambda module, args: observed.append(args[2].clone()))
    with patch.object(pipeline.model, 'q_sample', wraps=pipeline.model.q_sample) as corrupt:
        with patch.object(pipeline.criterion, 'forward', wraps=pipeline.criterion.forward) as criterion:
            losses = pipeline._losses_for_batch(batch)
    hook.remove()
    clean, times = corrupt.call_args.args
    assert torch.equal(observed[0], times) and (times > 0).any() and times.unique().numel() > 1
    outputs, target = criterion.call_args.args
    matches = outputs['pose_target_matches']
    assert criterion.call_args.kwargs['final_matches'] is matches
    assert criterion.call_args.kwargs['auxiliary_matches'][0] is None
    assert criterion.call_args.kwargs['auxiliary_matches'][1] is matches
    for index, (queries, people) in enumerate(matches):
        expected = (target[index]['keypoints'][people, :4] * 2 - 1) * pipeline.model.scale
        assert torch.equal(clean[index, queries], expected)
    assert losses['paired_proposal/target_coverage'] == (0 if empty else 1)
    assert losses['paired_proposal/matched_count'] == (0 if empty else 3)
    assert torch.isfinite(losses['total'])
    assert ('dn_loss_bbox' in losses) == training
    losses['total'].backward()
    if not empty:
        for module in (pipeline.model.pose_encoder, pipeline.model.pose_readers[-1], pipeline.model.person_attention[-1]):
            assert any(value.grad is not None and value.grad.abs().sum() > 0 for value in module.parameters())


def test_config_initial_weights_default_forward_rng_and_inference_are_unchanged():
    settings = paired_config()
    torch.manual_seed(42)
    paired = settings.build(max_epochs=80).eval()
    torch.manual_seed(42)
    control = config(pose_readout_mode='box_center_residual', training_timestep_policy='uniform').build(max_epochs=80).eval()
    assert list(paired.state_dict()) == list(control.state_dict())
    assert all(torch.equal(value, control.state_dict()[name]) for name, value in paired.state_dict().items())
    paired.load_state_dict(control.state_dict(), strict=True)
    image, state, times = torch.randn(1, 3, 64, 64), torch.randn(1, 2, 4), torch.tensor([333])
    rng = torch.get_rng_state().clone()
    with torch.no_grad():
        actual = paired.model(image, state, times)
        assert torch.equal(rng, torch.get_rng_state())
        expected = control.model(image, state, times)
    assert 'pose_target_matches' not in actual
    assert all(torch.equal(actual[name], expected[name]) for name in ('pred_logits', 'pred_boxes', 'pred_keypoints'))
    assert paired.hparams['target_assignment'] == 'proposal_hungarian_fixed'
    with initialize_config_dir(version_base=None, config_dir=str(Path(__file__).resolve().parents[1] / 'conf')):
        preset = compose(config_name='benchmark_poseflow_v42_paired_proposal')
    assert preset.training.max_epochs == 80 and preset.pipeline.quick_eval_steps == [1, 4, 8]
    assert preset.pipeline.training_timestep_policy == 'uniform'
    assert preset.pipeline.target_assignment == 'proposal_hungarian_fixed'
    assert preset.pipeline.pose_readout_mode == 'box_center_residual'

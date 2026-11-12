from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from src.models.pipelines.pose_estimators.diffusionregpose.training import DiffusionRegPoseMatcher
from src.models.pipelines.pose_estimators.poseflow.sampling_v42 import (
    match_proposals_to_noisy_poses, sample_v42_with_proposal_pose_matching,
)
from tests.test_poseflow_v42 import config


def inputs():
    boxes = torch.tensor([[[.2, .3, .2, .2], [.8, .7, .2, .2]],
                          [[.8, .7, .2, .2], [.2, .3, .2, .2]]])
    xy = torch.tensor([[[.1, .2, .3, .4], [.7, .6, .9, .8]]]).repeat(2, 1, 1)
    return boxes, (xy * 2 - 1) * 6


def test_existing_matcher_assigns_actual_pose_boxes_with_batch_isolation_and_no_rng():
    boxes, state = inputs()
    matcher = DiffusionRegPoseMatcher(num_body_points=2)
    rng = torch.get_rng_state().clone()
    with patch.object(matcher, 'forward', wraps=matcher.forward) as call:
        permutation = match_proposals_to_noisy_poses(boxes, state, scale=6, matcher=matcher)
    assert permutation.tolist() == [[0, 1], [1, 0]]
    assert torch.equal(rng, torch.get_rng_state())
    assert call.call_args.kwargs == {'include_keypoints': False}
    actual_targets = call.call_args.args[1]
    assert torch.allclose(actual_targets[0]['boxes'], boxes[0])
    assert torch.equal(actual_targets[0]['labels'], torch.ones(2, dtype=torch.long))
    ordered = state.gather(1, permutation[..., None].expand_as(state))
    assert torch.equal(ordered[0], state[0]) and torch.equal(ordered[1], state[1].flip(0))
    assert not permutation.requires_grad


def test_class_scores_cannot_change_complete_same_class_state_assignment():
    boxes, state = inputs()
    matcher = DiffusionRegPoseMatcher(num_body_points=2)
    first = match_proposals_to_noisy_poses(boxes, state, scale=6, matcher=matcher)
    other = match_proposals_to_noisy_poses(boxes, state, scale=6, matcher=matcher,
                                         selected_logits=torch.tensor([[[8., -8.], [-4., 4.]]]).repeat(2, 1, 1))
    assert torch.equal(first, other)


def test_degenerate_outside_image_and_tied_states_still_form_complete_finite_assignment():
    matcher = DiffusionRegPoseMatcher(num_body_points=2)
    boxes = torch.full((2, 3, 4), .2)
    state = torch.full((2, 3, 4), 20.)
    result = match_proposals_to_noisy_poses(boxes, state, scale=6, matcher=matcher)
    assert torch.equal(result.sort(1).values, torch.arange(3)[None].expand(2, -1))
    assert torch.equal(result, match_proposals_to_noisy_poses(boxes, state, scale=6, matcher=matcher))


@pytest.mark.parametrize('kind', ['nonfinite', 'count', 'points', 'scale'])
def test_invalid_matching_input_is_rejected(kind):
    boxes, state = inputs()
    if kind == 'nonfinite': state[0, 0, 0] = float('nan')
    if kind == 'count': state = state[:, :1]
    if kind == 'points': state = state[..., :2]
    with pytest.raises(ValueError):
        match_proposals_to_noisy_poses(boxes, state, scale=0 if kind == 'scale' else 6,
                                      matcher=DiffusionRegPoseMatcher(num_body_points=2))


class OrderedStateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_proposals = 2
        self.transformer = SimpleNamespace(decoder=SimpleNamespace(num_body_points=2))
        self.scale, self.ddim_sampling_eta = 6., 0.
        self.inputs, self.residual_inputs, self.steps = [], [], []

    def ddim_timestep_pairs(self):
        return [(999, 499), (499, -1)]

    def forward(self, images, state, times, *, padding_mask=None, pose_input_builder):
        # Force a reversed order using boxes from the exact incoming coordinates.
        xy = (state / self.scale + 1) / 2
        points = xy.reshape(1, 2, 2, 2)
        lower, upper = points.amin(2), points.amax(2)
        boxes = torch.cat(((lower + upper) / 2, (upper - lower).clamp_min(1e-6)), -1).flip(1)
        used, matches = pose_input_builder(torch.zeros(1, 2, 2), boxes, times, None)
        assert matches is None
        self.inputs.append(used.clone())
        return dict(pred_keypoints=torch.cat(((used / self.scale + 1) / 2, torch.ones(1, 2, 2)), -1),
                    pred_logits=torch.zeros(1, 2, 2), pred_boxes=boxes)

    def predict_noise_from_start(self, state, times, clean):
        self.residual_inputs.append(state.clone())
        assert torch.allclose(state, clean, atol=1e-6)
        return torch.zeros_like(state)

    def ddim_step(self, clean, noise, time, next_time, step_noise):
        self.steps.append((time, next_time))
        return clean


def test_every_step_permutation_reaches_both_denoiser_and_ddim_residual():
    model = OrderedStateModel().eval()
    observed = []
    generator = torch.Generator().manual_seed(91)
    initial = torch.randn(1, 2, 4, generator=torch.Generator().manual_seed(91))
    result = sample_v42_with_proposal_pose_matching(model, torch.zeros(1, 3, 4, 4),
        matcher=DiffusionRegPoseMatcher(num_body_points=2), generator=generator,
        assignment_observer=lambda time, permutation: observed.append((time, permutation.clone())))
    assert torch.equal(model.inputs[0], initial.flip(1))
    assert torch.allclose(model.inputs[1], initial, atol=1e-6)
    assert all(torch.equal(a, b) for a, b in zip(model.inputs, model.residual_inputs))
    assert [time for time, _ in observed] == [999, 499]
    assert all(permutation.tolist() == [[1, 0]] for _, permutation in observed)
    assert torch.allclose(result['diffusion_keypoints'], initial, atol=1e-6)
    assert model.steps == [(999, 499), (499, -1)]


def test_disabled_sampler_is_native_bitwise_with_identical_private_and_global_rng():
    model = config(pose_readout_mode='box_center_residual').build(max_epochs=80).model.eval()
    model.sampling_timesteps = 4
    image = torch.randn(1, 3, 64, 64)
    rng = torch.get_rng_state().clone()
    left_gen, right_gen = torch.Generator().manual_seed(42), torch.Generator().manual_seed(42)
    with torch.no_grad():
        expected = model.sample_keypoints(image, generator=left_gen)
        expected_rng = torch.get_rng_state().clone()
        torch.set_rng_state(rng)
        actual = sample_v42_with_proposal_pose_matching(model, image,
            matcher=DiffusionRegPoseMatcher(num_body_points=2), generator=right_gen, enabled=False)
    assert torch.equal(expected_rng, torch.get_rng_state())
    assert torch.equal(left_gen.get_state(), right_gen.get_state())
    assert actual.keys() == expected.keys()
    assert all(torch.equal(actual[key], expected[key]) for key in ('pred_logits', 'pred_boxes', 'pred_keypoints', 'diffusion_keypoints'))


def test_actual_v42_uses_one_forward_per_step_and_preserves_model_state_and_rng():
    model = config(pose_readout_mode='box_center_residual').build(max_epochs=80).model.eval()
    model.sampling_timesteps = 4
    before = {key:value.clone() for key,value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    times, permutations = [], []
    hook = model.register_forward_pre_hook(lambda module, args: times.append(int(args[2][0])))
    result = sample_v42_with_proposal_pose_matching(model, torch.randn(1, 3, 64, 64, generator=torch.Generator().manual_seed(8)),
        matcher=DiffusionRegPoseMatcher(num_body_points=2), generator=torch.Generator().manual_seed(42),
        assignment_observer=lambda time, permutation: permutations.append(permutation))
    hook.remove()
    assert times == [999, 749, 499, 249] and len(permutations) == 4
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(before[key], value) for key,value in model.state_dict().items())
    assert all(torch.isfinite(result[key]).all() for key in ('pred_logits', 'pred_boxes', 'pred_keypoints', 'diffusion_keypoints'))

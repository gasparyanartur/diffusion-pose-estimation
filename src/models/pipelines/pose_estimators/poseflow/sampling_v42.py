from collections.abc import Callable

import torch
from torch import Tensor

from ..diffusionregpose.training import CROWDPOSE_PERSON_LABEL, DiffusionRegPoseMatcher


@torch.no_grad()
def match_proposals_to_noisy_poses(
    reference_boxes: Tensor,
    state: Tensor,
    *,
    scale: float,
    matcher: DiffusionRegPoseMatcher,
    selected_logits: Tensor | None = None,
) -> Tensor:
    """Return incoming-state indices in current proposal order, without GT."""
    if scale <= 0 or reference_boxes.ndim != 3 or reference_boxes.shape[-1] != 4:
        raise ValueError('Matching requires positive scale and reference boxes [B, N, 4]')
    batch_size, count, _ = reference_boxes.shape
    expected = (batch_size, count, 2 * matcher.num_body_points)
    if batch_size < 1 or count < 1 or tuple(state.shape) != expected:
        raise ValueError(f'Incoming pose state must have nonempty shape {expected}')
    if state.device != reference_boxes.device:
        raise ValueError('Incoming poses and references must be on the same device')
    if not torch.isfinite(state).all() or not torch.isfinite(reference_boxes).all():
        raise ValueError('Incoming poses and reference boxes must be finite')
    if not state.is_floating_point() or not reference_boxes.is_floating_point():
        raise ValueError('Incoming poses and reference boxes must be floating point')
    if selected_logits is None:
        selected_logits = reference_boxes.new_zeros(batch_size, count, CROWDPOSE_PERSON_LABEL + 1)
    if (selected_logits.ndim != 3 or selected_logits.shape[:2] != (batch_size, count)
            or selected_logits.shape[-1] <= CROWDPOSE_PERSON_LABEL
            or selected_logits.device != state.device or not torch.isfinite(selected_logits).all()):
        raise ValueError('Selected class logits must be finite [B, N, C] on the state device')

    # B = images, N = person slots, K = joints. Coordinates may lie outside the image.
    points = ((state / scale + 1) / 2).reshape(batch_size, count, matcher.num_body_points, 2)
    lower, upper = points.amin(2), points.amax(2)
    boxes = torch.cat(((lower + upper) / 2, (upper - lower).clamp_min(1e-6)), -1)
    labels = torch.full((count,), CROWDPOSE_PERSON_LABEL, device=state.device, dtype=torch.long)
    # All target-side labels agree, so class costs cancel in a complete permutation.
    targets = [dict(labels=labels, boxes=box) for box in boxes]
    matches = matcher(dict(pred_logits=selected_logits, pred_boxes=reference_boxes),
                      targets, include_keypoints=False)
    permutation = torch.empty(batch_size, count, device=state.device, dtype=torch.long)
    expected_indices = torch.arange(count, device=state.device)
    for image, (queries, states) in enumerate(matches):
        if (not torch.equal(queries.sort().values, expected_indices)
                or not torch.equal(states.sort().values, expected_indices)):
            raise ValueError('Proposal/state matching must produce a complete one-to-one permutation')
        permutation[image, queries] = states
    return permutation


@torch.no_grad()
def sample_v42_with_proposal_pose_matching(
    model,
    images: Tensor,
    *,
    matcher: DiffusionRegPoseMatcher,
    padding_mask: Tensor | None = None,
    generator: torch.Generator | None = None,
    enabled: bool = True,
    assignment_observer: Callable[[int, Tensor], None] | None = None,
) -> dict:
    if not enabled:
        return model.sample_keypoints(images, padding_mask=padding_mask, generator=generator)
    if model.training:
        raise ValueError('Proposal/pose matching sampling requires model.eval()')
    if matcher.num_body_points != model.transformer.decoder.num_body_points:
        raise ValueError('Matcher and model must use the same number of joints')
    state = torch.randn(images.shape[0], model.num_proposals,
        model.transformer.decoder.num_body_points * 2, device=images.device, generator=generator)
    output = None
    for time, next_time in model.ddim_timestep_pairs():
        timestep = torch.full((images.shape[0],), time, dtype=torch.long, device=images.device)
        captured = {}

        def build_pose_inputs(logits, references, times, targets):
            permutation = match_proposals_to_noisy_poses(references, state,
                scale=model.scale, matcher=matcher, selected_logits=logits)
            ordered = state.gather(1, permutation[..., None].expand_as(state))
            captured['state'] = ordered
            if assignment_observer is not None:
                assignment_observer(time, permutation)
            return ordered, None

        output = model(images, state, timestep, padding_mask=padding_mask,
                       pose_input_builder=build_pose_inputs)
        if 'state' not in captured:
            raise RuntimeError('V42 model did not invoke its pose-input selection hook')
        # Predictions are in proposal order; the DDIM residual must use that same order.
        state = captured['state']
        coordinates = output['pred_keypoints'][..., :state.shape[-1]]
        clean = ((coordinates * 2 - 1) * model.scale).clamp(-model.scale, model.scale)
        noise = model.predict_noise_from_start(state, timestep, clean).to(state.dtype)
        step_noise = torch.zeros_like(state) if model.ddim_sampling_eta == 0 else None
        state = model.ddim_step(clean, noise, time, next_time, step_noise)
    if output is None:
        raise RuntimeError('Diffusion schedule produced no timestep pairs')
    output['diffusion_keypoints'] = state
    return output

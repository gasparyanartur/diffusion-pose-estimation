from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from hydra import compose, initialize_config_dir

from src.models.pipelines.pose_estimators.poseflow import training_v42_noisy_proposal as variant
from src.models.pipelines.pose_estimators.diffusionregpose import training as native_training
from src.models.pipelines.pose_estimators.poseflow.quick_eval_v42 import evaluate_v42_grid
from tests.test_poseflow_v42_paired_proposal import paired_config
from tests.test_poseflow_v42_uniform_t import repeated_batch


def noisy_config():
    values = paired_config().model_dump(exclude={'target_assignment'})
    return variant.PoseFlowV42NoisyProposalPipelineConfig(**values)


def test_foreground_background_swap_moves_source_labels_without_changing_states():
    state = torch.arange(24.).reshape(2, 3, 4)
    source_matches = [(torch.tensor([0, 2]), torch.tensor([1, 0])),
                      (torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long))]
    permutation = torch.tensor([[2, 0, 1], [1, 2, 0]])
    matcher = object()
    model = noisy_config().build(max_epochs=80).model
    boxes, logits, times = torch.zeros(2, 3, 4), torch.zeros(2, 3, 2), torch.tensor([0, 999])
    with patch.object(variant, 'build_proposal_paired_pose_inputs', return_value=(state, source_matches)):
        with patch.object(variant, 'match_proposals_to_noisy_poses', return_value=permutation) as associate:
            ordered, matches = variant.build_noisy_proposal_pose_inputs(
                logits, boxes, times, [{}, {}], diffusion=model, matcher=matcher)
    assert torch.equal(ordered, state.gather(1, permutation[..., None].expand_as(state)))
    assert matches[0][0].tolist() == [1, 0] and matches[0][1].tolist() == [1, 0]
    assert matches[1][0].numel() == 0
    assert associate.call_args.args[1] is state
    assert associate.call_args.kwargs['matcher'] is matcher
    assert set(associate.call_args.kwargs) == {'scale', 'matcher', 'selected_logits'}


@pytest.mark.parametrize('training', [False, True])
@pytest.mark.parametrize('empty', [False, True])
def test_real_corruption_and_fixed_losses_follow_actual_permutation(training, empty):
    pipeline = noisy_config().build(max_epochs=80).train(training)
    batch = repeated_batch(3)
    batch['image'] = torch.randn(3, 3, 64, 64)
    if empty:
        batch['gt_labeled'].zero_()
    corruptions, assignments, encoded, forward_times = [], [], [], []
    original_q = pipeline.model.q_sample
    original_match = variant.match_proposals_to_noisy_poses

    def corrupt(clean, times, noise):
        result = original_q(clean, times, noise)
        corruptions.append((clean.clone(), times.clone(), result.to(clean).clamp(-6, 6)))
        return result

    def associate(boxes, state, **kwargs):
        result = original_match(boxes, state, **kwargs)
        assignments.append((state.clone(), result.clone()))
        return result

    hooks = [pipeline.model.register_forward_pre_hook(lambda module, args: forward_times.append(args[2])),
             pipeline.model.pose_encoder.register_forward_pre_hook(lambda module, args: encoded.append(args[0]))]
    with patch.object(pipeline.model, 'q_sample', side_effect=corrupt):
        with patch.object(variant, 'match_proposals_to_noisy_poses', side_effect=associate):
            with patch.object(pipeline.criterion, 'forward', wraps=pipeline.criterion.forward) as criterion:
                losses = pipeline._losses_for_batch(batch)
    for hook in hooks:
        hook.remove()
    clean, times, noisy = corruptions[0]
    state, permutation = assignments[0]
    assert torch.equal(times, forward_times[0]) and times.unique().numel() > 1 and (times > 0).any()
    assert torch.equal(state, noisy)
    expected = noisy.gather(1, permutation[..., None].expand_as(noisy))
    assert torch.equal(encoded[0], (expected / 6 + 1) / 2)
    outputs, targets = criterion.call_args.args
    matches = outputs['pose_target_matches']
    assert criterion.call_args.kwargs['final_matches'] is matches
    assert criterion.call_args.kwargs['auxiliary_matches'] == [None, matches]
    for image, (queries, people) in enumerate(matches):
        source = clean[image, permutation[image, queries]]
        target = (targets[image]['keypoints'][people, :4] * 2 - 1) * 6
        assert torch.equal(source, target)
    assert losses['paired_proposal/target_coverage'] == (0 if empty else 1)
    assert torch.isfinite(losses['total'])
    assert ('dn_loss_bbox' in losses) == training
    losses['total'].backward()
    if not empty:
        assert pipeline.model.pose_encoder.layers[0].weight.grad.abs().sum() > 0


def test_new_config_weights_restore_and_matched_evaluation(tmp_path):
    torch.manual_seed(42)
    pipeline = noisy_config().build(max_epochs=80).train()
    torch.manual_seed(42)
    control = paired_config().build(max_epochs=80)
    assert all(torch.equal(value, control.state_dict()[name]) for name, value in pipeline.state_dict().items())
    pipeline.load_state_dict(control.state_dict(), strict=True)
    assert pipeline.hparams['target_assignment'] == 'noisy_proposal_source_fixed'
    assert pipeline.hparams['inference_association'] == 'proposal_noisy_pose_hungarian'
    batch = dict(image=torch.randn(1, 3, 64, 64), image_id=torch.tensor([7]), img_hw=torch.tensor([[64, 64]]))
    rng = torch.get_rng_state().clone()
    with patch.object(variant, 'sample_v42_with_proposal_pose_matching',
                      wraps=variant.sample_v42_with_proposal_pose_matching) as sample:
        report = evaluate_v42_grid(pipeline.model, [batch], pipeline.v42_config, tmp_path, 'unused', [7],
            metric_evaluator=lambda *args: {'pose_AP': .1, 'bbox_AP': .2},
            **pipeline._evaluation_sampler_kwargs())
    assert len(sample.call_args_list) == 3
    assert all(call.kwargs['matcher'] is pipeline.criterion.matcher for call in sample.call_args_list)
    assert 'proposal/noisy-pose Hungarian' in report['sampler']
    assert pipeline.model.training and torch.equal(rng, torch.get_rng_state())
    assert control._evaluation_sampler_kwargs() == {}
    with initialize_config_dir(version_base=None, config_dir=str(Path(__file__).resolve().parents[1] / 'conf')):
        preset = compose(config_name='benchmark_poseflow_v42_noisy_proposal')
    assert preset.training.max_epochs == 80 and preset.pipeline.quick_eval_steps == [1, 4, 8]
    assert preset.pipeline.training_timestep_policy == 'uniform'
    assert preset.pipeline.target_assignment == 'noisy_proposal_source_fixed'
    assert preset.pipeline.inference_association == 'proposal_noisy_pose_hungarian'


def test_periodic_validation_ap_uses_matching_with_legacy_native_default():
    pipeline = noisy_config().build(max_epochs=80).eval()
    control = paired_config().build(max_epochs=80).eval()
    image = torch.randn(1, 3, 64, 64)
    options = dict(padding_mask=None, generator=torch.Generator().manual_seed(42))
    with patch.object(variant, 'sample_v42_with_proposal_pose_matching', return_value={'matched': True}) as sample:
        output = pipeline._sample_validation_outputs(image, **options)
    assert output == {'matched': True}
    assert sample.call_args.args == (pipeline.model, image)
    assert sample.call_args.kwargs['matcher'] is pipeline.criterion.matcher
    with patch.object(control.model, 'sample_keypoints', return_value={'native': True}) as native:
        output = control._sample_validation_outputs(image, **options)
    assert output == {'native': True} and native.call_args.kwargs == options
    pipeline._val_bbox_ap = object()
    with patch.object(pipeline, '_losses_for_batch', return_value={'total': torch.tensor(0.)}), \
         patch.object(pipeline, '_log_losses'), patch.object(pipeline, '_ap_epoch_active', return_value=True), \
         patch.object(native_training, '_postprocess_diffusionregpose_ap', return_value=[]), \
         patch.object(native_training, 'update_diffusionregpose_ap_metric'), \
         patch.object(native_training, 'update_diffusionregpose_bbox_ap_metric'), \
         patch.object(variant, 'sample_v42_with_proposal_pose_matching', return_value={}) as sample:
        pipeline.validation_step({'image': image}, 0)
    assert sample.call_count == 1 and sample.call_args.args[0] is pipeline.model

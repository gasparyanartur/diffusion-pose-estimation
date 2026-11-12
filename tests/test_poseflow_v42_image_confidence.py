from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from hydra import compose, initialize_config_dir

from src.models.pipelines.pose_estimators.poseflow.training_v42_image_confidence import (
    PoseFlowV42ImageConfidencePipelineConfig,
)
from tests.test_poseflow_v42_paired_proposal import paired_config
from tests.test_poseflow_v42_uniform_t import repeated_batch


def image_config():
    return PoseFlowV42ImageConfidencePipelineConfig(**paired_config().model_dump(exclude={'confidence_mode'}))


def test_initial_shared_tensors_rng_and_separate_head_are_exact():
    torch.manual_seed(42)
    control = paired_config().build(max_epochs=80)
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(42)
    variant = image_config().build(max_epochs=80)
    assert torch.equal(expected_rng, torch.get_rng_state())
    expected = control.state_dict()
    actual = variant.state_dict()
    assert set(actual) - set(expected) == {'model.image_confidence_head.weight', 'model.image_confidence_head.bias'}
    assert all(torch.equal(value, actual[name]) for name, value in expected.items())
    original = variant.model.class_embed[variant.model.transformer.decoder.num_box_decoder_layers - 1]
    head = variant.model.image_confidence_head
    for name, value in head.state_dict().items():
        assert torch.equal(value, original.state_dict()[name])
    assert head.weight.data_ptr() != original.weight.data_ptr()
    assert not hasattr(control.model, 'image_confidence_head')
    assert variant.hparams['confidence_mode'] == 'image_only'


def test_scores_use_actual_selected_normalized_image_features_and_ignore_noisy_pose():
    model = image_config().build(max_epochs=80).model.eval()
    image = torch.randn(2, 3, 64, 64)
    times = torch.tensor([0, 999])
    records, inputs, selected = [], [], []
    head = model.class_embed[model.transformer.decoder.num_box_decoder_layers - 1]
    hooks = [head.register_forward_hook(lambda module, args, output: records.append(output)),
             model.image_confidence_head.register_forward_pre_hook(lambda module, args: inputs.append(args[0]))]

    def observe(logits, boxes, actual_times, targets):
        selected.append(logits)
        assert len(inputs) == 1
        return first_state, None

    first_state = torch.randn(2, 2, 4)
    with torch.no_grad():
        first = model(image, first_state, times, pose_input_builder=observe)
        second = model(image, torch.randn_like(first_state) * 6, times)
    for hook in hooks:
        hook.remove()
    raw = records[1]
    indices = raw.max(-1).values.topk(2, dim=0).indices.transpose(0, 1)
    assert torch.equal(raw.transpose(0, 1).gather(1, indices[..., None].expand(-1, -1, 2)), selected[0])
    expected = records[0].gather(1, indices[..., None].expand(-1, -1, 2))
    assert torch.allclose(first['pred_logits'], expected, atol=1e-6)
    assert torch.equal(first['pred_logits'], second['pred_logits'])
    assert not torch.equal(first['pred_keypoints'], second['pred_keypoints'])
    assert first['aux_outputs'][1]['pred_logits'] is first['pred_logits']


def test_initial_geometry_sampling_and_strict_restore(tmp_path):
    torch.manual_seed(42)
    control = paired_config().build(max_epochs=80).model.eval()
    torch.manual_seed(42)
    pipeline = image_config().build(max_epochs=80).eval()
    model = pipeline.model
    image = torch.randn(1, 3, 64, 64)
    for steps in (1, 4, 8):
        control.sampling_timesteps = model.sampling_timesteps = steps
        with torch.no_grad():
            expected = control.sample_keypoints(image, generator=torch.Generator().manual_seed(42))
            actual = model.sample_keypoints(image, generator=torch.Generator().manual_seed(42))
        for key in ('pred_keypoints', 'pred_boxes', 'diffusion_keypoints'):
            assert torch.equal(expected[key], actual[key])
    path = tmp_path / 'state.pt'
    torch.save(pipeline.state_dict(), path)
    restored = image_config().build(max_epochs=80)
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    assert all(torch.equal(value, restored.state_dict()[name]) for name, value in pipeline.state_dict().items())
    with pytest.raises(RuntimeError):
        control.load_state_dict(model.state_dict(), strict=True)


@pytest.mark.parametrize('training,empty', [(True, False), (False, False), (True, True)])
def test_fixed_labels_time_and_classification_geometry_gradient_routes(training, empty):
    pipeline = image_config().build(max_epochs=80).train(training)
    batch = repeated_batch(3)
    batch['image'] = torch.randn(3, 3, 64, 64)
    if empty:
        batch['gt_labeled'].zero_()
    with patch.object(pipeline.criterion, 'forward', wraps=pipeline.criterion.forward) as criterion:
        with patch.object(pipeline.model, 'q_sample', wraps=pipeline.model.q_sample) as corrupt:
            losses = pipeline._losses_for_batch(batch)
    outputs, targets = criterion.call_args.args
    matches = outputs['pose_target_matches']
    assert criterion.call_args.kwargs['final_matches'] is matches
    assert criterion.call_args.kwargs['auxiliary_matches'] == [None, matches]
    clean, times = corrupt.call_args.args
    assert (times > 0).any() and times.unique().numel() > 1
    for image, (queries, people) in enumerate(matches):
        assert torch.equal(clean[image, queries], (targets[image]['keypoints'][people, :4] * 2 - 1) * 6)
    assert torch.isfinite(losses['total'])
    assert ('dn_loss_bbox' in losses) == training
    losses['loss_ce'].backward(retain_graph=True)
    assert pipeline.model.image_confidence_head.weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in pipeline.model.backbone.parameters())
    for module in (pipeline.model.pose_encoder, pipeline.model.pose_readers, pipeline.model.person_attention):
        assert all(p.grad is None or p.grad.count_nonzero() == 0 for p in module.parameters())
    pipeline.zero_grad(set_to_none=True)
    losses['total'].backward()
    if not empty:
        for module in (pipeline.model.pose_encoder, pipeline.model.pose_readers, pipeline.model.person_attention):
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())


def test_preset_and_configuration_dispatch():
    from src.models.pipelines.pose_estimators.diffusionregpose import DiffusionRegPosePipelineConfig
    with initialize_config_dir(version_base=None, config_dir=str(Path(__file__).resolve().parents[1] / 'conf')):
        preset = compose(config_name='benchmark_poseflow_v42_image_confidence')
    settings = DiffusionRegPosePipelineConfig.from_hydra_config(preset.pipeline)
    assert isinstance(settings, PoseFlowV42ImageConfidencePipelineConfig)
    assert preset.training.max_epochs == 80 and preset.pipeline.quick_eval_steps == [1, 4, 8]
    assert settings.confidence_mode == 'image_only' and settings.training_timestep_policy == 'uniform'
    assert settings.target_assignment == 'proposal_hungarian_fixed'
    with pytest.raises(ValueError):
        image_config().model_copy().model_validate({**image_config().model_dump(), 'confidence_mode': 'pose_conditioned'})

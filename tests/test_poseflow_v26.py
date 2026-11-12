from copy import deepcopy

import pytest
import torch

from src.models.pipelines.pose_estimators.poseflow.latent_v26 import (
    build_v26_targets, v26_task_loss,
)


def batch_fixture(counts=(2, 1), joints=14):
    count = max(max(counts), 1)
    points = torch.full((len(counts), count, joints, 2), -0.5)
    labeled = torch.zeros(len(counts), count, joints, dtype=torch.bool)
    boxes = torch.zeros(len(counts), count, 4)
    for image, people in enumerate(counts):
        for person in range(people):
            points[image, person] += person * .25 + image * .1
            labeled[image, person] = True
            boxes[image, person] = torch.tensor([.3 + person*.1, .4, .2, .3])
    return dict(gt_keypoints=points, gt_labeled=labeled, gt_boxes_canon=boxes,
                person_mask=labeled.any(-1), img_hw=torch.tensor([[100., 200.]]*len(counts)),
                image_id=torch.arange(len(counts)), image=torch.randn(len(counts), 3, 64, 64))


def test_proposals_cover_gt_preserve_indices_and_transfer_donors():
    batch = batch_fixture()
    batch['img_hw'][1] = torch.tensor([200., 400.])
    target = build_v26_targets(batch, proposals=7, generator=torch.Generator().manual_seed(5))
    assert target['input_pixels'].shape == (2, 7, 28)
    assert target['positive'].sum(-1).tolist() == [4, 4]
    assert target['target_indices'][0, :2].tolist() == [0, 1]
    assert target['target_indices'][1, 0].item() == 0
    assert (target['target_indices'][~target['positive']] == -1).all()
    for recipient in range(2):
        negative = (~target['positive'][recipient]).nonzero().flatten()
        donor = 1-recipient
        assert (target['source_batch'][recipient, negative, 0] == donor).all()
        for slot in negative:
            person = target['source_person'][recipient, slot, 0]
            expected = (batch['gt_keypoints'][donor, person]+1)/2
            expected = expected * batch['img_hw'][recipient].flip(0)
            torch.testing.assert_close(target['input_pixels'][recipient, slot], expected.flatten())


def test_fallback_empty_recipient_and_overflow():
    batch = batch_fixture((1,))
    target = build_v26_targets(batch, proposals=4, generator=torch.Generator().manual_seed(9), fallback_noise_pixels=0)
    assert target['positive'].sum() == 2
    assert target['synthetic'].sum() == 2
    torch.testing.assert_close(target['input_pixels'][0, 0], target['input_pixels'][0, 2])
    empty = build_v26_targets(batch_fixture((0, 1)), proposals=5)
    assert not empty['positive'][0].any()
    assert empty['positive'][1].sum() == 3
    with pytest.raises(ValueError, match='budget'):
        build_v26_targets(batch_fixture((3,)), proposals=2)
    with pytest.raises(ValueError, match='no poses'):
        build_v26_targets(batch_fixture((0,)), proposals=4)


def test_original_pose_fallback_and_unlabeled_center():
    batch = batch_fixture((1,))
    batch['v26_original_points'] = batch['gt_keypoints'].clone()
    batch['v26_original_labeled'] = batch['gt_labeled'].clone()
    batch['v26_original_people'] = batch['person_mask'].clone()
    batch['person_mask'].zero_(); batch['gt_labeled'].zero_()
    target = build_v26_targets(batch, proposals=4, fallback_noise_pixels=0)
    assert target['synthetic'].all() and not target['positive'].any()
    batch = batch_fixture((1,))
    batch['gt_labeled'][0, 0, 0] = False
    target = build_v26_targets(batch, proposals=2)
    torch.testing.assert_close(target['input_pixels'][0, 0, :2], torch.tensor([100., 50.]))


def test_direct_loss_masks_negative_geometry_and_missing_joints():
    target = build_v26_targets(batch_fixture(), proposals=6)
    target['labeled'][:, :, 0] = False
    output = dict(existence_logits=torch.zeros(2, 6, requires_grad=True),
                  pred_boxes=torch.full((2, 6, 4), .4, requires_grad=True),
                  joint_xy=torch.full((2, 6, 14, 2), .6, requires_grad=True))
    loss = v26_task_loss(output, target)
    assert torch.isfinite(loss['total'])
    loss['total'].backward()
    assert output['existence_logits'].grad[~target['positive']].abs().sum() > 0
    assert output['pred_boxes'].grad[~target['positive']].abs().sum() == 0
    assert output['joint_xy'].grad[~target['positive']].abs().sum() == 0
    assert output['joint_xy'].grad[:, :, 0].abs().sum() == 0


def tiny_model():
    from src.models.pipelines.pose_estimators.poseflow.model_v26 import PoseFlowV26Model
    return PoseFlowV26Model(backbone_pretrained=False, hidden_dim=32, num_heads=4,
                           feedforward_dim=64, num_proposals=5, num_feature_levels=2)


def test_latent_schedule_detach_and_both_branch_gradients():
    model = tiny_model().train()
    images = torch.randn(2, 3, 64, 64)
    pixels = torch.rand(2, 5, 28)*60
    clean, recovered, noisy, times = model.training_outputs(images, pixels,
        timesteps=torch.tensor([0, 999]), noise=torch.ones(2, 5, 32))
    assert not noisy.requires_grad
    expected = model.q_sample(clean['pred_latents'].detach(), times, torch.ones_like(noisy))
    torch.testing.assert_close(noisy, expected)
    latent = (recovered['pred_latents']-clean['pred_latents'].detach()).square().mean()
    latent.backward(retain_graph=True)
    assert all(p.grad is None for p in model.clean_layers.parameters())
    assert model.input_mlp[0].weight.grad is None
    assert model.denoiser_projection.weight.grad.abs().sum() > 0
    task = clean['joint_xy'].mean()+clean['existence_logits'].mean()+recovered['joint_xy'].mean()
    task.backward()
    assert model.input_mlp[0].weight.grad.abs().sum() > 0
    for layers in (model.clean_layers, model.denoiser_layers):
        for layer in layers:
            assert layer.image_attention.in_proj_weight.grad.abs().sum() > 0
    assert torch.isfinite(latent)


@pytest.mark.parametrize('batch_size', [2, 4])
def test_latent_permutation_image_isolation_and_padding(batch_size):
    model = tiny_model().eval()
    state = torch.randn(batch_size, 5, 32)
    memory = torch.randn(batch_size, 7, 32)
    positions = torch.randn_like(memory)
    mask = torch.zeros(batch_size, 7, dtype=torch.bool); mask[:, -2:] = True
    times = torch.arange(batch_size)*200
    with torch.no_grad():
        full = model.denoise(state, times, (memory, positions, mask))
        separate = [model.denoise(state[i:i+1], times[i:i+1],
                    (memory[i:i+1], positions[i:i+1], mask[i:i+1])) for i in range(batch_size)]
        permutation = torch.tensor([3, 0, 4, 1, 2])
        permuted = model.denoise(state[:, permutation], times, (memory, positions, mask))
        changed = memory.clone(); changed[:, -2:] += 1000
        padded = model.denoise(state, times, (changed, positions, mask))
        changed[1:] += 100
        isolated = model.denoise(state, times, (changed, positions, mask))
    for key in ('pred_boxes', 'pred_logits', 'pred_keypoints', 'pred_latents'):
        torch.testing.assert_close(full[key], torch.cat([row[key] for row in separate]), atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(permuted[key], full[key][:, permutation], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(padded[key], full[key], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(isolated[key][0], full[key][0], atol=1e-6, rtol=1e-5)


def test_gaussian_sampling_strict_restore_and_no_clean_branch():
    model = tiny_model().eval()
    other = tiny_model().eval(); other.load_state_dict(model.state_dict(), strict=True)
    def forbidden(*args, **kwargs):
        raise AssertionError('sampling accessed clean encoder')
    model.encode_clean = forbidden
    image = torch.randn(1, 3, 64, 64)
    for steps in (1, 4):
        model.flow_sampling_steps = other.flow_sampling_steps = steps
        with torch.no_grad():
            a = model.sample_keypoints(image, generator=torch.Generator().manual_seed(42))
            b = other.sample_keypoints(image, generator=torch.Generator().manual_seed(42))
        assert a['pred_logits'].shape == (1, 5, 2)
        for key in ('pred_boxes', 'pred_logits', 'pred_keypoints'):
            torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
            assert torch.isfinite(a[key]).all()


@pytest.mark.parametrize('denoised_task_weight', [0., .5])
def test_pipeline_rng_restore_eval_and_no_matcher(tmp_path, denoised_task_weight):
    from src.models.pipelines.pose_estimators.poseflow.training_v26 import PoseFlowV26PipelineConfig
    from scripts.evaluation.evaluate_poseflow import load_poseflow_model
    from omegaconf import OmegaConf
    config = PoseFlowV26PipelineConfig(backbone_pretrained=False, pretrained_checkpoint=None,
        hidden_dim=32, num_heads=4, feedforward_dim=64, num_feature_levels=2,
        num_queries=6, num_proposals=6, num_group=6, quick_eval_enabled=False,
        denoised_task_loss_weight=denoised_task_weight)
    pipeline = config.build(max_epochs=2)
    batch = batch_fixture()
    first = pipeline._losses_for_batch(batch, training=True)
    assert torch.isfinite(first['total'])
    state = deepcopy(pipeline.state_dict())
    restored = config.build(max_epochs=2); restored.load_state_dict(state, strict=True)
    a = pipeline._draw_training_inputs(batch)
    b = restored._draw_training_inputs(batch)
    for left, right in zip(a[1:], b[1:]):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    for key in a[0]:
        torch.testing.assert_close(a[0][key], b[0][key], atol=0, rtol=0)
    rng = pipeline.training_rng_state.clone()
    pipeline._losses_for_batch(batch, training=False)
    assert torch.equal(rng, pipeline.training_rng_state)
    torch.save(dict(state_dict=pipeline.state_dict()), tmp_path/'model.ckpt')
    OmegaConf.save(OmegaConf.create({'pipeline':config.model_dump() | {
        '_target_':'src.models.pipelines.pose_estimators.poseflow.PoseFlowV26PipelineConfig'}}), tmp_path/'config.yaml')
    loaded, _ = load_poseflow_model(tmp_path/'model.ckpt', tmp_path/'config.yaml')
    assert loaded.sampler_kind == 'latent DDIM eta0'
    assert loaded.flow_sampling_steps == 4


@pytest.mark.parametrize('denoised_task_weight', [0., .5])
def test_denoised_task_weight_preserves_objective_and_gradient_routes(denoised_task_weight):
    from src.models.pipelines.pose_estimators.poseflow.training_v26 import PoseFlowV26PipelineConfig

    config = PoseFlowV26PipelineConfig(backbone_pretrained=False, pretrained_checkpoint=None,
        hidden_dim=32, num_heads=4, feedforward_dim=64, num_feature_levels=2,
        num_queries=6, num_proposals=6, num_group=6, quick_eval_enabled=False,
        denoised_task_loss_weight=denoised_task_weight)
    assert config.denoised_task_loss_weight == denoised_task_weight
    pipeline = config.build(max_epochs=2)
    losses = pipeline._losses_for_batch(batch_fixture(), training=False)
    expected = .5*losses['clean/total']+losses['loss_latent']
    if denoised_task_weight:
        expected = expected+denoised_task_weight*losses['denoised/total']
        legacy = (losses['clean/total']+losses['denoised/total'])/2+losses['loss_latent']
        torch.testing.assert_close(losses['total'], legacy, atol=0, rtol=0)
    torch.testing.assert_close(losses['total'], expected, atol=1e-6, rtol=1e-6)
    assert losses['denoised/total'].requires_grad == bool(denoised_task_weight)
    parameters = [parameter for parameter in pipeline.parameters() if parameter.requires_grad]
    actual_gradients = torch.autograd.grad(losses['total'], parameters, retain_graph=True, allow_unused=True)
    expected_gradients = torch.autograd.grad(expected, parameters, retain_graph=True, allow_unused=True)
    for actual, reference in zip(actual_gradients, expected_gradients):
        if reference is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    latent_parameters = [pipeline.model.input_mlp[0].weight,
                         pipeline.model.joint_head.layers[-1].weight,
                         pipeline.model.denoiser_projection.weight]
    source, readout, denoiser = torch.autograd.grad(
        losses['loss_latent'], latent_parameters, retain_graph=True, allow_unused=True)
    assert source is None and readout is None
    assert denoiser is not None and denoiser.abs().sum() > 0
    losses['total'].backward()
    assert pipeline.model.input_mlp[0].weight.grad.abs().sum() > 0
    assert pipeline.model.joint_head.layers[-1].weight.grad.abs().sum() > 0


def test_v26_recipe_and_no_deformable_modules():
    from hydra import compose, initialize_config_dir
    from pathlib import Path
    from src.models.pipelines.pose_estimators.poseflow.training import PoseFlowPipelineConfig
    from src.models.pipelines.pose_estimators.diffusionregpose.transformer import MultiScaleDeformableAttention
    with initialize_config_dir(config_dir=str(Path('conf').resolve()), version_base=None):
        config = compose(config_name='benchmark_poseflow_v26')
    assert config.data.batch_size == 4 and config.training.accumulate_grad_batches == 2
    assert config.training.max_epochs == 80
    typed = PoseFlowPipelineConfig.from_hydra_config(config.pipeline)
    assert typed.matching_strategy == 'constructed'
    assert typed.flow_sampling_steps == 4 and typed.dn_number == 0
    assert typed.denoised_task_loss_weight == .5
    assert not any(isinstance(module, MultiScaleDeformableAttention) for module in tiny_model().modules())


def test_latent_only_recipe_changes_only_denoised_task_supervision():
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from pathlib import Path
    from src.models.pipelines.pose_estimators.poseflow.training import PoseFlowPipelineConfig

    with initialize_config_dir(config_dir=str(Path('conf').resolve()), version_base=None):
        original = compose(config_name='benchmark_poseflow_v26')
        ablation = compose(config_name='benchmark_poseflow_v26_latent_only')
    for section in ('data', 'training'):
        assert OmegaConf.to_container(original[section]) == OmegaConf.to_container(ablation[section])
    original_pipeline = PoseFlowPipelineConfig.from_hydra_config(original.pipeline).model_dump()
    ablation_pipeline = PoseFlowPipelineConfig.from_hydra_config(ablation.pipeline).model_dump()
    assert original_pipeline.pop('denoised_task_loss_weight') == .5
    assert ablation_pipeline.pop('denoised_task_loss_weight') == 0
    assert original_pipeline == ablation_pipeline
    assert ablation.run_name == 'poseflow_v26_latent_only'


def test_all_negative_loss_is_finite_and_has_no_geometry_gradient():
    batch = batch_fixture((0, 1))
    targets = {name:value[:1] for name, value in build_v26_targets(batch, proposals=5).items()}
    output = dict(existence_logits=torch.zeros(1, 5, requires_grad=True),
                  pred_boxes=torch.full((1, 5, 4), .5, requires_grad=True),
                  joint_xy=torch.full((1, 5, 14, 2), .5, requires_grad=True))
    losses = v26_task_loss(output, targets)
    assert torch.isfinite(losses['total'])
    losses['total'].backward()
    assert output['existence_logits'].grad.abs().sum() > 0
    assert output['pred_boxes'].grad.abs().sum() == 0
    assert output['joint_xy'].grad.abs().sum() == 0


def test_image_extraction_batch_equivalence_and_import_coverage(tmp_path):
    from src.models.pipelines.pose_estimators.poseflow.training_v26 import initialize_v26_image_weights
    model = tiny_model().eval()
    source = {('transformer.level_embed' if name == 'level_embed' else name):value.clone()
              for name, value in model.state_dict().items()
              if name.startswith(('backbone.', 'input_proj.', 'level_embed'))}
    checkpoint = tmp_path/'source.pt'
    torch.save({'model':source}, checkpoint)
    other = tiny_model().eval()
    report = initialize_v26_image_weights(other, checkpoint)
    assert report['image_tensor_coverage'] == 1
    images = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        expected = model.encode_image(images)
        actual = other.encode_image(images)
        separate = [other.encode_image(images[index:index+1]) for index in range(2)]
        changed_images = images.clone()
        changed_images[1] += 10
        changed = other.encode_image(changed_images)
    for index, (left, right) in enumerate(zip(expected, actual)):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
        # FP32 convolution kernels differ slightly between B1 and B2.
        torch.testing.assert_close(right, torch.cat([row[index] for row in separate]), atol=3e-5, rtol=1e-5)
        torch.testing.assert_close(right[0], changed[index][0], atol=0, rtol=0)
    source.pop(next(iter(source)))
    torch.save({'model':source}, checkpoint)
    with pytest.raises(ValueError, match='missing/incompatible'):
        initialize_v26_image_weights(other, checkpoint)


def test_coverage_diagnostic_and_preaugmentation_capture():
    from src.models.pipelines.pose_estimators.poseflow.training_v26 import PoseFlowV26PipelineConfig
    from src.models.pipelines.pose_estimators.poseflow.quick_eval_v26 import coverage_diagnostics
    batch = batch_fixture((1,))
    batch['gt_boxes_pixel'] = torch.tensor([[[40., 25., 40., 30.]]])
    points = ((batch['gt_keypoints']+1)/2).expand(1, 3, 14, 2).clone()
    output = dict(pred_keypoints=torch.cat((points.flatten(-2), torch.ones(1, 3, 14)), -1),
        pred_boxes=torch.ones(1, 3, 4), existence_logits=torch.ones(1, 3))
    diagnostic = coverage_diagnostics(output, batch)
    assert diagnostic['covered_people'] == 1 and diagnostic['duplicate_predictions'] == 2
    config = PoseFlowV26PipelineConfig(backbone_pretrained=False, hidden_dim=32,
        num_heads=4, feedforward_dim=64, num_feature_levels=2, quick_eval_enabled=False)
    pipeline = config.build(max_epochs=2)
    batch['image'] = [batch['image'][0]]
    captured = pipeline.on_before_batch_transfer(batch, 0)
    batch['gt_labeled'].zero_()
    assert captured['v26_original_labeled'].any()

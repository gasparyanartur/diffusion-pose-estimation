import pytest
import torch

from src.models.pipelines.pose_estimators.poseflow import PoseFlowV11PipelineConfig
from src.models.pipelines.pose_estimators.poseflow.model_v11 import PoseFlowV11DecoderLayer
from src.models.pipelines.pose_estimators.poseflow.training_v11 import build_poseflow_v11_targets
from src.models.pipelines.pose_estimators.diffusionregpose.training import DiffusionRegPoseOTAMatcher
from tests.test_poseflow import _batch


def _config(**overrides):
    return PoseFlowV11PipelineConfig(
        backbone_pretrained=False, num_queries=4, num_proposals=2, num_group=2,
        num_encoder_layers=1, num_decoder_layers=2, num_feature_levels=3,
        hidden_dim=32, feedforward_dim=64, num_heads=8, num_body_points=2,
        num_box_decoder_layers=1, dn_number=2, **overrides,
    )


def test_coarse_matcher_does_not_require_pose():
    matcher = DiffusionRegPoseOTAMatcher(num_body_points=2)
    output = {'pred_logits': torch.tensor([[[-4., 4.], [-4., -4.]]]),
              'pred_boxes': torch.tensor([[[.5, .5, .2, .4], [.1, .1, .1, .1]]])}
    target = {'labels': torch.tensor([1]), 'boxes': output['pred_boxes'][0, :1]}
    indices = matcher(output, [target], include_keypoints=False)
    assert indices[0][0].tolist() == [0]


def test_v11_layout_conditioning_and_shared_readouts():
    model = _config().build(max_epochs=2).model.eval()
    latent = torch.randn(1, 4, 32)
    state = torch.randn(1, 4, 8)
    selected = torch.tensor([[0, 2]])
    coarse = model.predict_coarse(latent)
    assert 'pred_keypoints' not in coarse
    assert coarse['pred_boxes'].shape == (1, 4, 4)
    tokens = model.expand_tokens(latent, state, torch.ones(1), selected)
    assert tokens.shape == (1, 2, 5, 32)
    changed = state.clone()
    changed[:, 0, 4:6] += .3
    altered = model.expand_tokens(latent, changed, torch.ones(1), selected)
    assert torch.equal(tokens[:, :, :3], altered[:, :, :3])
    assert not torch.equal(tokens[:, 0, 3], altered[:, 0, 3])
    assert torch.equal(tokens[:, 0, 4], altered[:, 0, 4])
    prediction, references = model.predict_tokens(tokens)
    assert prediction['pred_keypoints'].shape == (1, 2, 6)
    assert references.shape == (1, 2, 5, 4)
    assert torch.equal(references[:, :, 0], prediction['pred_boxes'])
    assert torch.equal(references[:, :, 2], prediction['pred_boxes'])
    assert torch.equal(references[..., 2:], prediction['pred_boxes'][:, :, None, 2:].expand(-1, -1, 5, -1))


def test_v11_sampling_joint_conditioning_matches_zeroed_branch(monkeypatch):
    model = _config(flow_sampling_steps=3).build(max_epochs=2).model.eval()
    assert model.sampling_joint_conditioning is False
    images = torch.randn(1, 3, 64, 64)
    sampling_controls = []
    original_forward = model.forward

    def record_forward(*args, **kwargs):
        sampling_controls.append(kwargs['apply_joint_conditioning'])
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(model, 'forward', record_forward)
    with torch.no_grad():
        torch.manual_seed(77)
        corrected = model.sample_keypoints(images)
        assert sampling_controls == [False, False, False]
        sampling_controls.clear()
        model.sampling_joint_conditioning = True
        hook = model.joint_conditioning.register_forward_hook(
            lambda module, inputs, output: torch.zeros_like(output),
        )
        try:
            torch.manual_seed(77)
            reference = model.sample_keypoints(images)
        finally:
            hook.remove()
        assert sampling_controls == [True, True, True]
    for field in ('pred_logits', 'pred_boxes', 'pred_keypoints', 'flow_person_state'):
        torch.testing.assert_close(corrected[field], reference[field], rtol=0, atol=0)


def test_v11_sampling_control_preserves_direct_forward_conditioning():
    model = _config(sampling_joint_conditioning=False).build(max_epochs=2).model.eval()
    latent = torch.randn(1, 4, 32)
    state = torch.randn(1, 4, 8)
    selected = torch.tensor([[0, 2]])
    times = torch.ones(1)
    conditioned = model.expand_tokens(latent, state, times, selected)
    unconditioned = model.expand_tokens(latent, state, times, selected,
                                       apply_joint_conditioning=False)
    expected = latent.gather(1, selected[..., None].expand(-1, -1, 32))[:, :, None]
    expected = expected + model.role_embeddings.weight[None, None]
    torch.testing.assert_close(unconditioned, expected, rtol=0, atol=0)
    assert torch.equal(conditioned[:, :, :3], unconditioned[:, :, :3])
    assert not torch.equal(conditioned[:, :, 3:], unconditioned[:, :, 3:])
    legacy = _config(sampling_joint_conditioning=True).build(max_epochs=2).model
    legacy.load_state_dict(model.state_dict(), strict=True)
    assert legacy.sampling_joint_conditioning is True


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required'))])
def test_v11_losses_gradients_dn_and_sampling(device):
    pipeline = _config().build(max_epochs=2).to(device)
    batch = {name: value.to(device) for name, value in _batch().items()}
    batch['image'] = torch.randn(1, 3, 64, 64, device=device)
    losses = pipeline._losses_for_batch(batch)
    assert torch.isfinite(losses['total'])
    assert not any('corner' in name or 'center_' in name for name in losses)
    assert losses['loss_keypoints_0'] == 0
    assert losses['dn_loss_bbox'] > 0
    losses['total'].backward()
    for module in (pipeline.model.point_head, pipeline.model.size_head,
                   pipeline.model.existence_head, pipeline.model.joint_conditioning,
                   pipeline.model.dn_box_conditioning):
        assert module.layers[-1].weight.grad is not None
        assert torch.isfinite(module.layers[-1].weight.grad).all()
        assert module.layers[-1].weight.grad.abs().sum() > 0
    pipeline.eval()
    with torch.no_grad():
        sampled = pipeline.model.sample_keypoints(batch['image'], collect_step_diagnostics=True)
    assert torch.isfinite(sampled['flow_person_state']).all()
    state = (sampled['flow_person_state'] / pipeline.model.scale + 1) / 2
    assert (state[..., 2:4] > 0).all()
    assert len(sampled['flow_step_diagnostics']) == 3
    assert len(sampled['flow_step_diagnostics'][0]['token_diagnostics']) == 2
    assert all(not value.requires_grad for row in sampled['flow_step_diagnostics'][0]['token_diagnostics'] for value in row.values())


def test_v11_center_size_target_geometry_and_empty_people():
    model = _config().build(max_epochs=2).model
    batch = _batch()
    state, source, times, targets = build_poseflow_v11_targets(
        batch, model, num_queries=4, num_body_points=2,
        padding_policy='source_prior', noise_seed=42,
    )
    endpoint = (state - (1 - times[:, None, None]) * source) / times[:, None, None]
    normalized = (endpoint / model.scale + 1) / 2
    assert torch.allclose(normalized[0, :2, :4], batch['gt_boxes_canon'][0], atol=1e-5)
    assert torch.allclose(targets[0]['boxes'], batch['gt_boxes_canon'][0])
    assert (normalized[..., 2:4] > 0).all()
    batch['gt_labeled'].zero_()
    batch['person_mask'] = torch.zeros(1, 2, dtype=torch.bool)
    empty, _, _, targets = build_poseflow_v11_targets(
        batch, model, num_queries=4, num_body_points=2,
        padding_policy='source_prior', noise_seed=42,
    )
    assert targets[0]['labels'].numel() == 0
    assert torch.isfinite(empty).all()


def test_v11_decoder_order_and_attention_isolation():
    layer = PoseFlowV11DecoderLayer(d_model=32, d_ffn=64, dropout=0, num_levels=1, num_heads=8).eval()
    events = []
    hooks = [module.register_forward_pre_hook(lambda module, args, name=name: events.append(name))
             for name, module in [('image', layer.cross_attn), ('person', layer.self_attn), ('exist', layer.exist_attn), ('ffn', layer.linear1)]]
    tokens = torch.randn(1, 2, 5, 32)
    kwargs = dict(references=torch.rand(1, 10, 1, 4), memory=torch.randn(1, 4, 32),
                  spatial_shapes=torch.tensor([[2, 2]]), level_start_index=torch.tensor([0]))
    output, _ = layer(tokens, None, **kwargs)
    for hook in hooks:
        hook.remove()
    assert events == ['image', 'person', 'exist', 'ffn']
    changed = tokens.clone()
    changed[:, 1] += torch.randn_like(changed[:, 1])
    altered, _ = layer(changed, None, **kwargs)
    assert torch.equal(output[:, 0, 1:], altered[:, 0, 1:])
    assert not torch.equal(output[:, 0, 0], altered[:, 0, 0])


def test_v11_dn_group_and_clean_isolation():
    model = _config().build(max_epochs=2).model.eval()
    target = {'labels': torch.tensor([1])}
    mask = model.existence_attention_mask([target], 1, torch.device('cpu'))
    assert mask[:, 2:, :2].all()
    assert mask[:, 0, 1].all() and mask[:, 1, 0].all()
    assert not mask[:, 2:, 2:].any()
    layer = model.transformer.decoder.layers[0].eval()
    clean, dn = torch.randn(1, 2, 5, 32), torch.randn(1, 2, 3, 32)
    kwargs = dict(references=torch.rand(1, 16, 3, 4), memory=torch.randn(1, 6, 32),
                  spatial_shapes=torch.tensor([[2, 2], [1, 1], [1, 1]]),
                  level_start_index=torch.tensor([0, 4, 5]), existence_mask=mask)
    base_clean, base_dn = layer(clean, dn, **kwargs)
    changed = dn.clone()
    changed[:, 1] += torch.randn_like(changed[:, 1])
    altered_clean, altered_dn = layer(clean, changed, **kwargs)
    assert torch.equal(base_clean, altered_clean)
    assert torch.equal(base_dn[:, 0], altered_dn[:, 0])


def test_v11_references_keep_gradients_between_layers():
    model = _config().build(max_epochs=2).model.eval()
    references = []
    def capture(module, args):
        args[1].retain_grad()
        references.append(args[1])
    hooks = [layer.cross_attn.register_forward_pre_hook(capture)
             for layer in model.transformer.decoder.layers]
    output = model(torch.randn(1, 3, 64, 64), torch.randn(1, 4, 4),
                   torch.full((1,), .5), person_state=torch.randn(1, 4, 8))
    output['pred_keypoints'].sum().backward()
    for hook in hooks:
        hook.remove()
    assert len(references) == 2
    assert all(reference.grad is not None and reference.grad.abs().sum() > 0 for reference in references)


def test_v11_preset_and_exact_pretrained_transfer():
    from pathlib import Path
    from hydra import compose, initialize_config_dir
    from src.models.pipelines.pose_estimators.diffusionregpose.training import remap_diffusionregpose_checkpoint_layers
    from src.models.pipelines.pose_estimators.poseflow import PoseFlowPipelineConfig
    config_dir = Path(__file__).resolve().parents[1] / 'conf'
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        resolved = compose(config_name='benchmark_poseflow_v11')
    config = PoseFlowPipelineConfig.from_hydra_config(resolved.pipeline)
    assert config.hidden_dim == 256 and config.feedforward_dim == 2048
    assert config.num_body_points == 14 and config.num_queries == 200 and config.num_proposals == 50
    assert config.scheduler_type == 'two_cycle_cosine'
    assert config.lr == 1e-4 and config.lr_backbone == 5e-5
    assert config.flow_sampling_steps == 3
    pipeline = config.build(max_epochs=80)
    model = pipeline.model
    checkpoint = torch.load(config.pretrained_checkpoint, map_location='cpu', weights_only=False)
    state = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
    state = {key.removeprefix('module.'): value for key, value in state.items()}
    state = remap_diffusionregpose_checkpoint_layers(
        state, encoder_layer_indices=config.pretrained_keep_encoder_layers,
        decoder_layer_indices=config.pretrained_keep_decoder_layers,
        target_num_encoder_layers=3, target_num_decoder_layers=3, target_num_box_decoder_layers=1,
    )
    keys = [name for name in model.state_dict() if name.startswith(('backbone.', 'input_proj.', 'transformer.encoder.layers.'))]
    keys += [name for name in model.state_dict() if name.startswith('transformer.decoder.layers.')
             and '.exist_' not in name]
    assert keys and all(torch.equal(model.state_dict()[name], state[name]) for name in keys)
    assert model.role_embeddings.num_embeddings == 17
    assert model.hypothesis_stage.state_width == 4
    model.load_state_dict(model.state_dict(), strict=True)
    optimizer = pipeline.configure_optimizers()['optimizer']
    rates = {group['schedule_group']: group['lr'] for group in optimizer.param_groups}
    assert rates == pytest.approx({'main': 1e-4, 'backbone': 5e-5})


def test_v11_bounded_validation_diagnostics_preserve_predictions_and_rng(monkeypatch):
    from src.eval.poseflow_diagnostics import FEATURE_PATHS
    pipeline = _config(solution_feature_diagnostics=True).build(max_epochs=2).eval()
    image = torch.randn(1, 3, 64, 64)
    generator = torch.Generator().manual_seed(42)
    rng = torch.random.get_rng_state()
    baseline = pipeline.model.sample_keypoints(image, generator=generator)
    expected_generator = generator.get_state()
    expected_rng = torch.random.get_rng_state()
    torch.random.set_rng_state(rng)
    generator.manual_seed(42)
    output = pipeline._validation_sample(image, None, generator, collect_diagnostics=True)
    for name in ('pred_logits', 'pred_boxes', 'pred_keypoints'):
        torch.testing.assert_close(output[name], baseline[name], rtol=0, atol=0)
    assert torch.equal(generator.get_state(), expected_generator)
    assert torch.equal(torch.random.get_rng_state(), expected_rng)
    assert set(pipeline._validation_feature_outputs) == set(FEATURE_PATHS)
    assert all(not module._forward_hooks for module in pipeline.model.modules())
    batch = dict(image=image, gt_keypoints=torch.zeros(1, 1, 2, 2),
                 gt_labeled=torch.ones(1, 1, 2), person_mask=torch.ones(1, 1),
                 gt_boxes_pixel=torch.tensor([[[24., 24., 16., 16.]]]), img_hw=torch.tensor([[64., 64.]]))
    results = [dict(keypoints=torch.zeros(0, 6), scores=torch.zeros(0))]
    logged = {}
    monkeypatch.setattr(pipeline, 'log', lambda name, value, **kwargs: logged.update({name: value}))
    # This logger uses CrowdPose sigmas: exercise its full 14-joint contract below;
    # the summary helper also accepts reduced-joint unit-test configurations.
    from src.eval.poseflow_training_diagnostics import summarize_validation_solution, summarize_feature_intervention
    summary = summarize_validation_solution(batch, output, results, torch.full((2,), .1))
    assert 'stage/hypothesis_all/oracle_box_iou' in summary
    assert 'stage/hypothesis_all/oracle_oks' not in summary
    assert 'stage/refiner_1/paired_oks' in summary
    changed = summarize_feature_intervention(batch, output, output, torch.full((2,), .1))
    assert 'initial_oracle_oks_change_per_gt' not in changed
    assert changed['initial_box_l1'] == 0
    pipeline._log_validation_solution(batch, output, results, batch_index=1)
    assert not logged
    output = pipeline._validation_sample(image, None, generator, collect_diagnostics=False)
    assert 'flow_step_diagnostics' not in output
    assert not pipeline._validation_feature_outputs


def test_v11_empty_targets_have_finite_classification_only_losses():
    pipeline = _config().build(max_epochs=2)
    batch = _batch() | {'image': torch.randn(1, 3, 64, 64),
                        'person_mask': torch.zeros(1, 2, dtype=torch.bool)}
    batch['gt_labeled'].zero_()
    losses = pipeline._losses_for_batch(batch)
    assert torch.isfinite(losses['total'])
    assert losses['loss_ce'] > 0
    assert losses['loss_bbox'] == losses['loss_keypoints'] == 0
    losses['total'].backward()

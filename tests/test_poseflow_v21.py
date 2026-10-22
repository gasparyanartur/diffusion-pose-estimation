import pytest
import torch

from src.models.pipelines.pose_estimators.poseflow.model_v21 import PoseFlowV21Model
from src.models.pipelines.pose_estimators.poseflow.training_v21 import PoseFlowV21PipelineConfig
from tests.test_poseflow import _batch


def configuration(**overrides):
    return PoseFlowV21PipelineConfig(**(dict(
        backbone_pretrained=False, num_queries=3, num_proposals=3, num_group=3,
        num_encoder_layers=1, num_decoder_layers=3, num_box_decoder_layers=1,
        hidden_dim=32, feedforward_dim=64, num_heads=4, num_feature_levels=2,
        num_body_points=2, dn_number=2, pretrained_checkpoint=None,
    ) | overrides))


def test_fixed_counts_and_v1_recipe():
    from hydra import compose, initialize_config_dir
    from pathlib import Path
    with initialize_config_dir(config_dir=str(Path('conf').resolve()), version_base=None):
        config = compose(config_name='benchmark_poseflow_v21')
    assert config.pipeline.num_queries == config.pipeline.num_proposals == config.pipeline.num_group == 50
    assert config.pipeline.dn_number == 50 and config.training.grad_clip == .1
    assert config.pipeline.lr == 2e-4 and config.pipeline.scheduler_type == 'multistep'
    with pytest.raises(ValueError, match='equal'):
        configuration(num_queries=4)


def test_ordered_expansion_and_hypothesis_masks():
    model = configuration()._build_poseflow_model().eval()
    features = torch.randn(1, 3, 32)
    references = torch.rand(1, 3, 4)*.5+.25
    expanded, _ = model.expand_ordered(features, references, torch.rand(1, 3, 4), dn_count=0)
    torch.testing.assert_close(expanded[:, ::3], features, atol=0, rtol=0)
    model.class_embed[0].weight.data.normal_(std=5)
    again, _ = model.expand_ordered(features, references, torch.rand(1, 3, 4), dn_count=0)
    torch.testing.assert_close(again[:, ::3], features, atol=0, rtol=0)
    mask = model.hypothesis_attention_mask(batch_size=1, dn_count=0, expanded=True, device=features.device)
    assert not mask[0, 0, 3] and not mask[0, 0, 9]
    assert mask[0, 1, 3] and mask[0, 1, 10] and not mask[0, 1, 9]
    assert not mask[0, 9, :3].any() and mask[0, 9, 3:9].all()


def test_time_read_gate_and_detector_gradient_without_proposal_loss():
    model = configuration(dn_number=0)._build_poseflow_model().eval()
    image = torch.randn(1, 3, 64, 64)
    state = torch.randn(1, 3, 4)
    output = model(image, state, torch.tensor([.5]))
    assert output['hypothesis_memory'].shape == (1, 3, 32)
    output['pred_logits'].square().sum().backward()
    assert model.dense_read.gate.grad.abs() > 0
    model.zero_grad(set_to_none=True)
    model.dense_read.gate.data.fill_(.2)
    output = model(image, state, torch.tensor([.5]))
    output['pred_logits'].square().sum().backward()
    assert model.dense_read.attention.in_proj_weight.grad.abs().sum() > 0
    assert model.hypothesis_slots.grad.abs().sum() > 0
    torch.testing.assert_close(model.dense_strength(torch.tensor([0., 1.])), torch.tensor([.1, 1.]))


@pytest.mark.parametrize('order', ['parallel', 'h_then_g'])
def test_dn_loss_sampler_carry_and_strict_restore(tmp_path, order):
    from scripts.evaluation.evaluate_poseflow import load_poseflow_model
    from omegaconf import OmegaConf
    config = configuration(hypothesis_update_order=order)
    pipeline = config.build(max_epochs=2).train()
    losses = pipeline._losses_for_batch(_batch() | {'image':torch.randn(1, 3, 64, 64)})
    assert losses['total'].isfinite() and 'dn_loss_ce' in losses and 'loss_ce_interm' in losses
    losses['total'].backward()
    model = pipeline.model.eval()
    image = torch.randn(1, 3, 64, 64)
    def sample(steps, persist):
        model.flow_sampling_steps = steps
        return model.sample_keypoints(image, generator=torch.Generator().manual_seed(8), persist_hypotheses=persist)
    first, reset = sample(1, True), sample(1, False)
    for key in ['pred_logits', 'pred_boxes', 'pred_keypoints', 'flow_keypoints', 'hypothesis_memory']:
        torch.testing.assert_close(first[key], reset[key], atol=0, rtol=0)
    later, reset = sample(3, True), sample(3, False)
    assert not torch.equal(later['hypothesis_memory'], reset['hypothesis_memory'])
    torch.save({'state_dict': pipeline.state_dict()}, tmp_path/'model.ckpt')
    OmegaConf.save(OmegaConf.create({'pipeline':config.model_dump() | {
        '_target_':'src.models.pipelines.pose_estimators.poseflow.PoseFlowV21PipelineConfig'}}), tmp_path/'config.yaml')
    restored, _ = load_poseflow_model(tmp_path/'model.ckpt', tmp_path/'config.yaml')
    assert isinstance(restored, PoseFlowV21Model)
    restored.eval()
    restored.flow_sampling_steps = model.flow_sampling_steps
    actual = restored.sample_keypoints(image, generator=torch.Generator().manual_seed(8))
    torch.testing.assert_close(actual['pred_boxes'], later['pred_boxes'], atol=0, rtol=0)


@pytest.mark.parametrize('order', ['parallel', 'h_then_g'])
def test_geometry_block_matches_v1_when_hypothesis_read_is_masked(order):
    model = configuration(dn_number=0, hypothesis_update_order=order)._build_poseflow_model().eval()
    image = torch.randn(1, 3, 64, 64)
    inputs = model.prepare_transformer_inputs(image, torch.zeros(1, 64, 64, dtype=torch.bool))
    memory = model.encode_features(inputs, torch.tensor([.3]))
    features, hypotheses = torch.randn(1, 3, 32), torch.randn(1, 3, 32)
    references = torch.rand(1, 3, 4)*.5+.25
    decoder = model.transformer.decoder
    layer = decoder.layers[0]
    scaled = references[:, :, None]*torch.cat((inputs.valid_ratios, inputs.valid_ratios), -1)[:, None]
    from src.models.pipelines.pose_estimators.diffusionregpose.transformer import sine_embedding_for_position
    position = decoder.ref_point_head(sine_embedding_for_position(scaled[:, :, 0], num_pos_feats=16))
    expected = layer(features.transpose(0, 1), tgt_query_pos=position.transpose(0, 1),
        tgt_reference_points=scaled.transpose(0, 1), memory=memory,
        memory_spatial_shapes=inputs.spatial_shapes, memory_level_start_index=inputs.level_start_index,
        memory_key_padding_mask=inputs.padding_mask)
    mask = model.hypothesis_attention_mask(batch_size=1, dn_count=0, expanded=False, device=image.device)
    mask[:, :3, 3:] = True
    actual, _ = model.refine_with_hypotheses(layer, features, references, hypotheses, inputs, memory, mask)
    torch.testing.assert_close(actual.transpose(0, 1), expected, atol=1e-6, rtol=1e-6)


def test_original_coordinate_fusion_and_reference_updates_on_two_images():
    model = configuration(dn_number=0)._build_poseflow_model().eval()
    decoder = model.transformer.decoder
    features = torch.randn(2, 3, 32)
    features[:, :, 0] = torch.tensor([3., 2., 1.])
    references, noisy = torch.rand(2, 3, 4)*.5+.25, torch.randn(2, 3, 4)
    with torch.no_grad():
        model.class_embed[0].weight.zero_()
        model.class_embed[0].bias.fill_(-10)
        model.class_embed[0].weight[1, 0] = 1
        model.class_embed[0].bias[1] = 0
    expected_features, expected_references, _ = decoder._expand_keypoint_queries(
        features.transpose(0, 1), references.transpose(0, 1), noisy, state_keypoint_gate=None)
    actual_features, actual_references = model.expand_ordered(features, references, noisy, dn_count=0)
    torch.testing.assert_close(actual_features, expected_features.transpose(0, 1), atol=0, rtol=0)
    torch.testing.assert_close(actual_references, expected_references.transpose(0, 1), atol=0, rtol=0)
    _, expected = decoder._update_expanded_queries(1, expected_features, expected_references)
    actual = model.update_pose_references(1, actual_features, actual_references, 0)
    torch.testing.assert_close(actual, expected.transpose(0, 1), atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize('order', ['parallel', 'h_then_g'])
def test_no_sorting_empty_targets_padding_and_memory_validation(monkeypatch, order):
    pipeline = configuration(hypothesis_update_order=order).build(max_epochs=2).train()
    empty = _batch() | {'image':torch.randn(1, 3, 64, 64), 'gt_labeled':torch.zeros(1, 2, 2, dtype=torch.bool)}
    loss = pipeline._losses_for_batch(empty)['total']
    assert loss.isfinite()
    loss.backward()
    model = pipeline.model.eval()
    def forbidden(*args, **kwargs):
        raise AssertionError('fixed-slot forward must not select or sort candidates')
    monkeypatch.setattr(torch, 'topk', forbidden)
    output = model(empty['image'], torch.randn(1, 3, 4), torch.zeros(1))
    assert output['pred_boxes'].shape == (1, 3, 4)
    with pytest.raises(ValueError, match='memory'):
        model(empty['image'], torch.randn(1, 3, 4), torch.zeros(1), hypothesis_memory=torch.zeros(1, 2, 32))
    with pytest.raises(ValueError, match='pixels'):
        model(empty['image'], torch.randn(1, 3, 4), torch.zeros(1), padding_mask=torch.ones(1, 64, 64, dtype=torch.bool))
    mask = model.hypothesis_attention_mask(batch_size=1, dn_count=2, expanded=True,
                                          device=torch.device('cpu'), targets=[{'labels':torch.empty(0, dtype=torch.long)}])
    assert mask[:, 2:, :2].all() and mask[:, :2, -3:].all()


def test_next_call_loss_reaches_previous_hypothesis_without_coordinate_handoff():
    model = configuration(dn_number=0)._build_poseflow_model().eval()
    image, source = torch.randn(1, 3, 64, 64), torch.randn(1, 3, 4)
    first = model(image, source, torch.zeros(1))
    second = model(image, source.detach(), torch.full((1,), .5), hypothesis_memory=first['hypothesis_memory'])
    gradient = torch.autograd.grad(second['pred_logits'].square().sum(), first['hypothesis_memory'])[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0

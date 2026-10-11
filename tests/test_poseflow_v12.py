import pytest
import torch

from tests.test_poseflow import _batch
from src.models.pipelines.pose_estimators.poseflow.latent_v12 import (
    V12StateCodec, build_v12_targets, prediction_loss, latent_loss,
    cosine_alpha, corrupt_latents, ddim_update,
)


def test_v12_targets_are_fixed_reproducible_and_empty_boxes_enclose_joints():
    targets = build_v12_targets(_batch(), groups=4, generator=torch.Generator().manual_seed(42))
    repeated = build_v12_targets(_batch(), groups=4, generator=torch.Generator().manual_seed(42))
    for key in targets:
        assert torch.equal(targets[key], repeated[key])
    assert targets['existence'].shape == (1, 4, 1)
    assert targets['real'].sum() == 2
    assert targets['labeled'][targets['real']].sum() == 3
    empty = ~targets['real']
    joints = targets['joints'][empty]
    bounds = torch.cat(((joints.amin(-2)+joints.amax(-2))/2,
                        (joints.amax(-2)-joints.amin(-2)).clamp_min(1e-6)), -1)
    torch.testing.assert_close(targets['boxes'][empty], bounds)
    overflow = build_v12_targets(_batch(), groups=1, generator=torch.Generator().manual_seed(42))
    assert overflow['overflow'].item() == 1
    empty_batch = _batch()
    empty_batch['gt_labeled'].zero_()
    assert build_v12_targets(empty_batch, groups=4)['real'].sum() == 0


def test_v12_clean_codec_scale_gradients_and_empty_geometry_mask():
    targets = build_v12_targets(_batch(), groups=4)
    codec = V12StateCodec(width=32, joints=2)
    clean = codec.encode(targets)
    assert clean.shape == (1, 4, 5, 32)
    torch.testing.assert_close(clean.mean(-1), torch.zeros(1, 4, 5), atol=1e-5, rtol=0)
    torch.testing.assert_close(clean.var(-1, unbiased=False), torch.ones(1, 4, 5), atol=2e-3, rtol=0)
    loss = prediction_loss(codec.decode(clean), targets, reconstruct=True)['total']
    loss.backward()
    assert codec.joint_encoder.layers[0].weight.grad.abs().sum() > 0
    assert codec.point_head.layers[0].weight.grad.abs().sum() > 0
    output = codec.decode(clean.detach())
    changed = {**output, 'pred_boxes': output['pred_boxes'].clone(),
               'pred_keypoints': output['pred_keypoints'].clone()}
    changed['pred_boxes'][~targets['real']] = 0.1
    changed['pred_keypoints'][~targets['real']] = 0.1
    torch.testing.assert_close(prediction_loss(output, targets)['total'],
                               prediction_loss(changed, targets)['total'])


def test_v12_noise_endpoints_ddim_and_detached_role_balanced_loss():
    clean = torch.randn(2, 4, 5, 32, requires_grad=True)
    noise = torch.randn_like(clean)
    times = torch.tensor([0., 1.])
    alpha = cosine_alpha(times)
    assert torch.equal(alpha, torch.tensor([1., 0.]))
    corrupted = corrupt_latents(clean.detach(), noise, times)
    assert torch.equal(corrupted[0], clean[0])
    assert torch.equal(corrupted[1], noise[1])
    predicted = clean.detach().clone().requires_grad_()
    loss = latent_loss(predicted, clean.detach())
    loss.backward()
    assert clean.grad is None
    torch.testing.assert_close(ddim_update(corrupted[1:], clean.detach()[1:],
                                          torch.ones(1), torch.zeros(1)), clean.detach()[1:])


def _model():
    from src.models.pipelines.pose_estimators.diffusionregpose.model import DiffusionRegPoseModel
    from src.models.pipelines.pose_estimators.poseflow.model_v12 import PoseFlowV12Model
    image_model = DiffusionRegPoseModel(backbone_pretrained=False, hidden_dim=32,
        feedforward_dim=64, num_heads=8, num_feature_levels=3, num_encoder_layers=1,
        num_decoder_layers=2, num_body_points=2, num_box_decoder_layers=1,
        num_queries=3, num_proposals=3, num_group=3, dn_number=0)
    return PoseFlowV12Model(image_model, sampling_steps=3)


def test_v12_first_layer_sees_image_and_padding_is_masked():
    model = _model().eval()
    noisy = torch.randn(1, 3, 5, 32)
    memory, position = torch.randn(1, 7, 32), torch.randn(1, 7, 32)
    mask = torch.tensor([[False, False, False, False, True, True, True]])
    times = torch.ones(1)
    output = model.denoise(noisy, times, (memory, position, mask))
    changed = memory.clone()
    changed[:, :4] += 1
    altered = model.denoise(noisy, times, (changed, position, mask))
    assert not torch.equal(output['aux_outputs'][0]['pred_latents'], altered['aux_outputs'][0]['pred_latents'])
    changed = memory.clone()
    changed[:, 4:] += 100
    masked = model.denoise(noisy, times, (changed, position, mask))
    torch.testing.assert_close(output['pred_latents'], masked['pred_latents'], rtol=0, atol=0)
    assert output['pred_latents'].shape == noisy.shape
    output['pred_latents'].square().mean().backward()
    assert model.layers[0].image_attention.in_proj_weight.grad.abs().sum() > 0
    assert model.layers[0].person_attention.in_proj_weight.grad.abs().sum() > 0


def test_v12_sampling_is_latent_only_reproducible_and_encodes_image_once(monkeypatch):
    model = _model().eval()
    image = torch.randn(1, 3, 64, 64)
    calls = []
    original = model.encode_image
    def encode(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(model, 'encode_image', encode)
    output = model.sample_keypoints(image, generator=torch.Generator().manual_seed(42),
                                   collect_step_diagnostics=True)
    assert len(calls) == 1
    with torch.no_grad():
        for head in (model.codec.point_head, model.codec.size_head, model.codec.existence_head):
            head.layers[-1].weight.add_(10)
    changed = model.sample_keypoints(image, generator=torch.Generator().manual_seed(42),
                                    collect_step_diagnostics=True)
    assert len(output['latent_step_diagnostics']) == 3
    for first, second in zip(output['latent_step_diagnostics'], changed['latent_step_diagnostics']):
        torch.testing.assert_close(first['latent_state'], second['latent_state'], rtol=0, atol=0)
    torch.testing.assert_close(output['latent_state'], changed['latent_state'], rtol=0, atol=0)
    assert not torch.equal(output['pred_keypoints'], changed['pred_keypoints'])
    state_keys = model.state_dict()
    assert not any('bbox_embed' in key or 'hypothesis_stage' in key for key in state_keys)


def _config(**overrides):
    from src.models.pipelines.pose_estimators.poseflow import PoseFlowV12PipelineConfig
    defaults = dict(backbone_pretrained=False, pretrained_checkpoint=None,
        num_queries=3, num_proposals=3, num_group=3, num_body_points=2,
        hidden_dim=32, feedforward_dim=64, num_heads=8, num_encoder_layers=1,
        num_decoder_layers=2, num_feature_levels=3, lr_cycle_restart_epoch=1)
    return PoseFlowV12PipelineConfig(**(defaults | overrides))


def test_v12_training_detaches_encoder_and_handles_empty_images():
    pipeline = _config().build(max_epochs=2)
    batch = _batch() | {'image': torch.randn(1,3,64,64)}
    losses = pipeline._losses_for_batch(batch)
    assert all(torch.isfinite(value) for value in losses.values())
    losses['loss_latent'].backward(retain_graph=True)
    assert pipeline.model.codec.joint_encoder.layers[0].weight.grad is None
    assert pipeline.model.layers[0].latent_head.weight.grad.abs().sum() > 0
    pipeline.zero_grad(set_to_none=True)
    losses['total'].backward()
    assert pipeline.model.codec.joint_encoder.layers[0].weight.grad.abs().sum() > 0
    batch['gt_labeled'].zero_()
    empty_losses = pipeline._losses_for_batch(batch)
    assert all(torch.isfinite(value) for value in empty_losses.values())
    assert empty_losses['loss_bbox'] == 0 and empty_losses['loss_keypoints'] == 0


def test_v12_config_restore_category_adapter_and_sampler_options(tmp_path):
    from omegaconf import OmegaConf
    from scripts.evaluation.evaluate_poseflow import load_poseflow_model, apply_flow_sampler_overrides
    from types import SimpleNamespace
    pipeline = _config().build(max_epochs=2).eval()
    checkpoint = tmp_path/'epoch000.ckpt'
    torch.save({'state_dict':pipeline.state_dict()}, checkpoint)
    config = tmp_path/'config.yaml'
    OmegaConf.save(OmegaConf.create({'pipeline':{'_target_':
        'src.models.pipelines.pose_estimators.poseflow.PoseFlowV12PipelineConfig',
        **pipeline.v12_config.model_dump(mode='json')}}), config)
    model, loaded_config = load_poseflow_model(checkpoint, config)
    model.eval()
    image = torch.randn(1,3,64,64)
    one = pipeline.model.sample_keypoints(image, generator=torch.Generator().manual_seed(7))
    two = model.sample_keypoints(image, generator=torch.Generator().manual_seed(7))
    torch.testing.assert_close(one['latent_state'], two['latent_state'], rtol=0, atol=0)
    assert loaded_config.matching_strategy == 'fixed'
    results = model.postprocess(two, torch.tensor([[64.,64.]]))
    assert (results[0]['labels'] == 1).all()
    with pytest.raises(ValueError, match='V12'):
        apply_flow_sampler_overrides(model, SimpleNamespace(flow_renewal=True,
            flow_renewal_score_threshold=None, flow_ensemble=False))
    with pytest.raises(ValueError):
        _config(num_queries=4)


def test_v12_hydra_preset_is_separate_fixed_group_family():
    from hydra import compose, initialize_config_dir
    from pathlib import Path
    from src.models.pipelines.pose_estimators.poseflow import PoseFlowPipelineConfig
    with initialize_config_dir(config_dir=str(Path('conf').resolve()), version_base=None):
        config = compose(config_name='benchmark_poseflow_v12')
    parsed = PoseFlowPipelineConfig.from_hydra_config(config.pipeline)
    assert parsed.num_queries == parsed.num_proposals == parsed.num_group == 50
    assert parsed.dn_number == 0 and parsed.matching_strategy == 'fixed'
    assert parsed.num_encoder_layers == parsed.num_decoder_layers == 3
    assert parsed.flow_sampling_steps == 3 and parsed.timesteps == 1000
    assert config.out_dir == 'results/poseflow_v12_training'


def test_v12_intermediate_ddim_equation_and_group_permutation():
    clean, noise = torch.randn(1,3,5,32), torch.randn(1,3,5,32)
    times, next_times = torch.tensor([.75]), torch.tensor([.4])
    noisy = corrupt_latents(clean, noise, times)
    torch.testing.assert_close(ddim_update(noisy, clean, times, next_times),
                               corrupt_latents(clean, noise, next_times), atol=1e-6, rtol=1e-6)
    model = _model().eval()
    memory = (torch.randn(1,7,32), torch.randn(1,7,32), torch.zeros(1,7,dtype=torch.bool))
    permutation = torch.tensor([2,0,1])
    original = model.denoise(noisy, times, memory)
    changed = model.denoise(noisy[:,permutation], times, memory)
    torch.testing.assert_close(changed['pred_latents'], original['pred_latents'][:,permutation],
                               atol=2e-6, rtol=2e-5)
    with pytest.raises(ValueError):
        cosine_alpha(torch.tensor([-0.1]))

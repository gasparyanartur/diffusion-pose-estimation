from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.models.pipelines.pose_estimators.poseflow.training_v42 import PoseFlowV42PipelineConfig
from tests.test_diffusionregpose_training import _batch


def config(**overrides):
    return PoseFlowV42PipelineConfig(
        backbone_pretrained=False, pretrained_checkpoint=None,
        num_queries=4, num_proposals=2, num_group=2, dn_number=2,
        num_encoder_layers=1, num_decoder_layers=3, num_box_decoder_layers=1,
        num_feature_levels=3, hidden_dim=32, feedforward_dim=64,
        num_heads=8, num_body_points=2, **overrides,
    )


def test_no_expansion_box_references_and_matched_read_count():
    model = config().build(max_epochs=2).model.eval()
    seen = []
    hooks = []
    for index, layer in enumerate(model.transformer.decoder.layers):
        attention = layer.cross_attn if index == 0 else model.person_attention[index-1]
        hooks.append(attention.register_forward_pre_hook(
            lambda module, args: seen.append((args[0].shape[1], args[1].shape[-1], module.num_points))))
    noisy = torch.randn(2, 2, 4, requires_grad=True)
    output = model(torch.randn(2, 3, 64, 64), noisy, torch.tensor([0, 999]))
    for hook in hooks:
        hook.remove()
    assert seen == [(4, 4, 4), (2, 4, 12), (2, 4, 12)]
    assert output['pred_keypoints'].shape == (2, 2, 6)
    assert output['pose_features'].shape == (2, 2, 32)
    assert torch.allclose(output['pred_keypoints'][..., 4:], torch.ones(1).sigmoid())
    output['pred_keypoints'][..., :4].sum().backward()
    assert noisy.grad.abs().sum() > 0
    assert model.pose_encoder.layers[0].weight.grad.abs().sum() > 0
    assert all(reader.layers[-1].weight.grad is not None for reader in model.pose_readers[-1:])
    assert not hasattr(model.transformer.decoder, 'keypoint_embed')
    for original, replacement in zip(model.transformer.decoder.layers[1:], model.person_attention):
        assert torch.equal(original.cross_attn.value_proj.weight, replacement.value_proj.weight)
        assert replacement.num_points == 3 * original.cross_attn.num_points


@pytest.mark.parametrize('empty', [False, True])
def test_original_losses_dn_and_checkpoint_roundtrip(empty):
    pipeline = config().build(max_epochs=2)
    batch = _batch()
    batch['image'] = torch.randn(1, 3, 64, 64)
    if empty:
        batch['gt_labeled'].zero_()
    loss = pipeline._losses_for_batch(batch)
    assert torch.isfinite(loss['total'])
    assert 'dn_loss_bbox' in loss and 'loss_keypoints' in loss
    loss['total'].backward()
    assert pipeline.model.person_attention[0].sampling_offsets.weight.grad is not None
    restored = config().build(max_epochs=2, load_pretrained_checkpoint=False)
    restored.load_state_dict(pipeline.state_dict(), strict=True)
    assert all(torch.equal(value, restored.state_dict()[name]) for name, value in pipeline.state_dict().items())


def test_original_coordinate_corruption_and_ddim_calls():
    from src.models.pipelines.pose_estimators.diffusionregpose.training import build_diffusionregpose_targets
    model = config().build(max_epochs=2).model.eval()
    state, times, _ = build_diffusionregpose_targets(_batch(), model, num_proposals=2, num_body_points=2)
    assert times.tolist() == [0]
    calls = []
    hook = model.pose_encoder.register_forward_hook(lambda module, args, output: calls.append(args[0].shape))
    model.sampling_timesteps = 4
    result = model.sample_keypoints(torch.randn(1, 3, 64, 64), generator=torch.Generator().manual_seed(42))
    hook.remove()
    assert len(calls) == 4 and all(shape == (1, 2, 4) for shape in calls)
    assert result['diffusion_keypoints'].shape == (1, 2, 4)
    assert torch.isfinite(result['pred_keypoints']).all()


def test_dn_budget_reference_detach_and_batch_isolation():
    pipeline = config().build(max_epochs=2)
    model = pipeline.model
    from src.models.pipelines.pose_estimators.diffusionregpose.training import build_diffusionregpose_targets
    batch = _batch()
    noisy, times, targets = build_diffusionregpose_targets(batch, model, num_proposals=2, num_body_points=2)
    reads = []
    hooks = []
    for layer in model.transformer.decoder.layers[1:]:
        hooks.append(layer.cross_attn.register_forward_pre_hook(
            lambda module, args: reads.append((args[0].shape[1], module.num_points, args[1].requires_grad))))
    output = model(torch.randn(1, 3, 64, 64), noisy, times, targets)
    for hook in hooks:
        hook.remove()
    assert reads == [(2, 4, False), (2, 4, False)]
    assert [row['pred_boxes'].shape[1] for row in output['aux_outputs']] == [4, 2]
    model.eval()
    images = torch.randn(2, 3, 64, 64)
    noisy = torch.randn(2, 2, 4)
    times = torch.tensor([0, 999])
    with torch.no_grad():
        together = model(images, noisy, times)
        reversed_batch = model(images.flip(0), noisy.flip(0), times.flip(0))
    for name in ('pred_boxes', 'pred_logits', 'pred_keypoints'):
        assert torch.allclose(together[name], reversed_batch[name].flip(0), atol=1e-6)


def test_preset_and_quick_grid_contract():
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    with initialize_config_dir(version_base=None, config_dir=str(Path(__file__).resolve().parents[1] / 'conf')):
        base = compose(config_name='benchmark_diffusionregpose_development_compatible_256_keep_layers')
        variant = compose(config_name='benchmark_poseflow_v42')
    assert OmegaConf.to_container(base.data, resolve=True) == OmegaConf.to_container(variant.data, resolve=True)
    assert variant.pipeline.matching_strategy == 'hungarian'
    assert variant.pipeline.quick_eval_steps == [1, 4, 8]
    assert variant.pipeline.num_box_decoder_layers == 1
    assert variant.pipeline.pose_readout_mode == 'box_center_residual'
    assert variant.training.max_epochs == 80 and variant.training.accumulate_grad_batches == 2


def test_quick_grid_restores_settings_rng_and_uses_paired_sources(tmp_path):
    from src.models.pipelines.pose_estimators.poseflow.quick_eval_v42 import evaluate_v42_grid

    class Sampler(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))
            self.sampling_timesteps = 2
            self.transformer = SimpleNamespace(decoder=SimpleNamespace(num_body_points=2))
            self.sources = []

        def sample_keypoints(self, image, *, padding_mask, generator):
            self.sources.append(torch.randn(4, generator=generator))
            return dict(pred_logits=torch.zeros(1, 2, 2), pred_boxes=torch.full((1, 2, 4), 0.5),
                pred_keypoints=torch.full((1, 2, 6), 0.5))

    model = Sampler().train()
    batch = dict(image=torch.randn(1, 3, 16, 16), image_id=torch.tensor([7]), img_hw=torch.tensor([[16, 16]]))
    rng = torch.get_rng_state().clone()
    report = evaluate_v42_grid(model, [batch], config(), tmp_path, 'unused.json', [7],
        metric_evaluator=lambda records, annotations, ids: {'pose_AP': 0.1, 'bbox_AP': 0.2})
    assert model.training and model.sampling_timesteps == 2
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(model.sources[0], source) for source in model.sources)
    assert [row['steps'] for row in report['conditions'].values()] == [1, 4, 8]
    assert all(row['metrics'] == {'pose_AP': 0.1, 'bbox_AP': 0.2} for row in report['conditions'].values())


def test_box_center_reader_anchor_and_person_isolation():
    model = config(pose_readout_mode='box_center_residual').build(max_epochs=2).model
    reader = model.pose_readers[0]
    torch.nn.init.zeros_(reader.layers[-1].weight)
    torch.nn.init.zeros_(reader.layers[-1].bias)
    features = torch.zeros(2, 2, 32)
    references = torch.tensor([[[.2, .3, .1, .2], [.7, .8, .2, .1]],
                               [[.4, .6, .1, .1], [.6, .4, .2, .2]]])
    xy = model.read_pose(features, references, layer_index=0)
    assert torch.allclose(xy, references[..., :2].repeat(1, 1, 2))
    moved = references.clone()
    moved[1, 0, :2] += .1
    changed = model.read_pose(features, moved, layer_index=0)
    assert torch.allclose(changed[1, 0] - xy[1, 0], torch.full((4,), .1))
    assert torch.equal(changed[0], xy[0])
    assert torch.equal(changed[1, 1], xy[1, 1])


def test_anchored_reader_boundaries_gradients_and_forward_references():
    model = config(pose_readout_mode='box_center_residual').build(max_epochs=2).model.eval()
    features = torch.randn(1, 2, 32, requires_grad=True)
    references = torch.tensor([[[0., 1., 0., 0.], [1., 0., 1e-8, 1e-8]]])
    xy = model.read_pose(features, references, layer_index=0)
    assert torch.isfinite(xy).all() and ((xy > 0) & (xy < 1)).all()
    xy.sum().backward()
    assert torch.isfinite(features.grad).all() and features.grad.abs().sum() > 0
    seen = []
    original = model.read_pose
    def capture(features, references, *, layer_index):
        seen.append(references)
        return original(features, references, layer_index=layer_index)
    model.read_pose = capture
    model(torch.randn(1, 3, 64, 64), torch.randn(1, 2, 4), torch.zeros(1, dtype=torch.long))
    assert len(seen) == 2 and all(boxes.shape == (1, 2, 4) for boxes in seen)
    assert all(not boxes.requires_grad for boxes in seen)


def test_absolute_reader_default_and_mode_restore_contract():
    torch.manual_seed(42)
    absolute = config().build(max_epochs=2)
    torch.manual_seed(42)
    anchored = config(pose_readout_mode='box_center_residual').build(max_epochs=2)
    assert absolute.model.pose_readout_mode == 'absolute'
    anchored.load_state_dict(absolute.state_dict(), strict=True)
    assert absolute.state_dict().keys() == anchored.state_dict().keys()
    assert all(torch.equal(value, anchored.state_dict()[name]) for name, value in absolute.state_dict().items())
    features = torch.randn(1, 2, 32)
    references = torch.tensor([[[.2, .3, .1, .1], [.7, .8, .2, .2]]])
    raw = absolute.model.pose_readers[0](features)
    assert torch.equal(absolute.model.read_pose(features, references, layer_index=0), raw.sigmoid())
    from src.models.pipelines.pose_estimators.diffusionregpose.transformer import inverse_sigmoid
    expected = (raw + inverse_sigmoid(references[..., :2]).repeat(1, 1, 2)).sigmoid()
    assert torch.equal(anchored.model.read_pose(features, references, layer_index=0), expected)
    assert anchored.hparams['pose_readout_mode'] == 'box_center_residual'
    with pytest.raises(ValueError):
        config(pose_readout_mode='unknown')

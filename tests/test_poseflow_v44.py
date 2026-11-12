from copy import deepcopy

import pytest
import torch

from src.models.pipelines.pose_estimators.diffusionregpose.model import DiffusionRegPoseModel
from src.models.pipelines.pose_estimators.poseflow.model_v44 import PoseFlowV44Model
from src.models.pipelines.pose_estimators.poseflow.training_v44 import PoseFlowV44Pipeline, PoseFlowV44PipelineConfig
from tests.test_poseflow_v43_codec import small_codec
from tests.test_poseflow_v43_training import batch


def arguments():
    return dict(backbone_pretrained=False, num_queries=12, num_proposals=6, num_group=6,
        num_encoder_layers=1, num_decoder_layers=3, num_box_decoder_layers=1,
        num_feature_levels=3, hidden_dim=32, feedforward_dim=64, num_heads=4,
        num_body_points=14, dn_number=2)


def model():
    return PoseFlowV44Model(codec=small_codec(), **arguments()).eval()


def pipeline():
    config = PoseFlowV44PipelineConfig(native=arguments(), codec_width=32, latent_dim=8,
        pretrained_checkpoint=None, quick_eval_enabled=False)
    return PoseFlowV44Pipeline(config=config, model=model(), max_epochs=2).eval()


def test_native_prefix_parity_and_time_independence(monkeypatch):
    native = DiffusionRegPoseModel(**arguments()).eval()
    network = PoseFlowV44Model(codec=small_codec(), native_state=native.state_dict(), **arguments()).eval()
    images = torch.randn(1, 3, 128, 96)
    mask = torch.zeros(1, 128, 96, dtype=torch.bool)
    inputs = native.prepare_transformer_inputs(images, mask)
    memory = native.encode_features(inputs, None)
    features, references, _, _ = native.select_two_stage_queries(memory, inputs)
    selected = {}
    original = native.transformer.decoder._expand_keypoint_queries

    class StopAtExpansion(Exception):
        pass

    def capture(features, boxes, *args, **kwargs):
        _, _, metadata = original(features, boxes, *args, **kwargs)
        selected.update(metadata)
        raise StopAtExpansion

    monkeypatch.setattr(native.transformer.decoder, '_expand_keypoint_queries', capture)
    with pytest.raises(StopAtExpansion):
        native.transformer.decoder(features.transpose(0, 1), native.decoder_feature_memory(inputs, memory),
            refpoints_unsigmoid=references.transpose(0, 1), level_start_index=inputs.level_start_index,
            spatial_shapes=inputs.spatial_shapes, valid_ratios=inputs.valid_ratios,
            memory_key_padding_mask=inputs.padding_mask, diffusion_keypoints=torch.zeros(1, 6, 28))
    context = network.prepare_context(images, mask)
    torch.testing.assert_close(context['features'], selected['selected_person_features'], rtol=0, atol=0)
    torch.testing.assert_close(context['boxes'], selected['selected_person_references'], rtol=0, atol=0)
    assert torch.equal(context['indices'], selected['selected_person_indices'])
    assert not context['boxes'].requires_grad
    assert not hasattr(network.transformer.decoder, 'keypoint_embed')


def test_latent_geometry_gradient_and_frozen_codec():
    network = model()
    before = deepcopy(network.codec.state_dict())
    context = network.prepare_context(torch.randn(2, 3, 128, 128))
    state = torch.randn(2, 6, 8, requires_grad=True)
    sizes = torch.tensor([[128., 100.], [95., 128.]])
    output = network.denoise(state, torch.tensor([0, 999]), context, sizes)
    expected = network.codec.decode_latents(output['pred_latents'], sizes[:, None].expand(-1, 6, -1))
    torch.testing.assert_close(output['keypoints_px'], expected['keypoints_px'])
    assert len(output['aux_outputs']) == 1
    output['keypoints_px'].square().mean().backward()
    assert state.grad.abs().sum() > 0 and torch.isfinite(state.grad).all()
    assert network.latent_heads[-1].weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in network.codec.parameters())
    for name, value in before.items():
        assert torch.equal(value, network.codec.state_dict()[name])
    with pytest.raises(ValueError):
        network.denoise(state[..., :4], torch.tensor([0, 999]), context, sizes)


def test_cached_sampling_fixed_slots_no_encoding_and_latent_oracle(monkeypatch):
    network = model()
    images, sizes = torch.randn(1, 3, 128, 128), torch.tensor([[128., 100.]])
    contexts, steps = [], []
    prepare, denoise = network.prepare_context, network.denoise
    def context(*args, **kwargs):
        contexts.append(1)
        return prepare(*args, **kwargs)
    def step(state, times, cached, image_wh):
        steps.append((cached, times.clone()))
        return denoise(state, times, cached, image_wh)
    monkeypatch.setattr(network, 'prepare_context', context)
    monkeypatch.setattr(network, 'denoise', step)
    monkeypatch.setattr(network.codec, 'encode_posterior', lambda *_: pytest.fail('Inference encoded GT'))
    network.sampling_timesteps = 4
    result = network.sample_keypoints(images, image_wh=sizes, generator=torch.Generator().manual_seed(42))
    assert len(contexts) == 1 and len(steps) == 4
    assert all(item[0] is steps[0][0] for item in steps)
    assert torch.equal(result['reference_indices'], steps[0][0]['indices'])
    clean, noise, times = torch.randn(2, 6, 8), torch.randn(2, 6, 8), torch.tensor([0, 999])
    noisy = network.q_sample(clean, times, noise)
    torch.testing.assert_close(network.predict_noise_from_start(noisy, times, clean), noise, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(network.ddim_step(clean, noise, 20, -1, noise=torch.zeros_like(noise)), clean)


@pytest.mark.parametrize('counts', [(2, 1), (0,)])
def test_single_slot_assignment_auxiliary_masks_and_restore(counts):
    module = pipeline()
    sample = batch(counts)
    sample['gt_labeled'] = sample['gt_labeled'].clone()
    sample['gt_labeled'][..., 3] = False
    calls = []
    hook = module.slot_matcher.register_forward_hook(lambda *_: calls.append(1))
    saved = deepcopy(module.state_dict())
    loss = module._losses_for_batch(sample)
    hook.remove()
    assert len(calls) == 1 and torch.isfinite(loss['total'])
    assert loss['debug/positive_slots'] == sum(counts)
    assert 'aux0/latent/positive' in loss
    loss['total'].backward()
    assert module.model.latent_heads[0].weight.grad is not None
    assert module.model.transformer.decoder.layers[-1].cross_attn.sampling_offsets.weight.grad is not None
    restored = pipeline()
    restored.load_state_dict(saved, strict=True)
    module.load_state_dict(saved, strict=True)
    first = module._losses_for_batch(sample)
    second = restored._losses_for_batch(sample)
    torch.testing.assert_close(first['total'], second['total'], rtol=0, atol=0)
    assert first['debug/timestep_mean'] > 0
    invalid = batch((2,))
    for name in ('gt_boxes_pixel', 'gt_keypoints', 'gt_labeled', 'gt_box_labeled', 'person_mask'):
        repeats = [1] * invalid[name].ndim
        repeats[1] = 4
        invalid[name] = invalid[name].repeat(*repeats)
    with pytest.raises(ValueError, match='capacity'):
        module._losses_for_batch(invalid)


def test_train_mean_statistics_and_val_grid(tmp_path):
    from src.models.pipelines.pose_estimators.poseflow.statistics_v44 import MeanCodeMoments
    from src.models.pipelines.pose_estimators.poseflow.quick_eval_v44 import evaluate_v44_batches
    values = torch.randn(7, 8)
    moments = MeanCodeMoments(8)
    moments.update(values[:2])
    moments.update(values[2:])
    mean, std = moments.finalize()
    torch.testing.assert_close(mean, values.double().mean(0))
    torch.testing.assert_close(std, values.double().std(0, unbiased=False))
    network = model()
    sample = batch((2,))
    rng = torch.get_rng_state().clone()
    report = evaluate_v44_batches(network, [sample], tmp_path, 'unused.json', [0], steps=(1, 4),
        metric_evaluator=lambda *args: dict(pose_AP=1., bbox_AP=2.))
    assert torch.equal(rng, torch.get_rng_state())
    assert report['images'] == 1 and report['split'] == 'val'
    assert len(report['conditions']) == 2
    assert report['raw_confidence_max_difference'] == 0
    assert report['small_noise']['noise_5/input_latent_mse'] > 0


def test_val_transform_is_forced_even_when_trainer_is_training(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from src.data.datasets import PoseDataset
    from src.models.pipelines.pose_estimators.poseflow import quick_eval_v44
    sample = batch((2,))
    data = SimpleNamespace(val_ann_path='unused.json', img_dir='unused', eval_resize_scale=128,
        eval_max_size=128, trainer=SimpleNamespace(training=True))
    monkeypatch.setattr(PoseDataset, 'from_path', lambda *args, **kwargs: SimpleNamespace(samples=[dict(image_id=0)]))
    def batches(evaluation_data, *args):
        assert not evaluation_data.trainer.training
        assert evaluation_data is not data
        return [sample]
    monkeypatch.setattr(quick_eval_v44, 'v21_test_batches', batches)
    original = quick_eval_v44.evaluate_v44_batches
    monkeypatch.setattr(quick_eval_v44, 'evaluate_v44_batches', lambda *args, **kwargs:
        original(*args, **kwargs, metric_evaluator=lambda *_: dict(pose_AP=0., bbox_AP=0.)))
    report = quick_eval_v44.run_v44_evaluation(model(), data, tmp_path, epoch=-1, count=1, steps=(1,))
    assert report['images'] == 1 and data.trainer.training


def test_preset_uses_train_only_and_bounded_checkpoint_retention():
    from pathlib import Path
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(version_base=None, config_dir=str(Path('conf').resolve())):
        config = compose(config_name='benchmark_poseflow_v44')
    assert config.data.train_ann.endswith('crowdpose_train.json')
    assert config.data.test_ann == config.data.val_ann
    assert config.training.checkpoint.save_top_k == 1
    assert config.training.checkpoint.mode == 'max'
    assert config.training.max_epochs == 80 and config.training.accumulate_grad_batches == 2
    native = PoseFlowV44PipelineConfig.from_hydra_config(config.pipeline).native_config()
    assert (native.num_encoder_layers, native.num_box_decoder_layers, native.num_decoder_layers) == (3, 1, 3)

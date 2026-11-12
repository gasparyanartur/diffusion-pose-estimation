from copy import deepcopy
from pathlib import Path

import pytest
import torch


def tiny_model():
    from src.models.pipelines.pose_estimators.poseflow.model_v27 import PoseFlowV27Model
    return PoseFlowV27Model(backbone_pretrained=False, hidden_dim=32,
                           num_heads=4, feedforward_dim=64, num_proposals=5)


def memories(batch_size):
    return tuple((torch.randn(batch_size, count, 32), torch.randn(batch_size, count, 32),
                  torch.zeros(batch_size, count, dtype=torch.bool)) for count in (6, 24, 96))


def test_image_levels_route_coarse_to_fine_in_both_stacks():
    model = tiny_model().eval()
    image = torch.randn(2, 3, 64, 96)
    padding = torch.zeros(2, 64, 96, dtype=torch.bool)
    padding[0, :, 64:] = True
    with torch.no_grad():
        encoded = model.encode_image(image, padding)
    assert [level[0].shape[1] for level in encoded] == [6, 24, 96]
    assert all(level[2][0].any() and not level[2][1].any() for level in encoded)
    observed = []
    handles = [layer.image_attention.register_forward_pre_hook(
        lambda module, args: observed.append(args[1].shape[1]))
        for layer in list(model.clean_layers)+list(model.denoiser_layers)]
    with torch.no_grad():
        model.encode_clean(torch.rand(2, 5, 28)*60, encoded)
        model.denoise(torch.randn(2, 5, 32), torch.tensor([0, 999]), encoded)
    for handle in handles:
        handle.remove()
    assert observed == [6, 24, 96]*2


def test_fine_features_cannot_change_earlier_blocks():
    model = tiny_model().eval()
    encoded = memories(2)
    changed = list(encoded)
    changed[-1] = (encoded[-1][0]+10, *encoded[-1][1:])
    states = []
    handles = [layer.register_forward_hook(lambda module, args, output: states.append(output.clone()))
               for layer in model.denoiser_layers]
    noisy, times = torch.randn(2, 5, 32), torch.tensor([10, 500])
    with torch.no_grad():
        model.denoise(noisy, times, encoded)
        model.denoise(noisy, times, tuple(changed))
    for handle in handles:
        handle.remove()
    torch.testing.assert_close(states[0], states[3], atol=0, rtol=0)
    torch.testing.assert_close(states[1], states[4], atol=0, rtol=0)
    assert not torch.allclose(states[2], states[5])


@pytest.mark.parametrize('batch_size', [2, 4])
def test_scale_attention_permutation_padding_and_image_isolation(batch_size):
    model = tiny_model().eval()
    encoded = memories(batch_size)
    for _, _, mask in encoded:
        mask[:, -2:] = True
    changed = tuple((memory.clone(), position, mask) for memory, position, mask in encoded)
    for memory, _, _ in changed:
        memory[:, -2:] += 1000
        memory[1:] += 10
    state, times = torch.randn(batch_size, 5, 32), torch.arange(batch_size)*100
    permutation = torch.tensor([3, 1, 4, 0, 2])
    with torch.no_grad():
        full = model.denoise(state, times, encoded)
        independent = [model.denoise(state[i:i+1], times[i:i+1],
            tuple(tuple(value[i:i+1] for value in level) for level in encoded)) for i in range(batch_size)]
        permuted = model.denoise(state[:, permutation], times, encoded)
        isolated = model.denoise(state, times, changed)
    for key in ('pred_latents', 'pred_boxes', 'pred_keypoints', 'pred_logits'):
        torch.testing.assert_close(full[key], torch.cat([row[key] for row in independent]), atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(permuted[key], full[key][:, permutation], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(isolated[key][0], full[key][0], atol=0, rtol=0)


def test_latent_only_gradients_and_strict_pipeline_restore(tmp_path):
    from src.models.pipelines.pose_estimators.poseflow.training_v27 import PoseFlowV27PipelineConfig
    from scripts.evaluation.evaluate_poseflow import load_poseflow_model
    from omegaconf import OmegaConf

    config = PoseFlowV27PipelineConfig(backbone_pretrained=False, pretrained_checkpoint=None,
        hidden_dim=32, num_heads=4, feedforward_dim=64, num_proposals=5,
        num_queries=5, num_group=5, quick_eval_enabled=False)
    pipeline = config.build(max_epochs=2)
    batch = dict(image=torch.randn(2, 3, 64, 96), image_id=torch.arange(2),
        img_hw=torch.tensor([[64., 96.]]*2), gt_keypoints=torch.zeros(2, 1, 14, 2),
        gt_labeled=torch.ones(2, 1, 14, dtype=torch.bool), person_mask=torch.ones(2, 1, dtype=torch.bool),
        gt_boxes_canon=torch.tensor([[[.5, .5, .3, .5]]]*2))
    losses = pipeline._losses_for_batch(batch)
    assert not losses['denoised/total'].requires_grad
    torch.testing.assert_close(losses['total'], .5*losses['clean/total']+losses['loss_latent'], atol=0, rtol=0)
    source_grad = torch.autograd.grad(losses['loss_latent'], pipeline.model.input_mlp[0].weight,
                                     allow_unused=True, retain_graph=True)[0]
    assert source_grad is None
    losses['total'].backward()
    for stack in (pipeline.model.clean_layers, pipeline.model.denoiser_layers):
        for layer in stack:
            assert layer.image_attention.in_proj_weight.grad.abs().sum() > 0
    assert pipeline.model.joint_head.layers[-1].weight.grad.abs().sum() > 0
    other = config.build(max_epochs=2)
    other.load_state_dict(deepcopy(pipeline.state_dict()), strict=True)
    for left, right in zip(pipeline._draw_training_inputs(batch)[1:], other._draw_training_inputs(batch)[1:]):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    torch.save(dict(state_dict=pipeline.state_dict()), tmp_path/'model.ckpt')
    dumped = config.model_dump() | {'_target_':
        'src.models.pipelines.pose_estimators.poseflow.PoseFlowV27PipelineConfig',
        'pretrained_checkpoint':'/missing/initialization.pth'}
    OmegaConf.save(OmegaConf.create(dict(pipeline=dumped)), tmp_path/'config.yaml')
    restored, _ = load_poseflow_model(tmp_path/'model.ckpt', tmp_path/'config.yaml')
    def forbidden(*args, **kwargs):
        raise AssertionError('Gaussian inference accessed clean encoder')
    pipeline.model.encode_clean = forbidden
    pipeline.model.eval(); restored.eval()
    for calls in (1, 4):
        pipeline.model.flow_sampling_steps = restored.flow_sampling_steps = calls
        with torch.no_grad():
            actual = pipeline.model.sample_keypoints(batch['image'], generator=torch.Generator().manual_seed(42))
            expected = restored.sample_keypoints(batch['image'], generator=torch.Generator().manual_seed(42))
        for key in ('pred_latents', 'pred_keypoints', 'pred_boxes'):
            torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)


def test_explicit_image_level_subset_import(tmp_path):
    from src.models.pipelines.pose_estimators.poseflow.training_v26 import initialize_v26_image_weights
    model = tiny_model()
    source = {name:value.clone() for name,value in model.state_dict().items()
              if name.startswith(('backbone.', 'input_proj.'))}
    embedding = torch.randn(4, 32)
    source['transformer.level_embed'] = embedding
    path = tmp_path/'source.pt'
    torch.save(dict(model=source), path)
    with pytest.raises(ValueError, match='missing/incompatible'):
        initialize_v26_image_weights(model, path)
    report = initialize_v26_image_weights(model, path, feature_level_indices=(0, 1, 2))
    assert report['source_feature_levels'] == [0, 1, 2]
    assert report['image_tensor_coverage'] == 1.
    torch.testing.assert_close(model.level_embed, embedding[:3], atol=0, rtol=0)


def test_v27_recipe_and_fixed_feature_count():
    from hydra import compose, initialize_config_dir
    from src.models.pipelines.pose_estimators.poseflow.training import PoseFlowPipelineConfig
    from src.models.pipelines.pose_estimators.poseflow.training_v27 import PoseFlowV27PipelineConfig
    from omegaconf import OmegaConf

    with initialize_config_dir(config_dir=str(Path('conf').resolve()), version_base=None):
        v26 = compose(config_name='benchmark_poseflow_v26_latent_only')
        v27 = compose(config_name='benchmark_poseflow_v27')
    assert OmegaConf.to_container(v26.data) == OmegaConf.to_container(v27.data)
    assert OmegaConf.to_container(v26.training) == OmegaConf.to_container(v27.training)
    typed = PoseFlowPipelineConfig.from_hydra_config(v27.pipeline)
    assert isinstance(typed, PoseFlowV27PipelineConfig)
    assert typed.num_feature_levels == 3 and typed.denoised_task_loss_weight == 0
    assert typed.flow_sampling_steps == 4
    with pytest.raises(ValueError):
        PoseFlowV27PipelineConfig(num_feature_levels=4)

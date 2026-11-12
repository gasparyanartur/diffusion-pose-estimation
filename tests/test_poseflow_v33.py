from copy import deepcopy
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from tests.test_poseflow_v32 import FakeTeacher, make_batch


def make_pipeline(**kwargs):
    from src.models.pipelines.pose_estimators.poseflow.training_v33 import PoseFlowV33PipelineConfig

    config = PoseFlowV33PipelineConfig(backbone_pretrained=False, pretrained_checkpoint=None,
        hidden_dim=32, num_heads=4, feedforward_dim=64, num_queries=5,
        num_proposals=5, num_group=5, dino_grid_size=2, quick_eval_enabled=False, **kwargs)
    return config, config.build(max_epochs=2)


def test_grouped_readout_contract_permutation_and_no_box_coordinate_dependency():
    _, pipeline = make_pipeline()
    model = pipeline.model.eval()
    latent = torch.randn(2, 5, 32)
    with torch.no_grad():
        output = model.readout(latent)
        assert output['joint_xy'].shape == (2, 5, 14, 2)
        assert output['pred_boxes'].shape == (2, 5, 4)
        assert output['existence_logits'].shape == (2, 5)
        assert output['pred_latents'] is latent
        assert model.pose_decoder(latent).shape == (2, 5, 17, 32)
        assert len(model.pose_decoder.layers) == 3
        assert isinstance(model.joint_head, torch.nn.Linear)
        assert isinstance(model.box_head, torch.nn.Linear)
        permutation = torch.tensor([3, 0, 4, 1, 2])
        permuted = model.readout(latent[:, permutation])
        independent = model.readout(latent[:1, :1])
        changed = latent.clone(); changed[1] += 10; changed[0, 1:] -= 20
        isolated = model.readout(changed)
        for key in output:
            torch.testing.assert_close(permuted[key], output[key][:, permutation], atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(independent[key], output[key][:1, :1], atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(isolated[key][:1, :1], output[key][:1, :1], atol=1e-6, rtol=1e-5)
        model.box_head.bias.add_(100)
        after = model.readout(latent)
        assert not torch.equal(after['pred_boxes'], output['pred_boxes'])
        torch.testing.assert_close(after['joint_xy'], output['joint_xy'], atol=0, rtol=0)


def test_readout_initialization_preserves_common_tensors_and_global_rng():
    from src.models.pipelines.pose_estimators.poseflow.training_v32 import PoseFlowV32PipelineConfig

    torch.manual_seed(42)
    config, pipeline = make_pipeline()
    state = torch.get_rng_state().clone()
    torch.manual_seed(42)
    reference = PoseFlowV32PipelineConfig(**config.model_dump()).build(max_epochs=2)
    torch.testing.assert_close(state, torch.get_rng_state(), atol=0, rtol=0)
    for name, value in reference.model.state_dict().items():
        if not name.startswith(('joint_head.', 'box_head.', 'existence_head.')):
            torch.testing.assert_close(value, pipeline.model.state_dict()[name], atol=0, rtol=0)
    assert pipeline.model.joint_head.weight.shape == (2, 32)
    assert all(layer.dropout.p == 0 for layer in pipeline.model.pose_decoder.layers)
    pipeline.model.eval(); reference.model.eval()
    images = make_batch()['image']
    with torch.no_grad():
        before = reference.model.sample_keypoints(images, generator=torch.Generator().manual_seed(8))
        after = pipeline.model.sample_keypoints(images, generator=torch.Generator().manual_seed(8))
    torch.testing.assert_close(before['pred_latents'], after['pred_latents'], atol=0, rtol=0)
    torch.testing.assert_close(before['latent_state'], after['latent_state'], atol=0, rtol=0)


def test_structured_task_gradients_and_existing_latent_dino_routes():
    _, pipeline = make_pipeline()
    model = pipeline.model
    pipeline._dino_teacher = FakeTeacher()
    original = model.training_outputs
    observed = {}

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        observed['clean'] = result[0]['pred_latents']
        observed['clean'].retain_grad()
        assert not result[2].requires_grad
        return result

    model.training_outputs = capture
    losses = pipeline._losses_for_batch(make_batch())
    torch.testing.assert_close(losses['total'], .5*losses['clean/total']+
        .5*losses['denoised/total']+losses['loss_latent']+losses['loss_dino'], atol=0, rtol=0)
    losses['denoised/total'].backward(retain_graph=True)
    assert observed['clean'].grad is None
    assert (model.pose_decoder.roles.grad.abs().sum(-1) > 0).all()
    for name in ('pose_decoder', 'joint_head', 'box_head', 'existence_head',
                 'denoiser_layers', 'denoiser_projection', 'backbone'):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in getattr(model, name).parameters()), name
    assert all(p.grad is None for p in model.clean_layers.parameters())
    pipeline.zero_grad(set_to_none=True)
    losses['loss_latent'].backward(retain_graph=True)
    assert observed['clean'].grad is None
    assert all(p.grad is None for p in model.pose_decoder.parameters())
    pipeline.zero_grad(set_to_none=True)
    losses['loss_dino_clean'].backward(retain_graph=True)
    assert observed['clean'].grad.abs().sum() > 0
    assert all(p.grad is None for p in model.pose_decoder.parameters())
    pipeline.zero_grad(set_to_none=True)
    losses['total'].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_strict_restore_rng_and_teacher_free_gaussian_inference(tmp_path):
    from scripts.evaluation.evaluate_poseflow import load_poseflow_model

    config, pipeline = make_pipeline()
    pipeline._dino_teacher = FakeTeacher()
    batch = make_batch()
    pipeline._losses_for_batch(batch)
    restored = config.build(max_epochs=2)
    restored.load_state_dict(deepcopy(pipeline.state_dict()), strict=True)
    for before, after in zip(pipeline._draw_training_inputs(batch), restored._draw_training_inputs(batch)):
        if isinstance(before, dict):
            for key in before:
                torch.testing.assert_close(before[key], after[key], atol=0, rtol=0)
        else:
            torch.testing.assert_close(before, after, atol=0, rtol=0)
    torch.save(dict(state_dict=pipeline.state_dict()), tmp_path/'model.ckpt')
    saved = config.model_dump() | {'_target_':
        'src.models.pipelines.pose_estimators.poseflow.PoseFlowV33PipelineConfig',
        'pretrained_checkpoint':'/missing/image.pth', 'dino_teacher_model':'/missing/dino'}
    OmegaConf.save(OmegaConf.create(dict(pipeline=saved)), tmp_path/'config.yaml')
    model, _ = load_poseflow_model(tmp_path/'model.ckpt', tmp_path/'config.yaml')

    def forbidden(*args, **kwargs):
        raise AssertionError('Gaussian inference must not call clean encoding or DINO')

    model.encode_clean = forbidden
    model.dino_head.forward = forbidden
    model.eval(); pipeline.model.eval()
    with torch.no_grad():
        expected = pipeline.model.sample_keypoints(batch['image'], generator=torch.Generator().manual_seed(8))
        actual = model.sample_keypoints(batch['image'], generator=torch.Generator().manual_seed(8))
    torch.testing.assert_close(expected['pred_latents'], actual['pred_latents'], atol=0, rtol=0)
    torch.testing.assert_close(expected['joint_xy'], actual['joint_xy'], atol=0, rtol=0)


def test_v33_recipe_changes_only_family_and_namespace():
    with initialize_config_dir(config_dir=str(Path('conf').resolve()), version_base=None):
        before = compose(config_name='benchmark_poseflow_v32')
        after = compose(config_name='benchmark_poseflow_v33')
    first, second = (OmegaConf.to_container(c, resolve=True) for c in (before, after))
    for config in (first, second):
        config.pop('run_name'); config.pop('out_dir'); config['pipeline'].pop('_target_')
    assert first == second

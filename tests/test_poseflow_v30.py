from copy import deepcopy
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def make_pipeline():
    from src.models.pipelines.pose_estimators.poseflow.training_v30 import PoseFlowV30PipelineConfig

    config = PoseFlowV30PipelineConfig(backbone_pretrained=False, pretrained_checkpoint=None,
        hidden_dim=32, num_heads=4, feedforward_dim=64, num_proposals=5,
        num_queries=5, num_group=5, quick_eval_enabled=False)
    return config, config.build(max_epochs=2)


def make_batch():
    return dict(image=torch.randn(2, 3, 64, 96), image_id=torch.arange(2),
        img_hw=torch.tensor([[64., 96.]]*2), gt_keypoints=torch.zeros(2, 1, 14, 2),
        gt_labeled=torch.ones(2, 1, 14, dtype=torch.bool),
        person_mask=torch.ones(2, 1, dtype=torch.bool),
        gt_boxes_canon=torch.tensor([[[.5, .5, .3, .5]]]*2))


def test_recovered_task_updates_only_heads_and_latent_loss_keeps_original_routes():
    _, pipeline = make_pipeline()
    model = pipeline.model
    losses = pipeline._losses_for_batch(make_batch())
    torch.testing.assert_close(losses['total'], .5*losses['clean/total']+
        .5*losses['denoised/total']+losses['loss_latent'], atol=0, rtol=0)
    losses['denoised/total'].backward(retain_graph=True)
    heads = ('box_head', 'joint_head', 'existence_head')
    for name, parameter in model.named_parameters():
        if name.split('.')[0] in heads:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        else:
            assert parameter.grad is None, name
    for name in heads:
        assert any(p.grad.abs().sum() > 0 for p in getattr(model, name).parameters())
    pipeline.zero_grad(set_to_none=True)
    losses['loss_latent'].backward(retain_graph=True)
    for name in ('backbone', 'input_proj', 'denoiser_layers', 'denoiser_projection'):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in getattr(model, name).parameters())
    for name in (*heads, 'input_mlp', 'clean_layers'):
        assert all(p.grad is None for p in getattr(model, name).parameters())
    pipeline.zero_grad(set_to_none=True)
    losses['clean/total'].backward(retain_graph=True)
    assert all(p.grad is None for p in model.backbone.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.clean_layers.parameters())
    pipeline.zero_grad(set_to_none=True)
    losses['total'].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_detached_readout_preserves_values_and_strict_restore_rng(tmp_path):
    from scripts.evaluation.evaluate_poseflow import load_poseflow_model

    config, pipeline = make_pipeline()
    batch = make_batch()
    pipeline._losses_for_batch(batch)
    recovered = pipeline.model.readout(torch.randn(2, 5, 32, requires_grad=True))
    detached = pipeline._denoised_task_predictions(recovered)
    for key in recovered:
        torch.testing.assert_close(recovered[key], detached[key], atol=0, rtol=0)
    assert not detached['pred_latents'].requires_grad
    assert detached['joint_xy'].requires_grad
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
        'src.models.pipelines.pose_estimators.poseflow.PoseFlowV30PipelineConfig',
        'pretrained_checkpoint':'/missing/image.pth'}
    OmegaConf.save(OmegaConf.create(dict(pipeline=saved)), tmp_path/'config.yaml')
    model, _ = load_poseflow_model(tmp_path/'model.ckpt', tmp_path/'config.yaml')
    pipeline.model.eval(); model.eval()
    with torch.no_grad():
        expected = pipeline.model.sample_keypoints(batch['image'], generator=torch.Generator().manual_seed(8))
        actual = model.sample_keypoints(batch['image'], generator=torch.Generator().manual_seed(8))
    torch.testing.assert_close(expected['pred_latents'], actual['pred_latents'], atol=0, rtol=0)


def test_v30_recipe_changes_only_family_objective_and_namespace():
    with initialize_config_dir(config_dir=str(Path('conf').resolve()), version_base=None):
        before = compose(config_name='benchmark_poseflow_v28')
        after = compose(config_name='benchmark_poseflow_v30')
    first, second = (OmegaConf.to_container(c, resolve=True) for c in (before, after))
    assert second['pipeline']['denoised_task_loss_weight'] == .5
    for config in (first, second):
        config.pop('run_name'); config.pop('out_dir')
        config['pipeline'].pop('_target_'); config['pipeline'].pop('denoised_task_loss_weight')
    assert first == second

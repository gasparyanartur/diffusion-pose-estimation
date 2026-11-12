from copy import deepcopy
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def make_pipeline():
    from src.models.pipelines.pose_estimators.poseflow.training_v31 import PoseFlowV31PipelineConfig

    config = PoseFlowV31PipelineConfig(backbone_pretrained=False, pretrained_checkpoint=None,
        hidden_dim=32, num_heads=4, feedforward_dim=64, num_queries=5,
        num_proposals=5, num_group=5, quick_eval_enabled=False)
    return config, config.build(max_epochs=2)


def make_batch():
    return dict(image=torch.randn(2, 3, 64, 96), image_id=torch.arange(2),
        img_hw=torch.tensor([[64., 96.]]*2), gt_keypoints=torch.zeros(2, 1, 14, 2),
        gt_labeled=torch.ones(2, 1, 14, dtype=torch.bool),
        person_mask=torch.ones(2, 1, dtype=torch.bool),
        gt_boxes_canon=torch.tensor([[[.5, .5, .3, .5]]]*2))


def test_recovered_task_trains_denoiser_image_and_heads_but_not_clean_latents():
    _, pipeline = make_pipeline()
    model = pipeline.model
    observed = {}
    original = model.training_outputs

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        result[0]['pred_latents'].retain_grad()
        observed['clean'], observed['noisy'] = result[0]['pred_latents'], result[2]
        return result

    model.training_outputs = capture
    losses = pipeline._losses_for_batch(make_batch())
    assert not observed['noisy'].requires_grad
    torch.testing.assert_close(losses['total'], .5*losses['clean/total']+
        .5*losses['denoised/total']+losses['loss_latent'], atol=0, rtol=0)
    losses['denoised/total'].backward(retain_graph=True)
    assert observed['clean'].grad is None
    for name in ('backbone', 'input_proj', 'denoiser_layers', 'denoiser_projection',
                 'box_head', 'joint_head', 'existence_head'):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in getattr(model, name).parameters()), name
    assert model.level_embed.grad.abs().sum() > 0
    for name in ('input_mlp', 'clean_layers'):
        assert all(p.grad is None for p in getattr(model, name).parameters())
    pipeline.zero_grad(set_to_none=True)
    losses['loss_latent'].backward(retain_graph=True)
    assert observed['clean'].grad is None
    for name in ('input_mlp', 'clean_layers', 'box_head', 'joint_head', 'existence_head'):
        assert all(p.grad is None for p in getattr(model, name).parameters())
    pipeline.zero_grad(set_to_none=True)
    losses['clean/total'].backward()
    assert observed['clean'].grad.abs().sum() > 0
    assert all(p.grad is None for p in model.backbone.parameters())
    assert model.level_embed.grad is None


def test_v31_initialization_loss_and_strict_restore_match_live_v28_readout(tmp_path):
    from src.models.pipelines.pose_estimators.poseflow.training_v28 import PoseFlowV28PipelineConfig
    from scripts.evaluation.evaluate_poseflow import load_poseflow_model

    torch.manual_seed(42)
    config, pipeline = make_pipeline()
    torch.manual_seed(42)
    reference = PoseFlowV28PipelineConfig(**config.model_dump()).build(max_epochs=2)
    for name, value in pipeline.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[name], atol=0, rtol=0)
    batch = make_batch()
    losses, reference_losses = pipeline._losses_for_batch(batch), reference._losses_for_batch(batch)
    for name in losses:
        torch.testing.assert_close(losses[name], reference_losses[name], atol=0, rtol=0)
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
        'src.models.pipelines.pose_estimators.poseflow.PoseFlowV31PipelineConfig',
        'pretrained_checkpoint':'/missing/image.pth'}
    OmegaConf.save(OmegaConf.create(dict(pipeline=saved)), tmp_path/'config.yaml')
    model, _ = load_poseflow_model(tmp_path/'model.ckpt', tmp_path/'config.yaml')
    pipeline.model.eval(); model.eval()
    with torch.no_grad():
        expected = pipeline.model.sample_keypoints(batch['image'], generator=torch.Generator().manual_seed(8))
        actual = model.sample_keypoints(batch['image'], generator=torch.Generator().manual_seed(8))
    torch.testing.assert_close(expected['pred_latents'], actual['pred_latents'], atol=0, rtol=0)


def test_v31_recipe_preserves_v30_settings_except_family_and_namespace():
    with initialize_config_dir(config_dir=str(Path('conf').resolve()), version_base=None):
        before = compose(config_name='benchmark_poseflow_v30')
        after = compose(config_name='benchmark_poseflow_v31')
    first, second = (OmegaConf.to_container(c, resolve=True) for c in (before, after))
    assert second['pipeline']['denoised_task_loss_weight'] == .5
    for config in (first, second):
        config.pop('run_name'); config.pop('out_dir'); config['pipeline'].pop('_target_')
    assert first == second

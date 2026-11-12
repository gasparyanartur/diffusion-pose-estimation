from copy import deepcopy

import torch
from hydra import compose, initialize_config_dir
from pathlib import Path


def model_arguments():
    return dict(backbone_pretrained=False, hidden_dim=32, num_heads=4,
                feedforward_dim=64, num_proposals=5, num_feature_levels=4)


def test_v28_preserves_initialization_forward_and_image_dependence():
    from src.models.pipelines.pose_estimators.poseflow.model_v26 import PoseFlowV26Model
    from src.models.pipelines.pose_estimators.poseflow.model_v28 import PoseFlowV28Model

    torch.manual_seed(42)
    original = PoseFlowV26Model(**model_arguments()).eval()
    torch.manual_seed(42)
    model = PoseFlowV28Model(**model_arguments()).eval()
    for name, value in original.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[name], atol=0, rtol=0)
    images, pixels = torch.randn(2, 3, 64, 96), torch.rand(2, 5, 28)*60
    with torch.no_grad():
        encoded = model.encode_image(images)
        original_clean = original.encode_clean(pixels, encoded)
        clean = model.encode_clean(pixels, encoded)
        changed = model.encode_clean(pixels, (encoded[0]+1, encoded[1], encoded[2]))
        for key in clean:
            torch.testing.assert_close(clean[key], original_clean[key], atol=0, rtol=0)
        assert not torch.equal(clean['pred_latents'], changed['pred_latents'])
        for calls in (1, 4):
            original.flow_sampling_steps = model.flow_sampling_steps = calls
            before = original.sample_keypoints(images, generator=torch.Generator().manual_seed(12))
            after = model.sample_keypoints(images, generator=torch.Generator().manual_seed(12))
            torch.testing.assert_close(before['pred_latents'], after['pred_latents'], atol=0, rtol=0)


def test_clean_and_latent_losses_have_separate_image_gradient_routes():
    from src.models.pipelines.pose_estimators.poseflow.model_v28 import PoseFlowV28Model

    model = PoseFlowV28Model(**model_arguments()).train()
    clean, recovered, noisy, _ = model.training_outputs(torch.randn(2, 3, 64, 96),
        torch.rand(2, 5, 28)*60, timesteps=torch.tensor([0, 999]), noise=torch.randn(2, 5, 32))
    assert not noisy.requires_grad
    clean_loss = clean['joint_xy'].mean()+clean['pred_boxes'].mean()+clean['existence_logits'].mean()
    clean_loss.backward(retain_graph=True)
    assert all(p.grad is None for p in model.backbone.parameters())
    assert all(p.grad is None for p in model.input_proj.parameters())
    assert model.level_embed.grad is None
    for module in (model.input_mlp, model.clean_layers, model.box_head, model.joint_head, model.existence_head):
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters())
    model.zero_grad(set_to_none=True)
    latent_loss = (recovered['pred_latents']-clean['pred_latents'].detach()).square().mean()
    latent_loss.backward()
    for module in (model.backbone, model.input_proj, model.denoiser_layers, model.denoiser_projection):
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters())
    assert model.level_embed.grad.abs().sum()>0
    for module in (model.input_mlp, model.clean_layers, model.box_head, model.joint_head, model.existence_head):
        assert all(p.grad is None for p in module.parameters())


def test_v28_pipeline_restore_rng_and_latent_only_contract(tmp_path):
    from src.models.pipelines.pose_estimators.poseflow.training_v28 import PoseFlowV28PipelineConfig
    from src.models.pipelines.pose_estimators.poseflow.model_v28 import PoseFlowV28Model
    from scripts.evaluation.evaluate_poseflow import load_poseflow_model
    from omegaconf import OmegaConf

    config = PoseFlowV28PipelineConfig(**model_arguments(), pretrained_checkpoint=None,
        num_queries=5, num_group=5, quick_eval_enabled=False)
    pipeline = config.build(max_epochs=2)
    batch = dict(image=torch.randn(2, 3, 64, 96), image_id=torch.arange(2),
        img_hw=torch.tensor([[64., 96.]]*2), gt_keypoints=torch.zeros(2, 1, 14, 2),
        gt_labeled=torch.ones(2, 1, 14, dtype=torch.bool), person_mask=torch.ones(2, 1, dtype=torch.bool),
        gt_boxes_canon=torch.tensor([[[.5, .5, .3, .5]]]*2))
    losses = pipeline._losses_for_batch(batch)
    assert not losses['denoised/total'].requires_grad
    torch.testing.assert_close(losses['total'], .5*losses['clean/total']+losses['loss_latent'], atol=0, rtol=0)
    checkpoint = tmp_path/'model.ckpt'
    torch.save(dict(state_dict=pipeline.state_dict()), checkpoint)
    restored = config.build(max_epochs=2)
    restored.load_state_dict(deepcopy(pipeline.state_dict()), strict=True)
    for before, after in zip(pipeline._draw_training_inputs(batch), restored._draw_training_inputs(batch)):
        if isinstance(before, dict):
            for key in before:
                torch.testing.assert_close(before[key], after[key], atol=0, rtol=0)
        else:
            torch.testing.assert_close(before, after, atol=0, rtol=0)
    path = tmp_path/'config.yaml'
    saved = config.model_dump() | {'_target_':
        'src.models.pipelines.pose_estimators.poseflow.PoseFlowV28PipelineConfig',
        'pretrained_checkpoint':'/missing/initialization.pth'}
    OmegaConf.save(OmegaConf.create({'pipeline':saved}), path)
    model, _ = load_poseflow_model(checkpoint, path)
    assert isinstance(model, PoseFlowV28Model)
    model.eval();pipeline.model.eval()
    with torch.no_grad():
        expected = pipeline.model.sample_keypoints(batch['image'], generator=torch.Generator().manual_seed(8))
        actual = model.sample_keypoints(batch['image'], generator=torch.Generator().manual_seed(8))
    torch.testing.assert_close(expected['pred_latents'], actual['pred_latents'], atol=0, rtol=0)


def test_v28_recipe_matches_latent_only_except_family_and_namespace():
    with initialize_config_dir(config_dir=str(Path('conf').resolve()), version_base=None):
        before = compose(config_name='benchmark_poseflow_v26_latent_only')
        after = compose(config_name='benchmark_poseflow_v28')
    from omegaconf import OmegaConf
    first, second = (OmegaConf.to_container(c, resolve=True) for c in (before, after))
    for config in (first, second):
        config.pop('run_name');config.pop('out_dir');config['pipeline'].pop('_target_')
        # V26's loss coefficient is supplied at the benchmark level.
        config['pipeline']['denoised_task_loss_weight'] = float(config['pipeline'].get('denoised_task_loss_weight', 0.))
    assert first == second

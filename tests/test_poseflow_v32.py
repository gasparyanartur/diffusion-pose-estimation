from copy import deepcopy
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.models.pipelines.pose_estimators.poseflow.dino_box_teacher import BoxFeatureTargets


def make_batch():
    return dict(image=torch.randn(2, 3, 64, 96), image_id=torch.arange(2),
        img_hw=torch.tensor([[64., 96.]]*2), gt_keypoints=torch.zeros(2, 1, 14, 2),
        gt_labeled=torch.ones(2, 1, 14, dtype=torch.bool),
        person_mask=torch.ones(2, 1, dtype=torch.bool),
        gt_boxes_canon=torch.tensor([[[.5, .5, .3, .5]]]*2))


def make_pipeline(**kwargs):
    from src.models.pipelines.pose_estimators.poseflow.training_v32 import PoseFlowV32PipelineConfig

    config = PoseFlowV32PipelineConfig(backbone_pretrained=False, pretrained_checkpoint=None,
        hidden_dim=32, num_heads=4, feedforward_dim=64, num_queries=5,
        num_proposals=5, num_group=5, dino_grid_size=2, quick_eval_enabled=False, **kwargs)
    return config, config.build(max_epochs=2)


class FakeTeacher:
    def __init__(self):
        self.calls = 0
        self.last_targets = []

    def targets(self, images, image_hw, boxes):
        self.calls += 1
        self.last_targets = [BoxFeatureTargets(
            torch.randn(len(box), 4, 384, device=images.device, requires_grad=True),
            torch.ones(len(box), dtype=torch.bool, device=images.device)) for box in boxes]
        return self.last_targets


def test_person_correspondence_preserves_original_rows_duplicates_and_empty_images():
    from src.models.pipelines.pose_estimators.poseflow.training_v32 import dino_person_correspondence

    targets = dict(unique_people=torch.tensor([2, 0]),
        target_indices=torch.tensor([[3, 8, 8, -1, -1], [-1]*5]),
        positive=torch.tensor([[True, True, True, False, False], [False]*5]),
        boxes=torch.rand(2, 5, 4))
    boxes, matches = dino_person_correspondence(targets)
    torch.testing.assert_close(boxes[0], targets['boxes'][0, :2], atol=0, rtol=0)
    assert boxes[1].shape == (0, 4)
    assert matches[0][0].tolist() == [0, 1, 2]
    assert matches[0][1].tolist() == [0, 1, 1]
    assert not matches[1][0].numel() and not matches[1][1].numel()


def test_both_dino_routes_live_clean_image_gradients_and_detached_clean_source():
    _, pipeline = make_pipeline()
    model = pipeline.model
    teacher = FakeTeacher()
    pipeline._dino_teacher = teacher
    observed = {}
    grids = []
    original = model.training_outputs

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        result[0]['pred_latents'].retain_grad()
        observed['clean'], observed['noisy'] = result[0]['pred_latents'], result[2]
        return result

    def capture_grid(module, args, result):
        result.retain_grad()
        grids.append(result)

    model.training_outputs = capture
    handle = model.dino_head.register_forward_hook(capture_grid)
    losses = pipeline._losses_for_batch(make_batch())
    handle.remove()
    assert teacher.calls == 1 and len(grids) == 2
    assert not observed['noisy'].requires_grad
    torch.testing.assert_close(losses['loss_dino'],
        .5*(losses['loss_dino_clean']+losses['loss_dino_denoised']), atol=0, rtol=0)
    torch.testing.assert_close(losses['total'], .5*losses['clean/total']+
        .5*losses['denoised/total']+losses['loss_latent']+losses['loss_dino'], atol=0, rtol=0)

    losses['loss_dino_clean'].backward(retain_graph=True)
    assert observed['clean'].grad.abs().sum() > 0
    for name in ('backbone', 'input_proj', 'input_mlp', 'clean_layers', 'dino_head'):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in getattr(model, name).parameters()), name
    assert all(p.grad is None for p in model.denoiser_layers.parameters())
    assert grids[0].grad[:, 3:].count_nonzero() == 0
    pipeline.zero_grad(set_to_none=True); observed['clean'].grad = None

    losses['loss_dino_denoised'].backward(retain_graph=True)
    assert observed['clean'].grad is None
    for name in ('backbone', 'input_proj', 'denoiser_layers', 'denoiser_projection', 'dino_head'):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in getattr(model, name).parameters()), name
    for name in ('input_mlp', 'clean_layers', 'box_head', 'joint_head', 'existence_head'):
        assert all(p.grad is None for p in getattr(model, name).parameters()), name
    assert grids[1].grad[:, 3:].count_nonzero() == 0
    assert all(target.features.grad is None for target in teacher.last_targets)
    pipeline.zero_grad(set_to_none=True)
    losses['loss_latent'].backward(retain_graph=True)
    assert observed['clean'].grad is None
    assert all(p.grad is None for p in model.clean_layers.parameters())
    pipeline.zero_grad(set_to_none=True)
    losses['clean/total'].backward(retain_graph=True)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.backbone.parameters())
    pipeline.zero_grad(set_to_none=True)
    losses['total'].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_zero_weight_exact_base_loss_initialization_and_rng():
    from src.models.pipelines.pose_estimators.poseflow.training_v26 import PoseFlowV26PipelineConfig

    torch.manual_seed(42)
    config, pipeline = make_pipeline(dino_loss_weight=0.)
    state = torch.get_rng_state().clone()
    torch.manual_seed(42)
    reference = PoseFlowV26PipelineConfig(**config.model_dump()).build(max_epochs=2)
    torch.testing.assert_close(state, torch.get_rng_state(), atol=0, rtol=0)
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(value, pipeline.state_dict()[name], atol=0, rtol=0)
    teacher = FakeTeacher(); pipeline._dino_teacher = teacher
    batch = make_batch()
    expected, actual = reference._losses_for_batch(batch), pipeline._losses_for_batch(batch)
    assert teacher.calls == 0
    for name in expected:
        torch.testing.assert_close(expected[name], actual[name], atol=0, rtol=0)


def test_no_positive_slots_skips_teacher_and_restores_strictly_without_teacher(tmp_path):
    from scripts.evaluation.evaluate_poseflow import load_poseflow_model

    config, pipeline = make_pipeline()
    teacher = FakeTeacher(); pipeline._dino_teacher = teacher
    batch = make_batch()
    batch['v26_original_points'] = batch['gt_keypoints'].clone()
    batch['v26_original_labeled'] = batch['gt_labeled'].clone()
    batch['v26_original_people'] = batch['person_mask'].clone()
    batch['person_mask'].zero_(); batch['gt_labeled'].zero_()
    losses = pipeline._losses_for_batch(batch)
    assert losses['loss_dino'] == 0 and teacher.calls == 0
    assert not any('teacher' in name for name in pipeline.state_dict())
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
        'src.models.pipelines.pose_estimators.poseflow.PoseFlowV32PipelineConfig',
        'pretrained_checkpoint':'/missing/image.pth', 'dino_teacher_model':'/missing/dino'}
    OmegaConf.save(OmegaConf.create(dict(pipeline=saved)), tmp_path/'config.yaml')
    model, _ = load_poseflow_model(tmp_path/'model.ckpt', tmp_path/'config.yaml')
    model.eval(); pipeline.model.eval()
    with torch.no_grad():
        expected = pipeline.model.sample_keypoints(batch['image'], generator=torch.Generator().manual_seed(8))
        actual = model.sample_keypoints(batch['image'], generator=torch.Generator().manual_seed(8))
    torch.testing.assert_close(expected['pred_latents'], actual['pred_latents'], atol=0, rtol=0)


def test_v32_recipe_retains_task_and_sampling_contract():
    with initialize_config_dir(config_dir=str(Path('conf').resolve()), version_base=None):
        before = compose(config_name='benchmark_poseflow_v31')
        after = compose(config_name='benchmark_poseflow_v32')
    first, second = (OmegaConf.to_container(c, resolve=True) for c in (before, after))
    for config in (first, second):
        config.pop('run_name'); config.pop('out_dir'); config['pipeline'].pop('_target_')
    assert first == second

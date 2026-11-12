from copy import deepcopy
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from tests.test_poseflow_v26 import batch_fixture
from tests.test_poseflow_v32 import FakeTeacher


def make_pipeline():
    from src.models.pipelines.pose_estimators.poseflow.training_v34 import PoseFlowV34PipelineConfig

    config = PoseFlowV34PipelineConfig(backbone_pretrained=False, pretrained_checkpoint=None,
        hidden_dim=32, num_heads=4, feedforward_dim=64, num_queries=5,
        num_proposals=5, num_group=5, dino_grid_size=2, quick_eval_enabled=False)
    return config, config.build(max_epochs=2)


class SwappedMatcher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, output, targets):
        self.calls += 1
        return [(torch.tensor([0, 4]), torch.tensor([1, 0])) for _ in targets]


def test_one_assignment_controls_recovered_task_latent_and_dino_with_active_gradients():
    from src.models.pipelines.pose_estimators.poseflow.latent_v26 import v26_task_loss
    from src.models.pipelines.pose_estimators.poseflow.latent_v29 import build_v29_ota_targets
    from src.models.pipelines.pose_estimators.poseflow.training_v25 import matched_grid_loss

    _, pipeline = make_pipeline()
    pipeline.matcher = SwappedMatcher()
    teacher = FakeTeacher(); pipeline._dino_teacher = teacher
    observed = {}
    draw = pipeline._draw_training_inputs
    forward = pipeline.model.training_outputs

    def capture_inputs(*args, **kwargs):
        result = draw(*args, **kwargs)
        observed['targets'] = result[0]
        return result

    def capture_outputs(*args, **kwargs):
        result = forward(*args, **kwargs)
        observed['clean'], observed['recovered'] = result[:2]
        result[0]['pred_latents'].retain_grad()
        for key in ('pred_boxes', 'joint_xy', 'existence_logits'):
            result[1][key].retain_grad()
        assert not result[2].requires_grad
        return result

    pipeline._draw_training_inputs = capture_inputs
    pipeline.model.training_outputs = capture_outputs
    losses = pipeline._losses_for_batch(batch_fixture((2, 2)))
    assert pipeline.matcher.calls == 1 and teacher.calls == 1
    clean, recovered, targets = (observed[k] for k in ('clean', 'recovered', 'targets'))
    assigned, latent_targets, metadata = build_v29_ota_targets(clean, recovered, targets, SwappedMatcher())
    expected_task = v26_task_loss(recovered, assigned)
    expected_clean = v26_task_loss(clean, targets)
    assert assigned['positive'].tolist() == [[True, False, False, False, True]]*2
    assert metadata['matched_person'][:, [0, 4]].tolist() == [[1, 0]]*2
    for name, value in expected_task.items():
        torch.testing.assert_close(losses[f'denoised/{name}'], value, atol=0, rtol=0)
        torch.testing.assert_close(losses[f'clean/{name}'], expected_clean[name], atol=0, rtol=0)
    torch.testing.assert_close(losses['loss_latent'],
        (recovered['pred_latents']-latent_targets).square().mean(), atol=0, rtol=0)
    matches = [(torch.tensor([0, 4]), torch.tensor([1, 0]))]*2
    expected_dino = matched_grid_loss(pipeline.model.dino_head(recovered['pred_latents']),
                                     teacher.last_targets, matches)
    torch.testing.assert_close(losses['loss_dino_denoised'], expected_dino['loss_h_dino'], atol=0, rtol=0)
    torch.testing.assert_close(losses['total'], .5*losses['clean/total']+
        .5*losses['denoised/total']+losses['loss_latent']+losses['loss_dino'], atol=0, rtol=0)

    losses['denoised/total'].backward(retain_graph=True)
    unmatched = ~assigned['positive']
    assert recovered['pred_boxes'].grad[unmatched].count_nonzero() == 0
    assert recovered['joint_xy'].grad[unmatched].count_nonzero() == 0
    assert (recovered['existence_logits'].grad[unmatched] > 0).all()
    assert clean['pred_latents'].grad is None
    for name in ('backbone', 'denoiser_layers', 'pose_decoder', 'box_head', 'joint_head', 'existence_head'):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in getattr(pipeline.model, name).parameters()), name
    assert all(p.grad is None for p in pipeline.model.clean_layers.parameters())
    pipeline.zero_grad(set_to_none=True)
    losses['loss_latent'].backward(retain_graph=True)
    assert clean['pred_latents'].grad is None
    pipeline.zero_grad(set_to_none=True)
    losses['loss_dino_denoised'].backward(retain_graph=True)
    assert clean['pred_latents'].grad is None
    assert all(p.grad is None for p in pipeline.model.clean_layers.parameters())
    pipeline.zero_grad(set_to_none=True)
    losses['total'].backward()
    assert all(torch.isfinite(p.grad).all() for p in pipeline.parameters() if p.grad is not None)


@pytest.mark.parametrize('counts', [(1, 0), (0, 0)])
def test_empty_images_and_original_pose_fallback(counts):
    _, pipeline = make_pipeline()
    teacher = FakeTeacher(); pipeline._dino_teacher = teacher
    batch = batch_fixture(counts)
    if not any(counts):
        original = batch_fixture((1, 1))
        batch.update(v26_original_points=original['gt_keypoints'],
            v26_original_labeled=original['gt_labeled'], v26_original_people=original['person_mask'])
    losses = pipeline._losses_for_batch(batch)
    assert losses['denoised/total'].requires_grad
    assert pipeline._diagnostics['ota/unique_gt_coverage'] == (1 if any(counts) else 0)
    assert all(torch.isfinite(value) for value in losses.values())
    if not any(counts):
        assert teacher.calls == 0 and losses['loss_dino'] == 0
    losses['total'].backward()


def test_model_initialization_rng_strict_pipeline_restore_and_asset_free_evaluation(tmp_path):
    from src.models.pipelines.pose_estimators.poseflow.training_v33 import PoseFlowV33PipelineConfig
    from scripts.evaluation.evaluate_poseflow import load_poseflow_model

    torch.manual_seed(42)
    config, pipeline = make_pipeline()
    state = torch.get_rng_state().clone()
    arguments = config.model_dump(); arguments['matching_strategy'] = 'constructed'
    torch.manual_seed(42)
    reference = PoseFlowV33PipelineConfig(**arguments).build(max_epochs=2)
    torch.testing.assert_close(state, torch.get_rng_state(), atol=0, rtol=0)
    for name, value in reference.model.state_dict().items():
        torch.testing.assert_close(value, pipeline.model.state_dict()[name], atol=0, rtol=0)
    pipeline._dino_teacher = FakeTeacher()
    batch = batch_fixture()
    pipeline._losses_for_batch(batch)
    restored = config.build(max_epochs=2)
    restored.load_state_dict(deepcopy(pipeline.state_dict()), strict=True)
    assert any(name.startswith('matcher.') for name in restored.state_dict())
    for before, after in zip(pipeline._draw_training_inputs(batch), restored._draw_training_inputs(batch)):
        if isinstance(before, dict):
            for key in before:
                torch.testing.assert_close(before[key], after[key], atol=0, rtol=0)
        else:
            torch.testing.assert_close(before, after, atol=0, rtol=0)
    torch.save(dict(state_dict=pipeline.state_dict()), tmp_path/'model.ckpt')
    saved = config.model_dump() | {'_target_':
        'src.models.pipelines.pose_estimators.poseflow.PoseFlowV34PipelineConfig',
        'pretrained_checkpoint':'/missing/image.pth', 'dino_teacher_model':'/missing/dino'}
    OmegaConf.save(OmegaConf.create(dict(pipeline=saved)), tmp_path/'config.yaml')
    model, _ = load_poseflow_model(tmp_path/'model.ckpt', tmp_path/'config.yaml')
    model.eval(); pipeline.model.eval()
    with torch.no_grad():
        expected = pipeline.model.sample_keypoints(batch['image'], generator=torch.Generator().manual_seed(8))
        actual = model.sample_keypoints(batch['image'], generator=torch.Generator().manual_seed(8))
    torch.testing.assert_close(expected['pred_latents'], actual['pred_latents'], atol=0, rtol=0)
    torch.testing.assert_close(expected['joint_xy'], actual['joint_xy'], atol=0, rtol=0)


def test_recipe_changes_only_assignment_and_namespace():
    with initialize_config_dir(config_dir=str(Path('conf').resolve()), version_base=None):
        before = compose(config_name='benchmark_poseflow_v33')
        after = compose(config_name='benchmark_poseflow_v34')
    first, second = (OmegaConf.to_container(c, resolve=True) for c in (before, after))
    assert second['pipeline']['matching_strategy'] == 'ota'
    for config in (first, second):
        config.pop('run_name'); config.pop('out_dir'); config['pipeline'].pop('_target_')
        config['pipeline'].pop('matching_strategy')
    assert first == second

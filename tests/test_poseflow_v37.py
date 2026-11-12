from copy import deepcopy
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from tests.test_poseflow_v35 import batch, pipeline as v35_pipeline


def pipeline():
    from src.models.pipelines.pose_estimators.poseflow.training_v37 import PoseFlowV37PipelineConfig
    return PoseFlowV37PipelineConfig(backbone_pretrained=False, pretrained_checkpoint=None,
        hidden_dim=32, num_heads=4, feedforward_dim=64, num_queries=6, num_proposals=6,
        num_group=6, grid_rows=2, grid_columns=3, quick_eval_enabled=False).build(max_epochs=2)


def test_recipe_branches_from_spatial_with_latent_matching():
    with initialize_config_dir(config_dir=str(Path('conf').resolve()), version_base=None):
        config = compose(config_name='benchmark_poseflow_v37')
    assert config.pipeline.spatial_queries and config.pipeline.spatial_cost_weight == 0
    assert config.pipeline.matching_strategy == 'latent_ota'
    assert config.pipeline.denoised_task_loss_weight == .5
    assert config.pipeline.latent_loss_weight == 10
    assert config.pipeline.alignment_existence_weight == 4
    assert config.pipeline.num_proposals == 50 and config.training.max_epochs == 80


def test_distance_and_dynamic_k_use_only_latents():
    from src.models.pipelines.pose_estimators.poseflow.latent_v37 import latent_ota_assignment, pairwise_latent_distance
    clean = torch.tensor([[0., 0.], [2., 2.], [5., 5.]])
    recovered = clean.flip(0)
    distance = pairwise_latent_distance(recovered, clean)
    torch.testing.assert_close(distance, (recovered[:, None]-clean[None]).square().mean(-1))
    selected, counts = latent_ota_assignment(distance, torch.tensor([0, 1, -1]), candidate_topk=3)
    assert selected.tolist() == [2, 1, 0]
    expected = (-distance).exp().topk(3, dim=0).values.sum(0).long().clamp_min(1)
    torch.testing.assert_close(counts, expected)
    _, close_counts = latent_ota_assignment(torch.zeros(4, 4), torch.full((4,), -1), candidate_topk=3)
    assert close_counts.tolist() == [3]*4


def test_coverage_repair_duplicates_fakes_ties_and_full_occupancy():
    from src.models.pipelines.pose_estimators.poseflow.latent_v37 import latent_ota_assignment
    costs = torch.tensor([[1., 1., 9., 0.], [2., 2., 8., 0.],
                          [3., 3., 7., 0.], [4., 4., 6., 0.]])
    people = torch.tensor([0, 0, 1, -1])
    chosen, _ = latent_ota_assignment(costs, people)
    assigned_people = people[chosen]
    assert set(assigned_people.tolist()) == {-1, 0, 1}
    assert len(chosen) == 4
    for ids in (torch.full((4,), -1), torch.arange(4), people):
        first, _ = latent_ota_assignment(torch.zeros(4, 4), ids)
        second, _ = latent_ota_assignment(torch.zeros(4, 4), ids)
        torch.testing.assert_close(first, second, atol=0, rtol=0)
        assert set(ids[ids >= 0].tolist()) <= set(ids[first].tolist())


def test_one_assignment_controls_all_targets_and_ignores_decoded_predictions():
    from src.models.pipelines.pose_estimators.poseflow.latent_v35 import build_v35_targets
    from src.models.pipelines.pose_estimators.poseflow.latent_v37 import assign_latent_targets
    targets = build_v35_targets(batch(), proposals=6)
    latents = torch.eye(6)[None].repeat(2, 1, 1).requires_grad_()
    permutation = torch.tensor([5, 4, 3, 2, 1, 0])
    clean = dict(pred_latents=latents)
    recovered = dict(pred_latents=latents.detach()[:, permutation])
    assigned, latent_targets, metadata = assign_latent_targets(clean, recovered, targets)
    torch.testing.assert_close(latent_targets, latents[:, permutation])
    torch.testing.assert_close(assigned['positive'], targets['positive'][:, permutation])
    torch.testing.assert_close(assigned['boxes'], targets['boxes'][:, permutation])
    assert not latent_targets.requires_grad and metadata['unique_gt_coverage'] == 1
    recovered.update(pred_boxes=torch.randn(2, 6, 4)*100, existence_logits=torch.randn(2, 6)*100)
    other = assign_latent_targets(clean, recovered, targets)
    torch.testing.assert_close(other[1], latent_targets)
    assert metadata['fake_target_utilization'] == 1


def test_alignment_targets_and_balancing_include_background_geometry():
    from src.models.pipelines.pose_estimators.poseflow.latent_v37 import LatentMetricAlignment
    alignment = LatentMetricAlignment()
    targets = dict(positive=torch.tensor([[True, False, False]]),
        boxes=torch.tensor([[[.1, .1, .2, .2], [0., 0., 0., 0.], [0., 0., 0., 0.]]]),
        input_boxes=torch.tensor([[[.9, .9, .2, .2], [.1, .1, .2, .2], [.6, .1, .4, .2]]]))
    latents = torch.tensor([[[0., 0.], [0., 1.], [1., 0.]]], requires_grad=True)
    loss, metrics = alignment(latents, targets)
    rows, columns = torch.triu_indices(3, 3, offset=1)
    geometry = torch.where(targets['positive'][..., None], targets['boxes'], targets['input_boxes'])
    values = torch.stack(((-((geometry[0, rows, :2]-geometry[0, columns, :2]).square().sum(-1))/.25**2).exp(),
        (-((geometry[0, rows, 2:]-geometry[0, columns, 2:]).square().sum(-1))/.25**2).exp(),
        (targets['positive'][0, rows] == targets['positive'][0, columns]).float()), -1)
    distance = (latents[0, rows]-latents[0, columns]).square().mean(-1)
    element = torch.nn.functional.binary_cross_entropy_with_logits(-distance[:, None].expand_as(values), values, reduction='none')
    expected = (element[:2].mean(0)+element[2])/2
    torch.testing.assert_close(loss, (expected*torch.tensor([1., 1., 4.])).sum())
    loss.backward()
    assert latents.grad.abs().sum() > 0 and alignment.raw_scale.grad.abs().sum() > 0
    assert metrics['pairs/mixed'] == 2 and metrics['pairs/fake_fake'] == 1


@pytest.mark.parametrize('count', [1, 4])
def test_alignment_empty_groups_and_collapsed_codes_are_finite(count):
    from src.models.pipelines.pose_estimators.poseflow.latent_v37 import LatentMetricAlignment
    latents = torch.zeros(1, count, 4, requires_grad=True)
    targets = dict(positive=torch.zeros(1, count, dtype=torch.bool),
        boxes=torch.zeros(1, count, 4), input_boxes=torch.zeros(1, count, 4))
    loss, metrics = LatentMetricAlignment()(latents, targets)
    assert torch.isfinite(loss) and all(torch.isfinite(value) for value in metrics.values())
    loss.backward()
    assert torch.isfinite(latents.grad).all()


def test_model_rng_parity_gradient_routes_and_optimizer_membership():
    torch.manual_seed(42); baseline = v35_pipeline(True).eval()
    baseline_rng = torch.get_rng_state().clone()
    torch.manual_seed(42); module = pipeline().eval()
    torch.testing.assert_close(torch.get_rng_state(), baseline_rng, atol=0, rtol=0)
    for name, value in baseline.model.state_dict().items():
        torch.testing.assert_close(value, module.model.state_dict()[name], atol=0, rtol=0)
    data = batch()
    with torch.no_grad():
        left = baseline.model.sample_boxes(data['image'], generator=torch.Generator().manual_seed(8))
        right = module.model.sample_boxes(data['image'], generator=torch.Generator().manual_seed(8))
    for key in left:
        if isinstance(left[key], torch.Tensor):
            torch.testing.assert_close(left[key], right[key], atol=0, rtol=0)
    captured = {}
    original = module.model.training_outputs
    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        captured['clean'] = result[0]['pred_latents']; captured['clean'].retain_grad()
        assert not result[2].requires_grad
        return result
    module.model.training_outputs = capture
    losses = module._losses_for_batch(data)
    torch.testing.assert_close(losses['total'], .5*losses['clean/total']+.5*losses['denoised/total']+10*losses['loss_latent']+losses['loss_alignment'])
    losses['loss_latent'].backward(retain_graph=True)
    assert captured['clean'].grad is None
    assert all(p.grad is None for p in module.model.clean_layers.parameters())
    assert all(p.grad is None for p in module.model.box_head.parameters())
    assert module.model.query_projection.weight.grad.abs().sum() > 0
    module.zero_grad(); losses['loss_alignment'].backward(retain_graph=True)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.model.clean_layers.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.model.backbone.parameters())
    optimizer = module.configure_optimizers()['optimizer']
    parameters = {id(p) for group in optimizer.param_groups for p in group['params']}
    assert all(id(p) in parameters for p in module.alignment.parameters())


def test_restore_pipeline_optimizer_next_draws_and_empty_people():
    module = pipeline().eval()
    losses = module._losses_for_batch(batch((0, 0)))
    assert all(torch.isfinite(value) for value in losses.values())
    optimizer = module.configure_optimizers()['optimizer']
    losses['total'].backward(); optimizer.step()
    restored = pipeline().eval(); restored.load_state_dict(deepcopy(module.state_dict()), strict=True)
    other_optimizer = restored.configure_optimizers()['optimizer']
    other_optimizer.load_state_dict(deepcopy(optimizer.state_dict()))
    for original, actual in zip(module.alignment.parameters(), restored.alignment.parameters()):
        for key in ('step', 'exp_avg', 'exp_avg_sq'):
            torch.testing.assert_close(optimizer.state[original][key], other_optimizer.state[actual][key], atol=0, rtol=0)
    for name, value in module.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], atol=0, rtol=0)
    left = module._draw_training_inputs(batch()); right = restored._draw_training_inputs(batch())
    for name in left[0]:
        torch.testing.assert_close(left[0][name], right[0][name], atol=0, rtol=0)
    torch.testing.assert_close(left[1], right[1]); torch.testing.assert_close(left[2], right[2])


def test_alignment_permutation_invariance_and_strict_model_only_loading(tmp_path):
    from omegaconf import OmegaConf
    from scripts.evaluation.evaluate_poseflow import load_poseflow_model
    from src.models.pipelines.pose_estimators.poseflow.latent_v35 import build_v35_targets
    module = pipeline().eval()
    targets = build_v35_targets(batch(), proposals=6)
    latents = torch.randn(2, 6, 32)
    permutation = torch.tensor([5, 3, 1, 0, 2, 4])
    permuted = {name: value[:, permutation] for name, value in targets.items()
                if name in ('positive', 'boxes', 'input_boxes')}
    left, _ = module.alignment(latents, targets)
    right, _ = module.alignment(latents[:, permutation], permuted)
    torch.testing.assert_close(left, right)
    config_path = tmp_path/'config.yaml'
    config = module.v26_config.model_dump()
    config['_target_'] = 'src.models.pipelines.pose_estimators.poseflow.PoseFlowV37PipelineConfig'
    OmegaConf.save(OmegaConf.create(dict(pipeline=config)), config_path)
    checkpoint = tmp_path/'checkpoint.ckpt'
    torch.save(dict(state_dict=module.state_dict()), checkpoint)
    model, _ = load_poseflow_model(checkpoint, config_path)
    for name, value in module.model.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[name], atol=0, rtol=0)

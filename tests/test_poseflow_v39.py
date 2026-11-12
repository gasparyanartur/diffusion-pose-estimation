from copy import deepcopy

import pytest
import torch

from tests.test_poseflow_v35 import batch


def pipeline():
    from src.models.pipelines.pose_estimators.poseflow.training_v39 import PoseFlowV39PipelineConfig
    return PoseFlowV39PipelineConfig(backbone_pretrained=False, pretrained_checkpoint=None,
        hidden_dim=32, num_heads=4, feedforward_dim=64, num_queries=6, num_proposals=6,
        num_group=6, grid_rows=2, grid_columns=3, quick_eval_enabled=False).build(max_epochs=2)


def test_unique_people_and_permuted_provenance_without_matching(monkeypatch):
    from src.models.pipelines.pose_estimators.poseflow.latent_fixed_source import build_fixed_source_targets

    def forbidden(*args, **kwargs):
        raise AssertionError('Fixed-source supervision must not call a matcher')

    monkeypatch.setattr('src.models.pipelines.pose_estimators.poseflow.latent_v35.linear_sum_assignment', forbidden)
    monkeypatch.setattr('src.models.pipelines.pose_estimators.poseflow.latent_v35.BoxOTAMatcher.__init__', forbidden)
    data = batch()
    targets = build_fixed_source_targets(data, proposals=6, generator=torch.Generator().manual_seed(3))
    assert targets['positive'].sum(-1).tolist() == [2, 1]
    for image, count in enumerate((2, 1)):
        positive = targets['positive'][image]
        assert targets['target_indices'][image, positive].sort().values.tolist() == list(range(count))
        slots = targets['canonical_slots'][image, :count]
        assert targets['target_indices'][image, slots].tolist() == list(range(count))
        torch.testing.assert_close(targets['boxes'][image, slots], data['gt_boxes_canon'][image, :count])
        assert (targets['source_image_ids'][image, ~positive] != image).all()
        donors = targets['source_batch'][image]
        people = targets['source_person'][image]
        torch.testing.assert_close(targets['input_boxes'][image], data['gt_boxes_canon'][donors, people])
        assert targets['slot_to_original'][image].sort().values.tolist() == list(range(6))
    module = pipeline()
    assert not hasattr(module, 'matcher') or isinstance(module.matcher, torch.nn.Identity)
    module._losses_for_batch(data)


def test_empty_full_invalid_and_no_foreign_donor():
    from src.models.pipelines.pose_estimators.poseflow.latent_fixed_source import build_fixed_source_targets
    for counts, proposals in [((0,), 6), ((1,), 6), ((2,), 2)]:
        targets = build_fixed_source_targets(batch(counts), proposals=proposals,
            generator=torch.Generator().manual_seed(5))
        assert targets['positive'].sum() == counts[0]
        fake = ~targets['positive']
        corners = torch.cat((targets['input_boxes'][..., :2]-targets['input_boxes'][..., 2:]/2,
                             targets['input_boxes'][..., :2]+targets['input_boxes'][..., 2:]/2), -1)
        assert (corners[fake] >= 0).all() and (corners[fake] <= 1).all()
        assert (targets['input_boxes'][..., 2:][fake] > 0).all()
    invalid = batch()
    invalid['gt_boxes_canon'][0, 1, 0] = float('nan')
    invalid['gt_box_labeled'][1, 0] = False
    assert build_fixed_source_targets(invalid)['positive'].sum(-1).tolist() == [1, 0]
    with pytest.raises(ValueError, match='budget'):
        build_fixed_source_targets(batch(), proposals=1)
    same_image = batch()
    same_image['image_id'].zero_()
    targets = build_fixed_source_targets(same_image, proposals=6)
    assert targets['random_negative'][~targets['positive']].all()


def test_only_positive_latent_targets_are_reconstructed_and_detached():
    from src.models.pipelines.pose_estimators.poseflow.latent_fixed_source import positive_latent_mse
    prediction = torch.tensor([[[1., 3.], [90., 80.]]], requires_grad=True)
    target = torch.zeros_like(prediction, requires_grad=True)
    positive = torch.tensor([[True, False]])
    loss = positive_latent_mse(prediction, target, positive)
    torch.testing.assert_close(loss, torch.tensor(5.))
    loss.backward()
    assert target.grad is None
    assert prediction.grad[0, 0].abs().sum() > 0
    assert prediction.grad[0, 1].abs().sum() == 0
    prediction.grad = None
    empty_loss = positive_latent_mse(prediction, target, torch.zeros_like(positive))
    empty_loss.backward()
    assert empty_loss == 0 and prediction.grad.abs().sum() == 0


def test_original_pairing_controls_geometry_and_fake_classification():
    from src.models.pipelines.pose_estimators.poseflow.latent_v35 import box_task_loss
    boxes = torch.tensor([[[.2, .3, .1, .2], [.8, .7, .2, .3], [.5, .5, .2, .2]]], requires_grad=True)
    logits = torch.zeros(1, 3, requires_grad=True)
    targets = dict(positive=torch.tensor([[True, True, False]]), boxes=boxes.detach().clone())
    original = box_task_loss(dict(pred_boxes=boxes, existence_logits=logits), targets)
    swapped = box_task_loss(dict(pred_boxes=boxes[:, [1, 0, 2]], existence_logits=logits), targets)
    assert original['loss_bbox'] == 0 and swapped['loss_bbox'] > 0
    original['total'].backward()
    assert boxes.grad[0, 2].abs().sum() == 0 and logits.grad[0, 2] != 0


def test_objective_gradient_routes_and_strict_restore():
    module = pipeline().eval()
    data = batch()
    clean_latents = []
    hook = module.model.clean_norm.register_forward_hook(lambda _, args, output: clean_latents.append(output))
    losses = module._losses_for_batch(data)
    clean_latents[0].retain_grad()
    torch.testing.assert_close(losses['total'], .5*losses['clean/total']+.5*losses['denoised/total']+10*losses['loss_latent'])
    losses['loss_latent'].backward(retain_graph=True)
    assert clean_latents[0].grad is None
    assert all(p.grad is None for p in module.model.clean_layers.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.model.denoiser_layers.parameters())
    module.zero_grad()
    losses['total'].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.model.clean_layers.parameters())
    hook.remove()
    restored = pipeline().eval()
    restored.load_state_dict(deepcopy(module.state_dict()), strict=True)
    rng = module.training_rng_state.clone()
    module._draw_training_inputs(data, training=False, batch_idx=8)
    assert torch.equal(rng, module.training_rng_state)
    left, right = module._draw_training_inputs(data), restored._draw_training_inputs(data)
    for name in left[0]:
        torch.testing.assert_close(left[0][name], right[0][name], atol=0, rtol=0)
    torch.testing.assert_close(left[1], right[1], atol=0, rtol=0)
    torch.testing.assert_close(left[2], right[2], atol=0, rtol=0)
    with torch.no_grad():
        for steps in (1, 4):
            restored.model.flow_sampling_steps = steps
            output = restored.model.sample_boxes(data['image'], generator=torch.Generator().manual_seed(3))
            assert output['pred_boxes'].shape == (2, 6, 4)
            assert torch.isfinite(output['pred_boxes']).all() and torch.isfinite(output['existence_logits']).all()


def test_empty_batch_finite_loss_and_noise_metrics_use_positive_counts():
    module = pipeline()
    losses = module._losses_for_batch(batch((0,)))
    losses['total'].backward()
    assert losses['loss_latent'] == 0 and torch.isfinite(losses['total'])
    from src.models.pipelines.pose_estimators.poseflow.training_v39 import fixed_source_noise_metrics
    positive = torch.tensor([[True, False, False], [True, True, False], [False, False, False]])
    targets = dict(positive=positive, boxes=torch.zeros(3, 3, 4))
    output = dict(pred_boxes=torch.ones(3, 3, 4), existence_logits=torch.zeros(3, 3))
    metrics = fixed_source_noise_metrics(output, targets, torch.ones(3, 3), torch.tensor([0, 1, 999]), 1000)
    assert metrics['bbox_l1/low'][1] == 3
    assert metrics['latent_mse/low'][1] == 3
    assert 'bbox_l1/high' not in metrics
    assert metrics['fake_score/high'][1] == 3

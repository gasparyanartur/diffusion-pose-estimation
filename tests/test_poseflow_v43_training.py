from copy import deepcopy

import pytest
import torch

from src.models.pipelines.pose_estimators.poseflow.training_v43 import (
    PoseFlowV43Pipeline, PoseFlowV43PipelineConfig, assign_latent_sources,
    balanced_latent_loss, geometry_targets,
)
from src.models.pipelines.pose_estimators.poseflow.latent_v43 import reference_statistics
from src.models.pipelines.pose_estimators.diffusionregpose.training import DiffusionRegPoseMatcher
from tests.test_poseflow_v43 import model


def batch(counts=(2, 1)):
    images = len(counts)
    boxes = torch.tensor([[[20., 20., 30., 50.], [75., 70., 20., 30.]]]*images)
    mask = torch.arange(2)[None] < torch.tensor(counts)[:, None]
    points = (boxes[..., :2]+boxes[..., 2:]/2)[:, :, None].expand(-1, -1, 14, -1)/128*2-1
    return dict(image=torch.randn(images, 3, 128, 128), img_hw=torch.tensor([[128., 128.]]*images),
        gt_boxes_pixel=boxes, gt_keypoints=points, gt_labeled=mask[..., None].expand(-1, -1, 14),
        gt_box_labeled=mask, person_mask=mask, image_id=torch.arange(images))


def pipeline():
    config = PoseFlowV43PipelineConfig(hidden_dim=32, num_heads=4, feedforward_dim=64,
        num_proposals=6, reference_candidates=12, reference_encoder_source_layers=(2,),
        codec_width=32, latent_dim=8, backbone_pretrained=False, pretrained_checkpoint=None,
        quick_eval_enabled=False)
    return PoseFlowV43Pipeline(model=model(), config=config, max_epochs=2).eval()


def test_reference_diagnostics_distinguish_candidate_coverage_from_selected_coverage():
    boxes = torch.tensor([[[.2, .3, .1, .2], [.8, .7, .2, .3]]])
    candidates = dict(pred_boxes=boxes.requires_grad_())
    references = dict(boxes=boxes[:, :1])
    targets = [dict(boxes=boxes[0].detach())]
    result = reference_statistics(candidates, references, targets,
        [(torch.tensor([0]), torch.tensor([0]))])
    assert result['candidate_recall_iou50'] == 1
    assert result['selected_recall_iou50'] == .5
    assert result['selected_matched_iou'] == 1 and result['selected_matched_l1'] == 0
    assert not any(value.requires_grad for value in result.values())


def test_complete_source_identity_background_null_and_positive_background_reductions():
    codec = model().codec
    geometry, targets = geometry_targets(batch())
    matches = [(torch.tensor([3, 0]), torch.tensor([1, 0])), (torch.tensor([4]), torch.tensor([0]))]
    sources = assign_latent_sources(codec, geometry, targets, matches, slots=6,
        generator=torch.Generator().manual_seed(42))
    codes = codec.sample_latents(geometry, generator=torch.Generator().manual_seed(42))
    torch.testing.assert_close(sources['latents'][0, 3], codes[1])
    torch.testing.assert_close(sources['latents'][0, 0], codes[0])
    torch.testing.assert_close(sources['latents'][1, 4], codes[2])
    assert sources['target_indices'][0, 3] == 1
    assert not sources['latents'][~sources['positive']].any()
    predicted = sources['latents'].detach().clone().requires_grad_()
    predicted.data[sources['positive']] += 2
    predicted.data[~sources['positive']] += 3
    loss = balanced_latent_loss(predicted, sources['latents'], sources['positive'])
    assert loss['positive'] == 4 and loss['background'] == 9 and loss['total'] == 13
    loss['total'].backward()
    assert torch.isfinite(predicted.grad).all()


def test_fixed_assignments_objective_and_geometry_only_gradient_routes():
    module = pipeline()
    matcher_calls = []
    hook = module.matcher.register_forward_hook(lambda *_: matcher_calls.append(1))
    saved = deepcopy(module.model.codec.state_dict())
    losses = module._losses_for_batch(batch())
    hook.remove()
    assert len(matcher_calls) == 2
    torch.testing.assert_close(losses['total'], losses['proposal/total']+2*losses['loss_slot_focal']
                               +10*losses['loss_latent']+.5*losses['geometry/total'])
    losses['geometry/total'].backward()
    assert module.model.latent_output.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in module.model.codec.parameters())
    assert module.model.slot_confidence_head.weight.grad is None
    for key, value in saved.items():
        torch.testing.assert_close(value, module.model.codec.state_dict()[key], rtol=0, atol=0)
    optimizer = module.configure_optimizers()['optimizer']
    selected = {id(p) for group in optimizer.param_groups for p in group['params']}
    assert not any(id(p) in selected for p in module.model.codec.parameters())


def test_empty_targets_capacity_and_validation_rng():
    module = pipeline()
    losses = module._losses_for_batch(batch((0,)))
    assert losses['geometry/total'] == 0 and losses['latent/positive'] == 0
    assert torch.isfinite(losses['total'])
    losses['total'].backward()
    state = (module.training_rng_state.clone(), module.posterior_rng_state.clone())
    left = module._losses_for_batch(batch((1,)), training=False)
    assert torch.equal(module.training_rng_state, state[0]) and torch.equal(module.posterior_rng_state, state[1])
    geometry, targets = geometry_targets(batch((2,)))
    with pytest.raises(ValueError, match='capacity|budget'):
        assign_latent_sources(module.model.codec, geometry, targets,
            [(torch.tensor([0]), torch.tensor([0]))], slots=1, generator=torch.Generator())


def test_model_optimizer_scheduler_and_next_update_restore_without_assets():
    original = pipeline()
    data = batch((1,))
    optimizer = original.configure_optimizers()
    optimizer['optimizer'].zero_grad()
    original._losses_for_batch(data)['total'].backward()
    torch.nn.utils.clip_grad_norm_(original.parameters(), .1)
    optimizer['optimizer'].step()
    optimizer['lr_scheduler']['scheduler'].step()
    restored = pipeline()
    restored.load_state_dict(deepcopy(original.state_dict()), strict=True)
    configuration = restored.configure_optimizers()
    configuration['optimizer'].load_state_dict(deepcopy(optimizer['optimizer'].state_dict()))
    configuration['lr_scheduler']['scheduler'].load_state_dict(deepcopy(optimizer['lr_scheduler']['scheduler'].state_dict()))
    for module, setup in ((original, optimizer), (restored, configuration)):
        setup['optimizer'].zero_grad()
        module._losses_for_batch(data)['total'].backward()
        torch.nn.utils.clip_grad_norm_(module.parameters(), .1)
        setup['optimizer'].step()
    for key, value in original.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)

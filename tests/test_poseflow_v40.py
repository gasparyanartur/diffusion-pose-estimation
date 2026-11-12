from copy import deepcopy

import pytest
import torch

from tests.test_poseflow_v35 import batch


def pipeline(slots=6):
    from src.models.pipelines.pose_estimators.poseflow.training_v40 import PoseFlowV40PipelineConfig
    return PoseFlowV40PipelineConfig(backbone_pretrained=False, pretrained_checkpoint=None,
        hidden_dim=32, num_heads=4, feedforward_dim=64, num_queries=slots,
        num_proposals=slots, num_group=slots, grid_rows=1, grid_columns=slots,
        reference_candidates=12, reference_encoder_source_layers=(2,),
        quick_eval_enabled=False).build(max_epochs=2)


def test_assignment_retains_low_rank_matches_orders_and_preserves_live_references():
    from src.models.pipelines.pose_estimators.poseflow.latent_v40 import select_training_references
    module = pipeline(slots=2)
    data = batch((2,))
    boxes = torch.tensor([[[.5, .5, .9, .9], [.5, .5, .8, .8],
                           [.2, .3, .1, .2], [.8, .7, .2, .3]]], requires_grad=True)
    features = torch.randn(1, 4, 32, requires_grad=True)
    logits = torch.tensor([[0., -1., -2., -3.]])
    candidates = dict(pred_boxes=boxes, features=features, existence_logits=logits,
        pred_logits=torch.stack((logits-20, logits), -1))
    references, targets, proposal_targets = select_training_references(
        candidates, data, module.matcher, slots=2)
    assert references['indices'].tolist() == [[2, 3]]
    torch.testing.assert_close(targets['boxes'], data['gt_boxes_canon'])
    assert targets['positive'].all() and proposal_targets['positive'].tolist() == [[False, False, True, True]]
    torch.testing.assert_close(targets['input_pixels'], targets['boxes']*128)
    (references['features'].sum()+references['boxes'].sum()).backward()
    assert features.grad[0, 2:].abs().sum() > 0 and boxes.grad[0, 2:].abs().sum() > 0
    assert features.grad[0, :2].abs().sum() == 0


def test_empty_full_invalid_targets_and_no_post_denoise_matching(monkeypatch):
    module = pipeline(slots=2).eval()
    for counts in ((0,), (1,), (2,)):
        calls = []
        handle = module.matcher.register_forward_hook(lambda *args: calls.append(1))
        losses = module._losses_for_batch(batch(counts))
        handle.remove()
        assert len(calls) == 1
        losses['total'].backward()
        assert torch.isfinite(losses['total'])
        if counts == (0,):
            for key in ('loss_latent', 'clean/loss_bbox', 'denoised/loss_bbox', 'proposal/loss_bbox'):
                assert losses[key] == 0
        module.zero_grad(set_to_none=True)
    data = batch((2,))
    data['gt_boxes_canon'][0, 0, 0] = float('nan')
    data['gt_box_labeled'][0, 1] = False
    losses = module._losses_for_batch(data)
    assert losses['loss_latent'] == 0 and torch.isfinite(losses['total'])
    with pytest.raises(ValueError, match='budget'):
        pipeline(slots=1)._losses_for_batch(batch((2,)))


def test_objective_masks_detaches_and_live_reference_gradient_routes():
    module = pipeline().eval()
    for layer in module.model.denoiser_layers:
        torch.nn.init.normal_(layer.modulation.weight, std=.02)
    captured = {}
    clean_hook = module.model.clean_norm.register_forward_hook(
        lambda _, args, output: captured.update(clean=output))
    original = module.model.propose
    def propose(*args, **kwargs):
        output = original(*args, **kwargs)
        output['features'].retain_grad()
        output['pred_boxes'].retain_grad()
        captured.update(proposals=output)
        return output
    module.model.propose = propose
    losses = module._losses_for_batch(batch())
    clean_hook.remove()
    captured['clean'].retain_grad()
    expected = (losses['proposal/total']+.5*losses['clean/total']
                +.5*losses['denoised/total']+10*losses['loss_latent'])
    torch.testing.assert_close(losses['total'], expected)
    losses['loss_latent'].backward(retain_graph=True)
    assert captured['clean'].grad is None
    assert all(p.grad is None for p in module.model.clean_layers.parameters())
    assert captured['proposals']['features'].grad.abs().sum() > 0
    assert captured['proposals']['pred_boxes'].grad.abs().sum() > 0
    module.zero_grad(set_to_none=True)
    losses['total'].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.model.clean_layers.parameters())
    from src.models.pipelines.pose_estimators.poseflow.latent_fixed_source import positive_latent_mse
    predicted = torch.randn(1, 3, 4, requires_grad=True)
    target = torch.randn_like(predicted, requires_grad=True)
    positive_latent_mse(predicted, target, torch.tensor([[True, False, False]])).backward()
    assert target.grad is None and predicted.grad[:, 1:].abs().sum() == 0


def test_cached_proposal_stage_and_fixed_reference_identity_across_steps():
    module = pipeline().eval()
    calls, references = [], []
    original_propose, original_denoise = module.model.propose, module.model.denoise
    def propose(*args, **kwargs):
        calls.append(1)
        return original_propose(*args, **kwargs)
    def denoise(*args, **kwargs):
        references.append(kwargs['references'])
        return original_denoise(*args, **kwargs)
    module.model.propose, module.model.denoise = propose, denoise
    with torch.no_grad():
        module.model.flow_sampling_steps = 4
        output = module.model.sample_boxes(batch()['image'], generator=torch.Generator().manual_seed(4))
    assert len(calls) == 1 and len(references) == 4
    assert all(item is references[0] for item in references)
    assert output['pred_boxes'].shape == (2, 6, 4)
    torch.testing.assert_close(output['pred_logits'][..., 1], output['existence_logits'])


def test_batch_isolation_and_reference_conditioning_changes_output():
    model = pipeline().model.eval()
    for layer in model.denoiser_layers:
        torch.nn.init.normal_(layer.modulation.weight, std=.02)
    data = batch()
    with torch.no_grad():
        context = model.encode_context(data['image'])
        candidates = model.propose(context)
        references = model.select_references(candidates)
        noise = torch.randn(2, 6, 32)
        times = torch.tensor([0, 999])
        output = model.denoise(noise, times, context['image'], references=references)
        altered = {k: v.clone() for k, v in references.items()}
        altered['features'][1].add_(5)
        changed = model.denoise(noise, times, context['image'], references=altered)
        torch.testing.assert_close(output['pred_boxes'][0], changed['pred_boxes'][0], atol=0, rtol=0)
        assert not torch.equal(output['pred_boxes'][1], changed['pred_boxes'][1])


def test_strict_restore_optimizer_and_next_draws_without_initialization_assets():
    module = pipeline().eval()
    data = batch()
    optimizer = module.configure_optimizers()['optimizer']
    module._losses_for_batch(data)['total'].backward()
    optimizer.step()
    restored = pipeline().eval()
    restored.load_state_dict(deepcopy(module.state_dict()), strict=True)
    restored_optimizer = restored.configure_optimizers()['optimizer']
    restored_optimizer.load_state_dict(deepcopy(optimizer.state_dict()))
    assert len(restored_optimizer.state) == len(optimizer.state)
    state = module.training_rng_state.clone()
    module._draw_training_inputs(data, training=False)
    assert torch.equal(state, module.training_rng_state)
    for left, right in zip(module._draw_training_inputs(data), restored._draw_training_inputs(data)):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    with torch.no_grad():
        for steps in (1, 4):
            module.model.flow_sampling_steps = restored.model.flow_sampling_steps = steps
            left = module.model.sample_boxes(data['image'], generator=torch.Generator().manual_seed(7))
            right = restored.model.sample_boxes(data['image'], generator=torch.Generator().manual_seed(7))
            torch.testing.assert_close(left['pred_boxes'], right['pred_boxes'], atol=0, rtol=0)


def test_initialization_maps_selected_encoder_heads_and_rejects_missing_tensors(tmp_path):
    from src.models.pipelines.pose_estimators.poseflow.training_v40 import initialize_v40_weights
    model = pipeline().model
    state = model.state_dict()
    source = {}
    for name, tensor in state.items():
        source_name = model.pretrained_source_name(name, (2,))
        if source_name is not None:
            source['module.'+source_name] = torch.full_like(tensor, .125)
    path = tmp_path/'edpose.pth'
    torch.save({'model': source}, path)
    untouched = state['input_mlp.0.weight'].clone()
    report = initialize_v40_weights(model, path, encoder_source_layers=(2,))
    assert report['tensor_coverage'] == 1 and report['encoder_source_layers'] == [2]
    torch.testing.assert_close(model.proposal_encoder.layers[0].linear1.weight,
                               torch.full_like(model.proposal_encoder.layers[0].linear1.weight, .125))
    torch.testing.assert_close(model.input_mlp[0].weight, untouched, atol=0, rtol=0)
    source.pop(next(k for k in source if 'enc_out_bbox_embed' in k))
    torch.save({'model': source}, path)
    with pytest.raises(ValueError, match='missing|incompatible'):
        initialize_v40_weights(model, path, encoder_source_layers=(2,))


def test_full_fifty_assignment_and_native_class_channel_background_gradients():
    from src.models.pipelines.pose_estimators.poseflow.latent_v40 import (
        proposal_detection_loss, select_training_references,
    )
    module = pipeline()
    truth = torch.rand(1, 50, 4)*.6+.1
    truth[..., 2:] = .08
    data = dict(gt_boxes_canon=truth, person_mask=torch.ones(1, 50, dtype=torch.bool),
                img_hw=torch.tensor([[256, 427]]))
    logits = torch.zeros(1, 200, 2, requires_grad=True)
    candidates = dict(pred_boxes=torch.cat((truth, torch.full((1, 150, 4), .9)), 1),
        features=torch.randn(1, 200, 32), pred_logits=logits, existence_logits=logits[..., 1])
    refs, targets, proposal_targets = select_training_references(candidates, data, module.matcher, slots=50)
    assert refs['features'].shape == (1, 50, 32) and targets['positive'].all()
    assert targets['target_indices'].unique().numel() == 50
    losses = proposal_detection_loss(candidates, proposal_targets, weights=(2., 5., 2.))
    losses['total'].backward()
    assert (logits.grad[..., 0] > 0).all()
    assert (logits.grad[0, 50:, 1] > 0).all()
    assert (logits.grad[0, :50, 1] < 0).all()

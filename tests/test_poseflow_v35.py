from copy import deepcopy

import pytest
import torch
from scipy.optimize import linear_sum_assignment


def batch(counts=(2, 1)):
    boxes = torch.tensor([[[.2, .3, .1, .2], [.8, .7, .2, .3]]]*len(counts))
    mask = torch.arange(2)[None] < torch.tensor(counts)[:, None]
    return dict(image=torch.randn(len(counts), 3, 128, 128), gt_boxes_canon=boxes,
                person_mask=mask, gt_box_labeled=mask, img_hw=torch.tensor([[128, 128]]*len(counts)),
                image_id=torch.arange(len(counts)))


def pipeline(spatial=False):
    from src.models.pipelines.pose_estimators.poseflow.training_v35 import PoseFlowV35PipelineConfig
    config = PoseFlowV35PipelineConfig(backbone_pretrained=False, pretrained_checkpoint=None,
        hidden_dim=32, num_heads=4, feedforward_dim=64, num_queries=6, num_proposals=6,
        num_group=6, grid_rows=2, grid_columns=3, spatial_queries=spatial,
        quick_eval_enabled=False)
    return config.build(max_epochs=2)


def test_counts_actual_negative_geometry_global_sort_and_canonical_indices():
    from src.models.pipelines.pose_estimators.poseflow.latent_v35 import build_v35_targets, grid_anchors
    data = batch()
    plain = build_v35_targets(data, proposals=6, generator=torch.Generator().manual_seed(1))
    anchors = grid_anchors(2, 3)
    sorted_targets = build_v35_targets(data, proposals=6, anchors=anchors,
                                     generator=torch.Generator().manual_seed(1))
    assert plain['positive'].sum(-1).tolist() == [4, 3]
    assert (plain['input_boxes'][~plain['positive']][:, 2:] > 0).all()
    for image, count in enumerate([2, 1]):
        permutation = sorted_targets['slot_to_original'][image]
        assert permutation.sort().values.tolist() == list(range(6))
        for name, value in plain.items():
            if name in ('canonical_slots', 'slot_to_original', 'unique_people'):
                continue
            torch.testing.assert_close(sorted_targets[name][image], value[image, permutation])
        canonical = sorted_targets['canonical_slots'][image, :count]
        assert sorted_targets['target_indices'][image, canonical].tolist() == list(range(count))
        distance = torch.cdist(anchors, plain['input_boxes'][image, :, :2]).square()
        rows, columns = linear_sum_assignment(distance.numpy())
        torch.testing.assert_close(distance[torch.arange(6), permutation].sum(), distance[rows, columns].sum())


def test_positive_perturbations_odd_spares_and_zero_fallback():
    from src.models.pipelines.pose_estimators.poseflow.latent_v35 import build_v35_targets
    targets = build_v35_targets(batch((1,)), proposals=7, generator=torch.Generator().manual_seed(7))
    assert targets['positive'].all()
    assert (targets['input_boxes'][0, 1:, :2] != targets['boxes'][0, 1:, :2]).any()
    torch.testing.assert_close(targets['input_boxes'][0, :, 2:], targets['boxes'][0, :, 2:])
    empty = build_v35_targets(batch((0,)), proposals=7)
    assert not empty['positive'].any() and not empty['input_pixels'].any()
    odd = build_v35_targets(batch((2, 2)), proposals=7)
    assert odd['positive'].sum(-1).tolist() == [4, 4]
    with pytest.raises(ValueError, match='budget'):
        build_v35_targets(batch(), proposals=1)


def test_box_loss_masks_geometry_but_trains_negative_existence():
    from src.models.pipelines.pose_estimators.poseflow.latent_v35 import box_task_loss
    boxes = torch.full((1, 2, 4), .4, requires_grad=True)
    logits = torch.zeros(1, 2, requires_grad=True)
    targets = dict(positive=torch.tensor([[True, False]]), boxes=torch.tensor([[[.3, .3, .2, .2], [0., 0., 0., 0.]]]))
    losses = box_task_loss(dict(pred_boxes=boxes, existence_logits=logits), targets)
    losses['total'].backward()
    assert boxes.grad[0, 1].abs().sum() == 0 and logits.grad[0, 1] != 0
    assert set(losses) == {'loss_ce', 'loss_bbox', 'loss_giou', 'total'}


def test_box_matching_zero_weight_parity_and_spatial_preference():
    from src.models.pipelines.pose_estimators.poseflow.latent_v35 import BoxOTAMatcher, grid_anchors
    from src.models.pipelines.pose_estimators.diffusionregpose.training import DiffusionRegPoseOTAMatcher
    anchors = grid_anchors(2, 2)
    output = dict(pred_boxes=torch.tensor([[[.75, .75, .2, .2]]*4]), pred_logits=torch.zeros(1, 4, 2))
    targets = [dict(labels=torch.ones(1, dtype=torch.long), boxes=output['pred_boxes'][0, :1])]
    reference = DiffusionRegPoseOTAMatcher(candidate_topk=1)(output, targets, include_keypoints=False)
    actual = BoxOTAMatcher(anchors=anchors, spatial_cost_weight=0, candidate_topk=1)(output, targets)
    for a, b in zip(reference[0], actual[0]):
        torch.testing.assert_close(a, b)
    chosen = BoxOTAMatcher(anchors=anchors, spatial_cost_weight=10, candidate_topk=1)(output, targets)
    assert chosen[0][0].tolist() == [3]


def test_clean_anchor_free_common_initialization_and_detached_routes():
    torch.manual_seed(42); plain = pipeline()
    torch.manual_seed(42); spatial = pipeline(True)
    for name, value in plain.model.state_dict().items():
        torch.testing.assert_close(value, spatial.model.state_dict()[name], atol=0, rtol=0)
    assert not any(word in n for n, _ in spatial.model.named_parameters() for word in ('dino', 'joint_head', 'pose_decoder'))
    data = batch()
    inputs = spatial._draw_training_inputs(data)[0]['input_pixels']
    plain.model.eval(); spatial.model.eval()
    with torch.no_grad():
        image_features = plain.model.encode_image(data['image'])
        left = plain.model.encode_clean(inputs, image_features)
        right = spatial.model.encode_clean(inputs, image_features)
        torch.testing.assert_close(left['pred_latents'], right['pred_latents'], atol=0, rtol=0)
    captured = {}
    original = spatial.model.training_outputs

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        captured['clean'] = result[0]['pred_latents']; captured['clean'].retain_grad()
        assert not result[2].requires_grad
        return result

    spatial.model.training_outputs = capture
    losses = spatial._losses_for_batch(data)
    torch.testing.assert_close(losses['total'], .5*losses['clean/total']+.5*losses['denoised/total']+10*losses['loss_latent'])
    losses['loss_latent'].backward(retain_graph=True)
    assert captured['clean'].grad is None
    assert all(p.grad is None for p in spatial.model.clean_layers.parameters())
    assert spatial.model.query_projection.weight.grad.abs().sum() > 0
    spatial.zero_grad(); losses['total'].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in spatial.model.backbone.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in spatial.model.clean_layers.parameters())


def test_restore_next_draws_and_box_only_gaussian_inference():
    module = pipeline(True).eval()
    data = batch()
    module._losses_for_batch(data)
    restored = pipeline(True).eval(); restored.load_state_dict(deepcopy(module.state_dict()), strict=True)
    first = module._draw_training_inputs(data); second = restored._draw_training_inputs(data)
    for name in first[0]:
        torch.testing.assert_close(first[0][name], second[0][name], atol=0, rtol=0)
    torch.testing.assert_close(first[1], second[1]); torch.testing.assert_close(first[2], second[2])
    with torch.no_grad():
        output = restored.model.sample_keypoints(data['image'], generator=torch.Generator().manual_seed(5))
        independent = restored.model.sample_keypoints(data['image'][:1], generator=torch.Generator().manual_seed(5))
    assert output['pred_boxes'].shape == (2, 6, 4) and 'pred_keypoints' not in output
    assert torch.isfinite(output['pred_boxes']).all()
    torch.testing.assert_close(output['pred_boxes'][:1], independent['pred_boxes'], atol=1e-5, rtol=1e-5)


def test_one_swapped_match_controls_box_and_latent_after_global_sort():
    from src.models.pipelines.pose_estimators.poseflow.latent_v35 import assign_box_targets, build_v35_targets, grid_anchors
    targets = build_v35_targets(batch(), proposals=6, anchors=grid_anchors(2, 3))
    latents = torch.randn(2, 6, 32, requires_grad=True)
    clean = dict(pred_latents=latents)
    recovered = dict(pred_latents=torch.randn_like(latents))

    class Matcher:
        def __call__(self, output, gt):
            return [(torch.tensor([5, 1]), torch.tensor([1, 0])), (torch.tensor([4]), torch.tensor([0]))]

    assigned, expected, metadata = assign_box_targets(clean, recovered, targets, Matcher())
    slot = targets['canonical_slots'][0, 1]
    torch.testing.assert_close(assigned['boxes'][0, 5], targets['boxes'][0, slot])
    torch.testing.assert_close(expected[0, 5], latents[0, slot])
    assert not expected.requires_grad and metadata['matched_person'][0, 5] == 1
    positive_only = build_v35_targets(batch((1,)), proposals=6)
    class OneMatch:
        def __call__(self, output, gt):
            return [(torch.tensor([0]), torch.tensor([0]))]
    _, fallback, metadata = assign_box_targets(dict(pred_latents=latents[:1]),
        dict(pred_latents=recovered['pred_latents'][:1]), positive_only, OneMatch())
    assert not fallback[0, 1:].any() and metadata['zero_background_latent_fraction'] > 0


def test_position_enters_attention_queries_keys_but_not_values(monkeypatch):
    from src.models.pipelines.pose_estimators.poseflow.model_v26 import LatentImageBlock
    block = LatentImageBlock(32, 4, 64)
    seen = {}
    def capture_self(q, k, v, **kwargs):
        seen['self'] = (q, k, v)
        return torch.zeros_like(q), None
    def capture_image(q, k, v, **kwargs):
        seen['image'] = (q, k, v)
        return torch.zeros_like(q), None
    monkeypatch.setattr(block.self_attention, 'forward', capture_self)
    monkeypatch.setattr(block.image_attention, 'forward', capture_image)
    state = torch.randn(2, 6, 32)
    position = torch.randn(1, 6, 32)
    memory = torch.randn(2, 8, 32); image_position = torch.randn_like(memory)
    block(state, torch.zeros(2, 32), (memory, image_position, None), query_position=position)
    q, k, v = seen['self']
    torch.testing.assert_close(q-v, position.expand_as(q))
    torch.testing.assert_close(k, q)
    torch.testing.assert_close(seen['image'][0], block.norms[1](state)+position)
    torch.testing.assert_close(seen['image'][1], memory+image_position)
    torch.testing.assert_close(seen['image'][2], memory)


def test_box_evaluation_has_no_pose_dependency_and_preserves_scaling(tmp_path):
    import json
    from src.models.pipelines.pose_estimators.poseflow.quick_eval_v35 import (
        box_records, evaluate_box_records, postprocess_boxes,
    )
    output = dict(pred_boxes=torch.tensor([[[.5, .5, .5, .5]]]), existence_logits=torch.tensor([[10.]]))
    result = postprocess_boxes(output, torch.tensor([[200., 100.]]))[0]
    records = box_records(1, result)
    assert records[0]['bbox'] == [25., 50., 50., 100.] and records[0]['category_id'] == 1
    assert 'keypoints' not in records[0]
    annotations = dict(images=[dict(id=1, width=100, height=200, file_name='unused.jpg')],
        categories=[dict(id=1, name='person')], annotations=[dict(id=1, image_id=1, category_id=1,
            bbox=[25., 50., 50., 100.], area=5000., iscrowd=0)])
    path = tmp_path/'boxes.json'; path.write_text(json.dumps(annotations))
    metrics = evaluate_box_records(records, path, [1])
    assert metrics['AP'] > .99
    assert evaluate_box_records([], path, [1])['AP'] == 0


def test_batched_fixed_state_isolation_padding_and_anchor_permutation():
    model = pipeline(True).model.eval()
    memory = torch.randn(4, 9, 32)
    positions = torch.randn_like(memory)
    mask = torch.zeros(4, 9, dtype=torch.bool); mask[:, -2:] = True
    state = torch.randn(4, 6, 32); times = torch.tensor([0, 100, 500, 999])
    with torch.no_grad():
        output = model.denoise(state, times, (memory, positions, mask))
        for size in (1, 2):
            result = model.denoise(state[:size], times[:size], (memory[:size], positions[:size], mask[:size]))
            torch.testing.assert_close(result['pred_boxes'], output['pred_boxes'][:size], atol=1e-6, rtol=1e-5)
        changed = memory.clone(); changed[:, -2:] += 10000
        result = model.denoise(state, times, (changed, positions, mask))
        torch.testing.assert_close(result['pred_boxes'], output['pred_boxes'], atol=0, rtol=0)
        permutation = torch.tensor([5, 1, 3, 0, 4, 2])
        model.query_anchors.copy_(model.query_anchors[permutation])
        result = model.denoise(state[:, permutation], times, (memory, positions, mask))
        torch.testing.assert_close(result['pred_boxes'], output['pred_boxes'][:, permutation], atol=1e-6, rtol=1e-5)


def test_v35_recipes_are_box_only_and_differ_only_by_spatial_bundle():
    from pathlib import Path
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    with initialize_config_dir(config_dir=str(Path('conf').resolve()), version_base=None):
        left = OmegaConf.to_container(compose(config_name='benchmark_poseflow_v35'), resolve=True)
        right = OmegaConf.to_container(compose(config_name='benchmark_poseflow_v35_spatial'), resolve=True)
    assert left['pipeline']['latent_loss_weight'] == 10 and left['training']['grad_clip'] == .1
    assert left['pipeline']['weights']['keypoints'] == 0 and left['pipeline']['weights']['oks'] == 0
    left['pipeline'].update(spatial_queries=True, spatial_cost_weight=1.)
    left.update(run_name=right['run_name'], out_dir=right['out_dir'])
    assert left == right

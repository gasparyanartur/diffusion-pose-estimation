from __future__ import annotations

import pytest
import torch
import json

from scripts.investigation.diffusionregpose_inspection.trace import (
    capture_sampling,
    reconstruct_sampling,
    query_lineage,
)
from scripts.investigation.diffusionregpose_inspection.analysis import (
    evaluate_records, postprocess_with_queries, rank_images,
)
from src.models.pipelines.pose_estimators.diffusionregpose.model import DiffusionRegPoseModel


def make_model():
    return DiffusionRegPoseModel(
        backbone_pretrained=False, hidden_dim=32, num_heads=4,
        num_encoder_layers=1, num_decoder_layers=3, feedforward_dim=64,
        num_queries=5, num_group=2, num_proposals=2, num_box_decoder_layers=1,
        sampling_timesteps=2,
    ).eval()


def test_trace_preserves_sampling_and_cleans_up_on_failure():
    torch.manual_seed(7)
    model = make_model()
    image = torch.randn(1, 3, 64, 80)
    with torch.no_grad():
        plain = model.sample_keypoints(image, generator=torch.Generator().manual_seed(42))
        with capture_sampling(model) as calls:
            traced = model.sample_keypoints(image, generator=torch.Generator().manual_seed(42))
    for key in ('pred_logits', 'pred_boxes', 'pred_keypoints', 'diffusion_keypoints'):
        torch.testing.assert_close(plain[key], traced[key], rtol=0, atol=0)
    assert len(calls) == 2
    assert [call['time'] for call in calls] == [999, 499]
    for call in calls:
        assert len(call['layers']) == 3
        assert call['layers'][0]['mask'] is None
        assert call['layers'][1]['mask'].shape == (30, 30)
        assert call['selected_boxes'].shape == (2,)
        for layer in call['layers']:
            assert torch.isfinite(layer['locations']).all()
            torch.testing.assert_close(
                layer['weights'].sum((-1, -2)),
                torch.ones_like(layer['weights'].sum((-1, -2))),
            )
    with pytest.raises(RuntimeError, match='intentional'):
        with capture_sampling(model):
            raise RuntimeError('intentional')
    for module in model.modules():
        assert not module._forward_pre_hooks
        assert not module._forward_hooks


def test_query_lineage_maps_selected_box_and_expanded_person_tokens():
    call = {'selected_boxes': torch.tensor([4, 1]), 'num_box_layers': 1, 'num_joints': 14}
    assert query_lineage(call, 0, 1, joint=None) == 1
    assert query_lineage(call, 1, 1, joint=None) == 15
    assert query_lineage(call, 2, 1, joint=4) == 20
    with pytest.raises(ValueError, match='box-only'):
        query_lineage(call, 0, 1, joint=4)


def test_sampling_locations_respect_reference_extent_and_level_coordinates():
    references = torch.tensor([[[[0.25, 0.5, 0.4, 0.8]]]])
    offsets = torch.tensor([[[[[[1.0, -1.0], [0.0, 0.0]]]]]])
    logits = torch.zeros(1, 1, 1, 1, 2)
    locations, weights = reconstruct_sampling(references, offsets, logits, torch.tensor([[4, 8]]))
    torch.testing.assert_close(locations[0, 0, 0, 0, 0], torch.tensor([0.35, 0.3]))
    torch.testing.assert_close(weights, torch.full_like(weights, .5))
    locations, _ = reconstruct_sampling(references[..., :2], offsets, logits, torch.tensor([[4, 8]]))
    torch.testing.assert_close(locations[0, 0, 0, 0, 0], torch.tensor([.375, .25]))


def test_ranking_excludes_undefined_scores_and_breaks_ties_by_image_id():
    rows = [{'image_id': i, 'AP': score} for i, score in
            [(9, .9), (2, .9), (1, .1), (7, 0.), (8, None), (5, float('nan'))]]
    ranked = rank_images(rows, count=2)
    assert ranked == {'best': [2, 9], 'worst': [7, 1], 'undefined': [5, 8]}


def test_per_image_ap_handles_perfect_empty_and_ignored_ground_truth(tmp_path):
    keypoints = [coordinate for index in range(14) for coordinate in (10+index*2, 20+index*2, 2)]
    annotation = {'id': 1, 'image_id': 1, 'category_id': 1, 'keypoints': keypoints,
                  'num_keypoints': 14, 'bbox': [5, 5, 80, 80], 'iscrowd': 0}
    path = tmp_path / 'annotations.json'
    path.write_text(json.dumps({
        'images': [{'id': index, 'width': 100, 'height': 100} for index in (1, 2)],
        'annotations': [annotation, {**annotation, 'id': 2, 'image_id': 2, 'iscrowd': 1}],
        'categories': [{'id': 1, 'name': 'person', 'keypoints': list(range(14))}],
    }))
    records = [{**annotation, 'score': .9}]
    aggregate, rows = evaluate_records(path, records, [1, 2])
    assert aggregate['AP'] == pytest.approx(1.)
    assert rows[0]['AP'] == pytest.approx(1.)
    assert rows[1]['AP'] is None
    aggregate, rows = evaluate_records(path, [], [1, 2])
    assert aggregate['AP'] == 0
    assert rows[0]['AP'] == 0
    assert rows[1]['AP'] is None


@pytest.mark.parametrize('keep', [[2, 0], []])
def test_native_postprocessing_query_provenance(monkeypatch, keep):
    from scripts.investigation.diffusionregpose_inspection import analysis
    from src.models.pipelines.pose_estimators.diffusionregpose import evaluation
    def select(*args):
        return torch.tensor(keep, dtype=torch.long)
    monkeypatch.setattr(analysis, '_crowdpose_pose_nms_keep', select)
    monkeypatch.setattr(evaluation, '_crowdpose_pose_nms_keep', select)
    output = {'pred_logits': torch.tensor([[[-3., 3.], [-3., 2.], [-3., 1.]]]),
              'pred_boxes': torch.rand(1, 3, 4), 'pred_keypoints': torch.rand(1, 3, 42)}
    result = postprocess_with_queries(output, (100, 200), (50, 100))
    assert len(result['scores']) == len(keep)
    for query, score, keypoints in zip(result['query_ids'], result['scores'], result['keypoints']):
        assert score == output['pred_logits'][0, query, 1].sigmoid()
        torch.testing.assert_close(keypoints.reshape(14, 3)[:, :2],
                                   output['pred_keypoints'][0, query, :28].reshape(14, 2)*torch.tensor([200, 100]))


@pytest.mark.parametrize('reference_x', [.4, -.4])
def test_attention_plot_shows_true_offsets_strengths_and_legends(tmp_path, reference_x):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    from PIL import Image
    from scripts.investigation.diffusionregpose_inspection.plots import attention_figure

    image_path = tmp_path / 'image.png'
    Image.new('RGB', (100, 80), 'gray').save(image_path)
    reference = torch.tensor([reference_x, .5, .1, .1])
    offsets = torch.tensor([[.01, .02], [-.02, .01], [.03, -.02], [-.01, -.03]])
    points = reference[:2] + offsets
    weights = torch.arange(1, 17, dtype=torch.float32).reshape(4, 4)/136
    layer = {'references': reference[None],
             'locations': points[None, None, None].expand(1, 1, 4, 4, 2),
             'weights': weights[None, None], 'spatial_shapes': torch.tensor([[10, 12]]*4)}
    calls = [{'layers': [layer], 'selected_boxes': torch.tensor([0]),
              'num_box_layers': 1, 'num_joints': 14}]
    trace_path = tmp_path / 'trace.pt'
    torch.save(calls, trace_path)
    cache = {'samples': [{'image_id': 1, 'img_path': str(image_path), 'img_hw': (80, 100)}],
             'trace_paths': {(1, 1): str(trace_path)}, 'num_heads': 1}
    figure = attention_figure(cache, 1, 1, 0, 0, head=0, layer_index=0)
    try:
        images = [axis for axis in figure.axes if axis.get_gid() == 'attention-image']
        bars = [axis for axis in figure.axes if axis.get_gid() == 'attention-strength']
        assert len(images) == len(bars) == 4
        expected_start = reference[:2]*torch.tensor([100, 80])
        for level, (image_axis, strength_axis) in enumerate(zip(images, bars)):
            arrows = [patch for patch in image_axis.patches if isinstance(patch, FancyArrowPatch)]
            assert len(arrows) == 4
            for point, arrow in enumerate(arrows):
                torch.testing.assert_close(torch.tensor(arrow._posA_posB[0]), expected_start)
                torch.testing.assert_close(torch.tensor(arrow._posA_posB[1]), points[point]*torch.tensor([100, 80]))
            assert arrows[-1].get_linewidth() > arrows[0].get_linewidth()
            assert arrows[-1].get_alpha() > arrows[0].get_alpha()
            assert [patch.get_width() for patch in strength_axis.patches] == pytest.approx(weights[level].numpy()*100)
            xlim = sorted(image_axis.get_xlim())
            assert xlim[0] <= expected_start[0] <= xlim[1]
        labels = [text.get_text() for text in figure.legends[0].get_texts()]
        assert labels == ['Reference point', 'Predicted offset', 'Sample 1', 'Sample 2', 'Sample 3', 'Sample 4']
    finally:
        plt.close(figure)


def test_attention_context_preserves_person_extremities_and_head():
    from scripts.investigation.diffusionregpose_inspection.plots import person_attention_context
    points = torch.tensor([[45., 20.], [55., 20.], [35., 35.], [65., 35.],
                           [20., 45.], [80., 45.], [45., 50.], [55., 50.],
                           [43., 70.], [57., 70.], [40., 90.], [60., 90.],
                           [50., 10.], [50., 18.]])
    sample = {'img_hw': (100, 100), 'keypoints': torch.stack([points, points+200]),
              'visibility': torch.ones(2, 14), 'gt_boxes_pixel': torch.tensor([[15., 5., 70., 90.], [200., 200., 70., 90.]])}
    prediction = {'pred_keypoints': torch.cat([points.flatten()/100, torch.ones(14)])[None]}
    reference = torch.tensor([20., 45.]).numpy()
    samples = torch.tensor([[19., 44.], [22., 45.], [21., 47.], [23., 43.]]).numpy()
    context = person_attention_context(sample, prediction, 0, reference, samples)
    assert context['ground_truth_person'] == 0
    lower, upper = context['person_limits']
    assert (lower <= points.amin(0).numpy()).all()
    assert (upper >= points.amax(0).numpy()).all()
    lower, upper = context['detail_limits']
    assert (lower <= points[12].numpy()).all()
    assert (upper >= points[12].numpy()).all()

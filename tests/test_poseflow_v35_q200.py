from copy import deepcopy
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from tests.test_poseflow_v35 import batch
from src.models.pipelines.pose_estimators.poseflow.latent_v35 import build_v35_targets, grid_anchors
from src.models.pipelines.pose_estimators.poseflow.training_v35 import PoseFlowV35PipelineConfig


def configuration():
    with initialize_config_dir(config_dir=str(Path('conf').resolve()), version_base=None):
        return compose(config_name='benchmark_poseflow_v35_spatial_early_cost_q200')


def test_q200_config_changes_only_count_grid_and_namespace():
    with initialize_config_dir(config_dir=str(Path('conf').resolve()), version_base=None):
        base = OmegaConf.to_container(compose(config_name='benchmark_poseflow_v35_spatial_early_cost'), resolve=True)
    actual = OmegaConf.to_container(configuration(), resolve=True)
    base['pipeline'].update(num_proposals=200, num_queries=200, num_group=200, grid_rows=10, grid_columns=20)
    base.update(run_name='poseflow_v35_spatial_early_cost_q200', out_dir='results/poseflow_v35_spatial_early_cost_q200')
    assert actual == base


def test_q200_construction_preserves_coverage_and_fallbacks():
    config = configuration()
    anchors = grid_anchors(config.pipeline.grid_rows, config.pipeline.grid_columns)
    assert anchors.shape == (200, 2) and anchors.unique(dim=0).shape[0] == 200
    targets = build_v35_targets(batch((2, 1, 0)), proposals=200, anchors=anchors, generator=torch.Generator().manual_seed(42))
    assert targets['positive'].sum(-1).tolist() == [101, 100, 0]
    assert targets['input_boxes'].shape == (3, 200, 4)
    for image, count in enumerate((2, 1)):
        slots = targets['canonical_slots'][image, :count]
        assert targets['target_indices'][image, slots].tolist() == list(range(count))
    assert (targets['input_boxes'][~targets['positive']][..., 2:] > 0).all()
    assert build_v35_targets(batch((1,)), proposals=200, anchors=anchors)['positive'].all()
    empty = build_v35_targets(batch((0,)), proposals=200, anchors=anchors)
    assert empty['zero_fallback'].all() and not empty['input_pixels'].any()


def test_q200_strict_rng_restore_and_gaussian_inference():
    options = OmegaConf.to_container(configuration().pipeline, resolve=True)
    options.update(backbone_pretrained=False, pretrained_checkpoint=None, hidden_dim=32,
                   num_heads=4, feedforward_dim=64, quick_eval_enabled=False)
    config = PoseFlowV35PipelineConfig.from_hydra_config(OmegaConf.create(options))
    module = config.build(max_epochs=2).eval()
    data = batch()
    targets, times, noise = module._draw_training_inputs(data)
    assert noise.shape == (2, 200, 32)
    restored = config.build(max_epochs=2).eval()
    restored.load_state_dict(deepcopy(module.state_dict()), strict=True)
    expected = module._draw_training_inputs(data)
    actual = restored._draw_training_inputs(data)
    for key in expected[0]:
        torch.testing.assert_close(expected[0][key], actual[0][key], atol=0, rtol=0)
    torch.testing.assert_close(expected[1], actual[1], atol=0, rtol=0)
    torch.testing.assert_close(expected[2], actual[2], atol=0, rtol=0)
    with torch.no_grad():
        first = module.model.sample_boxes(data['image'][:1], generator=torch.Generator().manual_seed(42))
        second = restored.model.sample_boxes(data['image'][:1], generator=torch.Generator().manual_seed(42))
    assert first['pred_boxes'].shape == (1, 200, 4)
    for key in first:
        if torch.is_tensor(first[key]):
            assert torch.isfinite(first[key]).all()
            torch.testing.assert_close(first[key], second[key], atol=0, rtol=0)

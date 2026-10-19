from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from scripts.evaluation.evaluate_poseflow import load_poseflow_model
from src.models.pipelines.pose_estimators.poseflow import PoseFlowPipelineConfig
from src.models.pipelines.pose_estimators.poseflow.training_dense_read_v19 import (
    PoseFlowV19DenseReadPipelineConfig,
)
from tests.test_poseflow import _batch
from tests.test_poseflow_v19 import configuration


def dense_configuration(**overrides):
    return PoseFlowV19DenseReadPipelineConfig(**(
        configuration().model_dump() | {'detach_decoder_geometry': True} | overrides))


def test_configuration_strict_checkpoint_and_one_call_parity(tmp_path):
    with initialize_config_dir(config_dir=str(Path('conf').resolve()), version_base=None):
        resolved = compose(config_name='benchmark_poseflow_v19_dense_read')
    config = PoseFlowPipelineConfig.from_hydra_config(resolved.pipeline)
    assert isinstance(config, PoseFlowV19DenseReadPipelineConfig)
    assert config.detach_decoder_geometry and not config.detach_sampling_references
    assert config.rollout_probability == .5 and config.rollout_call_counts == [2, 3]
    assert resolved.training.grad_clip == 2 and resolved.training.max_epochs == 80
    assert resolved.run_name == 'poseflow_v19_dense_read_training'
    with pytest.raises(ValueError, match='full geometry'):
        dense_configuration(detach_decoder_geometry=False)
    pipeline = dense_configuration().build(max_epochs=2).eval()
    assert 'evidence_rank' not in str(pipeline.model.state_dict().keys())
    pipeline.model.dense_read.gate.data.fill_(.2)
    torch.save({'state_dict': pipeline.state_dict()}, tmp_path/'dense.ckpt')
    config_path = tmp_path/'config.yaml'
    OmegaConf.save(OmegaConf.create({'pipeline': pipeline.v12_config.model_dump() | {
        '_target_': 'src.models.pipelines.pose_estimators.poseflow.PoseFlowV19DenseReadPipelineConfig'}}), config_path)
    restored, _ = load_poseflow_model(tmp_path/'dense.ckpt', config_path)
    restored.eval()
    image, state = torch.randn(1, 3, 64, 64), torch.randn(1, 3, 8)
    with torch.no_grad():
        for steps in [1, 3]:
            expected = pipeline.model.sample_keypoints(image, initial_latents=state, sampling_steps=steps)
            actual = restored.sample_keypoints(image, initial_latents=state, sampling_steps=steps)
            for key in ('pred_coordinates', 'pred_logits', 'hypothesis_memory', 'coordinate_state'):
                torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
        direct = restored.denoise(state, torch.ones(1), restored.encode_image(image))
        persistent = restored.sample_keypoints(image, initial_latents=state, sampling_steps=1)
        reset = restored.sample_keypoints(image, initial_latents=state, sampling_steps=1,
                                         persist_hypotheses=False)
        for key in ('pred_coordinates', 'pred_logits', 'hypothesis_memory'):
            torch.testing.assert_close(direct[key], persistent[key], atol=0, rtol=0)
            torch.testing.assert_close(reset[key], persistent[key], atol=0, rtol=0)
    state_dict = pipeline.state_dict()
    state_dict.pop('model.dense_read.gate')
    torch.save({'state_dict': state_dict}, tmp_path/'missing.ckpt')
    with pytest.raises(RuntimeError, match='Missing key'):
        load_poseflow_model(tmp_path/'missing.ckpt', config_path)


def test_training_objective_parity_rollout_and_diagnostics():
    torch.manual_seed(42)
    baseline = configuration(detach_decoder_geometry=True, rollout_probability=0.).build(max_epochs=2).train()
    pipeline = dense_configuration(rollout_probability=0.).build(max_epochs=2).train()
    pipeline.model.load_baseline_state_dict(baseline.model.state_dict())
    batch = _batch() | {'image': torch.randn(1, 3, 64, 64)}
    batch['gt_labeled'][:, :, 1] = False
    torch.manual_seed(71)
    expected = baseline._losses_for_batch(batch)
    torch.manual_seed(71)
    actual = pipeline._losses_for_batch(batch)
    assert expected.keys() == actual.keys()
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
    assert actual['total'] == actual['loss_prediction']
    assert 'dense_read/gate' in '/'.join(pipeline._v12_diagnostics)
    architecture = pipeline._quick_architecture()
    assert architecture['training_variant'] == 'V19-dense-read'
    assert architecture['dense_read_calls_per_denoise'] == 1
    pipeline.v12_config.rollout_probability = 1.
    pipeline.v12_config.rollout_call_counts = [3]
    losses = pipeline._losses_for_batch(batch)
    assert pipeline._rollout_call_count == 3 and torch.isfinite(losses['total'])
    losses['total'].backward()
    assert pipeline.model.dense_read.gate.grad.abs() > 0
    empty = batch | {'person_mask': torch.zeros(1, 2, dtype=torch.bool)}
    assert torch.isfinite(pipeline._losses_for_batch(empty)['total'])

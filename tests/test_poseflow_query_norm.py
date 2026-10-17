import torch
from torch import nn

from src.models.pipelines.pose_estimators.poseflow.query_norm import (
    PoseFlowQueryNormModel,
    PoseFlowQueryNormPipelineConfig,
)
from src.models.pipelines.pose_estimators.poseflow.model_v15 import V15HypothesisUpdate
from tests.test_poseflow_v15_rollout import _config


def test_normalized_queries_preserve_original_retention_and_gradients():
    writer = V15HypothesisUpdate(32, 4, 0.)
    writer.query_norm = nn.LayerNorm(32, elementwise_affine=False)
    memory = (torch.randn(2, 50, 32) * .02 + .4).requires_grad_()
    globals = torch.randn(2, 3, 32, requires_grad=True)
    queries = writer.query_norm(memory)
    evidence = writer.evidence_norm(writer.attention(queries, globals, globals, need_weights=False)[0])
    retention = writer.retention_logit.sigmoid()
    actual, _ = writer(memory, globals)
    torch.testing.assert_close(actual, retention * memory + (1-retention) * evidence)
    assert not torch.allclose(actual, writer.evidence_norm(actual))
    actual.square().mean().backward()
    assert memory.grad.abs().sum() > 0 and globals.grad.abs().sum() > 0
    assert writer.attention.in_proj_weight.grad[:32].abs().sum() > 0
    assert not list(writer.query_norm.parameters())
    constant, _ = writer(torch.ones_like(memory), globals)
    assert torch.isfinite(constant).all()


def test_model_keeps_state_schema_count_and_rollout_contract():
    base_config = _config()
    config = PoseFlowQueryNormPipelineConfig(**base_config.model_dump())
    base = base_config.build(max_epochs=2).model
    model = config.build(max_epochs=2).model.eval()
    assert isinstance(model, PoseFlowQueryNormModel)
    assert set(model.state_dict()) == set(base.state_dict())
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in base.parameters())
    model.load_state_dict(base.state_dict(), strict=True)
    assert isinstance(base.hypothesis_update.query_norm, nn.Identity)
    image = torch.randn(1, 3, 64, 64)
    state = torch.randn(1, 3, 3, 32)
    encoded = model.encode_image(image)
    first = model.denoise(state, torch.ones(1), encoded)
    second = model.denoise(first['pred_latents'], torch.tensor([.5]), encoded,
                           hypothesis_memory=first['hypothesis_memory'])
    gradient, = torch.autograd.grad(second['pred_latents'].square().mean(), first['hypothesis_memory'])
    assert gradient.abs().sum() > 0
    permutation = torch.tensor([2, 0, 1])
    permuted = model.denoise(state[:, permutation], torch.ones(1), encoded)
    torch.testing.assert_close(permuted['hypothesis_memory'], first['hypothesis_memory'], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(permuted['pred_latents'], first['pred_latents'][:, permutation], atol=2e-5, rtol=2e-5)


def test_config_and_strict_evaluator_restore(tmp_path):
    from pathlib import Path
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from scripts.evaluation.evaluate_poseflow import load_poseflow_model
    from src.models.pipelines.pose_estimators.poseflow import PoseFlowPipelineConfig
    with initialize_config_dir(config_dir=str(Path('conf').resolve()), version_base=None):
        resolved = compose(config_name='benchmark_poseflow_v15_rollout_query_norm')
    parsed = PoseFlowPipelineConfig.from_hydra_config(resolved.pipeline)
    assert isinstance(parsed, PoseFlowQueryNormPipelineConfig)
    assert parsed.rollout_probability == .5 and parsed.rollout_call_counts == [2, 3]
    assert resolved.training.max_epochs == 80 and resolved.data.batch_size == 4
    assert resolved.training.accumulate_grad_batches == 2 and resolved.training.grad_clip == 2
    config = PoseFlowQueryNormPipelineConfig(**_config().model_dump())
    pipeline = config.build(max_epochs=2).eval()
    torch.save({'state_dict': pipeline.state_dict()}, tmp_path/'model.ckpt')
    OmegaConf.save(OmegaConf.create({'pipeline': config.model_dump() | {
        '_target_': 'src.models.pipelines.pose_estimators.poseflow.PoseFlowQueryNormPipelineConfig'}}), tmp_path/'config.yaml')
    restored, _ = load_poseflow_model(tmp_path/'model.ckpt', tmp_path/'config.yaml')
    assert isinstance(restored.hypothesis_update.query_norm, nn.LayerNorm)
    image = torch.randn(1, 3, 64, 64)
    state = torch.randn(1, 3, 3, 32)
    for steps in [1, 3]:
        expected = pipeline.model.sample_keypoints(image, initial_latents=state, sampling_steps=steps)
        actual = restored.eval().sample_keypoints(image, initial_latents=state, sampling_steps=steps)
        torch.testing.assert_close(actual['pred_latents'], expected['pred_latents'])
        torch.testing.assert_close(actual['hypothesis_memory'], expected['hypothesis_memory'])

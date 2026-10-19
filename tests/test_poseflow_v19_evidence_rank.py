import copy

import pytest
import torch

from tests.test_poseflow_v19 import configuration, source_model
from src.models.pipelines.pose_estimators.poseflow.model_v19 import PoseFlowV19Model
from src.models.pipelines.pose_estimators.poseflow.evidence_rank_v19 import (
    PoseFlowV19EvidenceRankModel, corrupt_evidence, evidence_ranking_loss,
)


def test_shared_noise_rms_scaling_and_zero_evidence():
    evidence = torch.tensor([[[[3., 4.], [0., 0.]]]], requires_grad=True)
    noise = torch.ones_like(evidence)
    levels = corrupt_evidence(evidence, [0., .25, .5], noise=noise)
    assert len(levels) == 3
    torch.testing.assert_close(levels[0], evidence, atol=0, rtol=0)
    scale = evidence.detach().square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
    torch.testing.assert_close(levels[1], evidence + .25*scale)
    torch.testing.assert_close(levels[2]-evidence, 2*(levels[1]-evidence))
    assert all(torch.isfinite(level).all() for level in levels)
    gradient = torch.autograd.grad(levels[2].sum(), evidence)[0]
    torch.testing.assert_close(gradient, torch.ones_like(evidence))


def test_ranking_margins_positive_normalization_and_empty_targets():
    scores = torch.zeros(1, 2, 3, requires_grad=True)
    positive = torch.tensor([[True, False]])
    loss, metrics = evidence_ranking_loss(scores, positive, [0., .25, .5], margin=1.)
    torch.testing.assert_close(loss, torch.tensor((.25+1.+.75)/3))
    assert metrics['pair02/violation_fraction'] == 1
    loss.backward()
    torch.testing.assert_close(scores.grad[0, 1], torch.zeros(3))
    assert scores.grad[0, 0, 0] < 0 and scores.grad[0, 0, 2] > 0
    ordered = torch.tensor([[[2., 1., 0.], [0., 0., 0.]]])
    assert evidence_ranking_loss(ordered, positive, [0., .25, .5])[0] == 0
    empty = evidence_ranking_loss(scores, torch.zeros_like(positive), [0., .25, .5])[0]
    assert empty.requires_grad and empty == 0


def test_clean_branch_parity_single_read_and_shared_prior(monkeypatch):
    torch.manual_seed(42)
    baseline = PoseFlowV19Model(source_model()).train()
    baseline.detach_decoder_geometry = True
    model = PoseFlowV19EvidenceRankModel(source_model()).train()
    missing = model.load_state_dict(baseline.state_dict(), strict=False)
    assert missing.missing_keys == ['evidence_confidence.weight'] and not missing.unexpected_keys
    image, state = torch.randn(1, 3, 64, 64), torch.randn(1, 3, 8)
    encoded = baseline.encode_image(image)
    expected = baseline.denoise(state, torch.ones(1), encoded)
    calls, inputs, outputs = [], [], []
    for layer in model.layers:
        layer.block.cross_attn.register_forward_hook(lambda *args: calls.append(1))
        original = layer.update_from_evidence
        def capture(features, memory, positions, evidence, *, collect_weights=False, update=original):
            inputs.append((features, memory, positions, evidence))
            result = update(features, memory, positions, evidence, collect_weights=collect_weights)
            outputs.append(result)
            return result
        monkeypatch.setattr(layer, 'update_from_evidence', capture)
    actual = model.denoise(state, torch.ones(1), encoded)
    assert len(calls) == len(model.layers) and len(inputs) == 3*len(model.layers)
    for offset in range(0, len(inputs), 3):
        for branch in [1, 2]:
            assert all(inputs[offset][i] is inputs[offset+branch][i] for i in range(3))
    for name in ['pred_coordinates', 'pred_logits', 'hypothesis_memory']:
        torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)
    assert actual['evidence_rank_scores'].shape == (1, 3, 3)
    assert model.evidence_confidence.bias is None
    score_gradients = torch.autograd.grad(actual['evidence_rank_scores'].square().sum(),
        [outputs[-3+i][1] for i in range(3)], retain_graph=True)
    assert all(g.abs().sum() > 0 for g in score_gradients)
    base_gradient = torch.autograd.grad(expected['pred_logits'].sum(), baseline.layers[0].block.cross_attn.value_proj.weight)[0]
    actual_gradient = torch.autograd.grad(actual['pred_logits'].sum(), model.layers[0].block.cross_attn.value_proj.weight)[0]
    torch.testing.assert_close(actual_gradient, base_gradient, atol=0, rtol=0)
    model.eval()
    no_ranking = model.denoise(state, torch.ones(1), model.encode_image(image))
    assert 'evidence_rank_scores' not in no_ranking
    model.train()
    with torch.no_grad():
        no_ranking = model.denoise(state, torch.ones(1), model.encode_image(image))
    assert 'evidence_rank_scores' not in no_ranking


def rank_configuration(**overrides):
    from src.models.pipelines.pose_estimators.poseflow.training_evidence_rank_v19 import PoseFlowV19EvidenceRankPipelineConfig
    return PoseFlowV19EvidenceRankPipelineConfig(**(configuration().model_dump() | {
        'detach_decoder_geometry': True} | overrides))


def test_configuration_loss_averaging_empty_and_gradients(monkeypatch):
    from tests.test_poseflow import _batch
    from src.models.pipelines.pose_estimators.poseflow.latent_v12 import build_v12_targets
    pipeline = rank_configuration().build(max_epochs=2).train()
    batch = _batch() | {'image': torch.randn(1, 3, 64, 64)}
    targets = build_v12_targets(batch, groups=3)
    state = torch.randn(1, 3, 8)
    schedule = torch.tensor([[1., .7, .3, 0.]])
    outputs, calls = [], []
    original = pipeline.model.denoise
    def capture(*args, **kwargs):
        output = original(*args, **kwargs)
        outputs.append(output)
        return output
    monkeypatch.setattr(pipeline.model, 'denoise', capture)
    original_loss = pipeline._denoising_losses
    def capture_loss(*args):
        losses = original_loss(*args)
        calls.append(losses)
        return losses
    monkeypatch.setattr(pipeline, '_denoising_losses', capture_loss)
    losses = pipeline._trajectory_losses(batch, targets, state, schedule)
    assert len(outputs) == 3
    torch.testing.assert_close(losses['total'], losses['loss_prediction']+.1*losses['loss_evidence_rank'])
    torch.testing.assert_close(losses['loss_evidence_rank'], torch.stack([c['loss_evidence_rank'] for c in calls]).mean())
    assert 'rollout/call2/evidence_rank/layer0/pair02/score_gap' in pipeline._v12_diagnostics
    last_loss = calls[-1]['total']
    gradients = torch.autograd.grad(last_loss, [outputs[0]['pred_coordinates'], outputs[0]['hypothesis_memory']],
                                    retain_graph=True)
    assert all(g.abs().sum() > 0 and torch.isfinite(g).all() for g in gradients)
    ranking_gradients = torch.autograd.grad(losses['loss_evidence_rank'],
        [pipeline.model.evidence_confidence.weight, pipeline.model.hypothesis_slots,
         pipeline.model.layers[0].block.cross_attn.value_proj.weight], retain_graph=True)
    assert all(g.abs().sum() > 0 and torch.isfinite(g).all() for g in ranking_gradients)
    losses['total'].backward()
    empty = targets | {'real': torch.zeros_like(targets['real'])}
    loss = original_loss(outputs[-1], empty)
    assert loss['loss_evidence_rank'] == 0 and torch.isfinite(loss['total'])
    for levels in [[0., .25, .25], [0., -.1, .5], [.1, .2, .3], [0., .5], [0., float('nan'), .5]]:
        with pytest.raises(ValueError):
            rank_configuration(evidence_noise_levels=levels)
    with pytest.raises(ValueError):
        rank_configuration(detach_decoder_geometry=False)


def test_new_namespace_strict_restore_and_eval_baseline(tmp_path):
    from pathlib import Path
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from scripts.evaluation.evaluate_poseflow import load_poseflow_model
    from src.models.pipelines.pose_estimators.poseflow import PoseFlowPipelineConfig
    from tests.test_poseflow import _batch
    with initialize_config_dir(config_dir=str(Path('conf').resolve()), version_base=None):
        resolved = compose(config_name='benchmark_poseflow_v19_evidence_rank')
    config = PoseFlowPipelineConfig.from_hydra_config(resolved.pipeline)
    assert config.evidence_rank_weight == .1 and config.detach_decoder_geometry
    pipeline = rank_configuration(evidence_noise_levels=[0., .1, .4]).build(max_epochs=2).eval()
    assert pipeline.model.evidence_noise_levels == [0., .1, .4]
    losses = pipeline._losses_for_batch(_batch() | {'image': torch.randn(1, 3, 64, 64)})
    assert 'loss_evidence_rank' not in losses and losses['total'] == losses['loss_prediction']
    assert pipeline._rollout_call_count == 1
    torch.save({'state_dict': pipeline.state_dict()}, tmp_path/'model.ckpt')
    OmegaConf.save(OmegaConf.create({'pipeline': pipeline.v12_config.model_dump() | {
        '_target_': 'src.models.pipelines.pose_estimators.poseflow.PoseFlowV19EvidenceRankPipelineConfig'}}),
        tmp_path/'config.yaml')
    restored, _ = load_poseflow_model(tmp_path/'model.ckpt', tmp_path/'config.yaml')
    restored.eval()
    assert restored.evidence_noise_levels == [0., .1, .4]
    image, state = torch.randn(1, 3, 64, 64), torch.randn(1, 3, 8)
    for steps in [1, 3]:
        expected = pipeline.model.sample_keypoints(image, initial_latents=state, sampling_steps=steps)
        actual = restored.sample_keypoints(image, initial_latents=state, sampling_steps=steps)
        torch.testing.assert_close(actual['pred_coordinates'], expected['pred_coordinates'], atol=0, rtol=0)
    incompatible = copy.deepcopy(pipeline.state_dict())
    incompatible.pop('model.evidence_confidence.weight')
    with pytest.raises(RuntimeError):
        pipeline.load_state_dict(incompatible, strict=True)


def test_fresh_noise_despite_fixed_attention_seed_and_disabled_parity(monkeypatch):
    from tests.test_poseflow import _batch
    from src.models.pipelines.pose_estimators.poseflow import evidence_rank_v19
    pipeline = rank_configuration(rollout_probability=0., deterministic_seed=42).build(max_epochs=2).train()
    batch = _batch() | {'image': torch.randn(1, 3, 64, 64)}
    draws = []
    original = evidence_rank_v19.corrupt_evidence
    def capture(evidence, levels):
        noise = torch.randn_like(evidence)
        draws.append(noise.detach())
        return original(evidence, levels, noise=noise)
    monkeypatch.setattr(evidence_rank_v19, 'corrupt_evidence', capture)
    torch.manual_seed(17)
    pipeline._losses_for_batch(batch)
    first = draws[0].clone()
    draws.clear()
    pipeline._losses_for_batch(batch)
    assert not torch.equal(first, draws[0])
    draws.clear()
    torch.manual_seed(17)
    pipeline._losses_for_batch(batch)
    torch.testing.assert_close(first, draws[0], atol=0, rtol=0)
    disabled = rank_configuration(evidence_rank_weight=0., rollout_probability=0.).build(max_epochs=2).train()
    baseline = configuration(detach_decoder_geometry=True, rollout_probability=0.).build(max_epochs=2).train()
    baseline.model.load_state_dict({name: value for name, value in disabled.model.state_dict().items()
                                   if name != 'evidence_confidence.weight'})
    torch.manual_seed(19)
    expected = baseline._losses_for_batch(batch)
    torch.manual_seed(19)
    actual = disabled._losses_for_batch(batch)
    torch.testing.assert_close(actual['total'], expected['total'], atol=0, rtol=0)
    assert 'loss_evidence_rank' not in actual


def test_ota_duplicates_and_invisible_joints_use_only_assignment_mask():
    from tests.test_poseflow import _batch
    from src.models.pipelines.pose_estimators.poseflow.latent_v12 import build_v12_targets
    pipeline = rank_configuration().build(max_epochs=2)
    batch = _batch() | {'person_mask': torch.tensor([[True, False]])}
    batch['gt_labeled'][:, :, 1] = False
    targets = build_v12_targets(batch, groups=3)
    slot = targets['real'][0].nonzero().item()
    points = targets['joints'][:, slot:slot+1].expand(-1, 3, -1, -1).clone()
    points[:, :, 1] = .99
    stage = {
        'pred_logits': torch.full((1, 3, 1), 8., requires_grad=True),
        'pred_boxes': targets['boxes'][:, slot:slot+1].expand(-1, 3, -1),
        'pred_keypoints': torch.cat((points.flatten(-2), torch.ones(1, 3, 2)), -1),
        'evidence_rank_scores': torch.tensor([[[0., 0., 0.], [2., 1., 0.], [0., 0., 0.]]], requires_grad=True),
        'evidence_rank_statistics': {},
    }
    losses = pipeline._stage_losses(stage, targets, pipeline.v12_config.weights.to_dataclass())
    assert losses['positive_assignments'] == 3 and losses['loss_keypoints'] == 0
    torch.testing.assert_close(losses['loss_evidence_rank'], torch.tensor(4/9))


def test_branch_attention_dropout_is_rejected():
    source = source_model()
    source.transformer.decoder.layers[0].self_attn.dropout = .1
    with pytest.raises(ValueError, match='zero decoder dropout'):
        PoseFlowV19EvidenceRankModel(source)

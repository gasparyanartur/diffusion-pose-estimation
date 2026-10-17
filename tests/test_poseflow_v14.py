import pytest
import torch

from tests.test_poseflow import _batch
from tests.test_poseflow_v12 import _config as v12_config
from src.models.pipelines.pose_estimators.poseflow.latent_v12 import build_v12_targets, ddim_update
from src.models.pipelines.pose_estimators.poseflow.latent_v14 import V14StateCodec
from src.models.pipelines.pose_estimators.poseflow.training_v14 import PoseFlowV14PipelineConfig


def _config(**overrides):
    return PoseFlowV14PipelineConfig(**(v12_config().model_dump() | {
        'matching_strategy': 'ota', 'quick_eval_enabled': False,
    } | overrides))


def test_codec_has_only_box_and_joint_latents():
    targets = build_v12_targets(_batch(), groups=3)
    codec = V14StateCodec(width=32, joints=2)
    clean = codec.encode(targets)
    assert clean.shape == (1, 3, 3, 32)
    changed = dict(targets, existence=1-targets['existence'])
    torch.testing.assert_close(codec.encode(changed), clean)
    output = codec.decode(clean)
    assert output['pred_boxes'].shape == (1, 3, 4)
    assert output['pred_logits'].shape == (1, 3, 1)
    assert output['pred_keypoints'].shape == (1, 3, 6)
    assert all(torch.isfinite(value).all() for value in output.values())


def test_recurrence_is_equivariant_and_all_iterations_receive_gradients():
    model = _config().build(max_epochs=2).model.eval()
    image = torch.randn(1, 3, 64, 64)
    encoded = model.encode_image(image)
    state = torch.randn(1, 3, 3, 32)
    times = torch.tensor([.5])
    output = model.denoise(state, times, encoded, collect_diagnostics=True)
    permutation = torch.tensor([2, 0, 1])
    permuted = model.denoise(state[:, permutation], times, encoded)
    torch.testing.assert_close(permuted['pred_latents'], output['pred_latents'][:, permutation], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(permuted['hypothesis_memory'], output['hypothesis_memory'], atol=2e-5, rtol=2e-5)
    repeated = model.denoise(state, times, encoded)
    torch.testing.assert_close(repeated['pred_latents'], output['pred_latents'])
    assert output['hypothesis_memory'].var(dim=1).mean() > 0
    stages = output['aux_outputs'] + [output]
    for stage in stages:
        gradients = torch.autograd.grad(stage['pred_latents'].square().mean(),
            [model.evidence_extractor.image_attention.value_proj.weight,
             model.hypothesis_update.attention.in_proj_weight], retain_graph=True)
        assert all(gradient.abs().sum() > 0 for gradient in gradients)
    sum(stage['pred_latents'].square().mean() for stage in stages).backward()
    assert model.global_token.grad.abs().sum() > 0
    assert all(layer.aggregate.attention.in_proj_weight.grad.abs().sum() > 0 for layer in model.layers)
    assert len(output['latent_diagnostics']) == 2


def test_memory_lifecycle_and_explicit_ddim_parity(monkeypatch):
    model = _config().build(max_epochs=2).model.eval()
    image = torch.randn(1, 3, 64, 64)
    encoded = model.encode_image(image)
    state = torch.randn(1, 3, 3, 32)
    initial_globals = []
    handle = model.layers[0].aggregate.register_forward_pre_hook(
        lambda module, args: initial_globals.append(args[0][:, -1].detach().clone()))
    seen_memories = []
    original = model.denoise
    def capture(*args, **kwargs):
        seen_memories.append(kwargs.get('hypothesis_memory'))
        return original(*args, **kwargs)
    monkeypatch.setattr(model, 'denoise', capture)
    sampled = model.sample_keypoints(image, initial_latents=state, encoded_image=encoded, sampling_steps=3)
    assert seen_memories[0] is None and all(value is not None for value in seen_memories[1:])
    assert all(torch.equal(value, initial_globals[0]) for value in initial_globals)
    handle.remove()
    explicit, memory = state.clone(), None
    for step in range(3):
        times = torch.tensor([1-step/3])
        output = original(explicit, times, encoded, hypothesis_memory=memory)
        explicit = ddim_update(explicit, output['pred_latents'], times, torch.tensor([1-(step+1)/3]))
        memory = output['hypothesis_memory']
    torch.testing.assert_close(sampled['latent_state'], explicit)
    torch.testing.assert_close(sampled['hypothesis_memory'], memory)
    seen_memories.clear()
    single = model.sample_keypoints(image, initial_latents=state, encoded_image=encoded, sampling_steps=1)
    assert seen_memories == [None]
    direct = original(state, torch.ones(1), encoded)
    torch.testing.assert_close(single['pred_latents'], direct['pred_latents'])
    torch.testing.assert_close(single['latent_state'], direct['pred_latents'])
    seen_memories.clear()
    reset = model.sample_keypoints(image, initial_latents=state, encoded_image=encoded,
                                  sampling_steps=3, persist_hypotheses=False)
    assert seen_memories == [None, None, None]
    assert not torch.allclose(reset['pred_latents'], sampled['pred_latents'])


def test_padding_and_batch_isolation():
    from dataclasses import replace
    model = _config().build(max_epochs=2).model.eval()
    image = torch.randn(2, 3, 64, 64)
    mask = torch.zeros(2, 64, 64, dtype=torch.bool)
    mask[:, :, 48:] = True
    encoded = model.encode_image(image, mask)
    assert encoded.padding_mask.any()
    changed_memory = encoded.memory.masked_fill(encoded.padding_mask[..., None], 1e6)
    state, times = torch.randn(2, 3, 3, 32), torch.tensor([.4, .7])
    baseline = model.denoise(state, times, encoded)
    changed = model.denoise(state, times, replace(encoded, memory=changed_memory))
    torch.testing.assert_close(baseline['pred_latents'], changed['pred_latents'])
    altered = state.clone()
    altered[1] *= 10
    second = model.denoise(altered, times, encoded)
    torch.testing.assert_close(baseline['pred_latents'][0], second['pred_latents'][0])
    torch.testing.assert_close(baseline['hypothesis_memory'][0], second['hypothesis_memory'][0])
    for invalid in [torch.zeros(2, 49, 32), torch.full((2, 50, 32), float('nan')),
                    torch.zeros(2, 50, 32, dtype=torch.float64)]:
        with pytest.raises(ValueError, match='hypothesis memory'):
            model.denoise(state, times, encoded, hypothesis_memory=invalid)


@pytest.mark.parametrize('empty', [False, True])
def test_training_masks_unmatched_and_invisible_latents_and_detaches_codec(empty):
    from src.models.pipelines.pose_estimators.poseflow.losses_v14 import matched_v14_layer_losses
    pipeline = _config().build(max_epochs=2)
    batch = _batch() | {'image': torch.randn(1, 3, 64, 64)}
    if empty:
        batch['person_mask'] = torch.zeros(1, 2, dtype=torch.bool)
    losses = pipeline._losses_for_batch(batch)
    assert all(torch.isfinite(value) for value in losses.values())
    torch.testing.assert_close(losses['total'], losses['loss_latent_mean'] + losses['loss_prediction']
                               + losses['loss_reconstruction_weighted'])
    if empty:
        assert losses['loss_latent'] == losses['loss_bbox'] == losses['loss_keypoints'] == 0
    losses['loss_latent_mean'].backward(retain_graph=True)
    assert pipeline.model.codec.box_encoder.layers[0].weight.grad is None
    pipeline.zero_grad(set_to_none=True)
    losses['total'].backward()
    assert pipeline.model.codec.box_encoder.layers[0].weight.grad.abs().sum() > 0
    assert 'clean/box_rms' in pipeline._v12_diagnostics
    assert 'clean/existence_rms' not in pipeline._v12_diagnostics
    targets = build_v12_targets(batch, groups=3)
    clean = pipeline.model.codec.encode(targets)
    predicted = torch.randn_like(clean, requires_grad=True)
    output = pipeline.model.codec.decode(predicted) | {'pred_latents': predicted}
    # Match one candidate to one target, leaving two candidates unsupervised in latent space.
    slots = targets['real'][0].nonzero().flatten()
    if not empty:
        targets['labeled'][0, slots[0], 1] = False
    matcher = lambda *args: [(torch.tensor([0]) if not empty else torch.empty(0, dtype=torch.long),
                             torch.tensor([0]) if not empty else torch.empty(0, dtype=torch.long))]
    values = matched_v14_layer_losses(output, targets, clean, pipeline.model.codec,
                                      matcher, pipeline.v12_config.weights.to_dataclass())
    values['loss_latent'].backward()
    assert predicted.grad[:, 1:].count_nonzero() == 0
    if not empty:
        assert predicted.grad[0, 0, 2].count_nonzero() == 0
        assert predicted.grad[0, 0, :2].abs().sum() > 0


def test_ota_duplicates_share_targets_and_positive_normalization():
    from src.models.pipelines.pose_estimators.poseflow.losses_v14 import matched_v14_layer_losses
    pipeline = _config().build(max_epochs=2)
    targets = build_v12_targets(_batch() | {'person_mask': torch.tensor([[True, False]])}, groups=3)
    clean = pipeline.model.codec.encode(targets)
    slot = targets['real'][0].nonzero().item()
    latent = clean[:, slot:slot+1].detach().expand(-1, 3, -1, -1).clone().requires_grad_()
    logits = torch.full((1, 3, 1), 8., requires_grad=True)
    output = {'pred_latents': latent, 'pred_logits': logits,
              'pred_boxes': targets['boxes'][:, slot:slot+1].expand(-1, 3, -1),
              'pred_keypoints': torch.cat((targets['joints'][:, slot:slot+1].flatten(-2),
                  targets['labeled'][:, slot:slot+1].float()), -1).expand(-1, 3, -1)}
    values = matched_v14_layer_losses(output, targets, clean, pipeline.model.codec,
                                      pipeline.matcher, pipeline.v12_config.weights.to_dataclass())
    assert values['positive_assignments'] == 3 and values['loss_latent'] == 0
    expected = (.25 * torch.nn.functional.softplus(-logits) * (1-logits.sigmoid())**2).mean()
    torch.testing.assert_close(values['loss_ce'], expected)
    values['loss_ce'].backward()
    assert (logits.grad < 0).all()


def test_v14_config_and_strict_evaluator_restore(tmp_path):
    from pathlib import Path
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from scripts.evaluation.evaluate_poseflow import load_poseflow_model
    from src.models.pipelines.pose_estimators.poseflow import PoseFlowPipelineConfig
    with initialize_config_dir(config_dir=str(Path('conf').resolve()), version_base=None):
        config = compose(config_name='benchmark_poseflow_v14')
    parsed = PoseFlowPipelineConfig.from_hydra_config(config.pipeline)
    assert isinstance(parsed, PoseFlowV14PipelineConfig)
    assert parsed.matching_strategy == 'ota' and parsed.ota_candidate_topk == 5
    assert parsed.hidden_dim == 256 and parsed.num_proposals == 50 and parsed.num_decoder_layers == 3
    assert config.training.grad_clip == 2 and config.training.resume is None
    assert parsed.reconstruction_weight_end == .01 and parsed.reconstruction_decay_epochs == 40
    assert config.out_dir == 'results/poseflow_v14_training'
    pipeline = _config().build(max_epochs=2).eval()
    torch.save({'state_dict': pipeline.state_dict()}, tmp_path/'model.ckpt')
    OmegaConf.save(OmegaConf.create({'pipeline': {'_target_':
        'src.models.pipelines.pose_estimators.poseflow.PoseFlowV14PipelineConfig',
        **pipeline.v12_config.model_dump(mode='json')}}), tmp_path/'config.yaml')
    restored, _ = load_poseflow_model(tmp_path/'model.ckpt', tmp_path/'config.yaml')
    restored.eval()
    image = torch.randn(1, 3, 64, 64)
    initial = torch.randn(1, 3, 3, 32)
    torch.testing.assert_close(pipeline.model.sample_keypoints(image, initial_latents=initial)['pred_latents'],
                               restored.sample_keypoints(image, initial_latents=initial)['pred_latents'])


def test_quick_grid_pairs_persistent_and_reset_memory(tmp_path):
    from src.models.pipelines.pose_estimators.poseflow.quick_eval_v13 import evaluate_quick_grid
    model = _config().build(max_epochs=2).model.train()
    batch = _batch() | {'image': torch.randn(1, 3, 64, 64), 'image_id': torch.tensor([7]),
                       'img_hw': torch.tensor([[64., 64.]]), 'person_mask': torch.ones(1, 2, dtype=torch.bool)}
    boxes = batch['gt_boxes_canon']
    batch['gt_boxes_pixel'] = torch.cat((boxes[..., :2]-boxes[..., 2:]/2, boxes[..., 2:]), -1)*64
    starts = []
    original = model.sample_keypoints
    def capture(*args, **kwargs):
        starts.append((kwargs['persist_hypotheses'], kwargs['initial_latents'].clone()))
        return original(*args, **kwargs)
    model.sample_keypoints = capture
    rng = torch.get_rng_state().clone()
    report = evaluate_quick_grid(model, [batch], _config(), tmp_path, 'test.json', [7],
        hypothesis_modes=('persistent', 'reset'),
        metric_evaluator=lambda *args: {'pose_AP': 0., 'bbox_AP': 0.})
    assert len(report['conditions']) == 18
    assert model.training and torch.equal(rng, torch.get_rng_state())
    for index in range(0, len(starts), 2):
        assert starts[index][0] is True and starts[index+1][0] is False
        torch.testing.assert_close(starts[index][1], starts[index+1][1], atol=0, rtol=0)
    assert all('raw_bbox_coverage50' in row['metrics'] and 'steps_quality' in row
               for row in report['conditions'].values())


def test_box_diagnostics_distinguish_coverage_duplicates_and_confidence():
    from src.models.pipelines.pose_estimators.poseflow.quick_eval_v14 import box_quality, aggregate_box_quality
    from src.models.pipelines.pose_estimators.poseflow.optimization_v13 import parameter_component
    batch = _batch() | {'person_mask': torch.tensor([[True, False]])}
    target = batch['gt_boxes_canon'][0, 0]
    output = {'pred_boxes': torch.stack((target, target, torch.tensor([.9, .9, .05, .05])))[None],
              'pred_logits': torch.logit(torch.tensor([.8, .6, .1]))[None, :, None]}
    metrics = aggregate_box_quality([box_quality(output, batch)])
    assert metrics['raw_bbox_coverage50'] == metrics['raw_bbox_coverage75'] == 1
    assert metrics['raw_bbox_duplicate_fraction50'] == pytest.approx(1/3)
    assert metrics['raw_good_bbox_score_mean'] == pytest.approx(.7)
    assert metrics['raw_poor_bbox_score_mean'] == pytest.approx(.1)
    assert parameter_component('model.latent_head.weight') == 'latent_heads'


def test_v14_quick_report_declares_memory_modes_and_target_contract(monkeypatch):
    from src.models.pipelines.pose_estimators.poseflow import quick_eval_v13
    pipeline = _config().build(max_epochs=2)
    arguments = []
    def capture(*args, **kwargs):
        arguments.append(kwargs)
        return {'conditions': {}}
    monkeypatch.setattr(quick_eval_v13, 'evaluate_quick_grid', capture)
    report = pipeline._evaluate_quick_grid()
    assert arguments == [{'hypothesis_modes': ('persistent', 'reset')}]
    assert 'person_mask' in report['raw_box_quality_contract']['targets']
    assert pipeline._quick_architecture()['state_tokens'] == 3
    assert pipeline._quick_architecture()['hypothesis_slots'] == 50


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_backward_and_mixed_precision_memory_carry():
    pipeline = _config().build(max_epochs=2).cuda()
    batch = {key: value.cuda() for key, value in _batch().items()}
    batch['image'] = torch.randn(1, 3, 64, 64, device='cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        losses = pipeline._losses_for_batch(batch)
    losses['total'].backward()
    assert torch.isfinite(losses['total'])
    assert pipeline.model.evidence_extractor.image_attention.value_proj.weight.grad.abs().sum() > 0
    pipeline.eval()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        output = pipeline.model.sample_keypoints(batch['image'], collect_step_diagnostics=True)
    assert output['latent_state'].shape == (1, 3, 3, 32)
    assert len(output['latent_step_diagnostics']) == 3
    assert torch.isfinite(output['hypothesis_memory']).all()

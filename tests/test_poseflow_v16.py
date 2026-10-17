import torch

from tests.test_poseflow import _batch
from tests.test_poseflow_v15 import _config as v15_config
from src.models.pipelines.pose_estimators.poseflow.latent_v12 import build_v12_targets, ddim_update
from src.models.pipelines.pose_estimators.poseflow.latent_v16 import V16StateCodec
from src.models.pipelines.pose_estimators.poseflow.training_v16 import PoseFlowV16PipelineConfig


def _config(**overrides):
    return PoseFlowV16PipelineConfig(**(v15_config().model_dump() | overrides))


def test_codec_separates_confidence_center_size_and_joints():
    targets = build_v12_targets(_batch(), groups=3)
    codec = V16StateCodec(width=32, joints=2)
    clean = codec.encode(targets)
    assert clean.shape == (1, 3, 5, 32)
    for role, name in enumerate(('existence', 'center', 'size')):
        changed = {key: value.clone() for key, value in targets.items()}
        if role == 0:
            changed['existence'] = 1-changed['existence']
        else:
            changed['boxes'][..., :2 if role == 1 else 4] += .1
            if role == 2:
                changed['boxes'][..., :2] = targets['boxes'][..., :2]
        encoded = codec.encode(changed)
        assert not torch.equal(encoded[:, :, role], clean[:, :, role])
        for other in set(range(5))-{role}:
            torch.testing.assert_close(encoded[:, :, other], clean[:, :, other])
    output = codec.decode(clean)
    assert output['pred_keypoints'].shape == (1, 3, 6)
    assert all(torch.isfinite(value).all() for value in output.values())
    altered = clean.clone()
    altered[:, :, 0] += torch.randn_like(altered[:, :, 0])
    torch.testing.assert_close(codec.decode(altered)['pred_boxes'], output['pred_boxes'])
    altered = clean.clone()
    altered[:, :, 1] += torch.randn_like(altered[:, :, 1])
    torch.testing.assert_close(codec.decode(altered)['pred_boxes'][..., 2:], output['pred_boxes'][..., 2:])
    torch.testing.assert_close(codec.decode(altered)['pred_logits'], output['pred_logits'])


def test_recurrence_memory_equivariance_and_gradients():
    model = _config().build(max_epochs=2).model.eval()
    image, state = torch.randn(1, 3, 64, 64), torch.randn(1, 3, 5, 32)
    encoded = model.encode_image(image)
    direct = model.denoise(state, torch.ones(1), encoded, collect_diagnostics=True)
    assert direct['pred_latents'].shape == state.shape
    assert len(direct['aux_outputs']) == 1
    for stage in direct['aux_outputs']+[direct]:
        gradient, = torch.autograd.grad(stage['pred_latents'].square().mean(),
            model.evidence_extractor.image_attention.value_proj.weight, retain_graph=True)
        assert gradient.abs().sum() > 0
    direct['pred_latents'].square().mean().backward()
    assert all(layer.aggregate.attention.in_proj_weight.grad.abs().sum() > 0 for layer in model.layers)
    assert model.hypothesis_update.retention_logit.grad.abs() > 0
    single = model.sample_keypoints(image, encoded_image=encoded, initial_latents=state, sampling_steps=1)
    torch.testing.assert_close(single['latent_state'], direct['pred_latents'])
    permutation = torch.tensor([2, 0, 1])
    permuted = model.denoise(state[:, permutation], torch.ones(1), encoded)
    torch.testing.assert_close(permuted['pred_latents'], direct['pred_latents'][:, permutation], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(permuted['hypothesis_memory'], direct['hypothesis_memory'], atol=2e-5, rtol=2e-5)
    explicit, memory = state, None
    for step in range(3):
        times, next_times = torch.tensor([1-step/3]), torch.tensor([1-(step+1)/3])
        output = model.denoise(explicit, times, encoded, hypothesis_memory=memory)
        explicit = ddim_update(explicit, output['pred_latents'], times, next_times)
        memory = output['hypothesis_memory']
    sampled = model.sample_keypoints(image, encoded_image=encoded, initial_latents=state, sampling_steps=3)
    torch.testing.assert_close(sampled['latent_state'], explicit)
    torch.testing.assert_close(sampled['hypothesis_memory'], memory)


def test_losses_mask_unmatched_and_invisible_and_preserve_family_balance():
    from src.models.pipelines.pose_estimators.poseflow.losses_v14 import matched_v14_layer_losses
    pipeline = _config().build(max_epochs=2)
    targets = build_v12_targets(_batch(), groups=3)
    slot = targets['real'][0].nonzero().flatten()[0]
    targets['labeled'][0, slot, 1] = False
    clean = pipeline.model.codec.encode(targets)
    predicted = torch.randn_like(clean, requires_grad=True)
    output = pipeline.model.codec.decode(predicted) | {'pred_latents': predicted}
    matcher = lambda *args: [(torch.tensor([0]), torch.tensor([0]))]
    losses = matched_v14_layer_losses(output, targets, clean, pipeline.model.codec, matcher,
                                      pipeline.v12_config.weights.to_dataclass())
    torch.testing.assert_close(losses['loss_latent_box'],
        (losses['loss_latent_confidence']+losses['loss_latent_center']+losses['loss_latent_size'])/3)
    torch.testing.assert_close(losses['loss_latent'],
        (losses['loss_latent_box']+losses['loss_latent_joints'])/2)
    losses['loss_latent'].backward()
    assert predicted.grad[:, 1:].count_nonzero() == 0
    assert predicted.grad[0, 0, 4].count_nonzero() == 0
    assert all(predicted.grad[0, 0, role].abs().sum() > 0 for role in range(4))
    assert all(parameter.grad is None for parameter in pipeline.model.codec.parameters())
    for empty in (False, True):
        batch = _batch() | {'image': torch.randn(1, 3, 64, 64)}
        if empty:
            batch['person_mask'] = torch.zeros(1, 2, dtype=torch.bool)
        losses = pipeline._losses_for_batch(batch)
        assert all(torch.isfinite(value) for value in losses.values())
        if empty:
            assert losses['loss_latent'] == losses['loss_bbox'] == losses['loss_keypoints'] == 0
        pipeline.zero_grad(set_to_none=True)
        losses['total'].backward()
        assert pipeline.model.codec.center_encoder.layers[0].weight.grad.abs().sum() > 0
        assert 'clean/confidence_rms' in pipeline._v12_diagnostics


def test_box_role_references_padding_global_reset_and_batch_isolation():
    from dataclasses import replace
    model = _config().build(max_epochs=2).model.eval()
    images = torch.randn(2, 3, 64, 64)
    mask = torch.zeros(2, 64, 64, dtype=torch.bool)
    mask[:, :, 48:] = True
    encoded = model.encode_image(images, mask)
    state, times = torch.randn(2, 3, 5, 32), torch.tensor([.4, .7])
    references, geometries, globals_in = [], [], []
    handles = [model.evidence_extractor.image_attention.register_forward_pre_hook(
        lambda module, args: references.append(args[1].clone())),
        model.evidence_extractor.register_forward_pre_hook(
        lambda module, args: geometries.append(args[1])),
        model.layers[0].state_update.register_forward_pre_hook(
        lambda module, args: globals_in.append(args[0].reshape(2, 3, 6, 32)[:, :, -1].clone()))]
    output = model.denoise(state, times, encoded)
    for index, geometry in enumerate(geometries):
        ratios = encoded.valid_ratios[:, None]
        boxes = geometry['pred_boxes'][:, :, None].expand(-1, -1, 3, -1).flatten(1, 2)[:, :, None]
        torch.testing.assert_close(references[2*index], boxes*torch.cat((ratios, ratios), -1))
        joints = geometry['pred_keypoints'][..., :4].reshape(2, 3, 2, 2).flatten(1, 2)[:, :, None]
        torch.testing.assert_close(references[2*index+1], joints*ratios)
    continued = model.denoise(state, times, encoded, hypothesis_memory=output['hypothesis_memory'])
    torch.testing.assert_close(globals_in[0], globals_in[1])
    assert not torch.equal(continued['hypothesis_memory'], output['hypothesis_memory'])
    for handle in handles:
        handle.remove()
    changed_image = replace(encoded, memory=encoded.memory.masked_fill(encoded.padding_mask[..., None], 1e6))
    masked = model.denoise(state, times, changed_image)
    torch.testing.assert_close(masked['pred_latents'], output['pred_latents'])
    changed_state = state.clone()
    changed_state[1] *= 10
    isolated = model.denoise(changed_state, times, encoded)
    torch.testing.assert_close(isolated['pred_latents'][0], output['pred_latents'][0])
    torch.testing.assert_close(isolated['hypothesis_memory'][0], output['hypothesis_memory'][0])


def test_duplicate_assignments_and_strict_checkpoint_restore(tmp_path, monkeypatch):
    from pathlib import Path
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from scripts.evaluation.evaluate_poseflow import load_poseflow_model
    from src.models.pipelines.pose_estimators.poseflow import PoseFlowPipelineConfig
    import tests.test_poseflow_v14 as baseline
    monkeypatch.setattr(baseline, '_config', _config)
    baseline.test_ota_duplicates_share_targets_and_positive_normalization()
    with initialize_config_dir(config_dir=str(Path('conf').resolve()), version_base=None):
        config = compose(config_name='benchmark_poseflow_v16')
    parsed = PoseFlowPipelineConfig.from_hydra_config(config.pipeline)
    assert isinstance(parsed, PoseFlowV16PipelineConfig)
    assert config.training.max_epochs == 80 and config.training.grad_clip == 2
    assert config.data.batch_size == 4 and config.training.accumulate_grad_batches == 2
    assert parsed.matching_strategy == 'ota' and parsed.ota_candidate_topk == 5
    assert config.out_dir == 'results/poseflow_v16_training'
    pipeline = _config().build(max_epochs=2).eval()
    checkpoint = tmp_path/'model.ckpt'
    torch.save({'state_dict': pipeline.state_dict()}, checkpoint)
    config_path = tmp_path/'config.yaml'
    OmegaConf.save(OmegaConf.create({'pipeline': {'_target_':
        'src.models.pipelines.pose_estimators.poseflow.PoseFlowV16PipelineConfig',
        **pipeline.v12_config.model_dump(mode='json')}}), config_path)
    restored, _ = load_poseflow_model(checkpoint, config_path)
    restored.eval()
    state, image = torch.randn(1, 3, 5, 32), torch.randn(1, 3, 64, 64)
    torch.testing.assert_close(restored.sample_keypoints(image, initial_latents=state)['pred_latents'],
        pipeline.model.sample_keypoints(image, initial_latents=state)['pred_latents'])
    assert pipeline._quick_architecture()['state_tokens'] == 5
    assert pipeline.model.initialization_report['decoder_transferred'] == []


def test_direct_image_route_with_hypothesis_messages_disabled_and_shared_updates():
    model = _config().build(max_epochs=2).model.eval()
    extractor_ids, memory_ids = [], []
    handles = [model.evidence_extractor.register_forward_hook(
        lambda module, args, output: extractor_ids.append(id(module))),
        model.hypothesis_update.register_forward_hook(
        lambda module, args, output: memory_ids.append(id(module)))]
    for layer in model.layers:
        with torch.no_grad():
            layer.state_update.attention.out_proj.weight.zero_()
            layer.state_update.attention.out_proj.bias.zero_()
    output = model(torch.randn(1, 3, 64, 64), torch.randn(1, 3, 5, 32), torch.ones(1))
    for stage in output['aux_outputs']+[output]:
        gradient, = torch.autograd.grad(stage['pred_boxes'].sum()+stage['pred_keypoints'].sum(),
            model.evidence_extractor.image_attention.value_proj.weight, retain_graph=True)
        assert gradient.abs().sum() > 0
    assert len(extractor_ids) == len(model.layers) and len(set(extractor_ids)) == 1
    assert len(memory_ids) == len(model.layers) and len(set(memory_ids)) == 1
    for handle in handles:
        handle.remove()

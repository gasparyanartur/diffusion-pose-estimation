import math
import pytest
import torch
from tests.test_poseflow_v14 import _config as v14_config
from src.models.pipelines.pose_estimators.poseflow.training_v15 import PoseFlowV15PipelineConfig
from src.models.pipelines.pose_estimators.poseflow.model_v15 import V15HypothesisUpdate


def _config(**overrides):
    return PoseFlowV15PipelineConfig(**(v14_config().model_dump() | overrides))


def test_gate_arithmetic_and_gradient():
    update = V15HypothesisUpdate(32, 4, 0.)
    memory, globals = torch.randn(2, 50, 32), torch.randn(2, 3, 32)
    assert update.retention_logit.item() == torch.tensor(math.log(9)).item()
    evidence = update.evidence_norm(update.attention(memory, globals, globals, need_weights=False)[0])
    actual, _ = update(memory, globals)
    torch.testing.assert_close(actual, .9*memory + .1*evidence)
    actual.square().mean().backward()
    assert update.retention_logit.grad.abs() > 0


def test_local_route_and_updated_geometry():
    model = _config().build(max_epochs=2).model.eval()
    image = torch.randn(1, 3, 64, 64)
    encoded = model.encode_image(image)
    state = torch.randn(1, 3, 3, 32)
    references, expected = [], []
    for layer in model.layers:
        # Disable only the cross-attention message; retain residual/FFN computation.
        layer.state_update.attention.out_proj.weight.data.zero_()
        layer.state_update.attention.out_proj.bias.data.zero_()
        layer.state_update.register_forward_hook(lambda module,args,out: expected.append(
            model.codec.decode(model.latent_head(out[0].reshape(1,3,4,32)[:,:,:-1]))['pred_boxes']))
    model.evidence_extractor.register_forward_pre_hook(lambda module,args: references.append(args[1]['pred_boxes']))
    output = model.denoise(state, torch.ones(1), encoded, collect_diagnostics=True)
    for actual, wanted in zip(references, expected):
        torch.testing.assert_close(actual, wanted)
    for stage in output['aux_outputs']+[output]:
        gradient, = torch.autograd.grad(stage['pred_latents'].square().mean(),
            model.evidence_extractor.image_attention.value_proj.weight, retain_graph=True)
        assert gradient.abs().sum() > 0
    sum(stage['pred_latents'].square().mean() for stage in output['aux_outputs']+[output]).backward()
    assert model.global_token.grad.abs().sum() > 0
    assert all(layer.aggregate.attention.in_proj_weight.grad.abs().sum() > 0 for layer in model.layers)
    assert 'hypothesis_attention_entropy_per_head' in output['latent_diagnostics'][0]


def test_memory_lifecycle_equivariance_and_ddim():
    from src.models.pipelines.pose_estimators.poseflow.latent_v12 import ddim_update
    model = _config().build(max_epochs=2).model.eval()
    image, state = torch.randn(1, 3, 64, 64), torch.randn(1, 3, 3, 32)
    encoded = model.encode_image(image)
    initial_globals = []
    handle = model.layers[0].state_update.register_forward_pre_hook(
        lambda module,args: initial_globals.append(args[0][:,3::4].detach().clone()))
    sampled = model.sample_keypoints(image, encoded_image=encoded, initial_latents=state, sampling_steps=3)
    assert len(initial_globals) == 3
    assert all(torch.equal(value, initial_globals[0]) for value in initial_globals)
    handle.remove()
    explicit, memory = state, None
    for step in range(3):
        out = model.denoise(explicit, torch.tensor([1-step/3]), encoded, hypothesis_memory=memory)
        memory = out['hypothesis_memory']
        explicit = ddim_update(explicit, out['pred_latents'], torch.tensor([1-step/3]), torch.tensor([1-(step+1)/3]))
    torch.testing.assert_close(sampled['latent_state'], explicit)
    torch.testing.assert_close(sampled['hypothesis_memory'], memory)
    again = model.sample_keypoints(image, encoded_image=encoded, initial_latents=state, sampling_steps=3)
    torch.testing.assert_close(sampled['pred_latents'], again['pred_latents'])
    direct = model.denoise(state, torch.ones(1), encoded)
    single = model.sample_keypoints(image, encoded_image=encoded, initial_latents=state, sampling_steps=1)
    torch.testing.assert_close(single['latent_state'], direct['pred_latents'])
    permutation = torch.tensor([2,0,1])
    changed = model.denoise(state[:,permutation], torch.ones(1), encoded)
    torch.testing.assert_close(changed['pred_latents'], direct['pred_latents'][:,permutation], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(changed['hypothesis_memory'], direct['hypothesis_memory'], atol=2e-5, rtol=2e-5)
    gradient, = torch.autograd.grad(direct['hypothesis_memory'].square().mean(), model.hypothesis_update.retention_logit)
    assert gradient.abs() > 0


def test_v15_config_and_strict_restore(tmp_path):
    from pathlib import Path
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from scripts.evaluation.evaluate_poseflow import load_poseflow_model
    from src.models.pipelines.pose_estimators.poseflow import PoseFlowPipelineConfig
    with initialize_config_dir(config_dir=str(Path('conf').resolve()), version_base=None):
        config = compose(config_name='benchmark_poseflow_v15')
    parsed = PoseFlowPipelineConfig.from_hydra_config(config.pipeline)
    assert isinstance(parsed, PoseFlowV15PipelineConfig)
    assert parsed.num_proposals == 50 and parsed.hidden_dim == 256 and parsed.num_decoder_layers == 3
    assert parsed.matching_strategy == 'ota' and parsed.ota_candidate_topk == 5
    assert parsed.num_heads == 8 and parsed.feedforward_dim == 2048
    assert parsed.num_body_points == 14 and parsed.num_feature_levels == 4
    assert config.training.grad_clip == 2 and config.training.max_epochs == 80
    assert config.data.batch_size == 4 and config.training.accumulate_grad_batches == 2
    assert config.out_dir == 'results/poseflow_v15_training'
    pipeline = _config().build(max_epochs=2).eval()
    torch.save({'state_dict':pipeline.state_dict()},tmp_path/'model.ckpt')
    OmegaConf.save(OmegaConf.create({'pipeline':{'_target_':'src.models.pipelines.pose_estimators.poseflow.PoseFlowV15PipelineConfig',**pipeline.v12_config.model_dump(mode='json')}}),tmp_path/'config.yaml')
    model, _ = load_poseflow_model(tmp_path/'model.ckpt',tmp_path/'config.yaml')
    model.eval()
    image,state=torch.randn(1,3,64,64),torch.randn(1,3,3,32)
    torch.testing.assert_close(model.sample_keypoints(image,initial_latents=state)['pred_latents'],pipeline.model.sample_keypoints(image,initial_latents=state)['pred_latents'])


@pytest.mark.parametrize('name', [
    'test_padding_and_batch_isolation',
    'test_ota_duplicates_share_targets_and_positive_normalization',
    'test_quick_grid_pairs_persistent_and_reset_memory',
    'test_cuda_backward_and_mixed_precision_memory_carry',
])
def test_inherited_contracts(name, monkeypatch, tmp_path):
    import tests.test_poseflow_v14 as baseline
    monkeypatch.setattr(baseline, '_config', _config)
    if name == 'test_cuda_backward_and_mixed_precision_memory_carry' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    test = getattr(baseline, name)
    if name == 'test_quick_grid_pairs_persistent_and_reset_memory':
        test(tmp_path)
    else:
        test()


@pytest.mark.parametrize('empty', [False, True])
def test_training_masks(empty, monkeypatch):
    import tests.test_poseflow_v14 as baseline
    monkeypatch.setattr(baseline, '_config', _config)
    baseline.test_training_masks_unmatched_and_invisible_latents_and_detaches_codec(empty)


def test_shared_blocks_carried_globals_and_memory_gradients():
    model = _config().build(max_epochs=2).model.eval()
    globals_in, globals_out, extractor_ids, updater_ids = [], [], [], []
    handles = []
    for layer in model.layers:
        handles.append(layer.state_update.register_forward_pre_hook(
            lambda module,args: globals_in.append(args[0][:,3::4].clone())))
        handles.append(layer.aggregate.register_forward_hook(
            lambda module,args,out: globals_out.append(out[0][:,-1].reshape(1,3,32))))
    handles.append(model.evidence_extractor.register_forward_hook(lambda module,args,out: extractor_ids.append(id(module))))
    handles.append(model.hypothesis_update.register_forward_hook(lambda module,args,out: updater_ids.append(id(module))))
    output = model(torch.randn(1,3,64,64), torch.randn(1,3,3,32), torch.tensor([.5]))
    torch.testing.assert_close(globals_in[1], globals_out[0])
    assert not torch.equal(globals_out[0], globals_in[0])
    assert extractor_ids == [id(model.evidence_extractor)]*len(model.layers)
    assert updater_ids == [id(model.hypothesis_update)]*len(model.layers)
    output['pred_latents'].square().mean().backward()
    assert model.hypothesis_update.attention.in_proj_weight.grad.abs().sum() > 0
    assert model.hypothesis_update.retention_logit.grad.abs() > 0
    assert all(layer.state_update.attention.in_proj_weight.grad.abs().sum() > 0 for layer in model.layers)
    for handle in handles:
        handle.remove()


def test_local_residual_once_and_reference_ratios():
    model = _config().build(max_epochs=2).model.eval()
    extractor = model.evidence_extractor
    image = torch.randn(1,3,64,64)
    mask = torch.zeros(1,64,64,dtype=torch.bool)
    mask[:,:,48:] = True
    encoded = model.encode_image(image,mask)
    state = torch.randn(1,3,3,32)
    geometry = model.codec.decode(state)
    references = []
    def zero_message(module,args,out):
        references.append(args[1])
        return torch.zeros_like(out)
    handle = extractor.image_attention.register_forward_hook(zero_message)
    extractor.norm = extractor.ffn_norm = torch.nn.Identity()
    with torch.no_grad():
        for parameter in extractor.ffn.parameters():
            parameter.zero_()
    actual = extractor(state,geometry,encoded)
    torch.testing.assert_close(actual,state)
    ratios = encoded.valid_ratios[:,None]
    torch.testing.assert_close(references[0],geometry['pred_boxes'][:,:,None]*torch.cat((ratios,ratios),-1))
    joints=geometry['pred_keypoints'][...,:4].reshape(1,6,2)
    torch.testing.assert_close(references[1],joints[:,:,None]*ratios)
    handle.remove()

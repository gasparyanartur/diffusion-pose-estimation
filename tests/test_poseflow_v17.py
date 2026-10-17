import torch
from tests.test_poseflow_v15_rollout import _config as rollout_config
from src.models.pipelines.pose_estimators.poseflow.training_v17 import PoseFlowV17PipelineConfig
from src.models.pipelines.pose_estimators.poseflow.latent_v12 import ddim_update


def _config(**overrides):
    return PoseFlowV17PipelineConfig(**(rollout_config().model_dump() | overrides))


def test_tied_recurrence_and_removed_modules():
    model = _config().build(max_epochs=2).model.eval()
    assert model.num_hypotheses == model.num_proposals == 3
    assert not any('retention' in k or 'hypothesis_update' in k or 'state_update' in k for k in model.state_dict())
    image, state = torch.randn(1, 3, 64, 64), torch.randn(1, 3, 3, 32)
    local_inputs, local_outputs, bank_outputs, image_inputs = [], [], [], []
    for layer in model.layers:
        layer.register_forward_pre_hook(lambda m,args: local_inputs.append(args[0]))
        layer.register_forward_hook(lambda m,args,out: local_outputs.append(out[0]))
    model.hypothesis_attention.register_forward_hook(lambda m,args,out: bank_outputs.append(out[0]))
    model.evidence_extractor.register_forward_pre_hook(lambda m,args:image_inputs.append(args[0]))
    out = model.denoise(state, torch.ones(1), model.encode_image(image), collect_diagnostics=True)
    assert len(bank_outputs) == len(model.layers)
    for i, tokens in enumerate(local_inputs):
        assert tokens.shape == (3, 5, 32)
        torch.testing.assert_close(tokens[:, -1], bank_outputs[i].reshape(3,32))
        if i:
            torch.testing.assert_close(image_inputs[i], local_outputs[i-1].reshape(1,3,5,32)[:, :, :3])
            torch.testing.assert_close(tokens[:, -2], local_outputs[i-1][:, -2])
    torch.testing.assert_close(out['hypothesis_memory'], local_outputs[-1][:,-1].reshape(1,3,32))
    assert 'memory_retention' not in out['latent_diagnostics'][0]
    assert 'local_hypothesis_read_mass' in out['latent_diagnostics'][0]
    assert all(torch.isfinite(v).all() for v in [out['pred_latents'],out['hypothesis_memory']])


def test_candidate_pairing_and_global_communication():
    model = _config().build(max_epochs=2).model.eval()
    tokens = torch.randn(3,5,32)
    before = model.layers[0](tokens,tokens)[0]
    changed = tokens.clone();changed[1] += torch.randn_like(changed[1])
    torch.testing.assert_close(model.layers[0](changed,changed)[0][0],before[0])
    image,state = torch.randn(1,3,64,64),torch.randn(1,3,3,32)
    memory = torch.randn(1,3,32,requires_grad=True)
    encoded = model.encode_image(image)
    out = model.denoise(state,torch.ones(1),encoded,hypothesis_memory=memory)
    gradient, = torch.autograd.grad(out['pred_latents'][:,0].square().mean(),memory)
    assert gradient[:,1:].abs().sum()>0
    permutation = torch.tensor([2,0,1])
    other = model.denoise(state[:,permutation],torch.ones(1),encoded,hypothesis_memory=memory[:,permutation])
    torch.testing.assert_close(other['pred_latents'],out['pred_latents'][:,permutation],atol=3e-5,rtol=3e-5)
    torch.testing.assert_close(other['hypothesis_memory'],out['hypothesis_memory'][:,permutation],atol=3e-5,rtol=3e-5)


def test_memory_global_lifecycle_ddim_and_gradients():
    model = _config().build(max_epochs=2).model.eval()
    image,state = torch.randn(1,3,64,64),torch.randn(1,3,3,32)
    encoded = model.encode_image(image)
    globals_seen=[]
    handle=model.layers[0].register_forward_pre_hook(lambda m,args:globals_seen.append(args[0][:,-2].clone()))
    actual=model.sample_keypoints(image,encoded_image=encoded,initial_latents=state,sampling_steps=3)
    handle.remove()
    assert len(globals_seen)==3
    for value in globals_seen:torch.testing.assert_close(value,model.global_token.expand(1,3,32).reshape(3,32))
    memory=None;current=state;first=None
    for i in range(3):
        out=model.denoise(current,torch.tensor([1-i/3]),encoded,hypothesis_memory=memory)
        if first is None:first=out
        memory=out['hypothesis_memory']
        current=ddim_update(current,out['pred_latents'],torch.tensor([1-i/3]),torch.tensor([1-(i+1)/3]))
    torch.testing.assert_close(actual['latent_state'],current)
    torch.testing.assert_close(actual['hypothesis_memory'],memory)
    memory_grad,latent_grad=torch.autograd.grad(out['pred_latents'].square().mean(),
        [first['hypothesis_memory'],first['pred_latents']],retain_graph=True)
    assert memory_grad.abs().sum()>0 and latent_grad.abs().sum()>0
    out['pred_latents'].square().mean().backward()
    assert model.hypothesis_attention.attention.in_proj_weight.grad.abs().sum()>0
    assert all(layer.attention.in_proj_weight.grad.abs().sum()>0 for layer in model.layers)
    assert model.evidence_extractor.image_attention.value_proj.weight.grad.abs().sum()>0
    again=model.sample_keypoints(image,encoded_image=encoded,initial_latents=state,sampling_steps=3)
    torch.testing.assert_close(actual['pred_latents'],again['pred_latents'])
    direct=model.denoise(state,torch.ones(1),encoded)
    single=model.sample_keypoints(image,encoded_image=encoded,initial_latents=state,sampling_steps=1)
    torch.testing.assert_close(direct['pred_latents'],single['latent_state'])


def test_v17_config_and_strict_restore(tmp_path):
    from pathlib import Path
    from hydra import compose,initialize_config_dir
    from omegaconf import OmegaConf
    from scripts.evaluation.evaluate_poseflow import load_poseflow_model
    from src.models.pipelines.pose_estimators.poseflow import PoseFlowPipelineConfig
    with initialize_config_dir(config_dir=str(Path('conf').resolve()),version_base=None):
        resolved=compose(config_name='benchmark_poseflow_v17')
    config=PoseFlowPipelineConfig.from_hydra_config(resolved.pipeline)
    assert isinstance(config,PoseFlowV17PipelineConfig)
    assert config.rollout_probability==.5 and config.rollout_call_counts==[2,3]
    assert resolved.training.max_epochs==80 and resolved.data.batch_size==4
    pipeline=_config().build(max_epochs=2).eval()
    architecture=pipeline._quick_architecture()
    assert architecture['decoder_representation_tokens']==5
    assert 'hypothesis_update' not in architecture['shared_components']
    torch.save({'state_dict':pipeline.state_dict()},tmp_path/'model.ckpt')
    OmegaConf.save(OmegaConf.create({'pipeline':pipeline.v12_config.model_dump()|{
        '_target_':'src.models.pipelines.pose_estimators.poseflow.PoseFlowV17PipelineConfig'}}),tmp_path/'config.yaml')
    model,_=load_poseflow_model(tmp_path/'model.ckpt',tmp_path/'config.yaml');model.eval()
    image,state=torch.randn(1,3,64,64),torch.randn(1,3,3,32)
    for steps in [1,3]:
        expected=pipeline.model.sample_keypoints(image,initial_latents=state,sampling_steps=steps)
        actual=model.sample_keypoints(image,initial_latents=state,sampling_steps=steps)
        torch.testing.assert_close(expected['pred_latents'],actual['pred_latents'])
        torch.testing.assert_close(expected['hypothesis_memory'],actual['hypothesis_memory'])


def test_rollout_loss_with_empty_and_invisible_targets():
    from tests.test_poseflow import _batch
    pipeline = _config(rollout_probability=1, rollout_call_counts=[3]).build(max_epochs=2)
    for empty in [False, True]:
        batch = _batch() | {'image': torch.randn(1,3,64,64)}
        if empty:
            batch['person_mask'] = torch.zeros(1,2,dtype=torch.bool)
        else:
            batch['gt_labeled'][:,:,1] = False
        losses = pipeline._losses_for_batch(batch)
        assert all(torch.isfinite(value) for value in losses.values())
        torch.testing.assert_close(losses['total'], losses['loss_latent_mean']+
            losses['loss_prediction']+losses['loss_reconstruction_weighted'])
        if empty:
            assert losses['loss_latent'] == losses['loss_bbox'] == losses['loss_keypoints'] == 0
        pipeline.zero_grad(set_to_none=True)
        losses['total'].backward()
        assert pipeline.model.hypothesis_attention.attention.in_proj_weight.grad.abs().sum()>0


def test_batch_isolation_and_memory_validation():
    import pytest
    model = _config().build(max_epochs=2).model.eval()
    image,state = torch.randn(2,3,64,64),torch.randn(2,3,3,32)
    memory = torch.randn(2,3,32)
    encoded = model.encode_image(image)
    first = model.denoise(state,torch.ones(2),encoded,hypothesis_memory=memory)
    changed_state,changed_memory = state.clone(),memory.clone()
    changed_state[1] = torch.randn_like(state[1]);changed_memory[1] = torch.randn_like(memory[1])
    second = model.denoise(changed_state,torch.ones(2),encoded,hypothesis_memory=changed_memory)
    torch.testing.assert_close(first['pred_latents'][0],second['pred_latents'][0])
    torch.testing.assert_close(first['hypothesis_memory'][0],second['hypothesis_memory'][0])
    for invalid in [torch.zeros(2,50,32), memory.double(), memory*float('nan')]:
        with pytest.raises(ValueError,match='memory'):
            model.denoise(state,torch.ones(2),encoded,hypothesis_memory=invalid)

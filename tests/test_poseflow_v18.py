import torch
from torch import nn
from tests.test_poseflow_v17 import _config as v17_config
from src.models.pipelines.pose_estimators.poseflow.model_v18 import PreImageRefinement
from src.models.pipelines.pose_estimators.poseflow.training_v18 import PoseFlowV18PipelineConfig


def _config(**overrides):
    return PoseFlowV18PipelineConfig(**(v17_config().model_dump() | overrides))


def test_refinement_arithmetic_and_all_role_propagation():
    block = PreImageRefinement(32,64,0.)
    inputs = torch.randn(1,3,4,32,requires_grad=True)
    normalized = block.input_norm(inputs)
    expected = block.output_norm(normalized+block.ffn(normalized))
    torch.testing.assert_close(block(inputs),expected)
    block(inputs).square().mean().backward()
    assert inputs.grad.abs().sum()>0
    model = _config().build(max_epochs=2).model.eval()
    before,refined,evidence,geometry,local = [],[],[],[],[]
    for layer in model.pre_image_updates:
        layer.register_forward_pre_hook(lambda m,args:before.append(args[0]))
        layer.register_forward_hook(lambda m,args,out:refined.append(out))
    def capture_evidence(module, arguments):
        evidence.append(arguments[0])
        geometry.append(arguments[1])
    model.evidence_extractor.register_forward_pre_hook(capture_evidence)
    for layer in model.layers:
        layer.register_forward_pre_hook(lambda m,args:local.append(args[0]))
    image,state = torch.randn(1,3,64,64),torch.randn(1,3,3,32)
    out=model.denoise(state,torch.ones(1),model.encode_image(image),collect_diagnostics=True)
    assert len(refined)==len(model.layers) and refined[0].shape==(1,3,4,32)
    assert model.pre_image_updates[0] is not model.pre_image_updates[1]
    for index,value in enumerate(refined):
        torch.testing.assert_close(evidence[index],value[:,:,:3])
        torch.testing.assert_close(local[index][:,-2],value[:,:,-1].reshape(3,32))
        wanted=model.codec.decode(model.latent_head(value[:,:,:3]))
        torch.testing.assert_close(geometry[index]['pred_boxes'],wanted['pred_boxes'])
    assert 'pre_image_update_rms' in out['latent_diagnostics'][0]
    out['pred_latents'].square().mean().backward()
    assert all(layer.ffn[0].weight.grad.abs().sum()>0 for layer in model.pre_image_updates)


def test_identity_hook_preserves_v17_and_v18_recurrence():
    from src.models.pipelines.pose_estimators.poseflow.latent_v12 import ddim_update
    base=v17_config().build(max_epochs=2).model.eval()
    model=_config().build(max_epochs=2).model.eval()
    missing,_=model.load_state_dict(base.state_dict(),strict=False)
    assert all(k.startswith('pre_image_updates.') for k in missing)
    image,state=torch.randn(1,3,64,64),torch.randn(1,3,3,32)
    memory=torch.randn(1,3,32)
    encoded=base.encode_image(image)
    saved=model.pre_image_updates
    model.pre_image_updates=nn.ModuleList([nn.Identity() for _ in model.layers])
    actual=model.denoise(state,torch.ones(1),encoded,hypothesis_memory=memory)
    wanted=base.denoise(state,torch.ones(1),encoded,hypothesis_memory=memory)
    torch.testing.assert_close(actual['pred_latents'],wanted['pred_latents'],atol=0,rtol=0)
    torch.testing.assert_close(actual['hypothesis_memory'],wanted['hypothesis_memory'],atol=0,rtol=0)
    assert not any(k.startswith('pre_image_updates') for k in base.state_dict())
    model.pre_image_updates=saved
    encoded=model.encode_image(image)
    current=state;bank=None;first=None
    for i in range(3):
        out=model.denoise(current,torch.tensor([1-i/3]),encoded,hypothesis_memory=bank)
        if first is None:first=out
        bank=out['hypothesis_memory'];current=ddim_update(current,out['pred_latents'],torch.tensor([1-i/3]),torch.tensor([1-(i+1)/3]))
    native=model.sample_keypoints(image,encoded_image=encoded,initial_latents=state,sampling_steps=3)
    torch.testing.assert_close(native['latent_state'],current)
    torch.testing.assert_close(native['hypothesis_memory'],bank)
    gradients=torch.autograd.grad(out['pred_latents'].square().mean(),[first['hypothesis_memory'],first['pred_latents']])
    assert all(g.abs().sum()>0 for g in gradients)
    permutation=torch.tensor([2,0,1])
    a=model.denoise(state,torch.ones(1),encoded,hypothesis_memory=memory)
    b=model.denoise(state[:,permutation],torch.ones(1),encoded,hypothesis_memory=memory[:,permutation])
    torch.testing.assert_close(b['pred_latents'],a['pred_latents'][:,permutation],atol=3e-5,rtol=3e-5)
    torch.testing.assert_close(b['hypothesis_memory'],a['hypothesis_memory'][:,permutation],atol=3e-5,rtol=3e-5)


def test_config_and_strict_restore(tmp_path):
    from pathlib import Path
    from hydra import compose,initialize_config_dir
    from omegaconf import OmegaConf
    from src.models.pipelines.pose_estimators.poseflow import PoseFlowPipelineConfig
    from scripts.evaluation.evaluate_poseflow import load_poseflow_model
    with initialize_config_dir(config_dir=str(Path('conf').resolve()),version_base=None):
        resolved=compose(config_name='benchmark_poseflow_v18')
    parsed=PoseFlowPipelineConfig.from_hydra_config(resolved.pipeline)
    assert isinstance(parsed,PoseFlowV18PipelineConfig)
    assert parsed.rollout_probability==.5 and parsed.rollout_call_counts==[2,3]
    assert resolved.training.max_epochs==80 and resolved.data.batch_size==4
    pipeline=_config().build(max_epochs=2).eval()
    assert pipeline._quick_architecture()['training_variant']=='V18-preimage-FFN'
    torch.save({'state_dict':pipeline.state_dict()},tmp_path/'model.ckpt')
    OmegaConf.save(OmegaConf.create({'pipeline':pipeline.v12_config.model_dump()|{
        '_target_':'src.models.pipelines.pose_estimators.poseflow.PoseFlowV18PipelineConfig'}}),tmp_path/'config.yaml')
    model,_=load_poseflow_model(tmp_path/'model.ckpt',tmp_path/'config.yaml');model.eval()
    image,state=torch.randn(1,3,64,64),torch.randn(1,3,3,32)
    for steps in [1,3]:
        expected=pipeline.model.sample_keypoints(image,initial_latents=state,sampling_steps=steps)
        actual=model.sample_keypoints(image,initial_latents=state,sampling_steps=steps)
        torch.testing.assert_close(actual['pred_latents'],expected['pred_latents'])
        torch.testing.assert_close(actual['hypothesis_memory'],expected['hypothesis_memory'])


def test_reset_and_empty_target_training():
    from tests.test_poseflow import _batch
    pipeline=_config(rollout_probability=1,rollout_call_counts=[3]).build(max_epochs=2)
    for empty in [False,True]:
        batch=_batch()|{'image':torch.randn(1,3,64,64)}
        batch['gt_labeled'][:,:,1]=False
        if empty:batch['person_mask']=torch.zeros(1,2,dtype=torch.bool)
        losses=pipeline._losses_for_batch(batch)
        assert all(torch.isfinite(value) for value in losses.values())
        if empty:assert losses['loss_latent']==losses['loss_bbox']==losses['loss_keypoints']==0
        pipeline.zero_grad(set_to_none=True)
        losses['total'].backward()
        assert all(layer.ffn[0].weight.grad.abs().sum()>0 for layer in pipeline.model.pre_image_updates)
    model=pipeline.model.eval()
    image,state=torch.randn(1,3,64,64),torch.randn(1,3,3,32)
    globals_seen=[]
    handle=model.pre_image_updates[0].register_forward_pre_hook(
        lambda m,args:globals_seen.append(args[0][:,:,-1].clone()))
    first=model.sample_keypoints(image,initial_latents=state,sampling_steps=3)
    handle.remove()
    assert len(globals_seen)==3
    for value in globals_seen:torch.testing.assert_close(value,model.global_token.expand(1,3,32))
    again=model.sample_keypoints(image,initial_latents=state,sampling_steps=3)
    torch.testing.assert_close(first['hypothesis_memory'],again['hypothesis_memory'])
    encoded=model.encode_image(image)
    direct=model.denoise(state,torch.ones(1),encoded)
    single=model.sample_keypoints(image,initial_latents=state,encoded_image=encoded,sampling_steps=1)
    torch.testing.assert_close(single['latent_state'],direct['pred_latents'])

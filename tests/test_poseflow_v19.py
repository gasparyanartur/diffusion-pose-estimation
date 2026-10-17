import pytest
import torch

from tests.test_poseflow_v17 import _config as previous_config
from src.models.pipelines.pose_estimators.poseflow.coordinate_v19 import CoordinateState
from src.models.pipelines.pose_estimators.poseflow.model_v19 import PoseFlowV19Model
from src.models.pipelines.pose_estimators.diffusionregpose.model import DiffusionRegPoseModel


def source_model():
    config = previous_config()
    names = ('num_queries','num_proposals','num_classes','num_feature_levels',
             'num_encoder_layers','num_decoder_layers','hidden_dim','feedforward_dim',
             'num_heads','num_body_points','num_box_decoder_layers','num_group','dn_number')
    return DiffusionRegPoseModel(backbone_pretrained=False, **{n:getattr(config,n) for n in names})


def test_coordinate_state_and_pretrained_zero_delta_identity():
    codec = CoordinateState(2)
    targets = {'boxes':torch.tensor([[[.2,.4,.3,.5]]]),
               'joints':torch.tensor([[[[0.,1.],[.7,.8]]]])}
    state = codec.encode(targets)
    assert state.shape == (1,1,8) and not list(codec.parameters())
    decoded = codec.decode(state)
    torch.testing.assert_close(decoded['pred_boxes'],targets['boxes'])
    torch.testing.assert_close(decoded['pred_keypoints'][...,:4].reshape(1,1,2,2),
                               targets['joints'].clamp(codec.epsilon,1-codec.epsilon))
    source = source_model()
    model = PoseFlowV19Model(source).eval()
    assert not any('global_token' in n or 'latent_head' in n for n in model.state_dict())
    for i,layer in enumerate(model.layers):
        torch.testing.assert_close(layer.block.cross_attn.value_proj.weight,
                                   source.transformer.decoder.layers[i].cross_attn.value_proj.weight)
        torch.testing.assert_close(layer.class_head.weight,source.class_embed[i].weight[1:2])
        for head in [layer.box_head,layer.joint_head]:
            torch.nn.init.zeros_(head.layers[-1].weight)
            torch.nn.init.zeros_(head.layers[-1].bias)
    image = torch.randn(1,3,64,64)
    state = torch.randn(1,3,8)
    references = []
    for layer in model.layers:
        layer.register_forward_pre_hook(lambda m,args:references.append(args[2].clone()))
    out = model.denoise(state,torch.ones(1),model.encode_image(image))
    for stage in out['aux_outputs']+[out]:
        torch.testing.assert_close(stage['pred_coordinates'],state,atol=0,rtol=0)
    for reference in references:
        torch.testing.assert_close(reference[:,:,0],state[:,:,:4].sigmoid())
    assert out['hypothesis_memory'].shape == (1,3,32)


def test_coordinate_recurrence_memory_and_gradients():
    from src.models.pipelines.pose_estimators.poseflow.latent_v12 import ddim_update
    source = source_model()
    for head in list(source.bbox_embed)+list(source.pose_embed):
        torch.nn.init.normal_(head.layers[-1].weight,std=.01)
    model = PoseFlowV19Model(source).eval()
    image,state = torch.randn(1,3,64,64),torch.randn(1,3,8)
    encoded = model.encode_image(image)
    local_shapes=[]
    for layer in model.layers:
        layer.block.self_attn.register_forward_pre_hook(lambda m,args:local_shapes.append(args[0].shape))
    first=model.denoise(state,torch.ones(1),encoded,collect_diagnostics=True)
    assert all(s==(4,3,32) for s in local_shapes)
    current=state;memory=None
    for j in range(3):
        out=model.denoise(current,torch.tensor([1-j/3]),encoded,hypothesis_memory=memory)
        if j==0:initial=out
        memory=out['hypothesis_memory']
        current=ddim_update(current,out['pred_coordinates'],torch.tensor([1-j/3]),torch.tensor([1-(j+1)/3]))
    gradients=torch.autograd.grad(out['pred_coordinates'].square().mean(),
        [initial['pred_coordinates'],initial['hypothesis_memory']],retain_graph=True)
    assert all(g.abs().sum()>0 for g in gradients)
    out['pred_coordinates'].square().mean().backward()
    assert all(l.block.cross_attn.value_proj.weight.grad.abs().sum()>0 for l in model.layers)
    actual=model.sample_keypoints(image,encoded_image=encoded,initial_latents=state,sampling_steps=3)
    torch.testing.assert_close(actual['coordinate_state'],current)
    torch.testing.assert_close(actual['hypothesis_memory'],memory)
    pair=torch.tensor([2,0,1]);bank=torch.randn(1,3,32)
    a=model.denoise(state,torch.ones(1),encoded,hypothesis_memory=bank)
    b=model.denoise(state[:,pair],torch.ones(1),encoded,hypothesis_memory=bank[:,pair])
    torch.testing.assert_close(a['pred_coordinates'][:,pair],b['pred_coordinates'],atol=3e-5,rtol=3e-5)
    torch.testing.assert_close(a['hypothesis_memory'][:,pair],b['hypothesis_memory'],atol=3e-5,rtol=3e-5)
    again=model.sample_keypoints(image,encoded_image=encoded,initial_latents=state,sampling_steps=3)
    torch.testing.assert_close(actual['hypothesis_memory'],again['hypothesis_memory'])


def configuration(**overrides):
    from src.models.pipelines.pose_estimators.poseflow.training_v19 import PoseFlowV19PipelineConfig
    values = previous_config().model_dump()
    selected = {name:value for name,value in values.items() if name in PoseFlowV19PipelineConfig.model_fields}
    return PoseFlowV19PipelineConfig(**(selected|overrides))


def test_rollout_objective_no_codec_loss_and_edge_targets(monkeypatch):
    from tests.test_poseflow import _batch
    from src.models.pipelines.pose_estimators.poseflow.latent_v12 import ddim_update
    pipeline=configuration(rollout_probability=1,rollout_call_counts=[3]).build(max_epochs=2)
    calls=[];losses=[];original=pipeline.model.denoise;original_loss=pipeline._denoising_losses
    def denoise(state,times,encoded,**kwargs):
        out=original(state,times,encoded,**kwargs)
        calls.append((state,times,kwargs.get('hypothesis_memory'),out))
        return out
    def loss(*args):
        out=original_loss(*args);losses.append(out);return out
    monkeypatch.setattr(pipeline.model,'denoise',denoise)
    monkeypatch.setattr(pipeline,'_denoising_losses',loss)
    batch=_batch()|{'image':torch.randn(1,3,64,64)}
    batch['gt_labeled'][:,:,1]=False
    values=pipeline._losses_for_batch(batch)
    assert len(calls)==3 and not any('latent' in k or 'reconstruction' in k for k in values)
    torch.testing.assert_close(values['total'],torch.stack([v['total'] for v in losses]).mean())
    for i in [1,2]:
        assert calls[i][2] is calls[i-1][3]['hypothesis_memory']
        torch.testing.assert_close(calls[i][0],ddim_update(calls[i-1][0],
            calls[i-1][3]['pred_coordinates'],calls[i-1][1],calls[i][1]))
    values['total'].backward()
    assert pipeline.model.hypothesis_slots.grad.abs().sum()>0
    empty=batch|{'person_mask':torch.zeros(batch['gt_labeled'].shape[:2],dtype=torch.bool)}
    values=pipeline._losses_for_batch(empty)
    assert values['loss_bbox']==values['loss_keypoints']==0 and torch.isfinite(values['total'])
    assert values['loss_ce']>0


def test_config_strict_restore_sampling_and_validation_branch(tmp_path):
    from pathlib import Path
    from hydra import compose,initialize_config_dir
    from omegaconf import OmegaConf
    from src.models.pipelines.pose_estimators.poseflow import PoseFlowPipelineConfig
    from scripts.evaluation.evaluate_poseflow import load_poseflow_model
    from tests.test_poseflow import _batch
    with initialize_config_dir(config_dir=str(Path('conf').resolve()),version_base=None):
        resolved=compose(config_name='benchmark_poseflow_v19')
    cfg=PoseFlowPipelineConfig.from_hydra_config(resolved.pipeline)
    assert cfg.rollout_probability==.5 and resolved.training.grad_clip==2
    assert 'reconstruction_weight_end' not in type(cfg).model_fields
    pipeline=configuration().build(max_epochs=2).eval()
    pipeline._losses_for_batch(_batch()|{'image':torch.randn(1,3,64,64)})
    assert pipeline._rollout_call_count==1
    torch.save({'state_dict':pipeline.state_dict()},tmp_path/'model.ckpt')
    OmegaConf.save(OmegaConf.create({'pipeline':pipeline.v12_config.model_dump()|{
        '_target_':'src.models.pipelines.pose_estimators.poseflow.PoseFlowV19PipelineConfig'}}),tmp_path/'config.yaml')
    model,_=load_poseflow_model(tmp_path/'model.ckpt',tmp_path/'config.yaml');model.eval()
    image,state=torch.randn(1,3,64,64),torch.randn(1,3,8)
    for steps in [1,3]:
        a=pipeline.model.sample_keypoints(image,initial_latents=state,sampling_steps=steps)
        b=model.sample_keypoints(image,initial_latents=state,sampling_steps=steps)
        torch.testing.assert_close(a['pred_coordinates'],b['pred_coordinates'])
        torch.testing.assert_close(a['hypothesis_memory'],b['hypothesis_memory'])
    a=model.sample_keypoints(image,initial_latents=state,sampling_steps=1)
    b=model.sample_keypoints(image,initial_latents=state,sampling_steps=1,persist_hypotheses=False)
    torch.testing.assert_close(a['pred_coordinates'],b['pred_coordinates'])


def test_padding_references_and_duplicate_assignment_masking():
    from src.models.pipelines.pose_estimators.poseflow.coordinate_v19 import matched_coordinate_loss
    from src.models.pipelines.pose_estimators.poseflow.latent_v12 import build_v12_targets
    from tests.test_poseflow import _batch
    model=PoseFlowV19Model(source_model()).eval()
    image=torch.randn(1,3,64,96);mask=torch.zeros(1,64,96,dtype=torch.bool);mask[:,:,64:]=True
    encoded=model.encode_image(image,mask)
    state=torch.randn(1,3,8);refs=[]
    model.layers[0].block.cross_attn.register_forward_pre_hook(lambda m,args:refs.append(args[1]))
    out=model.denoise(state,torch.ones(1),encoded)
    decoded=state.sigmoid();extents=model.joint_extents.weight[:2].sigmoid()[None,None]*decoded[:,:,None,2:4]
    unscaled=model._references(state,extents).flatten(1,2)
    ratios=torch.cat((encoded.valid_ratios,encoded.valid_ratios),-1)
    torch.testing.assert_close(refs[0],unscaled[:,:,None]*ratios[:,None])
    assert torch.isfinite(out['pred_coordinates']).all() and encoded.padding_mask.any()
    pipeline=configuration().build(max_epochs=2)
    batch=_batch()|{'person_mask':torch.tensor([[True,False]])}
    batch['gt_labeled'][:,:,1]=False
    targets=build_v12_targets(batch,groups=3);slot=targets['real'][0].nonzero().item()
    logits=torch.full((1,3,1),8.,requires_grad=True)
    boxes=targets['boxes'][:,slot:slot+1].expand(-1,3,-1)
    points=targets['joints'][:,slot:slot+1].expand(-1,3,-1,-1).clone()
    points[:,:,1]=.99
    prediction={'pred_logits':logits,'pred_boxes':boxes,
        'pred_keypoints':torch.cat((points.flatten(-2),torch.ones(1,3,2)),-1)}
    losses=matched_coordinate_loss(prediction,targets,pipeline.matcher,pipeline.v12_config.weights.to_dataclass())
    assert losses['positive_assignments']==3 and losses['loss_keypoints']==0
    expected=(.25*torch.nn.functional.softplus(-logits)*(1-logits.sigmoid())**2).mean()
    torch.testing.assert_close(losses['loss_ce'],expected)
    losses['total'].backward()
    assert (logits.grad<0).all()


def test_transfer_audit_distinguishes_checkpoint_from_fresh_parameters(tmp_path):
    from src.models.pipelines.pose_estimators.poseflow.initialization_v19 import checkpoint_transfer_coverage
    source=source_model();model=PoseFlowV19Model(source)
    checkpoint=tmp_path/'source.pt'
    torch.save({'model':source.state_dict()},checkpoint)
    cfg=configuration(pretrained_checkpoint=str(checkpoint))
    report=checkpoint_transfer_coverage(model,cfg)
    assert 0<report['verified_checkpoint_fraction']<1
    assert 'hypothesis_slots' in report['fresh_state_keys']
    assert not any(k.startswith('layers.') for k in report['fresh_state_keys'])
    with torch.no_grad():model.layers[0].box_head.layers[0].weight.add_(1)
    with pytest.raises(ValueError,match='transfer mismatch'):
        checkpoint_transfer_coverage(model,cfg)

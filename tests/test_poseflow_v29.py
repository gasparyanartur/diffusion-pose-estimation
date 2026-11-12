from copy import deepcopy
from pathlib import Path

import pytest
import torch


def constructed_fixture(people=2, proposals=5):
    points = torch.tensor([.2, .8]).repeat_interleave(28).reshape(1, 2, 14, 2)
    positive = torch.zeros(1, proposals, dtype=torch.bool)
    positive[:, :people] = True
    target = dict(unique_people=torch.tensor([people]), positive=positive,
        joints=torch.zeros(1, proposals, 14, 2), boxes=torch.zeros(1, proposals, 4),
        labeled=torch.zeros(1, proposals, 14, dtype=torch.bool),
        target_indices=torch.full((1, proposals), -1, dtype=torch.long))
    for index in range(people):
        target['joints'][0,index] = points[0,index%2]
        target['boxes'][0,index] = torch.tensor([.2 if index==0 else .8, .2 if index==0 else .8, .2, .2])
        target['labeled'][0,index] = True
        target['target_indices'][0,index] = index*2+1
    clean = dict(pred_latents=torch.arange(proposals*4).float().reshape(1,proposals,4).requires_grad_())
    return target, clean


class FixedMatcher:
    def __init__(self, source, target):
        self.source, self.target = source, target
        self.observed = None

    def __call__(self, output, targets):
        self.observed = targets
        return [(torch.tensor(self.source,dtype=torch.long),torch.tensor(self.target,dtype=torch.long))]


def test_rematched_latents_geometry_and_closest_background():
    from src.models.pipelines.pose_estimators.poseflow.latent_v29 import build_v29_ota_targets
    targets, clean = constructed_fixture()
    original = deepcopy(targets)
    recovered = dict(pred_latents=clean['pred_latents'].detach().flip(1).clone().requires_grad_())
    matcher = FixedMatcher([0,4],[1,0])
    assigned, latent, metadata = build_v29_ota_targets(clean,recovered,targets,matcher)
    assert matcher.observed[0]['labels'].numel()==2
    assert assigned['positive'].tolist()==[[True,False,False,False,True]]
    assert metadata['latent_indices'][0,0]==1 and metadata['latent_indices'][0,4]==0
    assert assigned['target_indices'][0,0]==3 and assigned['target_indices'][0,4]==1
    assert metadata['latent_indices'][0,1:4].tolist()==[3,2,2]
    assert not latent.requires_grad
    torch.testing.assert_close(latent[0,0],clean['pred_latents'][0,1],atol=0,rtol=0)
    torch.testing.assert_close(assigned['joints'][0,4],targets['joints'][0,0])
    for key in targets:
        torch.testing.assert_close(targets[key],original[key],atol=0,rtol=0)


@pytest.mark.parametrize('people,proposals',[(0,5),(2,2)])
def test_empty_recipient_and_full_positive_budget(people,proposals):
    from src.models.pipelines.pose_estimators.poseflow.latent_v29 import build_v29_ota_targets
    target,clean=constructed_fixture(people,proposals)
    matcher=FixedMatcher(list(range(people)),list(range(people)))
    recovered=dict(pred_latents=clean['pred_latents'].detach().clone())
    assigned,latent,metadata=build_v29_ota_targets(clean,recovered,target,matcher)
    assert assigned['positive'].sum()==people
    assert (metadata['latent_indices']>=0).all()
    torch.testing.assert_close(latent,clean['pred_latents'],atol=0,rtol=0)


def test_actual_ota_reassigns_swapped_poses_and_covers_unique_people():
    from src.models.pipelines.pose_estimators.poseflow.latent_v29 import build_v29_ota_targets
    from src.models.pipelines.pose_estimators.diffusionregpose.training import DiffusionRegPoseOTAMatcher
    target,clean=constructed_fixture()
    order=torch.tensor([1,0,1,0,0]);joints=target['joints'][:,order]
    output=dict(pred_latents=clean['pred_latents'].detach().clone(),
        pred_boxes=target['boxes'][:,order],pred_logits=torch.tensor([[[-20.,4.]]*5]),
        pred_keypoints=torch.cat((joints.flatten(-2),torch.ones(1,5,14)),-1))
    assigned,latent,metadata=build_v29_ota_targets(clean,output,target,DiffusionRegPoseOTAMatcher())
    assert assigned['positive'].all()
    assert metadata['matched_person'].tolist()==[order.tolist()]
    assert assigned['target_indices'][0].unique().tolist()==[1,3]
    # Query permutation leaves the detached assignment/target values equivariant.
    permutation=torch.tensor([4,2,0,3,1])
    changed={key:value[:,permutation] for key,value in output.items()}
    new_tasks,new_latents,_=build_v29_ota_targets(clean,changed,target,DiffusionRegPoseOTAMatcher())
    torch.testing.assert_close(new_latents,latent[:,permutation],atol=0,rtol=0)
    torch.testing.assert_close(new_tasks['joints'],assigned['joints'][:,permutation])


def test_missing_negative_bank_rejected_and_conflicts_repaired():
    from src.models.pipelines.pose_estimators.poseflow.latent_v29 import build_v29_ota_targets
    from src.models.pipelines.pose_estimators.diffusionregpose.training import DiffusionRegPoseOTAMatcher
    targets,clean=constructed_fixture(2,2)
    with pytest.raises(ValueError,match='negative'):
        build_v29_ota_targets(clean,dict(pred_latents=clean['pred_latents']),targets,FixedMatcher([0],[0]))
    matcher=DiffusionRegPoseOTAMatcher()
    sources,people=matcher._dynamic_k_assignment(torch.tensor([[0.,0.],[1.,2.],[3.,4.]]),torch.ones(3,2))
    assert sources.unique().numel()==len(sources) and people.unique().numel()==2


def test_pipeline_loss_gradient_restore_and_rng(tmp_path):
    from src.models.pipelines.pose_estimators.poseflow.training_v29 import PoseFlowV29PipelineConfig,PoseFlowV29Pipeline
    from src.models.pipelines.pose_estimators.poseflow.training_v28 import PoseFlowV28PipelineConfig
    from scripts.evaluation.evaluate_poseflow import load_poseflow_model
    from omegaconf import OmegaConf
    arguments=dict(backbone_pretrained=False,pretrained_checkpoint=None,hidden_dim=32,
        num_heads=4,feedforward_dim=64,num_proposals=5,num_queries=5,num_group=5,quick_eval_enabled=False)
    config=PoseFlowV29PipelineConfig(**arguments)
    torch.manual_seed(42);pipeline=config.build(max_epochs=2)
    torch.manual_seed(42);reference=PoseFlowV28PipelineConfig(**arguments).build(max_epochs=2)
    assert isinstance(pipeline,PoseFlowV29Pipeline)
    for key,value in reference.model.state_dict().items():
        torch.testing.assert_close(value,pipeline.model.state_dict()[key],atol=0,rtol=0)
    batch=dict(image=torch.randn(2,3,64,96),image_id=torch.arange(2),img_hw=torch.tensor([[64.,96.]]*2),
        gt_keypoints=torch.zeros(2,1,14,2),gt_labeled=torch.ones(2,1,14,dtype=torch.bool),
        person_mask=torch.ones(2,1,dtype=torch.bool),gt_boxes_canon=torch.tensor([[[.5,.5,.3,.5]]]*2))
    losses=pipeline._losses_for_batch(batch)
    assert not losses['denoised/total'].requires_grad
    torch.testing.assert_close(losses['total'],.5*losses['clean/total']+losses['loss_latent'],atol=0,rtol=0)
    losses['clean/total'].backward(retain_graph=True)
    assert all(p.grad is None for p in pipeline.model.backbone.parameters())
    assert pipeline.model.level_embed.grad is None
    pipeline.zero_grad(set_to_none=True);losses['loss_latent'].backward()
    assert pipeline.model.input_mlp[0].weight.grad is None
    assert pipeline.model.joint_head.layers[-1].weight.grad is None
    assert pipeline.model.level_embed.grad.abs().sum()>0
    assert pipeline.model.denoiser_projection.weight.grad.abs().sum()>0
    restored=config.build(max_epochs=2);restored.load_state_dict(deepcopy(pipeline.state_dict()),strict=True)
    for left,right in zip(pipeline._draw_training_inputs(batch)[1:],restored._draw_training_inputs(batch)[1:]):
        torch.testing.assert_close(left,right,atol=0,rtol=0)
    torch.save(dict(state_dict=pipeline.state_dict()),tmp_path/'model.ckpt')
    saved=config.model_dump()|{'_target_':'src.models.pipelines.pose_estimators.poseflow.PoseFlowV29PipelineConfig','pretrained_checkpoint':'/missing/image.pth'}
    OmegaConf.save(OmegaConf.create(dict(pipeline=saved)),tmp_path/'config.yaml')
    model,_=load_poseflow_model(tmp_path/'model.ckpt',tmp_path/'config.yaml')
    model.eval();reference.model.load_state_dict(model.state_dict(),strict=True);reference.model.eval()
    for calls in (1,4):
        model.flow_sampling_steps=reference.model.flow_sampling_steps=calls
        with torch.no_grad():
            actual=model.sample_keypoints(batch['image'],generator=torch.Generator().manual_seed(8))
            expected=reference.model.sample_keypoints(batch['image'],generator=torch.Generator().manual_seed(8))
        torch.testing.assert_close(actual['pred_latents'],expected['pred_latents'],atol=0,rtol=0)

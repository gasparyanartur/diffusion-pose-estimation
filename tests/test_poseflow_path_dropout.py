import pytest
import torch

from tests.test_poseflow import _batch
from tests.test_poseflow_v15_rollout import _config as rollout_config
from src.models.pipelines.pose_estimators.poseflow.path_dropout import JointMessageDropout
from src.models.pipelines.pose_estimators.poseflow.training_path_dropout import PoseFlowPathDropoutPipelineConfig


def _config(**overrides):
    return PoseFlowPathDropoutPipelineConfig(**(rollout_config().model_dump() | overrides))


def test_mask_boundaries_identity_scaling_reproducibility_and_joint_granularity():
    value = torch.randn(2, 3, 16, 32, requires_grad=True)
    dropout = JointMessageDropout(.25, slice(1, -1)).train()
    torch.manual_seed(7)
    changed = dropout(value)
    mask = dropout.last_drop_mask.clone()
    torch.manual_seed(7)
    assert torch.equal(dropout(value), changed)
    assert torch.equal(dropout.last_drop_mask, mask)
    torch.testing.assert_close(changed[:, :, 0], value[:, :, 0])
    torch.testing.assert_close(changed[:, :, -1], value[:, :, -1])
    torch.testing.assert_close(changed[:, :, 1:-1], value[:, :, 1:-1]*(~mask)[..., None]/.75)
    dropout.probability = 0
    assert dropout(value) is value
    dropout.probability = 1
    changed = dropout(value)
    assert changed[:, :, 1:-1].count_nonzero() == 0
    torch.testing.assert_close(changed[:, :, -1], value[:, :, -1])
    changed.sum().backward()
    assert value.grad[:, :, 1:-1].count_nonzero() == 0
    assert value.grad[:, :, -1].count_nonzero() > 0
    dropout.eval()
    assert dropout(value) is value


def test_independent_mask_frequencies():
    messages = torch.ones(128, 50, 15, 4)
    image = JointMessageDropout(.25, slice(1, None)).train()
    attention = JointMessageDropout(.25, slice(1, None)).train()
    image(messages); attention(messages)
    first, second = image.last_drop_mask, attention.last_drop_mask
    assert abs(float(first.float().mean())-.25) < .01
    assert abs(float(second.float().mean())-.25) < .01
    assert abs(float((first & second).float().mean())-.0625) < .005


def test_eval_and_zero_probability_training_match_original_without_parameter_changes():
    baseline = rollout_config().build(max_epochs=2)
    changed = _config(joint_image_path_dropout_probability=0,
        joint_self_attention_path_dropout_probability=0).build(max_epochs=2)
    changed.load_state_dict(baseline.state_dict(), strict=True)
    assert sum(p.numel() for p in changed.parameters()) == sum(p.numel() for p in baseline.parameters())
    batch = _batch() | {'image': torch.randn(1, 3, 64, 64)}
    for training in (True, False):
        baseline.train(training);changed.train(training)
        torch.manual_seed(9); expected = baseline._losses_for_batch(batch)
        torch.manual_seed(9); actual = changed._losses_for_batch(batch)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    changed.v12_config.joint_image_path_dropout_probability = .25
    changed.model.configure_path_dropout(.25, .25)
    changed.eval();baseline.eval()
    image,state=torch.randn(1,3,64,64),torch.randn(1,3,3,32)
    torch.testing.assert_close(changed.model.sample_keypoints(image,initial_latents=state)['pred_latents'],
        baseline.model.sample_keypoints(image,initial_latents=state)['pred_latents'],rtol=0,atol=0)


def test_forced_both_off_preserves_global_aggregation_and_hypothesis_gradients():
    model = _config(joint_image_path_dropout_probability=1,
        joint_self_attention_path_dropout_probability=1).build(max_epochs=2).model.train()
    output=model(torch.randn(1,3,64,64),torch.randn(1,3,3,32),torch.ones(1),collect_diagnostics=True)
    for row in output['latent_diagnostics']:
        assert row['joint_image_drop_fraction'] == row['joint_self_drop_fraction'] == row['joint_both_drop_fraction'] == 1
    gradient, = torch.autograd.grad(output['pred_keypoints'][...,:4].square().mean(),
        model.layers[-1].state_update.attention.in_proj_weight,retain_graph=True)
    assert gradient.abs().sum() > 0
    output['pred_latents'].square().mean().backward()
    assert model.hypothesis_update.retention_logit.grad.abs() > 0
    assert model.global_token.grad.abs().sum() > 0
    assert all(layer.aggregate.attention.in_proj_weight.grad.abs().sum() > 0 for layer in model.layers)


def test_masks_change_across_fixed_attention_seed_batches_and_generator_is_local():
    pipeline = _config(rollout_probability=0).build(max_epochs=2).train()
    pipeline.deterministic_seed = 42
    masks = []
    handle=pipeline.model.evidence_extractor.image_message_transform.register_forward_hook(
        lambda module,args,output:masks.append(module.last_drop_mask.clone()))
    batch=_batch() | {'image':torch.randn(1,3,64,64)}
    torch.manual_seed(19)
    pipeline._losses_for_batch(batch)
    pipeline._losses_for_batch(batch)
    assert not torch.equal(masks[0],masks[len(pipeline.model.layers)])
    first=masks[0].clone()
    masks.clear()
    torch.manual_seed(19)
    pipeline._losses_for_batch(batch)
    assert torch.equal(first,masks[0])
    assert pipeline.model.evidence_extractor.image_message_transform.generator is None
    handle.remove()
    pipeline.model.eval().configure_path_dropout(.25,.25)
    assert not pipeline.model.evidence_extractor.image_message_transform.training


def test_message_only_mask_preserves_residual_norm_and_ffn():
    from src.models.pipelines.pose_estimators.poseflow.model_v14 import V14AttentionUpdate
    block=V14AttentionUpdate(32,4,64,0).train()
    block.message_transform=JointMessageDropout(1,slice(1,-1))
    queries=torch.randn(2,16,32)
    result,_=block(queries,queries)
    normalized=block.attention_norm(queries[:,1:-1])
    expected=block.ffn_norm(normalized+block.ffn(normalized))
    torch.testing.assert_close(result[:,1:-1],expected)


def test_rollout_memory_gradients_empty_targets_and_checkpoint_config(tmp_path,monkeypatch):
    import tests.test_poseflow_v15_rollout as baseline
    monkeypatch.setattr(baseline,'_config',_config)
    baseline.test_rollout_recurrence_gradients_and_equal_losses(monkeypatch)
    for empty in (False,True):
        pipeline=_config(rollout_probability=1,rollout_call_counts=[3]).build(max_epochs=2)
        batch=_batch() | {'image':torch.randn(1,3,64,64)}
        if empty:batch['person_mask']=torch.zeros(1,2,dtype=torch.bool)
        losses=pipeline._losses_for_batch(batch)
        assert all(torch.isfinite(value) for value in losses.values())
        losses['total'].backward()
        if empty:assert losses['loss_latent'] == losses['loss_bbox'] == losses['loss_keypoints'] == 0
    from pathlib import Path
    from hydra import compose,initialize_config_dir
    from omegaconf import OmegaConf
    from scripts.evaluation.evaluate_poseflow import load_poseflow_model
    from src.models.pipelines.pose_estimators.poseflow import PoseFlowPipelineConfig
    with initialize_config_dir(config_dir=str(Path('conf').resolve()),version_base=None):
        config=compose(config_name='benchmark_poseflow_v15_rollout_path_dropout')
    parsed=PoseFlowPipelineConfig.from_hydra_config(config.pipeline)
    assert isinstance(parsed,PoseFlowPathDropoutPipelineConfig)
    assert parsed.joint_image_path_dropout_probability == parsed.joint_self_attention_path_dropout_probability == .25
    assert parsed.rollout_probability == .5 and parsed.rollout_call_counts == [2,3]
    pipeline=_config(joint_image_path_dropout_probability=.4).build(max_epochs=2).eval()
    torch.save({'state_dict':pipeline.state_dict()},tmp_path/'model.ckpt')
    OmegaConf.save(OmegaConf.create({'pipeline':{'_target_':'src.models.pipelines.pose_estimators.poseflow.PoseFlowPathDropoutPipelineConfig',**pipeline.v12_config.model_dump(mode='json')}}),tmp_path/'config.yaml')
    model,_=load_poseflow_model(tmp_path/'model.ckpt',tmp_path/'config.yaml');model.eval()
    assert model.evidence_extractor.image_message_transform.probability == .4
    image,state=torch.randn(1,3,64,64),torch.randn(1,3,3,32)
    torch.testing.assert_close(model.sample_keypoints(image,initial_latents=state)['pred_latents'],pipeline.model.sample_keypoints(image,initial_latents=state)['pred_latents'])


@pytest.mark.parametrize('probability',[-.1,1.1,float('nan'),float('inf')])
def test_invalid_probability(probability):
    with pytest.raises(ValueError):_config(joint_image_path_dropout_probability=probability)

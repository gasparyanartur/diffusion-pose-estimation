import pytest
import torch

from tests.test_poseflow import _batch
from tests.test_poseflow_v15 import _config as baseline_config
from src.models.pipelines.pose_estimators.poseflow.training_v15 import PoseFlowV15Pipeline
from src.models.pipelines.pose_estimators.poseflow.training_v15_rollout import (
    PoseFlowV15RolloutPipelineConfig, sample_rollout_schedule,
)
from src.models.pipelines.pose_estimators.poseflow.latent_v12 import ddim_update


def _config(**overrides):
    return PoseFlowV15RolloutPipelineConfig(**(baseline_config().model_dump() | overrides))


def test_schedule_bounds_independence_and_reproducibility():
    arguments = dict(batch_size=8, calls=3, timesteps=1000, device=torch.device('cpu'), dtype=torch.float32)
    first = sample_rollout_schedule(**arguments, generator=torch.Generator().manual_seed(42))
    second = sample_rollout_schedule(**arguments, generator=torch.Generator().manual_seed(42))
    torch.testing.assert_close(first, second)
    assert first.shape == (8,4) and (first[:,0] == 1).all() and (first[:,-1] == 0).all()
    assert (first[:,:-1] > first[:,1:]).all()
    assert torch.unique(first[:,1:-1],dim=0).shape[0] > 1
    with pytest.raises(ValueError):
        sample_rollout_schedule(**(arguments | {'timesteps':3}))


def test_single_branch_and_validation_parity():
    pipeline = _config(rollout_probability=0).build(max_epochs=2)
    batch = _batch() | {'image':torch.randn(1,3,64,64)}
    torch.manual_seed(9)
    actual = pipeline._losses_for_batch(batch)
    torch.manual_seed(9)
    expected = PoseFlowV15Pipeline._losses_for_batch(pipeline,batch)
    for key in expected:
        torch.testing.assert_close(actual[key],expected[key])
    pipeline.v12_config.rollout_probability = 1
    pipeline.eval()
    torch.manual_seed(9)
    actual = pipeline._losses_for_batch(batch)
    torch.manual_seed(9)
    expected = PoseFlowV15Pipeline._losses_for_batch(pipeline,batch)
    for key in expected:
        torch.testing.assert_close(actual[key],expected[key])


def test_rollout_recurrence_gradients_and_equal_losses(monkeypatch):
    pipeline = _config(rollout_probability=1,rollout_call_counts=[3]).build(max_epochs=2)
    batch = _batch() | {'image':torch.randn(1,3,64,64)}
    model = pipeline.model
    calls, layer_losses, encodings, initial_globals = [], [], [], []
    original_denoise, original_losses = model.denoise, pipeline._denoising_losses
    model.encoder.register_forward_hook(lambda *args:encodings.append(1))
    model.layers[0].state_update.register_forward_pre_hook(lambda module,args:initial_globals.append(args[0][:,3::4].detach().clone()))
    def denoise(state,times,image,**kwargs):
        out = original_denoise(state,times,image,**kwargs)
        calls.append((state,times,kwargs.get('hypothesis_memory'),out))
        return out
    def loss(*args):
        values=original_losses(*args);layer_losses.append(values);return values
    monkeypatch.setattr(model,'denoise',denoise)
    monkeypatch.setattr(pipeline,'_denoising_losses',loss)
    values=pipeline._losses_for_batch(batch)
    assert len(calls)==3 and len(encodings)==1 and calls[0][2] is None
    assert (calls[0][1]==1).all()
    for index in [1,2]:
        assert calls[index][2] is calls[index-1][3]['hypothesis_memory']
        expected=ddim_update(calls[index-1][0],calls[index-1][3]['pred_latents'],calls[index-1][1],calls[index][1])
        torch.testing.assert_close(calls[index][0],expected)
        assert calls[index][0].grad_fn is not None and calls[index][2].grad_fn is not None
    assert all(torch.equal(g,initial_globals[0]) for g in initial_globals)
    expected=torch.stack([v['loss_latent_mean']+v['loss_prediction'] for v in layer_losses]).mean()+values['loss_reconstruction_weighted']
    torch.testing.assert_close(values['total'],expected)
    gradients=torch.autograd.grad(layer_losses[-1]['loss_prediction'],
        [calls[0][3]['pred_latents'],calls[0][3]['hypothesis_memory']],retain_graph=True)
    assert all(g.abs().sum()>0 for g in gradients)
    values['total'].backward()
    assert model.hypothesis_update.retention_logit.grad.abs()>0
    assert model.codec.box_encoder.layers[0].weight.grad.abs().sum()>0
    assert 'rollout/call2/incoming_memory_rms' in pipeline._v12_diagnostics


@pytest.mark.parametrize('overrides',[{'rollout_call_counts':[]},{'rollout_call_counts':[1]},{'rollout_call_counts':[2,2]},{'rollout_call_counts':[3],'timesteps':3}])
def test_invalid_configuration(overrides):
    with pytest.raises(ValueError):
        _config(**overrides)


def test_config_and_strict_model_restore(tmp_path):
    from pathlib import Path
    from hydra import compose,initialize_config_dir
    from omegaconf import OmegaConf
    from scripts.evaluation.evaluate_poseflow import load_poseflow_model
    from src.models.pipelines.pose_estimators.poseflow import PoseFlowPipelineConfig
    with initialize_config_dir(config_dir=str(Path('conf').resolve()),version_base=None):
        config=compose(config_name='benchmark_poseflow_v15_rollout')
    parsed=PoseFlowPipelineConfig.from_hydra_config(config.pipeline)
    assert isinstance(parsed,PoseFlowV15RolloutPipelineConfig)
    assert parsed.rollout_probability == .5 and parsed.rollout_call_counts == [2,3]
    assert config.out_dir == 'results/poseflow_v15_rollout_training'
    assert config.training.resume is None and config.training.max_epochs == 80
    assert config.data.batch_size == 4 and config.training.accumulate_grad_batches == 2
    assert parsed.lr_cycle_restart_epoch == 40 and parsed.ota_candidate_topk == 5
    pipeline=_config().build(max_epochs=2).eval()
    baseline=baseline_config().build(max_epochs=2).eval()
    assert sum(p.numel() for p in pipeline.parameters()) == sum(p.numel() for p in baseline.parameters())
    baseline.model.load_state_dict(pipeline.model.state_dict(),strict=True)
    torch.save({'state_dict':pipeline.state_dict()},tmp_path/'model.ckpt')
    OmegaConf.save(OmegaConf.create({'pipeline':{'_target_':'src.models.pipelines.pose_estimators.poseflow.PoseFlowV15RolloutPipelineConfig',**pipeline.v12_config.model_dump(mode='json')}}),tmp_path/'config.yaml')
    restored,config=load_poseflow_model(tmp_path/'model.ckpt',tmp_path/'config.yaml')
    restored.eval()
    image,initial=torch.randn(1,3,64,64),torch.randn(1,3,3,32)
    torch.testing.assert_close(restored.sample_keypoints(image,initial_latents=initial)['pred_latents'],baseline.model.sample_keypoints(image,initial_latents=initial)['pred_latents'])


@pytest.mark.parametrize('empty',[False,True])
def test_empty_and_invisible_rollout_targets(empty):
    pipeline=_config(rollout_probability=1,rollout_call_counts=[2]).build(max_epochs=2)
    batch=_batch() | {'image':torch.randn(1,3,64,64)}
    batch['gt_labeled'][...,1]=False
    if empty:
        batch['person_mask']=torch.zeros(1,2,dtype=torch.bool)
    losses=pipeline._losses_for_batch(batch)
    assert all(torch.isfinite(v) for v in losses.values())
    if empty:
        assert losses['loss_latent_mean']==losses['loss_bbox']==losses['loss_keypoints']==0
    losses['total'].backward()
    assert torch.isfinite(pipeline.model.hypothesis_update.retention_logit.grad)


def test_rollout_does_not_leak_memory_between_batches(monkeypatch):
    pipeline=_config(rollout_probability=1,rollout_call_counts=[2]).build(max_epochs=2)
    batch=_batch() | {'image':torch.randn(1,3,64,64)}
    memories=[];original=pipeline.model.denoise
    def capture(*args,**kwargs):
        memories.append(kwargs.get('hypothesis_memory'))
        return original(*args,**kwargs)
    monkeypatch.setattr(pipeline.model,'denoise',capture)
    pipeline._losses_for_batch(batch)
    pipeline._losses_for_batch(batch)
    assert memories[0] is None and memories[2] is None
    assert memories[1] is not memories[3]


def test_mixture_branch_and_call_distribution(monkeypatch):
    pipeline=_config().build(max_epochs=2)
    calls=[]
    monkeypatch.setattr(PoseFlowV15Pipeline,'_losses_for_batch',lambda *args:{'total':torch.tensor(0.)})
    def record(*args):
        calls.append(args[-1].shape[1]-1)
        return {'total':torch.tensor(0.)},{}
    monkeypatch.setattr(pipeline,'_rollout_losses',record)
    batch=_batch() | {'image':torch.randn(1,3,64,64)}
    torch.manual_seed(42)
    for _ in range(400):
        pipeline._losses_for_batch(batch)
    assert 160 < len(calls) < 240
    assert 60 < calls.count(2) < 140 and 60 < calls.count(3) < 140

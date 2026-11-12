from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from tests.test_poseflow_v35 import batch, pipeline


def test_early_ramp_is_exact_reflection_with_equal_mean():
    from src.models.pipelines.pose_estimators.poseflow.training_v35 import (
        early_linear_spatial_multipliers, late_linear_spatial_multipliers,
    )
    times=torch.arange(1000)
    early=early_linear_spatial_multipliers(times,cutoff=500,num_timesteps=1000)
    late=late_linear_spatial_multipliers(times,cutoff=500,num_timesteps=1000)
    torch.testing.assert_close(early,late.flip(0),atol=0,rtol=0)
    torch.testing.assert_close(early.mean(),late.mean(),atol=1e-7,rtol=0)
    torch.testing.assert_close(early[[0,499,500,749,999]],torch.tensor([0.,0.,.002,.5,1.]),atol=1e-7,rtol=0)
    torch.testing.assert_close(early.mean(),torch.tensor(.2505),atol=1e-7,rtol=0)


def test_actual_early_pipeline_is_per_image_and_strictly_restores():
    module=pipeline(True)
    module.v26_config.spatial_cost_schedule='early_linear'
    data=batch();targets,times,noise=module._draw_training_inputs(data)
    times=torch.tensor([999,0]);module._draw_training_inputs=lambda *args,**kwargs:(targets,times,noise)
    seen=[];original=module.matcher.forward
    def capture(output,targets,*,spatial_multipliers=None):
        seen.append(spatial_multipliers)
        return original(output,targets,spatial_multipliers=spatial_multipliers)
    module.matcher.forward=capture
    losses=module._losses_for_batch(data)
    assert len(seen)==1 and losses['denoised/total'].requires_grad
    torch.testing.assert_close(seen[0],torch.tensor([1.,0.]),atol=0,rtol=0)
    torch.testing.assert_close(losses['total'],.5*losses['clean/total']+.5*losses['denoised/total']+10*losses['loss_latent'])
    losses['total'].backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in module.parameters())
    restored=module.v26_config.build(max_epochs=2);restored.load_state_dict(module.state_dict(),strict=True)
    for training in (False,True):
        expected=type(restored)._draw_training_inputs(module,data,training=training,batch_idx=9)
        actual=restored._draw_training_inputs(data,training=training,batch_idx=9)
        for left,right in zip(expected,actual):
            if isinstance(left,dict):
                for key in left:torch.testing.assert_close(left[key],right[key],atol=0,rtol=0)
            else:torch.testing.assert_close(left,right,atol=0,rtol=0)


def test_early_config_changes_only_direction_and_namespace():
    with initialize_config_dir(config_dir=str(Path('conf').resolve()),version_base=None):
        late=OmegaConf.to_container(compose(config_name='benchmark_poseflow_v35_spatial_timed_cost'),resolve=True)
        early=OmegaConf.to_container(compose(config_name='benchmark_poseflow_v35_spatial_early_cost'),resolve=True)
    late['pipeline']['spatial_cost_schedule']='early_linear'
    late.update(run_name='poseflow_v35_spatial_early_cost',out_dir='results/poseflow_v35_spatial_early_cost')
    assert late==early

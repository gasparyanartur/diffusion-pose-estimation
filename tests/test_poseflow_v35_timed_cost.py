from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from tests.test_poseflow_v35 import batch, pipeline


def test_late_linear_schedule_endpoints_monotonicity_and_validation():
    from src.models.pipelines.pose_estimators.poseflow.training_v35 import late_linear_spatial_multipliers
    times = torch.tensor([0, 250, 499, 500, 999])
    actual = late_linear_spatial_multipliers(times, cutoff=500, num_timesteps=1000)
    torch.testing.assert_close(actual, torch.tensor([1., .5, .002, 0., 0.]), atol=1e-7, rtol=0)
    assert (actual[:-1] >= actual[1:]).all()
    for times, cutoff in [(torch.tensor([-1]), 500), (torch.tensor([1000]), 500), (torch.tensor([0]), 0)]:
        with pytest.raises(ValueError):
            late_linear_spatial_multipliers(times, cutoff=cutoff, num_timesteps=1000)


def test_mixed_image_matching_matches_zero_and_full_cost_and_empty_targets():
    from src.models.pipelines.pose_estimators.poseflow.latent_v35 import BoxOTAMatcher, grid_anchors
    anchors = grid_anchors(2, 2)
    output = dict(pred_boxes=torch.tensor([[[.75, .75, .2, .2]]*4]*3), pred_logits=torch.zeros(3, 4, 2))
    targets = [dict(labels=torch.ones(1, dtype=torch.long), boxes=output['pred_boxes'][0, :1]) for _ in range(2)]
    targets.append(dict(labels=torch.empty(0, dtype=torch.long), boxes=torch.empty(0, 4)))
    matcher = BoxOTAMatcher(anchors=anchors, spatial_cost_weight=1., candidate_topk=1)
    actual = matcher(output, targets, spatial_multipliers=torch.tensor([0., 1., .5]))
    for i, weight in enumerate([0., 1.]):
        reference = BoxOTAMatcher(anchors=anchors, spatial_cost_weight=weight, candidate_topk=1)
        expected = reference({k:v[i:i+1] for k,v in output.items()}, targets[i:i+1])[0]
        for a,b in zip(actual[i], expected):
            torch.testing.assert_close(a,b,atol=0,rtol=0)
    assert actual[2][0].numel() == 0
    assert float(matcher.diagnostics['spatial_weight_mean']) == .5
    raw = torch.cdist(anchors, targets[0]['boxes'][:, :2]).square().mean()
    torch.testing.assert_close(matcher.diagnostics['spatial_cost_weighted'], raw/2)
    legacy = matcher(output, targets)
    ones = matcher(output, targets, spatial_multipliers=torch.ones(3))
    for left,right in zip(legacy,ones):
        for a,b in zip(left,right):torch.testing.assert_close(a,b,atol=0,rtol=0)
    for invalid in (torch.ones(2), torch.tensor([0.,1.,float('nan')]), torch.tensor([0.,1.,2.])):
        with pytest.raises(ValueError):matcher(output, targets, spatial_multipliers=invalid)


def test_actual_pipeline_passes_each_timestep_and_restores_rng():
    module = pipeline(True)
    module.v26_config.spatial_cost_schedule = 'late_linear'
    module.v26_config.spatial_cost_ramp_cutoff = 500
    data = batch()
    targets,times,noise = module._draw_training_inputs(data)
    times = torch.tensor([999, 0])
    module._draw_training_inputs = lambda *args, **kwargs: (targets,times,noise)
    observed=[]
    original = module.matcher.forward
    def capture(output, targets, *, spatial_multipliers=None):
        observed.append(spatial_multipliers)
        return original(output,targets,spatial_multipliers=spatial_multipliers)
    module.matcher.forward = capture
    losses = module._losses_for_batch(data)
    torch.testing.assert_close(observed[0],torch.tensor([0.,1.]),atol=0,rtol=0)
    assert len(observed) == 1 and losses['denoised/total'].requires_grad
    torch.testing.assert_close(losses['total'],.5*losses['clean/total']+.5*losses['denoised/total']+10*losses['loss_latent'])
    losses['total'].backward()
    for component in (module.model.clean_layers,module.model.denoiser_layers,module.model.box_head,module.model.backbone):
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in component.parameters())
    restored=module.v26_config.build(max_epochs=2)
    restored.load_state_dict(module.state_dict(),strict=True)
    for training in (False, True):
        reference=type(restored)._draw_training_inputs(module,data,training=training,batch_idx=4)
        actual=restored._draw_training_inputs(data,training=training,batch_idx=4)
        for left,right in zip(reference,actual):
            if isinstance(left,dict):
                for key in left:torch.testing.assert_close(left[key],right[key],atol=0,rtol=0)
            else:torch.testing.assert_close(left,right,atol=0,rtol=0)


def test_variant_configuration_is_only_schedule_and_namespace_change():
    with initialize_config_dir(config_dir=str(Path('conf').resolve()),version_base=None):
        baseline=OmegaConf.to_container(compose(config_name='benchmark_poseflow_v35_spatial'),resolve=True)
        actual=OmegaConf.to_container(compose(config_name='benchmark_poseflow_v35_spatial_timed_cost'),resolve=True)
    baseline['pipeline']['spatial_cost_schedule']='late_linear'
    baseline.update(run_name='poseflow_v35_spatial_timed_cost',out_dir='results/poseflow_v35_spatial_timed_cost')
    assert actual == baseline

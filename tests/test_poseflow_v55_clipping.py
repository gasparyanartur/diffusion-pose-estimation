from copy import deepcopy
import math

import pytest
import torch

from src.models.pipelines.pose_estimators.poseflow.balance_v51 import clip_families, gradient_families
from tests.test_poseflow_v55_training import pipeline


def test_shared_context_scaling_cannot_change_private_geometry_clipping():
    module = pipeline()
    parameters = dict(module.model.named_parameters())
    context = parameters['proposal_context.attention.out_proj.weight']
    joint = module.model.pose_decoder.pose_heads[0].layers[-1].weight
    observed = []
    for scale in (1., 10.):
        module.zero_grad(set_to_none=True)
        context.grad = torch.full_like(context, scale)
        joint.grad = torch.ones_like(joint)
        report = module.configure_gradient_clipping(None, .1, 'norm')
        observed.append(joint.grad.clone())
        norm = sum(float(parameter.grad.double().square().sum()) for parameter in parameters.values()
                   if parameter.grad is not None)**.5
        assert norm <= .1
        assert report['shared_context/postclip'] <= .1/math.sqrt(6)+1e-8
        assert 'geometry/clip_factor' not in report
    torch.testing.assert_close(*observed, atol=0, rtol=0)


def test_explicit_budgets_ownership_validation_and_legacy_parity():
    from src.models.pipelines.pose_estimators.poseflow.balance_v51 import clip_families_with_budgets
    parameters = [torch.nn.Parameter(torch.ones(2)) for _ in range(3)]
    for index, parameter in enumerate(parameters):
        parameter.grad = torch.full_like(parameter, 100.**index)
    copies = [torch.nn.Parameter(parameter.detach().clone()) for parameter in parameters]
    for original, copied in zip(parameters, copies):
        copied.grad = original.grad.clone()
    families = {str(index): [parameter] for index, parameter in enumerate(parameters)}
    old = clip_families(families)
    budgets = {name: .1/math.sqrt(3) for name in families}
    new = clip_families_with_budgets({str(index): [p] for index, p in enumerate(copies)}, budgets)
    for original, copied in zip(parameters, copies):
        torch.testing.assert_close(original.grad, copied.grad, atol=0, rtol=0)
    for name in old:
        torch.testing.assert_close(old[name], new[name], atol=0, rtol=0)
    for invalid in ({}, {'0': .2, '1': .1, '2': .1}, {'0': float('nan'), '1': .01, '2': .01}):
        with pytest.raises(ValueError):
            clip_families_with_budgets(families, invalid)
    with pytest.raises(ValueError, match='disjoint'):
        clip_families_with_budgets({'a': parameters, 'b': parameters}, {'a': .05, 'b': .05})
    report = clip_families_with_budgets({'empty': [], 'absent': [torch.nn.Parameter(torch.ones(1))]},
                                     {'empty': .05, 'absent': .05})
    assert report['empty/preclip'] == 0 and report['absent/preclip'] == 0
    parameters[0].grad.fill_(float('inf'))
    with pytest.raises(RuntimeError, match='non-finite'):
        clip_families_with_budgets({'bad': parameters[:1]}, {'bad': .1})


def test_pipeline_accumulation_budgets_logging_and_strict_contract():
    module = pipeline()
    for parameter in module.model.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.full_like(parameter, 1.)
            parameter.grad.add_(2.)
    report = module.configure_gradient_clipping(None, .1, 'norm')
    groups = module.gradient_clipping_families()
    identities = [id(parameter) for values in groups.values() for parameter in values]
    assert len(identities) == len(set(identities))
    assert set(identities) == {id(p) for p in module.model.parameters() if p.requires_grad}
    assert {id(p) for p in groups['shared_context']} == {id(p) for p in module.model.proposal_context.parameters()}
    for name, parameters in groups.items():
        expected = .1/math.sqrt(6 if name in ('shared_context', 'private_geometry') else 3)
        norm = sum(float(p.grad.double().square().sum()) for p in parameters if p.grad is not None)**.5
        assert norm == pytest.approx(expected, rel=2e-5)
    torch.testing.assert_close(report['geometry/postclip'], torch.sqrt(
        report['shared_context/postclip']**2+report['private_geometry/postclip']**2))
    assert 'shared_context/clip_factor' in report
    assert module.checkpoint_contract()['version'] == 2
    record = dict(state_dict=deepcopy(module.state_dict()))
    module.on_save_checkpoint(record)
    broken = deepcopy(record)
    broken['v55_contract']['version'] = 1
    with pytest.raises(ValueError, match='V55 contract'):
        pipeline().on_load_checkpoint(broken)
    broken = deepcopy(record)
    broken['v55_contract']['gradients'] = 'disjoint_native_selector_geometry;each_.1/sqrt3;shared_context_in_geometry;classification_and_geometry_train_context;raw_context_inputs_detached'
    with pytest.raises(ValueError, match='V55 contract'):
        pipeline().on_load_checkpoint(broken)
    with pytest.raises(ValueError, match='bound'):
        module.configure_gradient_clipping(None, .2, 'norm')


def test_legacy_geometry_family_stays_unchanged():
    module = pipeline()
    groups = gradient_families(module.model)
    assert list(groups) == ['native', 'selector', 'geometry']
    assert {id(p) for p in module.model.proposal_context.parameters()} <= {id(p) for p in groups['geometry']}

from copy import deepcopy

import torch

from src.models.pipelines.pose_estimators.poseflow.proposal_context_v55 import ProposalContext, ContextualProposalSelector


def contextualizer():
    torch.manual_seed(55)
    return ProposalContext(32, 4)


def opened(module):
    torch.nn.init.eye_(module.attention.out_proj.weight)
    return module


def inputs():
    torch.manual_seed(551)
    return torch.randn(2, 6, 32), torch.randn(2, 3, 4, 32), torch.tensor(
        [[[False]*4, [True, False, False, False], [True, True, False, False]]]*2)


def test_initial_identity_empty_prefix_padding_and_permutation():
    module = contextualizer()
    candidates, prefix, valid = inputs()
    corrected = module(candidates, prefix, valid)
    torch.testing.assert_close(corrected[..., :-1, :], candidates[:, None].expand(-1, 3, -1, -1), atol=0, rtol=0)
    opened(module)
    corrected = module(candidates, prefix, valid)
    poisoned = prefix.masked_fill(~valid[..., None], float('nan'))
    torch.testing.assert_close(module(candidates, poisoned, valid), corrected, atol=0, rtol=0)
    order = torch.tensor([3, 1, 0, 2])
    torch.testing.assert_close(module(candidates, prefix[:, :, order], valid[:, :, order]), corrected, atol=2e-6, rtol=2e-5)
    assert torch.isfinite(corrected).all()
    changed = prefix.clone()
    changed[:, 2, 0] += torch.randn(32)
    altered = module(candidates, changed, valid)
    assert not torch.allclose(altered[:, 2], corrected[:, 2])
    assert torch.equal(altered[:, :2], corrected[:, :2])


def test_eos_query_scale_matches_proposals_without_amplifying_shared_correction():
    module = contextualizer()
    assert .5 < float(module.eos_query.square().mean().sqrt()) < 1.5
    selector = ContextualProposalSelector(32)
    candidates, prefix, valid = inputs()
    used = torch.zeros(2, 3, 6, dtype=torch.bool)
    original = module.eos_query.detach().clone()
    norms = []
    for factor in (1., .02):
        with torch.no_grad():
            module.eos_query.copy_(original*factor)
        _, logits = selector(module(candidates, prefix, valid), used)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
        gradient, = torch.autograd.grad(loss, module.attention.out_proj.weight)
        norms.append(gradient.norm())
    assert norms[0] < .2*norms[1]


def test_same_mean_prefixes_are_distinguished_and_candidates_retrieve_differently():
    module = opened(contextualizer())
    candidates, prefix, valid = inputs()
    valid.fill_(True)
    altered = prefix.clone()
    delta = torch.randn_like(prefix[:, :, 0])
    altered[:, :, 0] += delta
    altered[:, :, 1] -= delta
    torch.testing.assert_close(prefix.mean(-2), altered.mean(-2))
    result, other = module(candidates, prefix, valid), module(candidates, altered, valid)
    assert not torch.allclose(result, other)
    correction = result[:, :, :-1]-candidates[:, None]
    assert not torch.allclose(correction[:, :, 0], correction[:, :, 1])


def test_row_gradient_parity_detach_boundaries_and_zero_projection_learns():
    module = contextualizer()
    candidates, prefix, valid = inputs()
    candidates.requires_grad_()
    prefix.requires_grad_()
    result = module(candidates, prefix, valid)
    result.square().sum().backward()
    assert candidates.grad is None and prefix.grad is None
    assert module.attention.out_proj.weight.grad.abs().sum() > 0
    with torch.no_grad():
        module.attention.out_proj.weight -= .001*module.attention.out_proj.weight.grad
    module.zero_grad()
    independent = deepcopy(module)
    module(candidates, prefix, valid).square().sum().backward()
    sum(independent(candidates, prefix[:, row:row+1], valid[:, row:row+1]).square().sum()
        for row in range(3)).backward()
    assert module.attention.in_proj_weight.grad.abs().sum() > 0
    for name, parameter in module.named_parameters():
        torch.testing.assert_close(parameter.grad, dict(independent.named_parameters())[name].grad, atol=3e-5, rtol=3e-4)


def test_scores_used_mask_exhaustion_and_geometry_route():
    module = contextualizer()
    selector = ContextualProposalSelector(32)
    candidates, prefix, valid = inputs()
    context = module(candidates, prefix, valid)
    used = torch.zeros(2, 3, 6, dtype=torch.bool)
    used[..., 0] = True
    used[:, 2] = True
    scores, eos = selector(context, used)
    assert (scores[..., 0] == -1e9).all() and (eos[:, 2] == 30).all()
    assert torch.isfinite(eos).all()
    (scores[:, :2, 1:].square().mean()+eos[:, :2].square().mean()).backward()
    assert module.attention.out_proj.weight.grad.abs().sum() > 0
    assert selector.eos_bias.grad.abs() > 0


def model():
    from src.models.pipelines.pose_estimators.diffusionregpose.model import DiffusionRegPoseModel
    from src.models.pipelines.pose_estimators.poseflow.model_v55 import PoseFlowV55Model
    from tests.test_poseflow_v44 import arguments
    return PoseFlowV55Model(native=DiffusionRegPoseModel(**arguments()), capacity=6).eval()


def test_model_native_same_proposal_parity_context_influence_and_fixed_births():
    from tests.test_poseflow_v53 import model as original_model, SimpleCache
    from tests.test_poseflow_v52 import context
    from src.models.pipelines.pose_estimators.poseflow.rows_v48 import training_rows
    network, original = model(), original_model()
    original.load_state_dict({name: value for name, value in network.state_dict().items()
                              if not name.startswith(('proposal_context.', 'selector.'))}, strict=False)
    for instance in (network, original):
        for head in (*instance.pose_decoder.pose_heads, *instance.pose_decoder.box_heads):
            torch.nn.init.constant_(head.layers[-1].weight, .01)
    torch.manual_seed(552)
    cached = context(SimpleCache(network))
    rows = training_rows(torch.rand(2, 6, 32), torch.tensor([2, 1]))
    rows['proposal_indices'] = torch.arange(3).expand(2, 3, 3).clone()
    values = {name: rows[name] for name in network.row_input_fields}
    actual = network.forward_rows(cached, **values)
    expected = original.forward_rows(cached, **values)
    torch.testing.assert_close(actual['pred_coordinates'], expected['pred_coordinates'], atol=0, rtol=0)
    opened(network.proposal_context)
    together = network.forward_rows(cached, **values)
    for row in range(3):
        alone = network.forward_rows(cached, **{name: value[:, row:row+1] for name, value in values.items()})
        for name in ('pred_coordinates', 'selection_logits', 'eos_logits'):
            torch.testing.assert_close(together[name][..., row:row+1, :] if name != 'eos_logits' else together[name][..., row:row+1],
                                       alone[name], atol=3e-6, rtol=3e-5)
    network.selector.eos_bias.data.fill_(-1e6)
    sizes = torch.tensor([[80., 64.], [64., 80.]])
    generated = network.predict(cached, sizes, collect_trace=True)
    assert generated['person_counts'].tolist() == [6, 6]
    for step in generated['trace']:
        if step['layer'] == 2:
            torch.testing.assert_close(generated['pred_coordinates'][:, step['step']-1], step['pred_coordinates'], atol=0, rtol=0)
    network.selector.eos_bias.data.fill_(1e6)
    assert network.predict(cached, sizes)['person_counts'].tolist() == [0, 0]


def test_per_image_stopping_and_batch_independent_predictions(monkeypatch):
    from tests.test_poseflow_v53 import SimpleCache
    from tests.test_poseflow_v52 import context
    from src.models.pipelines.pose_estimators.poseflow.model_v53 import select_context
    network = model()
    opened(network.proposal_context)
    cached = context(SimpleCache(network))
    with torch.no_grad():
        cached['features'][:, 0, 0] = torch.tensor([-100., 100.])
    select = network.selector.forward
    def scheduled(features, used):
        scores, _ = select(features, used)
        limit = torch.where(features[:, :, 0, 0] < 0, 1, 3)
        return scores, torch.where(used.sum(-1) >= limit, 20., -20.)
    monkeypatch.setattr(network.selector, 'forward', scheduled)
    sizes = torch.tensor([[80., 64.], [64., 80.]])
    together = network.predict(cached, sizes)
    assert together['person_counts'].tolist() == [1, 3]
    assert together['generation_steps'].tolist() == [2, 4]
    for image in range(2):
        alone = network.predict(select_context(cached, torch.tensor([image])), sizes[image:image+1])
        torch.testing.assert_close(together['pred_coordinates'][image:image+1], alone['pred_coordinates'], atol=3e-6, rtol=3e-5)

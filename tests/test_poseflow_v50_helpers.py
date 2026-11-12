import torch

from src.models.pipelines.pose_estimators.poseflow.proposals_v50 import ProposalSelector, gather_identity, residual_coordinates


def test_attention_permutation_masks_and_gradients():
    torch.manual_seed(4)
    selector = ProposalSelector(16)
    prefix = torch.randn(2, 3, 16, requires_grad=True)
    candidates = torch.randn(2, 5, 16, requires_grad=True)
    used = torch.zeros(2, 3, 5, dtype=torch.bool)
    used[:, :, 1] = True
    scores, eos = selector(prefix, candidates, used)
    permutation = torch.tensor([4, 2, 0, 3, 1])
    other, other_eos = selector(prefix, candidates[:, permutation], used[:, :, permutation])
    torch.testing.assert_close(scores[:, :, permutation], other)
    torch.testing.assert_close(eos, other_eos)
    assert (scores[:, :, 1] < -1e8).all()
    (scores[:, :, 0].sum()+eos.sum()).backward()
    assert candidates.grad.abs().sum() > 0
    all_used = torch.ones_like(used)
    _, certain_eos = selector(prefix.detach(), candidates.detach(), all_used)
    assert certain_eos.sigmoid().eq(1).all() and torch.isfinite(certain_eos).all()


def test_gather_detaches_only_identity_branch():
    features = torch.randn(2, 5, 16, requires_grad=True)
    boxes = torch.rand(2, 5, 4, requires_grad=True)
    selected = torch.tensor([[[2, -1]], [[4, 0]]])
    identity, reference = gather_identity(features, boxes, selected)
    assert not identity.requires_grad and not reference.requires_grad
    assert identity[0, 0, 1].eq(0).all()
    torch.testing.assert_close(identity[0, 0, 0], features[0, 2])


def test_residual_zero_identity_positive_sizes_and_gradients():
    clean = torch.rand(2, 3, 32)
    state = 6*(2*clean-1)
    delta = torch.randn_like(clean, requires_grad=True)
    zero = residual_coordinates(state, torch.zeros(2, 3), delta, scale=6.)
    assert torch.equal(zero, (state/6+1)/2)
    sigma = torch.full((2, 3), .1)
    result = residual_coordinates(state, sigma, delta, scale=6.)
    assert result[..., 30:].gt(0).all()
    result.sum().backward()
    assert delta.grad.abs().sum() > 0

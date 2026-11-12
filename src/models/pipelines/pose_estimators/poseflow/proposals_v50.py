import math

import torch
from torch import nn


class ProposalSelector(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.query = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
        self.key = nn.Linear(width, width)
        self.empty_prefix = nn.Parameter(torch.randn(width)*.02)
        self.eos_key = nn.Parameter(torch.randn(width)*.02)
        self.eos_bias = nn.Parameter(torch.zeros(()))

    def forward(self, prefix, candidates, used):
        query = self.query(prefix+self.empty_prefix)
        scores = torch.einsum('brd,bkd->brk', query, self.key(candidates))/math.sqrt(query.shape[-1])
        scores = scores.masked_fill(used, -1e9)
        eos_score = (query*self.eos_key).sum(-1)/math.sqrt(query.shape[-1])+self.eos_bias
        eos_logit = eos_score-torch.logsumexp(scores, -1)
        # Finite certain EOS when every image-local candidate has been consumed.
        eos_logit = torch.where(used.all(-1), torch.full_like(eos_logit, 30.), eos_logit)
        return scores, eos_logit


def gather_identity(features, boxes, indices):
    valid = indices >= 0
    batch_indices = torch.arange(len(features), device=features.device).reshape(-1, *([1]*(indices.ndim-1)))
    identity = features[batch_indices, indices.clamp_min(0)].detach()
    reference = boxes[batch_indices, indices.clamp_min(0)].detach()
    return identity.masked_fill(~valid[..., None], 0.), reference.masked_fill(~valid[..., None], 0.)


def residual_coordinates(noisy, sigmas, delta, *, scale):
    alpha = (1-sigmas.square()).clamp_min(1e-8).sqrt()
    base = (noisy/alpha[..., None]/scale+1)/2
    xy = base[..., :30]+sigmas[..., None]*delta[..., :30]/(2*scale)
    size = base[..., 30:].clamp_min(1e-6)*torch.exp((sigmas[..., None]*delta[..., 30:]).clamp(-10, 10))
    result = torch.cat((xy, size), -1)
    # Preserve the exact input arithmetic at sigma zero, including box sizes.
    return torch.where((sigmas == 0)[..., None], (noisy/scale+1)/2, result)

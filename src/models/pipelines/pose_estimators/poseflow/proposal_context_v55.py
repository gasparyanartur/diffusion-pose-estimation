import torch
from torch import nn


class ProposalContext(nn.Module):
    def __init__(self, width, heads, *, detach_prefix=True):
        super().__init__()
        self.detach_prefix = detach_prefix
        self.query_norm = nn.LayerNorm(width)
        self.prefix_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)
        self.empty_prefix = nn.Parameter(torch.randn(width)*.02)
        self.eos_query = nn.Parameter(torch.randn(width))
        nn.init.zeros_(self.attention.out_proj.weight)
        nn.init.zeros_(self.attention.out_proj.bias)

    def forward(self, candidates, prefix, valid):
        # B images, R independent rows, K candidates, Q prefix positions, D channels.
        batch, rows, queries, width = prefix.shape
        candidates = candidates.detach()[:, None].expand(-1, rows, -1, -1)
        eos = self.eos_query.expand(batch, rows, 1, width)
        query = torch.cat((candidates, eos), -2)
        prefix = (prefix.detach() if self.detach_prefix else prefix).masked_fill(~valid[..., None], 0.)
        empty = self.empty_prefix.expand(batch, rows, 1, width)
        keys = self.prefix_norm(torch.cat((prefix, empty), -2)).reshape(batch*rows, queries+1, width)
        key_mask = torch.cat((~valid, valid.any(-1, keepdim=True)), -1).reshape(batch*rows, queries+1)
        attended = self.attention(self.query_norm(query).reshape(batch*rows, -1, width), keys, keys,
                                  key_padding_mask=key_mask, need_weights=False)[0]
        return query+attended.reshape_as(query)


class ContextualProposalSelector(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.candidate_score = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))
        self.eos_score = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))
        self.eos_bias = nn.Parameter(torch.zeros(()))

    def forward(self, contextualized, used):
        scores = self.candidate_score(contextualized[..., :-1, :]).squeeze(-1).masked_fill(used, -1e9)
        eos = self.eos_score(contextualized[..., -1, :]).squeeze(-1)+self.eos_bias-torch.logsumexp(scores, -1)
        return scores, torch.where(used.all(-1), torch.full_like(eos, 30.), eos)

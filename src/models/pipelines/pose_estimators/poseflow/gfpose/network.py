# Adapted from Embracing/GFPose under Apache-2.0; see NOTICE and LICENSE.
import math

import torch
from torch import nn


def timestep_embedding(times, width=512):
    half = width//2
    frequencies = torch.exp(torch.arange(half, dtype=torch.float32, device=times.device)
                            *(-math.log(10000)/(half-1)))
    angles = times.float()[:, None]*frequencies[None]
    return torch.cat((angles.sin(), angles.cos()), dim=1)


class GFPoseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        width, embedding = 1024, 512
        self.pre_dense = nn.Linear(51, width)
        self.pre_dense_t = nn.Linear(embedding, width)
        self.pre_dense_cond = nn.Linear(width, width)
        self.pre_gnorm = nn.GroupNorm(32, width)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(.25)
        self.shared_time_embed = nn.Sequential(nn.Linear(embedding, embedding), self.act)
        self.register_buffer('sigmas', torch.logspace(math.log10(50), math.log10(.01), 1000,
                                                     dtype=torch.float64))
        self.cond_embed = nn.Sequential(nn.Linear(51, width), self.act)
        for block in (1, 2):
            for layer in (1, 2):
                prefix = f'b{block}_dense{layer}'
                setattr(self, prefix, nn.Linear(width, width))
                setattr(self, prefix+'_t', nn.Linear(embedding, width))
                setattr(self, prefix+'_cond', nn.Linear(width, width))
                setattr(self, f'b{block}_gnorm{layer}', nn.GroupNorm(32, width))
        self.post_dense = nn.Linear(width, 51)

    def forward(self, poses, labels):
        batch = poses.shape[0]
        time = self.shared_time_embed(timestep_embedding(labels))
        # Upstream null evaluation masks the relative condition before embedding.
        condition = self.cond_embed(torch.zeros_like(poses).reshape(batch, 51))
        hidden = self.pre_dense(poses.reshape(batch, 51))
        hidden = hidden+self.pre_dense_t(time)
        hidden = hidden+self.pre_dense_cond(condition)
        hidden = self.dropout(self.act(self.pre_gnorm(hidden)))
        for block in (1, 2):
            first = self._block_layer(hidden, time, condition, block, 1)
            second = self._block_layer(first, time, condition, block, 2)
            hidden = hidden+second
        return self.post_dense(hidden).reshape(batch, 17, 3)

    def _block_layer(self, hidden, time, condition, block, layer):
        prefix = f'b{block}_dense{layer}'
        value = getattr(self, prefix)(hidden)
        value = value+getattr(self, prefix+'_t')(time)
        value = value+getattr(self, prefix+'_cond')(condition)
        value = getattr(self, f'b{block}_gnorm{layer}')(value)
        return self.dropout(self.act(value))

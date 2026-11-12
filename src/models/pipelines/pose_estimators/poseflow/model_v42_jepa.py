from copy import deepcopy

import torch
from torch import nn


class EMAPoseEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.pose_encoder = deepcopy(model.pose_encoder)
        self.norm = deepcopy(model.transformer.decoder.norm)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, coordinates):
        return self.norm(self.pose_encoder(coordinates))

    @torch.no_grad()
    def update_from(self, model, decay):
        for teacher, online in ((self.pose_encoder, model.pose_encoder),
                                (self.norm, model.transformer.decoder.norm)):
            online_parameters = dict(online.named_parameters())
            for name, value in teacher.named_parameters():
                value.lerp_(online_parameters[name].detach(), 1 - decay)
            online_buffers = dict(online.named_buffers())
            for name, value in teacher.named_buffers():
                source = online_buffers[name].detach()
                if value.is_floating_point():
                    value.lerp_(source, 1 - decay)
                else:
                    value.copy_(source)

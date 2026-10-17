import math

import torch
from torch import nn

from .model_v15 import PoseFlowV15Model


class JointMessageDropout(nn.Module):
    def __init__(self, probability, joint_slice):
        super().__init__()
        self.probability = probability
        self.joint_slice = joint_slice
        self.last_drop_mask = None
        self.generator = None

    def forward(self, message):
        self.last_drop_mask = None
        if not math.isfinite(self.probability) or not 0 <= self.probability <= 1:
            raise ValueError('Joint message dropout probability must be in [0,1]')
        if not self.training or self.probability == 0:
            return message
        joints = message[..., self.joint_slice, :]
        mask_shape = joints.shape[:-1]
        if self.probability == 1:
            self.last_drop_mask = torch.ones(mask_shape, device=message.device, dtype=torch.bool)
            updated = joints*0
        else:
            self.last_drop_mask = torch.rand(mask_shape, device=message.device, generator=self.generator) < self.probability
            updated = joints*(~self.last_drop_mask)[..., None]/(1-self.probability)
        start = self.joint_slice.start
        stop = self.joint_slice.stop
        if stop is None:
            stop = message.shape[-2]
        elif stop < 0:
            stop += message.shape[-2]
        return torch.cat((message[..., :start, :], updated, message[..., stop:, :]), dim=-2)


class PoseFlowPathDropoutModel(PoseFlowV15Model):
    def __init__(self, image_model, *, sampling_steps=3):
        super().__init__(image_model, sampling_steps=sampling_steps)
        self.configure_path_dropout(.25, .25)

    def configure_path_dropout(self, image_probability, self_probability):
        self.evidence_extractor.image_message_transform = JointMessageDropout(image_probability, slice(1, None)).train(self.training)
        for layer in self.layers:
            layer.aggregate.message_transform = JointMessageDropout(self_probability, slice(1, -1)).train(self.training)

    def set_path_dropout_generator(self, generator):
        self.evidence_extractor.image_message_transform.generator = generator
        for layer in self.layers:
            layer.aggregate.message_transform.generator = generator

    def _iteration_diagnostics(self, index, clean, global_tokens, previous_global,
                               hypotheses, previous_memory, attention, geometry):
        row = super()._iteration_diagnostics(index, clean, global_tokens, previous_global,
                                            hypotheses, previous_memory, attention, geometry)
        image_mask = self.evidence_extractor.image_message_transform.last_drop_mask
        self_mask = self.layers[index].aggregate.message_transform.last_drop_mask
        zero = clean.detach().new_zeros(())
        row['joint_image_drop_fraction'] = zero if image_mask is None else image_mask.float().mean()
        row['joint_self_drop_fraction'] = zero if self_mask is None else self_mask.float().mean()
        row['joint_both_drop_fraction'] = (zero if image_mask is None or self_mask is None else
            (image_mask & self_mask.reshape_as(image_mask)).float().mean())
        return row

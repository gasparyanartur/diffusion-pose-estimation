import torch
from torch import nn

from .model_v35 import PoseFlowV35Model


class PoseFlowV36Model(PoseFlowV35Model):
    def __init__(self, *, clean_existence_input=False, clean_box_input_format='pixels', **kwargs):
        super().__init__(**kwargs)
        if clean_box_input_format not in ('pixels', 'normalized'):
            raise ValueError('Unknown clean box input format')
        self.clean_box_input_format = clean_box_input_format
        self.clean_existence_input = clean_existence_input
        del self.clean_layers
        if clean_existence_input:
            with torch.random.fork_rng(devices=[]):
                original = self.input_mlp[0]
                expanded = nn.Linear(5, self.hidden_dim)
                with torch.no_grad():
                    expanded.weight[:, :4].copy_(original.weight)
                    expanded.bias.copy_(original.bias)
                self.input_mlp[0] = expanded

    def encode_clean(self, input_pixels, *, existence=None):
        if input_pixels.ndim != 3 or input_pixels.shape[-2:] != (self.num_proposals, 4):
            raise ValueError('Clean boxes must have shape [B, P, 4]')
        features = input_pixels if self.clean_box_input_format == 'normalized' else input_pixels/100.
        if self.clean_existence_input:
            if existence is None or existence.shape != input_pixels.shape[:2]:
                raise ValueError('Clean existence labels must have shape [B, P]')
            if not ((existence == 0) | (existence == 1)).all():
                raise ValueError('Clean existence labels must be binary')
            features = torch.cat((features, existence.to(features)[..., None]), -1)
        elif existence is not None:
            raise ValueError('Box-only clean encoder does not accept existence labels')
        return self.readout(self.clean_norm(self.input_mlp(features)))

    def training_outputs(self, images, input_pixels, *, timesteps, noise, padding_mask=None,
                         existence=None):
        clean = self.encode_clean(input_pixels, existence=existence)
        encoded = self.encode_image(images, padding_mask)
        noisy = self.q_sample(clean['pred_latents'].detach(), timesteps, noise)
        recovered = self.denoise(noisy, timesteps, encoded)
        return clean, recovered, noisy, timesteps

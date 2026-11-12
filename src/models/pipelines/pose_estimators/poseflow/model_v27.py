import torch
from torch import Tensor

from .model_v26 import PoseFlowV26Model


class PoseFlowV27Model(PoseFlowV26Model):
    feature_strides = (32, 16, 8)

    def __init__(self, *, num_feature_levels=3, **kwargs):
        if num_feature_levels != 3:
            raise ValueError('V27 requires three image levels: strides32,16,8')
        super().__init__(num_feature_levels=num_feature_levels, **kwargs)

    def encode_image(self, images: Tensor, padding_mask: Tensor | None = None):
        if padding_mask is None:
            padding_mask = torch.zeros(images.shape[0], *images.shape[-2:],
                                       dtype=torch.bool, device=images.device)
        features, masks, positions = self.backbone(images, padding_mask)
        levels = []
        for index in (2, 1, 0):
            feature = self.input_proj[index](features[index])
            memory = feature.flatten(2).transpose(1, 2)  # <B, H_level*W_level, D>
            position = positions[index].to(feature.dtype).flatten(2).transpose(1, 2)
            position = position+self.level_embed[index][None, None]
            mask = masks[index].flatten(1)
            if mask.all(-1).any():
                raise ValueError('V27 requires valid image tokens at every feature level')
            levels.append((memory, position, mask))
        return tuple(levels)

    def _run_latent_stack(self, state, timesteps, encoded_image, layers):
        if len(encoded_image) != 3 or len(layers) != 3:
            raise ValueError('V27 routes exactly three blocks to three image levels')
        time_features = self.time_embedding(timesteps.to(state.dtype))
        for layer, level in zip(layers, encoded_image):
            state = layer(state, time_features, level)
        return state

    def encode_clean(self, input_pixels: Tensor, encoded_image):
        state = self.input_mlp(input_pixels/100.)
        times = state.new_zeros(state.shape[0])
        state = self._run_latent_stack(state, times, encoded_image, self.clean_layers)
        return self.readout(self.clean_norm(state))

    def denoise(self, noisy_latents: Tensor, timesteps: Tensor, encoded_image):
        expected = (encoded_image[0][0].shape[0], self.num_proposals, self.hidden_dim)
        if noisy_latents.shape != expected or timesteps.shape != (expected[0],):
            raise ValueError('V27 latent or timestep shape disagrees with the proposal set')
        if ((timesteps < 0) | (timesteps >= self.num_timesteps)).any():
            raise ValueError('V27 timestep outside diffusion schedule')
        state = self._run_latent_stack(noisy_latents, timesteps, encoded_image, self.denoiser_layers)
        return self.readout(self.denoiser_projection(state))

import torch
from torch import nn

from .model_v35 import PoseFlowV35Model


class PoseFlowV38Model(PoseFlowV35Model):
    def __init__(self, *, posterior_logvar_init=-4., posterior_logvar_min=-10.,
                 posterior_logvar_max=2., **kwargs):
        super().__init__(**kwargs)
        if not posterior_logvar_min <= posterior_logvar_init <= posterior_logvar_max:
            raise ValueError('Posterior initialization must be inside the logvariance bounds')
        self.posterior_logvar_min = posterior_logvar_min
        self.posterior_logvar_max = posterior_logvar_max
        with torch.random.fork_rng(devices=[]):
            self.posterior_mean = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.posterior_logvar = nn.Linear(self.hidden_dim, self.hidden_dim)
            nn.init.eye_(self.posterior_mean.weight)
            nn.init.zeros_(self.posterior_mean.bias)
            nn.init.zeros_(self.posterior_logvar.weight)
            nn.init.constant_(self.posterior_logvar.bias, posterior_logvar_init)

    def encode_clean(self, input_pixels, encoded_image):
        state = self.input_mlp(input_pixels/100.)
        time_features = self.time_embedding(state.new_zeros(state.shape[0]))
        for layer in self.clean_layers:
            state = layer(state, time_features, encoded_image)
        features = self.clean_norm(state)
        mean = self.posterior_mean(features)
        logvar = self.posterior_logvar(features).clamp(self.posterior_logvar_min, self.posterior_logvar_max)
        return dict(self.readout(mean), posterior_logvar=logvar)

    def training_outputs(self, images, input_pixels, *, timesteps, noise, posterior_noise,
                         padding_mask=None):
        encoded = self.encode_image(images, padding_mask)
        clean = self.encode_clean(input_pixels, encoded)
        mean = clean['pred_latents']
        if posterior_noise.shape != mean.shape:
            raise ValueError('Posterior noise must match the clean latent shape')
        sample = mean+(clean['posterior_logvar']*.5).exp()*posterior_noise
        sampled = self.readout(sample)
        clean.update(posterior_sample=sample, sampled_pred_boxes=sampled['pred_boxes'],
                     sampled_existence_logits=sampled['existence_logits'])
        noisy = self.q_sample(mean.detach(), timesteps, noise)
        recovered = self.denoise(noisy, timesteps, encoded)
        return clean, recovered, noisy, timesteps

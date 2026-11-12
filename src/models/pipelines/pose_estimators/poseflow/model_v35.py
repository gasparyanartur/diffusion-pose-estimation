import math

import torch
from torch import nn

from .latent_v35 import grid_anchors
from .model_v26 import PoseFlowV26Model


class PoseFlowV35Model(PoseFlowV26Model):
    def __init__(self, *, spatial_queries=False, grid_rows=5, grid_columns=10, **kwargs):
        super().__init__(**kwargs)
        self.spatial_queries = spatial_queries
        if grid_rows*grid_columns != self.num_proposals:
            raise ValueError('Grid rows*columns must equal proposal budget')
        del self.joint_head
        with torch.random.fork_rng(devices=[]):
            self.input_mlp = nn.Sequential(nn.Linear(4, self.hidden_dim), nn.GELU(),
                                           nn.Linear(self.hidden_dim, self.hidden_dim))
            self.box_head = nn.Linear(self.hidden_dim, 4)
            self.existence_head = nn.Linear(self.hidden_dim, 1)
            if spatial_queries:
                self.register_buffer('query_anchors', grid_anchors(grid_rows, grid_columns))
                self.register_buffer('query_frequencies', torch.tensor([1., 2., 4., 8.]))
                self.query_projection = nn.Linear(16, self.hidden_dim)

    def query_positions(self):
        phase = 2*math.pi*self.query_anchors[..., None]*self.query_frequencies
        features = torch.stack((phase.sin(), phase.cos()), -1).flatten(1)
        return self.query_projection(features)[None]  # <1, P, D>

    def readout(self, latents):
        existence = self.existence_head(latents).squeeze(-1)
        return dict(pred_latents=latents, existence_logits=existence,
            pred_logits=torch.stack((torch.full_like(existence, -20.), existence), -1),
            pred_boxes=self.box_head(latents).sigmoid())

    def denoise(self, noisy_latents, timesteps, encoded_image):
        if not self.spatial_queries:
            return super().denoise(noisy_latents, timesteps, encoded_image)
        expected = (encoded_image[0].shape[0], self.num_proposals, self.hidden_dim)
        if noisy_latents.shape != expected or timesteps.shape != (expected[0],):
            raise ValueError('Latent/timestep shape disagrees with proposal budget')
        if ((timesteps < 0) | (timesteps >= self.num_timesteps)).any():
            raise ValueError('Timestep outside diffusion schedule')
        state = noisy_latents
        time_features = self.time_embedding(timesteps.to(state.dtype))
        position = self.query_positions()
        for layer in self.denoiser_layers:
            state = layer(state, time_features, encoded_image, query_position=position)
        return self.readout(self.denoiser_projection(state))

    def sample_boxes(self, images, **kwargs):
        return self.sample_keypoints(images, **kwargs)

    def postprocess(self, output, target_sizes, **kwargs):
        from .quick_eval_v35 import postprocess_boxes
        return postprocess_boxes(output, target_sizes)

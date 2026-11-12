import torch
from torch import nn

from ..diffusionregpose.model import DiffusionRegPoseModel
from ..diffusionregpose.transformer import MLP, SinusoidalPositionEmbeddings
from .human_context import NativeHumanContext


class PoseFlowV44Model(NativeHumanContext, DiffusionRegPoseModel):
    sampler_kind = 'frozen human latent cosine DDIM eta0; native cached proposals'

    def __init__(self, *, codec, native_state=None, prediction_type='x0', **kwargs):
        super().__init__(**kwargs)
        if prediction_type not in ('x0', 'v'):
            raise ValueError('Unknown V44 prediction type')
        self.prediction_type = prediction_type
        if native_state is not None:
            self.load_state_dict(native_state, strict=True)
        self.ddim_sampling_eta = 0.
        self.scale = 1.
        decoder = self.transformer.decoder
        if not 1 <= decoder.num_box_decoder_layers < decoder.num_layers:
            raise ValueError('V44 needs native box layers followed by latent layers')
        if self.num_proposals != decoder.num_group or self.num_proposals > self.num_queries:
            raise ValueError('V44 requires groups == proposals <= queries')
        self.codec = codec.requires_grad_(False).eval()
        self.latent_dim = codec.latent_dim
        self.num_body_points = decoder.num_body_points
        width = decoder.d_model
        self.latent_input = MLP(self.latent_dim, width, width, 2)
        self.latent_time = nn.Sequential(SinusoidalPositionEmbeddings(width),
            nn.Linear(width, width * 4), nn.GELU(), nn.Linear(width * 4, width))
        self.latent_heads = nn.ModuleList(nn.Linear(width, self.latent_dim)
            for _ in decoder.layers[decoder.num_box_decoder_layers:])
        for name in ('pose_embed', 'pose_hw_embed', 'expansion_pose_embed'):
            delattr(self, name)
            delattr(decoder, name)
        for name in ('hw', 'keypoint_embed', 'emb_Q', 'emb_K', 'emb_V', 'emb_SQ', 'emb_SK', 'emb_SV'):
            delattr(decoder, name)
        self.initialization_report = {}

    @property
    def flow_sampling_steps(self):
        return self.sampling_timesteps

    @flow_sampling_steps.setter
    def flow_sampling_steps(self, value):
        if not 1 <= value <= self.num_timesteps:
            raise ValueError('Sampling steps outside diffusion schedule')
        self.sampling_timesteps = int(value)

    def _decode(self, latent, context, image_wh):
        sizes = image_wh[:, None].expand(-1, self.num_proposals, -1).to(latent)
        geometry = self.codec.decode_latents(latent, sizes)
        points = geometry['keypoints_px'] / sizes[..., None, :]
        logits = context['existence_logits']
        return dict(geometry, pred_latents=latent, existence_logits=logits,
            pred_logits=torch.stack((torch.full_like(logits, -20.), logits), -1),
            pred_boxes=geometry['boxes_cxcywh_px'] / sizes.repeat(1, 1, 2),
            pred_keypoints=torch.cat((points.flatten(-2), torch.ones_like(points[..., 0])), -1))

    def denoise(self, h_t, t, context, image_wh):
        batch = len(context['features'])
        if h_t.shape != (batch, self.num_proposals, self.latent_dim):
            raise ValueError('Latent state shape disagrees with selected proposals')
        if t.shape != (batch,) or t.dtype != torch.long or ((t < 0) | (t >= self.num_timesteps)).any():
            raise ValueError('Expected valid long timesteps shaped [B]')
        if image_wh.shape != (batch, 2) or not torch.isfinite(image_wh).all() or (image_wh <= 0).any():
            raise ValueError('Expected finite positive image sizes shaped [B,2]')
        features = context['features'] + self.latent_input(h_t) + self.latent_time(t.to(h_t.dtype))[:, None]
        features = features.transpose(0, 1)
        references = context['boxes'].transpose(0, 1)
        decoder, outputs = self.transformer.decoder, []
        for layer, head in zip(decoder.layers[decoder.num_box_decoder_layers:], self.latent_heads):
            features = self._decoder_layer(layer, features, references, context['memory'], context['inputs'])
            prediction = head(decoder.norm(features).transpose(0, 1))
            latent = self.clean_from_velocity(h_t, prediction, t) if self.prediction_type == 'v' else prediction
            output = self._decode(latent, context, image_wh)
            if self.prediction_type == 'v':
                output['pred_velocity'] = prediction
            outputs.append(output)
        return dict(outputs[-1], aux_outputs=outputs[:-1])

    def _velocity_coefficients(self, state, timesteps):
        shape = (len(state),) + (1,) * (state.ndim - 1)
        alpha = self.sqrt_alphas_cumprod[timesteps].to(state).reshape(shape)
        sigma = self.sqrt_one_minus_alphas_cumprod[timesteps].to(state).reshape(shape)
        return alpha, sigma

    def velocity_target(self, clean, noise, timesteps):
        alpha, sigma = self._velocity_coefficients(clean, timesteps)
        return alpha * noise - sigma * clean

    def clean_from_velocity(self, state, velocity, timesteps):
        alpha, sigma = self._velocity_coefficients(state, timesteps)
        return alpha * state - sigma * velocity

    def forward(self, images, noisy_latents, timesteps, *, image_wh, padding_mask=None):
        return self.denoise(noisy_latents, timesteps, self.prepare_context(images, padding_mask), image_wh)

    def q_sample(self, x_start, timesteps, noise=None):
        return super().q_sample(x_start, timesteps, noise).to(x_start)

    def predict_noise_from_start(self, x_t, timesteps, x_start):
        return super().predict_noise_from_start(x_t, timesteps, x_start).to(x_t)

    def ddim_step(self, x_start, predicted_noise, time, time_next, noise=None):
        return super().ddim_step(x_start, predicted_noise, time, time_next, noise).to(x_start)

    @torch.no_grad()
    def sample(self, context, image_wh, steps=None, generator=None, *, collect_step_diagnostics=False):
        count = self.sampling_timesteps if steps is None else steps
        if not 1 <= count <= self.num_timesteps:
            raise ValueError('Sampling steps outside diffusion schedule')
        reference = context['features']
        state = torch.randn(len(reference), self.num_proposals, self.latent_dim,
            device=reference.device, dtype=reference.dtype, generator=generator)
        # Construct locally so sampling does not mutate the model's schedule configuration.
        times = torch.linspace(-1, self.num_timesteps - 1, count + 1).long().tolist()[::-1]
        diagnostics = []
        for time, next_time in zip(times[:-1], times[1:]):
            timestep = torch.full((len(state),), time, device=state.device, dtype=torch.long)
            output = self.denoise(state, timestep, context, image_wh)
            noise = self.predict_noise_from_start(state, timestep, output['pred_latents']).to(state)
            state = self.ddim_step(output['pred_latents'], noise, time, next_time, noise=torch.zeros_like(state)).to(state)
            if collect_step_diagnostics:
                diagnostics.append(dict(time=time, next_time=next_time, latent_rms=float(state.square().mean().sqrt())))
        return dict(output, latent_state=state, latent_step_diagnostics=diagnostics,
            reference_indices=context['indices'], reference_boxes=context['boxes'])

    @torch.no_grad()
    def sample_keypoints(self, images, *, image_wh, padding_mask=None, generator=None,
                         collect_step_diagnostics=False):
        return self.sample(self.prepare_context(images, padding_mask), image_wh, generator=generator,
            collect_step_diagnostics=collect_step_diagnostics)

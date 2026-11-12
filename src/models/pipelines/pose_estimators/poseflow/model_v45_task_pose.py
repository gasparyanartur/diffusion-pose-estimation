from copy import deepcopy

import torch
from torch import nn

from ..diffusionregpose.transformer import SinusoidalPositionEmbeddings, sine_embedding_for_position
from .model_v42 import PoseFlowV42Model
from .model_v44 import PoseFlowV44Model


class TaskPoseCodec(nn.Module):
    def __init__(self, coordinate_count, width=256, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(coordinate_count, width), nn.GELU(), nn.Linear(width, latent_dim))
        self.norm = nn.LayerNorm(latent_dim, elementwise_affine=False)
        self.reader = nn.Sequential(nn.Linear(latent_dim, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, coordinate_count), nn.Sigmoid())

    def encode(self, coordinates):
        return self.norm(self.encoder(coordinates))

    def decode(self, latent):
        return self.reader(latent)


class EMATaskPoseEncoder(nn.Module):
    def __init__(self, codec):
        super().__init__()
        self.encoder = deepcopy(codec.encoder)
        self.norm = deepcopy(codec.norm)
        self.requires_grad_(False).eval()

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, coordinates):
        return self.norm(self.encoder(coordinates))

    @torch.no_grad()
    def update_from(self, codec, decay):
        online = dict(codec.encoder.named_parameters())
        for name, value in self.encoder.named_parameters():
            value.lerp_(online[name].detach(), 1 - decay)


class PoseFlowV45TaskPoseModel(PoseFlowV42Model):
    sampler_kind = 'native coordinate cosine DDIM eta0; cached time-independent human context; task-pose codec'
    confidence_description = 'cached time-independent selected image features; no coordinate dependence'
    _decoder_layer = PoseFlowV44Model._decoder_layer

    def __init__(self, *, latent_dim=32, **kwargs):
        super().__init__(pose_readout_mode='absolute', confidence_mode='image_only', **kwargs)
        decoder = self.transformer.decoder
        width = decoder.d_model
        self.latent_dim = latent_dim
        del self.pose_encoder
        del self.pose_readers
        self.pose_codec = TaskPoseCodec(2 * decoder.num_body_points, width, latent_dim)
        self.latent_adapter = nn.Linear(latent_dim, width)
        self.pose_time = nn.Sequential(SinusoidalPositionEmbeddings(width),
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
        self.latent_residual = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, latent_dim))
        nn.init.zeros_(self.latent_residual[-1].weight)
        nn.init.zeros_(self.latent_residual[-1].bias)

    def prepare_context(self, images, padding_mask=None, *, targets=None):
        context = PoseFlowV44Model.prepare_context(self, images, padding_mask, targets=targets)
        context['image_logits'] = self.image_confidence_head(self.transformer.decoder.norm(context['features']))
        return context

    def denoise_coordinates(self, state, timesteps, context):
        decoder = self.transformer.decoder
        expected = (len(context['features']), self.num_proposals, 2 * decoder.num_body_points)
        if state.shape != expected:
            raise ValueError(f'Expected coordinate state shaped {expected}')
        if (timesteps.shape != (expected[0],) or timesteps.dtype != torch.long
                or timesteps.device != state.device or ((timesteps < 0) | (timesteps >= self.num_timesteps)).any()):
            raise ValueError('Expected valid long timesteps [B] on the coordinate device')
        encoded = self.pose_codec.encode((state / self.scale + 1) / 2)
        features = (context['features'] + self.latent_adapter(encoded)
                    + self.pose_time(timesteps.to(state.dtype))[:, None]).transpose(0, 1)
        references = context['boxes'].transpose(0, 1)
        inputs = context['inputs']
        scaled = references[:, :, None] * torch.cat((inputs.valid_ratios, inputs.valid_ratios), -1)[None]
        position = decoder.ref_point_head(sine_embedding_for_position(scaled[:, :, 0], num_pos_feats=decoder.d_model // 2))
        predictions = []
        for layer, attention in zip(decoder.layers[decoder.num_box_decoder_layers:], self.person_attention):
            features = self._person_layer(layer, attention, features, position, scaled,
                context['memory'], inputs, None, 0)
            normalized = decoder.norm(features).transpose(0, 1)
            latent = self.pose_codec.norm(encoded + self.latent_residual(normalized))
            xy = self.pose_codec.decode(latent)
            visibility = xy.new_ones(*xy.shape[:2], decoder.num_body_points).sigmoid()
            predictions.append(dict(pred_logits=context['image_logits'], pred_boxes=context['boxes'],
                pred_keypoints=torch.cat((xy, visibility), -1), pred_latents=latent))
        return dict(predictions[-1], aux_outputs=predictions[:-1])

    def forward(self, images, diffusion_keypoints, timesteps, targets=None, padding_mask=None, *, pose_input_builder=None):
        context = self.prepare_context(images, padding_mask, targets=targets)
        matches = None
        if pose_input_builder is not None:
            diffusion_keypoints, matches = pose_input_builder(
                context['pred_logits'], context['boxes'], timesteps, targets)
        outputs = self.denoise_coordinates(diffusion_keypoints, timesteps, context)
        points = self.transformer.decoder.num_body_points
        frontend = [dict(row, pred_keypoints=row['pred_boxes'].new_zeros(*row['pred_boxes'].shape[:2], 3 * points))
                    for row in context['frontend_outputs']]
        outputs['aux_outputs'] = frontend + outputs['aux_outputs']
        if matches is not None:
            outputs['pose_target_matches'] = matches
        if self.training:
            outputs['interm_outputs'] = dict(context['encoder_output'],
                pred_keypoints=context['boxes'].new_zeros(len(images), self.num_queries, 3 * points))
            outputs.update(context.get('dn_output', {}))
        return outputs

    @torch.no_grad()
    def sample_coordinates(self, context, steps=None, generator=None):
        steps = self.sampling_timesteps if steps is None else steps
        if not 1 <= steps <= self.num_timesteps:
            raise ValueError('Sampling steps outside the diffusion schedule')
        reference = context['features']
        state = torch.randn(len(reference), self.num_proposals, 2 * self.transformer.decoder.num_body_points,
            device=reference.device, dtype=reference.dtype, generator=generator)
        times = torch.linspace(-1, self.num_timesteps - 1, steps + 1).long().tolist()[::-1]
        for time, next_time in zip(times[:-1], times[1:]):
            timesteps = torch.full((len(state),), time, device=state.device, dtype=torch.long)
            outputs = self.denoise_coordinates(state, timesteps, context)
            clean = ((outputs['pred_keypoints'][..., :state.shape[-1]] * 2 - 1) * self.scale).clamp(-self.scale, self.scale)
            noise = self.predict_noise_from_start(state, timesteps, clean).to(state)
            state = self.ddim_step(clean, noise, time, next_time, torch.zeros_like(state)).to(state)
        return dict(outputs, diffusion_keypoints=state, reference_indices=context['indices'], reference_boxes=context['boxes'])

    @torch.no_grad()
    def sample_keypoints(self, images, *, padding_mask=None, num_proposals=None, generator=None):
        if num_proposals is not None and num_proposals != self.num_proposals:
            raise ValueError('Sampling proposal count must equal selected person capacity')
        return self.sample_coordinates(self.prepare_context(images, padding_mask), generator=generator)

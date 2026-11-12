import math
import torch
from torch import nn

from ..diffusionregpose.backbone import DiffusionRegPoseResNetBackbone, PositionEmbeddingSineHW
from ..diffusionregpose.diffusion import DiffusionRegPoseDiffusion
from ..diffusionregpose.model import _BackboneJoiner
from ..diffusionregpose.transformer import SinusoidalPositionEmbeddings, build_input_projections
from .model_v40 import ReferenceLatentImageBlock, ReferenceProposalMixin, initialize_reference_frontend
from .noise_v43 import noise_coefficients, clean_from_velocity


class PoseFlowV43Model(ReferenceProposalMixin, DiffusionRegPoseDiffusion):
    sampler_kind = 'frozen pose-box latent cosine DDIM eta0'

    def __init__(self, *, codec, backbone_pretrained=True, timesteps=1000, sampling_steps=4,
                 num_proposals=50, hidden_dim=256, num_heads=8, feedforward_dim=2048,
                 num_feature_levels=4, reference_candidates=200, reference_encoder_layers=3,
                 freeze_backbone_stem_and_layer1=True, freeze_backbone=False, prediction_type='x0'):
        super().__init__(timesteps=timesteps, sampling_timesteps=sampling_steps, scale=1., ddim_sampling_eta=0.)
        if reference_candidates < num_proposals or num_proposals < 1 or hidden_dim % num_heads:
            raise ValueError('Invalid reference budget or attention dimensions')
        if prediction_type not in ('x0', 'v'):
            raise ValueError('Unknown prediction type')
        self.prediction_type = prediction_type
        self.codec = codec
        self.latent_dim, self.hidden_dim = codec.latent_dim, hidden_dim
        self.num_proposals = self.num_queries = num_proposals
        self.num_body_points, self.num_feature_levels = 14, num_feature_levels
        self.reference_candidates = reference_candidates
        self.backbone = _BackboneJoiner(DiffusionRegPoseResNetBackbone(
            pretrained=backbone_pretrained, freeze_stem_and_layer1=freeze_backbone_stem_and_layer1),
            PositionEmbeddingSineHW(hidden_dim//2, temperature_h=20, temperature_w=20))
        if freeze_backbone:
            self.backbone.requires_grad_(False)
        self.input_proj = build_input_projections((512, 1024, 2048), hidden_dim=hidden_dim, num_levels=num_feature_levels)
        for projection in self.input_proj:
            nn.init.xavier_uniform_(projection[0].weight)
            nn.init.zeros_(projection[0].bias)
        self.level_embed = nn.Parameter(torch.randn(num_feature_levels, hidden_dim))
        initialize_reference_frontend(self, reference_encoder_layers=reference_encoder_layers,
            feedforward_dim=feedforward_dim, num_heads=num_heads)
        self.latent_input = nn.Linear(self.latent_dim, hidden_dim)
        self.time_embedding = nn.Sequential(SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim*4), nn.GELU(), nn.Linear(hidden_dim*4, hidden_dim))
        self.denoiser_layers = nn.ModuleList(
            ReferenceLatentImageBlock(hidden_dim, num_heads, feedforward_dim) for _ in range(3))
        self.latent_output = nn.Linear(hidden_dim, self.latent_dim)
        if prediction_type == 'v':
            nn.init.zeros_(self.latent_output.weight)
            nn.init.zeros_(self.latent_output.bias)
        self.slot_confidence_head = nn.Linear(hidden_dim, 1)
        self.initialization_report = {}
        self.initialize_slot_confidence()

    @torch.no_grad()
    def initialize_slot_confidence(self):
        self.slot_confidence_head.weight.copy_(self.proposal_class_head.weight[1:2])
        self.slot_confidence_head.bias.copy_(self.proposal_class_head.bias[1:2])

    @property
    def flow_sampling_steps(self):
        return self.sampling_timesteps

    @flow_sampling_steps.setter
    def flow_sampling_steps(self, value):
        if not 1 <= value <= self.num_timesteps:
            raise ValueError('Sampling steps must fit the diffusion schedule')
        self.sampling_timesteps = int(value)

    def select_references(self, candidates):
        references = super().select_references(candidates)
        references['pred_logits'] = candidates['pred_logits'].gather(
            1, references['indices'][..., None].expand(-1, -1, 2))
        references['existence_logits'] = self.slot_confidence_head(references['features']).squeeze(-1)
        return references

    def denoise(self, noisy_latents, timesteps, encoded_image, *, references, image_wh):
        batch_size = len(noisy_latents)
        if timesteps.shape != (batch_size,) or timesteps.dtype != torch.long:
            raise ValueError('Timesteps must have long shape [B]')
        if ((timesteps < 0) | (timesteps >= self.num_timesteps)).any():
            raise ValueError('Timestep outside diffusion schedule')
        if self.prediction_type == 'v':
            sigma = self.sqrt_recipm1_alphas_cumprod[timesteps].to(noisy_latents)
            velocity = self._predict_head(noisy_latents, self._sigma_time(sigma), encoded_image,
                references=references, image_wh=image_wh)
            output = self._decode_output(clean_from_velocity(noisy_latents, velocity, sigma), references, image_wh)
            return dict(output, pred_velocity=velocity)
        predicted = self._predict_head(noisy_latents, timesteps, encoded_image,
            references=references, image_wh=image_wh)
        return self._decode_output(predicted, references, image_wh)

    def _sigma_time(self, sigma):
        return sigma.atan()*(2/math.pi)*(self.num_timesteps-1)

    def denoise_at_sigma(self, noisy_additive_latents, sigma, encoded_image, *, references, image_wh):
        alpha, _ = noise_coefficients(sigma, noisy_additive_latents)
        state = alpha*noisy_additive_latents
        sigma = torch.as_tensor(sigma, device=state.device, dtype=state.dtype)
        predicted = self._predict_head(state, self._sigma_time(sigma), encoded_image,
            references=references, image_wh=image_wh)
        if self.prediction_type == 'v':
            output = self._decode_output(clean_from_velocity(state, predicted, sigma), references, image_wh)
            return dict(output, pred_velocity=predicted)
        return self._decode_output(predicted, references, image_wh)

    def _predict_head(self, noisy_latents, timesteps, encoded_image, *, references, image_wh):
        batch_size = encoded_image[0].shape[0]
        if noisy_latents.shape != (batch_size, self.num_proposals, self.latent_dim):
            raise ValueError('Noisy latent shape disagrees with proposal/codec dimensions')
        if image_wh.shape != (batch_size, 2) or not torch.isfinite(image_wh).all() or (image_wh <= 0).any():
            raise ValueError('Expected finite positive unpadded image_wh shape [B,2]')
        condition = references['features']+self.reference_box_embedding(references['boxes'])
        state = self.latent_input(noisy_latents)
        time_features = self.time_embedding(timesteps.to(state.dtype))
        for layer in self.denoiser_layers:
            state = layer(state, time_features, encoded_image, reference_features=condition)
        return self.latent_output(state)

    def _decode_output(self, predicted, references, image_wh):
        sizes = image_wh[:, None].expand(-1, self.num_proposals, -1).to(predicted)
        geometry = self.codec.decode_latents(predicted, sizes)
        points = geometry['keypoints_px']/sizes[..., None, :]
        existence = references['existence_logits']
        return dict(geometry, pred_latents=predicted, existence_logits=existence,
            pred_logits=torch.stack((torch.full_like(existence, -20.), existence), -1),
            pred_boxes=geometry['boxes_cxcywh_px']/sizes.repeat(1, 1, 2), joint_xy=points,
            pred_keypoints=torch.cat((points.flatten(-2), torch.ones_like(points[..., 0])), -1))

    def forward(self, images, noisy_latents, timesteps, *, image_wh, padding_mask=None):
        context = self.encode_context(images, padding_mask)
        references = self.select_references(self.propose(context))
        return self.denoise(noisy_latents, timesteps, context['image'], references=references, image_wh=image_wh)

    def predict_image_context(self, encoded_image, *, references, image_wh):
        if self.prediction_type != 'x0':
            raise ValueError('Image alignment requires an absolute clean-latent head')
        memory = encoded_image[0]
        state = memory.new_zeros(len(memory), self.num_proposals, self.latent_dim)
        times = torch.zeros(len(memory), device=memory.device, dtype=torch.long)
        predicted = self._predict_head(state, times, encoded_image, references=references, image_wh=image_wh)
        return self._decode_output(predicted, references, image_wh)

    def predict_image(self, images, *, image_wh, padding_mask=None):
        context = self.encode_context(images, padding_mask)
        references = self.select_references(self.propose(context))
        return self.predict_image_context(context['image'], references=references, image_wh=image_wh)

    def q_sample(self, x_start, timesteps, noise=None):
        return super().q_sample(x_start, timesteps, noise).to(x_start)

    @torch.no_grad()
    def sample_keypoints(self, images, *, image_wh, padding_mask=None, generator=None,
                         collect_step_diagnostics=False):
        context = self.encode_context(images, padding_mask)
        candidates = self.propose(context)
        references = self.select_references(candidates)
        state = torch.randn(len(images), self.num_proposals, self.latent_dim,
                            device=images.device, dtype=images.dtype, generator=generator)
        diagnostics = []
        for time, next_time in self.ddim_timestep_pairs():
            timesteps = torch.full((len(images),), time, dtype=torch.long, device=images.device)
            output = self.denoise(state, timesteps, context['image'], references=references, image_wh=image_wh)
            noise = self.predict_noise_from_start(state, timesteps, output['pred_latents']).to(state)
            state = self.ddim_step(output['pred_latents'], noise, time, next_time, noise=torch.zeros_like(state))
            if collect_step_diagnostics:
                diagnostics.append(dict(time=time, next_time=next_time, latent_rms=float(state.square().mean().sqrt())))
        return dict(output, latent_state=state, latent_step_diagnostics=diagnostics,
            reference_boxes=references['boxes'], reference_indices=references['indices'],
            proposal_boxes=candidates['pred_boxes'], proposal_logits=candidates['pred_logits'])

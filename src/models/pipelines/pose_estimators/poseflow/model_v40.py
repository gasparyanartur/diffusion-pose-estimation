import torch
from torch import nn

from ..diffusionregpose.model import DiffusionRegPoseModel, gen_encoder_output_proposals
from ..diffusionregpose.transformer import (
    DeformableTransformerEncoder, DeformableTransformerEncoderLayer, MLP,
)
from .latent_v40 import gather_references
from .model_v26 import LatentImageBlock
from .model_v35 import PoseFlowV35Model


class ReferenceLatentImageBlock(LatentImageBlock):
    def forward(self, state, time_features, encoded_image, *, reference_features):
        memory, position, padding_mask = encoded_image
        modulation = self.modulation(time_features[:, None]+reference_features).chunk(6, -1)
        def conditioned(index):
            scale, shift = modulation[2*index:2*index+2]
            return self.norms[index](state)*(1+scale)+shift
        query = conditioned(0)
        state = state+self.self_attention(query, query, query, need_weights=False)[0]
        query = conditioned(1)
        state = state+self.image_attention(query, memory+position, memory,
            key_padding_mask=padding_mask, need_weights=False)[0]
        return state+self.feedforward(conditioned(2))


def initialize_reference_frontend(model, *, reference_encoder_layers, feedforward_dim, num_heads):
    layer = DeformableTransformerEncoderLayer(d_model=model.hidden_dim,
        d_ffn=feedforward_dim, dropout=0.,
        num_levels=model.num_feature_levels, num_heads=num_heads, num_points=4)
    model.proposal_encoder = DeformableTransformerEncoder(layer, reference_encoder_layers)
    model.proposal_output = nn.Linear(model.hidden_dim, model.hidden_dim)
    model.proposal_output_norm = nn.LayerNorm(model.hidden_dim)
    model.proposal_box_head = MLP(model.hidden_dim, model.hidden_dim, 4, 3)
    model.proposal_class_head = nn.Linear(model.hidden_dim, 2)
    model.reference_box_embedding = nn.Sequential(nn.Linear(4, model.hidden_dim), nn.GELU(),
                                                nn.Linear(model.hidden_dim, model.hidden_dim))


class ReferenceProposalMixin:
    @staticmethod
    def pretrained_source_name(name, encoder_source_layers):
        if name.startswith(('backbone.', 'input_proj.')):
            return name
        if name == 'level_embed':
            return 'transformer.level_embed'
        if name.startswith('proposal_encoder.layers.'):
            parts = name.split('.')
            parts[2] = str(encoder_source_layers[int(parts[2])]-1)
            return 'transformer.encoder.'+'.'.join(parts[1:])
        for target, source in (('proposal_output.', 'enc_output.'),
                               ('proposal_output_norm.', 'enc_output_norm.'),
                               ('proposal_box_head.', 'enc_out_bbox_embed.'),
                               ('proposal_class_head.', 'enc_out_class_embed.')):
            if name.startswith(target):
                return 'transformer.'+source+name[len(target):]
        return None

    def encode_context(self, images, padding_mask=None):
        if padding_mask is None:
            padding_mask = torch.zeros(images.shape[0], *images.shape[2:], dtype=torch.bool, device=images.device)
        inputs = DiffusionRegPoseModel.prepare_transformer_inputs(self, images, padding_mask)
        if inputs.padding_mask.all(-1).any():
            raise ValueError('Image requires valid feature tokens')
        offsets = self.level_embed.repeat_interleave(inputs.spatial_shapes.prod(-1), dim=0)[None]
        return dict(image=(inputs.memory, inputs.position+offsets, inputs.padding_mask), inputs=inputs)

    def propose(self, context):
        memory, position, padding_mask = context['image']
        inputs = context['inputs']
        memory = self.proposal_encoder(memory, position, inputs.spatial_shapes,
            inputs.level_start_index, inputs.valid_ratios, padding_mask)
        memory, anchors = gen_encoder_output_proposals(memory, padding_mask, inputs.spatial_shapes)
        features = self.proposal_output_norm(self.proposal_output(memory))
        logits = self.proposal_class_head(features)
        boxes = (self.proposal_box_head(features)+anchors).sigmoid()
        valid = torch.isfinite(anchors).all(-1)
        ranking = logits.amax(-1).masked_fill(~valid, -torch.inf).argsort(dim=-1, descending=True, stable=True)
        indices = []
        for image in range(len(boxes)):
            available = ranking[image, valid[image, ranking[image]]]
            if len(available) < self.reference_candidates:
                raise ValueError('Insufficient valid image positions for reference candidate budget')
            indices.append(available[:self.reference_candidates])
        indices = torch.stack(indices)
        features = features.gather(1, indices[..., None].expand(-1, -1, self.hidden_dim))
        boxes = boxes.gather(1, indices[..., None].expand(-1, -1, 4))
        logits = logits.gather(1, indices[..., None].expand(-1, -1, 2))
        return dict(features=features, pred_boxes=boxes, pred_logits=logits,
                    existence_logits=logits[..., 1], encoder_indices=indices)

    def select_references(self, candidates):
        indices = candidates['pred_logits'].amax(-1).argsort(dim=-1, descending=True, stable=True)[:, :self.num_proposals]
        return gather_references(candidates, indices)


class PoseFlowV40Model(ReferenceProposalMixin, PoseFlowV35Model):
    def __init__(self, *, reference_candidates=200, reference_encoder_layers=3, **kwargs):
        super().__init__(**kwargs)
        self.reference_candidates = reference_candidates
        if reference_candidates < self.num_proposals:
            raise ValueError('Reference candidate budget must cover latent slots')
        with torch.random.fork_rng(devices=[]):
            initialize_reference_frontend(self, reference_encoder_layers=reference_encoder_layers,
                feedforward_dim=kwargs.get('feedforward_dim', 2048), num_heads=kwargs.get('num_heads', 8))
            conditioned_layers = nn.ModuleList()
            for previous in self.denoiser_layers:
                block = ReferenceLatentImageBlock(self.hidden_dim, kwargs.get('num_heads', 8),
                                                 kwargs.get('feedforward_dim', 2048))
                block.load_state_dict(previous.state_dict(), strict=True)
                conditioned_layers.append(block)
            self.denoiser_layers = conditioned_layers

    def denoise(self, noisy_latents, timesteps, encoded_image, *, references):
        expected = (encoded_image[0].shape[0], self.num_proposals, self.hidden_dim)
        if noisy_latents.shape != expected or timesteps.shape != (expected[0],):
            raise ValueError('Latent/timestep shape disagrees with fixed proposal budget')
        if ((timesteps < 0) | (timesteps >= self.num_timesteps)).any():
            raise ValueError('Timestep outside diffusion schedule')
        if references['features'].shape != expected or references['boxes'].shape != (*expected[:2], 4):
            raise ValueError('Reference shape disagrees with fixed proposal budget')
        condition = references['features']+self.reference_box_embedding(references['boxes'])
        time_features = self.time_embedding(timesteps.to(noisy_latents.dtype))
        state = noisy_latents
        for layer in self.denoiser_layers:
            state = layer(state, time_features, encoded_image, reference_features=condition)
        return self.readout(self.denoiser_projection(state))

    def training_outputs(self, images, input_pixels, *, timesteps, noise, references,
                         padding_mask=None, context=None, latent_target=None, detach_corruption_source=True):
        context = self.encode_context(images, padding_mask) if context is None else context
        clean = self.encode_clean(input_pixels, context['image'])
        source = clean['pred_latents'] if latent_target is None else latent_target
        if source.shape != clean['pred_latents'].shape:
            raise ValueError('Latent target shape disagrees with clean codes')
        source = source.detach() if detach_corruption_source else source
        noisy = self.q_sample(source, timesteps, noise)
        denoised = self.denoise(noisy, timesteps, context['image'], references=references)
        return clean, denoised, noisy, timesteps

    def forward(self, images, noisy_latents, timesteps, padding_mask=None):
        context = self.encode_context(images, padding_mask)
        references = self.select_references(self.propose(context))
        return self.denoise(noisy_latents, timesteps, context['image'], references=references)

    @torch.no_grad()
    def sample_keypoints(self, images, *, padding_mask=None, generator=None, num_proposals=None,
                         collect_step_diagnostics=False):
        if num_proposals is not None and num_proposals != self.num_proposals:
            raise ValueError('V40 preserves its fixed latent slot budget')
        context = self.encode_context(images, padding_mask)
        candidates = self.propose(context)
        references = self.select_references(candidates)
        state = torch.randn(len(images), self.num_proposals, self.hidden_dim,
            device=images.device, dtype=images.dtype, generator=generator)
        diagnostics = []
        for time, next_time in self.ddim_timestep_pairs():
            times = torch.full((len(images),), time, device=images.device, dtype=torch.long)
            output = self.denoise(state, times, context['image'], references=references)
            noise = self.predict_noise_from_start(state, times, output['pred_latents']).to(state.dtype)
            state = self.ddim_step(output['pred_latents'], noise, time, next_time, noise=torch.zeros_like(state))
            if collect_step_diagnostics:
                diagnostics.append(dict(time=time, next_time=next_time, latent_rms=state.square().mean().sqrt().item()))
        return dict(output, latent_state=state, latent_step_diagnostics=diagnostics,
            reference_boxes=references['boxes'], proposal_boxes=candidates['pred_boxes'],
            proposal_logits=candidates['pred_logits'])

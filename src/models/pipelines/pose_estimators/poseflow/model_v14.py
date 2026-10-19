from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..diffusionregpose.model import DiffusionRegPoseModel
from ..diffusionregpose.transformer import MLP, MultiScaleDeformableAttention
from .latent_v12 import ddim_update
from .latent_v14 import V14StateCodec
from .model_v12 import PoseFlowV12Model


def tensor_rms(value: Tensor) -> Tensor:
    return value.detach().float().square().mean().sqrt()


@dataclass
class V14EncodedImage:
    memory: Tensor
    padding_mask: Tensor
    spatial_shapes: Tensor
    level_start_index: Tensor
    valid_ratios: Tensor
    position: Tensor | None = None


class V14AttentionUpdate(nn.Module):
    def __init__(self, width: int, heads: int, feedforward: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.attention_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, feedforward), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(feedforward, width))
        self.ffn_norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)
        self.message_transform = nn.Identity()

    def forward(self, queries: Tensor, context: Tensor, *, collect_weights: bool = False):
        update, weights = self.attention(queries, context, context, need_weights=collect_weights)
        update = self.message_transform(update)
        queries = self.attention_norm(queries + self.dropout(update))
        return self.ffn_norm(queries + self.dropout(self.ffn(queries))), weights


class V14EvidenceExtractor(nn.Module):
    def __init__(self, width: int, heads: int, levels: int, points: int, joints: int):
        super().__init__()
        self.image_attention = MultiScaleDeformableAttention(width, levels, heads, points)
        self.roles = nn.Embedding(joints + 1, width)
        self.box_position = MLP(4, width, width, 2)
        self.joint_position = MLP(2, width, width, 2)
        self.norm = nn.LayerNorm(width)

    def forward(self, representations: Tensor, geometry: dict, image: V14EncodedImage) -> Tensor:
        # B = batch, N = people, J = joints, D = width; representations <B,N,J+1,D>.
        batch, groups, roles, width = representations.shape
        boxes = geometry['pred_boxes']
        joints = geometry['pred_keypoints'][..., :2*(roles-1)].reshape(batch, groups, roles-1, 2)
        position = torch.cat((self.box_position(boxes)[:, :, None], self.joint_position(joints)), 2)
        queries = representations + position + self.roles.weight[None, None]
        ratios = image.valid_ratios[:, None]
        box_references = boxes[:, :, None] * torch.cat((ratios, ratios), -1)
        joint_references = joints.flatten(1, 2)[:, :, None] * ratios
        arguments = (image.memory, image.spatial_shapes, image.level_start_index, image.padding_mask)
        box_evidence = self.image_attention(queries[:, :, 0], box_references, *arguments)
        joint_evidence = self.image_attention(queries[:, :, 1:].flatten(1, 2), joint_references, *arguments)
        sampled = torch.cat((box_evidence[:, :, None],
                             joint_evidence.reshape(batch, groups, roles-1, width)), 2)
        return self.norm(queries + sampled)


class PoseFlowV14DecoderLayer(nn.Module):
    def __init__(self, width: int, heads: int, feedforward: int, dropout: float):
        super().__init__()
        self.aggregate = V14AttentionUpdate(width, heads, feedforward, dropout)
        self.state_update = V14AttentionUpdate(width, heads, feedforward, dropout)

    def aggregate_evidence(self, evidence: Tensor, global_tokens: Tensor) -> Tensor:
        batch, groups, roles, width = evidence.shape
        grouped = torch.cat((evidence, global_tokens[:, :, None]), 2).reshape(batch*groups, roles+1, width)
        aggregated, _ = self.aggregate(grouped, grouped)
        return aggregated.reshape(batch, groups, roles+1, width)[:, :, -1]

    def update_state(self, representations: Tensor, hypotheses: Tensor) -> Tensor:
        updated, _ = self.state_update(representations.flatten(1, 2), hypotheses)
        return updated.reshape_as(representations)


class PoseFlowV14Model(PoseFlowV12Model):
    @property
    def num_state_tokens(self):
        return self.codec.roles.num_embeddings

    def __init__(self, image_model: DiffusionRegPoseModel, *, sampling_steps: int = 3):
        nn.Module.__init__(self)
        decoder = image_model.transformer.decoder
        original = decoder.layers[0]
        self.num_proposals = self.num_queries = image_model.num_proposals
        self.num_body_points = decoder.num_body_points
        self.num_feature_levels = image_model.num_feature_levels
        self.hidden_dim = decoder.d_model
        self.num_hypotheses = 50
        self.backbone = image_model.backbone
        self.input_proj = image_model.input_proj
        self.encoder = image_model.transformer.encoder
        self.level_embed = image_model.transformer.level_embed
        self.time_embedding = nn.Sequential(image_model.transformer.time_mlp, image_model.transformer.block_time_mlp)
        width, heads = self.hidden_dim, original.self_attn.num_heads
        feedforward, dropout = original.linear1.out_features, original.dropout1.p
        self.codec = V14StateCodec(width=width, joints=self.num_body_points)
        self.input_projection = nn.Linear(width, width)
        self.state_roles = nn.Embedding(self.num_body_points + 1, width)
        self.global_token = nn.Parameter(torch.randn(1, 1, width) * .02)
        self.hypothesis_slots = nn.Parameter(torch.randn(1, self.num_hypotheses, width) * .02)
        self.evidence_extractor = V14EvidenceExtractor(width, heads, self.num_feature_levels,
                                                      original.cross_attn.num_points, self.num_body_points)
        self.hypothesis_update = V14AttentionUpdate(width, heads, feedforward, dropout)
        self.layers = nn.ModuleList([PoseFlowV14DecoderLayer(width, heads, feedforward, dropout)
                                     for _ in decoder.layers])
        self.latent_head = nn.Linear(width, width)
        self.flow_sampling_steps = sampling_steps
        self.flow_renewal = self.flow_ensemble = False
        self.flow_renewal_score_threshold = .5
        image_keys = [key for key in self.state_dict()
                      if key.startswith(('backbone.', 'input_proj.', 'encoder.', 'level_embed.'))]
        self.initialization_report = {
            'image_components': ['backbone', 'input_proj', 'encoder', 'level_embed'],
            'retained_image_tensors': len(image_keys),
            'retained_image_elements': sum(self.state_dict()[key].numel() for key in image_keys),
            'compatible_component_copy_coverage': 1.,
            'decoder_transferred': [],
            'fresh_components': ['codec', 'evidence_extractor', 'hypothesis_update', 'layers',
                                 'latent_head', 'input_projection', 'state_roles', 'global_token',
                                 'hypothesis_slots'],
        }

    def encode_image(self, images: Tensor, padding_mask: Tensor | None = None) -> V14EncodedImage:
        if padding_mask is None:
            padding_mask = torch.zeros(images.shape[0], *images.shape[-2:], device=images.device, dtype=torch.bool)
        inputs = DiffusionRegPoseModel.prepare_transformer_inputs(self, images, padding_mask)
        offsets = self.level_embed.repeat_interleave(inputs.spatial_shapes.prod(-1), dim=0)[None]
        position = inputs.position + offsets
        memory = self.encoder(inputs.memory, position, inputs.spatial_shapes,
                              inputs.level_start_index, inputs.valid_ratios, inputs.padding_mask)
        return V14EncodedImage(memory, inputs.padding_mask, inputs.spatial_shapes,
                               inputs.level_start_index, inputs.valid_ratios, position)

    def denoise(self, noisy_latents: Tensor, timesteps: Tensor, encoded_image: V14EncodedImage,
                *, hypothesis_memory: Tensor | None = None, collect_diagnostics: bool = False) -> dict:
        expected = (noisy_latents.shape[0], self.num_proposals, self.num_state_tokens, self.hidden_dim)
        if noisy_latents.shape != expected or timesteps.shape != (expected[0],):
            raise ValueError('V14 latent/timestep shapes disagree with configuration')
        if not torch.isfinite(noisy_latents).all() or not torch.isfinite(timesteps).all() or ((timesteps < 0) | (timesteps > 1)).any():
            raise ValueError('V14 requires finite latents and times in [0,1]')
        if hypothesis_memory is None:
            hypothesis_memory = self.hypothesis_slots.expand(expected[0], -1, -1)
        if (hypothesis_memory.shape != (expected[0], self.num_hypotheses, self.hidden_dim)
                or hypothesis_memory.device != noisy_latents.device or hypothesis_memory.dtype != noisy_latents.dtype
                or not torch.isfinite(hypothesis_memory).all()):
            raise ValueError('V14 hypothesis memory shape/device/dtype/values disagree with state')
        time_features = self.time_embedding(timesteps)[:, None, None]
        representations = self.input_projection(noisy_latents) + self.state_roles.weight[None, None] + time_features
        global_tokens = self.global_token.expand(expected[0], self.num_proposals, -1)
        geometry = self.codec.decode(noisy_latents)
        predictions, diagnostics = [], []
        for index, layer in enumerate(self.layers):
            evidence = self.evidence_extractor(representations, geometry, encoded_image)
            previous_global, previous_memory = global_tokens, hypothesis_memory
            global_tokens = layer.aggregate_evidence(evidence, global_tokens)
            hypothesis_memory, attention = self.hypothesis_update(
                hypothesis_memory, global_tokens, collect_weights=collect_diagnostics)
            representations = layer.update_state(representations, hypothesis_memory)
            clean = self.latent_head(representations)
            geometry = self.codec.decode(clean)
            predictions.append({'pred_latents': clean, **geometry})
            if collect_diagnostics:
                diagnostics.append(self._iteration_diagnostics(index, clean, global_tokens,
                    previous_global, hypothesis_memory, previous_memory, attention, geometry))
        return {**predictions[-1], 'aux_outputs': predictions[:-1],
                'hypothesis_memory': hypothesis_memory, 'latent_diagnostics': diagnostics}

    def _iteration_diagnostics(self, index, clean, global_tokens, previous_global,
                               hypotheses, previous_memory, attention, geometry):
        probabilities = attention.detach().float()
        row = {'layer': index, 'global_rms': tensor_rms(global_tokens),
               'global_update_rms': tensor_rms(global_tokens-previous_global),
               'hypothesis_rms': tensor_rms(hypotheses),
               'hypothesis_update_rms': tensor_rms(hypotheses-previous_memory),
               'hypothesis_slot_variance': hypotheses.detach().float().var(dim=1, unbiased=False).mean(),
               'hypothesis_attention_entropy': -(probabilities * probabilities.clamp_min(1e-8).log()).sum(-1).mean(),
               'existence_count': (geometry['pred_logits'].sigmoid() >= .5).sum().detach().float()}
        for name, subset in getattr(self.codec, 'latent_role_slices',
                [('box', slice(0, 1)), ('joints', slice(1, None))]):
            values = clean[:, :, subset].detach().float()
            row[name+'_rms'] = tensor_rms(values)
            row[name+'_variance'] = values.var(unbiased=False)
        return row

    def forward(self, images, noisy_latents, timesteps, padding_mask=None, *,
                hypothesis_memory=None, collect_diagnostics=False):
        return self.denoise(noisy_latents, timesteps, self.encode_image(images, padding_mask),
                            hypothesis_memory=hypothesis_memory, collect_diagnostics=collect_diagnostics)

    @torch.no_grad()
    def sample_keypoints(self, images, *, padding_mask=None, num_proposals=None, generator=None,
                         collect_step_diagnostics=False, initial_latents=None, start_time=1.,
                         sampling_steps=None, encoded_image=None, persist_hypotheses=True):
        if num_proposals is not None and num_proposals != self.num_proposals:
            raise ValueError('V14 preserves its configured group count')
        steps = self.flow_sampling_steps if sampling_steps is None else sampling_steps
        if not isinstance(steps, int) or steps < 1 or not 0 < start_time <= 1:
            raise ValueError('V14 requires positive steps and a start time in (0,1]')
        image_memory = self.encode_image(images, padding_mask) if encoded_image is None else encoded_image
        shape = (images.shape[0], self.num_proposals, self.num_state_tokens, self.hidden_dim)
        if initial_latents is None:
            if start_time != 1:
                raise ValueError('Partial-noise sampling requires explicit initial latents')
            state = torch.randn(shape, device=images.device, dtype=images.dtype, generator=generator)
        else:
            if (initial_latents.shape != shape or not torch.isfinite(initial_latents).all()
                    or initial_latents.device != images.device or initial_latents.dtype != images.dtype):
                raise ValueError('V14 initial latents have invalid shape/device/dtype/values')
            state = initial_latents.clone()
        hypotheses, diagnostics = None, []
        for step in range(steps):
            times = torch.full((images.shape[0],), start_time*(1-step/steps), device=images.device, dtype=images.dtype)
            next_times = torch.full_like(times, start_time*(1-(step+1)/steps))
            output = self.denoise(state, times, image_memory, hypothesis_memory=hypotheses,
                                  collect_diagnostics=collect_step_diagnostics)
            next_state = ddim_update(state, output['pred_latents'], times, next_times)
            hypotheses = output['hypothesis_memory'] if persist_hypotheses else None
            if collect_step_diagnostics:
                stages = output['aux_outputs'] + [output]
                diagnostics.append({'time': times[0].item(), 'latent_state': next_state.clone(),
                    'update_rms': (next_state-state).float().square().mean().sqrt(),
                    'layers': output['latent_diagnostics'],
                    'predictions': [{name: stage[name] for name in
                        ('pred_logits', 'pred_boxes', 'pred_keypoints')} for stage in stages]})
            state = next_state
        return {**output, 'latent_state': state, 'latent_step_diagnostics': diagnostics}

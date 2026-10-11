"""Fixed-person latent diffusion with readout-only pose coordinates."""

from copy import deepcopy

import torch
from torch import Tensor, nn

from ..diffusionregpose.model import DiffusionRegPoseModel
from ..diffusionregpose.evaluation import postprocess_crowdpose
from .latent_v12 import V12StateCodec, ddim_update


class PoseFlowV12DecoderLayer(nn.Module):
    def __init__(self, original):
        super().__init__()
        width = original.self_attn.embed_dim
        heads = original.self_attn.num_heads
        dropout = original.dropout1.p
        self.gather_attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.distribute_attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.gather_attention.load_state_dict(original.self_attn.state_dict())
        self.distribute_attention.load_state_dict(original.self_attn.state_dict())
        self.image_attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.person_attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.gather_norm = deepcopy(original.norm2)
        self.distribute_norm = deepcopy(original.norm2)
        self.image_norm = nn.LayerNorm(width)
        self.person_norm = nn.LayerNorm(width)
        self.ffn_norm = deepcopy(original.norm3)
        self.linear1, self.linear2 = deepcopy(original.linear1), deepcopy(original.linear2)
        self.dropout = nn.Dropout(dropout)
        self.latent_head = nn.Linear(width, width)

    def forward(self, tokens: Tensor, memory: Tensor, position: Tensor,
                padding_mask: Tensor) -> Tensor:
        batch, groups, count, width = tokens.shape
        grouped = tokens.reshape(batch * groups, count, width)
        update = self.gather_attention(grouped, grouped, grouped, need_weights=False)[0]
        grouped = self.gather_norm(grouped + self.dropout(update))
        tokens = grouped.reshape_as(tokens)
        helper = tokens[:, :, -1]
        update = self.image_attention(helper, memory + position, memory,
                                      key_padding_mask=padding_mask, need_weights=False)[0]
        helper = self.image_norm(helper + self.dropout(update))
        update = self.person_attention(helper, helper, helper, need_weights=False)[0]
        helper = self.person_norm(helper + self.dropout(update))
        grouped = torch.cat((tokens[:, :, :-1], helper[:, :, None]), 2).reshape_as(grouped)
        update = self.distribute_attention(grouped, grouped, grouped, need_weights=False)[0]
        grouped = self.distribute_norm(grouped + self.dropout(update))
        update = self.linear2(self.dropout(self.linear1(grouped).relu()))
        return self.ffn_norm(grouped + self.dropout(update)).reshape_as(tokens)


class PoseFlowV12Model(nn.Module):
    def __init__(self, image_model: DiffusionRegPoseModel, *, sampling_steps: int = 3):
        super().__init__()
        self.num_proposals = image_model.num_proposals
        self.num_queries = self.num_proposals
        self.num_body_points = image_model.transformer.decoder.num_body_points
        self.num_feature_levels = image_model.num_feature_levels
        self.hidden_dim = image_model.transformer.decoder.d_model
        self.backbone = image_model.backbone
        self.input_proj = image_model.input_proj
        self.encoder = image_model.transformer.encoder
        self.level_embed = image_model.transformer.level_embed
        self.time_embedding = nn.Sequential(image_model.transformer.time_mlp,
                                             image_model.transformer.block_time_mlp)
        self.codec = V12StateCodec(width=self.hidden_dim, joints=self.num_body_points)
        self.input_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.state_roles = nn.Embedding(self.num_body_points + 3, self.hidden_dim)
        self.global_token = nn.Parameter(torch.randn(1, 1, 1, self.hidden_dim) * .02)
        self.layers = nn.ModuleList([PoseFlowV12DecoderLayer(layer)
                                     for layer in image_model.transformer.decoder.layers])
        self.flow_sampling_steps = sampling_steps
        self.flow_renewal = False
        self.flow_ensemble = False
        self.flow_renewal_score_threshold = 0.5
        self.initialization_report = {
            'image_components': ['backbone', 'input_proj', 'encoder', 'level_embed'],
            'decoder_transferred': ['gather_attention', 'distribute_attention', 'linear1', 'linear2',
                                    'gather_norm', 'distribute_norm', 'ffn_norm'],
            'fresh_components': ['codec', 'image_attention', 'person_attention', 'latent_head',
                                 'input_projection', 'state_roles', 'global_token'],
        }
        state = self.state_dict()
        image_keys = [key for key in state if key.startswith(('backbone.', 'input_proj.', 'encoder.', 'level_embed'))]
        decoder_keys = [key for key in state if key.startswith('layers.') and any(
            f'.{component}.' in key for component in self.initialization_report['decoder_transferred']
        )]
        self.initialization_report.update(
            retained_image_tensors=len(image_keys), copied_decoder_tensors=len(decoder_keys),
            total_tensors=len(state), compatible_component_copy_coverage=1.0,
        )

    def encode_image(self, images: Tensor, padding_mask: Tensor | None = None):
        if padding_mask is None:
            padding_mask = torch.zeros(images.shape[0], *images.shape[-2:], device=images.device, dtype=torch.bool)
        inputs = DiffusionRegPoseModel.prepare_transformer_inputs(self, images, padding_mask)
        offsets = self.level_embed.repeat_interleave(inputs.spatial_shapes.prod(-1), dim=0)[None]
        position = inputs.position + offsets
        memory = self.encoder(inputs.memory, position, inputs.spatial_shapes,
                              inputs.level_start_index, inputs.valid_ratios, inputs.padding_mask)
        return memory, position, inputs.padding_mask

    def denoise(self, noisy_latents: Tensor, timesteps: Tensor, encoded_image,
                *, collect_diagnostics: bool = False) -> dict:
        expected = (noisy_latents.shape[0], self.num_proposals, self.num_body_points + 3, self.hidden_dim)
        if noisy_latents.shape != expected or timesteps.shape != (noisy_latents.shape[0],):
            raise ValueError('V12 latent/timestep shapes disagree with the fixed group configuration')
        if not torch.isfinite(timesteps).all() or ((timesteps < 0) | (timesteps > 1)).any():
            raise ValueError('V12 timesteps must be in [0,1]')
        time_features = self.time_embedding(timesteps)[:, None, None]
        state = self.input_projection(noisy_latents) + self.state_roles.weight[None, None] + time_features
        helper = self.global_token.expand(state.shape[0], self.num_proposals, 1, -1) + time_features
        tokens = torch.cat((state, helper), 2)
        predictions, diagnostics = [], []
        for index, layer in enumerate(self.layers):
            tokens = layer(tokens, *encoded_image)
            clean = layer.latent_head(tokens[:, :, :-1])
            prediction = {'pred_latents': clean, **self.codec.decode(clean)}
            predictions.append(prediction)
            if collect_diagnostics:
                row = {'layer': index}
                for name, subset in [('existence', slice(0,1)), ('center', slice(1,2)),
                                     ('size', slice(2,3)), ('joints', slice(3,None))]:
                    values = clean[:, :, subset].detach().float()
                    row[name + '_rms'] = values.square().mean().sqrt()
                    row[name + '_variance'] = values.var(unbiased=False)
                row['existence_count'] = (prediction['pred_logits'].sigmoid() >= .5).sum().detach()
                diagnostics.append(row)
        return {**predictions[-1], 'aux_outputs': predictions[:-1], 'latent_diagnostics': diagnostics}

    def forward(self, images: Tensor, noisy_latents: Tensor, timesteps: Tensor,
                padding_mask: Tensor | None = None, *, collect_diagnostics: bool = False) -> dict:
        return self.denoise(noisy_latents, timesteps, self.encode_image(images, padding_mask),
                            collect_diagnostics=collect_diagnostics)

    @torch.no_grad()
    def sample_keypoints(self, images: Tensor, *, padding_mask=None, num_proposals=None,
                         generator=None, collect_step_diagnostics=False) -> dict:
        if num_proposals is not None and num_proposals != self.num_proposals:
            raise ValueError('V12 sampling preserves its configured group count')
        steps = self.flow_sampling_steps
        if steps < 1:
            raise ValueError('V12 requires positive DDIM sampling steps')
        memory = self.encode_image(images, padding_mask)
        state = torch.randn(images.shape[0], self.num_proposals, self.num_body_points + 3,
                            self.hidden_dim, device=images.device, dtype=images.dtype, generator=generator)
        diagnostics = []
        for step in range(steps):
            times = torch.full((images.shape[0],), 1 - step/steps, device=images.device, dtype=images.dtype)
            next_times = torch.full_like(times, 1 - (step+1)/steps)
            output = self.denoise(state, times, memory, collect_diagnostics=collect_step_diagnostics)
            next_state = ddim_update(state, output['pred_latents'], times, next_times)
            if collect_step_diagnostics:
                stages = output['aux_outputs'] + [output]
                diagnostics.append({'time': times[0].item(), 'latent_state': next_state.clone(),
                    'update_rms': (next_state - state).float().square().mean().sqrt(),
                    'layers': output['latent_diagnostics'],
                    'predictions': [{name: stage[name] for name in
                        ('pred_logits', 'pred_boxes', 'pred_keypoints')} for stage in stages]})
            state = next_state
        return {**output, 'latent_state': state, 'latent_step_diagnostics': diagnostics}

    def postprocess(self, output, target_sizes, *, nms_target_sizes=None):
        results = postprocess_crowdpose(output, target_sizes, num_body_points=self.num_body_points,
                                       nms_target_sizes=nms_target_sizes)
        for result in results:
            result['labels'] = torch.ones_like(result['labels'])
        return results

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ..diffusionregpose.transformer import MultiScaleDeformableAttention
from .model_v50 import PoseFlowV50Model
from .proposals_v50 import gather_identity
from .rows_token_age import PAD, PERSON, PREDICT


def decode_joint_geometry(raw):
    center = raw[..., 28:30].sigmoid()
    joints = raw[..., :28].unflatten(-1, (14, 2))+center[..., None, :]
    sizes = F.softplus(raw[..., 30:])+1e-6
    return torch.cat((joints.flatten(-2), center, sizes), -1)


class DeformablePersonLayer(nn.Module):
    def __init__(self, width, heads, feedforward_dim, levels, points):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)
        self.image_attention = MultiScaleDeformableAttention(width, levels, heads, points)
        self.image_projection = self.image_attention.value_proj
        self.image_attention.value_proj = nn.Identity()
        self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in range(3))
        self.time_modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 2*width))
        self.reference_position = nn.Linear(4, width)
        self.feedforward = nn.Sequential(nn.Linear(width, feedforward_dim), nn.GELU(), nn.Linear(feedforward_dim, width))
        nn.init.normal_(self.time_modulation[-1].weight, std=.01)
        nn.init.zeros_(self.time_modulation[-1].bias)
        # Keep the radial sampling initialization while opening the time-query gradient path.
        nn.init.normal_(self.image_attention.sampling_offsets.weight, std=1e-3)
        nn.init.normal_(self.image_attention.attention_weights.weight, std=1e-3)

    def forward(self, state, references, times, padding_mask, image_values, spatial_shapes, level_start_index, valid_ratios):
        # B images, R independent rows, Q people, D feature channels.
        batch, rows, queries, width = state.shape
        query = self.norms[0](state).reshape(batch*rows, queries, width)
        update = self.self_attention(query, query, query,
            key_padding_mask=padding_mask.reshape(batch*rows, queries), need_weights=False)[0]
        state = state+update.reshape_as(state)
        shift, scale = self.time_modulation(times).chunk(2, -1)
        query = self.norms[1](state)*(1+scale)+shift+self.reference_position(references)
        ratio = torch.cat((valid_ratios, valid_ratios), -1)
        scaled_references = references.reshape(batch, rows*queries, 1, 4)*ratio[:, None]
        update = self.image_attention(query.reshape(batch, rows*queries, width), scaled_references,
            image_values, spatial_shapes, level_start_index)
        state = state+update.reshape_as(state)
        state = state+self.feedforward(self.norms[2](state))
        return state.masked_fill(padding_mask[..., None], 0.)


class PoseFlowV52Model(PoseFlowV50Model):
    def __init__(self, *, native, sampling_points=4, **kwargs):
        super().__init__(native=native, **kwargs)
        del self.person_codec, self.latent_input, self.latent_heads, self.residual_heads
        self.latent_dim = self.hidden_dim
        self.coordinate_input = nn.Sequential(nn.Linear(32, self.hidden_dim), nn.GELU(), nn.Linear(self.hidden_dim, self.hidden_dim))
        self.layers = nn.ModuleList(DeformablePersonLayer(self.hidden_dim, self.num_heads,
            self.feedforward_dim, self.num_feature_levels, sampling_points) for _ in range(2))
        self.geometry_heads = nn.ModuleList(nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(), nn.Linear(self.hidden_dim, 32)) for _ in self.layers)
        self.sampling_points = sampling_points

    def _project_image_context(self, memory, position, padding_mask):
        if padding_mask.all(-1).any():
            raise ValueError('Image requires valid feature tokens')
        values = tuple(layer.image_projection(memory).masked_fill(padding_mask[..., None], 0.) for layer in self.layers)
        return dict(memory=memory, position=position, padding_mask=padding_mask, image_values=values)

    def _time_features(self, sigmas):
        features = super()._time_features(sigmas)
        return features/features.square().mean(-1, keepdim=True).clamp_min(1.).sqrt()

    def _decode_geometry(self, state, references, times, padding_mask, spatial_shapes, level_start_index, valid_ratios, *values):
        coordinates, features = [], []
        for index, layer in enumerate(self.layers):
            state = layer(state, references, times, padding_mask, values[index], spatial_shapes, level_start_index, valid_ratios)
            readout = self.readout_norms[index](state)
            prediction = decode_joint_geometry(self.geometry_heads[index](readout))
            prediction = prediction.masked_fill(padding_mask[..., None], 0.)
            coordinates.append(prediction)
            features.append(readout.masked_fill(padding_mask[..., None], 0.))
            references = prediction[..., 28:].detach()
        return torch.stack(coordinates), torch.stack(features)

    def _validate_rows(self, context, coordinates, sigmas, token_types, padding_mask, indices):
        shape = coordinates.shape[:-1]
        if (coordinates.ndim != 4 or coordinates.shape[-1] != 32 or shape[0] != len(context['memory'])
                or shape[2] > self.rows or any(value.shape != shape for value in (sigmas, token_types, padding_mask, indices))):
            raise ValueError('Invalid V52 row shapes')
        person = token_types == PERSON
        if (not torch.equal(padding_mask, token_types == PAD)
                or not ((token_types == PAD) | person | (token_types == PREDICT)).all()
                or ((token_types == PREDICT).sum(-1) != 1).any()
                or not torch.isfinite(sigmas).all() or ((sigmas < 0) | (sigmas > 1)).any()
                or (person & (sigmas >= 1)).any() or not torch.isfinite(coordinates[~padding_mask]).all()
                or (indices < -1).any() or (indices >= context['features'].shape[1]).any()
                or (person & (indices < 0)).any()):
            raise ValueError('Invalid V52 token types, noise or proposal identities')

    def forward_rows(self, context, noisy_coordinates, noise_sigmas, token_types, padding_mask, proposal_indices):
        self._validate_rows(context, noisy_coordinates, noise_sigmas, token_types, padding_mask, proposal_indices)
        person = token_types == PERSON
        coordinates = (noisy_coordinates.masked_fill(padding_mask[..., None], 0.)/self.scale+1)/2
        relative = coordinates[..., :28].unflatten(-1, (14, 2))-coordinates[..., None, 28:30]
        state = self.coordinate_input(torch.cat((relative.flatten(-2), coordinates[..., 28:]), -1))
        identity, reference = gather_identity(context['features'], context['boxes'], proposal_indices)
        prefix_state = state+identity+self.reference_input(reference)
        prefix = prefix_state.masked_fill(~person[..., None], 0.).sum(-2)/person.sum(-1, keepdim=True).clamp_min(1)
        candidates = torch.arange(context['features'].shape[1], device=state.device)
        used = ((proposal_indices[..., None] == candidates) & person[..., None]).any(-2)
        selection, eos = self.selector(prefix.detach(), context['features'], used)
        chosen = selection.argmax(-1)
        indices = torch.where((token_types == PREDICT) & (proposal_indices < 0), chosen[..., None], proposal_indices)
        identity, reference = gather_identity(context['features'], context['boxes'], indices)
        state = state+identity+self.reference_input(reference)
        state = state+(token_types == PREDICT)[..., None]*self.predict_embedding
        state = state+self.rank_embedding.weight[:state.shape[2]][None, None]
        times = self._time_features(noise_sigmas)
        state = (state+times).masked_fill(padding_mask[..., None], 0.)
        alpha = (1-noise_sigmas.masked_fill(~person, 0.).square()).clamp_min(1e-8).sqrt()
        base = (noisy_coordinates.masked_fill(~person[..., None], 0.)/alpha[..., None]/self.scale+1)/2
        current_box = torch.cat((base[..., 28:30], base[..., 30:].clamp(1e-4, 2.)), -1)
        references = torch.where(person[..., None], current_box, reference).detach()
        inputs = context['inputs']
        args = (state, references, times, padding_mask, inputs.spatial_shapes, inputs.level_start_index,
                inputs.valid_ratios, *context['image_values'])
        if self.training and torch.is_grad_enabled() and self.activation_checkpointing:
            predictions, features = checkpoint(self._decode_geometry, *args, use_reentrant=False, preserve_rng_state=True)
        else:
            predictions, features = self._decode_geometry(*args)
        shape = token_types.shape
        return dict(pred_coordinates=predictions, pred_features=features,
            eos_logits=eos[None, ..., None].expand(len(self.layers), *shape),
            selection_logits=selection, selected_proposals=indices, used_candidates=used)

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ..diffusionregpose.model import DiffusionRegPoseModel
from ..diffusionregpose.transformer import SinusoidalPositionEmbeddings
from .coordinates_v47 import coordinate_geometry
from .rows_v46 import PAD, PERSON, PREDICT, draw_noise, row_alpha_bar
from .sampling_v46 import ddim_transition


class PersonCoordinateCodec(nn.Module):
    def __init__(self, width=256, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(32, width), nn.GELU(), nn.Linear(width, latent_dim),
                                     nn.LayerNorm(latent_dim, elementwise_affine=False))
        self.reader = nn.Sequential(nn.Linear(latent_dim, width), nn.GELU(), nn.Linear(width, width),
                                    nn.GELU(), nn.Linear(width, 32))

    def encode(self, coordinates):
        return self.encoder(coordinates)

    def decode(self, latent):
        coordinates = self.reader(latent)
        return torch.cat((coordinates[..., :30], F.softplus(coordinates[..., 30:])+1e-6), -1)


class CachedRowLayer(nn.Module):
    def __init__(self, width, heads, feedforward_dim):
        super().__init__()
        self.heads = heads
        self.self_attention = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)
        self.image_attention = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)
        self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in range(3))
        self.feedforward = nn.Sequential(nn.Linear(width, feedforward_dim), nn.GELU(),
                                         nn.Linear(feedforward_dim, width))

    def _heads(self, value):
        return value.unflatten(-1, (self.heads, -1)).transpose(1, 2)

    def project_image(self, memory, position):
        _, key_weight, value_weight = self.image_attention.in_proj_weight.chunk(3)
        _, key_bias, value_bias = self.image_attention.in_proj_bias.chunk(3)
        return (self._heads(F.linear(memory+position, key_weight, key_bias)),
                self._heads(F.linear(memory, value_weight, value_bias)))

    def forward(self, state, padding_mask, image_keys, image_values, image_mask):
        # B images, R independent rows, Q token positions, W features.
        batch, rows, queries, width = state.shape
        query = self.norms[0](state).reshape(batch*rows, queries, width)
        update = self.self_attention(query, query, query,
            key_padding_mask=padding_mask.reshape(batch*rows, queries), need_weights=False)[0]
        state = state+update.reshape_as(state)
        query = self.norms[1](state).reshape(batch, rows*queries, width)
        query_weight = self.image_attention.in_proj_weight[:width]
        query_bias = self.image_attention.in_proj_bias[:width]
        query = self._heads(F.linear(query, query_weight, query_bias))
        update = F.scaled_dot_product_attention(query, image_keys, image_values,
            attn_mask=~image_mask[:, None, None, :], dropout_p=0.)
        update = update.transpose(1, 2).flatten(-2)
        state = state+self.image_attention.out_proj(update).reshape_as(state)
        state = state+self.feedforward(self.norms[2](state))
        return state.masked_fill(padding_mask[..., None], 0.)


class PoseFlowV47Model(nn.Module):
    prepare_transformer_inputs = DiffusionRegPoseModel.prepare_transformer_inputs
    encode_features = DiffusionRegPoseModel.encode_features
    _level_position_offsets = DiffusionRegPoseModel._level_position_offsets
    row_input_fields = ('noisy_coordinates', 'row_times', 'token_types', 'padding_mask')

    def __init__(self, *, native, rows=64, capacity=50, hidden_dim=256, num_heads=8,
                 feedforward_dim=2048, codec_width=256, latent_dim=32, scale=6., activation_checkpointing=True):
        super().__init__()
        if not 0 < capacity < rows or scale <= 0:
            raise ValueError('Expected positive coordinate scale and capacity below row count')
        self.rows, self.capacity, self.scale = rows, capacity, scale
        self.hidden_dim, self.num_heads, self.feedforward_dim = hidden_dim, num_heads, feedforward_dim
        self.codec_width, self.latent_dim = codec_width, latent_dim
        self.num_body_points, self.num_feature_levels = 14, native.num_feature_levels
        self.backbone, self.input_proj = native.backbone, native.input_proj
        self.transformer = nn.Module()
        self.transformer.encoder = native.transformer.encoder
        self.transformer.level_embed = native.transformer.level_embed
        self.person_codec = PersonCoordinateCodec(codec_width, latent_dim)
        self.latent_input = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.predict_embedding = nn.Parameter(torch.zeros(hidden_dim))
        self.rank_embedding = nn.Embedding(rows, hidden_dim)
        self.time_embedding = nn.Sequential(SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim*4), nn.GELU(), nn.Linear(hidden_dim*4, hidden_dim))
        self.layers = nn.ModuleList(CachedRowLayer(hidden_dim, num_heads, feedforward_dim) for _ in range(2))
        self.readout_norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(2))
        self.latent_heads = nn.ModuleList(nn.Linear(hidden_dim, latent_dim) for _ in range(2))
        self.eos_heads = nn.ModuleList(nn.Linear(hidden_dim, 1) for _ in range(2))
        self.activation_checkpointing = activation_checkpointing
        self.initialization_report = {}

    def _project_image_context(self, memory, position, padding_mask):
        if padding_mask.all(-1).any():
            raise ValueError('Image requires valid feature tokens')
        projected = [layer.project_image(memory, position) for layer in self.layers]
        return dict(memory=memory, position=position, padding_mask=padding_mask,
                    image_keys=tuple(pair[0] for pair in projected), image_values=tuple(pair[1] for pair in projected))

    def prepare_context(self, images, padding_mask=None):
        if padding_mask is None:
            padding_mask = torch.zeros(len(images), *images.shape[2:], device=images.device, dtype=torch.bool)
        inputs = self.prepare_transformer_inputs(images, padding_mask)
        return self._project_image_context(self.encode_features(inputs),
            inputs.position+self._level_position_offsets(inputs), inputs.padding_mask)

    def _decode_rows(self, state, padding_mask, image_mask, *projected):
        latents, coordinates, eos = [], [], []
        for index, (layer, norm, head, eos_head) in enumerate(zip(
                self.layers, self.readout_norms, self.latent_heads, self.eos_heads)):
            state = layer(state, padding_mask, projected[2*index], projected[2*index+1], image_mask)
            readout = norm(state)
            latent = head(readout)
            latents.append(latent.masked_fill(padding_mask[..., None], 0.))
            coordinates.append(self.person_codec.decode(latent).masked_fill(padding_mask[..., None], 0.))
            eos.append(eos_head(readout).squeeze(-1).masked_fill(padding_mask, 0.))
        return torch.stack(coordinates), torch.stack(latents), torch.stack(eos)

    def forward_rows(self, context, noisy_coordinates, row_times, token_types, padding_mask):
        if noisy_coordinates.ndim != 4 or noisy_coordinates.shape[-1] != 32:
            raise ValueError('Expected coordinate state [B,R,Q,32]')
        batch, rows, queries, _ = noisy_coordinates.shape
        if (batch != len(context['memory']) or queries > self.rows or row_times.shape != (rows,)
                or token_types.shape != (batch, rows, queries) or padding_mask.shape != token_types.shape):
            raise ValueError('Row shapes disagree with model/context')
        if (not torch.isfinite(row_times).all() or ((row_times < 0) | (row_times > 1)).any()
                or not torch.equal(padding_mask, token_types == PAD)
                or not ((token_types == PAD) | (token_types == PERSON) | (token_types == PREDICT)).all()
                or ((token_types == PREDICT).sum(-1) != 1).any()):
            raise ValueError('Invalid row times, padding or token types')
        person = token_types == PERSON
        latent = noisy_coordinates.new_zeros(batch, rows, queries, self.latent_dim)
        if person.any():
            encoded = self.person_codec.encode((noisy_coordinates[person]/self.scale+1)/2)
            latent = latent.masked_scatter(person[..., None].expand_as(latent), encoded)
        state = torch.where(person[..., None], self.latent_input(latent), self.predict_embedding)
        state = state+self.rank_embedding.weight[:queries][None, None]
        state = state+self.time_embedding(row_times*999)[None, :, None]
        state = state.masked_fill(padding_mask[..., None], 0.)
        projected = tuple(value for pair in zip(context['image_keys'], context['image_values']) for value in pair)
        args = (state, padding_mask, context['padding_mask'], *projected)
        if self.training and torch.is_grad_enabled() and self.activation_checkpointing:
            coordinates, latents, eos = checkpoint(self._decode_rows, *args, use_reentrant=False, preserve_rng_state=True)
        else:
            coordinates, latents, eos = self._decode_rows(*args)
        return dict(pred_coordinates=coordinates, pred_latents=latents, eos_logits=eos)

    def _geometry_output(self, coordinates, counts, scores, image_wh):
        sizes = image_wh[:, None].expand(-1, self.capacity, -1).to(coordinates)
        geometry = coordinate_geometry(coordinates, sizes)
        valid = torch.arange(self.capacity, device=coordinates.device)[None] < counts[:, None]
        logits = torch.logit(scores.clamp(1e-8, 1-1e-7)).masked_fill(~valid, -30.)
        points = coordinates[..., :28]
        return dict(geometry, pred_coordinates=coordinates, scores=scores.masked_fill(~valid, 0.),
            existence_logits=logits, pred_logits=torch.stack((torch.full_like(logits, -30.), logits), -1),
            pred_boxes=coordinates[..., 28:],
            pred_keypoints=torch.cat((points, torch.ones_like(coordinates[..., :14])), -1),
            valid_mask=valid, person_counts=counts,
            sequence_indices=torch.arange(self.capacity, device=coordinates.device)[None].expand(len(coordinates), -1).masked_fill(~valid, -1))

    @torch.no_grad()
    def predict(self, context, image_wh, generator=None, collect_trace=False):
        reference = context['memory']
        batch = len(reference)
        if image_wh.shape != (batch, 2) or not torch.isfinite(image_wh).all() or (image_wh <= 0).any():
            raise ValueError('Expected positive finite image sizes [B,2]')
        if isinstance(generator, (list, tuple)) and len(generator) != batch:
            raise ValueError('Per-image RNG list disagrees with batch size')
        clean = reference.new_zeros(batch, self.capacity, 32)
        noisy, scores = clean.clone(), reference.new_zeros(batch, self.capacity)
        counts = torch.zeros(batch, device=reference.device, dtype=torch.long)
        stopped = torch.zeros(batch, device=reference.device, dtype=torch.bool)
        stop_reason, eos_row = counts.clone(), counts.clone()
        times = torch.linspace(1., 0., self.rows, device=reference.device, dtype=reference.dtype)
        ranks = torch.arange(self.capacity+1, device=reference.device)[None]
        trace = []
        for row in range(self.rows):
            types = torch.where(ranks < counts[:, None], PERSON,
                                torch.where(ranks == counts[:, None], PREDICT, PAD))[:, None]
            inputs = F.pad(noisy, (0, 0, 0, 1))[:, None]
            output = self.forward_rows(context, inputs, times[row:row+1], types, types == PAD)
            prediction = output['pred_coordinates'][-1, :, 0]
            stop_probability = output['eos_logits'][-1, :, 0].gather(1, counts[:, None]).squeeze(-1).sigmoid()
            eos = ~stopped & (stop_probability >= .5)
            birth = ~stopped & ~eos & (counts < self.capacity)
            old_counts = counts.clone()
            active = ranks[:, :self.capacity] < counts[:, None]
            born = (ranks[:, :self.capacity] == counts[:, None]) & birth[:, None]
            clean = torch.where((active | born)[..., None], prediction[:, :self.capacity], clean)
            scores = torch.where(born, (1-stop_probability)[:, None], scores)
            counts = counts+birth.long()
            capacity_stop = ~stopped & ~eos & (counts == self.capacity)
            stop_reason = torch.where(eos, 1, torch.where(capacity_stop, 2, stop_reason))
            eos_row = torch.where(eos, row+1, eos_row)
            stopped = stopped | eos | capacity_stop
            if collect_trace:
                for layer in range(2):
                    trace.append(dict(row=row+1, layer=layer+1,
                        sequence_indices=ranks.expand(batch, -1).clone(),
                        pred_coordinates=output['pred_coordinates'][layer, :, 0].clone(),
                        eos_probability=output['eos_logits'][layer, :, 0].gather(1, old_counts[:, None]).squeeze(-1).sigmoid(),
                        prefix_counts=old_counts, person_counts=counts.clone(), birth=birth.clone(),
                        stopped=stopped.clone(), stop_reason=stop_reason.clone()))
            if row+1 < self.rows:
                if isinstance(generator, (list, tuple)):
                    noise = torch.cat([draw_noise((1, self.capacity, 32), clean, item) for item in generator])
                else:
                    noise = draw_noise(clean.shape, clean, generator)
                clean_state = self.scale*(2*clean-1)
                alpha = row_alpha_bar(times[row+1])
                newborn = alpha.sqrt()*clean_state+(1-alpha).sqrt()*noise
                continued = ddim_transition(noisy, clean_state, times[row], times[row+1])
                noisy = torch.where(active[..., None], continued, newborn)
        result = self._geometry_output(clean, counts, scores, image_wh)
        result.update(stop_reason=stop_reason, eos_row=eos_row)
        if collect_trace:
            result['trace'] = trace
        return result

    @torch.no_grad()
    def sample_keypoints(self, images, *, image_wh, padding_mask=None, generator=None, collect_trace=False):
        return self.predict(self.prepare_context(images, padding_mask), image_wh,
                            generator=generator, collect_trace=collect_trace)

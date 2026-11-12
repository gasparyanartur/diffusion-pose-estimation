import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..diffusionregpose.model import DiffusionRegPoseModel
from ..diffusionregpose.transformer import SinusoidalPositionEmbeddings
from .rows_v46 import PAD, PERSON, PREDICT, draw_noise, row_alpha_bar
from .sampling_v46 import ddim_transition


class DenseRowLayer(nn.Module):
    def __init__(self, width, heads, feedforward_dim):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)
        self.image_attention = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)
        self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in range(3))
        self.feedforward = nn.Sequential(nn.Linear(width, feedforward_dim), nn.GELU(),
                                         nn.Linear(feedforward_dim, width))

    def forward(self, state, padding_mask, memory, position, image_mask):
        batch, rows, queries, width = state.shape
        query = self.norms[0](state).reshape(batch*rows, queries, width)
        update = self.self_attention(query, query, query,
            key_padding_mask=padding_mask.reshape(batch*rows, queries), need_weights=False)[0]
        state = state+update.reshape_as(state)
        # Cross-attention queries are independent; image K/V are projected once per image.
        query = self.norms[1](state).reshape(batch, rows*queries, width)
        update = self.image_attention(query, memory+position, memory,
            key_padding_mask=image_mask, need_weights=False)[0]
        state = state+update.reshape_as(state)
        state = state+self.feedforward(self.norms[2](state))
        return state.masked_fill(padding_mask[..., None], 0.)


class PoseFlowV46Model(nn.Module):
    prepare_transformer_inputs = DiffusionRegPoseModel.prepare_transformer_inputs
    encode_features = DiffusionRegPoseModel.encode_features
    _level_position_offsets = DiffusionRegPoseModel._level_position_offsets

    def __init__(self, *, codec, native, rows=64, capacity=50, hidden_dim=256,
                 num_heads=8, feedforward_dim=2048, activation_checkpointing=True):
        super().__init__()
        if not 0 < capacity < rows:
            raise ValueError('Rows must exceed positive person capacity')
        self.rows, self.capacity = rows, capacity
        self.num_proposals = capacity
        self.num_body_points = 14
        self.hidden_dim = hidden_dim
        self.num_heads, self.feedforward_dim = num_heads, feedforward_dim
        self.num_feature_levels = native.num_feature_levels
        self.backbone, self.input_proj = native.backbone, native.input_proj
        self.transformer = nn.Module()
        self.transformer.encoder = native.transformer.encoder
        self.transformer.level_embed = native.transformer.level_embed
        self.codec = codec.requires_grad_(False).eval()
        self.latent_dim = codec.latent_dim
        self.activation_checkpointing = activation_checkpointing
        self.latent_input = nn.Sequential(nn.Linear(self.latent_dim, hidden_dim), nn.GELU(),
                                          nn.Linear(hidden_dim, hidden_dim))
        self.predict_embedding = nn.Parameter(torch.zeros(hidden_dim))
        self.rank_embedding = nn.Embedding(rows, hidden_dim)
        self.time_embedding = nn.Sequential(SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim*4), nn.GELU(), nn.Linear(hidden_dim*4, hidden_dim))
        self.layers = nn.ModuleList(DenseRowLayer(hidden_dim, num_heads, feedforward_dim) for _ in range(2))
        self.readout_norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(2))
        self.latent_heads = nn.ModuleList(nn.Linear(hidden_dim, self.latent_dim) for _ in range(2))
        self.eos_heads = nn.ModuleList(nn.Linear(hidden_dim, 1) for _ in range(2))
        self.initialization_report = {}

    @property
    def sampler_kind(self):
        return f'ordered sequence; {self.rows} cosine rows; independent whole-prefix re-noising'

    def sampler_description(self, sampler):
        if sampler == 'renoise':
            return self.sampler_kind
        if sampler == 'ddim':
            return f'ordered sequence; {self.rows} cosine rows; DDIM eta0 existing tokens; forward-noised newborns'
        raise ValueError('Unknown V46 sampler; expected renoise or ddim')

    def prepare_context(self, images, padding_mask=None):
        if padding_mask is None:
            padding_mask = torch.zeros(len(images), *images.shape[2:], device=images.device, dtype=torch.bool)
        inputs = self.prepare_transformer_inputs(images, padding_mask)
        if inputs.padding_mask.all(-1).any():
            raise ValueError('Image requires valid feature tokens')
        return dict(memory=self.encode_features(inputs),
                    position=inputs.position+self._level_position_offsets(inputs),
                    padding_mask=inputs.padding_mask)

    def _decode_rows(self, state, padding_mask, memory, position, image_mask):
        latents, eos = [], []
        for layer, norm, head, eos_head in zip(self.layers, self.readout_norms, self.latent_heads, self.eos_heads):
            state = layer(state, padding_mask, memory, position, image_mask)
            readout = norm(state)
            latents.append(head(readout).masked_fill(padding_mask[..., None], 0.))
            eos.append(eos_head(readout).squeeze(-1).masked_fill(padding_mask, 0.))
        return torch.stack(latents), torch.stack(eos)

    def forward_rows(self, context, noisy_latents, row_times, token_types, padding_mask):
        if noisy_latents.ndim != 4 or noisy_latents.shape[-1] != self.latent_dim:
            raise ValueError('Expected noisy latents [B,R,Q,D]')
        batch, rows, queries, _ = noisy_latents.shape
        if (batch != len(context['memory']) or queries > self.rows or token_types.shape != (batch, rows, queries)
                or padding_mask.shape != token_types.shape or row_times.shape != (rows,)):
            raise ValueError('Row shapes disagree with model/context')
        if (not torch.isfinite(row_times).all() or ((row_times < 0) | (row_times > 1)).any()
                or not torch.equal(padding_mask, token_types == PAD)
                or ((token_types == PREDICT).sum(-1) != 1).any()):
            raise ValueError('Invalid row times, padding or prediction tokens')
        person = token_types == PERSON
        state = self.latent_input(torch.where(person[..., None], noisy_latents, 0.))
        state = torch.where(person[..., None], state, self.predict_embedding)
        state = state+self.rank_embedding.weight[:queries][None, None]
        state = state+self.time_embedding(row_times*999)[None, :, None]
        state = state.masked_fill(padding_mask[..., None], 0.)
        args = (state, padding_mask, context['memory'], context['position'], context['padding_mask'])
        if self.training and torch.is_grad_enabled() and self.activation_checkpointing:
            latents, eos = checkpoint(self._decode_rows, *args, use_reentrant=False, preserve_rng_state=True)
        else:
            latents, eos = self._decode_rows(*args)
        return dict(pred_latents=latents, eos_logits=eos)

    def _geometry_output(self, latents, counts, scores, image_wh):
        sizes = image_wh[:, None].expand(-1, self.capacity, -1).to(latents)
        geometry = self.codec.decode_latents(latents, sizes)
        points = geometry['keypoints_px']/sizes[..., None, :]
        valid = torch.arange(self.capacity, device=latents.device)[None] < counts[:, None]
        logits = torch.logit(scores.clamp(1e-8, 1-1e-7)).masked_fill(~valid, -30.)
        return dict(geometry, pred_latents=latents, scores=scores.masked_fill(~valid, 0.),
            existence_logits=logits, pred_logits=torch.stack((torch.full_like(logits, -30.), logits), -1),
            pred_boxes=geometry['boxes_cxcywh_px']/sizes.repeat(1, 1, 2),
            pred_keypoints=torch.cat((points.flatten(-2), torch.ones_like(points[..., 0])), -1),
            valid_mask=valid, person_counts=counts,
            sequence_indices=torch.arange(self.capacity, device=latents.device)[None].expand(len(latents), -1).masked_fill(~valid, -1))

    @torch.no_grad()
    def predict(self, context, image_wh, generator=None, collect_trace=False, *, sampler='renoise'):
        self.sampler_description(sampler)
        reference = context['memory']
        batch = len(reference)
        if image_wh.shape != (batch, 2) or not torch.isfinite(image_wh).all() or (image_wh <= 0).any():
            raise ValueError('Expected positive finite image sizes [B,2]')
        if isinstance(generator, (list, tuple)) and len(generator) != batch:
            raise ValueError('Per-image RNG list disagrees with batch size')
        clean = reference.new_zeros(batch, self.capacity, self.latent_dim)
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
            inputs = torch.nn.functional.pad(noisy, (0, 0, 0, 1))[:, None]
            output = self.forward_rows(context, inputs, times[row:row+1], types, types == PAD)
            prediction = output['pred_latents'][-1, :, 0]
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
                        pred_latents=output['pred_latents'][layer, :, 0].clone(),
                        eos_probability=output['eos_logits'][layer, :, 0].gather(1, old_counts[:, None]).squeeze(-1).sigmoid(),
                        prefix_counts=old_counts, person_counts=counts.clone(), birth=birth.clone(),
                        stopped=stopped.clone(), stop_reason=stop_reason.clone()))
            if row+1 < self.rows:
                if isinstance(generator, (list, tuple)):
                    noise = torch.cat([draw_noise((1, self.capacity, self.latent_dim), clean, item) for item in generator])
                else:
                    noise = draw_noise(clean.shape, clean, generator)
                alpha = row_alpha_bar(times[row+1])
                renoised = alpha.sqrt()*clean+(1-alpha).sqrt()*noise
                if sampler == 'ddim':
                    continued = ddim_transition(noisy, clean, times[row], times[row+1])
                    noisy = torch.where(active[..., None], continued, renoised)
                else:
                    noisy = renoised
        result = self._geometry_output(clean, counts, scores, image_wh)
        result.update(stop_reason=stop_reason, eos_row=eos_row)
        if collect_trace:
            result['trace'] = trace
        return result

    @torch.no_grad()
    def sample_keypoints(self, images, *, image_wh, padding_mask=None, generator=None, collect_trace=False,
                         sampler='renoise'):
        return self.predict(self.prepare_context(images, padding_mask), image_wh,
                            generator=generator, collect_trace=collect_trace, sampler=sampler)

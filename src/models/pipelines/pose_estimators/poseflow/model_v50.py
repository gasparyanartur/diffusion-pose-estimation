import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .human_context import NativeHumanContext
from .model_v49 import PoseFlowV49Model
from .proposals_v50 import ProposalSelector, gather_identity, residual_coordinates
from .rows_token_age import PAD, PERSON, PREDICT, draw_noise, token_age_sigmas
from .sampling_v49 import ddim_transition


class ProposalFrontend(NativeHumanContext, nn.Module):
    def __init__(self, native):
        super().__init__()
        self.native = native
        decoder = native.transformer.decoder
        # Native diffusion-time MLPs are unused by image-only context encoding.
        del native.transformer.time_mlp, native.transformer.block_time_mlp
        # Only the box frontend runs; discard unused pose expansion and decoder layers.
        decoder.layers = decoder.layers[:decoder.num_box_decoder_layers]
        for owner in (native, decoder):
            for name in ('pose_embed', 'pose_hw_embed', 'expansion_pose_embed'):
                if hasattr(owner, name):
                    delattr(owner, name)
        for name in ('hw', 'keypoint_embed', 'emb_Q', 'emb_K', 'emb_V', 'emb_SQ', 'emb_SK', 'emb_SV'):
            if hasattr(decoder, name):
                delattr(decoder, name)
        native.bbox_embed = native.bbox_embed[:decoder.num_box_decoder_layers]
        native.class_embed = native.class_embed[:decoder.num_box_decoder_layers]
        decoder.bbox_embed, decoder.class_embed = native.bbox_embed, native.class_embed

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__('native'), name)


class PoseFlowV50Model(PoseFlowV49Model):
    row_input_fields = PoseFlowV49Model.row_input_fields+('proposal_indices',)

    def __init__(self, *, native, **kwargs):
        super().__init__(native=native, **kwargs)
        if self.capacity > native.num_proposals:
            raise ValueError('V50 capacity exceeds native proposal capacity')
        del self.backbone, self.input_proj, self.transformer, self.eos_heads
        self.frontend = ProposalFrontend(native)
        self.selector = ProposalSelector(self.hidden_dim)
        self.reference_input = nn.Linear(4, self.hidden_dim)
        self.residual_heads = nn.ModuleList(nn.Linear(self.hidden_dim, 32) for _ in self.layers)
        for head in self.residual_heads:
            nn.init.normal_(head.weight, std=1e-3)
            nn.init.zeros_(head.bias)

    def prepare_context(self, images, padding_mask=None, *, targets=None):
        native = self.frontend.prepare_context(images, padding_mask, targets=targets)
        inputs = native['inputs']
        dense = self._project_image_context(native['memory'],
            inputs.position+self.frontend._level_position_offsets(inputs), inputs.padding_mask)
        return dict(native, **dense)

    def _decode_geometry(self, state, padding_mask, image_mask, *projected):
        coordinates, latents, residuals = [], [], []
        for index, layer in enumerate(self.layers):
            state = layer(state, padding_mask, projected[2*index], projected[2*index+1], image_mask)
            readout = self.readout_norms[index](state)
            latent = self.latent_heads[index](readout)
            latents.append(latent)
            coordinates.append(self.person_codec.decode(latent))
            residuals.append(self.residual_heads[index](readout))
        return torch.stack(coordinates), torch.stack(latents), torch.stack(residuals)

    def _selector_prefix(self, prefix):
        return prefix

    def _time_features(self, sigmas):
        return self.time_embedding(sigmas.reshape(-1)*999).reshape(*sigmas.shape, self.hidden_dim)

    def _refine_coordinates(self, coordinates, sigmas, residuals):
        return residual_coordinates(coordinates, sigmas, residuals, scale=self.scale)

    def forward_rows(self, context, noisy_coordinates, noise_sigmas, token_types, padding_mask, proposal_indices):
        shape = noisy_coordinates.shape[:-1]
        if (noisy_coordinates.ndim != 4 or noisy_coordinates.shape[-1] != 32
                or shape[0] != len(context['memory']) or shape[2] > self.rows
                or any(value.shape != shape for value in (noise_sigmas, token_types, padding_mask, proposal_indices))
                or not torch.equal(padding_mask, token_types == PAD)
                or ((token_types == PREDICT).sum(-1) != 1).any()
                or not ((token_types == PAD) | (token_types == PERSON) | (token_types == PREDICT)).all()
                or not torch.isfinite(noise_sigmas).all() or ((noise_sigmas < 0) | (noise_sigmas > 1)).any()
                or ((token_types == PERSON) & (noise_sigmas >= 1)).any()
                or (proposal_indices < -1).any() or (proposal_indices >= context['features'].shape[1]).any()
                or ((token_types == PERSON) & (proposal_indices < 0)).any()):
            raise ValueError('Invalid V50 row state, sigma or sample-local proposal identities')
        identity, reference = gather_identity(context['features'], context['boxes'], proposal_indices)
        person = token_types == PERSON
        active = ~padding_mask
        latent = noisy_coordinates.new_zeros(*shape, self.latent_dim)
        encoded = self.person_codec.encode((noisy_coordinates[active]/self.scale+1)/2)
        latent = latent.masked_scatter(active[..., None].expand_as(latent), encoded)
        state = self.latent_input(latent)
        prefix_state = state+identity+self.reference_input(reference)
        prefix = (prefix_state*person[..., None]).sum(-2)/person.sum(-1, keepdim=True).clamp_min(1)
        candidates = torch.arange(context['features'].shape[1], device=state.device)
        used = ((proposal_indices[..., None] == candidates) & person[..., None]).any(-2)
        selection_logits, eos = self.selector(self._selector_prefix(prefix), context['features'], used)
        chosen = selection_logits.argmax(-1)
        indices = torch.where((token_types == PREDICT) & (proposal_indices < 0), chosen[..., None], proposal_indices)
        identity, reference = gather_identity(context['features'], context['boxes'], indices)
        state = state+identity+self.reference_input(reference)
        state = state+(token_types == PREDICT)[..., None]*self.predict_embedding
        state = state+self.rank_embedding.weight[:shape[2]][None, None]
        state = state+self._time_features(noise_sigmas)
        state = state.masked_fill(padding_mask[..., None], 0.)
        projected = tuple(value for pair in zip(context['image_keys'], context['image_values']) for value in pair)
        args = (state, padding_mask, context['padding_mask'], *projected)
        if self.training and torch.is_grad_enabled() and self.activation_checkpointing:
            absolute, latents, residuals = checkpoint(self._decode_geometry, *args, use_reentrant=False, preserve_rng_state=True)
        else:
            absolute, latents, residuals = self._decode_geometry(*args)
        residual_input = noisy_coordinates.masked_fill(~person[..., None], 0.)
        residual_sigmas = noise_sigmas.masked_fill(~person, 0.)
        refined = self._refine_coordinates(residual_input[None], residual_sigmas[None], residuals)
        coordinates = torch.where(person[None, ..., None], refined, absolute).masked_fill(padding_mask[None, ..., None], 0.)
        return dict(pred_coordinates=coordinates, pred_latents=latents,
            eos_logits=eos[None, ..., None].expand(len(self.layers), *shape),
            selection_logits=selection_logits, selected_proposals=indices, used_candidates=used)

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
        ranks = torch.arange(self.capacity+1, device=reference.device)[None]
        identities = torch.full((batch, self.capacity), -1, device=reference.device, dtype=torch.long)
        trace = []
        for row in range(self.rows):
            old_counts = counts.clone()
            active = ranks < counts[:, None]
            query = ranks == counts[:, None]
            types = torch.where(active, PERSON, torch.where(query, PREDICT, PAD))[:, None]
            ages = (row-ranks).clamp_min(0)
            sigma = token_age_sigmas(ages, dtype=reference.dtype)
            if row == self.rows-1:
                sigma = torch.zeros_like(sigma)
            sigmas = torch.where(active, sigma, torch.where(query, 1., 0.))[:, None]
            if isinstance(generator, (list, tuple)):
                noise = torch.cat([draw_noise((1, self.capacity+1, 32), clean, item) for item in generator])
            else:
                noise = draw_noise((batch, self.capacity+1, 32), clean, generator)
            inputs = torch.where(query[..., None], noise, F.pad(noisy, (0, 0, 0, 1)))[:, None]
            inputs = inputs.masked_fill((types == PAD)[..., None], 0.)
            output = self.forward_rows(context, inputs, sigmas, types, types == PAD, F.pad(identities, (0, 1), value=-1)[:, None])
            prediction = output['pred_coordinates'][-1, :, 0]
            probability = output['eos_logits'][-1, :, 0].gather(1, counts[:, None]).squeeze(-1).sigmoid()
            eos = ~stopped & (probability >= .5)
            birth = ~stopped & ~eos & (counts < self.capacity) & (row < self.rows-1)
            born = (ranks[:, :self.capacity] == counts[:, None]) & birth[:, None]
            people = active[:, :self.capacity] | born
            chosen = output['selected_proposals'][:, 0].gather(1, counts[:, None]).squeeze(-1)
            identities = torch.where(born, chosen[:, None], identities)
            clean = torch.where(people[..., None], prediction[:, :self.capacity], clean)
            objectness = context['existence_logits'].sigmoid().gather(1, chosen[:, None]).squeeze(-1)
            scores = torch.where(born, (objectness*(1-probability))[:, None], scores)
            counts = counts+birth.long()
            capacity_stop = ~stopped & ~eos & (counts == self.capacity)
            stop_reason = torch.where(eos, 1, torch.where(capacity_stop, 2, stop_reason))
            eos_row = torch.where(eos, row+1, eos_row)
            stopped = stopped | eos | capacity_stop
            if collect_trace:
                for layer in range(len(self.layers)):
                    trace.append(dict(row=row+1, layer=layer+1,
                        sequence_indices=ranks.expand(batch, -1).clone(),
                        pred_coordinates=output['pred_coordinates'][layer, :, 0].clone(),
                        noise_sigmas=sigmas[:, 0].clone(), prefix_counts=old_counts.clone(),
                        proposal_indices=F.pad(identities, (0, 1), value=-1).clone(),
                        eos_probability=output['eos_logits'][layer, :, 0].gather(1, old_counts[:, None]).squeeze(-1).sigmoid(),
                        person_counts=counts.clone(), birth=birth.clone(), stopped=stopped.clone(), stop_reason=stop_reason.clone()))
            if row+1 < self.rows:
                next_sigmas = token_age_sigmas((row+1-ranks[:, :self.capacity]).clamp_min(0), dtype=reference.dtype)
                if row+1 == self.rows-1:
                    next_sigmas = torch.zeros_like(next_sigmas)
                next_sigmas = torch.where(people, next_sigmas, 0.)
                noisy = ddim_transition(inputs[:, 0, :self.capacity], self.scale*(2*clean-1),
                    sigmas[:, 0, :self.capacity, None], next_sigmas[..., None])
                noisy = noisy.masked_fill(~people[..., None], 0.)
        result = self._geometry_output(clean, counts, scores, image_wh)
        result.update(stop_reason=stop_reason, eos_row=eos_row,
            generation_steps=torch.full_like(counts, self.rows))
        result['proposal_indices'] = identities
        result['original_query_indices'] = context['indices'].gather(1, identities.clamp_min(0)).masked_fill(identities < 0, -1)
        if collect_trace:
            result['trace'] = trace
        return result

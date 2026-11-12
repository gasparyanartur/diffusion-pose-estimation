import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .model_v47 import PoseFlowV47Model
from .rows_token_age import PAD, PERSON, PREDICT, draw_noise, token_age_sigmas
from .sampling_v49 import ddim_transition


class PoseFlowV49Model(PoseFlowV47Model):
    row_input_fields = ('noisy_coordinates', 'noise_sigmas', 'token_types', 'padding_mask')

    def forward_rows(self, context, noisy_coordinates, noise_sigmas, token_types, padding_mask):
        if noisy_coordinates.ndim != 4 or noisy_coordinates.shape[-1] != 32:
            raise ValueError('Expected coordinate state [B,R,Q,32]')
        batch, rows, queries, _ = noisy_coordinates.shape
        if (batch != len(context['memory']) or queries > self.rows
                or token_types.shape != (batch, rows, queries) or padding_mask.shape != token_types.shape
                or noise_sigmas.shape != token_types.shape):
            raise ValueError('Row shapes disagree with model/context')
        if (not torch.isfinite(noise_sigmas).all() or ((noise_sigmas < 0) | (noise_sigmas > 1)).any()
                or not torch.equal(padding_mask, token_types == PAD)
                or not ((token_types == PAD) | (token_types == PERSON) | (token_types == PREDICT)).all()
                or ((token_types == PREDICT).sum(-1) != 1).any()):
            raise ValueError('Invalid per-token sigma, padding or query types')
        active = ~padding_mask
        encoded = self.person_codec.encode((noisy_coordinates[active]/self.scale+1)/2)
        latent = noisy_coordinates.new_zeros(batch, rows, queries, self.latent_dim)
        latent = latent.masked_scatter(active[..., None].expand_as(latent), encoded)
        state = self.latent_input(latent)
        state = state+(token_types == PREDICT)[..., None]*self.predict_embedding
        state = state+self.rank_embedding.weight[:queries][None, None]
        times = self.time_embedding(noise_sigmas.reshape(-1)*999).reshape_as(state)
        state = (state+times).masked_fill(padding_mask[..., None], 0.)
        projected = tuple(value for pair in zip(context['image_keys'], context['image_values']) for value in pair)
        args = (state, padding_mask, context['padding_mask'], *projected)
        if self.training and torch.is_grad_enabled() and self.activation_checkpointing:
            coordinates, latents, eos = checkpoint(self._decode_rows, *args, use_reentrant=False, preserve_rng_state=True)
        else:
            coordinates, latents, eos = self._decode_rows(*args)
        return dict(pred_coordinates=coordinates, pred_latents=latents, eos_logits=eos)

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
            output = self.forward_rows(context, inputs, sigmas, types, types == PAD)
            prediction = output['pred_coordinates'][-1, :, 0]
            probability = output['eos_logits'][-1, :, 0].gather(1, counts[:, None]).squeeze(-1).sigmoid()
            eos = ~stopped & (probability >= .5)
            birth = ~stopped & ~eos & (counts < self.capacity) & (row < self.rows-1)
            born = (ranks[:, :self.capacity] == counts[:, None]) & birth[:, None]
            people = active[:, :self.capacity] | born
            clean = torch.where(people[..., None], prediction[:, :self.capacity], clean)
            scores = torch.where(born, (1-probability)[:, None], scores)
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
        if collect_trace:
            result['trace'] = trace
        return result

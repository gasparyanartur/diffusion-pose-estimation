import torch
from torch.utils.checkpoint import checkpoint

from .model_v47 import PoseFlowV47Model
from .rows_v46 import PAD, PERSON, PREDICT


def select_context(context, indices):
    return {name: tuple(item[indices] for item in value) if isinstance(value, tuple) else value[indices]
            for name, value in context.items()}


class PoseFlowV48Model(PoseFlowV47Model):
    row_input_fields = ('prefix_coordinates', 'token_types', 'padding_mask')

    def __init__(self, *, native, capacity=50, hidden_dim=256, num_heads=8,
                 feedforward_dim=2048, codec_width=256, latent_dim=32, activation_checkpointing=True):
        super().__init__(native=native, rows=capacity+1, capacity=capacity, hidden_dim=hidden_dim,
            num_heads=num_heads, feedforward_dim=feedforward_dim, codec_width=codec_width,
            latent_dim=latent_dim, activation_checkpointing=activation_checkpointing)
        del self.time_embedding

    def forward_rows(self, context, prefix_coordinates, token_types, padding_mask):
        # Coordinates <B,R,Q,32> are clean image-normalized person states.
        if prefix_coordinates.ndim != 4 or prefix_coordinates.shape[-1] != 32:
            raise ValueError('Expected clean coordinates [B,R,Q,32]')
        batch, rows, queries, _ = prefix_coordinates.shape
        if (batch != len(context['memory']) or queries > self.rows
                or token_types.shape != (batch, rows, queries) or padding_mask.shape != token_types.shape):
            raise ValueError('Row shapes disagree with model/context')
        if (not torch.equal(padding_mask, token_types == PAD)
                or not ((token_types == PAD) | (token_types == PERSON) | (token_types == PREDICT)).all()
                or ((token_types == PREDICT).sum(-1) != 1).any()):
            raise ValueError('Invalid padding or token types')
        person = token_types == PERSON
        latent = prefix_coordinates.new_zeros(batch, rows, queries, self.latent_dim)
        if person.any():
            # V47's clean endpoint: inverse scaling of x=6*(2*c-1) gives c.
            encoded = self.person_codec.encode(prefix_coordinates[person])
            latent = latent.masked_scatter(person[..., None].expand_as(latent), encoded)
        state = torch.where(person[..., None], self.latent_input(latent), self.predict_embedding)
        state = state+self.rank_embedding.weight[:queries][None, None]
        state = state.masked_fill(padding_mask[..., None], 0.)
        projected = tuple(value for pair in zip(context['image_keys'], context['image_values']) for value in pair)
        args = (state, padding_mask, context['padding_mask'], *projected)
        if self.training and torch.is_grad_enabled() and self.activation_checkpointing:
            coordinates, latents, eos = checkpoint(self._decode_rows, *args, use_reentrant=False, preserve_rng_state=True)
        else:
            coordinates, latents, eos = self._decode_rows(*args)
        return dict(pred_coordinates=coordinates, pred_latents=latents, eos_logits=eos)

    @torch.no_grad()
    def predict(self, context, image_wh, collect_trace=False):
        reference = context['memory']
        batch = len(reference)
        if image_wh.shape != (batch, 2) or not torch.isfinite(image_wh).all() or (image_wh <= 0).any():
            raise ValueError('Expected positive finite image sizes [B,2]')
        clean = reference.new_zeros(batch, self.capacity, 32)
        scores = reference.new_zeros(batch, self.capacity)
        counts = torch.zeros(batch, device=reference.device, dtype=torch.long)
        stop_reason, eos_row, steps = counts.clone(), counts.clone(), counts.clone()
        trace = []
        for rank in range(self.capacity):
            active = (stop_reason == 0).nonzero().flatten()
            if not len(active):
                break
            inputs = clean.new_zeros(len(active), 1, rank+1, 32)
            inputs[:, 0, :rank] = clean[active, :rank]
            types = torch.full(inputs.shape[:-1], PERSON, device=reference.device, dtype=torch.long)
            types[..., -1] = PREDICT
            output = self.forward_rows(select_context(context, active), inputs, types, types == PAD)
            probability = output['eos_logits'][-1, :, 0, rank].sigmoid()
            eos = probability >= .5
            birth = ~eos
            born_images = active[birth]
            clean[born_images, rank] = output['pred_coordinates'][-1, birth, 0, rank]
            scores[born_images, rank] = 1-probability[birth]
            counts[born_images] += 1
            stop_reason[active[eos]] = 1
            eos_row[active[eos]] = rank+1
            steps[active] += 1
            if rank+1 == self.capacity:
                stop_reason[born_images] = 2
            if collect_trace:
                for layer in range(2):
                    trace.append(dict(step=rank+1, layer=layer+1, image_indices=active.clone(),
                        sequence_indices=torch.full_like(active, rank),
                        pred_coordinates=output['pred_coordinates'][layer, :, 0, rank].clone(),
                        eos_probability=output['eos_logits'][layer, :, 0, rank].sigmoid(),
                        birth=birth.clone(), stop_reason=stop_reason[active].clone()))
        result = self._geometry_output(clean, counts, scores, image_wh)
        result.update(stop_reason=stop_reason, eos_row=eos_row, generation_steps=steps)
        if collect_trace:
            result['trace'] = trace
        return result

    @torch.no_grad()
    def sample_keypoints(self, images, *, image_wh, padding_mask=None, collect_trace=False):
        return self.predict(self.prepare_context(images, padding_mask), image_wh, collect_trace=collect_trace)

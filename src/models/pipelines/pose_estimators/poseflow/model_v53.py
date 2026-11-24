from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .expanded_pose_v53 import ExpandedPoseDecoder
from .model_v47 import PoseFlowV47Model
from .model_v50 import ProposalFrontend
from .proposals_v50 import ProposalSelector, gather_identity
from .rows_v46 import PAD, PERSON, PREDICT


def select_context(context, indices):
    result = {name: context[name][indices] for name in
              ('memory', 'features', 'boxes', 'indices', 'existence_logits')}
    inputs = context['inputs']
    result['inputs'] = SimpleNamespace(spatial_shapes=inputs.spatial_shapes, level_start_index=inputs.level_start_index,
                                       valid_ratios=inputs.valid_ratios[indices])
    result['image_values'] = tuple(value[indices] for value in context['image_values'])
    return result


class PoseFlowV53Model(nn.Module):
    row_input_fields = ('prefix_coordinates', 'token_types', 'padding_mask', 'proposal_indices')
    _geometry_output = PoseFlowV47Model._geometry_output

    def __init__(self, *, native, capacity=49, activation_checkpointing=True):
        super().__init__()
        if not 0 < capacity <= native.num_proposals:
            raise ValueError('V53 capacity exceeds proposal count')
        self.capacity, self.rows = capacity, capacity+1
        decoder = native.transformer.decoder
        self.hidden_dim, self.num_heads = decoder.d_model, decoder.num_heads
        self.num_body_points = decoder.num_body_points
        self.pose_decoder = ExpandedPoseDecoder(native)
        self.frontend = ProposalFrontend(native)
        self.coordinate_input = nn.Sequential(nn.Linear(32, self.hidden_dim), nn.GELU(), nn.Linear(self.hidden_dim, self.hidden_dim))
        self.reference_input = nn.Linear(4, self.hidden_dim)
        self.selector = ProposalSelector(self.hidden_dim)
        self.activation_checkpointing = activation_checkpointing
        self.initialization_report = {}

    @property
    def layers(self):
        return self.pose_decoder.layers

    def prepare_context(self, images, padding_mask=None, *, targets=None):
        context = self.frontend.prepare_context(images, padding_mask, targets=targets)
        context['image_values'] = self.pose_decoder.project_memory(context['memory'], context['inputs'].padding_mask)
        return context

    def select_birth(self, context, prefix, person, used, supplied):
        summary = prefix.sum(-2)/person.sum(-1, keepdim=True).clamp_min(1)
        selection, eos = self.selector(summary.detach(), context['features'], used)
        chosen = torch.where(supplied >= 0, supplied, selection.argmax(-1))
        feature, box = gather_identity(context['features'], context['boxes'], chosen)
        return selection, eos, chosen, feature, box

    def forward_rows(self, context, prefix_coordinates, token_types, padding_mask, proposal_indices):
        # B images, R independent teacher rows, Q prefix positions plus one PREDICT.
        shape = prefix_coordinates.shape[:-1]
        if (prefix_coordinates.ndim != 4 or prefix_coordinates.shape[-1] != 32 or shape[0] != len(context['memory'])
                or shape[2] > self.rows or any(value.shape != shape for value in (token_types, padding_mask, proposal_indices))
                or not torch.equal(padding_mask, token_types == PAD)
                or not ((token_types == PAD) | (token_types == PERSON) | (token_types == PREDICT)).all()
                or ((token_types == PREDICT).sum(-1) != 1).any()
                or (proposal_indices < -1).any() or (proposal_indices >= context['features'].shape[1]).any()):
            raise ValueError('Invalid V53 rows, padding or proposal identities')
        person = token_types == PERSON
        if not torch.isfinite(prefix_coordinates[person]).all() or (person & (proposal_indices < 0)).any():
            raise ValueError('Invalid V53 clean prefix')
        coordinates = prefix_coordinates.masked_fill(~person[..., None], 0.)
        identity, reference = gather_identity(context['features'], context['boxes'], proposal_indices)
        prefix = (self.coordinate_input(coordinates)+identity+self.reference_input(reference)).masked_fill(~person[..., None], 0.)
        candidates = torch.arange(context['features'].shape[1], device=prefix.device)
        used = ((proposal_indices[..., None] == candidates) & person[..., None]).any(-2)
        position = (token_types == PREDICT).long().argmax(-1)
        supplied = proposal_indices.gather(-1, position[..., None]).squeeze(-1)
        selection, eos, chosen, feature, box = self.select_birth(context, prefix, person, used, supplied)
        prefix_boxes = coordinates[..., 28:].masked_fill(~person[..., None], .5)
        inputs = context['inputs']
        arguments = (feature[..., None, :], box[..., None, :], prefix, prefix_boxes, ~person,
                     context['image_values'], inputs.spatial_shapes, inputs.level_start_index, inputs.valid_ratios)
        if self.training and torch.is_grad_enabled() and self.activation_checkpointing:
            predictions = checkpoint(self.pose_decoder, *arguments, use_reentrant=False, preserve_rng_state=True)
        else:
            predictions = self.pose_decoder(*arguments)
        return dict(pred_coordinates=predictions, eos_logits=eos[None].expand(len(self.layers), -1, -1),
                    selection_logits=selection, selected_proposals=chosen)

    @torch.no_grad()
    def predict(self, context, image_wh, collect_trace=False):
        memory = context['memory']
        batch = len(memory)
        if image_wh.shape != (batch, 2) or not torch.isfinite(image_wh).all() or (image_wh <= 0).any():
            raise ValueError('Expected positive finite image sizes [B,2]')
        clean = memory.new_zeros(batch, self.capacity, 32)
        scores = memory.new_zeros(batch, self.capacity)
        counts = torch.zeros(batch, device=memory.device, dtype=torch.long)
        identities = torch.full((batch, self.capacity), -1, device=memory.device, dtype=torch.long)
        stop_reason, eos_row, calls = counts.clone(), counts.clone(), counts.clone()
        trace = []
        for rank in range(self.capacity):
            images = (stop_reason == 0).nonzero().flatten()
            if not len(images):
                break
            coordinates = clean.new_zeros(len(images), 1, rank+1, 32)
            coordinates[:, 0, :rank] = clean[images, :rank]
            types = torch.full(coordinates.shape[:-1], PERSON, device=memory.device, dtype=torch.long)
            types[..., -1] = PREDICT
            proposals = torch.full_like(types, -1)
            proposals[:, 0, :rank] = identities[images, :rank]
            output = self.forward_rows(select_context(context, images), coordinates, types, types == PAD, proposals)
            probability = output['eos_logits'][-1, :, 0].sigmoid()
            birth = probability < .5
            chosen = output['selected_proposals'][:, 0]
            born = images[birth]
            clean[born, rank] = output['pred_coordinates'][-1, birth, 0]
            identities[born, rank] = chosen[birth]
            objectness = context['existence_logits'][images].sigmoid().gather(1, chosen[:, None]).squeeze(-1)
            scores[born, rank] = (objectness*(1-probability))[birth]
            counts[born] += 1
            calls[images] += 1
            stop_reason[images[~birth]] = 1
            eos_row[images[~birth]] = rank+1
            if rank+1 == self.capacity:
                stop_reason[born] = 2
            if collect_trace:
                for layer in range(len(self.layers)):
                    trace.append(dict(step=rank+1, layer=layer+1, image_indices=images.clone(), birth=birth.clone(),
                        proposal_indices=chosen.clone(), pred_coordinates=output['pred_coordinates'][layer, :, 0].clone(),
                        eos_probability=probability.clone(), stop_reason=stop_reason[images].clone()))
        result = self._geometry_output(clean, counts, scores, image_wh)
        original = context['indices'].gather(1, identities.clamp_min(0)).masked_fill(identities < 0, -1)
        result.update(proposal_indices=identities, original_query_indices=original,
                      stop_reason=stop_reason, eos_row=eos_row, generation_steps=calls)
        if collect_trace:
            result['trace'] = trace
        return result

    @torch.no_grad()
    def sample_keypoints(self, images, *, image_wh, padding_mask=None, collect_trace=False):
        return self.predict(self.prepare_context(images, padding_mask), image_wh, collect_trace=collect_trace)

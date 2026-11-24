import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .full_depth_birth import FullDepthBirthDecoder
from .model_v55 import PoseFlowV55Model
from .proposal_context_v55 import ContextualProposalSelector
from .proposals_v50 import gather_identity
from .rows_v46 import PAD, PERSON, PREDICT


class PoseFlowV56Model(PoseFlowV55Model):
    def __init__(self, *, native, capacity=49, activation_checkpointing=True):
        nn.Module.__init__(self)
        if not 0 < capacity <= native.num_proposals:
            raise ValueError('V56 capacity exceeds native proposal count')
        adapter = FullDepthBirthDecoder(native=native)
        self.pose_decoder = adapter.pose_decoder
        self.frontend = adapter.frontend
        self.proposal_context = adapter.proposal_context
        self.proposal_context.detach_prefix = False
        self.hidden_dim = self.pose_decoder.width
        self.num_heads = self.pose_decoder.layers[0].self_attn.num_heads
        self.num_body_points = self.pose_decoder.points
        self.capacity, self.rows = capacity, capacity+1
        self.selector = ContextualProposalSelector(self.hidden_dim)
        self.coordinate_input = nn.Sequential(nn.Linear(32, self.hidden_dim), nn.GELU(),
                                               nn.Linear(self.hidden_dim, self.hidden_dim))
        self.reference_input = nn.Linear(4, self.hidden_dim)
        self.activation_checkpointing = activation_checkpointing
        self.initialization_report = {}
        self.architecture = 'V56'

    def select_birth(self, context, prefix, person, used, supplied):
        selection, eos, chosen, corrected, box = super().select_birth(context, prefix, person, used, supplied)
        batch = torch.arange(len(chosen), device=chosen.device)[:, None]
        raw = context['features'][batch, chosen]
        return selection, eos, chosen, raw+(corrected-raw.detach()), box

    def forward_rows(self, context, prefix_coordinates, token_types, padding_mask, proposal_indices):
        # B images, R independent teacher rows, Q previous people plus PREDICT/padding.
        shape = prefix_coordinates.shape[:-1]
        if (prefix_coordinates.ndim != 4 or prefix_coordinates.shape[-1] != 32
                or shape[0] != len(context['memory']) or shape[2] > self.rows
                or any(value.shape != shape for value in (token_types, padding_mask, proposal_indices))
                or not torch.equal(padding_mask, token_types == PAD)
                or not ((token_types == PAD) | (token_types == PERSON) | (token_types == PREDICT)).all()
                or ((token_types == PREDICT).sum(-1) != 1).any()
                or (proposal_indices < -1).any() or (proposal_indices >= context['features'].shape[1]).any()):
            raise ValueError('Invalid V56 rows, padding or proposal identities')
        person = token_types == PERSON
        if not torch.isfinite(prefix_coordinates[person]).all() or (person & (proposal_indices < 0)).any():
            raise ValueError('Invalid V56 fixed prefix')
        coordinates = prefix_coordinates.masked_fill(~person[..., None], 0.)
        identity, reference = gather_identity(context['features'], context['boxes'], proposal_indices)
        prefix = (self.coordinate_input(coordinates)+identity+self.reference_input(reference)).masked_fill(~person[..., None], 0.)
        candidates = torch.arange(context['features'].shape[1], device=prefix.device)
        used = ((proposal_indices[..., None] == candidates) & person[..., None]).any(-2)
        position = (token_types == PREDICT).long().argmax(-1)
        supplied = proposal_indices.gather(-1, position[..., None]).squeeze(-1)
        selection, eos, chosen, feature, box = self.select_birth(context, prefix, person, used, supplied)
        batch, rows = chosen.shape
        empty = feature.new_zeros(batch, rows, 0, self.hidden_dim)
        empty_boxes = box.new_zeros(batch, rows, 0, 4)
        empty_mask = torch.zeros(batch, rows, 0, dtype=torch.bool, device=feature.device)
        inputs = context['inputs']
        arguments = (feature[..., None, :], box[..., None, :], empty, empty_boxes, empty_mask,
            context['image_values'], inputs.spatial_shapes, inputs.level_start_index, inputs.valid_ratios)
        if self.training and torch.is_grad_enabled() and self.activation_checkpointing:
            predictions = checkpoint(self.pose_decoder, *arguments, use_reentrant=False, preserve_rng_state=True)
        else:
            predictions = self.pose_decoder(*arguments)
        return dict(pred_coordinates=predictions, eos_logits=eos[None].expand(len(self.layers), -1, -1),
                    selection_logits=selection, selected_proposals=chosen)

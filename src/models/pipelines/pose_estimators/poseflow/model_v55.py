import torch
from torch.utils.checkpoint import checkpoint

from .model_v53 import PoseFlowV53Model
from .proposal_context_v55 import ContextualProposalSelector, ProposalContext
from .proposals_v50 import gather_identity


class PoseFlowV55Model(PoseFlowV53Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.proposal_context = ProposalContext(self.hidden_dim, self.num_heads)
        self.selector = ContextualProposalSelector(self.hidden_dim)
        self.architecture = 'V55'

    def select_birth(self, context, prefix, person, used, supplied):
        arguments = (context['features'], prefix, person)
        if self.training and torch.is_grad_enabled() and self.activation_checkpointing:
            contextualized = checkpoint(self.proposal_context, *arguments, use_reentrant=False, preserve_rng_state=True)
        else:
            contextualized = self.proposal_context(*arguments)
        selection, eos = self.selector(contextualized, used)
        chosen = torch.where(supplied >= 0, supplied, selection.argmax(-1))
        feature = contextualized.gather(-2, chosen[..., None, None].expand(*chosen.shape, 1, self.hidden_dim)).squeeze(-2)
        _, box = gather_identity(context['features'], context['boxes'], chosen)
        return selection, eos, chosen, feature, box

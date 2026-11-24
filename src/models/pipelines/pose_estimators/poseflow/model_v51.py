from copy import deepcopy

from .model_v50 import PoseFlowV50Model
from .proposals_v51 import clean_anchor_residual


class PoseFlowV51Model(PoseFlowV50Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.residual_heads = deepcopy(self.latent_heads)

    def _selector_prefix(self, prefix):
        self.prefix_rms = prefix.detach().square().mean().sqrt()
        return prefix.detach()

    def _time_features(self, sigmas):
        features = super()._time_features(sigmas)
        self.time_rms_before_cap = features.detach().square().mean(-1).sqrt().max()
        bounded = features/features.square().mean(-1, keepdim=True).clamp_min(1.).sqrt()
        self.time_rms_after_cap = bounded.detach().square().mean(-1).sqrt().max()
        return bounded

    def _refine_coordinates(self, coordinates, sigmas, residuals):
        anchor = self.person_codec.decode(residuals)
        return clean_anchor_residual(coordinates, sigmas, anchor, scale=self.scale)

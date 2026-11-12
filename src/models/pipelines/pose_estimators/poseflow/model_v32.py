import torch

from .model_v25 import HFeatureGridDecoder
from .model_v26 import PoseFlowV26Model


class PoseFlowV32Model(PoseFlowV26Model):
    def __init__(self, *, dino_grid_size=7, **kwargs):
        super().__init__(**kwargs)
        with torch.random.fork_rng(devices=[]):
            self.dino_head = HFeatureGridDecoder(self.hidden_dim, grid_size=dino_grid_size)

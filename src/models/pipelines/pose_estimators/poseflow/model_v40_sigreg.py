from torch import nn

from .model_v40 import PoseFlowV40Model


class PoseFlowV40SigRegModel(PoseFlowV40Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.clean_norm = nn.Identity()

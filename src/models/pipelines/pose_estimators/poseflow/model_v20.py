import torch
from torch import nn

from .dense_read_v19 import PoseFlowV19DenseReadModel
from .guidance_v20 import GFPOSE_SCALE


class PoseFlowV20Model(PoseFlowV19DenseReadModel):
    def __init__(self, image_model, *, sampling_steps=3):
        super().__init__(image_model, sampling_steps=sampling_steps)
        with torch.random.fork_rng(devices=[]):
            self.pose3d_head = nn.Linear(self.hidden_dim, 17*3)
            nn.init.normal_(self.pose3d_head.weight, std=.01)
            nn.init.zeros_(self.pose3d_head.bias)
        self.initialization_report['fresh_components'] += ['pose3d_head']
        total = sum(value.numel() for value in self.state_dict().values())
        self.initialization_report.update(total_state_tensors=len(self.state_dict()),
            total_state_elements=total,
            copied_source_fraction=self.initialization_report['copied_source_elements']/total)

    def read_pose3d(self, hypotheses):
        # B = images, N = candidates; teacher state <B,N,17,3>.
        return self.pose3d_head(hypotheses).reshape(*hypotheses.shape[:2], 17, 3)*GFPOSE_SCALE

    def load_dense_state_dict(self, state):
        current = self.state_dict()
        new_keys = {name for name in current if name.startswith('pose3d_head.')}
        if state.keys() != current.keys()-new_keys:
            raise ValueError('Expected exactly a V19 dense-read model state')
        if any(state[name].shape != current[name].shape for name in state):
            raise ValueError('Dense-read baseline tensor shapes do not match V20')
        result = self.load_state_dict(state, strict=False)
        if set(result.missing_keys) != new_keys or result.unexpected_keys:
            raise RuntimeError('Unexpected V20 baseline import boundary')
        return {'loaded_tensors': len(state), 'new_keys': sorted(new_keys)}

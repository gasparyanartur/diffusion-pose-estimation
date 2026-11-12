import torch
from torch import Tensor, nn

from .model_v24 import HypothesisContext, PoseFlowV24Model


class HFeatureGridDecoder(nn.Module):
    def __init__(self, width: int, *, grid_size: int = 7, feature_dim: int = 384):
        super().__init__()
        self.position = nn.Parameter(torch.empty(grid_size * grid_size, width))
        nn.init.normal_(self.position, std=.02)
        self.readout = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width * 2), nn.GELU(),
            nn.Linear(width * 2, feature_dim),
        )

    def forward(self, hypotheses: Tensor) -> Tensor:
        # B images, N person candidates, S spatial cells, D feature channels.
        return self.readout(hypotheses.unsqueeze(-2) + self.position)  # <B,N,S,D>


class PoseFlowV25Model(PoseFlowV24Model):
    def __init__(self, *, h_grid_size: int = 7, **kwargs):
        super().__init__(**kwargs)
        if self.h_branch is None:
            raise ValueError('V25 DINO supervision requires h')
        with torch.random.fork_rng(devices=[]):
            self.h_decoder = HFeatureGridDecoder(self.transformer.decoder.d_model,
                                                 grid_size=h_grid_size)

    def decoder_context_outputs(self, context: HypothesisContext) -> dict[str, Tensor]:
        if context.hypotheses is None:
            raise RuntimeError('V25 did not produce final person h')
        return {'h_final': context.hypotheses.clone()}

    def load_v24_state(self, state: dict[str, Tensor]) -> dict:
        own = self.state_dict()
        new_keys = {key for key in own if key.startswith('h_decoder.')}
        expected = set(own) - new_keys
        if set(state) != expected:
            raise RuntimeError('V24 import must contain every shared tensor and only omit h_decoder')
        if any(state[key].shape != own[key].shape for key in expected):
            raise RuntimeError('V24 import has incompatible shared tensor shapes')
        result = self.load_state_dict(state, strict=False)
        if set(result.missing_keys) != new_keys or result.unexpected_keys:
            raise RuntimeError('Unexpected V24 import mismatch')
        return {'shared_tensors': len(expected), 'new_state_keys': sorted(new_keys)}

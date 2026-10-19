import hashlib
from pathlib import Path

import torch

from .network import GFPoseNetwork
from ..guidance_v20 import gfpose_marginal


class FrozenGFPose:
    def __init__(self, checkpoint, expected_sha256):
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f'GFPose training weights not found: {path}; see gfpose/NOTICE')
        digest = self._digest(path)
        if digest != expected_sha256:
            raise ValueError('GFPose checkpoint SHA256 does not match the configured teacher')
        state = torch.load(path, map_location='cpu', weights_only=True)
        # Loading the auxiliary teacher must not consume the student's initialization/RNG stream.
        with torch.random.fork_rng(devices=[]):
            self.network = GFPoseNetwork()
        self.network.load_state_dict(state['model_state_dict'], strict=True)
        parameters = list(self.network.parameters())
        shadows = state['ema']['shadow_params']
        if len(parameters) != len(shadows):
            raise ValueError('GFPose EMA does not match the released architecture')
        # EMA is stored as a positional list; bind it to the audited upstream names.
        source_order = self._upstream_parameter_order()
        named = dict(self.network.named_parameters())
        if set(source_order) != set(named):
            raise ValueError('Unexpected GFPose parameter names')
        with torch.no_grad():
            for name, shadow in zip(source_order, shadows):
                if named[name].shape != shadow.shape:
                    raise ValueError(f'GFPose EMA shape mismatch: {name}')
                named[name].copy_(shadow)
        self.network.eval().requires_grad_(False)
        self.sha256 = digest

    @staticmethod
    def _digest(path):
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b''):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _upstream_parameter_order():
        modules = ['pre_dense', 'pre_dense_t', 'pre_dense_cond', 'pre_gnorm',
                   'shared_time_embed.0', 'cond_embed.0']
        for block in (1, 2):
            for layer in (1, 2):
                modules += [f'b{block}_dense{layer}', f'b{block}_dense{layer}_t',
                            f'b{block}_dense{layer}_cond', f'b{block}_gnorm{layer}']
        modules += ['post_dense']
        return [f'{module}.{part}' for module in modules for part in ('weight', 'bias')]

    @torch.no_grad()
    def score(self, poses, times):
        if poses.shape[-2:] != (17, 3) or times.shape != poses.shape[:1]:
            raise ValueError('GFPose requires poses [N,17,3] and times [N]')
        self.network.to(device=poses.device, dtype=torch.float32).eval()
        with torch.autocast(device_type=poses.device.type, enabled=False):
            _, sigma = gfpose_marginal(times.float())
            return -self.network(poses.float(), times.float()*999)/sigma[:, None, None]

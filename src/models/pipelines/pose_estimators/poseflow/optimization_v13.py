import math

import torch


def reconstruction_weight(start: float, end: float, decay_epochs: int, epoch: int) -> float:
    if not math.isfinite(start) or not math.isfinite(end) or min(start, end) <= 0 or decay_epochs < 2:
        raise ValueError('Reconstruction weights must be positive and decay must span at least two epochs')
    if epoch <= 0 or start == end:
        return start
    if epoch >= decay_epochs - 1:
        return end
    progress = epoch / (decay_epochs - 1)
    return math.exp(math.log(start) + progress * (math.log(end) - math.log(start)))


def parameter_component(name: str) -> str:
    name = name.removeprefix('model.')
    if name.startswith('codec.'):
        if '_encoder.' in name or name.startswith('codec.roles.'):
            return 'codec_encoder'
        if name.startswith('codec.existence_head.'):
            return 'existence_readout'
        return 'geometry_readouts'
    if name.startswith('backbone.'):
        return 'backbone'
    if name.startswith(('encoder.', 'input_proj.', 'level_embed')):
        return 'image_encoder'
    if '.latent_head.' in name:
        return 'latent_heads'
    return 'state_denoiser'


def gradient_norms(named_parameters, gradients) -> dict[str, torch.Tensor]:
    squared_norms = {}
    for (name, _), gradient in zip(named_parameters, gradients):
        if gradient is None:
            continue
        component = parameter_component(name)
        squared = gradient.detach().float().square().sum()
        squared_norms[component] = squared_norms.get(component, 0) + squared
    if not squared_norms:
        return {}
    norms = {name: value.sqrt() for name, value in squared_norms.items()}
    norms['global'] = torch.stack(list(squared_norms.values())).sum().sqrt()
    return norms

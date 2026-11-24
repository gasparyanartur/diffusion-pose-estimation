import torch


def clean_anchor_residual(noisy, sigmas, anchor, *, scale):
    alpha = (1-sigmas.square()).clamp_min(1e-8).sqrt()
    base = (noisy/alpha[..., None]/scale+1)/2
    gate = (sigmas/.1).clamp(0., 1.)[..., None]
    xy = base[..., :30]+gate*(anchor[..., :30]-base[..., :30])
    base_size = base[..., 30:].clamp_min(1e-6)
    log_ratio = (anchor[..., 30:].clamp_min(1e-6)/base_size).log()
    size = base_size*torch.exp((gate*log_ratio).clamp(-20., 20.))
    refined = torch.cat((xy, size), -1)
    return torch.where((sigmas == 0)[..., None], (noisy/scale+1)/2, refined)

import math

import torch
from torch import nn


class GradientBalance(nn.Module):
    def __init__(self, probes, *, decay=.95):
        super().__init__()
        self.decay = decay
        self.register_buffer('ema', torch.zeros(4, probes))
        self.register_buffer('seen', torch.zeros(4, probes, dtype=torch.bool))
        self.register_buffer('weights', torch.ones(4))
        self.register_buffer('measurements', torch.zeros((), dtype=torch.long))
        self.register_buffer('prior', torch.tensor([.125, .375, 1/3, 1/6]))

    @torch.no_grad()
    def update(self, norms):
        norms = torch.as_tensor(norms, device=self.ema.device, dtype=self.ema.dtype)
        if norms.shape != self.ema.shape or not torch.isfinite(norms).all() or (norms < 0).any():
            raise ValueError('Invalid V51 task/probe gradients')
        present = norms > 1e-12
        smoothed = torch.where(self.seen, self.decay*self.ema+(1-self.decay)*norms, norms)
        self.ema.copy_(torch.where(present, smoothed, self.ema))
        self.seen.logical_or_(present)
        shares = self.ema/self.ema.sum(0, keepdim=True).clamp_min(1e-12)
        log_score = (shares.clamp_min(1e-6).log()*self.seen).sum(-1)/self.seen.sum(-1).clamp_min(1)
        active = self.seen.any(-1)
        proposed = self.prior/log_score.exp().clamp_min(1e-6)
        if active.any():
            proposed = proposed/proposed[active].log().mean().exp()
        self.weights.copy_(torch.where(active, proposed.clamp(.1, 10.), 1.))
        self.measurements.add_(1)


def gradient_families(model):
    groups = dict(native=[], selector=[], geometry=[])
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            family = 'native' if name.startswith('frontend.') else 'selector' if name.startswith('selector.') else 'geometry'
            groups[family].append(parameter)
    return groups


def clip_families(families, *, total_bound=.1):
    if not families or total_bound <= 0:
        raise ValueError('Expected positive clipping bound and parameter families')
    bound = total_bound/math.sqrt(len(families))
    return clip_families_with_budgets(families, {name: bound for name in families}, total_bound=total_bound)


def clip_families_with_budgets(families, budgets, *, total_bound=.1):
    identities = [id(parameter) for parameters in families.values() for parameter in parameters]
    if len(identities) != len(set(identities)):
        raise ValueError('V51 gradient ownership must be disjoint')
    if not families or not math.isfinite(total_bound) or total_bound <= 0:
        raise ValueError('Expected positive clipping bound and parameter families')
    if (set(budgets) != set(families)
            or any(not math.isfinite(bound) or bound <= 0 for bound in budgets.values())):
        raise ValueError('Expected one finite positive clipping budget per family')
    if math.fsum(bound**2 for bound in budgets.values()) > total_bound**2*(1+1e-12):
        raise ValueError('Family clipping budgets exceed combined bound')
    result = {}
    for name, parameters in families.items():
        bound = budgets[name]
        norm = torch.nn.utils.clip_grad_norm_(parameters, bound, error_if_nonfinite=True)
        result[f'{name}/preclip'] = norm
        result[f'{name}/postclip'] = norm.clamp(max=bound)
        result[f'{name}/clip_factor'] = (bound/(norm+1e-6)).clamp(max=1.)
    return result

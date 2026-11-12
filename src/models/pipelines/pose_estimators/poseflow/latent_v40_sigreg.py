# Adapted from LeJEPA MINIMAL.md by Randall Balestriero and Yann LeCun.
# Revision c293d291ca87cd4fddee9d3fffe4e914c7272052, CC-BY-NC-4.0.
# Attribution and license: docs/licenses/lejepa-CC-BY-NC-4.0.txt.
import torch


def sigreg_penalty(codes, directions, *, knots=17, t_max=3.):
    # codes: <N, D>; directions: <D, M>, with unit-length columns.
    with torch.autocast(device_type=codes.device.type, enabled=False):
        codes = codes.float()
        if len(codes) < 2:
            return codes.sum()*0
        frequencies = torch.linspace(0, t_max, knots, device=codes.device, dtype=torch.float32)
        target = torch.exp(-frequencies.square()/2)
        weights = torch.full_like(frequencies, 2*t_max/(knots-1))
        weights[[0, -1]] *= .5
        phase = (codes @ directions.float())[..., None]*frequencies
        error = (phase.cos().mean(0)-target).square()+phase.sin().mean(0).square()
        return ((error*target*weights).sum(-1)*len(codes)).mean()


@torch.no_grad()
def latent_distribution_metrics(codes):
    if not len(codes):
        return {}
    codes = codes.detach().float()
    variance = codes.var(dim=0, unbiased=False)
    metrics = dict(mean=codes.mean(), rms=codes.square().mean().sqrt(),
                   variance_mean=variance.mean(), variance_min=variance.min())
    if len(codes) > 1:
        centered = codes-codes.mean(0)
        covariance = centered.T @ centered/len(codes)
        off_diagonal = covariance-torch.diag_embed(covariance.diag())
        metrics['covariance_offdiag_rms'] = off_diagonal.square().mean().sqrt()
    return metrics

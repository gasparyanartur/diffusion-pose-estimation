import torch


def ddim_transition(state, predicted_clean, current_sigma, next_sigma):
    current = torch.as_tensor(current_sigma, device=state.device, dtype=state.dtype)
    following = torch.as_tensor(next_sigma, device=state.device, dtype=state.dtype)
    if (not torch.isfinite(current).all() or not torch.isfinite(following).all()
            or ((following < 0) | (following > current) | (current > 1)).any()):
        raise ValueError('DDIM sigma must decrease within [0,1]')
    # The ratio form avoids dividing a coordinate residual by a tiny sigma.
    ratio = following/current.clamp_min(torch.finfo(state.dtype).tiny)
    result = (1-following.square()).sqrt()*predicted_clean+ratio*(
        state-(1-current.square()).sqrt()*predicted_clean)
    return torch.where(following == 0, predicted_clean, result)

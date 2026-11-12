import torch

from .rows_v46 import row_alpha_bar


def ddim_transition(state, predicted_clean, current_time, next_time):
    times = torch.stack([torch.as_tensor(value, device=state.device, dtype=state.dtype)
                         for value in (current_time, next_time)])
    if not torch.isfinite(times).all() or not (0 <= times[1] <= times[0] <= 1):
        raise ValueError('DDIM times must decrease within [0,1]')
    if times[1] == 0:
        return predicted_clean
    current_alpha, next_alpha = row_alpha_bar(times)
    residual = (state-current_alpha.sqrt()*predicted_clean)/(1-current_alpha).sqrt()
    return next_alpha.sqrt()*predicted_clean+(1-next_alpha).sqrt()*residual

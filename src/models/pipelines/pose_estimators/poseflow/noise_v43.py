import torch


def noise_coefficients(sigma, state):
    sigma = torch.as_tensor(sigma, device=state.device, dtype=state.dtype)
    if sigma.shape != (len(state),) or not torch.isfinite(sigma).all() or (sigma < 0).any():
        raise ValueError('Expected finite nonnegative sigma with shape [B]')
    sigma = sigma.reshape(len(state), *((1,) * (state.ndim-1)))
    alpha = (1+sigma.square()).rsqrt()
    return alpha, sigma*alpha


def velocity_target(clean, noise, sigma):
    alpha, beta = noise_coefficients(sigma, clean)
    return alpha*noise-beta*clean


def clean_from_velocity(state, velocity, sigma):
    alpha, beta = noise_coefficients(sigma, state)
    return alpha*state-beta*velocity

import random

import numpy as np
import torch

from .losses_v49_balanced import TASK_NAMES


def fixed_scales(norms):
    values = torch.as_tensor(norms, dtype=torch.float64, device='cpu')
    if (values.ndim != 2 or values.shape[1] != 4 or not len(values)
            or not torch.isfinite(values).all() or (values < 0).any()):
        raise ValueError('Invalid V49 calibration norms')
    if (values.max(0).values == 0).any():
        raise ValueError('Inactive V49 calibration task')
    return values.quantile(.5, dim=0).clamp_min(1e-8).float()


def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def calibrate(module, batches):
    if module.calibration_ready:
        raise RuntimeError('V49 gradient calibration is already frozen')
    rng, mode = capture_rng(), module.training
    initial = {name: value.detach().cpu().clone() for name, value in module.model.state_dict().items()}
    seed = module.config.calibration_seed
    norms, image_ids = [], []
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        module.train()
        parameters = tuple(module.model.layers.parameters())
        for index, batch in enumerate(batches):
            losses = module.calibration_losses(batch, index)
            values = []
            for task in TASK_NAMES:
                gradients = torch.autograd.grad(losses[task], parameters, retain_graph=True, allow_unused=True)
                squared = sum(float(gradient.detach().double().square().sum()) for gradient in gradients if gradient is not None)
                values.append(squared**.5)
            norms.append(values)
            image_ids.append(batch['image_id'].detach().cpu().tolist())
            print('V49_CALIBRATION_BATCH', index, values, flush=True)
            del losses, gradients
            if len(norms) == module.config.calibration_batches:
                break
        if len(norms) != module.config.calibration_batches:
            raise ValueError('Incomplete V49 gradient calibration')
        if any(not torch.equal(value.detach().cpu(), initial[name]) for name, value in module.model.state_dict().items()):
            raise AssertionError('V49 calibration modified model state')
        module.set_calibration(norms, dict(seed=seed, image_ids=image_ids, task_names=list(TASK_NAMES),
            policy='median_shared_two_layer_decoder_L2;TRAIN_only;no_optimizer_update'))
        return module.calibration_provenance
    finally:
        module.model.load_state_dict(initial, strict=True)
        module.train(mode)
        restore_rng(rng)

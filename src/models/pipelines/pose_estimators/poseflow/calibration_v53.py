import random

import numpy as np
import torch

from .calibration_v49_balanced import capture_rng, restore_rng
from .losses_v53 import TASK_NAMES


def fixed_scales(norms):
    values = torch.as_tensor(norms, dtype=torch.float64)
    if (values.ndim != 2 or values.shape[1] != 2 or not len(values) or not torch.isfinite(values).all()
            or (values < 0).any() or (values.max(0).values == 0).any()):
        raise ValueError('Invalid V53 birth task gradient norms')
    return values.quantile(.5, dim=0).clamp_min(1e-8).float()


def calibrate(module, batches):
    if module.calibration_ready:
        raise RuntimeError('V53 calibration is already frozen')
    rng, mode = capture_rng(), module.training
    initial = {name: value.detach().cpu().clone() for name, value in module.model.state_dict().items()}
    norms, image_ids = [], []
    try:
        seed = module.config.calibration_seed
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
                values.append(sum(float(gradient.detach().double().square().sum()) for gradient in gradients if gradient is not None)**.5)
            norms.append(values)
            image_ids.append(batch['image_id'].detach().cpu().tolist())
            print('V53_CALIBRATION_BATCH', index, values, flush=True)
            del losses, gradients
            if len(norms) == module.config.calibration_batches:
                break
        if len(norms) != module.config.calibration_batches:
            raise ValueError('Incomplete V53 TRAIN calibration')
        if any(not torch.equal(value.detach().cpu(), initial[name]) for name, value in module.model.state_dict().items()):
            raise AssertionError('V53 calibration modified model state')
        module.set_calibration(norms, dict(seed=seed, image_ids=image_ids, task_names=list(TASK_NAMES),
            policy='median_shared_native_pose_layers_L2;TRAIN_only;no_optimizer_update'))
        return module.calibration_provenance
    finally:
        module.model.load_state_dict(initial, strict=True)
        module.train(mode)
        restore_rng(rng)

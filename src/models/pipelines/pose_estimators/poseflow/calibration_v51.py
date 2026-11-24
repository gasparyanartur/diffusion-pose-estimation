from copy import deepcopy

import torch

from .calibration_v49_balanced import capture_rng, restore_rng


def calibrate(module, batches):
    if module.calibration_ready:
        raise RuntimeError('V51 calibration is already initialized')
    rng, mode = capture_rng(), module.training
    norms, image_ids = [], []
    try:
        torch.manual_seed(module.config.calibration_seed)
        module.train()
        for index,batch in enumerate(batches):
            losses = module._forward_losses(batch,torch.Generator().manual_seed(module.config.calibration_seed+index),
                                            torch.ones_like(module.gradient_scales))
            values = module.measure_gradients(losses).tolist()
            norms.append(values)
            image_ids.append(batch['image_id'].tolist())
            print('V51_CALIBRATION',index,values,flush=True)
            del losses
            if len(norms) == module.config.calibration_batches:
                break
        module.set_gradient_calibration(norms,dict(seed=module.config.calibration_seed,image_ids=image_ids,
            policy='TRAIN_only;four_shared_probe_blocks;no_optimizer_update'))
        return deepcopy(module.calibration_provenance)
    finally:
        module.train(mode)
        restore_rng(rng)

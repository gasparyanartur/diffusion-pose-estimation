from copy import deepcopy

import torch

from .training_v44 import PoseFlowV44Pipeline, PoseFlowV44PipelineConfig


def warmstart_comparison(record, *, prediction_type, max_epochs=10):
    if prediction_type not in ('x0', 'v') or record.get('v44_prediction_type', 'x0') != 'x0':
        raise ValueError('Comparison warmstart requires an explicit x0 source and valid target mode')
    if record.get('v44_contract') != dict(version=1, posterior_mode='mean', state='latent', fixed_assignment=True):
        raise ValueError('Comparison source does not satisfy the V44 contract')
    source = record['hyper_parameters']['config']
    config = PoseFlowV44PipelineConfig(**dict(source, prediction_type=prediction_type,
        lr=2e-5, lr_backbone=1e-5))
    module = PoseFlowV44Pipeline(config=config, max_epochs=max_epochs)
    module.load_state_dict(record['state_dict'], strict=True)
    module.codec_provenance = deepcopy(record['v44_codec_provenance'])
    if prediction_type == 'v':
        with torch.no_grad():
            for head in module.model.latent_heads:
                head.weight.neg_()
                head.bias.neg_()
    return module

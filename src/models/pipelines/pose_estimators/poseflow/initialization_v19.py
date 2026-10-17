from pathlib import Path

import torch

from ..diffusionregpose.training import (
    adapt_checkpoint_deformable_attention_levels, remap_diffusionregpose_checkpoint_layers,
)


def checkpoint_transfer_coverage(model, config):
    checkpoint = torch.load(config.pretrained_checkpoint, map_location='cpu', weights_only=False)
    raw = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
    raw = {name.removeprefix('module.'):value for name,value in raw.items() if torch.is_tensor(value)}
    if config.pretrained_keep_decoder_layers is not None:
        raw = remap_diffusionregpose_checkpoint_layers(raw,
            encoder_layer_indices=config.pretrained_keep_encoder_layers,
            decoder_layer_indices=config.pretrained_keep_decoder_layers,
            target_num_encoder_layers=config.num_encoder_layers,
            target_num_decoder_layers=config.num_decoder_layers,
            target_num_box_decoder_layers=config.num_box_decoder_layers)
    state = model.state_dict()
    source_view = {source:state[target] for target,source in model.transfer_sources.items()}
    raw, _ = adapt_checkpoint_deformable_attention_levels(source_view, raw, num_heads=config.num_heads)
    verified, fresh = {}, []
    for target, value in state.items():
        source = model.transfer_sources.get(target)
        if source not in raw:
            fresh.append(target)
            continue
        expected = raw[source]
        if '.class_head.' in target:
            row = model.initialization_report['person_class_row']
            expected = expected[row:row+1]
        if expected.shape != value.shape or not torch.equal(expected.cpu(), value.cpu()):
            raise ValueError(f'V19 checkpoint transfer mismatch: {source} -> {target}')
        verified[target] = value.numel()
    parameters = dict(model.named_parameters())
    elements = sum(value.numel() for value in state.values())
    parameter_count = sum(value.numel() for value in parameters.values())
    return {'checkpoint':str(Path(config.pretrained_checkpoint).resolve()),
            'verified_checkpoint_tensors':len(verified),
            'verified_checkpoint_elements':sum(verified.values()),
            'verified_checkpoint_fraction':sum(verified.values())/elements,
            'verified_checkpoint_parameter_fraction':sum(v for k,v in verified.items() if k in parameters)/parameter_count,
            'fresh_state_keys':fresh}

from __future__ import annotations

from contextlib import contextmanager

import torch


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [cpu_copy(item) for item in value]
    return value


def reconstruct_sampling(references, offsets, logits, spatial_shapes):
    # offsets: <B, Q, H, L, P, 2>; logits: <B, Q, H, L, P>
    weights = logits.flatten(-2).softmax(-1).reshape_as(logits)
    if references.shape[-1] == 4:
        locations = (references[:, :, None, :, None, :2]
                     + offsets / offsets.shape[-2]
                     * references[:, :, None, :, None, 2:] * .5)
    elif references.shape[-1] == 2:
        normalizer = spatial_shapes.flip(-1)
        locations = references[:, :, None, :, None] + offsets / normalizer[None, None, None, :, None]
    else:
        raise ValueError('references must have XY or XYWH coordinates')
    return locations, weights


def query_lineage(call, layer_index, person, joint=None):
    if not 0 <= person < len(call['selected_boxes']):
        raise ValueError('person group is out of range')
    if layer_index < call['num_box_layers']:
        if joint is not None:
            raise ValueError('joint queries do not exist in box-only layers')
        return int(call['selected_boxes'][person])
    if joint is not None and not 0 <= joint < call['num_joints']:
        raise ValueError('joint is out of range')
    return person * (call['num_joints'] + 1) + (0 if joint is None else joint + 1)


@contextmanager
def capture_sampling(model):
    """Observe native sample_keypoints without replacing forwards or activations."""
    if model.training:
        raise ValueError('capture requires model.eval()')
    calls, handles = [], []
    decoder = model.transformer.decoder

    def begin_call(module, args, kwargs):
        if args[0].shape[0] != 1:
            raise ValueError('inspection captures one image at a time')
        calls.append({
            'time': int(args[2][0]), 'state': cpu_copy(args[1][0]),
            'num_joints': decoder.num_body_points,
            'num_box_layers': decoder.num_box_decoder_layers,
            'layers': [{} for _ in decoder.layers],
        })

    def finish_call(module, args, kwargs, output):
        for layer, prediction in zip(calls[-1]['layers'], [*output['aux_outputs'], output]):
            layer['prediction'] = {key: cpu_copy(prediction[key][0]) for key in
                                   ('pred_logits', 'pred_boxes', 'pred_keypoints')}

    def finish_decoder(module, args, kwargs, output):
        calls[-1]['references_after'] = cpu_copy([item[0] for item in output[1][1:]])

    def layer_hooks(index):
        def before(module, args, kwargs):
            layer = calls[-1]['layers'][index]
            layer['references'] = cpu_copy(kwargs['raw_reference_points'][:, 0])
            mask = kwargs['self_attn_mask']
            layer['mask'] = None if mask is None else cpu_copy(mask[0] if mask.ndim == 3 else mask)
            if mask is not None and mask.ndim == 3:
                if not torch.equal(mask, mask[0].expand_as(mask)):
                    raise ValueError('inspection expects identical masks across heads')

        def after(module, args, kwargs, output):
            if index == decoder.num_box_decoder_layers - 1:
                scores = decoder.class_embed[index](output).max(-1).values
                selected = scores.topk(decoder.num_group, dim=0).indices[:, 0]
                calls[-1]['selected_boxes'] = cpu_copy(selected)

        return before, after

    def cross_hooks(index):
        def before(module, args):
            layer = calls[-1]['layers'][index]
            layer['scaled_references'] = cpu_copy(args[1])
            layer['spatial_shapes'] = cpu_copy(args[3])
            layer['padding_mask'] = cpu_copy(args[5][0]) if args[5] is not None else None

        def offsets(module, args, output):
            cross = decoder.layers[index].cross_attn
            calls[-1]['layers'][index]['offsets'] = cpu_copy(output.reshape(
                1, -1, cross.num_heads, cross.num_levels, cross.num_points, 2))

        def logits(module, args, output):
            cross = decoder.layers[index].cross_attn
            calls[-1]['layers'][index]['attention_logits'] = cpu_copy(output.reshape(
                1, -1, cross.num_heads, cross.num_levels, cross.num_points))

        def after(module, args, output):
            layer = calls[-1]['layers'][index]
            locations, weights = reconstruct_sampling(
                layer.pop('scaled_references'), layer.pop('offsets'),
                layer.pop('attention_logits'), layer['spatial_shapes'])
            layer['locations'], layer['weights'] = locations[0], weights[0]

        return before, offsets, logits, after

    try:
        handles.append(model.register_forward_pre_hook(begin_call, with_kwargs=True))
        handles.append(model.register_forward_hook(finish_call, with_kwargs=True))
        handles.append(decoder.register_forward_hook(finish_decoder, with_kwargs=True))
        for index, layer in enumerate(decoder.layers):
            before, after = layer_hooks(index)
            handles.append(layer.register_forward_pre_hook(before, with_kwargs=True))
            handles.append(layer.register_forward_hook(after, with_kwargs=True))
            before_cross, offsets, logits, after_cross = cross_hooks(index)
            handles.append(layer.cross_attn.register_forward_pre_hook(before_cross))
            handles.append(layer.cross_attn.sampling_offsets.register_forward_hook(offsets))
            handles.append(layer.cross_attn.attention_weights.register_forward_hook(logits))
            handles.append(layer.cross_attn.register_forward_hook(after_cross))
        yield calls
    finally:
        for handle in handles:
            handle.remove()

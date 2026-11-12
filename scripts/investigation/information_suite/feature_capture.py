from __future__ import annotations

from contextlib import contextmanager
from types import MethodType

import numpy as np
import torch
from torch.nn import functional as F

from evaluate_pilot import capture_features
from scoring import box_overlap


def sample_finest_map(memory, spatial_shapes, pixel_xy):
    height, width = map(int, spatial_shapes[0].tolist())
    feature = memory[:, :height*width].reshape(1, height, width, -1).permute(0, 3, 1, 2)
    grid = memory.new_tensor([[[[2*pixel_xy[0]/384-1, 2*pixel_xy[1]/256-1]]]])
    return F.grid_sample(feature, grid, mode='bilinear', padding_mode='border', align_corners=False)[0, :, 0, 0]


@contextmanager
def capture_maps(model, pixel_xy):
    prepare, encode = model.prepare_transformer_inputs, model.encode_features
    capture = {}
    def prepared(self, *args, **kwargs):
        inputs = prepare(*args, **kwargs)
        capture['B'] = sample_finest_map(inputs.memory, inputs.spatial_shapes, pixel_xy).detach().cpu().numpy().copy()
        return inputs
    def encoded(self, inputs, *args, **kwargs):
        output = encode(inputs, *args, **kwargs)
        capture['E'] = sample_finest_map(output, inputs.spatial_shapes, pixel_xy).detach().cpu().numpy().copy()
        return output
    model.prepare_transformer_inputs = MethodType(prepared, model)
    model.encode_features = MethodType(encoded, model)
    try:
        with capture_features(model) as decoder:
            yield capture, decoder
    finally:
        model.prepare_transformer_inputs, model.encode_features = prepare, encode


def normalized_boxes_to_pixels(boxes):
    boxes = np.asarray(boxes)
    return np.concatenate([boxes[..., :2]-boxes[..., 2:]/2,
                           boxes[..., :2]+boxes[..., 2:]/2], axis=-1)*[384, 256, 384, 256]


def select_features(capture, decoder, output, gt_box, target_joint):
    boxes = normalized_boxes_to_pixels(output['pred_boxes'][0].detach().cpu().numpy())
    overlaps = box_overlap([gt_box], boxes)[0]
    slot = int(overlaps.argmax())
    stage_boxes = normalized_boxes_to_pixels(output['aux_outputs'][-1]['pred_boxes'][0].detach().cpu().numpy())
    stage_overlap = float(box_overlap([gt_box], stage_boxes)[0, slot])
    selected = dict(capture)
    for stage in (1, 2):
        geometry = decoder[f'geometry_{stage}'][0].reshape(50, 15, 256)
        selected[f'G{stage}'] = geometry[slot, 1+target_joint]
        if stage == 2:
            selected['G2_elbow'] = geometry[slot, 3]
            selected['G2_shoulder'] = geometry[slot, 1]
    for source in ('h_initial', 'h_written_0', 'h_written_1'):
        if source in decoder:
            selected[source] = decoder[source][0, slot]
    pose = output['pred_keypoints'][0, slot, :28].detach().cpu().numpy()
    box = output['pred_boxes'][0, slot].detach().cpu().numpy()
    score = output['pred_logits'][0, slot].sigmoid().amax().detach().cpu().numpy().reshape(1)
    selected['native_prediction'] = np.r_[pose, box, score]
    matched = bool(overlaps[slot] >= .5 and stage_overlap >= .5)
    selected = {name:np.array(value, copy=True) for name, value in selected.items()}
    return selected, dict(slot=slot, final_box_iou=float(overlaps[slot]), stage1_box_iou=stage_overlap, matched=matched)

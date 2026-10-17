import torch
from torchvision.ops import box_iou

from ..diffusionregpose.evaluation import box_cxcywh_to_xyxy
from .quick_eval_v13 import raw_pose_quality


def box_quality(output, batch):
    valid = batch.get('person_mask', batch['gt_labeled'].any(-1))[0].bool()
    targets = batch['gt_boxes_canon'][0, valid]
    boxes = output['pred_boxes'][0]
    scores = output['pred_logits'][0].sigmoid().flatten()
    overlap = box_iou(box_cxcywh_to_xyxy(boxes), box_cxcywh_to_xyxy(targets))
    best_target = overlap.amax(0) if targets.numel() else boxes.new_empty(0)
    best_candidate = overlap.amax(1) if targets.numel() else scores.new_zeros(scores.shape)
    good = best_candidate >= .5
    best_ids = overlap.argmax(1)[good] if targets.numel() else torch.empty(0, device=boxes.device)
    excess_duplicates = max(int(good.sum())-best_ids.unique().numel(), 0)
    return {'people': len(targets), 'covered50': int((best_target >= .5).sum()),
            'covered75': int((best_target >= .75).sum()), 'candidates': len(scores),
            'duplicates50': excess_duplicates,
            'good_count': int(good.sum()), 'poor_count': int((~good).sum()),
            'good_score_sum': float(scores[good].sum()), 'poor_score_sum': float(scores[~good].sum())}


def aggregate_box_quality(rows):
    totals = {key: sum(row[key] for row in rows) for key in rows[0]}
    return {'raw_bbox_coverage50': totals['covered50']/max(totals['people'], 1),
            'raw_bbox_coverage75': totals['covered75']/max(totals['people'], 1),
            'raw_bbox_duplicate_fraction50': totals['duplicates50']/max(totals['candidates'], 1),
            'raw_good_bbox_score_mean': totals['good_score_sum']/max(totals['good_count'], 1),
            'raw_poor_bbox_score_mean': totals['poor_score_sum']/max(totals['poor_count'], 1),
            'raw_good_bbox_count': totals['good_count'], 'raw_poor_bbox_count': totals['poor_count']}


def step_quality(output, batch, sigmas):
    rows = []
    for index, step in enumerate(output['latent_step_diagnostics']):
        layers = []
        for prediction in step['predictions']:
            oks, covered, people = raw_pose_quality(prediction, batch, sigmas)
            layers.append({**aggregate_box_quality([box_quality(prediction, batch)]),
                           'oracle_oks': oks/max(people, 1), 'coverage75': covered/max(people, 1)})
        rows.append({'step': index, 'time': step['time'], 'latent_update_rms': float(step['update_rms']),
                     'latent_rms': float(step['latent_state'].float().square().mean().sqrt()),
                     'layers_quality': layers,
                     'layers_diagnostics': [{key: float(value) if torch.is_tensor(value) else value
                                             for key, value in row.items()} for row in step['layers']]})
    return rows

from __future__ import annotations

import builtins
from contextlib import contextmanager, redirect_stdout
from io import StringIO
import math

import numpy as np
import torch

from src.models.pipelines.pose_estimators.diffusionregpose.evaluation import (
    _crowdpose_pose_nms_keep,
    _select_crowdpose_candidates,
    postprocess_crowdpose,
)


def rank_images(rows, count=3):
    defined = [row for row in rows if row['AP'] is not None and math.isfinite(row['AP'])]
    undefined = sorted(row['image_id'] for row in rows if row not in defined)
    best = sorted(defined, key=lambda row: (-row['AP'], row['image_id']))[:count]
    worst = sorted(defined, key=lambda row: (row['AP'], row['image_id']))[:count]
    return {'best': [row['image_id'] for row in best],
            'worst': [row['image_id'] for row in worst], 'undefined': undefined}


@contextmanager
def crowdpose_compatibility():
    missing_unicode = not hasattr(builtins, 'unicode')
    missing_float = not hasattr(np, 'float')
    if missing_unicode:
        builtins.unicode = str
    if missing_float:
        np.float = float
    try:
        with redirect_stdout(StringIO()):
            yield
    finally:
        if missing_unicode:
            del builtins.unicode
        if missing_float:
            del np.float


def evaluate_records(annotation_path, records, image_ids, *, per_image=True):
    from crowdposetools.coco import COCO
    from crowdposetools.cocoeval import COCOeval

    with crowdpose_compatibility():
        ground_truth = COCO(str(annotation_path))
        selected = set(image_ids)
        ground_truth.dataset['images'] = [item for item in ground_truth.dataset['images'] if item['id'] in selected]
        ground_truth.dataset['annotations'] = [item for item in ground_truth.dataset['annotations'] if item['image_id'] in selected]
        ground_truth.createIndex()
        if records:
            detected = ground_truth.loadRes(records)
        else:
            detected = COCO()
            detected.dataset = {'images': ground_truth.dataset['images'],
                                'categories': ground_truth.dataset['categories'], 'annotations': []}
            detected.createIndex()

        def measure(ids, kind='keypoints'):
            evaluator = COCOeval(ground_truth, detected, kind)
            evaluator.params.imgIds = list(ids)
            evaluator.evaluate()
            evaluator.accumulate()
            # precision: <threshold, recall, category, area, maxDet>; all-area AP.
            precision = evaluator.eval['precision'][:, :, :, 0, -1]
            valid = precision[precision >= 0]
            return float(valid.mean()) if valid.size else None

        aggregate = {'AP': measure(image_ids), 'bbox/AP': measure(image_ids, 'bbox')}
        rows = [{'image_id': image_id, 'AP': measure([image_id])} for image_id in image_ids] if per_image else []
    return aggregate, rows


def postprocess_with_queries(output, original_hw, resized_hw):
    """Carry original query IDs alongside the unchanged native postprocessor."""
    device = output['pred_logits'].device
    sizes = torch.tensor([original_hw], device=device, dtype=torch.float32)
    nms_sizes = torch.tensor([resized_hw], device=device, dtype=torch.float32)
    result = postprocess_crowdpose(output, sizes, nms_target_sizes=nms_sizes)[0]
    scores, _, _, _, nms_points, queries = _select_crowdpose_candidates(
        output, sizes, num_body_points=14, nms_target_sizes=nms_sizes)
    keep = _crowdpose_pose_nms_keep(nms_points[0], scores[0], 14)
    kept_queries = queries[0, keep]
    logits = output['pred_logits'][0, kept_queries]
    indices = logits.sigmoid().flatten().topk(len(keep), sorted=True).indices
    result['query_ids'] = kept_queries[indices // logits.shape[-1]]
    return result

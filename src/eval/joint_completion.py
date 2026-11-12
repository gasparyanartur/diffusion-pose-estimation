from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np


def select_annotations(data: dict, image_ids: list[int]) -> dict[int, list[dict]]:
    if len(set(image_ids)) != len(image_ids):
        raise ValueError('duplicate image IDs')
    known_images = [image['id'] for image in data['images']]
    if len(set(known_images)) != len(known_images):
        raise ValueError('duplicate image IDs in annotation file')
    if not set(image_ids).issubset(known_images):
        raise ValueError('unknown image IDs')
    selected = {image_id: [] for image_id in image_ids}
    seen = set()
    for person in data['annotations']:
        if person['id'] in seen:
            raise ValueError('duplicate annotation ID')
        seen.add(person['id'])
        if (person['image_id'] in selected and not person.get('iscrowd', 0)
                and person.get('num_keypoints', 0) > 0):
            selected[person['image_id']].append(person)
    return {image_id: sorted(people, key=lambda person: person['id'])
            for image_id, people in selected.items()}


def joint_errors(predicted_xy, original_keypoints, bbox, sigmas) -> dict[str, np.ndarray]:
    """Per-joint diagnostics in original pixels; unlabeled entries remain undefined."""
    predicted = np.asarray(predicted_xy, dtype=np.float64)
    truth = np.asarray(original_keypoints, dtype=np.float64).reshape(-1, 3)
    box = np.asarray(bbox, dtype=np.float64)
    sigmas = np.asarray(sigmas, dtype=np.float64)
    if (box.shape != (4,) or not np.isfinite(box).all() or (box[2:] <= 0).any()):
        raise ValueError('box must have finite, positive width and height')
    if predicted.shape != (len(truth), 2) or sigmas.shape != (len(truth),):
        raise ValueError('joint shapes do not match')
    if not np.isfinite(predicted).all() or not np.isfinite(truth).all():
        raise ValueError('coordinates must be finite')
    if not np.isin(truth[:, 2], [0, 1, 2]).all() or (sigmas <= 0).any():
        raise ValueError('invalid flags or sigmas')
    squared_error = np.square(predicted - truth[:, :2]).sum(axis=-1)
    error = np.sqrt(squared_error)
    normalized = error / box[2:].max()
    # CrowdPose's evaluator uses bbox area * .53, regardless of stored annotation area.
    oks = np.exp(-squared_error / np.square(2 * sigmas) / (box[2] * box[3] * .53) / 2)
    missing = truth[:, 2] == 0
    for values in (error, normalized, oks):
        values[missing] = np.nan
    return dict(pixel_error=error, normalized_error=normalized, oks=oks)


def unflip_keypoints(predicted_xy, image_width: int, flip_indices) -> np.ndarray:
    restored = np.asarray(predicted_xy, dtype=np.float64)[..., flip_indices, :].copy()
    restored[..., 0] = image_width - 1 - restored[..., 0]
    return restored


def build_prediction_record(annotation: dict, predicted_xy, scores, flipped_xy,
                            teacher_id: str) -> dict[str, Any]:
    predicted = np.asarray(predicted_xy, dtype=np.float64)
    flipped = np.asarray(flipped_xy, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    joint_count = len(annotation['keypoints']) // 3
    if (predicted.shape != (joint_count, 2) or flipped.shape != predicted.shape
            or scores.shape != (joint_count,)):
        raise ValueError('prediction shapes do not match source annotations')
    if not all(np.isfinite(values).all() for values in (predicted, flipped, scores)):
        raise ValueError('predictions and scores must be finite')
    return dict(image_id=annotation['image_id'], annotation_id=annotation['id'],
                original_annotation=deepcopy(annotation), predicted_xy=predicted.tolist(),
                keypoint_scores=scores.tolist(), flipped_back_xy=flipped.tolist(),
                teacher_id=teacher_id)


def summarize_predictions(records: list[dict], joint_names: list[str], sigmas,
                          bootstrap_draws: int = 500, seed: int = 42) -> dict:
    if not records or bootstrap_draws < 1:
        raise ValueError('records and positive bootstrap_draws required')
    identities = [(record['image_id'], record['annotation_id']) for record in records]
    if len(set(identities)) != len(identities):
        raise ValueError('duplicate prediction identities')
    truth = np.asarray([record['original_annotation']['keypoints'] for record in records])
    truth = truth.reshape(len(records), len(joint_names), 3)  # <people, joints, xyz/flag>
    errors = [joint_errors(record['predicted_xy'], record['original_annotation']['keypoints'],
                           record['original_annotation']['bbox'], sigmas) for record in records]
    normalized = np.stack([error['normalized_error'] for error in errors])
    oks = np.stack([error['oks'] for error in errors])
    pixels = np.stack([error['pixel_error'] for error in errors])
    scores = np.asarray([record['keypoint_scores'] for record in records])
    predicted = np.asarray([record.get('single_pass_xy', record['predicted_xy']) for record in records])
    flipped = np.asarray([record['flipped_back_xy'] for record in records])
    boxes = np.asarray([record['original_annotation']['bbox'] for record in records])
    flip_error = np.linalg.norm(predicted - flipped, axis=-1) / boxes[:, None, 2:].max(axis=-1)
    image_ids, image_indices = np.unique([record['image_id'] for record in records],
                                         return_inverse=True)
    rng = np.random.default_rng(seed)
    sampled_images = rng.integers(len(image_ids), size=(bootstrap_draws, len(image_ids)))

    def describe(mask: np.ndarray, intervals: bool = False) -> dict:
        count = int(mask.sum())
        output = dict(count=count, mean_pixel_error=None, mean_normalized_error=None,
                      median_normalized_error=None, pck_005=None, pck_010=None, mean_oks=None)
        if not count:
            return output
        output.update(mean_pixel_error=float(pixels[mask].mean()),
                      mean_normalized_error=float(normalized[mask].mean()),
                      median_normalized_error=float(np.median(normalized[mask])))
        values = dict(pck_005=(normalized <= .05).astype(float),
                      pck_010=(normalized <= .10).astype(float), mean_oks=oks)
        image_counts = np.bincount(image_indices, weights=mask.sum(axis=1),
                                   minlength=len(image_ids))
        denominators = image_counts[sampled_images].sum(axis=1)
        for name, value in values.items():
            output[name] = float(value[mask].mean())
            if intervals:
                sums = np.bincount(image_indices, weights=np.where(mask, value, 0).sum(axis=1),
                                   minlength=len(image_ids))
                estimates = sums[sampled_images].sum(axis=1)[denominators > 0]
                estimates /= denominators[denominators > 0]
                output[name + '_ci95'] = np.quantile(estimates, [.025, .975]).tolist()
        return output

    labeled = truth[..., 2] > 0
    groups = dict(labeled=labeled, flag1=truth[..., 2] == 1, flag2=truth[..., 2] == 2)
    if len(joint_names) == 14:
        limbs = np.arange(14)[None, :] < 12
        groups.update(labeled_limbs=labeled & limbs, flag1_limbs=groups['flag1'] & limbs,
                      flag2_limbs=groups['flag2'] & limbs, head_neck=labeled & ~limbs)
    output = dict(people=len(records), images=len(image_ids),
                  bootstrap=dict(unit='image', draws=bootstrap_draws, seed=seed),
                  groups={name: describe(mask, True) for name, mask in groups.items()})
    output['per_joint'] = {
        name: {group: describe(mask & (np.arange(len(joint_names))[None, :] == joint), True)
               for group, mask in list(groups.items())[:3]}
        for joint, name in enumerate(joint_names)}
    missing = truth[..., 2] == 0
    output['missing'] = dict(count=int(missing.sum()), accuracy=None,
                             mean_score=float(scores[missing].mean()) if missing.any() else None,
                             flip_error_quantiles=np.quantile(flip_error[missing], [.5, .9, .95]).tolist()
                             if missing.any() else None)
    output['confidence_curve'] = []
    for threshold in [0, .1, .2, .3, .4, .5, .6, .7, .8]:
        accepted = scores >= threshold
        output['confidence_curve'].append(dict(
            threshold=threshold, labeled=describe(labeled & accepted),
            missing_count=int((missing & accepted).sum()),
            missing_coverage=float((missing & accepted).sum() / missing.sum()) if missing.any() else None))
    output['consistency_curve'] = []
    for threshold in [.02, .05, .1]:
        accepted = flip_error <= threshold
        output['consistency_curve'].append(dict(
            max_flip_error=threshold, labeled=describe(labeled & accepted),
            missing_count=int((missing & accepted).sum()),
            missing_coverage=float((missing & accepted).sum() / missing.sum()) if missing.any() else None))
    return output

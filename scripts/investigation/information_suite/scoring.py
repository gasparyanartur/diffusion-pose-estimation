import numpy as np


def box_overlap(gt_xywh, predicted_xyxy):
    gt = np.asarray(gt_xywh, dtype=float)
    predicted = np.asarray(predicted_xyxy, dtype=float).reshape(-1, 4)
    corners = np.concatenate((gt[:, :2], gt[:, :2]+gt[:, 2:]), axis=1)
    low = np.maximum(corners[:, None, :2], predicted[None, :, :2])
    high = np.minimum(corners[:, None, 2:], predicted[None, :, 2:])
    intersection = np.maximum(high-low, 0).prod(-1)
    area = np.maximum(predicted[:, 2:]-predicted[:, :2], 0).prod(-1)
    return intersection / np.maximum(gt[:, 2:].prod(-1)[:, None]+area[None]-intersection, 1e-9)


def score_people(result, keypoints, gt_boxes):
    scores, labels = np.asarray(result['scores']), np.asarray(result['labels'])
    candidates = np.asarray(result['keypoints']).reshape(-1, 14, 3)
    overlaps = box_overlap(gt_boxes, result['boxes'])
    owners = overlaps.argmax(0) if len(scores) else np.zeros(0, dtype=int)
    records = []
    for person, (truth, box) in enumerate(zip(np.asarray(keypoints), gt_boxes)):
        eligible = np.flatnonzero((owners == person) & (overlaps[person] >= .5) & (scores >= .5) & (labels == 1))
        record = {'person': person, 'detected': bool(len(eligible)), 'candidate_count': int(len(eligible)),
                  'tau': .04*float(np.sqrt(box[2]*box[3])), 'wrist_correct': False}
        if len(eligible):
            index = eligible[np.argmax(scores[eligible])]
            errors = np.linalg.norm(candidates[index, :, :2]-truth, axis=-1)
            record.update(score=float(scores[index]), iou=float(overlaps[person, index]),
                          wrist_error_pixels=float(errors[4]), wrist_correct=bool(errors[4] <= record['tau']),
                          mean_common_joint_error_pixels=float(errors[:12].mean()),
                          predicted_wrist=candidates[index, 4, :2].tolist(), candidate_index=int(index))
        records.append(record)
    return records

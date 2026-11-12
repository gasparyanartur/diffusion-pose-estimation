from copy import deepcopy
import json

import numpy as np
import pytest

from src.eval.joint_completion import (
    build_prediction_record,
    joint_errors,
    select_annotations,
    summarize_predictions,
    unflip_keypoints,
)


def annotation(annotation_id=5, image_id=9):
    return dict(id=annotation_id, image_id=image_id, bbox=[10, 20, 100, 200],
                keypoints=[30, 40, 2, 50, 60, 1, 0, 0, 0],
                num_keypoints=2, iscrowd=0)


def test_masked_error_arithmetic_and_crowdpose_oks():
    truth = np.array([[30, 40, 2], [50, 60, 1], [0, 0, 0]])
    predicted = np.array([[36, 48], [50, 60], [1000, 1000]])
    errors = joint_errors(predicted, truth, [10, 20, 100, 200], np.ones(3) * .1)
    np.testing.assert_allclose(errors['normalized_error'][:2], [.05, 0])
    expected = np.exp(-100 / ((2 * .1) ** 2) / (100 * 200 * .53) / 2)
    np.testing.assert_allclose(errors['oks'][:2], [expected, 1])
    assert np.isnan(errors['normalized_error'][2])
    assert np.isnan(errors['oks'][2])
    with pytest.raises(ValueError, match='box'):
        joint_errors(predicted, truth, [10, 20, 0, 200], np.ones(3) * .1)


def test_identity_order_and_duplicate_rejection():
    annotations = [annotation(7), annotation(5), annotation(8, 10),
                   dict(annotation(6), num_keypoints=0),
                   dict(annotation(4), iscrowd=1)]
    data = dict(images=[dict(id=9), dict(id=10)], annotations=annotations)
    selected = select_annotations(data, [9])
    assert [person['id'] for person in selected[9]] == [5, 7]
    data['annotations'].append(annotation(5))
    with pytest.raises(ValueError, match='duplicate'):
        select_annotations(data, [9])


def test_flip_and_native_inverse_affine_nonsquare_roundtrip():
    from mmpose.structures.bbox import get_warp_matrix
    import cv2

    predicted = np.array([[[11., 43.], [99., 18.], [190., 80.]]])
    flipped = unflip_keypoints(predicted, 317, [1, 0, 2])
    np.testing.assert_allclose(unflip_keypoints(flipped, 317, [1, 0, 2]), predicted)
    np.testing.assert_allclose(flipped[0, :, 0], [217, 305, 126])
    center, scale, size = np.array([91., 147.]), np.array([180., 240.]), (192, 256)
    matrix = get_warp_matrix(center, scale, 0, size)
    crop = cv2.transform(predicted.astype(np.float32), matrix)
    restored = crop / np.array(size) * scale + center - .5 * scale
    np.testing.assert_allclose(restored, predicted, atol=2e-5)


def test_serialized_record_preserves_labels_and_missing_flags():
    original = annotation()
    before = deepcopy(original)
    predicted = np.array([[36, 48], [50, 60], [101, -5]])
    record = build_prediction_record(original, predicted, [.9, .8, .7],
                                     predicted + 2, 'teacher-hash')
    serialized = json.loads(json.dumps(record, allow_nan=False))
    assert original == before
    assert serialized['original_annotation'] == before
    assert serialized['predicted_xy'][2] == [101, -5]
    assert serialized['original_annotation']['keypoints'][8] == 0
    assert serialized['teacher_id'] == 'teacher-hash'
    with pytest.raises(ValueError, match='finite'):
        build_prediction_record(original, predicted * np.nan, [.9, .8, .7],
                                predicted, 'teacher-hash')


def test_summary_empty_strata_and_image_cluster_bootstrap():
    original = annotation()
    predicted = np.array([[30, 40], [50, 60], [999, 999]])
    record = build_prediction_record(original, predicted, [.9, .8, .7],
                                     predicted, 'teacher-hash')
    record['single_pass_xy'] = (predicted + 30).tolist()
    record['flipped_back_xy'] = (predicted + 30).tolist()
    summary = summarize_predictions([record], ['a', 'b', 'c'], np.ones(3) * .1,
                                    bootstrap_draws=30)
    assert summary['groups']['labeled']['count'] == 2
    assert summary['groups']['labeled']['pck_005'] == 1
    assert summary['groups']['labeled']['pck_005_ci95'] == [1, 1]
    assert summary['per_joint']['c']['labeled']['count'] == 0
    assert summary['per_joint']['c']['labeled']['pck_005'] is None
    assert summary['missing']['count'] == 1
    assert summary['missing']['accuracy'] is None
    assert summary['consistency_curve'][0]['missing_coverage'] == 1


def test_native_flip_boxes_and_independent_consistency(monkeypatch):
    from types import SimpleNamespace
    from scripts.evaluation import evaluate_joint_completion as evaluation

    calls = []
    model = SimpleNamespace(test_cfg={'flip_test': True}, dataset_meta={'flip_indices': [1, 0, 2]})

    def predict(model, image, boxes, batch_size):
        calls.append((np.asarray(boxes).copy(), model.test_cfg['flip_test']))
        return np.zeros((1, 3, 2)), np.ones((1, 3))

    monkeypatch.setattr(evaluation, 'predict_people', predict)
    evaluation.infer_image(model, np.zeros((251, 317, 3), dtype=np.uint8), [annotation()], 8)
    np.testing.assert_allclose(calls[2][0][0], [206, 20, 100, 200])
    assert [call[1] for call in calls] == [True, False, False]
    assert model.test_cfg['flip_test'] is True

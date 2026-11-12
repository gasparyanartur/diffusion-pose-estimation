import numpy as np
import pytest

from scripts.investigation.information_suite.geometry import (
    align_rotation, camera_project, depth_for_length, patch_union, rigid_arm_pose,
)


def test_rotation_alignment_and_antiparallel_case():
    for target in ([0, 1, 0], [-1, 0, 0], [1, 0, 0]):
        rotation = align_rotation(np.array([1., 0, 0]), np.array(target, dtype=float))
        np.testing.assert_allclose(rotation @ [1., 0, 0], target, atol=1e-8)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)
        assert np.linalg.det(rotation) == pytest.approx(1)


def test_depth_preserves_length_and_rejects_impossible_projection():
    depth = depth_for_length(35., np.array([25., 20.]))
    assert np.linalg.norm([25., 20., depth]) == pytest.approx(35.)
    with pytest.raises(ValueError):
        depth_for_length(30., np.array([25., 20.]))


def test_arm_rotations_place_both_joints_exactly():
    rest_upper = np.array([.3, 0., 0.])
    rest_forearm = np.array([.25, 0., 0.])
    upper = np.array([.1, -.2, np.sqrt(.04)])
    forearm = np.array([-.1, -.1, np.sqrt(.0425)])
    shoulder, elbow = rigid_arm_pose(rest_upper, rest_forearm, upper, forearm)
    np.testing.assert_allclose(shoulder @ rest_upper, upper, atol=1e-8)
    np.testing.assert_allclose(shoulder @ elbow @ rest_forearm, forearm, atol=1e-8)


def test_projection_and_fixed_patch_union():
    points = np.array([[1., 2., 3.], [-1., -2., -3.]])
    np.testing.assert_allclose(camera_project(points, 10., [20., 30.]), [[30., 10.], [10., 50.]])
    mask = patch_union(np.array([[10., 10.], [30., 30.]]), 6, (50, 50))
    assert mask.sum() == 2 * 13 * 13
    with pytest.raises(ValueError):
        patch_union(np.array([[1., 1.]]), 6, (50, 50))


def test_composite_preserves_core_and_guard_exactly():
    from scripts.investigation.information_suite.assemble_association import compose_pair
    first = np.full((50, 50, 3), 50, dtype=np.uint8)
    second = np.full((50, 50, 3), 200, dtype=np.uint8)
    reference = np.full((50, 50, 3), 100, dtype=np.uint8)
    images, neutral, audit = compose_pair(first, second, reference, np.array([[20., 20.]]))
    assert audit['core_max_difference'] == audit['boundary_ring_max_difference'] == 0
    np.testing.assert_array_equal(images[0][20, 20], [100, 100, 100])
    assert neutral.shape == first.shape


def test_native_scoring_uses_highest_score_without_wrist_oracle():
    from scripts.investigation.information_suite.scoring import score_people
    truth = np.full((1, 14, 2), [20., 20.])
    points = np.ones((2, 14, 3))
    points[0, :, :2] = 20
    points[1, :, :2] = 35
    result = {'scores': [.6, .9], 'labels': [1, 1], 'boxes': [[0, 0, 40, 40]]*2, 'keypoints': points}
    row = score_people(result, truth, [[0, 0, 40, 40]])[0]
    assert row['detected'] and row['candidate_count'] == 2
    assert row['candidate_index'] == 1 and not row['wrist_correct']
    result['scores'] = [.1, .2]
    assert not score_people(result, truth, [[0, 0, 40, 40]])[0]['detected']


def test_matching_diagnostic_preserves_original_pixel_tolerance():
    from scripts.investigation.information_suite.audit_matching import score_matching_only
    truth = np.full((1, 14, 2), 20.)
    points = np.ones((1, 14, 3))
    points[0, :, :2] = [23., 20.]
    result = {'scores': [.9], 'labels': [1], 'boxes': [[0, 0, 40, 40]], 'keypoints': points}
    row = score_matching_only(result, truth, [[0, 0, 40, 40]], [[0, 0, 100, 100]])[0]
    assert row['detected'] and row['wrist_correct']
    assert row['tau'] == pytest.approx(4.)
    assert row['matching_box_tau'] == pytest.approx(1.6)


def test_shared_mesh_bounds_are_pair_invariant_and_tightly_padded():
    from scripts.investigation.information_suite.geometry import shared_projected_box
    first = np.array([[10., 20.], [40., 60.]])
    second = np.array([[12., 18.], [42., 58.]])
    expected = [9., 17., 34., 44.]
    np.testing.assert_allclose(shared_projected_box([first, second]), expected)
    np.testing.assert_allclose(shared_projected_box([second, first]), expected)


def test_signed_bone_depth_has_same_projection_and_opposite_order():
    from scripts.investigation.information_suite.geometry import bone_endpoint
    elbow = np.array([.2, .4, .1])
    endpoints = [bone_endpoint(elbow, [.3, .6], .4, sign) for sign in (-1, 1)]
    np.testing.assert_allclose(endpoints[0][:2], endpoints[1][:2])
    assert (endpoints[0][2]-elbow[2]) == pytest.approx(-(endpoints[1][2]-elbow[2]))
    for endpoint in endpoints:
        assert np.linalg.norm(endpoint-elbow) == pytest.approx(.4)


def test_pair_ranking_does_not_substitute_for_both_labels_correct():
    from scripts.investigation.information_suite.readouts import classified_pairs, paired_scores
    rows = [dict(family_id=1, condition='main_0', label=0),
            dict(family_id=1, condition='main_1', label=1)]
    assert paired_scores([1., 2.], rows)[1].tolist() == [1.]
    assert classified_pairs([1., 2.], rows)[1].tolist() == [0.]
    assert classified_pairs([-1., 2.], rows)[1].tolist() == [1.]

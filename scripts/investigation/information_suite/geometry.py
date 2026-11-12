from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

CROWDPOSE_TO_RIG = (16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8, 15, 12)


def unit(vector: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(vector)
    if length < 1e-10:
        raise ValueError('Cannot normalize a zero vector')
    return vector / length


def align_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source, target = unit(source), unit(target)
    cross = np.cross(source, target)
    sine = np.linalg.norm(cross)
    cosine = np.clip(source @ target, -1., 1.)
    if sine < 1e-10:
        if cosine > 0:
            return np.eye(3)
        axis = unit(np.cross(source, np.eye(3)[np.argmin(np.abs(source))]))
        return Rotation.from_rotvec(np.pi * axis).as_matrix()
    return Rotation.from_rotvec(np.arctan2(sine, cosine) * cross / sine).as_matrix()


def camera_project(points: np.ndarray, scale: float, center) -> np.ndarray:
    result = np.asarray(points)[..., :2] * np.array([scale, -scale])
    return result + np.asarray(center)


def unproject_xy(points: np.ndarray, scale: float, center) -> np.ndarray:
    return (np.asarray(points) - np.asarray(center)) / np.array([scale, -scale])


def depth_for_length(length: float, projected_delta: np.ndarray) -> float:
    squared = length * length - np.square(projected_delta).sum()
    if squared <= 0:
        raise ValueError('Projected displacement must be shorter than the bone')
    return float(np.sqrt(squared))


def bone_endpoint(origin, endpoint_xy, length, depth_sign):
    if depth_sign not in (-1, 1):
        raise ValueError('Depth sign must be -1 or +1')
    origin, endpoint_xy = np.asarray(origin), np.asarray(endpoint_xy)
    depth = depth_sign * depth_for_length(length, endpoint_xy-origin[:2])
    return np.r_[endpoint_xy, origin[2]+depth]


def shared_projected_box(projected_meshes, image_wh=(384, 256)):
    points = np.concatenate(projected_meshes)
    low = np.maximum(points.min(axis=0)-1, [0, 0])
    high = np.minimum(points.max(axis=0)+1, image_wh)
    if np.any(high <= low):
        raise ValueError('Empty projected mesh bounds')
    return np.r_[low, high-low]


def rigid_arm_pose(rest_upper, rest_forearm, upper, forearm):
    rest_upper, rest_forearm = np.asarray(rest_upper), np.asarray(rest_forearm)
    upper, forearm = np.asarray(upper), np.asarray(forearm)
    if not np.isclose(np.linalg.norm(upper), np.linalg.norm(rest_upper), atol=1e-7):
        raise ValueError('Upper arm length changed')
    if not np.isclose(np.linalg.norm(forearm), np.linalg.norm(rest_forearm), atol=1e-7):
        raise ValueError('Forearm length changed')
    shoulder = align_rotation(rest_upper, upper)
    bend_normal = np.cross(upper, forearm)
    if np.linalg.norm(bend_normal) > 1e-8:
        hinge = np.array([0., -1., 0.])
        hinge -= unit(rest_upper) * (hinge @ unit(rest_upper))
        current = unit(shoulder @ hinge)
        desired = unit(bend_normal)
        axis = unit(upper)
        angle = np.arctan2(axis @ np.cross(current, desired), current @ desired)
        shoulder = Rotation.from_rotvec(axis * angle).as_matrix() @ shoulder
    correction = align_rotation(shoulder @ rest_forearm, forearm)
    elbow = shoulder.T @ correction @ shoulder
    return shoulder, elbow


def patch_union(points: np.ndarray, radius: int, image_hw) -> np.ndarray:
    height, width = image_hw
    result = np.zeros((height, width), dtype=bool)
    for point in np.asarray(points).reshape(-1, 2):
        x, y = np.rint(point).astype(int)
        if x - radius < 0 or y - radius < 0 or x + radius >= width or y + radius >= height:
            raise ValueError('Patch leaves the image')
        result[y-radius:y+radius+1, x-radius:x+radius+1] = True
    return result


def pose_angles(upper, forearm, resting_upper):
    bend = np.degrees(np.arccos(np.clip(unit(upper) @ unit(forearm), -1, 1)))
    elevation = np.degrees(np.arccos(np.clip(unit(upper) @ np.array([0., -1., 0.]), -1, 1)))
    return {'elbow_flexion_degrees': float(bend), 'upperarm_elevation_degrees': float(elevation)}

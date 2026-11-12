from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from scripts.investigation.information_suite.geometry import (
    CROWDPOSE_TO_RIG, align_rotation, camera_project, depth_for_length,
    pose_angles, rigid_arm_pose, unproject_xy, unit, bone_endpoint,
)

REPO = Path(__file__).resolve().parents[3]
ARTIFACTS = Path('/storage/outputs/pose-graph-diff/diagnostics/h_control_information_20261023_epoch065')
BODY_MODELS = Path('/storage/data/models_smplx_v1_1/models')


def load_body_model():
    sys.path.insert(0, str(REPO / 'tmp/h_control_information_suite/deps'))
    import smplx
    return smplx.create(str(BODY_MODELS), model_type='smplx', gender='neutral',
                        ext='npz', use_pca=False, flat_hand_mean=False)


def body_rest(model, betas):
    with torch.no_grad():
        output = model(betas=torch.tensor(betas[None], dtype=torch.float32))
    return output.joints[0, :22].numpy(), output.vertices[0].numpy()


def posed_person(model, betas, root_pixel, root_depth, yaw, elbow_pixel,
                 wrist_pixel, scale, wrist_roll=0., rest_joints=None,
                 forearm_sign=1, elbow_sign=1):
    rest = body_rest(model, betas)[0] if rest_joints is None else rest_joints
    root_rotation = Rotation.from_euler('y', yaw).as_matrix()
    root = np.r_[unproject_xy(np.asarray(root_pixel), scale, [192, 128]), root_depth]
    shoulder = root_rotation @ (rest[16] - rest[0]) + root
    upper_rest, forearm_rest = rest[18] - rest[16], rest[20] - rest[18]
    elbow_xy = unproject_xy(elbow_pixel, scale, [192, 128])
    elbow = bone_endpoint(shoulder, elbow_xy, np.linalg.norm(upper_rest), elbow_sign)
    wrist_xy = unproject_xy(wrist_pixel, scale, [192, 128])
    wrist = bone_endpoint(elbow, wrist_xy, np.linalg.norm(forearm_rest), forearm_sign)
    upper, forearm = elbow - shoulder, wrist - elbow
    angles = pose_angles(upper, forearm, upper_rest)
    if not 5 <= angles['elbow_flexion_degrees'] <= 150 or angles['upperarm_elevation_degrees'] > 140:
        raise ValueError('Declared arm-angle limits exceeded')
    shoulder_global, elbow_local = rigid_arm_pose(upper_rest, forearm_rest, upper, forearm)
    pose = np.zeros((21, 3), dtype=np.float32)
    pose[15] = Rotation.from_matrix(root_rotation.T @ shoulder_global).as_rotvec()
    pose[17] = Rotation.from_matrix(elbow_local).as_rotvec()
    pose[19] = unit(forearm_rest) * wrist_roll
    right_rest = rest[19] - rest[17]
    right_world = root_rotation @ unit(np.array([-.20, -.97, .10]))
    pose[16] = Rotation.from_matrix(root_rotation.T @ align_rotation(right_rest, right_world)).as_rotvec()
    pose[18] = [0, .10, 0]
    hand_offsets = []
    for mean in (model.left_hand_mean, model.right_hand_mean):
        offsets = np.zeros((15, 3), dtype=np.float32)
        offsets[:12, 2] = .45 * np.sign(mean.numpy().reshape(15, 3)[:12, 2])
        hand_offsets.append(torch.tensor(offsets.reshape(1, -1)))
    with torch.no_grad():
        output = model(betas=torch.tensor(betas[None], dtype=torch.float32),
                       body_pose=torch.tensor(pose.reshape(1, -1)),
                       global_orient=torch.tensor(Rotation.from_matrix(root_rotation).as_rotvec()[None], dtype=torch.float32),
                       transl=torch.tensor((root - rest[0])[None], dtype=torch.float32),
                       left_hand_pose=hand_offsets[0], right_hand_pose=hand_offsets[1])
    joints, vertices = output.joints[0, :22].numpy(), output.vertices[0].numpy()
    projected = camera_project(joints, scale, [192, 128])
    error = max(np.linalg.norm(projected[18] - elbow_pixel), np.linalg.norm(projected[20] - wrist_pixel))
    if error > 1e-4:
        raise ValueError(f'IK projection error {error}')
    return {'vertices': vertices, 'joints': joints, 'pose': pose,
            'keypoints': projected[list(CROWDPOSE_TO_RIG)], 'angles': angles,
            'projection_error': float(error)}


def face_parts(model, rest_vertices):
    dominant = model.lbs_weights.numpy().argmax(-1)
    parts = np.zeros(len(dominant), dtype=np.int32)
    parts[np.isin(dominant, [0, 3, 6, 9, 13, 14, 16, 17])] = 1
    parts[np.isin(dominant, [1, 2, 4, 5, 7, 8])] = 2
    parts[np.isin(dominant, [10, 11])] = 3
    parts[(rest_vertices[:, 1] > .34) & (rest_vertices[:, 2] < .03)] = 4
    parts[np.isin(dominant, [23, 24])] = 4
    return np.rint(np.median(parts[model.faces], axis=1)).astype(np.int32)


def generate_family(model, family_id, directory, repair=False):
    rng = np.random.default_rng(202610230 + family_id)
    betas = rng.normal(0, .22, (2, 10)).astype(np.float32)
    rests = [body_rest(model, beta) for beta in betas]
    scale = 35 / np.mean([np.linalg.norm(j[20] - j[18]) for j, _ in rests])
    horizontal = float(rng.uniform(-5, 5))
    root_pixels = np.array([[155 + horizontal, 120], [(205 if repair else 225) + horizontal, 120]])
    yaws = np.array([rng.uniform(-.08, .08), (-np.pi/4 if repair else np.pi) + rng.uniform(-.08, .08)])
    depths = np.array([-.35, .35]) * (1 if family_id % 2 else -1)
    shoulders = []
    for (rest, _), root_pixel, yaw in zip(rests, root_pixels, yaws):
        root = np.r_[unproject_xy(root_pixel, scale, [192, 128]), 0]
        shoulder = Rotation.from_euler('y', yaw).apply(rest[16] - rest[0]) + root
        shoulders.append(camera_project(shoulder[None], scale, [192, 128])[0])
    elbow_y = max(point[1] for point in shoulders) + (20 if repair else rng.uniform(27, 29))
    elbows = np.array([[(182 if repair else 180) + horizontal, elbow_y], [200 + horizontal, elbow_y]])
    wrists = np.array([[(178 if repair else 175) + horizontal, elbow_y + (27 if repair else 20)], [(202 if repair else 205) + horizontal, elbow_y + (27 if repair else 20)]])
    shifts = {'main': (0., 0.), 'nuisance': (0., 0.), 'clear': (-45., 45.), 'visible': (-45., 45.)}
    half_width = 68 if repair else 65
    boxes = np.array([[root_pixels[0, 0]-half_width, 0, 2*half_width, 256],
                      [root_pixels[1, 0]-half_width, 0, 2*half_width, 256]])
    scenes, metadata = [], []
    style_seeds = {'main': int(rng.integers(1, 2**30)), 'nuisance': int(rng.integers(1, 2**30))}
    style_seeds['clear'] = style_seeds['main']
    style_seeds['visible'] = style_seeds['main']
    weights = model.lbs_weights.numpy()
    arm_core = (weights[:, [18, 19, 20, 21] + list(range(25, 55))].sum(1) > .8)[model.faces].all(1)
    torso_core = (weights[:, [0, 1, 2, 3, 6, 9]].sum(1) > .8)[model.faces].all(1)
    for style in (('main', 'nuisance', 'clear', 'visible') if repair else ('main', 'nuisance', 'clear')):
        style_seed = style_seeds[style]
        members = (0, 1, 'reference0', 'reference1') if style == 'clear' and repair else ((0, 1) if style == 'visible' else (0, 1, 'reference'))
        for member in members:
            owner = (family_id % 2) if member == 'reference' else (int(member[-1]) if isinstance(member, str) else member)
            people = []
            local_boxes = boxes.copy()
            for person in (0, 1):
                shift = np.array([shifts[style][person], 0.])
                wrist_index = person if owner == 0 else 1-person
                result = posed_person(model, betas[person], root_pixels[person]+shift,
                                      depths[person], yaws[person], elbows[person]+shift,
                                      wrists[wrist_index]+shift, scale,
                                      .30 if isinstance(member, str) else 0., rests[person][0])
                vertex_pixels = camera_project(result['vertices'], scale, [192, 128])
                local_boxes[person, 0] += shift[0]
                left, top, width, height = local_boxes[person]
                if vertex_pixels[:, 0].min() < left or vertex_pixels[:, 0].max() > left+width:
                    raise ValueError(f'Mesh leaves fixed anchor box: {style}/{member}/person{person}; '
                                     f'mesh x{vertex_pixels[:,0].min():.2f}..{vertex_pixels[:,0].max():.2f}, '
                                     f'box x{left:.2f}..{left+width:.2f}')
                people.append(result)
            label = f'{style}_{member}'
            filename = f'{label}.npz'
            vertices = np.concatenate([p['vertices'] for p in people])
            faces = np.concatenate([model.faces, model.faces+len(people[0]['vertices'])])
            materials = np.concatenate([face_parts(model, rests[p][1]) for p in (0, 1)])
            joints = np.stack([p['joints'] for p in people])
            record = {'family_id': family_id, 'condition': label, 'scale': scale,
                      'mesh': filename, 'rgb': label+'.png', 'style_seed': style_seed,
                      'keypoints': [p['keypoints'].tolist() for p in people],
                      'boxes': local_boxes.tolist(), 'target_joint': 4,
                      'projection_errors': [p['projection_error'] for p in people],
                      'arm_angles': [p['angles'] for p in people], 'member': member,
                      'repair': repair,
                      'neutral_reference_owner': family_id % 2,
                      'geometry_status': 'FK/length/declared angle limits passed; collision audit pending'}
            if repair and style != 'visible':
                panels = []
                for person in (0, 1):
                    shift = np.array([shifts[style][person], 0.])
                    panels.extend([(shoulders[person]+shift).tolist(), (elbows[person]+shift).tolist()])
                    panels.extend([(wrist+shift).tolist() for wrist in wrists])
                record['foreground_panels'] = {'centers_xy': panels, 'half_width_pixels': 9, 'world_z': 1.5}
            scenes.append((filename, vertices, faces, materials, joints))
            metadata.append(record)
    directory.mkdir(parents=True, exist_ok=False)
    for filename, vertices, faces, materials, joints in scenes:
        np.savez_compressed(directory/filename, vertices=vertices, faces=faces,
                            materials=materials, joints=joints,
                            arm_core=np.tile(arm_core, 2), torso_core=np.tile(torso_core, 2))
    (directory/'scenes.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--families', type=int, default=32)
    parser.add_argument('--start-id', type=int, default=100000)
    parser.add_argument('--repair', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    model = load_body_model()
    families, rejected = [], []
    for family_id in range(args.start_id, args.start_id+512):
        try:
            scenes = generate_family(model, family_id, args.output/f'family_{family_id}', args.repair)
            families.append({'family_id': family_id, 'path': f'family_{family_id}', 'scenes': len(scenes)})
            print('GENERATED', family_id, len(families), flush=True)
        except ValueError as error:
            rejected.append({'family_id': family_id, 'reason': str(error)})
        if len(families) == args.families:
            break
    (args.output/'families.json').write_text(json.dumps({'families': families, 'rejected': rejected}, indent=2)+'\n')
    if len(families) != args.families:
        raise RuntimeError(f'Only {len(families)} valid geometric families after512 candidates')


if __name__ == '__main__':
    main()

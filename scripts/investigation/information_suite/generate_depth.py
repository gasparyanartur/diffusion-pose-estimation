from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from scripts.investigation.information_suite.generate_association import (
    body_rest, face_parts, load_body_model, posed_person,
)
from scripts.investigation.information_suite.geometry import camera_project, shared_projected_box


def body_masks(model):
    weights = model.lbs_weights.numpy()
    arm = weights[:, [18, 19, 20, 21] + list(range(25, 55))].sum(1) > .8
    torso = weights[:, [0, 1, 2, 3, 6, 9]].sum(1) > .8
    hand = weights[:, [20] + list(range(25, 40))].sum(1) > .5
    return arm[model.faces].all(1), torso[model.faces].all(1), hand


def pose_settings(rest, scale, root_pixel, yaw, study, stratum, rng):
    rotation = Rotation.from_euler('y', yaw)
    shoulder = camera_project(rotation.apply((rest[16]-rest[0])[None]), scale, root_pixel)[0]
    upper = np.linalg.norm(rest[18]-rest[16])*scale
    forearm = np.linalg.norm(rest[20]-rest[18])*scale
    if study == 'B':
        elbow = shoulder + [2, .95*upper]
        lengths = [.35*forearm, .985*forearm]
        wrists = [elbow+[length, 0] for length in lengths]
        signs = [1, 1]
    else:
        elbow = shoulder + [.35*upper, .92*upper]
        length = rng.uniform(.56, .72)*forearm
        direction = np.array([-1 if stratum == 'occlusion' else 1, .22])
        direction /= np.linalg.norm(direction)
        wrists = [elbow+length*direction]*2
        signs = [-1, 1]
    return shoulder, elbow, wrists, signs, forearm


def make_person(model, beta, root, yaw, elbow, wrist, scale, rest, sign):
    return posed_person(model, beta, root, 0., yaw, elbow, wrist, scale,
                        rest_joints=rest, forearm_sign=sign)


def family_geometry(model, family_id, study, split, repair=False):
    rng = np.random.default_rng(202610240 + family_id)
    beta = rng.normal(0, .22, 10).astype(np.float32)
    rest, rest_vertices = body_rest(model, beta)
    scale = 35 / np.linalg.norm(rest[20]-rest[18])
    root = np.array([192+rng.uniform(-4, 4), 120.])
    azimuths = {'pilot': [-5, 5], 'train': [-20, 0, 20], 'val': [-10, 10], 'test': [-30, 30]}
    yaw_degrees = float(rng.choice(azimuths['pilot' if study == 'B' else split]))
    yaw = np.deg2rad(yaw_degrees)
    stratum = 'occlusion' if study == 'C' and family_id % 2 else 'surface'
    shoulder, elbow, wrists, signs, forearm = pose_settings(rest, scale, root, yaw, study, stratum, rng)
    if repair and study == 'C':
        direction = np.array([-1 if stratum == 'occlusion' else 1, .22])
        ratio = .44 if stratum == 'occlusion' else .65
        wrists = [elbow+ratio*forearm*direction/np.linalg.norm(direction)]*2
    people = [make_person(model, beta, root, yaw, elbow, wrist, scale, rest, sign)
              for wrist, sign in zip(wrists, signs)]
    points = [camera_project(person['vertices'], scale, [192, 128]) for person in people]
    if any(p.min() < 1 or p[:, 0].max() >= 383 or p[:, 1].max() >= 255 for p in points):
        raise ValueError('Body leaves image')
    box = shared_projected_box(points)
    tau = .04*np.sqrt(box[2]*box[3])
    arm_core, torso_core, hand = body_masks(model)
    panels = []
    if study == 'B':
        separation = float(np.linalg.norm(wrists[1]-wrists[0]))
        if separation < 3*tau:
            raise ValueError(f'Target separation{separation:.3f} below3tau={3*tau:.3f}')
        hand_points = np.concatenate([p[hand] for p in points])
        low, high = hand_points.min(0)-2, hand_points.max(0)+2
        if low[0] < elbow[0]+4:
            raise ValueError('Full-hand occluder would hide the visible proximal forearm')
        panels = [{'bounds_xy': [float(low[0]), float(low[1]), float(high[0]), float(high[1])], 'world_z': 1.5}]
        visible = people
    else:
        equality = float(np.abs(people[0]['keypoints']-people[1]['keypoints']).max())
        if equality > 1e-4:
            raise ValueError(f'Full2D mismatch{equality}')
        control_wrist = elbow+[.65*forearm, .1*forearm]
        visible = [make_person(model, beta, root, yaw, elbow, control_wrist, scale, rest, sign)
                   for sign in (-1, 1)]
    return dict(people=people, visible=visible, rest_vertices=rest_vertices, scale=scale,
                box=box, tau=float(tau), beta=beta, stratum=stratum, yaw_degrees=yaw_degrees,
                arm_core=arm_core, torso_core=torso_core, panels=panels, root=root,
                shoulder=shoulder, elbow=elbow, wrists=wrists,
                style_seed=int(rng.integers(1, 2**29)), nuisance_seed=int(rng.integers(1, 2**29)))


def generate_family(model, family_id, study, split, directory, repair=False, pilot=False):
    geometry = family_geometry(model, family_id, study, split, repair)
    materials = face_parts(model, geometry['rest_vertices'])
    scenes, payloads = [], []
    visible_controls = pilot or (study == 'B' and split == 'test')
    styles = ['main', 'nuisance'] + (['clear'] if visible_controls else [])
    light_angle = 35 if split == 'test' else 0
    for style in styles:
        people = geometry['visible'] if style == 'clear' else geometry['people']
        box = shared_projected_box([camera_project(p['vertices'], geometry['scale'], [192, 128]) for p in people])
        for member, person in enumerate(people):
            condition = f'{style}_{member}'
            scene = dict(family_id=family_id, study=study, split=split, stratum=geometry['stratum'],
                         condition=condition, member=member, mesh=condition+'.npz', rgb=condition+'.png',
                         scale=geometry['scale'], style_seed=geometry['nuisance_seed'] if style == 'nuisance' else geometry['style_seed'],
                         light_angle=light_angle, keypoints=[person['keypoints'].tolist()], boxes=[box.tolist()],
                         label=member, depth_delta=float(person['joints'][20, 2]-person['joints'][18, 2]),
                         projection_errors=[person['projection_error']], arm_angles=[person['angles']],
                         camera_azimuth=geometry['yaw_degrees'], target_joint=4)
            scene['rest_texture'] = repair
            if study == 'B' and style != 'clear':
                scene['foreground_rectangles'] = geometry['panels']
            scenes.append(scene)
            payloads.append(dict(vertices=person['vertices'], joints=person['joints'][None], faces=model.faces,
                                 materials=materials, rest_vertices=geometry['rest_vertices'],
                                 arm_core=geometry['arm_core'], torso_core=geometry['torso_core']))
    directory.mkdir(parents=True, exist_ok=False)
    for scene, payload in zip(scenes, payloads):
        np.savez_compressed(directory/scene['mesh'], **payload)
    render_rng = np.random.default_rng(987654+family_id)
    render_rng.shuffle(scenes)
    (directory/'scenes.json').write_text(json.dumps(scenes, indent=2)+'\n')
    metadata = {key: geometry[key] for key in ('tau', 'stratum', 'yaw_degrees')}
    metadata.update(family_id=family_id, study=study, split=split, repair=repair,
                    beta=geometry['beta'].tolist(), wrist_candidates=np.asarray(geometry['wrists']).tolist(),
                    target_separation=float(np.linalg.norm(geometry['wrists'][0]-geometry['wrists'][1])))
    (directory/'geometry.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--study', choices=['B', 'C'], required=True)
    parser.add_argument('--split', choices=['pilot', 'train', 'val', 'test'], default='pilot')
    parser.add_argument('--start-id', type=int, required=True)
    parser.add_argument('--families', type=int, default=40)
    parser.add_argument('--repair', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    model = load_body_model()
    accepted, rejected = [], []
    for family_id in range(args.start_id, args.start_id+20*args.families):
        try:
            record = generate_family(model, family_id, args.study, args.split,
                                     args.output/f'family_{family_id}', args.repair, args.split == 'pilot')
            accepted.append(record)
            print('GENERATED', family_id, len(accepted), flush=True)
        except ValueError as error:
            rejected.append(dict(family_id=family_id, reason=str(error)))
        if len(accepted) == args.families:
            break
    (args.output/'families.json').write_text(json.dumps(dict(families=accepted, rejected=rejected), indent=2)+'\n')
    if len(accepted) < args.families:
        raise RuntimeError(f'Only{len(accepted)} families; inspect rejection reasons')


if __name__ == '__main__':
    main()

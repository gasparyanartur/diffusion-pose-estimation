from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, distance_transform_edt

from scripts.investigation.information_suite.geometry import patch_union


def image_array(path):
    return np.asarray(Image.open(path).convert('RGB'))


def compose_pair(first, second, reference, points, radius=6):
    core = patch_union(points, radius, first.shape[:2])
    guard = patch_union(points, radius+2, first.shape[:2])
    feather = np.clip(1-distance_transform_edt(~guard)/2, 0, 1)[..., None]
    outputs = [np.rint(reference*feather + image*(1-feather)).astype(np.uint8) for image in (first, second)]
    assert np.array_equal(outputs[0][guard], outputs[1][guard])
    difference = np.any(outputs[0] != outputs[1], axis=-1)
    neutral = outputs[0].copy()
    neutral[binary_dilation(difference, iterations=2)] = np.rint(reference.mean((0, 1))).astype(np.uint8)
    return outputs, neutral, {'core_pixels': int(core.sum()), 'guard_pixels': int(guard.sum()),
                              'core_max_difference': int(np.max(np.abs(outputs[0][core].astype(int)-outputs[1][core]))),
                              'boundary_ring_max_difference': int(np.max(np.abs(outputs[0][guard & ~core].astype(int)-outputs[1][guard & ~core]))),
                              'neutral_changed_pixels': int(difference.sum()),
                              'reference': 'separate relaxed-wrist render; ownership balanced by family parity',
                              'core_radius': radius, 'constant_guard_radius': radius+2, 'feather_width': 2}


def invariant_error(first, second):
    result = 0.
    for joint in range(14):
        a, b = first[:, joint], second[:, joint]
        a = a[np.lexsort((a[:, 1], a[:, 0]))]
        b = b[np.lexsort((b[:, 1], b[:, 0]))]
        result = max(result, float(np.abs(a-b).max()))
    return result


def montage(images, filename, points=None):
    rows = 3 if 'visible_0' in images else 2
    output = Image.new('RGB', (384*3, 282*rows), 'white')
    names = ['main_0', 'main_1', 'neutral', 'clear_0', 'clear_1', 'nuisance_1']
    if rows == 3:
        names.extend(['visible_0','visible_1','nuisance_0'])
    for index, name in enumerate(names):
        picture = Image.fromarray(images[name]).copy()
        if points and name in points:
            draw = ImageDraw.Draw(picture)
            for person, keypoints in enumerate(points[name]):
                color = 'lime' if person == 0 else 'cyan'
                x, y = keypoints[4]
                draw.ellipse((x-3, y-3, x+3, y+3), outline=color, width=1)
        left, top = (index%3)*384, (index//3)*282
        output.paste(picture, (left, top+25))
        ImageDraw.Draw(output).text((left+5, top+5), name, fill='black')
    output.save(filename)


def process_family(directory, target_person):
    records = {r['condition']: r for r in json.loads((directory/'scenes.json').read_text())}
    for record in records.values():
        audit = json.loads((directory/record['rgb']).with_suffix('.render.json').read_text())
        if audit['inter_person_triangle_intersections'] != 0:
            raise ValueError(f"Inter-person mesh intersection in {record['condition']}")
        own = audit.get('forearm_hand_vs_own_torso_intersections')
        if own is None or any(own):
            raise ValueError(f"Forearm/torso audit failed in {record['condition']}: {own}")
    images, point_sets, conditions, audits = {}, {}, {}, {}
    for style in ('main', 'nuisance', 'clear'):
        first, second = records[style+'_0'], records[style+'_1']
        points_first, points_second = np.array(first['keypoints']), np.array(second['keypoints'])
        error = invariant_error(points_first, points_second)
        if (style != 'clear' and error > 1e-4) or first['boxes'] != second['boxes']:
            raise ValueError(f'Pair equality failed: {style}, {error}')
        for box, a, b in zip(first['boxes'], points_first, points_second):
            tau = .04*np.sqrt(box[2]*box[3])
            if np.linalg.norm(a[4]-b[4]) < 3*tau:
                raise ValueError('Candidate targets do not satisfy3*tau separation')
        if style == 'clear':
            outputs = []
            for member, record in enumerate((first,second)):
                original = image_array(directory/record['rgb'])
                reference = image_array(directory/records[f'clear_reference{member}']['rgb'])
                pair, _, _ = compose_pair(original,original,reference,np.asarray(record['keypoints']))
                outputs.append(pair[0])
            audit = {'joint_set_max_error_pixels':error,'equality_required':False,
                     'reason':'Separated controls have per-member references and different wrist point sets'}
        else:
            outputs, neutral, audit = compose_pair(image_array(directory/first['rgb']),
                                                   image_array(directory/second['rgb']),
                                                   image_array(directory/records[style+'_reference']['rgb']),
                                                   points_first)
            audit['joint_set_max_error_pixels'] = error
        audits[style] = audit
        for member, array in enumerate(outputs):
            name = f'{style}_{member}'
            path = directory/f'input_{name}.png'
            Image.fromarray(array).save(path)
            images[name] = array
            point_sets[name] = records[name]['keypoints']
            conditions[name] = {'path': str(path), 'keypoints': records[name]['keypoints'], 'boxes': records[name]['boxes']}
        if style == 'main':
            path = directory/'input_neutral.png'
            Image.fromarray(neutral).save(path)
            images['neutral'] = neutral
            conditions['neutral'] = {'path': str(path), 'keypoints': first['keypoints'], 'boxes': first['boxes'],
                                      'alternative_keypoints': second['keypoints']}
    for member in (0,1):
        name = f'visible_{member}'
        if name in records:
            record = records[name]
            images[name] = image_array(directory/record['rgb'])
            point_sets[name] = record['keypoints']
            conditions[name] = {'path':str(directory/record['rgb']), 'keypoints':record['keypoints'],'boxes':record['boxes']}
    montage(images, directory/'montage.png', point_sets)
    result = {'family_id': first['family_id'], 'target_person': target_person,
              'conditions': conditions, 'audits': audits,
              'physical_audits': 'zero inter-person and forearm-core/torso-core surface intersections',
              'status': 'geometric and pixel contracts passed; visual/cue/native gates pending'}
    (directory/'pair_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--families', type=int, default=32)
    args = parser.parse_args()
    accepted, rejected = [], []
    for path in sorted(args.root.glob('family_*/scenes.json')):
        try:
            accepted.append(process_family(path.parent, len(accepted)%2))
        except (ValueError, FileNotFoundError) as error:
            rejected.append({'family': path.parent.name, 'reason': str(error)})
        if len(accepted) == args.families:
            break
    status = {'families': accepted, 'rejected': rejected, 'requested': args.families,
              'complete': len(accepted) == args.families}
    (args.root/'pilot_manifest.json').write_text(json.dumps(status, indent=2)+'\n')
    print('ASSEMBLED', len(accepted), 'REJECTED', len(rejected), flush=True)
    if not status['complete']:
        raise RuntimeError('Insufficient physically/pixel-valid families; no native inference admitted')


if __name__ == '__main__':
    main()

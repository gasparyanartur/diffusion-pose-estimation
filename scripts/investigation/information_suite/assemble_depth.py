from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def audit_family(directory, study, images_required=True):
    scenes = json.loads((directory/'scenes.json').read_text())
    records = {scene['condition']:scene for scene in scenes}
    for scene in scenes:
        audit = json.loads((directory/scene['rgb']).with_suffix('.render.json').read_text())
        if audit['projection_max_error_pixels'] > 1e-4:
            raise ValueError('Renderer projection mismatch')
        if audit['inter_person_triangle_intersections'] or any(audit['forearm_hand_vs_own_torso_intersections']):
            raise ValueError('Surface intersection')
        if study == 'C':
            visible = audit['wrist_surface'][0]['first_surface_is_forearm_or_hand']
            expected_visible = not (scene['stratum'] == 'occlusion' and scene['member'] == 0
                                    and not scene['condition'].startswith('clear_'))
            if visible != expected_visible:
                raise ValueError('Camera-ray torso-occlusion stratum mismatch')
    differences = {}
    for style in ('main', 'nuisance'):
        first, second = [records[f'{style}_{member}'] for member in (0, 1)]
        points = [np.asarray(record['keypoints']) for record in (first, second)]
        compared = np.ones(14, dtype=bool)
        if study == 'B':
            compared[4] = False
        residual = float(np.abs(points[0][:, compared]-points[1][:, compared]).max())
        if residual > 1e-4 or first['boxes'] != second['boxes']:
            raise ValueError('Shared geometry or box mismatch')
        differences[style] = residual
        if study == 'B':
            wrist_delta = np.linalg.norm(points[0][0, 4]-points[1][0, 4])
            box = first['boxes'][0]
            tau = .04*np.sqrt(box[2]*box[3])
            if wrist_delta < 3*tau:
                raise ValueError('Candidate separation fails')
            panel = first['foreground_rectangles'][0]['bounds_xy']
            for point in [p[0, 4] for p in points]:
                if not panel[0] <= point[0] <= panel[2] or not panel[1] <= point[1] <= panel[3]:
                    raise ValueError('Target wrist outside occluder')
    conditions = {name:dict(path=str(directory/scene['rgb']), keypoints=scene['keypoints'],
                            boxes=scene['boxes'], label=scene['label'])
                  for name, scene in records.items()}
    if study == 'B' and images_required:
        a, b = [np.asarray(Image.open(directory/records[f'main_{member}']['rgb']).convert('RGB')) for member in (0, 1)]
        neutral = a.copy()
        neutral[np.any(a != b, axis=-1)] = [50, 55, 60]
        path = directory/'neutral.png'
        Image.fromarray(neutral).save(path)
        conditions['neutral'] = dict(path=str(path), keypoints=records['main_0']['keypoints'],
                                     alternative_keypoints=records['main_1']['keypoints'], boxes=records['main_0']['boxes'])
    return dict(conditions=conditions, projection_residuals=differences)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--study', choices=['B', 'C'], required=True)
    parser.add_argument('--families', type=int, default=32)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    families, rejected, strata = [], [], {'surface':0, 'occlusion':0}
    for directory in sorted(args.root.glob('family_*')):
        try:
            result = audit_family(directory, args.study, images_required=not args.preflight)
            metadata = json.loads((directory/'geometry.json').read_text())
            stratum = metadata['stratum']
            if args.study == 'C' and strata[stratum] == args.families//2:
                continue
            families.append(dict(family_id=metadata['family_id'], target_person=0, stratum=stratum, **result))
            strata[stratum] += 1
        except (ValueError, FileNotFoundError, KeyError) as error:
            rejected.append(dict(path=str(directory), reason=str(error)))
        if len(families) == args.families:
            break
    manifest = dict(complete=len(families)==args.families, phase=f'{args.study}_admission_pilot',
                    study=args.study, families=families, rejected=rejected, strata=strata,
                    selection='First physically valid families in ID order; C16 per stratum. No model outcomes used.',
                    matching='Within-pair mesh union with1px padding; same box for IoU and PCK scale.')
    (args.root/'pilot_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    if not args.preflight:
        canvas = Image.new('RGB', (384*3, 280*min(8, len(families))), 'white')
        for row, family in enumerate(families[:8]):
            for col, condition in enumerate(('main_0', 'main_1', 'clear_0')):
                if condition not in family['conditions']:
                    continue
                canvas.paste(Image.open(family['conditions'][condition]['path']), (384*col, 280*row+20))
                ImageDraw.Draw(canvas).text((384*col+4, 280*row+3), f"{family['family_id']} {family['stratum']} {condition}", fill='black')
        canvas.save(args.root/'montage.png')
        if manifest['complete'] and args.root.name.startswith('main_'):
            selected_ids = {family['family_id'] for family in families}
            removed = 0
            for directory in args.root.glob('family_*'):
                if int(directory.name.split('_')[1]) in selected_ids:
                    continue
                for mesh in directory.glob('*.npz'):
                    mesh.unlink()
                    removed += 1
            (args.root/'cache_cleanup.json').write_text(json.dumps(dict(
                removed_unselected_mesh_caches=removed,
                preserved='All selected meshes, source snapshot, rejected-scene parameters, seeds and physical/camera-ray audit JSON.',
                rationale='Generated unselected intermediate meshes are reproducible; preserve the registered artifact budget.'))+'\n')
    print(json.dumps(dict(complete=manifest['complete'], families=len(families), strata=strata, rejected=len(rejected))))
    if not manifest['complete']:
        raise RuntimeError('Not enough physically valid families')


if __name__ == '__main__':
    main()

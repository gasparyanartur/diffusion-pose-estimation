from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[3]
SCRIPTS = REPO/'scripts/investigation/information_suite'
ARTIFACTS = Path('/storage/outputs/pose-graph-diff/diagnostics/h_control_information_20261023_epoch065')
SUITE = REPO/'tmp/h_control_information_suite'
FROZEN = Path('/storage/outputs/pose-graph-diff/results/poseflow_v24_matched/20261022_192822/workspace')


def run(command, log, native=False, allow_failure=False):
    environment = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    if native:
        environment['PYTHONPATH'] = str(FROZEN)
    with log.open('a') as stream:
        result = subprocess.run([str(value) for value in command], cwd=REPO, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode and not allow_failure:
        raise RuntimeError(f'Command failed; inspect{log}')
    return result.returncode


def python_module(name, *arguments):
    return [sys.executable, '-m', 'scripts.investigation.information_suite.'+name, *arguments]


def renderer(root, log, audit=False, selection=False):
    arguments = ['/usr/local/bin/blender', '-b', '-t', '2', '-P', SCRIPTS/'render_blender.py', '--', '--root', root]
    if audit:
        run(arguments+['--audit-only'], log)
        return
    selected = json.loads((root/'pilot_manifest.json').read_text())['families']
    expected = sum(len(json.loads((root/f"family_{family['family_id']}"/'scenes.json').read_text())) for family in selected)
    for _ in range(expected//256+2):
        missing = sum(not (root/f"family_{family['family_id']}"/scene['rgb']).exists()
                      for family in selected
                      for scene in json.loads((root/f"family_{family['family_id']}"/'scenes.json').read_text()))
        if not missing:
            return
        run(arguments+['--selection', root/'pilot_manifest.json', '--limit', '256'], log)
    raise RuntimeError('Renderer did not finish selected images')


def prepare_split(study, split, count, first_id, logs):
    root = ARTIFACTS/f'main_{study}_{split}'
    root.mkdir(exist_ok=True)
    if (root/'PREPARED.json').exists():
        return root
    source = root/'code_snapshot'
    if not source.exists():
        shutil.copytree(SCRIPTS, source, ignore=shutil.ignore_patterns('__pycache__'))
    generated = sorted(root.glob('family_*/geometry.json'))
    candidate_id = max([int(p.parent.name.split('_')[1]) for p in generated], default=first_id-1)+1
    target = count
    for batch_index in range(30):
        used = sum(path.stat().st_size for path in ARTIFACTS.rglob('*') if path.is_file())
        if used > 9.5*1024**3 or shutil.disk_usage(ARTIFACTS).free < 12*1024**3:
            raise RuntimeError('Registered artifact/free-space budget reached; no automatic expansion')
        existing = json.loads((root/'pilot_manifest.json').read_text()) if (root/'pilot_manifest.json').exists() else {}
        if not existing.get('complete'):
            amount = min(256, max(64, target-len(existing.get('families', []))))
            run(python_module('generate_depth', '--output', root, '--study', study, '--split', split,
                              '--start-id', candidate_id, '--families', amount, '--repair'), logs/f'{study}_{split}_generate.log')
            shutil.copy2(root/'families.json', root/f'generation_batch_{candidate_id}.json')
            candidate_id = max(int(p.parent.name.split('_')[1]) for p in root.glob('family_*/geometry.json'))+1
        renderer(root, logs/f'{study}_{split}_audit.log', audit=True)
        result = run(python_module('assemble_depth', '--root', root, '--study', study, '--families', count, '--preflight'),
                     logs/f'{study}_{split}_assemble.log', allow_failure=True)
        if result == 0:
            break
    else:
        raise RuntimeError('Candidate budget exhausted before physical admission')
    shutil.copy2(root/'pilot_manifest.json', root/'selected_before_render.json')
    renderer(root, logs/f'{study}_{split}_render.log', selection=True)
    run(python_module('assemble_depth', '--root', root, '--study', study, '--families', count), logs/f'{study}_{split}_assemble.log')
    final = json.loads((root/'pilot_manifest.json').read_text())
    prior = json.loads((root/'selected_before_render.json').read_text())
    assert [f['family_id'] for f in prior['families']] == [f['family_id'] for f in final['families']]
    (root/'PREPARED.json').write_text(json.dumps(dict(study=study, split=split, families=count, selection='Physical and camera-ray checks only; fixed before inference'))+'\n')
    return root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', choices=['B', 'C'], required=True)
    args = parser.parse_args()
    study = args.study
    admission = json.loads((ARTIFACTS/f'pilot_{study}_repair1/admission.json').read_text())
    assert admission['admitted'], 'Main study is not admitted'
    assert shutil.disk_usage(ARTIFACTS).free >= 12*1024**3
    logs = ARTIFACTS/f'main_{study}_logs';logs.mkdir(exist_ok=True)
    status_path = logs/'status.json'
    start = time.monotonic()
    status = dict(study=study, phase='starting', complete=False, model_training_runs=0)
    def update(phase, **fields):
        status.update(phase=phase, seconds=time.monotonic()-start, **fields)
        status_path.write_text(json.dumps(status, indent=2)+'\n')
        print(json.dumps(status), flush=True)
    splits = [('train', 256, 420000), ('val', 64, 440000), ('test', 512, 400000)] if study == 'B' else [('train', 1024, 600000), ('val', 128, 700000), ('test', 512, 800000)]
    roots = {}
    try:
        for split, count, first_id in splits:
            update('constructing', split=split, requested_families=count)
            roots[split] = prepare_split(study, split, count, first_id, logs)
            arms = ['control', 'h'] + (['initialization'] if study == 'C' else [])
            for arm in arms:
                if (roots[split]/'features'/f'{arm}_DONE.json').exists():
                    continue
                update('extracting_features', split=split, arm=arm)
                run([sys.executable, SCRIPTS/'extract_features.py', '--root', roots[split], '--suite', SUITE, '--arm', arm],
                    logs/f'{split}_{arm}_features.log', native=True)
        if study == 'B' and not (roots['test']/'native'/'DONE.json').exists():
            update('native_behavior', split='test')
            run([sys.executable, SCRIPTS/'evaluate_pilot.py', '--root', roots['test'], '--suite', SUITE, '--reuse-feature-native'],
                logs/'test_native.log', native=True)
        update('fitting_readouts')
        output = ARTIFACTS/f'main_{study}_readouts/results.json'
        run(python_module('readouts', '--train', roots['train'], '--val', roots['val'], '--test', roots['test'], '--study', study, '--output', output),
            logs/'readouts.log')
        if study == 'C':
            update('auditing_image_shortcuts')
            run(python_module('image_audits', '--train', roots['train'], '--val', roots['val'], '--test', roots['test'],
                              '--output', output.parent/'image_audits.json'), logs/'image_audits.log')
        update('summarizing_native_predictions')
        run(python_module('summarize_behavior', '--root', roots['test'], '--output', output.parent/'native_behavior.json'),
            logs/'native_behavior.log')
        update('complete', complete=True, results=str(output))
    except Exception as error:
        update('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()

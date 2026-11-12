import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def summarize_comparison(root):
    reports = {mode: json.loads((root / mode / 'final_results.json').read_text()) for mode in ('x0', 'v')}
    batches = {mode: json.loads((root / mode / 'first_batches.json').read_text()) for mode in ('x0', 'v')}
    if batches['x0'] != batches['v']:
        raise ValueError('Paired initial augmented batches differ between arms')
    differences = {}
    for condition, control in reports['x0']['conditions'].items():
        candidate = reports['v']['conditions'][condition]
        differences[condition] = {name: candidate['metrics'][name] - control['metrics'][name]
            for name in ('pose_AP', 'bbox_AP')}
    recovery = {}
    for mode, report in reports.items():
        recovery[mode] = {f'noise_{level}/OKS_gain_over_input': report['small_noise'][f'noise_{level}/output_paired_OKS_mean']
            - report['small_noise'][f'noise_{level}/input_paired_OKS_mean'] for level in ('1', '5')}
    matched_age = {}
    for epoch in range(10):
        rows = {mode: json.loads((root / mode / 'quick_test' / f'epoch{epoch:03d}' / 'metrics.json').read_text())
            for mode in ('x0', 'v')}
        matched_age[str(epoch)] = {condition: {name: rows['v']['conditions'][condition]['metrics'][name]
            - control['metrics'][name] for name in ('pose_AP', 'bbox_AP')}
            for condition, control in rows['x0']['conditions'].items()}
    summary = dict(arms={mode:dict(checkpoint=report['checkpoint'], checkpoint_sha256=report['checkpoint_sha256'],
            conditions=report['conditions'], selected_epoch=report['epoch'], continuation_steps=report['continuation_steps'])
            for mode,report in reports.items()},
        selected_checkpoint_fullVAL_AP_differences=differences,
        fullVAL_scope='Best selected independently after equal update budgets; selected checkpoint ages may differ',
        matched_age_fixed32VAL_AP_differences=matched_age, conditional_recovery=recovery,
        first_augmented_batches_identical=True, automatic_promotion=False,
        decision='Review positive-noise recovery over direct decoding and matched-age pose AP with bbox retained; '
            'identity alone is insufficient. No further training is launched.')
    (root / 'comparison.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    manifest = json.loads((root / 'launch_manifest.json').read_text())
    workspace = root / 'workspace'
    for mode in ('v', 'x0'):
        output = root / mode
        output.mkdir(exist_ok=True)
        (root / 'queue_status.json').write_text(json.dumps(dict(phase='running', active=mode,
            queued=['x0'] if mode=='v' else [], pid=os.getpid())) + '\n')
        command = [sys.executable, '-u', str(workspace / 'scripts/training/train_poseflow_v44_comparison.py'),
            f'hydra.run.dir={output}', f'out_dir={root}', f'arm={mode}',
            f'source_checkpoint={root}/source.ckpt', f'source_sha256={manifest["source_sha256"]}',
            f'expected_initial_model_sha256={manifest["preflight"]["initial_hashes"][mode]}']
        with (output / 'launch.log').open('ab') as log:
            process = subprocess.run(command, cwd=workspace, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT)
        if process.returncode:
            (root / 'queue_status.json').write_text(json.dumps(dict(phase='failed', arm=mode,
                exit_code=process.returncode, remaining_not_started=mode=='v')) + '\n')
            raise SystemExit(process.returncode)
    summarize_comparison(root)
    (root / 'queue_status.json').write_text(json.dumps(dict(phase='complete', arms=['v','x0'], further_training=False)) + '\n')


if __name__ == '__main__':
    main()

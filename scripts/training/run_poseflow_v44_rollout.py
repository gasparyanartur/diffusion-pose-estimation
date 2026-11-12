import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def summarize_rollout(root):
    reports = {arm: json.loads((root/arm/'final_results.json').read_text()) for arm in ('forward', 'rollout')}
    batches = {arm: json.loads((root/arm/'first_batches.json').read_text()) for arm in reports}
    if batches['forward'] != batches['rollout']:
        raise ValueError('Paired initial augmented batches differ between rollout arms')
    control, candidate = reports['forward'], reports['rollout']
    def metric(report, steps, name):
        return report['conditions'][f'noise_var_100/steps_{steps}']['metrics'][name]
    full_gate = (metric(candidate, 4, 'pose_AP') > metric(control, 4, 'pose_AP')
        and metric(candidate, 1, 'pose_AP') >= metric(control, 1, 'pose_AP')-.01
        and all(metric(candidate, steps, 'bbox_AP') >= metric(control, steps, 'bbox_AP') for steps in (1,4)))
    recovery_gate = all(candidate['small_noise'][f'noise_{level}/output_paired_OKS_mean']
        >= control['small_noise'][f'noise_{level}/output_paired_OKS_mean'] for level in ('1','5'))
    matched_age = {}
    for epoch in range(2):
        rows = {arm: json.loads((root/arm/'quick_test'/f'epoch{epoch:03d}'/'metrics.json').read_text()) for arm in reports}
        matched_age[str(epoch)] = {condition: {name: rows['rollout']['conditions'][condition]['metrics'][name]
            - row['metrics'][name] for name in ('pose_AP', 'bbox_AP')}
            for condition, row in rows['forward']['conditions'].items()}
    last = matched_age['1']
    matched_gate = (last['noise_var_100/steps_4']['pose_AP'] > 0
        and last['noise_var_100/steps_4']['bbox_AP'] >= 0
        and last['noise_var_100/steps_1']['pose_AP'] >= -.01)
    trajectory_control, trajectory_candidate = (report['last_checkpoint_trajectory'] for report in (control, candidate))
    gap_gate = (trajectory_candidate['OKS_gap'] < trajectory_control['OKS_gap']
        and trajectory_candidate['branches']['generated']['paired_OKS_mean']
        > trajectory_control['branches']['generated']['paired_OKS_mean'])
    gates = dict(selected_fullVAL=full_gate, matched_age_fixed32=matched_gate,
        small_noise_retained=recovery_gate, trajectory_gap_reduced_with_generated_gain=gap_gate)
    summary = dict(arms={arm: dict(checkpoint=row['checkpoint'], checkpoint_sha256=row['checkpoint_sha256'],
        selected_epoch=row['epoch'], continuation_steps=row['continuation_steps'], conditions=row['conditions'])
        for arm, row in reports.items()}, gates=gates, candidate_passes_all_gates=all(gates.values()),
        matched_age_fixed32_AP_differences=matched_age, first_augmented_batches_identical=True,
        fullVAL_scope='Independently selected best checkpoints after equal budgets; selected ages may differ',
        trajectory={arm: report['last_checkpoint_trajectory'] for arm, report in reports.items()},
        automatic_promotion=False, further_training=False)
    (root/'comparison.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root.resolve()
    manifest = json.loads((root/'launch_manifest.json').read_text())
    for arm in ('rollout', 'forward'):
        output = root/arm
        output.mkdir(exist_ok=True)
        (root/'queue_status.json').write_text(json.dumps(dict(phase='running', active=arm,
            queued=['forward'] if arm == 'rollout' else [], pid=os.getpid()))+'\n')
        command = [sys.executable, '-u', str(root/'workspace/scripts/training/train_poseflow_v44_rollout.py'),
            f'hydra.run.dir={output}', f'out_dir={root}', f'arm={arm}', f'source_checkpoint={root}/source.ckpt',
            f'source_sha256={manifest["source_sha256"]}',
            f'expected_initial_model_sha256={manifest["preflight"]["initial_hashes"][arm]}']
        with (output/'launch.log').open('ab') as log:
            result = subprocess.run(command, cwd=root/'workspace', env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            (root/'queue_status.json').write_text(json.dumps(dict(phase='failed', arm=arm, exit_code=result.returncode))+'\n')
            raise SystemExit(result.returncode)
    summarize_rollout(root)
    (root/'queue_status.json').write_text(json.dumps(dict(phase='complete', arms=['rollout','forward'], further_training=False))+'\n')


if __name__ == '__main__':
    main()

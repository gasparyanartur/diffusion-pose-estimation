from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time
from zoneinfo import ZoneInfo

ARTIFACTS = Path('/storage/outputs/pose-graph-diff/diagnostics/h_control_information_20261023_epoch065')
REPO = Path(__file__).resolve().parents[3]
REPORT = REPO/'tmp/h_control_information_suite/execution_snapshot.md'


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def percentage(measure):
    return f"{100*measure['mean']:.1f}% [{100*measure['ci95'][0]:.1f}, {100*measure['ci95'][1]:.1f}]"


def draw_depth(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    stages = ['B', 'E', 'G1', 'G2']
    for arm, color in [('initialization', '#777777'), ('control', '#2166ac'), ('h', '#b2182b')]:
        values = [result['models'][arm][stage]['main_pair_accuracy'] for stage in stages]
        mean = np.array([value['mean'] for value in values])*100
        low = mean-np.array([value['ci95'][0] for value in values])*100
        high = np.array([value['ci95'][1] for value in values])*100-mean
        axes[0].errorbar(stages, mean, yerr=[low, high], marker='o', label=arm, color=color, capsize=3)
    axes[0].axhline(50, color='black', linestyle=':', label='shared2D tie')
    axes[0].axhline(60, color='black', linestyle='--', alpha=.4, label='useful margin')
    axes[0].set(title='Held-out depth order by stage', ylabel='Within-pair accuracy (%)', ylim=(0, 103))
    axes[0].legend(fontsize=8)
    names = ['G2_plus_elbow', 'G2_plus_h', 'G2_plus_shuffled_h']
    values = [result['models']['h'][name]['main_pair_accuracy'] for name in names]
    mean = np.array([value['mean'] for value in values])*100
    low = mean-np.array([value['ci95'][0] for value in values])*100
    high = np.array([value['ci95'][1] for value in values])*100-mean
    axes[1].bar(['wrist+elbow', 'wrist+h', 'wrist+shuffled h'], mean, color=['#2166ac', '#b2182b', '#aaaaaa'])
    axes[1].errorbar(range(3), mean, yerr=[low, high], color='black', linestyle='', capsize=4)
    axes[1].set(title='Equal512-input readers inside h model', ylabel='Within-pair accuracy (%)', ylim=(0, 103))
    figure.suptitle('Synthetic depth-label accessibility; not evidence of coherent3D reasoning')
    output = ARTIFACTS/'main_C_readouts/depth_summary.png'
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def write_report():
    states = {study:read_json(ARTIFACTS/f'main_{study}_logs/status.json') for study in ('B', 'C')}
    results = {study:read_json(ARTIFACTS/f'main_{study}_readouts/results.json') for study in ('B', 'C')}
    terminal = all(state and (state.get('complete') or state.get('phase') == 'failed') for state in states.values())
    lines = ['# Remaining information probes: execution and results', '',
             'Updated '+datetime.now(ZoneInfo('Europe/Stockholm')).isoformat(timespec='seconds')+'.', '',
             '**Complete.**' if all(state and state.get('complete') for state in states.values()) else '**Main execution remains incomplete.**', '',
             'The A ownership study remains rejected. Both B/C admission cohorts passed visible-wrist localization64/64 in each pinned model. RGB controls distinguished63/64 B images and64/64 C images. All data are synthetic renders. Checkpoints are the fixed seed42 V24 epoch065/step99000 pair; the initialization reference contains transferred ED-Pose weights. No estimator was trained or fine-tuned.', '',
             'Scores below are family-bootstrap95% intervals (10000 draws), conditional on common box-matched families for model-feature readers. Scene replication does not replicate model training seeds.', '',
             '## Execution status', '', '| Study | Stage | Split | Complete |', '|---|---|---|---|']
    for study, state in states.items():
        state = state or {}
        lines.append(f"|{study}|{state.get('phase', 'not started')}|{state.get('split', '')}|{state.get('complete', False)}|")
        if state.get('error'):
            lines += ['', f"{study} failed: `{state['error']}`. No missing result should be treated as a negative capability finding."]
    lines += ['', '## Foreshortening: native behavior versus accessibility', '']
    behavior = read_json(ARTIFACTS/'main_B_readouts/native_behavior.json')
    if behavior:
        lines += ['| Model | Visible wrists correct | Hidden wrists correct | Both hidden members correct | Strict single-candidate pair |', '|---|---:|---:|---|---|']
        for arm in ('control', 'h'):
            main, clear = behavior[arm]['main'], behavior[arm]['clear']
            lines.append(f"|{arm}|{clear['correct']}/{clear['images']}|{main['correct']}/{main['images']}|{percentage(main['paired_correct'])}|{percentage(main['strict_single_candidate_pair_correct'])}|")
        lines += ['', 'Native h-minus-control paired difference: '+percentage(behavior['h_minus_control_native_pair_correct'])+'. Zero-event bootstrap intervals can degenerate; they do not establish a population probability of exactly zero.']
    else:
        lines += ['Full native results pending. The admitted32-family pilot had visible64/64 and hidden0/64 for each model; those are pilot results only.']
    if results['B']:
        lines += ['', 'B task readers predict short/long from the shared VISIBLE elbow; primary success requires both members classified correctly. They receive new task supervision and are not native localizers.', '', '| Source | Control paired classification | h-model paired classification |', '|---|---|---|']
        for source in ('B', 'E', 'G1', 'G2', 'native_prediction'):
            lines.append('|'+source+'|'+percentage(results['B']['models']['control'][source]['main_pair_accuracy'])+'|'+percentage(results['B']['models']['h'][source]['main_pair_accuracy'])+'|')
        lines += ['', 'If frozen features support this task while the native wrist predictions fail, that establishes task information accessible to the fitted reader without reliable native utilization. It does not establish that the original decoder already knows the geometric rule. C depth-sign scores are not an explanation for B depth-magnitude failure.']
    lines += ['', '## Depth order with complete2D held fixed', '']
    if results['C']:
        result = results['C']
        lines += ['| Stage | Exact initialization | Trained control | Trained h |', '|---|---|---|---|']
        for source in ('B', 'E', 'G1', 'G2', 'native_prediction'):
            lines.append('|'+source+'|'+ '|'.join(percentage(result['models'][arm][source]['main_pair_accuracy']) for arm in ('initialization', 'control', 'h'))+'|')
        lines += ['', 'B = projected finest backbone map; E = encoded map; G1/G2 = target wrist token after the first/final pose layer. Native prediction is the aligned predicted2D pose/box/score, which may carry depth-dependent errors. Ordinary feature readers use all256 channels; C is selected only on validation. Main appearance alone is used for fitting.', '',
                  '| Model final stage | Surface cue stratum | Torso-occlusion stratum | Appearance repeat |', '|---|---|---|---|']
        for arm in ('initialization', 'control', 'h'):
            source = result['models'][arm]['G2']
            strata = source['strata']
            lines.append('|'+arm+'|'+percentage(strata['surface'])+'|'+percentage(strata['occlusion'])+'|'+percentage(source['nuisance_pair_accuracy'])+'|')
        lines += ['', '### Incremental h under matched reader size', '', '|512-input source, inside h model|Depth pair accuracy|', '|---|---|']
        for source in ('G2_plus_elbow', 'G2_plus_h', 'G2_plus_shuffled_h'):
            lines.append('|'+source+'|'+percentage(result['models']['h'][source]['main_pair_accuracy'])+'|')
        lines += ['', '### Paired contrasts', '', '| Contrast | Difference in percentage points,95% interval |', '|---|---|']
        for name, measure in result['paired_contrasts'].items():
            lines.append('|'+name+'|'+percentage(measure)+'|')
        lines += ['', 'The h margin is5points above BOTH equal-size controls; C accessibility margin is10points above the shared2D bound50%. A confidence interval crossing the relevant margin remains unresolved. More accessible information is a restricted-reader result, not proof of new Shannon information or native utilization.', '', '### Correspondence coverage', '']
        for arm, splits in result['retention'].items():
            test = splits['test']
            retained = test['retained_families']/test['original_families']
            lines.append(f"- {arm}: {test['retained_families']}/{test['original_families']} test families ({100*retained:.1f}%)."+(' Below90%: no population-level representation claim.' if retained < .9 else ''))
        figure = draw_depth(result)
        lines += ['', f'![Depth accessibility summary]({figure})']
    else:
        lines += ['Stage, initialization, h and appearance readouts pending. No depth representation result is available yet.']
    audit = read_json(ARTIFACTS/'main_C_readouts/image_audits.json')
    lines += ['', '## Shortcut and interpretation audit', '',
              'In the pilots, head/leg crops often ordered paired images despite weak per-image classification. Randomizing render order did not remove this. Pose/render/shadow proxies remain a competing explanation; label accessibility alone cannot establish coherent3D pose or reasoning.']
    if audit:
        lines += ['', '| Held-out image control | Pair order | Per-image accuracy |', '|---|---|---|']
        for source in ('raw2D', 'outside_arm'):
            lines.append('|'+source+'|'+percentage(audit[source]['test_pair_order'])+f"|{100*audit[source]['test_image_accuracy']:.1f}%|")
        lines += ['', '**Raw2D leakage invalidates depth interpretation:** '+str(audit['raw2D_leakage_invalidates'])+'.',
                  '**Outside-arm ordering warning:** '+str(audit['outside_arm_warning'])+'.',
                  'Literal shared2D controls tie at50%. The image auditors use all physical families, independently of box correspondence. An outside-arm success means forearm-specific or coherent-depth interpretations are not isolated by the current data.']
    lines += ['', '## Provenance and artifacts', '',
              'Current methods, construction history and interpretation: [consolidated investigation](/home/gasparyanartur/dev/pose-graph-diff/docs/reports/2026-10-24-image-information-report.md).', '',
              'Numerical results, fitted linear coefficients, split IDs, selected feature caches, compressed raw native predictions, synthetic AP tables, scoped physical/ray audits and progress logs are under:', '',
              '`/storage/outputs/pose-graph-diff/diagnostics/h_control_information_20261023_epoch065/`.', '',
              'All estimator forwards use fixed source/config/checkpoints and common per-family noise. Only generated unselected mesh intermediates are removed after physical selection; parameters/rejection audits and every selected mesh are preserved.']
    temporary = REPORT.with_suffix('.partial.md')
    temporary.write_text('\n'.join(lines)+'\n')
    temporary.replace(REPORT)
    status_path = REPO/'tmp/h_control_information_suite/status.json'
    status = read_json(status_path) or {}
    complete = all(state and state.get('complete') for state in states.values())
    status.update(stage='B_C_main_complete' if complete else ('B_C_main_failed' if terminal else 'B_C_main_running'),
                  suite_complete=complete, B=states['B'], C=states['C'],
                  report=str(REPO/'docs/reports/2026-10-24-image-information-report.md'), diagnostic_snapshot=str(REPORT), representation_probe_fits=sum(len(model) for result in results.values()
                    if result for model in result['models'].values()), native_call_total_before_main=2402)
    status_path.write_text(json.dumps(status, indent=2)+'\n')
    return terminal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    deadline = time.monotonic()+6*3600
    while True:
        terminal = write_report()
        if terminal or not args.watch or time.monotonic() >= deadline:
            return
        time.sleep(10)


if __name__ == '__main__':
    main()

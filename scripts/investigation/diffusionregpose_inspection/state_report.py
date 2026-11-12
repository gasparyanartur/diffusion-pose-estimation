from __future__ import annotations

import argparse
import json
from pathlib import Path


CONDITIONS = {
    'seed0': 'Baseline',
    'first_joint_image_bias_only': 'First pose: joint image read off',
    'final_joint_image_bias_only': 'Final pose: joint image read off',
    'first_joint_feature_updates_off': 'First pose: joint feature updates off',
    'first_updates_off_and_final_image_bias_only': 'First updates + final image read off',
}


def write_report(result, output, report_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output, report_path = Path(output), Path(report_path)
    times = [999, 665, 332]
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1, 1.25]})
    movements = {
        'state_mean_shift_px': 'Diffusion state (pixel equivalent)',
        'incoming_reference_mean_shift_px': 'Incoming joint references',
        'first_pose_mean_shift_px': 'First pose prediction',
        'final_mean_joint_shift_px': 'Final pose prediction',
    }
    for key, label in movements.items():
        axes[0].plot([1, 2, 3], [result['sensitivity'][str(time)][key]['mean'] for time in times],
                     marker='o', label=label, linewidth=2)
    axes[0].set(yscale='log', xticks=[1, 2, 3], xticklabels=['1 (t=999)', '2 (t=665)', '3 (t=332)'],
                xlabel='Model call within S=3', ylabel='Mean joint displacement in pixels (log scale)',
                title='Vary state; hold image and timestep fixed')
    axes[0].grid(alpha=.25, axis='y')
    axes[0].legend(fontsize=9, loc='center left')
    labels = ['Baseline', 'First joint image read off', 'Final joint image read off',
              'First joint feature updates off', 'First updates + final image read off']
    values = [100 * result['metrics'][f'{condition}_t332']['AP'] for condition in CONDITIONS]
    bars = axes[1].barh(labels, values, color=['#31824a', '#31824a', '#c68023', '#b64545', '#b64545'])
    axes[1].invert_yaxis()
    axes[1].set(xlim=(0, 60), xlabel='Keypoint AP (0–100)', title='Mask one route at a time (call 3, t=332)')
    for bar, value in zip(bars, values):
        axes[1].text(value + .6, bar.get_y() + bar.get_height() / 2, f'{value:.2f}', va='center')
    axes[1].grid(alpha=.2, axis='x')
    figure.suptitle('DiffusionRegPose: the first pose layer processes features, but its joint image read contributes zero', fontsize=13)
    figure.text(.5, .025, '32 test images · frozen epoch 79 · 5 state seeds · masks preserve box rows and image-projection bias',
                ha='center', fontsize=10)
    figure.tight_layout(rect=(0, .06, 1, .94))
    figure.savefig(output / 'summary.png', dpi=160)
    plt.close(figure)

    movement_rows = '\n'.join(
        f'| {index + 1} / {time} | ' + ' | '.join(
            f"{result['sensitivity'][str(time)][key]['mean']:.4f}" for key in movements) + ' |'
        for index, time in enumerate(times))
    mask_rows = '\n'.join(
        f'| {label} | ' + ' | '.join(f"{100 * result['metrics'][f'{condition}_t{time}']['AP']:.2f}"
                                    for time in times) + ' |'
        for condition, label in CONDITIONS.items())
    geometry = result['reference_geometry']
    reference_count = sum(row['joint_reference_count'] for row in geometry.values())
    metrics = result['metrics']
    final_range = [100 * metrics[f'seed{seed}_t332']['AP'] for seed in range(result['manifest']['seed_count'])]
    checkpoint = result['manifest']['settings']['checkpoint']
    text = f'''# DiffusionRegPose: state sensitivity and intermediate-stage ablations

Date: 2026-10-24. Same reduced 256/427 model and fixed 32-image test subset as
the inspection notebook. Frozen epoch-79 checkpoint; no training or model repair.

The model depends very weakly on the diffusion state in this test. The first pose
layer's **joint image read contributes exactly zero**, but the **layer itself is
essential**: removing its joint self-attention and feed-forward residual updates
collapses keypoint AP. The final layer's joint image read also matters strongly.
These findings distinguish a useless image-reading route from a useless stage.

![State movement and route masks](../../{output}/summary.png)

## Fixed image, fixed timestep, varied diffusion state

For each image, run five native S=3 trajectories with initial seeds
`1729 + image_id + 100000 * seed_index`, seed indices 0 through 4. Compare each
alternative with seed zero at the same image and timestep. Later states come from
the native deterministic DDIM trajectory, rather than arbitrary Gaussian states
at later times. The image encoder and initial box-query selection are unchanged
at a fixed image/time. Selected box-query slots were checked for exact equality.

Each entry below averages joint displacement across the 50 corresponding person
slots, 14 joints, 32 images and four alternative seeds. Units are original-image
pixels. Input-state displacement is **pixel equivalent**, using the model's
conversion `xy = (state / 6 + 1) / 2`; it does not mean those states are valid poses.
The comparison uses corresponding raw slots before pose NMS, without matching
alternative predictions to ground truth or reassigning person identities.

| Call / timestep | Input state, pixel equivalent | Incoming references | First pose prediction | Final prediction |
|---|---:|---:|---:|---:|
{movement_rows}

Final-call AP over the five seeds ranges from **{min(final_range):.5f} to
{max(final_range):.5f}**. The final joint shifts above are about one hundredth
of a pixel on average. This is strong evidence of weak state sensitivity on this
checkpoint/subset; it is not a claim of exact state independence or a full-test result.

## Why the incoming references can move while predictions barely move

Across all **{reference_count:,} joint-reference observations** (32 images × 5 seeds ×
3 calls × 50 groups × 14 joints), both incoming XY coordinates of the first pose
layer are negative. No reference center is inside the image. For every compared
seed, clamping those XY values to [0,1] produces exactly the same `(0,0)`.

The decoder's `inverse_sigmoid` clamps coordinates before taking their logit.
Consequently, the coordinate residual path receives the same boundary logit even
when the incoming reference moves. The raw reference still enters positional
embeddings, so a small remaining state effect is possible and is measured above.
This explains a concrete loss of geometric variation; it does not by itself prove
how training arrived at it. The first pose joint cross-attention output equals
its output-projection bias exactly in all 480 unmasked model calls, confirming
that the image-value contribution is zero throughout this expanded diagnostic.

## Frozen masks and performance

For each of the 96 seed-zero image/time inputs, replay one model call with the
same image, timestep and incoming state. Change only the named route. These are
**conditional single-call interventions**, not complete diffusion trajectories
run with masks enabled from the start. The final-call baseline exactly reproduces
the notebook's S=3 predictions and native postprocessing.

“First pose” means decoder layer 2; “final pose” means decoder layer 3. Layer 1
is the preceding box-only decoder layer. AP values below are on the 0–100 scale.

| Condition | Call 1 / t=999 | Call 2 / t=665 | Call 3 / t=332 |
|---|---:|---:|---:|
{mask_rows}

- **Joint image read off:** replace only joint-query cross-attention output with
  the learned output-projection bias. This removes image values without deleting
  the bias or changing box-query outputs. The first-pose mask leaves raw logits,
  boxes and joints unchanged; the final-pose mask removes a useful read.
- **First joint feature updates off:** zero only joint-query self-attention and
  feed-forward residual contributions in the first pose layer. Keep residual
  inputs, all normalizations, image attention, box-query updates and coordinate
  prediction heads. This is not equivalent to deleting the entire layer.
- **Combined mask:** remove those first-layer updates and the final joint image
  read together. Both this condition and the first-update mask have AP zero,
  so their AP interaction is floor-limited and cannot quantify compensation.

The masks preserve box rows at the intervention point. They do not guarantee
identical downstream box predictions or detections: later query interactions and
pose-dependent NMS can change retained detections.

## What this says about compensation

The broad hypothesis “an ineffective intermediate stage is rescued by the final
stage” is not supported by these ablations. The first pose stage is necessary,
and the final stage does **not** rescue its removal. A more precise description
of the observed computation is: image-conditioned query features enter a first
pose stage that performs essential feature/geometry processing despite a dead
joint image read, followed by a useful final joint image read.

This is consistent with the model relying on other routes around that dead read.
It does not establish that it *learned to compensate for it*, or what performance
would be possible with a working intermediate image read. Abrupt activation masks
also create states absent during normal inference.

A sharper follow-up would **attenuate** the first joint updates gradually (for
example 100%, 90%, 75%, 50% retained), separately for self-attention and the FFN,
then cross each level with the final image read on/off. Use baseline-fixed
GT assignments and visible-joint errors as well as AP to avoid the zero-AP floor.
Greater downstream error reduction under moderate perturbation would support a
functional recovery role. Establishing a learned compensatory mechanism needs
matched training controls that repair the image-read route while preserving the
other components. These additional runs were not launched in this iteration.

## Verification and reproducibility

- Checkpoint: `{checkpoint}`.
- Checkpoint SHA256: `{result['manifest']['checkpoint_sha256']}`.
- Split, IDs, resolved config, annotation hash, source hashes, seeds and schedule:
  [`manifest.json`](../../{output}/manifest.json).
- Final verified run: **864 model calls**: 480 native trajectory calls and
  384 conditional masked calls. Five seeds; 32 images; resize 256/427; eta=0.
- Seed-zero final logits, boxes, keypoints and final diffusion state match the
  previous inspection cache exactly for every image. Native postprocessed scores,
  boxes, keypoints and query IDs also match exactly, on the original GPU device.
- First joint image-value contribution has absolute maximum zero. Every mask hook
  is removed on exit, including exceptions. Focused diagnostic tests: 12 passed.
- Initial run used CPU postprocessing of saved activations, which changed native
  tie/threshold behavior and AP. It is superseded. All reported AP uses the repeated
  GPU-postprocessed run with exact cached-result parity; geometric findings agreed.
- Numerical results: [`results.json`](../../{output}/results.json).
  Small CPU observations: [`observations.pt`](../../{output}/observations.pt).
  Per-condition native prediction JSONs are beside these files.

Reproduce from the project root:

```bash
OMP_NUM_THREADS=2 TMPDIR="$PWD/tmp" venv/bin/python -m scripts.investigation.diffusionregpose_inspection.state_sensitivity --output {output}
OMP_NUM_THREADS=2 TMPDIR="$PWD/tmp" venv/bin/python -m scripts.investigation.diffusionregpose_inspection.state_report --output {output}
```
'''
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='tmp/diffusionregpose_state_sensitivity_verified')
    parser.add_argument('--report', default='tmp/diffusionregpose_state_sensitivity_verified/methods_snapshot.md')
    args = parser.parse_args()
    result = json.loads((Path(args.output) / 'results.json').read_text())
    write_report(result, args.output, args.report)

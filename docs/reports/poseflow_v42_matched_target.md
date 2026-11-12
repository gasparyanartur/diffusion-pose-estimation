# V42 matched targets from noisy inputs

`benchmark_poseflow_v42_matched_target` retains repaired V42's whole-person
model, box-center residual pose reader, coordinate corruption/DDIM, task weights,
EDPose initialization and 80-epoch seed-42 recipe. It changes training assignment.

After native target preparation and coordinate noising/clamping, normalized noisy
XY is matched one-to-one to all valid supervised people with Hungarian assignment.
The cost is the existing visibility-masked joint L1 coefficient10 plus OKS
coefficient4, using the native joint sigmas and target area. No class score,
predicted output or original corruption-source identity enters this assignment.
Even an all-zero normalized pose uses pose costs rather than the native
zero-keypoint sentinel for box-only stages.

This mapping indexes complete clean person records for final and post-selection
pose-auxiliary classification/box/keypoint/OKS losses. Those layers preserve the
same 50 input slot order. Unmatched slots are classification negatives and have no
geometry loss. Earlier 200-query box stages, encoder proposals and box/label DN
supervision retain their original matching rules. Random padding can become
matched; cropped-out people retained in the diffusion bank remain excluded from
the supervised target bank. Empty images and more GT than queries retain native
normalization/capacity semantics.

Inference is unchanged and needs no target matcher. The Gaussian state is still
coordinates and there is no JEPA objective. Training still uses timestep zero;
this experiment isolates assignment rather than resolving the noise-level gap.
Noisy geometry may conflict with the image proposal attached to its slot, and
severe-noise Hungarian targets may be arbitrary. Learned AP must test the idea.

Implementation:
- `src/models/pipelines/pose_estimators/poseflow/training_v42_matched_target.py`
- Native criterion accepts optional explicit per-auxiliary matches, defaultNone.
- Native pipeline shares its RNG-fenced forward helper; existing defaults retain
  bitwise-identical losses and RNG behavior.
- V42 config exposes a class-level pipeline factory without changing serialized
  base configs or model initialization. New configs explicitly save
  `target_assignment: noisy_input_hungarian`.

```bash
python scripts/training/train_diffusionregpose.py --config-name benchmark_poseflow_v42_matched_target
```

The frozen launch wrapper additionally evaluates fixed32 held-out test images
after each epoch at1/4/8calls and all 8,000 test images after epoch 79, with native
pose NMS, seed42+image_id Gaussian sources and paired conditions. Canonical
pose_AP/bbox_AP tags share TensorBoard33331's sixpins with V42 and prior runs.
`train/input_match/{cost,target_coverage,output_disagreement}` and validation
counterparts monitor assignment. Output Hungarian is computed for disagreement
only and never selects the variant's pose targets.

## Verification and running experiment

Seven intended-red tests were followed by 17 focused checks and 91 native
criterion/runner/V34 regressions (108 total), all passing. A frozen-code audit
checks 24 matching-cost cases and 36 complete legacy loss fields, including the
old criterion with identical configured coefficients: values and RNG are bitwise
identical. Small-model initialization, noising and seeded 1/4/8-call inference also
match V42 exactly.

Native FP32 CUDA verification used a real augmented batch 4 with accumulation 2,
two finite optimizer updates, strict model/AdamW/scheduler restoration and exact
restored predictions. A controlled continued update agreed within 1.28e-7.
All 481 surviving native tensors match; the full initial V42 model digest is
`ef71a0f0543d7109117605ee9e162052d81aaff935c3ea685e7e21db27373628`.
Two-image K1/K4/K8 evaluation only checks evaluator wiring and finite outputs;
it is not learned-performance evidence.

Full 80-epoch run started 2026-11-04 in tmux `poseflow-v42-matched-target`,
PID552828, at `/storage/outputs/pose-graph-diff/results/poseflow_v42_matched_target/20261104_012819`. The worker verified 628 frozen source files and 5
input hashes before training. The original V42 control remains running from
its own frozen source. No two-epoch cutoff or JEPA loss was introduced.

Launch verified at epoch 0, optimizer step400, with finite loss and
gradients. First learned AP is pending completion of epoch 0. TensorBoard33331
indexes `V42-matched-target/.` alongside the preserved 42 configured roots
(46 streams including resumes), and its training scalar is populated. All six
requested quick-test pose/box AP pins are present. Their new-run AP curves
populate after each epoch; missing measurements are not filled.

Run evidence: `manifest.json`, `launch.log`, `status.json`,
`launch_verification.json`, verification logs, resolved config and checkpoints.
Local evidence and helpers live under `tmp/v42_matched_target/`. Follow-up is
same-age fixed32 Gaussian AP and assignment diagnostics against repaired V42;
full-test results remain scheduled after epoch 79.


## Failure diagnosis, 2026-11-04

The launch is operational, but Gaussian-start pose estimation has failed. At
completed epoch31/48,000 updates, fixed32 K1/K4/K8 pose AP is 0/0/0; the same-age
V42 control gives 9.10/26.01/30.37 AP. These are subset diagnostics on the same
32 images and seeds, not full-test results. The worker remains running.

The frozen source matches the current implementation. Noisy input slots are
injected after top-k selection and retain their order through pose outputs.
Input assignment coverage is 1; epoch31 train input/output disagreement is
0.000267 and validation disagreement is zero. The inherited target builder
always corrupts at timestep zero, leaving GT-derived inputs almost clean.

A frozen epoch31 checkpoint was evaluated on the same fixed32 images with a
2x2 coordinate-input/timestep ablation. Almost-clean GT inputs give pose AP
12.74 at t0 and 11.05 at t999; Gaussian inputs give 0 at either timestep.
The Gaussian/t999 condition reproduces the saved K1 box AP exactly (0.5276 on
an AP100 scale). GT-conditioned measurements are recovery diagnostics and do
not represent inference performance. Evidence is in
`tmp/v42_matched_target/diagnosis/input_time_ablation.json` and `diagnosis.json`.

This demonstrates failure under Gaussian coordinate inputs even with training's
timestep. It supports a coordinate-noise coverage problem; it does not establish
that random-t training will solve it. A separate uniform-t matched-target repair
is proposed in `.plan` for review. No production changes or job interruptions
were made during diagnosis.


2026-11-04: User cancelled this failed matched-target experiment. PID552828 has exited; the interruption checkpoint, frozen source, logs and events are preserved. Recorded cancellation status was epoch33/step50950. The requested replacement is the ordinary output-Hungarian box-center run with uniform diffusion training timesteps, at `/storage/outputs/pose-graph-diff/results/poseflow_v42_uniform_t/20261104_133538`; this matched-target experiment was not restarted.


[2026-11-04] Failed matched-target semantics re-reviewed from frozen source
User asked whether we already trained proposal-to-noisy-state matching and should revisit the failure. Verified original helper against manifest/current helper: exact SHA9c90108e... . Its noisy-pose->clean-GT L1/OKS assignment runs BEFORE image forward/proposal creation. Selected image proposals remain indexwise attached to noisy states; no proposal<->incoming-state association occurs. Frozen noiser also hardcodes t0. Cancelled epoch33/50950, completed epoch32 poseAP0/0/0. Saved epoch31 conditional GTt0 AP12.744 versus Gaussian0 at either t0/999; this is a failed recipe, not a controlled rejection of the proposed new association or evidence uniform-t alone repairs it. Current paired variant instead matches proposal->GT before source construction/noising and trains uniform. New direct proposal<->noisy-state mechanism remains untested. Report docs/reports/poseflow_v42_matching_review.md; evidence tmp/v42_matched_target/review/audit.json. No restart/model/sampler/dashboard changes.

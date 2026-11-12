# V42 noisy-proposal pairing during training and inference

Approved 2026-11-05. New config `benchmark_poseflow_v42_noisy_proposal`, run name
`v42-noisy-proposal-paired`. This tests the previously inference-only association
under consistent training. It is separate from the cancelled t0 matched-target
experiment, which did not match image proposals to noisy poses.

## Assignment and labels

Start from paired-proposal V42, retaining its source bank and random draws:
selected image references are matched to valid GT records, clean coordinates are
placed in those slots, remaining slots receive Gaussian padding, and native
coordinate q_sample uses independent uniform per-image t0..999. The same actual
times condition the model. After corruption, the existing configured Hungarian
matcher associates current proposal boxes with enclosing boxes of the ACTUAL
noisy joints. The association uses no GT, visibility or source provenance. Its
complete50x50 permutation and box adapter are identical to sampling_v42.py.

Reorder noisy states into current proposal order. Invert that permutation to
move each source's complete GT target to its new query; background sources stay
background. Final and post-selection pose auxiliaries use the same fixed map.
Earlier box/encoder/DN losses retain native matching. There is one image forward,
no extra parameters and no JEPA. Initial weights and architecture match the
paired control exactly, including box-center readout and60 attention reads.

A clean source A remains supervised by A even if noisy geometry associates it
with a proposal over B. At high noise, identity may be effectively erased and
these labels difficult to infer. This is an experimental risk. Keeping labels
on the image proposal instead would define a different objective and is excluded.

## Evaluation and launch contract

Training and validation loss use this new association. Epoch-end fixed32
K1/K4/K8 and final8000 at epoch79 use the matching sampler, including the first
Gaussian call and every later call. The same ordered state enters encoding,
epsilon reconstruction and the DDIM residual/update. No inference GT enters.
Evaluator injection/hook defaults preserve native policies for older configs;
reports identify the actual sampler.

Full80 epochs, fresh EDPose seed42, existing data/resize256427, FP32 batch4 with
accumulation2/clip.1 and unchanged optimizer. No pilot cutoff. Freeze code/config
and input hashes; preserve controls and historical event files. Add a separate
stream to0.0.0.0:33332, preserving prior roots and six K1/K4/K8 pose/box pins.

## Verification

The focused new test first failed at the missing module. Seven new checks and146
V42/native diffusion/criterion/evaluator regression cases pass. Native CUDA
preflight performs two real accumulated optimizer updates, exact initial-weight
comparison and481 surviving native-tensor comparisons, strict model/optimizer/
scheduler restore and a controlled next update (maximum absolute difference
4.29e-8). Validation loss is finite; matched K1/K4/K8 sampling is finite on two
images. These are wiring checks, not learned performance. Source/target identity
is checked after the actual permutation; sampled q_sample/model times agree.

Artifacts: `tmp/v42_noisy_proposal/` contains tests, preflight, launch helpers,
run pointer and dashboard verification. Run status/manifest and source snapshot
record live launch evidence. Actual worker audits verify complete permutations,
source-following target/background labels, encoder state and time equality,
timestep coverage and changed-assignment fraction by noise bin. Same-age AP and
confidence comparisons are needed to judge whether this improves multi-call use.

Periodic validation AP also uses the matching sampler through `_sample_validation_outputs`; the inherited native default is retained for older pipelines. A failing regression exposed the missing hook, and the real validation-step route is covered. The first startup snapshot was superseded before its first epoch; the full experiment restarts fresh with this hook.


## Live launch

Corrected full80 job started2026-11-05T08:57+01:00, PID869864, tmux
`poseflow-v42-noisy-proposal`. Root:
`/storage/outputs/pose-graph-diff/results/poseflow_v42_noisy_proposal/20261105_085659`.
Worker verifies677 frozen source hashes and5 inputs, exact initial weights,
uniform time conditioning and source-following labels. At the current launch
audit: epoch0, optimizer step100, finite loss;
2751 positive sources checked,
zero mismatches, all states preserved. First validation loss and fixed32 AP are
pending epoch completion. This is launch health, not trained AP evidence.

The initial startup root20261105_085136 was preserved and superseded before
its first epoch to close the optional periodic validation AP route as well.
Current run is fresh initialization, not a resumed pilot. TensorBoard stream
`V42-noisy-proposal-paired` points to the corrected run; all prior historical
roots and six pins are preserved. Measured scalar indexing verification pending.


Final launch verification (2026-11-05T09:02+01:00): PID869864 active at epoch0/step300,
finite loss65.3695;677 frozen source hashes and5 input hashes match.
Actual times span0..999 (910 unique across2400 images), all10bins sampled;8132
positive sources checked with zero mismatch, complete state permutations and
preserved background labels. TensorBoard fully indexed49 scalar streams/45roots;
`V42-noisy-proposal-paired/.` has real training loss through step199.
All six saved K1/K4/K8 pose/box pins have measured data from existing runs; the
new run's AP/val loss arrive after its first epoch. Original44roots preserved.
Direct host-side HTTP to100.122.159.115:33332 passes; phone connectivity is not
measured here. Evidence: run `launch_verification.json`,
`dashboard_verification.json`, `manifest.json` and `status.json`. No learned
performance claim or further pilot cutoff.

[2026-11-05 11:57 CET] User-approved V42 noisy-proposal cancellation verified
- SentSIGINT only to verified PID869864; process exited, native interrupted checkpoint preserved at /storage/outputs/pose-graph-diff/results/poseflow_v42_noisy_proposal/20261105_085659/seed42/checkpoints/interrupted.ckpt (521764325bytes). Saved user_cancellation.json/status/history; latest completed epoch006/10500 fixed32 poseAP K1/K4/K8=5.856/1.865/1.435. Uniform-t/V43 controls and TensorBoard unchanged.
- Approved next step is frozen existing-head pre-pose confidence substitution with exact native query alignment, no geometry/DDIM/training change.3local red-to-green checks PASS; pinned paired epoch3/6000 and epoch31/48000, immutable capture/evaluator extension snapshot. Original frozen-source verification and fixed32 comparisons underway.

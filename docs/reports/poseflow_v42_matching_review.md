# Re-review of failed V42 matched-target experiment

Read-only review of the original frozen run, 2026-11-04. No model, config,
trainer, sampler or TensorBoard change; the cancelled run remains cancelled.

## What was actually matched

| Mechanism | Assignment | When |
|---|---|---|
| Failed v42-matched-target | Incoming noisy pose -> clean GT person, using masked joint L1/OKS | After noising, BEFORE image/model forward; fixes final and pose-auxiliary loss targets |
| Current v42-paired-proposal | Selected image proposal -> clean GT person, using native class/boxL1/GIoU | Inside model at selection boundary; constructs matched clean pose THEN adds noise |
| Operation under discussion | Selected image proposal <-> actual incoming noisy pose state | At the selection/injection boundary on EVERY inference call; possible training counterpart requires an explicit target contract |

The failed run does not implement the third mechanism. Its matching function
receives only noisy coordinate tensors, GT target dictionaries and native pose
cost parameters. It has no proposal features/boxes/logits. In its loss method,
noisy_pose_matching_costs and Hungarian assignment run before
_forward_diffusion_batch creates the proposals. The frozen model selects50
image proposals by score and adds noisy pose i to proposal feature i; it never
reorders those proposals using the noisy states.

This is a semantic distinction, not a renamed configuration. For example, if
noisy state i belongs to person B but proposal i corresponds to A, the failed
run assigns B's loss target to that slot without moving it to B's image proposal.
The current paired run instead constructs an A-source state for A's proposal.
The proposed direct state association would change which existing noisy state
is combined with the proposal, which neither historical mechanism does at
inference.

## What the failure establishes

The frozen target builder hardcodes timesteps to zero. The failed run never
trained the full0..999 diffusion range. It was cancelled during epoch33 at
step50950; last completed epoch32/49500 fixed32 Gaussian poseAP K1/K4/K8=0/0/0.
The run's final native sampler has no proposal-to-state matching.

A previously saved frozen epoch31/48000 fixed32 ablation reports:

| Supplied state | Model timestep | Pose AP (0..100) |
|---|---:|---:|
| Native near-clean GT-derived t0 state |0|12.744|
| Same GT-derived t0 state |999|11.049|
| Gaussian |0|0.000|
| Gaussian |999|0.000|

These are conditional diagnostics, not GT-free inference results. They show a
serious failure of that training recipe. They also show that changing only the
timestep embedding cannot rescue Gaussian inputs. Near-clean conditional recovery
is itself limited; therefore correcting timestep coverage cannot be assumed to
solve the whole failed model.

## Implication for the pending design

It is inappropriate to cite this old run as evidence that proposal-to-noisy-state
matching failed: it was never performed, and full-range noise training was absent.
A proposal/noisy-state experiment remains relevant and untested.

A clean future comparison must explicitly specify:
1. Which actual incoming-state geometry is matched to current image proposals;
   training GT boxes cannot silently stand in for noisy-state geometry.
2. Whether that association is used during training as well as inference.
3. How complete GT labels/background provenance follow the assignment during
   training, especially where high noise makes pose identity ambiguous.
4. Consistent state/proposal ordering for the denoiser AND DDIM residual/update.
5. Full-range uniform timesteps, native baseline comparison and separate provenance.

The existing pending inference-only proposal can test an inference intervention,
but cannot by itself establish that matching is consistent between training and
inference. Do not blindly restart the old matched-target config or silently change
its semantics. No new training or production implementation is authorized by this
read-only review request; current training jobs remain intact.

## Provenance

Run: /storage/outputs/pose-graph-diff/results/poseflow_v42_matched_target/20261104_012819
Frozen training helper SHA256:
`9c90108e85f8b43ed41e261bf439cc98a81563168c5eb09d7f091e6dedf1bb36`.
Current helper is byte-identical to frozen and matches its manifest hash.
Frozen model/noiser were inspected directly; current model has later optional
input-builder changes, so conclusions use the original worker source.
Evidence: run/manifest.json, status.json, user_cancellation.json,
seed42/quick_test/epoch032/metrics.json, and
`tmp/v42_matched_target/diagnosis/input_time_ablation.json`.

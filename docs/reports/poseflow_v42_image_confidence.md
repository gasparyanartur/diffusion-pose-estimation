# V42 frozen pre-pose image-confidence diagnostic

Approved2026-11-05 after the recommendation to stop failed noisy matching and
isolate confidence using existing paired checkpoints. No production source,
geometry, native sampler, optimizer or training defaults changed. Diagnostic
code and immutable extension snapshot are in
`tmp/v42_paired_proposal/image_confidence/`.

## Exact score intervention

Use the existing trained BOX-stage classifier logits on decoder-normalized
candidate features before noisy-pose injection. Gather those scores by the
EXACT native top50 indices from the raw-feature selection logits. These are
different classifier calls: independently sorting normalized logits or treating
raw selection logits as calibrated scores would be a different experiment.
The callback checks gathered raw logits against the actual selected logits.
The first box auxiliary output is supervised over200 candidates; the selected
pose bank has50. Query identity is retained throughout.

For each K, run native DDIM once, observing every forward without changing any
returned tensor. Create an additional final dictionary changing only pred_logits
to the final call's gathered pre-pose logits. All raw boxes, joints, visibility
and diffusion states are bitwise identical. Postprocessing and pose NMS receive
the substituted scores and may select different survivors. Image-only means
independent of the noisy pose at a FIXED image/timestep; the image encoder is
time-conditioned and proposals/scores are not asserted invariant across K.
No GT, oracle labels, score fitting or extra image forward is used for rescoring.

## Comparison

Pin audited paired epoch003/6000 and latest completed paired epoch031/48000 at
experiment start; the paired trainer was already cancelled. Both use identical
fixed32 held-out image IDs, seed42+image_id, resize256427, FP32/TF32off and native
CrowdPose postprocessor/evaluator. Values below are AP on0..100 scale.

| Epoch / updates | Confidence | K1 pose AP | K4 pose AP | K8 pose AP |
|---|---|---:|---:|---:|
| 3 / 6000 | native |23.442|9.820|9.108|
| 3 / 6000 | pre-pose image confidence |25.273|23.312|23.711|
| 31 / 48000 | native |40.073|13.038|10.207|
| 31 / 48000 | pre-pose image confidence |42.893|35.525|35.274|

| Epoch / updates | Confidence | K1 box AP | K4 box AP | K8 box AP |
|---|---|---:|---:|---:|
| 3 / 6000 | native |33.175|18.213|16.968|
| 3 / 6000 | pre-pose image confidence |35.330|35.916|36.111|
| 31 / 48000 | native |46.613|22.183|18.493|
| 31 / 48000 | pre-pose image confidence |47.841|43.229|42.753|

The score substitution improves all six pose-AP conditions and all six box-AP
conditions without changing raw geometry. At epoch31, K4 pose AP rises13.038 ->
35.525 (+22.486), K8 rises10.207 ->35.274 (+25.067). This confirms a large ranking
penalty from the pose-conditioned scores in these frozen trajectories. It does
not show that generated geometry improved: it was held exactly fixed.

A remaining K1-to-K4/K8 gap persists (42.893 versus35.525/35.274 at epoch31).
These results therefore do not establish that classification explains the
entire multicall deficit or that a retrained image-only head will improve
geometry/pose-state use. They are two-checkpoint fixed32 diagnostics, not full8000
or retrained performance. They support a separate reviewed image-only-confidence
training iteration while retaining clean proposal/source/target pairing.

## Verification and artifacts

Three minimum local tests pass after intended missing-module red: nonidentity
B>1 gather where normalized/raw rankings disagree; actual V42 same-image/time
score invariance under different noisy states, hook/forward cleanup including
exceptions; exact native K4 predictions/state/private/global RNG and one forward
per timestep. On CUDA, uninstrumented/native outputs for the first image at each
K/checkpoint replay bitwise exactly. All saved native AP/AR/box metrics reproduce
EXACTLY for all six baseline conditions. All638 frozen source hashes are verified
per checkpoint, checkpoint/config/baseline/annotation hashes saved, model weights
and CPU/CUDA RNG preserved. All192 native/rescored final prediction pairs have
identical raw geometry;832 instrumented forward calls validate actual selection.
Extension source hashes match the immutable implementation snapshot.

Checkpoint SHA256s:
- Epoch3:10ab704cbfd137b56bfa4914369652d0f74363cb13546b8af5438781fae8e357.
- Epoch31:acadef264c21da4da24a01a0541a402bc0ddc67d092c61bae167547e2d9ea3da.

Artifacts: selection.json, implementation_manifest.json, verification.json,
comparison.json, tests/logs, and epoch003/epoch031 report.json, traces.json and
native/image_only predictions under the local diagnostic folder. Original
paired run/workspace/checkpoints, existing TensorBoard curves and uniform-t/V43
workers remain unchanged.

## Failed noisy-matching run cancellation

User-approved SIGINT stopped PID869864 during epoch7, after completed epoch6
and roughly10650 updates. Native interrupted checkpoint is preserved at the
noisy-proposal run's seed42/checkpoints/interrupted.ckpt (521764325 bytes), with
status.json and user_cancellation.json. Process exit was verified. Completed
epoch6 poseAP K1/K4/K8=5.856/1.865/1.435. Historical TensorBoard data is retained.
No new training has been launched by this diagnostic.

## Same-age performance control

Corrected uniform-t V42 on the same fixed32 images at epoch31/48000 has
K1/K4/K8 poseAP43.163/42.422/41.622 (epoch3:26.484/23.393/23.878).
The rescored paired checkpoint42.893/35.525/35.274 therefore remains weaker
than the uniform-t control, especially at multiple calls. This score intervention
repairs much of the paired model's ranking failure; it does not establish paired
training superiority or useful diffusion-state refinement. Removing the classifier's
pose dependence during TRAINING may change learned geometry, but that is untested.

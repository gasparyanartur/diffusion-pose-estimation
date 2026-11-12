# V42 proposal-to-incoming-pose inference comparison

Approved 2026-11-05. Inference-only test on existing paired-proposal V42
checkpoints, preserving native training, inference defaults, full80 jobs and
TensorBoard curves. This tests the user's requested current-proposal/current-
noisy-state matching. It is distinct from earlier noisy-state-to-GT loss
matching and previous-proposal-to-current-proposal spatial reordering.

## Policy

On every native DDIM call, including the first Gaussian call, select the native
50 image proposals. Convert all50 incoming scaled joint states using
XY=(state/6+1)/2, form joint min/max boxes, preserve their center and floor each
extent at1e-6. Coordinates are not clamped merely to compute matching boxes.
Use the EXISTING configured native Hungarian matcher, box-only mode. All
synthetic target-side labels are the same, so the class cost is constant along
each proposal row and cancels across a complete permutation. Default geometry
weights remain5*boxL1 -2*GIoU. No GT, known-person masks, GT visibility or new
trained parameters are used. No states/proposals are discarded or replaced.

Permute the incoming state into current proposal order. Apply that exact tensor
to pose encoding, epsilon reconstruction AND the DDIM residual/update. Next
call rematches the actual updated coordinate state, not previous predicted boxes
or previous reference boxes. Image-proposal coordinates stay unchanged. The
existing forward input-builder hook avoids a second image/box forward.

This is an alternative inference association, while training still matches
proposals to GT before corruption. Raw noisy-joint boxes can be uninformative
at high noise and differ from annotated boxes. The policy is an explicit test,
not an assumption that these rules are equivalent or that background feedback
has already been identified as the causal failure mechanism.

## Verification and selection

The intended first test failed at the missing sampler module.10 focused tests
pass: known permutations/batch isolation, native matcher reuse, constant-class
cost equivalence, degenerate/out-of-frame/tied geometry, invalid input rejection,
complete value-preserving permutations, actual two-step encoder/DDIM ordering,
exact disabled native outputs/global/private RNG, one model forward per step,
finite real-model sampling and unchanged model state.59 focused/native
V42/diffusion/evaluator regressions pass.

Before starting GPU comparisons, pin the already-audited epoch003/6000 and the
latest completed checkpoint at selection time, epoch006/10500. Both use the
original frozen model/config and a separately frozen new sampler/evaluator
extension. For each checkpoint, replay native and matched K1/K4/K8 on identical
32 held-out images, seed42+image_id, resize256427, FP32/TF32off and native pose NMS.
Native saved AP/AR/box metrics must reproduce exactly before interpreting results.
All assignments and source/checkpoint/annotation/config hashes are saved.

Artifacts: `tmp/v42_paired_proposal/proposal_state_inference/selection.json`,
`implementation_manifest.json`, tests/logs, checkpoint-specific `report.json`,
`assignments.json`, predictions and combined `comparison.json`.

## Results


Both checkpoint comparisons completed. Values are AP on the0..100 scale; each uses the same32 held-out test images.

| Epoch / updates | Policy | K1 pose AP | K4 pose AP | K8 pose AP |
|---|---|---:|---:|---:|
| 3 / 6000 | native |23.442|9.820|9.108|
| 3 / 6000 | matched |23.186|6.984|6.289|
| 6 / 10500 | native |29.240|9.292|8.451|
| 6 / 10500 | matched |28.687|7.657|7.015|

| Epoch / updates | Policy | K1 box AP | K4 box AP | K8 box AP |
|---|---|---:|---:|---:|
| 3 / 6000 | native |33.175|18.213|16.968|
| 3 / 6000 | matched |32.176|12.912|11.575|
| 6 / 10500 | native |36.756|15.341|15.267|
| 6 / 10500 | matched |35.839|11.953|11.130|

Every native K1/K4/K8 AP/AR/box metric reproduces its saved run value exactly at
both checkpoints.638 original frozen source hashes are verified per replay;
the new sampler/evaluator/test source hashes match the immutable extension copy.
All832 recorded matching calls across both checkpoints have complete50-element
permutations, totaling41,600 state associations. This includes all first calls.
Checkpoint SHA256s:
- Epoch3: `10ab704cbfd137b56bfa4914369652d0f74363cb13546b8af5438781fae8e357`.
- Epoch6: `1f12a072356caaeb89657a93ff57918255ca395e94b8e271a8a04080eb6fe33a`.

The policy reduces pose AP and box AP in every tested condition. K1 changes
little; multicall AP becomes worse. At epoch6, mean person probability for K8
rises .2083->.2474, and queries above .5 rise3.25->5.06/image; epoch3 shows the
same direction (.2081->.2452,5.03->7.22). These are aggregate confidence summaries,
not per-query GT false-positive labels or a new causal isolation of why the
policy failed. Pose AR at epoch6 K4/K8 changes34.14/35.04->35.19/35.49 while AP
falls, consistent with the unresolved ranking problem.

Conclusion: this particular every-call noisy-joint-enclosing-box matching policy
is not a useful inference-only repair on the two tested checkpoints. It does not
establish that every proposal/state cost or a consistently trained association
will fail. Those are separate experiments. Native inference stays default;
no training, running worker or TensorBoard history was changed, and no new
training experiment was started. The opt-in sampler remains available for
reproducibility. No full8000 test comparison was run.


# V30: shared decoder tolerance for recovered latents

Approved 2026-10-26. V28 fixed correspondence is the baseline; V29 OTA is stopped
by user request, retaining epoch071.ckpt and last.ckpt.

Objective: `0.5*clean_task + 0.5*task(R(sg(hhat0))) + latent_MSE`.

Recovered task gradients train only the shared box/joint/existence heads. Latent
MSE trains the denoiser and image extraction; clean task trains the clean encoder
and heads with detached image features. No OTA or architecture changes. Initial
model weights match V28 exactly. Fresh seed42,80epochs,256/427 resize, batch4/
accumulation2,FP32/TF32off,AdamW2e-4 and backbone1e-4,weight decay1e-4,
milestones31/61,clip0.1. Four-call deterministic DDIM is primary, one-call is
diagnostic. Existing fixed32 monitoring and epoch79 all8000 K1/K4 evaluation
remain enabled; monitoring does not select checkpoints.

Verification: three new tests cover isolated backward routes, readout value
parity, strict restoration/next proposal+noise draws, initialization-free evaluator
loading, and unchanged recipe. Real CUDA batch4/accumulation2 optimizer smoke
passed (peak allocated0.917GiB/reserved1.053GiB). Lightning optimizer/validation/
quickAP smoke passed. Full regression suite:130 tests passed. A preexisting
V26 assertion comparing differently ordered FP32 sums needed tolerance; its exact
legacy-formula assertion and backward-route checks remain.

Hypothesis: heads trained on recovered representations may tolerate imperfect
latent recovery better. Success requires improved Gaussian-start pose/box AP;
falling latent MSE or lower reconstruction loss is insufficient. Full V26 also
differs in clean-image gradients, so this is a controlled comparison to V28.

Launched fresh seed42/80epochs at `/storage/outputs/pose-graph-diff/results/poseflow_v30_decoder_tolerance/20261026_112529`. Frozen source: `/storage/outputs/pose-graph-diff/results/poseflow_v30_decoder_tolerance/20261026_112529/workspace`;
resolved config: `seed42/.hydra/config.yaml`; source/input hashes and initialization
parity: `manifest.json`; persistent console: `launch.log`; status: `status.json`.
Verified 250 optimizer updates, finite losses and nonzero gradients in both
stacks, readout and image modules. Actual launch peak allocated GPU memory
1.470GiB (larger than the two-microbatch preflight).
This is launch evidence, not learned performance.

Shared TensorBoard remains on port33329, label **V30-decoder-tolerance**, with
populated training scalar data and Tailscale HTTP200 verified. First fixed32 AP
is expected after epoch0 validation; the completed Lightning smoke verifies that
path. A visible `V30-progress` window is selected in attached tmux session73.
No separate TensorBoard server was added. Original latent labels are retained.

Full8000 epoch79 deterministic K1/K4 evaluation is configured in frozen
`run_v30.py`, including checkpoint/config/annotation hashes and comparisons to
V28, original V26 and V24. Those results remain pending training completion.

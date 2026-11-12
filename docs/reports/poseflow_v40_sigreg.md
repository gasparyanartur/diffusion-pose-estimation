# V40-SigReg

Status: FAILED during epoch3 after latent-scale divergence. Run: `/storage/outputs/pose-graph-diff/results/poseflow_v40_sigreg/20261103_123752`. See [failure diagnosis](poseflow_v40_sigreg_failure.md).

## Model and gradient contract

V40-SigReg keeps V40's ED-Pose proposal frontend, 200 candidates, matched-preserving
50 references and box/existence decoder. There is one online clean encoder and
no EMA teacher. Only the final clean LayerNorm is replaced by identity; internal
norms remain. Shared initial parameter tensors and global RNG match V40, while
forward codes intentionally differ because output normalization changes.

The source h0 remains live through ht=a_t*h0+sigma_t*epsilon. Recovered detection
and latent-prediction losses can therefore train the clean encoder through ht.
Positive latent regression still uses stopgrad(h0). References/features/boxes
remain live; correspondence stays fixed after Hungarian assignment.

L = proposal_task +0.5clean_task +0.5recovered_task +10matched_latent_MSE
+0.05SIGReg(matched_clean_codes). Detection coefficients remain2/5/2. Classification
covers all50 slots; geometry/MSE cover matches. Background placeholders and
recovered codes are excluded from SIGReg. No auxiliary regularization projector.

## SIGReg definition and provenance

The characteristic-function statistic is adapted from Randall Balestriero and
Yann LeCun's LeJEPA minimal example, revision
`c293d291ca87cd4fddee9d3fffe4e914c7272052`:
https://github.com/galilai-group/lejepa/blob/c293d291ca87cd4fddee9d3fffe4e914c7272052/MINIMAL.md
Paper: https://arxiv.org/abs/2511.08544 . Original license: CC-BY-NC-4.0;
full text is in `docs/licenses/lejepa-CC-BY-NC-4.0.txt`.

Changes from the example: functional FP32 calculation, explicit configuration,
matched-person filtering, small-count skips and a separate checkpointed RNG.
Defaults: weight0.05,256 unit random projections,17 knots on[0,3], symmetric
trapezoidal weights with Gaussian window, average over projections, multiplied
by the number of matched codes. The weight is additive to the V40 objective;
it is not LeJEPA's full objective mixing coefficient.

SIGReg sees matched people from the current four training images together.
Fewer than two codes returns a connected zero; weight0 bypasses projection and
penalty computation. Projection sampling uses its own checkpointed CPU generator,
advancing on eligible training batches only. Validation directions are determined
by seed42+batch_index and never change training randomness. Inference computes
neither clean codes nor SIGReg. Restore requires the saved SigReg configuration
because model tensor shapes alone do not identify the normalization behavior.

## Verification and interpretation

Eight intended-red tests preceded45passing focused SIGReg/V40/EMA/V39/V35 checks.
Coverage includes reference characteristic-function quadrature, Gaussian versus
collapsed/shifted/rescaled distributions, FP32, masks/empty/single/full50/invalid
boxes, independent loss gradient routes, objective coefficients, shared initial
tensors/RNG, strict pipeline/optimizer/next-update restoration and cached K1/K4
references. Latent targets remain detached. Native verification passes; see below.

Raw/weighted SIGReg,count/skips and matched clean/recovered mean/RMS/variance/
covariance diagnostics augment existing proposals/background/noise-bin recovery.
A fixed four-image subset of held-out fixed32 supplies per-person clean-code
snapshots initially and every epoch, with MSE/cosine to initial and previous codes.
This probe is GT-conditioned and is not detection-performance evidence.

SIGReg constrains marginal shape; a rotating Gaussian remains Gaussian, so it
cannot guarantee stable person identity or correct reverse conditional states.
Comparisons with EMA combine changes in normalization, gradient routing and
regularization. Only one long training run is approved; no zero-weight control.

## Training recipe

Config: `benchmark_poseflow_v40_sigreg`; freshseed42/80epochs, original V40
optimizer/scheduler/data/augmentation,FP32,resize256/427,batch4,accumulation2,
clip0.1. Frozen source/configuration, ED-Pose tensor coverage and SIGReg provenance
are recorded in a timestamped poseflow_v40_sigreg namespace. Fixed32 held-out bbox
K1/K4 runs every epoch; full8000 K1/K4 follows epoch79. All50 outputs, no NMS,
fixed proposal references and denoiser-only final scores. Existing runs/history/
pinned dashboard comparisons are preserved. Launch health will be reported
separately from learned results.


## Native verification

Native FP32 CUDA completed two batch-4/accumulation-2 optimizer updates with
finite losses and gradients. Each of latent regression, recovered task loss and
SIGReg independently supplies a nonzero clean-encoder gradient. Selected
reference feature and coordinate gradients remain live. ED-Pose initialization
covers all 342 intended tensors. Initial online tensor hash and global RNG match
V40 exactly: d63da49a27c552fe050bb73a5a0b4fea132f12e48e08cdb76d2804a8efb38c26.

Strict model/pipeline/optimizer restoration is exact. The subsequent native CUDA
update agrees within FP32 tolerance (maximum parameter difference 1.14e-7;
deformable-attention backward reductions are not bitwise deterministic). CPU
next-update restoration is exact in focused tests. Model-only restoration uses
identity clean normalization and needs no initialization assets. Gaussian K1/K4
two-image sampling produced 100 detections for each setting; this is a wiring
smoke, not learned-performance evidence. Peak reserved CUDA memory: 2.227 GiB.

The fixed probe covers 17 people in four held-out images. Repeating it without
changing weights gives zero MSE; an injected 0.01 clean-output shift produces
MSE 0.0001. A real validation callback smoke verifies probe recording and logging
without advancing training projection RNG. After the final diagnostic logging
change, all eight SIGReg tests pass again. The verified launch is recorded below.

Launch verification caught an initial-probe transform-mode issue: the data module
could apply training augmentation during on_fit_start. The run-local probe now
uses a shallow evaluation-only data copy, independent of trainer.training. A real
train-mode versus eval-mode parity check gives identical clean codes. The first
namespace 20261103_123224 is preserved as superseded; only its SIGReg trainer was
interrupted. A corrected launch starts fresh from seed42, not its weights.


## Verified launch

Run: `/storage/outputs/pose-graph-diff/results/poseflow_v40_sigreg/20261103_123752`.
Trainer PID: 278823; tmux: `poseflow-v40-sigreg-seed42`.
At verification the corrected fresh run had completed 380 finite optimizer
updates and continued progressing after the initial 170-update check. Training
losses and gradients are finite. No learned AP result is claimed yet.

The manifest audits 607 frozen source/document files and five inputs, including
the resolved configuration. SIGReg source revision/license hashes and ED-Pose
initialization mappings are preserved. Live logs expose 65 scalar tags. The
detection dashboard includes V40-SigReg and preserves all 17 previous streams,
history and pinned K1/K4 AP comparisons. Initial/every-epoch held-out clean-code
probe snapshots use evaluation transforms regardless of trainer mode.

Per-epoch fixed32 K1/K4 and automatic full8000 K1/K4 after epoch79 are configured
and verified. The runtime disables the generic post-fit task-loss test in favor
of that explicit Gaussian bbox evaluation, disables the progress bar in favor
of persistent status, and uses a short local TMPDIR for worker IPC. These are
recorded in runtime_overrides.json; the optimization recipe is unchanged.

Artifacts: manifest.json, preflight.json, live_verification.json, launch.log,
status.json, seed42/.hydra/config.yaml, seed42/representation_probe/ and frozen
workspace/. The initial superseded namespace remains available for audit.


## Outcome update — 2026-11-03

The initial launch-health checks did not establish training stability. Clean-code
RMS grew from0.849 to2136.947 by epoch2, both K1/K4 AP and IoU0.5 coverage fell
to zero, and the gradient-norm check stopped training in epoch3. Checkpoints and
logs are preserved. See [failure diagnosis](poseflow_v40_sigreg_failure.md) for
measured gradient-route evidence, its limitations and the proposed correction.

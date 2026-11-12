# V41: reference-conditioned coordinate box diffusion with JEPA

V41 retains V40's image-derived proposals and changes the diffusion state from
256-dimensional learned features to four-dimensional box coordinates. The clean
target and noisy view share an encoder; a time-conditioned predictor learns the
matched clean embedding. Live target gradients and SIGReg are retained.

The user's correction explicitly retains the full proposal front end:

```text
image → backbone/projections → three deformable encoder layers
      → native grid-relative box/class heads → top 200 candidates
      → top 50 scored reference boxes and features

noisy box coordinates → shared encoder → timed, reference-conditioned predictor
                     → clean embedding → shared box/existence heads
clean GT boxes       → shared encoder → clean target embeddings
```

The image path uses four scales and width 256. ED-Pose initializes backbone,
projections, encoder layers 2/4/6, encoder-output transformation and proposal
heads. All 342 eligible tensors load with audited 100% coverage. Query modules
start fresh. Initial shared model tensors and global RNG match V40 exactly;
parameter count is 45,699,659. This is initialization parity, not forward parity.

Both training and inference select references by native classification score,
without GT-retaining selection. Selected boxes/features stay live in training.
Each predictor block receives reference features plus an embedded reference box
and timestep conditioning. Proposal classification/box supervision uses its own
Hungarian assignment. It does not determine the final denoiser assignment.

Boxes are normalized resized-image `cx,cy,w,h`. Diffusion coordinates are
`z = 6 * (2*b - 1)`. Every valid GT appears once in the training source, with
remaining slots padded by Gaussian vectors of standard deviation 6 directly in
centered coordinates. Invalid boxes are filtered; more than 50 valid people is
an error. Uniform per-image timesteps 0–999 use the existing 1,000-step cosine
schedule. Noised sources are coordinates, never learned features. Only the
encoder-input view is clipped to [-6,6] and converted back to normalized boxes.

A two-layer input MLP and three full-image attention blocks encode coordinates
into width-256 embeddings. Clean/noisy views share those parameters and fixed
zero-timestep encoder conditioning. The three predictor blocks use the actual
timestep and fixed proposal references. Final clean normalization is identity.
Shared linear heads predict absolute sigmoid boxes and existence; there is no
coordinate bypass around the embedding.

The clean GT bank uses each valid GT once, zero-padded to 50. Hungarian matching
of final predicted boxes/scores selects GT box, existence and clean-embedding
targets together, using classification/L1/GIoU costs 2/5/2. Embedding gathering
occurs outside `no_grad`, preserving target gradients. A final query can match
a different person from its proposal's auxiliary assignment or corruption source.

```text
L = L_proposal + 0.5 L_clean + 0.5 L_denoised + 10 L_JEPA + 0.05 SIGReg
```

Detection losses retain focal/L1/GIoU coefficients 2/5/2. Classification covers
all slots, geometry/JEPA only matches. JEPA averages squared error over matched
queries and embedding channels. SIGReg uses unique valid clean GT embeddings,
excluding padding, with the existing pinned LeJEPA implementation and license:
256 projections, 17 knots over [0,3], checkpointed independent projection RNG,
skip fewer than two codes. Both JEPA branches and image/reference features are
live. Empty geometry/embedding losses remain connected zeros.

Inference starts from independent Gaussian `[B,50,4]`. Image/proposal features
are computed once and references stay fixed. Every DDIM step re-encodes current
coordinates, predicts a clean box, converts it to centered coordinates, and
updates the original coordinate state with eta 0. Embeddings are not carried
between steps. Final boxes/scores come from the last denoiser output, with all
50 outputs, no renewal, ensemble, proposal-score multiplication or NMS.

Use `benchmark_poseflow_v41`; the public outputs are `pred_embeddings`,
`pred_boxes`, `pred_logits`, `existence_logits`, and sampling `coordinate_state`
plus optional `coordinate_step_diagnostics`. Historical model/configuration,
gradient, selection and restoration behavior is unchanged.

Verification: ten intended-red checks preceded implementation. The first suite
passed 62 focused checks across V41, V40, SIGReg, EMA, V35 and V39. Additional
checks cover reversed proposal/final identity and SIGReg skip/disabled RNG.
Native FP32 CUDA passed two updates at batch 4/accumulation 2, independent
JEPA/task/SIGReg encoder gradients, live reference gradients, strict model,
pipeline, optimizer and RNG restore, and next-update parity within FP32 tolerance
(maximum parameter difference 1.26e-7). Gaussian K1/K4 two-image sampling produces
100 finite detections each. The fixed 17-person probe repeats exactly and detects
an injected 0.01 feature shift (MSE 0.0001). Peak reserved memory is 2.463 GiB.
These are wiring checks, not learned-performance results.

The approved launch is fresh seed 42 for 80 epochs, with V40 data, augmentation,
optimizer/scheduler, resize 256/427, FP32, batch 4, accumulation 2 and clipping 0.1.
Fixed32 held-out bbox K1/K4 evaluation runs every epoch; full8000 K1/K4 follows
epoch 79. Frozen source/configuration, initialization provenance, persistent logs,
dashboard history, representation probes and checkpoints are retained. Numerical
failure or clean-embedding RMS above 10 stops with a forensic checkpoint.

Coordinate corruption is independent of learned feature drift; this does not
guarantee useful reverse states or stable representations. K1/K4 AP, coverage,
foreground/background scores, JEPA by noise range, feature RMS/variance, proposal
coverage/ranks and coordinate scale/clipping must be inspected during training.
This experiment remains box detection, not full pose estimation.

Final V41 checks: 14 passed, including reversed proposal/final identity and
SIGReg skip/disabled RNG; 65 distinct focused checks across regression/final
runs. Launch callback scale-guard smoke passed. V41-only checkpoint hooks now also save/restore Python, NumPy, Torch CPU and
initialized CUDA RNG state; a dedicated failing-then-passing test and native
CUDA RNG restoration check cover this. Fixed-batch next-update parity does not
claim exact dataloader worker/prefetch replay. Full training launch is verified below.


## Fresh 80-epoch launch

User-authorized fresh seed-42 run: `/storage/outputs/pose-graph-diff/results/poseflow_v41/20261103_181429`.
The trainer runs in persistent tmux session `poseflow-v41` (PID `424762`).
At 2026-11-03 18:18 Europe/Stockholm it had reached update 420 with finite
losses/gradients, clean embedding RMS 0.2945, and JEPA MSE
0.0226. This is launch health, not a learned AP result.

All 617 frozen source files and five input/configuration hashes were verified,
as were the uniform-noise recipe, live target flag, full proposal front end,
342-tensor initialization coverage, 80-epoch horizon, accumulation/clipping and
fixed32/final-full8000 K1/K4 evaluation settings. There are 64 scalar tags.
See `manifest.json`, `live_verification.json`, `status.json`, `launch.log`, frozen
workspace and `seed42/` logs/evaluations/checkpoints in that run directory.
The local active pointer is `tmp/v41/run_root.txt`.

[Detection dashboard](http://lynx.tail481642.ts.net:33330/): stream `V41`, with
all 20 prior streams, history and AP pins preserved. No existing trainer was
interrupted. The earlier V40-SigReg live-target run had already stopped at its
scale guard during epoch 13/update 20,466; its forensic checkpoint is intact.

The initial V41 namespace `20261103_181140` was frozen but never launched. It is
preserved with status `superseded_before_launch` because the final restore audit
added the required global-RNG hooks before the actual launch.

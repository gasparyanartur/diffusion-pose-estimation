# V40-EMA: slow full clean-code targets

This approved experiment retains V40's online model, proposal assignment and
reference-conditioned Gaussian DDIM sampler. The only objective-routing change
is using a slow EMA clean-code teacher for both corruption and latent targets.

Teacher scope: backbone/position extraction, projections/level embeddings, input
MLP, clean layers/norm and time embedding at time0. It is initialized from online
weights after ED-Pose initialization and updates with decay0.999 AFTER each AdamW
step. Accumulation microbatches and validation do not trigger EMA updates. Teacher
parameters are no-grad/eval and excluded from the optimizer. Floating buffers
follow EMA; nonfloating buffers copy. Full checkpoints save teacher and counter;
model-only evaluation loads only the unchanged online model.

L = proposal_task +0.5online_clean_task +0.5recovered_task
+10positive_teacher_latent_MSE. Detection weights2/5/2 and all V40 target masks
remain. Online clean reconstruction trains the online encoder. Teacher-code
readout loss is a no-grad diagnostic, not an additional objective. Neither
background latent supervision nor SigReg is added.

Diagnostics separately report online-teacher positive/background MSE/cosine and
mean/RMS, live-head reconstruction of teacher codes, EMA updates/decay and actual
teacher-target noise-bin recovery. Primary comparison: matched-age K4 bbox AP
and K4-K1 gap. EMA target MSE reduction is not proof of improved Gaussian detection.

Six intended-red checks preceded implementation. Initial online tensors and RNG
match V40 exactly; teacher-forward/loss equality is checked within FP32 tolerance.
37focused EMA/V40/V35/V39 checks pass, including accumulation cadence, detached
teacher source/target, optimizer exclusion, strict restore and Gaussian sampling.
Native CUDA verification and launch provenance will be appended.


## Native verification and launch

Six intended-red checks preceded37passing EMA/V40/V35/V39 checks. Native FP32
CUDA real-data batch4accum2 completed two finite optimizer updates. EMA counts
were[0,1,1,2], confirming no microbatch updates. Teacher gradients/optimizer slots
are absent; full teacher/model/optimizer/RNG and the next EMA update restore
strictly. Model-only K1/K4 restores without teacher/init assets. The online
initial model hash matches original V40 exactly; CPU/GPU RNG is preserved by
teacher initialization. Initial teacher forwards are checked at FP32 tolerance.

Online parameters:45,699,659; no-grad teacher parameters:35,603,648. Native smoke
peakreserved1.898GiB. Teacher source scope includes every clean-code dependency.
Live reference feature and box gradients remain positive in isolated probes.

Fresh seed42/80epoch run: `/storage/outputs/pose-graph-diff/results/poseflow_v40_ema/20261102_112758`.
The namespace is separate from V40; tmux session `poseflow-v40-ema-seed42`.
The V40 data/optimizer/scheduler/resize256/427/batch4accum2/clip0.1 recipe is
retained. Fixed32K1/K4 runs every epoch and full8000K1/K4 follows epoch79.
Teacher EMA updates occur after optimizer steps, while logged batch diagnostics
record the teacher state used for that forward pass.

Frozen source/configuration, initialization mappings/hashes, native preflight,
launch logs and live status are stored under the run root. Runtime overrides
replace the progress bar with persistent status and generic post-fit task-loss
testing with explicit held-out Gaussian bbox evaluation; short local worker
TMPDIR avoids IPC path-length failures.

First-epoch AP is pending. This confirms implementation and launch health; it
does not establish that EMA fixes multi-step ranking. SigReg remains deferred.

Live verification: trainerPID3933264, 160finite optimizer steps and
160EMA updates; counts match exactly. Frozen524source files and five inputs
hash-verified. Dashboard17runs retains all16prior streams/history/APpins;69V40-EMA
scalar tags are served, including teacher lag and update count.

Artifacts: `manifest.json`, `preflight.json`, `live_verification.json`, `status.json`,
`launch.log`, `seed42/.hydra/config.yaml` and immutable source `workspace/`.

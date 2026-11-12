# V36: independent clean box latents

Approved 2026-10-29. Two fresh seed42 controls branch from V35-spatial-latent-only:

| Variant | Clean input | Clean encoder |
|---|---|---|
| V36-box | pixel cxcywh /100 | Linear4->256, GELU, Linear256->256, non-affine LayerNorm |
| V36-box-c | same coordinates + binary existence | Linear5->256, GELU, Linear256->256, non-affine LayerNorm |

There is no clean image conditioning, self-attention, timestep modulation or
query position. Codes are independent per proposal. The additional existence
channel remains0/1 and is never scaled by100. Labels are accepted exclusively
by clean training encoding, never by the denoiser or Gaussian inference.
Shared model tensors and the first four input weight columns initialize exactly
identically; the fifth column is freshly initialized under isolated RNG.

The denoiser retains three V35 blocks: timestep-conditioned pre-normalization,
proposal self-attention, dense cross-attention to the four projected image scales,
and FFN; spatial Fourier query positions enter self/cross attention. Final linear
projection predicts256-dimensional clean code. Shared direct linear heads predict
sigmoid box coordinates and one existence logit. Image features are extracted
once. There is no DINO, joint head or token-expanded task decoder.

Objective:0.5 clean box/existence task +10 OTA-selected latent MSE. Recovered task
loss is monitoring-only. Clean task trains MLP/heads; denoising alone trains image
extraction/denoiser. Both corruption source and matched latent targets are detached.

Construction and matching are unchanged:50proposals, all real people once,
remaining split perturbed positives/foreign negatives with existing fallbacks;
global Hungarian alignment of every input to5x10anchors. Recovered predictions
use box Dynamic-K OTA(topk5, spatial cost1). Its assignment selects canonical clean
person codes; unmatched slots select nearest clean negative code or detachedzero
fallback. This can remap a corruption source to another person. It is not fixed
pair reconstruction. Box-only existence ambiguity is an intentional limitation:
identical geometry can represent a positive or borrowed negative.

Training recipe unchanged:80epochs, seed42, resize256/427, original augmentations,
batch4 accumulation2, FP32/TF32off, AdamW2e-4/backbone1e-4, wd1e-4,
milestones31/61, clip0.1. Image-only EDPose initialization; latent modules fresh.
Gaussian-start K1/K4 fixed32 monitoring; automatic epoch79 all8000 held-out bbox
K1/K4 evaluation, seed42+imageID, all50queries/noNMS. No checkpoint selection from
monitoring. Ordinary validation loss overlaps the trainval training split.

Verification: six intended red failures before implementation;20 focused tests
pass, including V35 regression behavior, label boundary, independent clean code,
isolated gradient routes, strict pipeline restore/next proposal-noise draws and
GT-free Gaussian prediction parity. Each variant receives native CUDA batch4,
accumulation2 two-update preflight, quick two-image bbox evaluation and strict
model loading without initialization assets. These are wiring checks, not evidence
of learned detection performance. Launch evidence is appended after verification.

Native preflight results:35,675,077 parameters(box) and35,675,333(box+c).
Both peak allocated1.005GiB/reserved1.170GiB. Image initialization imports282tensors
with100% compatible image coverage, SHA256
9ff5a0930cab53c42a903d1f10fec1ebbb49457a127987afab54e1c72d8d71d9.
Shared initial tensor hash excluding input weight matrix is identical:
829e5127fde09155ab3cfd6227949974d6367b2b33b03456ab121a9c168b8e49.
First four input weight columns and all other common tensors also pass exact
initialization equality in the focused test. Full per-variant initialization hashes
are asserted before production training.

Fresh frozen roots:
- `/storage/outputs/pose-graph-diff/results/poseflow_v36_box/20261029_004444`
- `/storage/outputs/pose-graph-diff/results/poseflow_v36_box_c/20261029_004447`

Each retains source/config/input hash manifests, preflight evidence, persistent
launch.log, atomic status.json, checkpointed proposal/noise RNG and automatic final
evaluation worker. Training tmux sessions:poseflow-v36-box andposeflow-v36-box_c;
progress windows73:V36-box and73:V36-box_c. Shared existing TensorBoard33329 exposes
V36-box andV36-box-c; TailscaleHTTP200 verified. Existing job sessions retained.

Launch verification: both90finiteoptimizerupdates, actual peakallocated1.139GiB.
Tailscale scalar API exposes40tags perrun, with populated train loss, latentMSE,
module/global gradient and construction diagnostics. Existing server was refreshed
once after the first scalar flush so new curves load before historical histories;
all labels retained. Full evidence:launch_verification.json in each run root.
No learned AP conclusion yet; fixed32bboxK1/K4 arrives after epoch validation.

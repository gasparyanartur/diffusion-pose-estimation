# V43 image-to-frozen-VAE alignment baseline

The hypothesis is that the image/proposal branch has not learned a transferable
mapping to the frozen VAE's pose-and-box representation. The prior single-image
model uses image conditioning but damages held-out geometry; identity-preserving
velocity alone supplied negligible recovery on the broader cache. See
[the thesis and frozen probes](poseflow_v43_thesis.md).

This baseline explicitly predicts a clean standardized 32-dimensional code from
images and selected proposal features. `predict_image` accepts only images,
unpadded image dimensions and optional padding. Internally, constant zero state
tokens and time zero enter the existing V43 prediction body; target or noisy
geometry never enters it. All predicted geometry passes through the frozen VAE.
The latent encoder runs only to construct supervision and evaluation ceilings.

Training uses clean posterior means, balanced foreground/background code error,
labeled-joint native OKS-weighted geometry, per-axis box-center error, log size
and GIoU, plus unchanged proposal and image-only confidence losses. A one-pixel
minimum center-normalization denominator handles crop slivers. No diffusion
noise, coordinate masking/jitter, posterior sampling, KL, or VAE update occurs.
Existing denoising APIs/default configurations remain unchanged. Checkpoints
record the training task and reject incompatible task restores.

The recipe is seed 42, 10 epochs, native crop/flip, batch 4 with accumulation 2,
FP32, AdamW main/backbone learning rates 2e-4/1e-4, weight decay 1e-4, cosine decay
to 2e-6, and gradient clipping 0.1. The frontend uses the existing EDPose transfer;
the V43 prediction body is fresh, without an epoch-011 V43 warmstart. The High-KL
codec checkpoint and audited standardization are fixed. The fixed statistics
include prior trainval geometry, and the image frontend has pretraining exposure;
neither is represented as an independently held-out detector initialization.

This stage trains on the 10,000 TRAIN images only and validates on the separate
2,000 VAL images, with image-ID overlap checked before initialization. The shared
DataModule eagerly initializes its unused test slot, so this preset points that
slot to VAL. The TEST annotations/images are not read or evaluated by this run.

Every epoch validates losses over the complete VAL population. A fixed 64-image
VAL monitor evaluates image-only native pose/box AP and paired source geometry
with correct and swapped image memory/appearance features. Swaps retain original
proposal boxes, confidence scores and query/target identity. Swaps are
counterfactual dependency checks, not detector-performance claims. Odd-sized
evaluation subsets use the first image as the last image's distinct donor.

Best full-VAL decoded geometry loss and last are the only retained checkpoints.
After 10 epochs both are automatically restored and evaluated over all 2,000 VAL
images, with native AP, paired OKS/IoU, median/p95 errors, foreground code error,
codec ceiling and qualitative geometry. The evaluator also checks exact image
coverage and records donor assignments. No next diffusion stage is launched.

Useful correct-image geometry/code prediction beyond wrong conditioning supports
learning a conditional representation. Train/VAL losses distinguish underfit
from overfit. Failure does not uniquely prove the VAE is unsuitable. Near-perfect
image-only AP is not required, and passing this baseline is not evidence of
small-noise recovery: that requires a later controlled denoiser comparison versus
raw noisy decoding at 1%, 2.5% and 5% noise with paired draws and geometry/AP gates.

Focused tests cover target-free deterministic inference, RNG isolation, frozen
decoder/input gradients, supervision-only targets, missing/empty targets,
task-specific checkpoint restoration, source/query/donor identity and disjoint
splits. A native CUDA smoke checks accumulated updates, strict restore and the
next optimizer update; its AP is wiring evidence only. Launch provenance and
trained results are appended after verification.

## Verified implementation and active training

Run: `/home/gasparyanartur/outputs/pose-graph-diff/poseflow_v43_alignment/20261105_173723`. The fresh 10-epoch stage is running in tmux
`poseflow-v43-alignment`, PID 956042; currently epoch 0,
optimizer step 200, with finite loss. This is launch evidence; no
learned held-out alignment result is available yet.

All 72 relevant regression checks pass, including five focused alignment tests.
The native FP32 CUDA smoke executes two accumulated optimizer updates; restoring
model, optimizer and scheduler reproduces the next loss exactly, with maximum
parameter difference 1.19209e-7. The frozen codec is unchanged. A three-image
native VAL evaluation verifies odd donor pairing, finite metrics and coverage.
Its zero AP after two updates is a wiring smoke, not trained accuracy.

All 615 source snapshot files were verified before worker initialization. The
initial model SHA256 exactly matches preflight:
`320dbd7f0b9a89b3f3d38efffe57e4131ae9ae9fab2f27e1666ee3a4229b6937`.
The worker checks this before optimization. The manifest checks disjoint TRAIN/VAL
IDs, hashes annotations and the pretrained codec/frontend/statistics inputs, and
records configuration and parameter counts (40,241,095 total; 39,444,135 trainable).

Scientific interpretation will use the complete VAL results of both best/last:
correct image conditioning should improve paired OKS/IoU and code error relative
to wrong conditioning, with native pose/box AP reported alongside. Mixed or tiny
effects remain inconclusive; no downstream diffusion optimization is automatic.
Natural annotation masks are encoder inputs, so image-only latent error can
include annotation-pattern uncertainty; decoded labeled geometry remains central.

The current TensorBoard label is `V43-image-alignment`. Dashboard preservation
and actual scalar indexing are verified: 57 scalar streams across 51 roots, all
56 prior streams and six AP pins preserved. New finite loss scalars are visible
through the direct address http://100.122.159.115:33332/.

## Cancelled by user; failure diagnosed

The first run is cancelled. The image/proposal frontend degraded severely, and
the regressor's pose/latent objectives fed conflicting gradients into proposal
boxes. The image-only path also starts with identical query states because its
reference modulation initializes to zero. See
[the measured failure diagnosis](poseflow_v43_alignment_failure.md). This recipe
is not a validated alignment solution. No replacement run has been started.

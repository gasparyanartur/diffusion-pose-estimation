# V33:17-token structured task readout

Approved2026-10-27. Baseline: completed V32 dual-DINO pilot. Each h expands to
14joint+box+existence+global tokens, with learned role-specific projections and
embeddings. Three pre-LN transformer blocks, width256,8heads,FFN2048,GELU,
dropout0, finalLayerNorm and single linear output heads. Global token has no
output target. Attention stays within each person's17-token group. No image
attention, cross-person attention, timestep input or cross-call state in readout.

Joint XY remains normalized image-space output and never depends arithmetically
on predicted boxes; internal box/joint features may exchange information. Shared
task readout handles clean/recovered latents. Diffusion remains[B,50,256]; DINO
head, clean image gradients, both h0 detach points, task/DINO weights and proposal
correspondence unchanged. New task heads replace old MLPs without unused tensors.
Common V32 image/clean/denoiser/DINO initialization and global RNG are preserved.

Same fresh seed42/80epochs,256/427 augmentations,batch4accum2,FP32/TF32off,
AdamW2e-4/backbone1e-4,wd1e-4,milestones31/61,clip0.1. GaussianK4primary,
K1diagnostic,fixed32 every epoch, finalepoch79full8000 K1/K4 with corrected NMS.
Task readout now costs three local transformer blocks per denoiser call; it is
not compute-matched to V32. No monitoring-based tuning/checkpoint selection.

Five focused tests cover shape/role layout, group isolation and permutation,
final-box-head perturbation leaves joints unchanged, common tensors/RNG, initial
Gaussian latent/DDIM trajectory parity, task/MSE/DINO routes, strict restore/next
proposal+noise draws and teacher/init-free evaluation. Initial test invocation
had a fixture import-path collection error, corrected to tests.test_poseflow_v32.
The corrected focused suite and all143 broad regression tests passed; final
role-gradient check confirms all17 roles receive task gradients. Real cached-teacher CUDA batch4accum2 optimizer smoke passed:
peak allocated1.792GiB/reserved1.908GiB,46,999,111 student parameters
(+4,662,758 vs V32). Teacher stays frozen/outside state and optimizer. Lightning
optimizer/validation/quickAP passed. Launch verified: 150 optimizer updates, finite losses/nonzero gradients
including pose_decoder and all heads. Actual task/DINO scalar streams populated.
This is launch evidence, not learned performance.

Frozen run: `/storage/outputs/pose-graph-diff/results/poseflow_v33_token_readout/20261027_105255`. Source/config and completed V31/V32 metric hashes recorded
in manifest.json. Resolved config seed42/.hydra/config.yaml; console launch.log;
live status.json. Actual launch peak allocated GPU memory
2.326GiB. Shared33329 TensorBoard **V33-token-readout**
has live training scalars and TailscaleHTTP200 verified; visible V33-progress
window selected in attached tmux session73. No additional TensorBoard server.
First fixed32 AP is expected after epoch0 validation. Frozen runner automatically
evaluates epoch79 all8000 at K1/K4 and compares against V32/V31/V26/V24 with
matching image/annotation checks. Historical jobs/checkpoints/events retained.

Hypothesis: structured per-joint decoding improves recovery from person latents.
Judge by same-age and final Gaussian pose/box AP versus V32, separating clean
reconstruction/conditional denoising from detection. A negative result would not
prove representation failure; optimization, matching and sampling remain possible
causes. No learned-performance claim from wiring/finite-loss evidence.

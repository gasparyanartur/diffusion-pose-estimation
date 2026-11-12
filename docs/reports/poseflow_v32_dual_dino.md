# V32: dual-branch DINO with live clean-image gradients

Approved2026-10-26: supervise both h0 and hhat0 with person-region DINO, and let
the clean path train image extraction. V31 remains active as the reference; this
combines two changes and is not an isolated DINO ablation.

Objective:
`0.5*clean_task + 0.5*recovered_task + latent_MSE + 0.5*(DINO_clean+DINO_recovered)`.
Default DINO weight1, configurable via pipeline.dino_loss_weight. Teacher is the
existing pinned DINOv3 ViT-S/16 checkpoint (revision114c1379950215c8b35dfcd4e90a5c251dde0d32;
SHA2564610ad75edef83e75afdebf162d148dc628045ea6cbb83d67d4708c709c4f91d).
Same frozen7x7x384 ROI features, shared grid predictor and cosine objective as V25.
Target features extracted once from recipient image and each unique GT box;
duplicates use source annotation correspondence, negatives excluded. Teacher has
no gradients, optimizer membership or checkpoint tensors.

Clean task and DINO train clean encoder and image features. Recovered task and
DINO train denoiser and image features. Both train their shared readout heads
(task heads or DINO head respectively). Clean h0 remains detached before adding
noise and as the MSE target. Denoising never backpropagates through clean h0.
No frozen clean encoder, no OTA, no new sampler. Inference executes neither DINO
teacher nor prediction head. Only dino_head.* adds model tensors; baseline model
initialization/global RNG and zero-DINO-weight numerical objective parity tested.

Five new tests cover original-row mapping, duplicates/empty images, both isolated
DINO gradients with live clean-image routes and h0 protections, negative zero
gradients/teacher detach, zero-weight parity/bypass, strict restore/next proposal
and noise draws, initialization-free/teacher-free evaluation and unchanged recipe.
Real cached-teacher CUDA batch4/accumulation2 optimizer smoke passed: peak allocated
1.243GiB/reserved1.420GiB. Teacher exclusion/freeze verified. Shared model tensor
initialization matches V31; the DINO head adds341632 parameters. All138 regression
tests passed, and real-teacher Lightning optimizer/validation/quickAP smoke passed.
Launch verified: 80 optimizer updates, finite losses/nonzero gradients
including shared DINO head, both DINO loss scalar streams populated. This is
launch evidence, not learned performance.

Frozen run: `/storage/outputs/pose-graph-diff/results/poseflow_v32_dual_dino/20261026_150316`. Source/config hashes, pinned teacher identity and V31
reference recorded in `manifest.json`. Console `launch.log`, live `status.json`,
strict resolved config `seed42/.hydra/config.yaml`. Actual launch peak allocated
GPU memory 1.731GiB.

Existing shared33329 TensorBoard label **V32-dual-DINO** contains separate clean/
recovered loss/cosine/variance/correct-minus-wrong diagnostics. Tailscale HTTP200
verified. V31 remains validation_and_fixed32 at update9000 on
inspection. Visible V32-progress window selected in attached tmux session73.
No new TensorBoard server; historical latent labels retained. Fixed32 AP begins
after first epoch; final full8000 K1/K4 evaluation is configured in frozen runner.

Fresh seed42/80epoch recipe: original256/427 augmentations, batch4accum2,FP32/
TF32off,AdamW2e-4/backbone1e-4,wd1e-4,milestones31/61,clip0.1. K4Gaussian DDIM
primary/K1diagnostic, fixed32 every epoch, automatic final epoch79 full8000 test.
No checkpoint selection/tuning from fixed32. Success requires Gaussian-start pose
and box AP; track DINO correct-minus-wrong cosine and variance to expose generic
feature prediction. Wiring/finite losses are not learned-performance evidence.

## Final outcome — 2026-10-27

Training and epoch79 full8000 K1/K4 evaluation completed. See
[final comparison](2026-10-27-v26-v31-v32-final-comparison.md) for results, matched
protocol verification and limits. Earlier launch-only evidence above is retained
as provenance.

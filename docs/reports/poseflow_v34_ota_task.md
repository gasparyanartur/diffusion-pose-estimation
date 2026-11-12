# V34: OTA-assigned task supervision through the denoiser

Approved2026-10-27. This tests what V29 did not: recovered task losses train the
denoiser/readout/image features under OTA assignment. V33 remains the parallel
control. No model/decoder/sampler changes; all model tensors initialize exactly
as V33. Existing Dynamic-K candidate_topk5/center_radius2.5 is reused.

One OTA mapping against unique GT people supplies recovered pose/box/existence,
detached clean-person latent targets and recovered DINO targets. Clean branch
task/DINO use original constructed correspondence. Unmatched queries: existence0,
no geometry/DINO loss and nearest-clean-negative latent target (same V29 policy).
Many-to-one matches allowed; this is not Hungarian one-to-one supervision.

Loss:0.5clean_task+0.5OTA_recovered_task+uniform_allslot_OTA_latent_MSE+
0.5(DINO_clean+DINO_OTA_recovered). Both h0 detaches remain; image gradients
remain live in both branches. Target remapping can change the person relative
to the corruption source; this is explicitly not the original fixed-pair loss.
This intervention also changes matched cardinality/background targets; a gain
would not isolate source ambiguity as the sole cause.

Five tests first failed before implementation, then passed. Forced swapped
assignment verifies identical mapping across all recovered losses, one matcher
call and one teacher extraction; original negatives may become positives.
Unmatched output geometry gradients are zero while negative-existence gradients
are active. Actual recovered task reaches backbone/denoiser/readout; clean h0
receives no gradient from recovered task/MSE/DINO. Empty-recipient/all-negative
fallback, exact initialization/global RNG, strict matcher-buffer restore/next
draws, asset-free evaluator and unchanged recipe are tested. All148 regression
tests passed. Real teacher CUDA batch4accum2 optimizer smoke passed: peak allocated
1.731GiB/reserved1.916GiB,
46,999,111 student parameters. Teacher remains frozen/outside optimizer
and checkpoint. Lightning optimizer/validation/quickAP passed. Launch verified: 60 optimizer updates with finite losses/nonzero
gradients. Actual recovered task, OTA assignment counts and recovered DINO
scalars populated. This is launch evidence, not learned performance.

Frozen run: `/storage/outputs/pose-graph-diff/results/poseflow_v34_ota_task/20261027_150734`. Source/input/config hashes, teacher pins and V33 reference
are recorded in manifest.json. Resolved config seed42/.hydra/config.yaml;
console launch.log; live status.json. Actual launch peak allocated GPU memory
2.242GiB. Shared33329 TensorBoard **V34-OTA-task**,
TailscaleHTTP200 verified. V33 remained training at
update33570 on inspection. V34-progress selected in
attached tmux session73; no separate TensorBoard server. Fixed32 AP begins after
epoch0 validation. Frozen worker schedules finalepoch79 all8000 deterministic
K1/K4 and comparison to V33 if available, plus V32/V31/V26/V24 references.

Original seed42/80epochs,256/427 augmentations,batch4accum2,FP32/TF32off,
AdamW2e-4/backbone1e-4,wd1e-4,milestones31/61,clip0.1. K4Gaussian DDIMprimary,
K1diagnostic; fixed32 each epoch and final epoch79 all8000 test. Monitoring does
not select checkpoints. Compare Gaussian AP to V33 at matched ages; conditional
loss/DINO/source-identity metrics have different targets and require labeling.

Track OTA counts/coverage/multiplicity/reassignment/background utilization, matched
and unmatched MSE, original-source MSE, and confidence under OTA labels. Correct
DINO similarity uses the assigned person. Low reassigned loss alone does not
establish successful Gaussian detection.

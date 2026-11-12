# V35 spatial latent-only denoiser

Approved2026-10-29, with explicit correction to use the spatial V35 base.
The only objective change is recovered task coefficient0.5->0:

`total = 0.5*clean_box_existence_task + 10*matched_latent_MSE`.

Clean task still trains the clean encoder, shared box/existence heads and live
image features. The denoiser and its image-conditioning route receive only
latent-MSE gradients. Both h0 corruption source and latent target stay detached.
Recovered box/existence task losses remain no-grad monitoring values; they do
not contribute to optimization. Shared heads learn from the clean path alone.

OTA still uses recovered decoded predictions under no-grad to choose the
canonical clean-person latent target. Unmatched queries retain the original
nearest-clean-negative/zero-background policy. This is not fixed-source
reconstruction and not a frozen autoencoder. Removing a task loss does not
remove decoding for assignment or Gaussian-start detection evaluation.

Model/state schema and all other settings remain V35-spatial: global Hungarian
proposal sorting, denoiser-only5x10 Fourier anchors, spatial cost1, latent10,
clip0.1, direct box/existence heads, no DINO/joints, seed42/80epochs and the same
FP32/TF32off data/optimizer recipe. New namespace; existing jobs retained.
Fixed32 Gaussian K1/K4 bbox AP and automatic finalepoch79 all8000 K1/K4 use the
same all50/noNMS evaluator. Monitor clean reconstruction, conditional denoising
and Gaussian-start detection separately; no monitoring-based checkpoint selection.

Fourteen focused V35 tests pass, including recovered-task no-grad, exact
denoiser total-gradient equality with10*MSE, no latent-MSE gradients into clean
source/shared heads, active clean task routes, default supervised forward-value
parity and config differences restricted to namespace/coefficient. Strict model
restoration and next proposal/noise draws remain covered by the V35 suite.

Native CUDA smoke and frozen launch evidence are recorded in
tmp/v35_spatial_latent_only and will be appended here after verification.

CUDA verification passed: two native batch4accum2 optimizer updates, quick2image
box evaluation, isolated latent-only backward (no clean-source/sharedhead grads)
and strict teacher/init-free model restoration. Common initial tensor hash
matches the V35-spatial control;41,590,981 parameters unchanged. Peak allocated
1.244GiB/reserved1.405GiB in smoke. This establishes wiring, not learned AP.

Frozen run: `/storage/outputs/pose-graph-diff/results/poseflow_v35_spatial_latent_only/20261029_001905`. Shared33329 label **V35-spatial-latent-only**;
visible tmux73 progress window and independent training session. Native loss,
latent/OTA diagnostics and gradient streams are configured; bboxAP K1/K4 follows
epoch0 validation. Existing V35-OTA/V35-spatial jobs are preserved.

Launch verified at70 finite optimizer updates, actual peak allocated1.381GiB.
The recovered task coefficient0, gradient norms, clean/recovered monitoring
losses, latent MSE and spatial-cost scalar streams are populated on the shared
Tailscale33329 dashboard. Existing V35 control runs remain active. Per-run
launch_verification.json pins the observed status/events; no learned AP claim.
The first launch encountered multiprocessing AF_UNIX path length before any
updates; only the new process was stopped and restarted with shorter local
TMPDIR tmp/v35lo. Source/config/objective stayed fixed; runtime_override.json,
startup_before_tmpdir_fix.json and console history document the repair.

## Failed first attempt and authorized restart (2026-10-29)

The original20261029_001905 attempt exited at00:42 Stockholm with SystemExit(1),
after1209gradientupdates in epoch0. Last full progress snapshot was update1200;
loss/gradients finite, no epoch checkpoint yet. There is no explanatory traceback
for this final exit. Earlier AF_UNIX startup failures precede the successful
00:21 restart and do not establish the cause of the later exit.

User authorized a freshseed42 relaunch. New root:
`/storage/outputs/pose-graph-diff/results/poseflow_v35_spatial_latent_only/20261029_010826`.
Its workspace is copied from the failed frozen snapshot, with all source hashes
verified and byte-identical resolvedconfig; current V36 edits are excluded.
No architecture/objective/data/optimizer/matching changes. Existing preflight
and tests retained. This is not checkpoint resume. Failed logs/events/status are
preserved. ShortlocalTMPDIRtmp/v35lo retained; PYTHONFAULTHANDLER and shell exit-code
logging added. Same training session/progress viewer, shared33329 dashboard;
failed attempt receives a separate label. Automatic final8000bboxK1/K4 retained.

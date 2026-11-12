# V43: frozen High-KL pose-and-box LDM

The reviewed design uses V40/V40-EMA's shared inference architecture with a frozen
32D pose-and-box VAE. Fresh ED-Pose image initialization and fresh denoiser weights
are specified; the trained V40-EMA checkpoint is not a warm start. Its moving EMA
target encoder is replaced by a fixed pretrained codec. The complete contract is
recorded in the 2026-11-05 V43 `.plan` entry.

## Codec adapter and augmentation audit

The unmasked High-KL checkpoint is
`sample_kl_high_clean/checkpoints/epoch099-score97.2048.ckpt` under
`/storage/outputs/pose-graph-diff/results/pose_vae_comparison/20261104_233335`.
SHA256: `7168bffc7452f83e5e7c70b062d55ff867d2cb7d1eb0932cd1339cdc9e0ee951`.
All codec parameters stay frozen/eval; predicted-latent input gradients remain
enabled through its decoder. Geometry uses transformed unpadded image sizes.

The full audit used all 12,000 trainval images, the actual historical detector
crop/flip/resize functions, and independent per-image transform seeds `42+image_id`.
Original annotation IDs survive filtering and qualitative export. Cropped geometry
is measured against its transformed targets, without original-image AP reassignment.

| Posterior mean reconstruction | Uncropped resize | Historical augmentation |
|---|---:|---:|
| Retained people |42,733|40,873|
| Mean paired OKS |0.986189|0.976550|
| Mean paired box IoU |0.986186|0.979346|
| Joint error median / box scale |0.007274|0.008036|
| Joint error p95 / box scale |0.029534|0.033722|

Normalization round-trip maximum error was `3.0518e-5` transformed pixels.
The original full-validation replay (2,000 images / 7,074 people) exactly reproduced
posterior-mean pose/box AP `97.246106/98.426622` and three-seed sampled averages
`97.207467/98.382681`. Ground-truth-copy and normalization-only AP were both100/100.
These are codec measurements, not detector results.

Fixed latent statistics include posterior variance:
`mean=E[mu]`, `std=sqrt(Var(mu)+E[exp(logvar)])`, with population moments and a
minimum standard deviation of1e-6. Only augmented trainval targets enter this pass.
Artifacts, target/source/annotation hashes, paired error slices and four qualitative
grids are saved at
`/storage/outputs/pose-graph-diff/results/poseflow_v43_frozen_vae/20261105_093111_codec_audit`.
The statistics file is `latent_statistics.pt`; no test annotations were accessed.

Seven focused adapter/audit tests pass after intended missing-module failures.
A real-data smoke caught missing annotation identity in the export path; preserving
the real source IDs fixed it, with a real-image regression test.

## Implemented LDM

The model has40,241,095 parameters,39,444,135 trainable. V40's shared image frontend
uses the342 ED-Pose tensors at100% required coverage, including encoder layers2/4/6.
Three width256 reference/time-conditioned blocks predict a32D clean latent from
the incoming noised latent. The pretrained codec alone decodes both pose and box;
its574,560 parameters and population-statistics buffers stay fixed. There is no
coordinate bypass or new geometry readout. The remaining frozen parameters belong
to the historical backbone stem/layer1.

Train and inference both select the top50 image references from200 candidates.
References remain fixed across DDIM calls. Independent box-only Hungarian maps
supervise the candidate proposal head and selected slots. Each selected slot's
complete geometry/latent target stays fixed after corruption. Unmatched slots have
zero standardized clean-latent targets and no geometry loss. The maximum valid
person count in the actual trainval annotations is13, below the50-slot capacity.

Final confidence comes from a separate image-only linear head initialized from
ED-Pose's person-class row. Its logits are cached before denoising. The same image
therefore has identical raw per-slot confidence for different latent draws and
K1/K4/K8; geometry-based pose NMS can still change which outputs survive.

The objective is `proposal +2*slot_focal +10*latent +0.5*geometry`. Foreground and
background latent MSE are reduced separately, then summed. Geometry uses the
codec's masked native OKS-denominator squared error, per-axis box-center MSE,
log-size MSE and1-GIoU. Independent posterior and diffusion generators, global
Python/NumPy/Torch/CUDA RNG state, codec weights, statistics, optimizer and scheduler
are checkpointed. Restoration constructs modules without initialization assets.

The separate Hydra entrypoint uses the reviewed80-epoch seed42 recipe, batch4 with
accumulation2, FP32, main/backbone AdamW rates2e-4/1e-4, weight decay1e-4, drops at
31/61 and clipping0.1. Every epoch and an independently advancing last checkpoint
are retained. The historical trainval/test and crop policy are explicit: val loss
overlaps training; fixed32 test AP is a development diagnostic, not an independent
validation result. The primary endpoint is epoch79, with full8000 K1/K4/K8 evaluation.

## Verification

Sixty-two focused checks pass in the combined codec/V43/V40/EMA/SigReg regression
suite. Coverage includes strict codec
identity, posterior moments, actual augmentation/export, missing/occluded labels,
tiny positive boxes, outside-image joints, source identity, background handling,
latent-only geometry, image-only confidence, RNG and native checkpoint restoration.
The shared V40 frontend extraction also exactly replays pre-extraction state keys,
initialization tensors, RNG, image context and candidate outputs.

Native FP32 CUDA verification used real CrowdPose batches, batch4/accumulation2 and
two Lightning optimizer updates. Strict restore without initialization assets
reproduced model tensors exactly; controlled next-update losses were both
41.9723625183, with maximum parameter difference9.9449e-8. Codec tensors/buffers
remained bitwise fixed and outside AdamW. Peak allocated CUDA memory was
2,171,255,296 bytes. Real two-image native CrowdPose evaluation ran at K1/K4/K8;
raw confidence differences were exactly0. These are finite/restore/evaluator
smokes, not learned detector-performance evidence.

Verification caught a wrapper eval-mode omission and missing global RNG snapshots;
both now have failing-then-passing regressions. The GPU comparison harness also
needed explicit post-Lightning device restoration and independent copies of AdamW
state, whose CPU step tensors would otherwise alias. Post-fit full evaluation now
explicitly returns the model to its configured device. Failed harness attempts
remain under local `tmp/v43/` for traceability; existing experiments were untouched.

Evaluation-only skip flags are rejected before training initialization and direct
the caller to the separate evaluator; both supported flag locations have regressions.

Native checkpoint,418 verified source/config/script hashes, test logs and a source
snapshot are preserved at
`/storage/outputs/pose-graph-diff/results/poseflow_v43_frozen_vae/20261105_100324_implementation`.
The selected codec and its reconstruction limitations are unchanged. The full80-epoch V43 run was subsequently launched as recorded below; learned
detector AP remains unmeasured at startup.

## TensorBoard diagnostic visibility

The existing port33332 dashboard now includes the separate
`V43-preflight-diagnostic` root, exported from the saved native verification JSON.
It contains single-point preflight checks and explicit scope text; it does not
contain a learning curve or learned detector AP. The diagnostic stream remains separate from the subsequently launched full run. Export provenance and HTTP verification are stored alongside the
implementation artifacts as `dashboard_export.json` and
`dashboard_verification.json`. Existing dashboard roots and six AP pins are retained.


## Full training launch (2026-11-05)

User authorization: "Do the training". Fresh full80 seed42 training runs at
`/home/gasparyanartur/outputs/pose-graph-diff/poseflow_v43_frozen_vae/20261105_103134`
in persistent session `poseflow-v43-frozen-vae`. The live port33332 TensorBoard label
is `V43-frozen-HighKL-VAE-live`. This home-filesystem namespace has sufficient space
for roughly36GiB of full checkpoint history; /storage had only19GB free.
Both V42 trainers remain running.

All418 tested source hashes match the implementation verification. The snapshot
contains579 source/config/test/document files and six verified initialization/annotation
inputs. The reviewed scientific recipe and model are unchanged. The initial
20261105_103027 attempt exited before any training updates: installed Lightning rejects
save_top_k80 with monitorNone. The corrected fresh run uses explicit runtime
checkpoint overrides monitor=epoch/mode=max, preserving all80epochs plus independent
last. The preset default correction remains a maintenance TODO; README includes the
working overrides.

Measured startup reached logged optimizer steps49 and99 with finite total losses
92.4344482 and16.6246567. These are startup checks, not an accuracy result. Native
manifest, frozen workspace, resolved configuration, input/source hashes and launch
verification are saved in the run namespace. Per-epoch fixed32 test K1/K4/K8 and
final full8000 evaluation remain scheduled.

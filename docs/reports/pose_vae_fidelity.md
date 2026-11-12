# Warm-started fidelity VAE comparison

The user accepted the deterministic control's 96.049 pose / 97.915 box validation
AP as the starting fidelity baseline and approved parallel VAE comparisons.
The previous AE's 97.5 gate does not block this experiment. No decoder-only stage,
auxiliary relative-joint loss, image input or detector integration is included.

All arms initialize from the accepted AE checkpoint
`pose_autoencoder_refinement/20261104_225527/clean/checkpoints/epoch098-score96.0492.ckpt`,
SHA256 `e7358ca464155a3a086ee2b6d92bc8501483dfb17bea3dee1dac646cc1cb8dcf`.
Encoder, posterior mean and decoder/head weights transfer exactly. A separate
variance head starts with zero weights and posterior standard deviations equal
to 5% of each initial training latent dimension's population standard deviation.
Exact mean and decoded geometry parity are verified before optimization.
The new family's logvariance safety bounds are [-30,10], versus [-10,2] in the
original VAE, allowing the training-scaled initialization down to its numerical
std floor of 1e-6. These bounds do not enforce a useful noise level. Posterior
shrinkage must therefore be reported explicitly; high sampled AP alone cannot
establish robustness. Original VAE behavior is unchanged.

## Matrix and objective

Each row runs both without synthetic masking and with per-person masking
probability sampled uniformly from 0 to .3: eight concurrent arms in total.

| Arm | Training latent | KL target fraction at initialization |
| --- | --- | ---: |
| Deterministic control | Posterior mean | 0 |
| Sampling diagnostic | Posterior sample | 0 |
| Low KL VAE | Posterior sample | .01 |
| High KL VAE | Posterior sample | .10 |

KL fractions calibrate fixed coefficients as
`beta = fraction * initial_sampled_reconstruction / initial_KL`, using all 35,659
training people, unflipped, with a fixed posterior draw. This is an initialization
calibration, not a constraint on later KL fractions. Initial reconstruction is
.166883 for samples versus .036609 at the mean; initial KL is 200.397919.
Coefficients are approximately 8.327568e-6 and 8.327568e-5, warmed over ten epochs.
KL sums latent dimensions and averages people. Sampling with KL zero is diagnostic;
it cannot establish a useful standard-normal prior.

The reconstruction objective reuses the AE's native OKS-denominator pose loss,
per-axis box-center error, log-size error and GIoU, with equal term weights.
All encoder, decoder and posterior parameters train jointly in sampled arms.
Deterministic controls freeze the unused variance head. Each arm trains 100
epochs, seed42, batch512, FP32, AdamW1e-4/weight decay1e-4, cosine to1e-6, clip1.
All arms retain horizontal flips; coordinate and box jitter remain disabled.

Masked inputs retain at least one observed labeled joint. Artificially hidden
coordinates remain clean reconstruction targets. Naturally unlabeled joints
remain excluded from loss and evaluation; labeled occluded joints are retained.
Separate checkpointed random streams for flips, masks and posterior noise keep
training input draws comparable across arms.

## Evaluation and verification

Every five epochs, full-validation mean and fixed posterior-sample native pose
and box AP are recorded. Checkpoint selection maximizes the minimum of those
four AP values. This is a comparison criterion, not evidence of completed quality
or a guarantee of missing-joint accuracy. The initial checkpoint is preserved.

Final assessment includes train/validation mean and sampled AP, paired OKS/IoU,
geometry tails, per-joint/person-size errors, KL and posterior/aggregate-mean
standard deviation ratios. Three fixed seeds measure latent noise at 0,1,2.5,5,10%
under both fixed initial training std and each final model's training std. Both
scales are needed to interpret latent rescaling. Held-out-joint evaluation hides
10,30,50% of labeled joints using paired draws over three seeds; observed and
hidden errors are reported separately, with mean and sampled AP.

Qualitative artifacts include varied and worst reconstructions, posterior samples
and unmatched standard-normal prior draws. Prior grids are inspection evidence,
not a measured generation benchmark. Naturally missing-joint recovery cannot be
scored without additional ground truth.

Six new focused checks plus 37 existing codec/shared-runner regressions pass,
including CPU/CUDA native model/optimizer/scheduler/RNG restoration with an exact
next update. Sampled/masked and deterministic two-epoch GPU smokes completed all
assessment paths. These are wiring evidence. Each arm saves best/last checkpoints,
resolved config, source/annotation/initialization hashes and explicit test-access
flags. Only train and val annotations are opened; test remains untouched.

## Usage and status

Single-arm entrypoint:

```bash
PYTHONPATH=. venv/bin/python scripts/training/train_pose_vae_fidelity.py \
  initial_checkpoint=/absolute/path/to/accepted_ae.ckpt \
  pipeline.mode=sampled pipeline.kl_fraction=0.01 pipeline.mask_max=0.3
```

Set `pipeline.kl_fraction=0` for the sampling diagnostic; also set
`pipeline.mode=deterministic` for the AE control. Set `pipeline.mask_max=0` for
unmasked arms. A shared `initialization_file=/absolute/path/to/initialization.pt`
preserves identical initialization and calibration across a comparison.

The eight-arm launch freezes source and the shared initialization into a fresh
`results/pose_vae_comparison/` namespace, retaining the exact orchestration script.
`launch.json`, `progress.json`, per-arm `fit_complete.json` and `completion.json`
distinguish running, fitted and fully assessed experiments. All eight assessments
are complete; the results below separate fidelity gains from remaining prior,
masking and noise limitations.

Eight arms launched 2026-11-04 23:34 CET: `/storage/outputs/pose-graph-diff/results/pose_vae_comparison/20261104_233335`. Supervisor PID819048; frozen source 509 files verified without mismatches. Training and assessment are complete, including the checkpoint audit described below.

## Completed eight-arm experiment

All eight arms completed 100 epochs / 7,000 optimizer updates each (56,000 total), then full train/validation, three-seed posterior/noise and held-out-joint assessments. All processes exited successfully. Results below use the selected checkpoint; posterior-sampled AP averages seeds1042/2042/3042. One training seed was used.

| Arm | Mask training | Clean sampled pose/box AP | 5% latent-noise pose/box AP | 30%-hidden joint median error |
| --- | --- | ---: | ---: | ---: |
| Deterministic | no | 97.399/98.536 | 80.662/87.685 | 14.672% |
| Deterministic | yes | 89.912/98.104 | 76.174/87.252 | 7.083% |
| Sample, KL0 | no | 97.302/98.422 | 81.816/87.971 | 14.599% |
| Sample, KL0 | yes | 89.892/98.051 | 76.834/87.517 | 7.073% |
| VAE, low KL | no | 97.308/98.467 | 81.924/87.982 | 14.611% |
| VAE, low KL | yes | 89.874/98.057 | 76.832/87.515 | 7.073% |
| VAE, high KL | no | 97.207/98.383 | 82.743/87.997 | 14.621% |
| VAE, high KL | yes | 89.761/98.049 | 77.184/87.620 | 7.074% |

Noise is applied around posterior means, with std5% of each final model’s training latent std. It is not additional noise on top of posterior samples. Fixed-initial-reference curves are also retained. Hidden-joint error is normalized by target max box extent; the table averages three seed medians at30% hiding and uses posterior-mean decoding. Naturally unknown labels remain excluded.

### Findings

Warm initialization plus the revised objectives recover sampled clean fidelity close to the deterministic control. The sampled unmasked arms are nearly tied around97.2–97.3pose AP; differences at this scale should not be treated as a robust model ranking from one training seed. The original VAE’s poor sampled reconstruction is therefore not an unavoidable consequence of this model width/latent dimension, though this combined intervention does not isolate which original recipe component caused its failure.

The gain coincides with strong posterior contraction. For unmasked KL0/low/high arms, the validation posterior-to-aggregate-mean std ratios are .125%/.207%/.638%, respectively, versus approximately5% initially. High-KL conditional std averages .00645, all32mean dimensions remain active, and logvariance clamp saturation is zero. Thus the contraction is learned rather than caused by the numerical safety bound. Pointwise KL is243.985nats for the high-KL arm; this is not an aggregate-posterior KL estimate.

The high-KL unmasked VAE improves 5%-noise pose/box AP from the matched deterministic control’s80.662/87.685 to82.743/87.998. This is a modest gain, with a large remaining drop from clean fidelity. The models remain sensitive to larger latent errors.

The ranking depends on the perturbation definition. At5% of the shared initial
training latent std, deterministic/KL0/low-KL/high-KL unmasked pose/box AP is
79.655/87.308,81.547/87.884,81.351/87.766,79.941/86.778 respectively. The
high-KL arm therefore does not improve both metrics under the fixed-reference
probe. Per-model std provides a scale-normalized probe; fixed-reference std
holds initial coordinate units constant. Both are reported, and no single
uniform robustness winner is established across these definitions.

Masking approximately halves hidden-joint median error:14.67%→7.08% of box scale in deterministic controls, and14.62%→7.07% in high-KL VAEs. However, high-KL clean sampled pose AP falls97.208→89.761 with masking training. Under30%input hiding, high-KL mean pose AP improves43.462→57.751; hidden-joint p95 remains approximately27.3% of box scale. This is useful but incomplete recovery, not reliable missing-joint reconstruction.

Standard-normal prior inspection still shows many malformed skeletons and incoherent pose/box pairs. This is qualitative evidence, not a generative benchmark. The empirical high-KL aggregate posterior mean norm is 11.7816 (standard-normal target0), with average diagonal variance 1.6854 (target1). Clean AP alone therefore does not establish useful prior matching.

### Checkpoint audit

The final audit found that the installed Lightning default `ModelCheckpoint.on_validation_end` saves `last` only when top-k saving occurs in that step. Consequently the original KL0-masked and low-KL-masked `last.ckpt` files retained epoch94/step6650 although both arms completed and validated epoch99/step7000. Their selected best checkpoints and assessments are valid.

The fidelity preset now enables the repository’s existing `save_last_every_epoch`
policy, with a failing-then-passing recipe regression. Both affected arms were
replayed for100epochs from identical initialization in fresh
`checkpoint_replays/` namespaces. **Selected model tensors and all20validation
snapshots match their originals exactly.** The replayed final native checkpoints
retain epoch99/step7000, including optimizer/scheduler/private RNG states.
Original files are preserved. Assessments were reused only after exact selected
model parity was verified. These are recovered replay artifacts, not claims of
having retained the original unsaved final tensors. Replay audits record their
paths and provenance; the two repeats add14,000verification updates, without
additional optimization of the selected scientific candidates.

### Artifacts and limits

Run: `/storage/outputs/pose-graph-diff/results/pose_vae_comparison/20261104_233335`. All509frozen source/config/test files and two annotation hashes verified without mismatches. All43unique focused/codec/shared-runner tests pass; the checkpoint-policy-focused19checks were rerun after the policy fix. `results_summary.json`, `comparison.md`, learning/noise/masking plots, raw geometry, per-joint/size slices, posterior/prior grids and original best checkpoints are retained.

All encoder, posterior and decoder optimization was joint. No decoder-only robustness stage, coordinate jitter, auxiliary relative-joint objective, test evaluation or downstream integration occurred. The experiments recover clean fidelity and expose remaining noise/prior/masking tradeoffs; they do not establish the requested broadly robust probabilistic codec.

Final checkpoint audit completed 2026-11-05 00:22 CET. All eight final native states are available at step7000. For both replayed arms, selected model, optimizer, scheduler and all private RNG states match exactly, as do all20validation snapshots. `checkpoint_recovery.json` maps final-state paths and hashes. Source and annotation audits remain clean; test was untouched.

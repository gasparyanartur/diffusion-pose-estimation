# V54/V55 pose bottleneck and prefix-attention audit

Date: 2026-11-16. Frozen diagnostics; no training or production model changes.

## Findings

The strongest measured issue is **loss of working pretrained behavior when the
native model is reduced**, rather than an inherent inability to predict people
independently. Retaining encoder layers 2/4/6, box layer 2 and pose layers 4/6
copies pretrained tensors but does not preserve the original computation.
Later blocks receive features and reference points without the preceding
blocks that produced their training inputs.

The trained model's remaining deficit is primarily joint geometry. Stopping,
confidence and proposal availability matter, but removing those limitations
does not recover 50 pose AP on the measured checkpoint. Prefix attention has
not demonstrated a clear pose regression or a substantial pose benefit.

## Scope and provenance

- Original V55 run:
  `/home/gasparyanartur/outputs/pose-graph-diff/poseflow_v55/20261114_184631`.
- Immutable selected checkpoint: `seed42/checkpoints/epoch046.ckpt`;
  SHA256 `bcfcd5f1187537756bbcf66d259279e04c32d9141022fe3d0021412154f3519b`.
  Epoch indices are zero-based; this checkpoint follows 47 completed epochs.
- Executed against that run's archived Python/native source through PYTHONPATH.
  All 644 pinned source hashes checked unchanged. The current split-clipping
  trainer was preserved; no optimizer steps, restart or dashboard changes.
- Same fixed32 **VAL** images, 103 valid posed people, FP32 native CUDA,
  resize 256/max427, TF32 disabled, native CrowdPose evaluator/postprocessing.
  This fixed32 contains no empty images. Missing-joint masks are preserved.
  TEST was not opened. These are subset diagnostics, not a new full-split result.
- Baseline replay exactly matches all five saved evaluation metrics. Replaying
  the generated prefixes/selected identities matches sequential coordinates
  within 1e-5. Trained checkpoint/model hashes remain unchanged.
- Pretrained asset SHA256:
  `0f50dae126a73ad2a15c41906c9ba6d8efa3421ef6cc2fbdc878238bd26b8f93`.
  Fresh reduced-model hash exactly matches the original run's initial hash
  `d4319444b4bd15731c243c4e206a1ec0d169c84b2ec57cf4190a33fed7897e64`.
- Pretraining exposure is not certified by an annotation manifest inside this
  asset. The bundled upstream CrowdPose training loader uses train+val. Thus
  pretrained VAL AP below is a transfer-preservation diagnostic; it must not
  be presented as proof of performance on unseen data.

All AP/AR numbers below use the 0–100 scale. JSON artifacts use 0–1.

## Does prefix attention make performance worse?

Completed 80-epoch runs, each selected by the established fixed32 rule and then
evaluated on all 2,000 VAL images:

| Model | Selected epoch index | Pose AP | Box AP |
| --- | ---: | ---: | ---: |
| V54 pooled prefix, empirical augmentation | 39 | 29.39 | 43.07 |
| V55 proposal-to-prefix attention, same augmentation | 46 | 29.52 | 42.39 |

Pose differs by +0.13 AP, box by -0.68 AP. Individual fixed32 epochs fluctuate;
this comparison does not establish that attention harms pose. Separate trained
runs and independently selected checkpoints also do not isolate causality.

Frozen intervention on the original V55 selected checkpoint holds generated
prefix coordinates, selected proposal identities, counts and scores fixed:

| Intervention | Pose AP | Box AP |
| --- | ---: | ---: |
| Actual V55 births | 34.70 | 45.79 |
| Replace corrected proposal feature with its raw image feature | 34.17 | 45.75 |
| Remove decoder prefix, keep corrected selected feature fixed | 34.70 | 45.67 |
| Raw feature and no decoder prefix | 34.17 | 45.75 |

The learned correction is slightly helpful at these trained weights. Removing
the decoder's separate prefix path changes coordinates, but no pose-AP threshold
crossings on this subset. This is not an untrained V54 reconstruction: V55's
downstream weights were trained with the correction.

Two distinct paths must be distinguished. `proposal_context_v55.py` lets each
proposal attend individual prior people before selection/expansion. Separately,
`expanded_pose_v53.py` lets the new person query attend prefix tokens, while
joint queries are masked from direct prefix attention. Joint effects through
that second path are indirect across the two pose layers. Thus adding prefix
attention does not itself provide additional joint-specific image decoding.

The earlier [trained gradient audit](poseflow_v55_trained_gradient_audit.md)
found shared context consuming about 98% of the geometry family's squared
gradient norm under the original clipping policy. The separate fresh
split-clipping run addresses that measured optimization coupling. This audit
does not identify it as the sole cause of the box difference, nor does it show
that attention is intrinsically harmful.

## What prevents the trained checkpoint from reaching 50 pose AP?

| Frozen condition | Pose AP | Pose AR | Box AP |
| --- | ---: | ---: | ---: |
| Actual sequential output | 34.70 | 41.36 | 45.79 |
| Disable pose NMS | 34.70 | 41.36 | 45.78 |
| Score with native objectness only | 34.93 | 41.36 | 45.96 |
| GT-assisted whole-pose quality scores | 41.49 | 41.36 | 45.87 |
| Predict all 50 proposals independently, empty prefixes | 37.26 | 45.83 | 51.69 |
| GT prefix and matched proposal identity, constant scores | 28.78 | 46.50 | 47.19 |
| Same teacher inputs with GT proposal box | 28.96 | 46.41 | 82.42 |
| Replace joints of already well-localized detections with GT joints | 74.71 | 76.12 | 45.78 |
| Exact valid GT geometry through normal postprocessing | 100.00 | 100.00 | 83.17 |

The top50 proposals cover 100/103 posed people at box IoU >=0.5 and 87/103 at
IoU >=0.75. Actual inference emits 85 people before NMS, 83 after. Stopping is
therefore imperfect, but all50 improves pose by only 2.56 AP while box improves
5.90 AP. It does not resolve joint localization.

The joint-replacement diagnostic pairs whole people one-to-one using generated
box IoU >=0.5. It keeps generated boxes and confidence, substitutes the matched
GT person's joints, and runs normal postprocessing. It demonstrates that the
existing detections could support substantially higher pose AP with accurate
joints. It uses GT and is not an inference result. Likewise, quality-score
replacement uses a whole-pose OKS score, never independently matched joints.

Teacher rows contain GT previous people and a known Hungarian-matched image
proposal, removing generated-prefix errors and incorrect next-person choice.
Their constant confidence makes AP unsuitable for a direct comparison with the
baseline's learned confidence; AR is the useful comparison. AR remains 46.50,
and replacing the proposal box with GT barely changes pose despite correcting
box AP. This is stronger evidence for articulation error than box drift alone.

On those teacher rows, mean same-person OKS (the benchmark's normalized
skeleton similarity) is 0.664; 47.6% of people exceed 0.75. Even among 85 people
with resized square-root box area >=64 pixels, mean OKS is only 0.693 and 51.8%
exceed 0.75. Small people are difficult, but small-person resolution alone
cannot explain this sample's deficit. Wrists/elbows show large pixel errors;
RMSE is sensitive to catastrophic outliers and is not a new AP metric.

Both pose layers work: actual births improve from 29.48 to 34.70 pose AP between
the first and second layer. The expansion references before pose decoding score
only 0.64 AP. There is no evidence here of an entirely disconnected pose head.

## The pretrained reduction loses the working pose computation

Frozen original pretrained weights, no optimization, identical fixed32 and
resize. Every condition predicts all50 image proposals, uses native proposal
objectness and the same postprocessing; there is no GT-conditioned sampling,
EOS or diffusion. Independent groups cannot attend another person's state.
These controls retain our 200-query/50-proposal settings and proposal-birth
objectness scoring. They restore original block depth, not every upstream
evaluation setting or the upstream final-stage classifier; they are not a
reproduction of the paper's published AP.

| Frontend and pose blocks | Pose AP |
| --- | ---: |
| Full six encoder/two box blocks; pose blocks 3/4/5/6, native simultaneous groups | 54.20 |
| Same full frontend; all four pose blocks, independent groups | 54.15 |
| Same full frontend; first pose block 3 only, independent groups | 51.11 |
| Same full frontend; first two consecutive pose blocks 3/4, independent groups | 53.93 |
| Same full frontend; only retained pose blocks 4/6, independent groups | 16.08 |
| Reduced encoder2/4/6 + box2; all four pose blocks, independent groups | 0.73 |
| Reduced encoder2/4/6 + box2; retained pose4/6, independent groups | 0.10 |

The first four rows are layer readouts from the same full pretrained decoder,
not separately trained models. The reduced frontend/four-pose condition and
full frontend/four-pose condition share the same original pose parameters.
The sparse two-block condition keeps the same expansion head and source4/6
pose blocks/heads, changing only which pose stages run.

This isolates two destructive changes at initialization:

1. With the full frontend fixed, skipping the initial pose stage and intermediate
   stage drops independent-group AP from 54.15 to 16.08. The first two consecutive
   stages already score 53.93. **The issue is not simply needing more than two
   layers:** preserving their trained sequence and reference updates matters.
2. With all four pose stages fixed, replacing the full frontend with the current
   reduced frontend drops AP from 54.15 to 0.73. This bundles encoder truncation
   and box-stage truncation; this audit does not separately attribute their
   contributions.

The original transfer coverage checks proved tensor loading, not functional
preservation. That distinction was missed in earlier verification. Training
raises reduced-model pose AP far above initialization, but it is relearning a
pose computation whose pretrained inputs were disrupted. This is a concrete
architectural/initialization problem, not proof of a hard asymptotic ceiling.

Independent pretrained groups retaining 54.15 AP also show that the inability
to reach 50 cannot be assigned to independent person decoding alone. These
controls do not prove that adding learned sequential selection/EOS preserves
54 AP; that remains a separate verification requirement.

For the temporary four-pose transfer control, the generic remapper required an
explicit duplicate mapping of source pose head0 to both expansion and pose
stage0. Its standard path reserves that head only for expansion. Full relevant
parameter coverage was asserted after correcting this **diagnostic** mapping;
production loading was untouched. Current source4/6 retention does not encounter
that head0 mapping case. Independent versus separated-group batched coordinates
also passed numerical parity. All measured model states remained unchanged.

## Implications and remaining work

The best supported next direction is to preserve and verify a working pretrained
frontend/pose path before adding sequential prefix selection. A minimum useful
control can retain the first two consecutive pose stages: adding many unrelated
loss/gradient mechanisms is not required to establish that baseline. This is a
recommendation, not an implemented or launched change.

Still unresolved: how much current training underperformance comes from reduced
frontend versus skipped pose stages after full optimization; whether sequential
selection/teacher forcing can retain the intact baseline; generalization under
certified pretraining exposure; and the independently trained split-clipping
outcome. No causal claim about those outcomes follows from this frozen audit.

## Reproducible artifacts

Repository-local `tmp/v55_pose_diagnosis/` contains `diagnose.py`, `results.json`,
all condition predictions, `transfer.py`, `transfer_results.json`,
`frontend_transfer.py`, `frontend_transfer_results.json`, logs and
`provenance.json`. Scripts use archived V55 source, OMP/MKL threads2 and native
CUDA. They assert baseline/row replay, immutable state, source/checkpoint hashes,
and retained parameter coverage. No production tests were needed for a frozen
diagnostic. A temporary loader coverage rejection exposed the head0 mapping
case described above; the corrected diagnostic asserts relevant key coverage.

# V42 proposal–pose correspondence audit

The current V42 training recipe does not align a noisy pose's source person
with its slot's image proposal or final supervision target. V40 had an explicit
training mechanism for that alignment. V41 replaced it with independent banks,
and V42 does not restore it. This is a verified correspondence problem; its
causal contribution to learned state suppression still requires a training
comparison.

## Measured V42 mismatch

Frozen corrected box-center checkpoint: epoch014, 22,500 updates, seed42,
256/427 resize, native evaluator and pose NMS. The fixed32 Gaussian K1 baseline
reproduces every saved metric exactly: pose AP32.573. No optimizer updates or
live training changes were made. Proposal boxes are the exact incoming first
pose-layer references, captured before the pose-stage box update.

Source identity is recovered from the uncorrupted coordinate bank by matching
visibility-masked coordinates to valid target records, requiring joint L1 below
1e-5. It covers all133 fixed32 people and all112 people in32 representative
augmented training images. Padding and cropped-out sources are excluded from
these known-person denominators. The augmented sample uses a frozen model in
evaluation mode and is not a replay of the live worker's batches.

| Where a known-source pose was injected | Same person supervised | Different person supervised | Slot unmatched/background |
|---|---:|---:|---:|
| Fixed32, near-clean t0 | 22/133 (16.5%) | 32/133 (24.1%) | 79/133 (59.4%) |
| Augmented train32, sampled timesteps | 17/112 (15.2%) | 30/112 (26.8%) | 65/112 (58.0%) |

The output labels above are the native final Hungarian assignment of the frozen
outputs. They establish what that criterion would supervise for these inputs;
they are not recorded training-gradient labels from the running worker.

At t0, only30.1% of fixed32 source slots' reference boxes have the source person
as their nearest GT by IoU (identity-permutation chance baseline24.1%). Among
source slots with a reference overlapping some GT at IoU>=.5, identity agreement
is33.3%. Thus the mismatch is not explained just by references lacking people.
Some reference covers94.0% of GT at IoU>=.5, but a source pose's own slot box
overlaps its person at that threshold only30.8% of the time. Mean same-slot
source-person IoU is.305, versus.807 under independent box-Hungarian alignment.

A GT-assisted t0 permutation moves each known-source pose to its box-Hungarian
proposal slot and preserves every input bank entry. Captured proposal references
remain unchanged. Source/person box IoU>=.5 coverage becomes93.2%, with mean
IoU.807. Conditional pose AP changes31.045 ->30.940. This oracle input ordering
is unavailable at inference, and the frozen model's AP does not recover merely
by reordering inputs. That is consistent with weak learned pose-state use; it
does not prove that aligned training will improve AP.

Artifacts: `tmp/v42_uniform_t/correspondence/{report,details}.json`; raw source,
final and geometry assignments are retained per image. The actual source bank
can include cropped-out people, so this audit identifies valid supervised source
people explicitly instead of equating the first GT indices across banks.

## What earlier versions did

| Version | Correspondence mechanism |
|---|---|
| V19 full pose | Current coordinate state defines its box/joint sampling references and residual updates. No independently ranked image proposal is added by slot. |
| V33/V34 full pose | The noisy person latent is the query state. Image attention updates that state; there is no separate ranked image-proposal bank fused indexwise. V34 still rematches denoised targets by OTA, so it does not retain corruption-source identity in all losses. |
| V40 box family | Hungarian-match image candidates first; gather each matched GT into that candidate's slot; encode and corrupt that bank; use the same targets for recovered task and latent losses. References remain fixed during sampling. This establishes training proposal/source/target correspondence. |
| V41 box family | Build GT-coordinate sources independently, select top50 references by score, and match proposal/final outputs separately. Final box and JEPA targets agree with one another, but can differ from proposal or corruption-source identity. References are cached across calls; caching alone does not align the banks. |
| Native DiffusionRegPose | No explicit GT-based proposal/source pairing. Its coordinate-fusion attention mixes noisy joints and proposal-derived joints across the banks. This is not a hard identity guarantee. |
| V42 | Adds the noisy-pose MLP encoding indexwise to score-ranked selected features; anchors image reads and pose readout to proposal boxes; independently matches final outputs. Proposals are recomputed each call. |

V40's implementation is
[select_training_references](../../src/models/pipelines/pose_estimators/poseflow/latent_v40.py#L24),
called before noising in
[the training path](../../src/models/pipelines/pose_estimators/poseflow/training_v40.py#L68).
Its source helper is byte-identical to the original frozen V40 helper, and that
frozen worker uses it for both clean/noisy source construction and denoised task
targets. Current V40 training also adds later EMA/SIGReg extension hooks; those
do not alter this correspondence mechanism. Two existing targeted tests pass:
retaining lower-ranked matched proposals with correctly gathered GT and live
references, and empty/full/invalid targets with no post-denoising rematching.

V41's
[training path](../../src/models/pipelines/pose_estimators/poseflow/training_v41.py#L76)
and coordinate-target builder are byte-identical to its frozen run. Frozen V33
and V34 have the same latent-query denoiser/readout methods inspected here.

V40's selection also keeps all matched candidates before filling the remaining
50 slots by score; inference uses score-top50. Reusing that mechanism therefore
requires an explicit decision about the training/inference selection policy.
Its alignment implementation is not evidence that all inference identity or
performance problems were solved.

## Interpretation and next decision

V42's pose input is informative about a known person, but its slot often reads
another person's image evidence and receives a different target or a background
label. The image/proposal branch provides a coherent competing solution. This
supports the proposed explanation for weak state use, alongside the earlier
coordinate ablations. It does not isolate the sole cause from model capacity,
optimization, matching or pretrained-path strength.

A correspondence-preserving training design must keep proposal assignment,
clean/noisy source construction and denoising supervision consistent. The
cancelled noisy-input-matched variant changed final supervision but did not align
the image reference bank. A new implementation or training comparison requires
a reviewed plan. No production model, losses, configs or running jobs were
changed during this audit. Cross-call proposal-identity stability was traced in
code but not measured here.

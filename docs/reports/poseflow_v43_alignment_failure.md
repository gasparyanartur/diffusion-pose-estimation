# V43 alignment failure: proposal damage and unsuitable query initialization

## Cancellation and logging

The user explicitly cancelled the alignment run. Only verified PID956042 received
SIGINT. It exited after the sixth epoch had begun, with the last progress message
at update6950. The best epoch000 checkpoint and last completed epoch004 checkpoint
(global step6250) are preserved with hashes in `user_cancellation.json`. There is
no interrupted checkpoint; no claim is made that mid-epoch weights were saved.
The original V43 LDM and the other two trainers were left unchanged.

Metrics were placed under `alignment-val/` by an implementation choice, while the
user's pinned cards display `quick-test/`. There was no technical need to make the
metrics undiscoverable in that view. This is a separate logging mistake. Future
alignment AP should use a clearly labeled `quick-test/image_alignment/...` condition
and be included in the comparison view, without pretending it is Gaussian-start
multistep diffusion. Existing history and dashboard roots are preserved.

## Confirmed training failure

The alignment stage did not isolate image-to-code learning: it trained the
pretrained image/proposal frontend jointly with an untrained latent regressor.
The proposal geometry and selection degraded badly during optimization. This
changes the source boxes, features and assignments on which the regressor depends.
It is a training/setup failure, not evidence that the frozen VAE is intrinsically
unusable or that the issue is merely generalization.

The full2000-image VAL logs give these batch/person-averaged coverage diagnostics:

| Completed epoch | Candidate recall at IoU0.5 | Selected50 recall at IoU0.5 | Matched selected IoU | VAL latent foreground MSE |
|---|---:|---:|---:|---:|
| 1 | 84.96% | 72.24% | 0.5908 | 0.3837 |
| 2 | 0.00% | 0.00% | 0.0581 | 0.5859 |
| 3 | 63.51% | 50.17% | 0.4407 | 0.4373 |
| 4 | 0.00% | 0.00% | 0.0618 | 0.5606 |
| 5 | 25.52% | 4.63% | 0.1705 | 0.4628 |

TRAIN also fails: selected-proposal recall averages about13.2%,22.8%,10.4%,14.5%
in epochs2–5, with large raw pose losses. This is not a healthy training fit with
a validation-only gap.

## Demonstrated damaging gradient route

The pose/box regressor conditions on differentiable proposal boxes through
`reference_box_embedding` (`model_v43.py` `_predict_head`). The raw OKS-denominator
squared pose loss backpropagates through that conditioning into the proposal-box
head as well as through the image features. The proposal head can therefore be
optimized to help encode a pose rather than remain an accurate detection box.
There is no stop-gradient or frozen frontend boundary for this alignment stage.

At the pinned epoch004 checkpoint, on real VAL image100163/eight people, gradients
with their actual training coefficients are:

| Loss term | Proposal-box head gradient norm | Cosine with native proposal gradient |
|---|---:|---:|
| proposal | 37.545 | 1.000 |
| positive_latent | 11.187 | -0.724 |
| background_latent | 19.895 | -0.681 |
| pose | 246.903 | -0.521 |
| center | 6.287 | -0.395 |

The weighted pose gradient is 6.58 times the native proposal gradient and
opposes it (cosine -0.521). This is direct evidence of conflicting
supervision at this checkpoint/batch; it is not a claim to have replayed every
actual AdamW update or proven a unique cause of the entire learning trajectory.
The checked batch's minimum box side is16.18pixels, so this conflict does not
require a pathological crop sliver. Unbounded squared OKS exponents and crop
slivers remain amplification risks; flooring the center denominator does not
bound the pose term.

A same-forward gradient isolation test detaches only reference boxes in the
conditioning path. Predicted codes remain bitwise identical, but the pose gradient
into the proposal-box head changes from246.903 to exactly0. This proves the route.
Other frontend image-feature gradients remain; detaching boxes alone is not a
claim that all frontend interference is solved.

## Initial person-query symmetry

The new image-only method constructs zero state for every query. The only
person-specific information enters via reference-conditioned modulation, whose
weights and bias initialize tozero (`model_v26.py` `LatentImageBlock`). Consequently
all50 query codes/decoded poses are identical within an image at initialization;
measured query-code spread is exactly0 on all8TRAIN and8VAL diagnostic images.

This symmetry does break during training: the last checkpoint has query spread
about0.31–0.33. Thus permanent identical-query collapse is rejected. Nevertheless,
removing the noisy-code input while retaining zero-gated conditioning discarded
the pretrained person-specific query signal at the start. The proposed alignment
baseline should have used proposal-specific feature queries directly and verified
their distinctness before full training.

## Frozen intervention and evaluator checks

On the same8TRAIN/8VAL images, restoring only the original pretrained frontend
inside the last-checkpoint model restores selected-proposal recall:

| Population | Original initialization | Trained last | Last body + restored initial frontend |
|---|---:|---:|---:|
| train eight images | 67.50% | 2.50% | 67.50% |
| val eight images | 47.62% | 9.52% | 47.62% |

Decoded poses remain poor after restoring the frontend (VAL pairedOKS about0.065).
Therefore frontend repair alone does not rescue the already-trained regressor,
and the intervention does not prove that this is the only source of failure.
It does show that useful proposals were lost, making the original experiment
confounded. Both TRAIN and VAL geometry are poor.

The full64-image saved native monitor is replayed exactly from the pinned last
checkpoint, including correct/wrong-image AP and paired geometry. This excludes
an accidental metric replay/version difference as the explanation. Frozen model
states are restored after interventions; no optimizer updates or production code
changes were made. The frozen VAE's established clean reconstruction ceiling is
not evidence of image-conditioned prediction quality.

## Concrete corrections before any new training

1. Keep the pretrained image/proposal frontend fixed while testing the mapping
   from its features into the frozen VAE. Verify proposal boxes, rankings and
   source assignments stay identical across regressor updates. This makes the
   alignment test interpretable.
2. Initialize each image-only query from its selected proposal feature/box
   embedding rather than identical zero tokens. Verify that different people
   produce distinct initial query states; do not silently alter legacy denoising.
3. Protect proposal boxes from latent/geometry feedback. The detachment test
   proves the precise head route; any later frontend fine-tuning must explicitly
   balance and inspect geometry/proposal gradients. Revisit raw squared pose loss
   sensitivity separately, rather than changing several objectives blindly.
4. Put labeled alignment AP back in the `quick-test` dashboard view. Preserve old
   tags/history; do not alias these metrics to nonexistent diffusion noise/steps.

These are proposed code corrections, not a newly launched experiment. Their
interface/parity/gradient/query/source-invariance checks must pass before another
training decision. No new optimization has been launched after cancellation.

Artifacts: `results.json`, `gradient_isolation.json`, `native_replay/epoch004`,
`diagnose.py`, logs and the cancellation record. The checkpoint and source hashes,
image IDs, replay assertion and zero-update scope are recorded in those artifacts.

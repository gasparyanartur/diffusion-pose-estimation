# V50 gradient and optimizer balancing audit

2026-11-12. Frozen epoch047 (48 completed epochs), SHA256 `d9ccf2b2471e7a8841e809cbe9f0194aa38046eb3959657c1f68b90ad7ac3d29`. Diagnostic only; production model/config/optimizer and live V50/V42 trainers unchanged. TEST unopened.

Eight augmented TRAIN microbatches (batch4), averaged in pairs to match accumulation2:32 images,104 annotated people, four effective batches. Full native training forward includes selected-slot assignment, native box/encoder/DN objectives, both row layers and activation checkpointing. AdamW moments and LRs are restored from the checkpoint. Per-task gradients are decomposed from each same forward; no optimizer steps on the model.

The old calibration was already gradient-based, but it measured only four geometry tasks on shared dense decoder parameters once at initialization. It excluded EOS, selection, native losses, coordinate encoding, time conditioning and optimizer/clipping effects. It does not establish whole-model balance.

## Parameter ownership and routes

- Coordinate encoder, coordinate projection and reference projection receive birth/refinement geometry plus selection/EOS gradients. Selector prefix pooling occurs before rank/time/row decoding; therefore selection/EOS do not train time embeddings or dense row layers.
- Selected proposal content and boxes are detached on the geometry identity path; selection keys still backpropagate into native proposal features. Geometry image cross-attention remains connected to image memory and frontend.
- Birth latent heads/codec reader receive birth geometry only. Residual heads receive refinement only, with separate joint and box output rows. Private head magnitudes cannot be compared as if all tasks share those parameters.
- Image backbone/projections/encoder receive geometry, selection/EOS and native detection losses. Native proposal decoder/heads receive classification/native losses, with no geometry gradient through the detached hard identity gather.

## Measured gradient magnitudes

Medians across four effective batches. Entries are L2 norms of already weighted gradients. Birth/refinement columns combine pose and box vectors before taking norms. Cross-task comparisons within one block are valid; use the accompanying RMS/relative-gradient CSV when comparing different block sizes.

| Parameter block | Birth geometry | Refinement geometry | EOS | Selection | Native detection |
|---|---:|---:|---:|---:|---:|
| coordinate_encoder | 0.00727 | 0.00483 | 0.08170 | 0.10603 | 0 |
| coordinate_projection | 0.05901 | 0.00642 | 0.24952 | 0.29100 | 0 |
| reference_projection | 0.02571 | 0.00286 | 0.14717 | 0.15791 | 0 |
| row_layer_0 | 0.11971 | 0.04550 | 0 | 0 | 0 |
| row_layer_1 | 0.01292 | 0.03138 | 0 | 0 | 0 |
| time_embedding | 0.03720 | 0.14615 | 0 | 0 | 0 |
| readout_norms | 0.01626 | 0.23477 | 0 | 0 | 0 |
| image_backbone | 0.17606 | 0.12013 | 0.72096 | 5.02437 | 10.74559 |
| image_encoder | 0.09830 | 0.07323 | 0.30136 | 2.04270 | 5.08663 |
| residual_heads | 0 | 0.92300 | 0 | 0 | 0 |

At the coordinate encoder, selection/pose-refinement median-norm ratio is32.4 and EOS/pose-refinement25.0; at the coordinate projection they are67.6/58.0. Adam-preconditioned directions retain the disparity (encoder selection/pose-refinement36.5). These directions use the historical second moment, not an exact additive attribution of Adam updates.

Imbalance reverses by location: refinement/birth family ratio is~0.66 in the coordinate encoder,~0.11 in coordinate projection,~0.38 in row layer0,~2.43 in row layer1,~3.93 in time conditioning and~14.4 in readout norms. Multiplying refinement globally cannot equalize all these blocks.

Pose/box coefficient priorities are not realized as gradient ratios. Birth box:pose coefficients3:1 produce almost1:1 individual gradient norms in row layer0 (.0771 vs.0714). In the backbone, box-refinement norm is.09364 versus pose-refinement.02078 (~4.5x), despite the older-token pose preference. Equal raw losses or coefficients do not establish intended optimization priorities.

## Clipping and AdamW

Global gradient norms are22.92/20.87/24.28/19.64; clipping at.1 multiplies all current gradients by.00436/.00479/.00412/.00509. Image/native-frontend parameter blocks contain98.25–99.65% of the total squared gradient norm. This is parameter ownership of the combined gradient, not a claim that one loss contributes exactly that fraction. Native detection and selection dominate those blocks.

Global clipping couples private geometry learning to large frontend gradients. AdamW adapts to historical gradient scales, so this is not equivalent to a200x lower effective learning rate. We therefore also evaluated the next-update formula with stored first/second moments, bias correction, original per-group LRs, weight decay and clipping. Exact task attribution is nonlinear; g_task dot delta_parameter is only a first-order local loss-change estimate.

The sum of all seven decomposed gradient vectors matches total-loss gradients to relative error at most2.08e-6. An independent float64 formula check matches torch AdamW exactly (maximum discrepancy0). All counterfactuals use the same saved moments; no AP or learning improvement is claimed.

## Counterfactual controls

- Global scalar norm equalization: inverse whole-model task norms, multipliers clipped[.1,10]. It increases some geometry descent estimates but cannot fix the parameter-specific imbalance; it mixes private and shared parameters when deciding weights. Not selected.
- Coordinate routing: remove selection/EOS gradient only from coordinate encoder/projection. This protects geometry features from classification training; private selector and candidate-key/native routes remain available. A production implementation should consider detaching the complete geometry prefix at the selector boundary, also protecting the shared reference projection; that exact broader detach is not simulated here.
- Routing plus bounded geometry-family balance: per-block birth/refinement multipliers[.25,4], only on geometry-shared blocks. This does not address global frontend clipping and does not reliably improve local descent versus current. Arbitrary equalization is insufficient.
- Disjoint family clipping: native/image parameters, private selector parameters, and all geometry parameters receive separate clipping thresholds .1/sqrt(3)=.057735 each, after accumulation. Thus the concatenated norm remains<=.1. This is a transparent diagnostic allocation, not an established optimal budget.
- Routing plus disjoint family clipping: geometry clipping allowance increases5.07–13.00x; predicted weighted pose-refinement decrease increases3.39–9.58x. Native loss descent estimates remain negative on all four pairs, but are smaller. This is an optimization trade-off, not a proven harmless change or an improvement in denoising/AP.

Historical optimizer momentum persists after routing: some encoder update directions still increase pose-refinement loss locally. A new recipe needs a fresh run or explicitly reviewed optimizer-state treatment; do not silently resume old moments under changed routing/scales.

## Recommended implementation sequence (not implemented)

1. Separate gradient responsibilities. Let the selector/EOS read geometry-prefix information through a stop-gradient boundary. Keep selector query/key parameters and native candidate-feature gradients trainable. Preserve image-only confidence/native losses; do not discard useful proposal training. Verify birth/refinement gradients still reach coordinate encoding.
2. Replace one global clip with disjoint native/selector/geometry clipping budgets after accumulation. Start the controlled comparison with equal squared-norm budgets (.057735 each; combined<=.1). Log norm before/after clipping per family and relative parameter updates. This directly removes measured private-geometry starvation by frontend scale.
3. Stabilize forward conditioning before recalibrating task weights: the earlier time-embedding outlier suppresses coordinate sensitivity. Gradient rescaling alone is not a validated repair; refinement-only fresh-noise controls previously failed too. A bounded additive conditioning or carefully initialized adaptive normalization is a separate reviewed architecture choice.
4. Recalibrate geometry on coordinate encoder/projection and both decoder layers, not only one concatenated decoder vector. Track birth/refinement and pose/box separately, then apply intended box/pose priorities after normalization. Use bounded, slowly updated multipliers rather than inverse loss values or an unlimited response to flat refinement. A proposed probe interval is100 updates, smoothing coefficient.95, multiplier range[.25,4] relative to a reviewed warm-up baseline; these settings are proposals requiring a controlled test. Near-zero/missing gradients must not be amplified. Scalar weights cannot equalize all modules; log residual imbalance and introduce per-block task scaling only if the simpler routing/clipping recipe fails.
5. Require fresh-noise fixed-identity learning and held-out-noise improvement before a full run. Include matched controls for routing, clipping and conditioning; report AP separately. Preserve the prior pose/box task definitions and age reductions unless a separately reviewed experiment changes them.

Periodic task probes add multiple backward passes; normal training should retain one backward per microbatch with cached scalar weights and ordinary gradient routing/clipping. A block-specific, task-specific balancer generally needs task-gradient separation; its cost and changed update rule would need explicit review.

Gradient conflict changes sign: encoder pose-refinement versus selection cosine ranges-.499 to+.468 across the four effective batches. The data support robust magnitude/routing issues more strongly than a uniformly antagonistic direction. Defer gradient projection initially. [PCGrad](https://arxiv.org/abs/2001.06782) projects conflicting gradients; [GradNorm](https://proceedings.mlr.press/v80/chen18a.html) dynamically tunes task gradient magnitudes. Neither method is implemented or validated by this audit, and the proposed bounded periodic policy is not a claim to reproduce either algorithm.

## Verification and delivery gates

- Exact route tests: classification gradient absent from protected geometry modules, present in selector/native keys; geometry gradients still present in codec/decoder and differentiable image memory.
- Disjoint parameter ownership, correct accumulation2 order, combined clipping bound, no accidental second global clip, FP32/empty-task/missing-joint behavior.
- Checkpoint/save optimizer, scheduler, RNG, gradient-statistics/multiplier state and policy metadata; reject incompatible resumes.
- TensorBoard: raw/weighted losses, per-task per-probe-block gradient norms, family clip factors, Adam relative update norms, forward time/content scales and fresh-noise denoising diagnostics.
- No full run on balancing metrics alone. The independent-noise refinement gate must improve beyond the bypass with seed/coverage reported; retain both native AP and birth/refinement diagnostics.

Artifacts: `tmp/v50/gradient_audit/` and run `diagnostics/gradient_audit_epoch047/`: report.json, summary.json, task_gradient_medians.csv, scripts, logs and provenance/source hashes. Current source hashes match immutable run imports. Existing training remains unchanged.


## Latest-checkpoint confirmation:63 completed epochs

A stable copy of the current last checkpoint (epoch index62/global step78750)
was captured during the audit; SHA256
`997afde2c7bd47aee4e56b0f07dacdd9202f2ff8575f56feae03234bbb5fe90c`.
Repeat the full native decomposition on two additional TRAIN microbatches
(8images/28people), averaged as one effective batch. This is a confirmation
sample, not a second32-image median study. Existing optimizer LR is2e-6 main,
1e-6 backbone and historical moments are retained in the counterfactuals.

Coordinate-encoder weighted gradient norms are pose-birth.003091,
box-birth.007443, pose-refine.0006269, box-refine.002667, EOS.108952 and
selection.104743. Selection/EOS are167x/174x pose-refine. Adam-preconditioned
selection/EOS norms (.02140/.02081) remain~165x/~161x pose-refine(.0001298).
The coordinate projection shows the same imbalance (selection.4690 versus
pose-refine.002681). In readout norms pose-refine(.08961) exceeds pose-birth
(.008398), illustrating why a global refinement multiplier is inappropriate.

Global preclip norm24.3259, clipfactor.0041108;99.556% of gradient energy is in
native image parameter blocks. Routing plus disjoint clipping gives12.95x more
geometry clipping allowance and4.95x larger first-order pose-refinement decrease.
This corroborates the earlier mechanism at the current training age; it remains
a local prediction with the denoising/learning gate unresolved. The latest
checkpoint snapshot, full results, scripts and source provenance are preserved
under `tmp/v50/gradient_audit/latest/` and archived in
`diagnostics/gradient_audit_epoch047/latest_checkpoint_confirmation/`.

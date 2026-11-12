# Removing decoded supervision and V36 failures

Read-only audit on2026-10-29. No model, config, active weights or run changed.
Evidence scripts/results:tmp/v35_decoded_loss_audit/{metrics,probe,image_probe}.{py,json,log}.
Pinned original checkpoints (hard links, to avoid training cleanup races):
/storage/outputs/pose-graph-diff/diagnostics/v35_decoded_loss_20261029/.

## Controlled matched-epoch results

V35 spatial supervised and V35 spatial latent-only resolved configs differ only
in recovered task coefficient0.5->0 and output namespaces. Same image-only
initialization, construction, matching, weights, clipping, data and sampling.
The latent-only restart uses the exact failed attempt's frozen source/config.

After12completedepochs (epoch011, step17999):

| Model | Clean box L1 | Recovered box L1 | Raw latent MSE | Gaussian K1 AP | Gaussian K4 AP |
|---|---:|---:|---:|---:|---:|
| V35 spatial supervised | .042035 | .100615 | .024988 |15.2621|14.0277|
| V35 spatial latent-only | .042343 | .255402 | .022656 |3.9764|3.0992|
| V36 box | .182528 | .298731 | .025885 |2.9127|.9985|
| V36 box+c | .184474 | .457993 | .032824 |3.8214|1.4828|

Losses are ordinary validation fit diagnostics (val overlaps trainval), averaged
across sampled timesteps. Box L1 is sum of four absolute normalized-coordinate
errors per positive proposal, not mean per coordinate. AP is fixed32 held-out,
all50queries/noNMS, seed42+imageID, percentpoints. Single seed and small monitoring
subset; not full8000 results. Do not equate these two evaluation regimes.

V35 clean reconstruction is essentially unchanged; removal hurts recovered
geometry despite slightly lower raw MSE. At the same epoch, score-free IoU50
coverage on133people is87/87 (K1/K4) for supervised versus49/38 for latent-only.
Its above0.5 confident predictions number187/94 versus60/87 supervised. This is
not explained solely by a confidence threshold: geometry/coverage is worse even
when all scores are admitted. Coverage is any-overlap, not one-to-one recall/AP.

## V36 has a missing-input problem before diffusion

Clean input is resized pixel cxcywh/100; target is cx/W,cy/H,w/W,h/H. Removing
all image/context input also removes access to W,H. Different resized images
have different aspect ratios and sizes (actual audit sizes include341x256,
256x385,427x246). Per-box MLP receives neither dimensions nor other boxes.

Exact counterexample: pixel box(128,128,64,64) has normalized target
(1/3,1/2,1/6,1/4) in H256,W384 but(1/2,1/3,1/4,1/6) in H384,W256.
Both V36 models must emit the same output for these inputs, even with c1.
An image-independent encoder is coherent, but the current pixel-to-normalized
contract omits necessary information. We overlooked this in the approved design.
It establishes ambiguity, not the fraction of AP loss caused by that ambiguity;
normalized inputs or explicit frame size require a separately reviewed change
and a controlled retraining check.

V36-box additionally cannot identify borrowed-vs-real existence from geometry
alone. Epoch011 clean positive/negative scores are.416/.406. Box+c fixes this
clean classification (.994/.0045), as expected when c is supplied directly, but
clean geometry remains poor. Perfect clean confidence is not image detection.
Neither outcome establishes that MLP latent representations are inherently bad.

## Frozen probes: what the small MSE hides

Same16images selected from fixed32, grouped4 to retain foreign-donor construction,
fixed proposals/noise, timesteps0/499/999. Pinned checkpoints: V35 supervised
epoch020, V35latent-onlyepoch011, V36boxepoch012, V36box+c epoch011. Unequal ages
mean these probes characterize each checkpoint; they are not a causal matched-age
comparison. The matched-age evidence is the preceding table.

For V35latent-only, t999 assigned-target MSE averages.05152 against target
variation.06199 per channel. The latter is the MSE of an oracle constant fitted
to each batch of assigned targets; the denoiser leaves~83% of that error. It is
not a deployable baseline and uses the current OTA-selected targets. A small
number relative to unit latent norm does not mean accurate recovery of the
variation separating boxes/people. Recovered variance.02067 versus target.06199
shows that only about one third of target variation remains at this high-noise
step. This is a one-call conditional t999 probe, NOT a measurement of later DDIM
calls. Lower-noise results are separately stored inprobe.json.

Decode the assigned clean target directly: box L1.05671. Decode the predicted
latent:box L1.47979. Thus this checkpoint has useful clean codes but fails to
recover enough of their geometry. Clean reconstruction on these held-out16 is
box L1.05658/IoU50 fraction.804. V36clean box L1.17766/.18696 andIoU50 fractions
.519/.495 are already much worse before corruption.

The heads are linear256->4/1 before sigmoid. At fixed weights all five outputs
read a subspace of rank at most5. Oracle correction of the error projected into
that subspace restores exactly the assigned clean decoding; removing error only
in the complementary subspace changes box outputs by under1.6e-6. This is an
algebraic diagnostic using targets, not an achievable AP improvement or learned
result. In latent-only at t999,~67% of squared error lies in the five directions
read by the heads, so it would be incorrect to say most of its error is harmless
or simply in unused dimensions.

At V35latent-only's predicted latent, counterfactual0.5task gradient norm is
~7.68 times the10MSE gradient norm at t999; mean gradient cosine is.292.
This is measured at hhat0, not at backbone/denoiser parameters or AdamW updates.
Removing decoded loss removes a substantial localization/classification signal,
not just a small auxiliary correction. Increasing the scalar MSE weight cannot
make its gradient direction equal to the task gradient. Both gradient directions
have positive average alignment, so this is not evidence of blanket opposition.

## Assignment changes the recovery problem

Actual assign_box_targets uses recovered decoded predictions for OTA, then maps
matched queries to canonical clean person codes; all other queries target a
nearest clean negative code (or zero background fallback). All targets detached.
The latent noised at a slot need not be the target of its MSE.

In the16image V35latent-only probe, construction has27.625positive inputs/image.
At t0, OTA retains6.125positive output assignments/image.45.25% of ALL slots
change from positive source to background target;56.9% of positive matches refer
to a different original person than their source. t0 is the lowest-noise schedule
step, not literally zero corruption. Original annotation indices were mapped
through canonical slots before comparing identities.

Consequently, nearly clean positive codes can be trained toward negative codes.
The optimized quantity is OTA-remapped set prediction in latent coordinates,
not recovery of the original clean sample under the stated forward corruption.
This is inherited from V35, not a newly introduced V36 code bug. The same issue
exists with decoded supervision and therefore cannot alone explain its removal's
AP effect. It is a concrete mismatch to investigate for multi-step deterioration:
DDIM updates assume an endpoint associated with the forward-noised sample while
target selection can switch identity/existence. No isolated causal test of that
mismatch was run in this audit.

## Image conditioning is used

Single fixed batch4, t999, clean/OTA targets held fixed; replace image features
with the other images' features (rolling values, positions and masks together).
Box L1 real->shuffled:V35supervised.218->.796;V35latent-only.547->.878;
V36box.528->.695;V36box+c.513->.963. These are unequal-age checkpoint probes,
not AP comparisons. All degrade: complete image blindness is not supported.
This does not establish sufficiently precise image localization or the exact
training mechanism responsible for the AP gap.

## Supported conclusion and next controls

There are two separable issues: removing decoded supervision leaves useful V35
clean codes but inadequate recovery of task-relevant differences; V36 additionally
weakens the clean geometry target through missing frame-size information, and
box-only lacks clean existence information. Low raw MSE has not established
successful denoising. OTA target remapping/background conversion remains another
measured complication, not a proven sole cause.

First correct/explicitly review V36's coordinate contract before treating these
runs as a test of MLP representation capacity. Then isolate source-target
consistency and preservation at low noise while keeping other settings fixed.
Do not launch another loss/architecture redesign merely from these probes. No
training changes or launches were performed as part of this audit.

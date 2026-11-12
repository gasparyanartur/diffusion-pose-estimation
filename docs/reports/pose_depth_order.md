# Visible-joint mean depth as canonical pose order

Investigation: 7 November 2026. No detector changes or training updates.

## Finding

The proposed target is well defined for people with at least one annotated visible
joint. It can serialize a person set into an ordered scene, provided output slots
are explicitly trained to represent depth ranks. The immediate obstacle on our
CrowdPose annotations is coverage: 41.27% of valid TRAIN people and 40.88% of valid
VAL people have no joints with visibility flag 2. Their requested mean is undefined.
These are annotation-mask counts, not proof that those people are physically
invisible. Some records have all labeled joints marked 1; the source-specific
quality/provenance of those visibility flags remains unresolved.

A cached frozen DA3-BASE teacher produces usable candidate orders, with measured
sensitivity near depth ties. This supports a bounded target-design experiment,
not a claim of accurate depth, improved poses, or successful diffusion training.

## Exact target

For each original image I, infer one shared full-image depth map D. For person i:

    V_i = {j: visibility_ij == 2, joint coordinate finite and inside image}
    d_i = sum(D(u_ij) for j in V_i) / len(V_i)

Use an unweighted arithmetic mean. Occluded (flag 1) and unlabeled (flag 0)
joints never contribute. Do not substitute a median, confidence-weighted average,
box average or occluded-joint estimate. All sampled visible coordinates in the
current TRAIN/VAL audit were finite and in bounds.

For DA3 depth, smaller values denote nearer surfaces; sort ascending. Define
an explicit secondary rule for exact ties (e.g. x-center, then annotation ID).
The diagnostic uses annotation ID for exact ties only. A tie-breaker cannot resolve
near-tie teacher inconsistency. No cross-image depth scale is required.

Raw relative inverse-depth estimators require their own documented convention:
averaging inverse depths and sorting descending is not in general equivalent to
averaging depths and sorting ascending. Do not infer the averaging contract merely
from a color visualization. DA3-BASE exposes relative depth, not metric meters.
See the [official model table](https://github.com/ByteDance-Seed/Depth-Anything-3#%EF%B8%8F-model-cards)
and [API](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/docs/API.md).

When V_i is empty, d_i and its rank are undefined. Zero depth, mean over flag>0,
mask-derived arbitrary depth, or treating this real person as background would
change the requested target. Unknown people require a separately reviewed policy.

## Full annotation coverage

Person filter matches the dataset's raw labeled-person filter: iscrowd==0 and
num_keypoints>0. All filtered boxes have positive width and height. This audit uses
TRAIN and VAL only; TEST annotations/images are not accessed.

| Measurement | TRAIN | VAL |
|---|---:|---:|
| Images | 10,000 | 2,000 |
| Valid labeled people | 35,659 | 7,074 |
| People with >=1 visible joint | 20,942 | 4,182 |
| People without visible joints | 14,717 (41.27%) | 2,892 (40.88%) |
| Images with every valid person rankable | 3,943 (39.43%) | 827 (41.35%) |
| Images with no rankable person | 5,551 | 1,090 |
| Images with mixed defined/undefined people | 506 | 83 |
| Out-of-bounds/nonfinite flag-2 coordinates | 0 | 0 |

Head/neck entries (joints 12/13) never have flag 2 in either file, so they contribute
nothing under the requested mask. This follows the files; no special head/neck
exception has been inserted. Dataset visibility helper:
`src/data/transforms.py:48`; raw person filtering: `src/data/datasets.py:90`.

## Frozen teacher diagnostic

- Teacher: cached DA3-BASE, revision f4a6c9b3c95e41c82048423d3493a81ec3fa810e.
- Weight SHA256: e01067dc1659613083d9145a9a2547ccdbe6ccbbf83c4fe7b3e8a4e2bdae78b5.
- Clean local teacher source commit: 3d835ec1a5802d64a8b8b15f817a1ab54809bfe4.
- 32 VAL image IDs sampled without replacement with NumPy default_rng(42), sorted
  for processing. Independent single-image inference; unrelated scenes are never
  passed as multi-view inputs. This is not the historical fixed32 TEST manifest.
- 89 valid people, 45 defined means and 44 undefined; 14 fully rankable images,
  13 images with at least two defined people, 75 comparable unordered person pairs.
- Primary longest-side process resolution 504; compare independent flipped input
  at 504 and original image at 420. Native upper_bound_resize, no crop.
- Native inference uses CUDA autocast (BF16 on this host), frozen eval/no-grad.
- Exact half-pixel bilinear sampling maps original pixels to the output grid.
  Independent torch.grid_sample cross-check passes. Image dimensions checked.
- Synthetic checks pass: arbitrarily large flag-1/flag-0 samples excluded,
  all-invisible mean undefined, sort direction, independent bilinear interpolation.

| Perturbation/check | Changed comparisons | Total | Rate |
|---|---:|---:|---:|
| Re-infer horizontally flipped RGB, corresponding mirrored joints | 11 | 75 pairs | 14.67% |
| Re-infer at resolution 420 instead of 504 | 2 | 75 pairs | 2.67% |
| Remove one visible joint from one person's mean | 15 | 1,059 leave-one-out comparisons | 1.42% |

For a descriptive near-pair criterion abs(d_a-d_b)/max(abs(d_a),abs(d_b))<.05,
29 pairs are near; 9/11 flip reversals and both resize reversals fall there.
Among 46 other pairs, 2 flip reversals and no resize reversals occur. This .05
threshold is an arbitrary diagnostic in this teacher's relative-depth units,
not calibrated uncertainty. Pair/leave-one-out comparisons share images/people
and are not independent trials. Single-visible-joint people cannot enter the
leave-one-out check. Five of the 13 multi-person defined scenes have any flip
reversal; two have any resize reversal.

Caching labels on original images and carrying them through flips/resizes would
avoid re-estimating contradictory labels each epoch. These perturbation results
measure teacher consistency, not actual training-label drift with such a cache.
No human depth-order ground truth is available in this diagnostic; stable labels
may still be wrong. No second teacher or checkpoint comparison was performed.

Reproducible evidence:
[manifest](assets/pose_depth_order/manifest.json),
[per-person means/pairs/summary](assets/pose_depth_order/results.json),
[temporary diagnostic source](assets/pose_depth_order/investigate.py).
Native depth arrays, log and initial overlays remain in `tmp/depth_order/`.
The harness expects the existing local teacher source/dependencies/cache via
PYTHONPATH=tmp/depth-anything-3/src:tmp/da3-deps and HF_HUB_OFFLINE=1.
Its output paths are the local tmp directory; it is not a production entrypoint.

## How this fits the current detector

1. `model_v44.py:prepare_context` selects 50 proposal features/references by
   classification score, not scene depth. `training_v44.py:_losses_for_batch`
   Hungarian-matches these boxes to GT. `latent_v43.py:assign_latent_sources`
   constructs clean codes in those matched proposal slots before corruption.
   Sorting GT rows alone leaves these semantics unchanged: matching still
   determines which target belongs to which proposal.
2. A canonical target would instead put complete pose/box/code and presence in
   rank slots 0..N-1, with absence thereafter. Slots need rank-specific identity;
   an equivariant network with identical queries and iid initial noise cannot
   distinguish an externally prescribed rank. The full scene may still use
   cross-person attention. Preserve the clean/noisy slot correspondence.
3. Proposal features and deformable-attention reference boxes must agree with
   this contract. Assigning the nearest person's target to the highest-score
   proposal creates a new association task, potentially at a distant reference.
   The existing cached confidence logits also describe proposal objectness,
   not necessarily existence of the kth-nearest person. Those assumptions need
   an explicit redesign, not just a target permutation. Keeping the existing
   proposal generator internally also retains its Hungarian detector losses;
   distinguish matching-free latent supervision from an entirely matching-free model.
4. Mean-depth ranking does not require a scalar depth output, depth loss,
   autoregression or a depth teacher at inference. It serializes the pose set;
   rank is not localization confidence and should not replace the AP score.
5. Cropping needs a fixed semantic choice. Native transforms clamp out-of-crop
   keypoints while retaining their visibility if the person's box survives
   (`src/data/data_module.py:240-262`). Sampling a depth map at those clamped
   coordinates would not be the requested original visible-joint measurement.
   A raw-image cache can preserve original mean-depth identity, subset surviving
   people and compact their ranks. Recomputing only in-crop visible joints instead
   defines a different, crop-dependent mean. Original annotation identity must
   be carried through the dataset/batch; currently image ID and per-image row
   order are available, but original annotation IDs are not returned.

## Literature boundary

[HMOR, ECCV 2020, section 3.2](https://arxiv.org/html/2008.00206v2#S3.SS2)
uses a person's arithmetic mean 3D joint position and pairwise instance depth
relations (equation 5). This is close precedent for mean-joint person ordering.
Its loss supervises relative depth of predicted 3D people, rather than assigning
canonical diffusion slots from a frozen RGB teacher's visible-only surface depths.

[Pavlakos et al., CVPR 2018](https://arxiv.org/html/1805.04095) use ordinal depths
of joints as weak supervision for 3D pose. This is precedent for ordinal geometric
supervision, with a different unit of ordering and different targets. Neither
paper establishes the proposed CrowdPose canonical-slot design. This bounded
lookup does not establish novelty or an exact absence of precedent.

## Proposed next iteration, not implemented

Keep the user's exact visible-only arithmetic mean. First resolve the annotation
coverage policy. The simplest controlled initial test would use only the 3,943
TRAIN / 827 VAL images where every valid person has a defined mean, cache teacher
labels once on raw images, and test a dedicated rank-slot design against a
left-to-right ordering on that SAME subset with identical training/evaluation
budgets. This tests whether the semantic ordering helps beyond deterministic
serialization. Subset selection is a distribution change and cannot stand in for
full CrowdPose results. No particular architecture or training budget is approved.

For a first implementation iteration, create and audit only a cache on that
complete-ranked subset, preserving mean, visible sample count, original annotation
ID, teacher revision and tie metadata. Cache coverage and sanity checks precede
any detector changes. Whole-dataset integration must explicitly retain unrankable
real people; dropping them or labeling them background is not an acceptable
silent fallback. A different visibility source or partial-order objective would
require its own reviewed scope.

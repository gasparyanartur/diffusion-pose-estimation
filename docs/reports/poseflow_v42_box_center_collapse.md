# Repaired V42 box-center Gaussian AP deterioration

Read-only audit, 2026-11-04. This is the historical t0-only repaired box-center
run, not the later corrected uniform-t run. No trainer, model, optimizer,
checkpoint or dashboard configuration was changed. Frozen native source and
strictly loaded checkpoints were used for inference only.

## Observed trajectory

The fixed32 K1 pose AP peaks at33.815 (epoch9) and reaches4.385 at epoch63.
K4/K8 peak at37.040/37.797 (epoch22), reaching17.159/23.339 at epoch63.
At epoch63 box AP remains37.738/46.203/48.058. Validation total loss falls
15.697 at epoch9 ->7.542 at epoch63; keypoint loss .4965 ->.2445.
The decline begins before the first LR reduction. This is declining Gaussian
pose sampling despite improving t0-conditioned training/validation objectives.

## Frozen counterfactuals

Same32 held-out test images/seed42/resize256427/native pose NMS. Values below
are pose AP on the 0..100 scale. GT-derived input conditions are recovery
diagnostics, not deployable Gaussian inference AP.

| Input and model timestep | Epoch1 / step3000 | Epoch63 / step96000 |
|---|---:|---:|
| Gaussian N(0,1), t999, native K1 |20.094|4.385|
| Same Gaussian input, t0 |19.369|4.336|
| Native GT-derived t0-corrupted bank, t0 |17.148|54.718|
| Zero coordinate bank, t0 |19.368|2.147|
| Remove entire pose-encoder output, t999 |19.306|1.369|

All saved K1 metrics are reproduced exactly at both checkpoints, including box
AP. At epoch63, varying only time for the same Gaussian input gives4.036 at
124,3.935 at249 and3.960 at499. Therefore the time embedding value alone does
not explain the failure. Actual native K8 ends at124 and achieves23.339,
whereas one Gaussian-input forward at124 achieves4.036: the evolved coordinate
state accounts for the improvement across calls.

## Content versus scale controls at epoch63

The native t0 source builder preserves nearly clean GT coordinate sources and
random padding scaled by6 before clamping; inference begins with N(0,1).
Randomizing only GT-derived input slots while preserving all original padding
and keeping t0 changes54.718 ->17.951 AP. Erasing those slots gives7.814.
This establishes useful GT-pose-content dependence in the late checkpoint,
beyond the background distribution or feature bias alone.

A fully random N(0,6) bank clamped to[-6,6] gives18.774 at t0 and17.223 at999.
Its partial recovery versus N(0,1)4.385 also identifies a coordinate-distribution
scale effect. These are alternative-input diagnostics, not changes to the
native sampler or a justified repair of its diffusion equations.

## Interpretation and limits

The old t0-only model increasingly learns recovery conditioned on training-style
pose inputs; those inputs also appear in validation loss. Gaussian inference
starts from a distribution it was never trained to denoise. Thus the objective
can improve while Gaussian AP deteriorates. Input content and scale controls
support this explanation; the timestep embedding alone is ruled out as the
primary route in these controls.

The late repaired checkpoint clearly uses informative pose content despite
output-Hungarian rematching and independently ordered proposal/source banks.
Earlier weak-state-use measurements concerned the younger corrected uniform-t
checkpoint; they must not be generalized across these different runs and ages.
This result also means proposal/source mismatch alone is not sufficient to
explain all V42 state behavior. The paired-proposal experiment remains a test
of correspondence, not an established cure for this collapse.

The corrected uniform-t and new paired-proposal recipes train t0..999 and
therefore expose the model to near-Gaussian states as well as near-clean ones.
That removes this specific t0-only training-distribution omission; it does not
establish their late-training AP outcome. The latest corrected-uniform fixed32
result inspected here is epoch20, K1/K4/K8=35.906/35.613/35.007; it is younger
than the collapsed control, not a matched-age proof.

## Provenance

Run: /storage/outputs/pose-graph-diff/results/poseflow_v42/box_center_pilot_20261103_213148
Frozen source: run/workspace/src; native target builder explicitly sets every
training time to zero. Native full80 training/control jobs remain intact.

Epoch63 checkpoint SHA256:
`abed4c085845c3ad8bd6396a7e09834b58f3ba5f126439f3e4faa3623ac497a4`.
A local immutable diagnostic copy is retained. Epoch1 uses the preserved original
checkpoint; both source paths, hashes, image IDs and per-condition predictions
are saved in `tmp/v42/repair/collapse_audit/`:
`report.json`, `early_epoch001/report.json`, `scale_controls/report.json`,
`trajectory.json`, `comparison.json`, scalar exports and reproducible scripts.
No optimizer updates or new training experiments were performed.

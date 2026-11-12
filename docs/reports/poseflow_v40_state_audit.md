# V40 noisy-state and proposal dependence audit

Verified 2026-11-04T19:31+01:00. Base V40 final checkpoint epoch079/120000, original fixed32
CrowdPose subset, seed42+image_id, resize256/427, FP32/TF32off, all50 queries,
no NMS, native bbox evaluator. These are **box AP**, not pose AP. Saved K1/K4
metrics reproduce exactly. The loaded denoiser, clean encoder, readout and
sampler methods match the frozen original V40 methods. No optimization or live
training changes were made.

## Inference counterfactuals

| Condition | Fixed32 box AP |
|---|---:|
| Native Gaussian K1 |43.2876|
| Different Gaussian seed, K1 |42.9505|
| Zero latent input, K1 |43.5072|
| Zero proposal-reference conditioning, K1 |0.2316|
| Same Gaussian latent, timestep0 |39.1816|
| Native Gaussian K4 |20.6213|
| Zero denoiser input at every K4 call |42.8150|
| K4 box geometry with K1 scores |43.4169|
| K1 box geometry with K4 scores |20.3638|

K1 versus K4 uses the same image, initial Gaussian seed, cached proposal-reference
slots and output query order. The hybrid checks swap only geometry or scores
between these paired outputs; no rematching, NMS or oracle ranking is used.
They isolate the fixed32 AP deterioration to the changed confidence scores,
with little AP penalty from K4 geometry when K1 scores are retained.

Average box-center displacement K1->K4 is1.14 resized-image pixels across all
slots. Average absolute probability change is.621. The number of queries with
score>=.5 rises2.66 ->22.66 ->44.00 ->49.75 out of50 along times999/749/499/249;
the subset contains133 people, or4.16/image. Mean person score rises.125->.746.
Zero-input-at-every-call keeps accepted queries near2.6–3.0 and recovers42.82 AP.
Generated-state feedback is therefore causally implicated in the measured
confidence degradation. This does not isolate the particular training route
or representation property responsible for the bad scores.

The zero-reference test removes the complete proposal feature+box condition
inside each denoiser block while retaining global image attention. It is an
out-of-distribution intervention. The zero-state test also changes the latent
input distribution; it is a diagnostic, not a recommended inference policy.

## Does an informative matched latent help?

The training-style conditional test first uses V40's GT-matched reference
selection, then constructs the matched clean latent bank, applies native
q_sample at the specified time and reuses identical references for the zero-state
comparison. All50 predictions are evaluated; matched errors cover133 positives.
This uses GT information and **does not represent inference AP**.

| Conditional input | Box AP |Matched box-coordinate MAE|Mean positive score|Mean negative score|
|---|---:|---:|---:|---:|
| Matched clean-bank corruption, t0 |69.89|.00500|.904|.000112|
| Same references, zero latent, t0 |41.16|.01791|.529|.05866|
| Matched clean-bank corruption, t500 |50.32|.01597|.906|.000186|
| Same references, zero latent, t500 |44.35|.01678|.578|.07100|

V40 uses informative aligned low-noise states, unlike the weak low-noise
coordinate dependence measured in V42. At Gaussian start its predictions are
mostly reference-driven. On the generated multi-call trajectory, state feedback
strongly increases background confidence and destroys ranking, while box
geometry changes little. Alignment and useful reverse-trajectory behavior are
separate properties.

The completed native full8000 result is K1/K4 bboxAP47.9955/25.8320. Ablations
here are fixed32 only, on one checkpoint/training seed. They do not establish
the corresponding full-test intervention metrics or the benefit of a retrained
alternative. EMA/SIGReg variants were not evaluated in this audit.

Artifacts: `tmp/v40_state_audit/report.json`, prediction/hybrid JSON files,
`frozen_methods.json`, `run.py`, and `run.log`. Checkpoint SHA256 is recorded in
the report. The source checkpoint and live V42 jobs were unchanged.

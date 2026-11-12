# Paired V42 multicall confidence audit

Frozen paired-proposal epoch003/6000 optimizer updates, fixed32 held-out images,
seed42+image_id, resize256427, FP32/TF32 off, native pose NMS/evaluator. This is
an early checkpoint and subset diagnostic, not a final/full-test result.

## Matched-age result

| Epoch3 /6000 updates | K1 pose AP | K4 pose AP | K8 pose AP |
|---|---:|---:|---:|
| Paired proposal |23.442|9.820|9.108|
| Corrected uniform-t control |26.484|23.393|23.878|

Early K1 improved at epochs0..2, but the advantage is not consistent: by epoch3
control K1 is higher. The large multicall penalty is clear at every observed
epoch. Latest paired epoch4 subsequently records22.92/7.18/6.14. Uniform training
is live and correct: all1000 times observed; the source/label audit has zero
mismatches. This is a different failure from the historical t0-only repair.

## Feedback and confidence controls

All saved native K1/K4/K8 metrics reproduce exactly. For each image we compare
the native final call with a second call at the SAME final timestep, image,
selected reference boxes and selection logits, replacing only its incoming state
with the ORIGINAL Gaussian input. References and selection logits are bitwise
identical between these paired final calls; no GT or query rematching is used.
Geometry means both normalized boxes and joints. Confidence means class logits.

| Frozen condition | Pose AP | Box AP |
|---|---:|---:|
| Native K1 |23.442|33.175|
| Native K4 |9.820|18.213|
| K4 finalt249 with original Gaussian input |21.499|33.353|
| K4 carried geometry + Gaussian-input scores |20.892|32.668|
| Gaussian-input geometry + K4 carried scores |9.659|18.024|
| Native K8 |9.108|16.968|
| K8 finalt124 with original Gaussian input |22.151|33.444|
| K8 carried geometry + Gaussian-input scores |22.086|33.022|
| Gaussian-input geometry + K8 carried scores |9.316|17.331|

The feedback-induced drop in AP is dominated by confidence ranking. The carried
geometry can support20.9/22.1 pose AP when paired with the controlled scores;
changing geometry alone retains the bad9.7/9.3 AP. This does not mean geometry
is identical: carried-vs-Gaussian final-call joint MAE is7.46/6.94 resized pixels
averaged over all50 slots, while box MAE is.00243/.00251. Native pose AR changes
33.01(K1)->30.00(K4)/29.85(K8), much less than AP.

Mean person probability rises.137 ->.161 ->.187 ->.204 over K4; queries above.5
rise1.31 ->1.94 ->3.28 ->4.75 per image. K8 ends at.208/5.03. These aggregate
counts are descriptive; without per-person labels they do not establish which
new high-confidence queries are false positives. Hybrid AP identifies the
ranking failure directly.

## Proposal order and state alignment

Selected image references are rebuilt at every time and have substantial
spatial reordering. A Hungarian L1 spatial match of t999 references to t249
references retains only8.0% of indices; same-index center displacement averages
53.58 resized pixels, versus7.61 after spatial matching. This is a spatial
reference comparison, not proof of exact person identity on duplicate boxes.

A separate inference-only counterfactual matches each previous selected box bank
to the next selected bank and permutes the complete carried coordinate state
accordingly. The permuted state is used for BOTH the model and DDIM residual/
update, preserving numerical consistency. All coordinate values are retained,
no GT is used. Custom no-remapping K4 reproduces native metrics exactly.

Reference-remapped K4 gives7.143 poseAP/13.182 boxAP, versus9.820/18.213 native.
Mean final confidence increases further to.233 with5.75 queries above.5. Thus
this explicit spatial remapping does not rescue inference. Reference changes
are real but cannot be assigned sole causality from this diagnostic; fixed
references or another matching rule is a different intervention.

## Interpretation

Pairing ensures that during a training forward the proposal, corrupted GT pose
source and task label refer to one person. It does not establish that feeding
model-generated outputs back will retain calibrated confidence. The model emits
coordinates for every slot, including background slots with no geometry loss;
the sampler feeds all of them back. Training inputs are forward-corrupted GT
poses or random background padding; generated trajectories contain the model's
own errors and generated background poses. A classifier learning to trust
pose-like state can therefore encounter misleading evidence on feedback.

This input-distribution/foreground-confidence mechanism is a hypothesis about
WHY the confidence drift was learned, not yet an isolated training cause.
The measured result is narrower and strong: generated-state feedback degrades
confidence ranking, and this accounts for most of the multicall AP loss at this
checkpoint. The predecessor V40 audit found a related score-dominated trajectory
failure; paired training correspondence had not solved inference calibration
there either. No AP benefit or failure outcome is claimed for a retrained fix.

## Provenance and scope

Checkpoint SHA256:
`10ab704cbfd137b56bfa4914369652d0f74363cb13546b8af5438781fae8e357`.
Run: /storage/outputs/pose-graph-diff/results/poseflow_v42_paired_proposal/20261104_203731
Frozen source under that run's workspace; strict restore, no optimizer updates.
Local checkpoint copy, image IDs, predictions, trace summaries and scripts:
`tmp/v42_paired_proposal/multistep_audit/` (`report.json`, `traces.json`,
`same_age.json`, `reference_remap.json`, `evaluate.py`, `remap.py`).
All trainers, production code and TensorBoard configuration remain unchanged.

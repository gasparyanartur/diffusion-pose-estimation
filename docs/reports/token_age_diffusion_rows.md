# Token-age diffusion forcing: row construction

Status: the approved first implementation iteration is complete. This is an
isolated training-input builder, not a model, sampler, training run or AP result.
Historical V46–V48 implementations are unchanged.

## Interface

`src/models/pipelines/pose_estimators/poseflow/rows_token_age.py` exposes:

```python
build_training_rows(
    clean_coordinates, counts,
    coordinate_mask=None, target_indices=None,
    row_count=50, generator=None,
)
```

Coordinates have shape `[B,C,32]`: 14 joint XY pairs and box CXCYWH. Counts are
int64 `[B]`. Missing input joints must already be filled, as in V47; their
boolean `[B,C,32]` supervision mask is preserved. Coordinate units and scaling
are caller-owned. The builder does not encode people or normalize coordinates.

Each call draws one permutation per image's valid people. The same permutation
applies to coordinates, coordinate masks and target identities and is reused in
every row. `sources` returns the shuffled clean targets, masks, original local
`source_indices`, optional external `target_indices`, and counts. Geometry can
be gathered through those identities; it is not rematched. Invalid source slots
are zero with false masks and identity -1. Inputs are not modified.

Noisy inputs have shape `[B,50,50,32]`. The output also contains per-token
`noise_sigmas`, `token_types`, `padding_mask`, `predict_indices`, `eos_targets`,
`birth_mask`, `refinement_mask`, `supervised`, and the sampled `noise` for
verification. Clean sources and loss masks are target metadata, not additional
decoder inputs. No future clean target is copied into an earlier row.

## Rows and corruption

For zero-indexed row `r`, the existing prefix has `min(r,P)` people. Its next
slot is always a PREDICT query, whether the target is a person or EOS. This query
contains pure Gaussian noise in both cases. It must use a noisy-coordinate input
path in the future decoder, rather than V46's embedding-only query behavior.
All subsequent slots are padding with zero input, noise and sigma.

Person rank `i` has age `r-i`. The Gaussian coefficient is:

- Birth, age 0: `sigma=1`.
- Age 1: `sigma=.1`.
- Age 2: `sigma=.05`.
- Later ages: `sigma=.1 * 2**(-(age-1))`.
- Final row: existing people have `sigma=0`; the EOS query remains pure noise.

Corruption uses `sqrt(1-sigma**2)*clean + sigma*epsilon`. Thus `.1` is a noise
coefficient, corresponding to `.01` noise variance, not a timestep percentage.
Gaussian draws are independent across rows, people and channels. Supplying a
generator controls both permutation and noise without consuming global RNG.

All active people have supervision masks, including the newly queried person
on birth rows. EOS and padding have no coordinate/geometry targets. After the
first EOS target row, subsequent rows keep the same people and continue to
provide refinement and EOS supervision. Empty images have EOS on all 50 rows.
Actual losses, both-layer reductions and per-person normalization belong to the
next iteration; this builder does not compute loss weights.

The final row is reserved for EOS/refinement, so 50 total rows support at most
49 births. Counts beyond either supplied storage or this budget raise before
RNG is consumed. No annotations are truncated. The TRAIN-only annotation audit
found a maximum of 15 annotations per image, leaving 35 rows after the last
possible introduction in that raw audit. No TEST annotations were accessed.

## Verification and remaining work

The test was first run with the new module absent, producing the expected
import failure. Focused tests cover permutation/identity/mask alignment, the
corruption equation, age-dependent noise, EOS query isolation, independent draws,
future/padding exclusion, final cleanup, counts 0/1/49 and mixed batches, seeded
repeatability, input preservation and rejection before RNG consumption.
All 16 focused checks pass, along with 19 existing V46/V47/V48 row regressions
(35 total). Evidence logs are under repository-local `tmp/token_age_diffusion/`.

The final clean-GT training row is an easy reconstruction endpoint. Its error
must not be presented as evidence that a generated prefix improves. Later
measurement work must separate teacher-forced reconstruction, generated birth
quality, generated refinement, duplicates/coverage/count errors, native AP and
compute cost.

Next work needs its own reviewed iteration: coordinate codec/decoder with
per-token noise conditioning, all-active losses and normalization; then DDIM
transitions, EOS/capacity tails, deterministic evaluation, strict restore and
native resource verification. Pure-noise DDIM endpoints require an explicit
finite transition contract. Age-dependent box/pose emphasis remains deferred.
No training entrypoint, checkpoint namespace or run has been created here.

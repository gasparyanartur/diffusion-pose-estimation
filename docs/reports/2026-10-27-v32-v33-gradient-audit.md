# V32/V33 gradient audit

The V33 readout dominates global gradient clipping in the measured batches.
Specifically, its final shared joint projection dominates; the expanded-token
transformer parameters themselves contribute less than 1% of the squared norm.
This is evidence of a gradient balance problem worth investigating, not proof
that it causes V33's four-call AP deficit.

## Scope and controls

Offline diagnostic requested by the user, with no optimizer updates, training
interruption, or production code changes. Native training losses, frozen pinned
DINO teacher, training mode, FP32/TF32 disabled, batch4/accumulation2. Four replayed
training microbatches give two optimizer-sized accumulated gradients. Seed7821;
image SHA256 and pre-draw proposal/noise RNG state match between both models.
Native weighted terms are .5clean task, .5recovered task, latent MSE,
.5clean DINO and .5recovered DINO. Gradients are measured before clipping.

V32 retained only epoch079 (120000 updates); V33's epoch044 rotated away during
inspection, so epoch045 (69000 updates) was copied locally before loading.
**This comparison is not matched-age.** Same-age historical scalar comparisons
remain separate. Both pipeline states restore strictly. Component-gradient sums
were checked against a direct backward of the native total on the first batch of
each model; objective sums, finite norms and identical replay inputs also pass.

Machine data, checkpoint hashes and executable diagnostic:
`tmp/v33_gradient_audit/results.json`, `audit.py`, `console.log`.
The original V32/V33 run paths and checkpoint identities are in the JSON.

## Accumulated gradients

Shares below are squared L2 norm shares, which add to 100% across disjoint modules.

| Quantity | V32 window1 / window2 | V33 window1 / window2 |
|---|---:|---:|
| Global norm |46.43 /47.29|85.01 /134.92|
| Task-readout share |76.8% /67.8%|90.0% /90.6%|
| Denoiser norm before clipping |16.30 /13.75|13.25 /16.18|
| Global0.1 clipping multiplier |.002154 /.002115|.001176 /.000741|
| Denoiser norm after that multiplier |.0351 /.0291|.0156 /.0120|
| Multiplier if readout excluded from global norm |.004474 /.003724|.003727 /.002411|

Excluding the readout from the clipping norm would allow V33's other gradients
to be about3.17–3.25 times larger for these fixed backward passes. This is only an
algebraic counterfactual; no clipping rule or optimizer was changed. V32's
corresponding factor is1.76–2.08.

V33's final shared Linear256->2 joint head alone contributes84.3% /89.0% of the
global squared norm. Its token transformer, including expansion and role
embeddings, contributes0.84% /0.82%. Box-head norms are18.77 /11.16; joint-head
norms78.06 /127.32; token-transformer norms7.77 /12.19. Therefore, attributing the
dominance simply to the number of new transformer parameters would be wrong.
The shared joint head receives contributions from all14 joint roles. Whether
gradient alignment across roles or the features' scale explains its large norm
has not been isolated.

## Loss routes inside the denoiser

These norms already include the actual objective coefficients.

| Denoiser gradient source | V32 window1 / window2 | V33 window1 / window2 |
|---|---:|---:|
| Recovered task |16.252 /13.718|13.171 /16.101|
| Latent MSE |.207 /.201|.281 /.271|
| Recovered DINO |.219 /.288|.190 /.235|
| Task/MSE norm ratio |78.4 /68.3|46.9 /59.4|
| Task/MSE gradient cosine |.188 /.096|.275 /.224|

Clean task and clean DINO contribute exactly zero to the denoiser parameter
group, as intended. Recovered task and latent MSE do not reach the clean encoder
through h0. Task and MSE gradients are weakly positively aligned in these
samples; there is no measured overall destructive task/MSE alignment.

Task supervision dominates latent reconstruction in both models. However, the
task/MSE norm ratio is lower in V33 here, so the audit does not support the claim
that V33 makes that particular imbalance worse. DINO's parameter gradient into
the denoiser is also small relative to task supervision despite its visible loss.

## Historical first-update evidence and limits

The runs' saved first-update module norms also show increased readout share:
42.7% V32 versus62.2% V33 of the recorded squared norm. Those callback records
omit shared time/level embeddings, so these are not exact global shares. Source
values are preserved in `tmp/v33_gradient_audit/first_update_evidence.json`.
They provide evidence from the start of training rather than only differently
aged checkpoints; no assumption of identical first-update batches is required
for interpreting each run's internal share.

Global clipping scales every parameter gradient by the same multiplier: it
does not change task/MSE balance within one backward pass. AdamW adapts using
gradient moments, so smaller clipped gradient norms cannot be translated into
proportionally smaller parameter updates. A stable common gradient scale can
largely cancel in that adaptation; changing batch-dependent scales can still
change moment weighting. Historical optimizer behavior was not replayed.

Conclusion: the proposed global-clipping mechanism is present locally, chiefly
through the joint output head. It remains a candidate explanation for poorer
iterative inference, rather than an established cause. A controlled clipping
intervention would be a separate user-reviewed experiment. No such experiment
was launched by this audit.

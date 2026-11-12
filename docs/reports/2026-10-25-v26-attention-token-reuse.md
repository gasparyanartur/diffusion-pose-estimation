# V26 latent-only: earlier-to-later attention token reuse

Checkpoint epoch046, fixed32 test subset, FP32/TF32off, frozen source and identical seed42+imageID Gaussian starts as the prior attention audit. No actual masks were applied. No model/config/training changes.

Definitions: for each query, sort valid image tokens by attention probability and retain the smallest prefix carrying90% or99% probability, including the threshold-crossing token. Padding is excluded. Union retained sets from earlier layers. Head-mean masks use averaged-head weights; head-aware masks retain the union of each head's retained set. Per-level control normalizes each feature level separately before selecting and unioning. Later mass outside these sets is descriptive probability mass that a proposed mask would remove before renormalization, not a measured masked-model AP loss.

Tables summarize equally weighted32image means, ten slots per image chosen by final-call existence score. Same proposal slots compared across layers/calls; no matching. All50 results also saved.

## Final K4 call: union of layers1+2 versus layer3

| Earlier selection rule | Valid tokens retained | Layer3 attention mass excluded | Layer3 salient-token count outside earlier selection |
|---|---:|---:|---:|
| head_mean_union, 90% | 4.71% | 55.32% | 91.7 of 133.0 (67.95%) |
| per_head_union, 90% | 12.61% | 22.74% | 45.7 of 133.0 (32.70%) |
| per_level_union, 90% | 53.74% | 19.54% | 44.5 of 133.0 (32.27%) |
| head_mean_union, 99% | 22.40% | 7.10% | 108.2 of 376.1 (27.84%) |
| per_head_union, 99% | 48.86% | 1.57% | 24.9 of 376.1 (6.25%) |
| per_level_union, 99% | 81.93% | 2.45% | 26.1 of 376.1 (6.57%) |

The late salient set uses the same90/99% coverage threshold as the earlier mask. Counts and percentages are separate averaged per-query statistics, so a ratio of average counts need not equal the averaged percentage.

## Findings

- Late-salient tokens are not a subset of earlier-salient tokens. Using head-mean90% sets from layers1+2,67.95% of layer3's90%-coverage tokens are new (~91.7 of133.0 tokens/query); those excluded tokens carry55.32% of all layer3 mass.
- Allowing any head to preserve a token reduces that to32.70% new late-salient tokens (~45.7 of133.0), but still excludes22.74% of layer3 attention.
- The conservative99%-per-head union preserves48.86% of valid tokens and excludes1.57% of layer3 mass. This could motivate an offline masking ablation, but it is not a guarantee that predictions/AP remain stable.
- At K16 final call, the conservative99%-per-head union preserves44.14% of valid tokens and excludes2.81% of layer3 mass. At90% it excludes26.22%.
- Within-scale control: among the top10% layer3 tokens by probability within stride16,5.38% were in the bottom50% in BOTH earlier layers. At stride32 this is13.09%, stride8 0.84%, stride64 0%. Thus novelty is not entirely caused by shifting total mass across scales.
- Between K4 calls, layer3 final attention places24.50% mass outside the first-call layer3 head-mean90% set. With99%-per-head union,3.28% remains outside, while71.62% of tokens are retained. A static first-call mask is not lossless either.

## Masking implications

Aggressive hard masking based on earlier attention alone is not supported as a lossless optimization for this checkpoint. A permissive union across ALL heads and BOTH earlier layers at99% is the most promising tested rule, based solely on retained probability. It must be tested by rerunning the masked model and comparing predictions/AP before accepting it; masks change softmax normalization and hidden states across subsequent calls.

Masks are per-proposal. Retaining~49% per query does not imply half the memory tokens can be discarded globally: the union across50 proposals can be larger. A dense attention implementation with a Boolean mask may not save compute; sparse execution/token gathering needs a separate implementation and benchmark.

Verification: 384 native weighted-value/output reconstruction checks; max absolute error 0.00000347 within the prior audit tolerances. Selection includes boundary token and excludes zero-weight padding. Attention probability is not causal value contribution. One checkpoint, one fixed32 monitoring subset and one seed; no generalization/AP claim.

Raw evidence: `/home/gasparyanartur/dev/pose-graph-diff/tmp/v26_attention_reuse/metrics.json`, `probe.py`, `console.log`.

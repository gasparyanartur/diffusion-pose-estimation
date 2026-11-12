# V26 latent-only: sampling-call sweep

Checkpoint: epoch046 (47 completed epochs), SHA256 `8763930455e9473b198a9bcb945d6b81f67f7fd39317f8efa451644c3d1c5c47`.

Fixed32 held-out test monitoring images, 133 GT people. Gaussian-start eta0 DDIM; identical seed42+imageID across call counts; FP32, TF32 disabled. Frozen training source, strict restore, native K4 bitwise parity. Corrected existing pose NMS; no renewal or ensemble. Image features cached once per image.

| Calls | Pose AP | Box AP | Raw coverage OKS >= .5 | Raw coverage OKS >= .75 |
|---:|---:|---:|---:|---:|
| 1 | 0.101 | 1.352 | 19/133 | 0/133 |
| 4 | 1.896 | 7.520 | 46/133 | 14/133 |
| 8 | 1.564 | 8.198 | 47/133 | 14/133 |
| 16 | 1.514 | 9.999 | 46/133 | 7/133 |
| 32 | 1.732 | 10.322 | 41/133 | 7/133 |
| 64 | 1.838 | 10.886 | 40/133 | 8/133 |
| 128 | 1.645 | 11.528 | 37/133 | 8/133 |

Raw coverage ignores existence scores and NMS: any of the 50 output poses can cover each GT person; multiple GT people may share a prediction in this diagnostic. It is not AP or one-to-one assignment.

Observation: box AP continues rising beyond four calls; pose AP does not. Four calls gives 1.896 pose AP versus 1.645 at128, while box AP rises7.520 to11.528. Across-slot latent variance grows0.03488 to0.12618, so extra calls create more spread without improved articulated-pose coverage. OKS .75 coverage falls14 to8 people. This does not support the claim that more calls will fix pose generation for this checkpoint. No causal attribution to a module or loss follows from this sweep.

Limits: one checkpoint, one seed, 32 monitoring images; descriptive subset diagnostic, not full-test AP or checkpoint selection. Training jobs unchanged.

Raw artifacts: `tmp/v26_step_sweep/metrics.json`, `pin.json`, `console.log`, and per-K predictions.

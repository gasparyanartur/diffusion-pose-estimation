# DiffusionRegPose selected-layer OTA comparison

This report compares the completed 256/427-pixel selected-layer OTA run with
the completed selected-layer Hungarian control. The runs use the same native
3-encoder/3-decoder model initialized from EDPose checkpoint layers 2, 4, and
6. The only resolved configuration difference is clean-query assignment:
`ota` versus `hungarian`.

## Artifacts

- OTA config: [`benchmark_diffusionregpose_development_compatible_256_keep_layers_ota.yaml`](../conf/benchmark_diffusionregpose_development_compatible_256_keep_layers_ota.yaml)
- OTA resolved config: [`config.yaml`](../results/diffusionregpose_development_compatible_256_keep_layers_ota/diffusionregpose_development_compatible_256_keep_layers_ota/20260912_224837/.hydra/config.yaml)
- OTA metrics: [`metrics.csv`](../results/diffusionregpose_development_compatible_256_keep_layers_ota/diffusionregpose_development_compatible_256_keep_layers_ota/20260912_224837/csv_logs/metrics.csv)
- OTA test results: [`test_results.json`](../results/diffusionregpose_development_compatible_256_keep_layers_ota/diffusionregpose_development_compatible_256_keep_layers_ota/20260912_224837/test_results.json)
- OTA test AP: [`metrics.json`](../results/diffusionregpose_development_compatible_256_keep_layers_ota/test_eval_256_v3/metrics.json) and [`predictions.json`](../results/diffusionregpose_development_compatible_256_keep_layers_ota/test_eval_256_v3/predictions.json)
- OTA checkpoint: [`epoch079.ckpt`](../results/diffusionregpose_development_compatible_256_keep_layers_ota/diffusionregpose_development_compatible_256_keep_layers_ota/20260912_224837/checkpoints/epoch079.ckpt)
- Hungarian resolved config: [`config.yaml`](../results/diffusionregpose_development_compatible_256_keep_layers/diffusionregpose_development_compatible_256_keep_layers/20260912_000637/.hydra/config.yaml)
- Hungarian metrics: [`metrics.csv`](../results/diffusionregpose_development_compatible_256_keep_layers/diffusionregpose_development_compatible_256_keep_layers/20260912_000637/csv_logs/metrics.csv)
- Hungarian test AP: [`metrics.json`](../results/diffusionregpose_development_compatible_256_keep_layers/test_eval_256_v3/metrics.json) and [`predictions.json`](../results/diffusionregpose_development_compatible_256_keep_layers/test_eval_256_v3/predictions.json)

## AP trajectory

AP is evaluated every ten completed epochs. The epoch numbers below are
one-based completed epochs; the CSV stores zero-based epoch indices.

| Completed epoch | OTA keypoint AP | Hungarian keypoint AP | Difference | OTA bbox AP | Hungarian bbox AP | Difference |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.341154 | 0.212985 | +0.128169 | 0.387421 | 0.325902 | +0.061520 |
| 20 | 0.462667 | 0.388829 | +0.073839 | 0.469302 | 0.439432 | +0.029870 |
| 30 | 0.515657 | 0.418571 | +0.097086 | 0.505586 | 0.460998 | +0.044588 |
| 40 | 0.641916 | 0.565507 | +0.076409 | 0.588820 | 0.553700 | +0.035120 |
| 50 | 0.667528 | 0.596590 | +0.070938 | 0.607194 | 0.573690 | +0.033504 |
| 60 | 0.689636 | 0.616428 | +0.073209 | 0.623890 | 0.588192 | +0.035698 |
| 70 | 0.700597 | 0.626144 | +0.074452 | 0.630129 | 0.593711 | +0.036418 |
| 80 | **0.703043** | 0.630153 | **+0.072890** | **0.632079** | 0.596394 | **+0.035685** |

OTA improves the final keypoint AP by 0.072890 and bbox AP by 0.035685. The
relative improvements are approximately 11.6% and 6.0%, respectively.

## Loss trajectory

| Completed epoch | OTA train loss | Hungarian train loss | OTA validation loss | Hungarian validation loss |
|---:|---:|---:|---:|---:|
| 1 | 42.1726 | 50.1479 | 28.0019 | 30.7284 |
| 10 | 22.8762 | 24.9720 | 16.5791 | 19.7023 |
| 20 | 19.4564 | 21.4713 | 13.8563 | 15.2259 |
| 40 | 13.6271 | 15.4076 | 9.5088 | 11.0362 |
| 60 | 12.5094 | 14.1783 | 8.4612 | 9.9503 |
| 80 | **12.1058** | 13.8019 | **8.0828** | 9.5740 |

The OTA loss is lower throughout this matched run, and the loss ordering agrees
with the AP ordering. These losses remain assignment-specific: OTA averages
over multiple clean positives, while Hungarian has one clean positive per
person.

## Test-split AP

The final epoch-79 checkpoints were evaluated independently on all 8,000
CrowdPose test images (35,283 annotations), using the same native evaluator,
256/427-pixel resize, one sampling step, and seed 42. The standalone evaluator
now preserves the selected-layer expansion-head topology while loading the
final Lightning checkpoint, and writes both keypoint and bbox records.

| Checkpoint | Keypoint AP | Bbox AP | Predictions |
|---|---:|---:|---:|
| OTA | **0.548950** | **0.518699** | 119,112 |
| Hungarian | 0.515739 | 0.508753 | 184,197 |
| OTA − Hungarian | **+0.033211** | **+0.009947** | — |

The test split confirms an OTA advantage for this seed and selected-layer
configuration, but the gain is smaller than on validation (0.703043 vs
0.630153 keypoint AP; 0.632079 vs 0.596394 bbox AP). Test AP is the correct
generalization check; the training runner's `test_after_fit` artifact remains
loss-only because its `test_step` does not accumulate AP.

## Assignment diagnostics

At the final validation epoch:

- OTA: 50.186 matched queries, 3.67699 positives/person on average, minimum
  2.564 and maximum 4.0 positives/person.
- Hungarian: 14.148 matched queries, exactly 1 positive/person.

The increased OTA cardinality is intentional and confirms that the run did not
silently fall back to Hungarian matching.

## Fairness audit

The resolved Hydra configs differ only in `pipeline.matching_strategy`,
`run_name`, and `out_dir`. They share architecture, selected checkpoint layers,
resolution, data splits/transforms, batch and accumulation, seed, EDPose
initialization, optimizer/schedule, precision, validation cadence, and
evaluator. Both AP metrics are produced by the same evaluator and reset after
each AP epoch.

The final checkpoints are different by design. Their SHA-256 hashes are:

```text
OTA:       97ee195be5f944f3cea424eb188a310277bc0bdee529e87be86fddbf483b5409
Hungarian: 6d53ee208f234debddbe7e26e24524f56345b90cb7ec9c42b4501ea0e2add31c
```

## Conclusion

For this selected-layer 256-pixel development model, pose-aware OTA is the
better observed training strategy: it achieves lower validation loss and higher
keypoint and bbox AP at every common evaluation point, and remains ahead on the
held-out test split (0.5490/0.5187 versus 0.5157/0.5088 keypoint/bbox AP). This
is evidence for OTA on this controlled experiment, not a paper-scale claim;
the model still uses reduced depth, reduced query groups, and 256/427
resolution.

## Full-scale follow-up

The OTA advantage is not confined to the validation trajectory: it also holds
on the untouched test split, with gains of +0.033211 keypoint AP and +0.009947
bbox AP. This is sufficient evidence to justify a full-scale experiment, but
it is not evidence that the gain will transfer unchanged to the full model.
The current result uses reduced encoder/decoder depth, reduced query groups,
256/427-pixel inputs, and one seed; scale changes can alter both optimization
and assignment behavior.

The next experiment should therefore pair full-scale OTA with a full-scale
Hungarian control, keeping initialization, architecture, data and transforms,
optimizer and schedule, seed, precision, sampling, and evaluator fixed. The
assignment strategy should remain the sole intended training difference. The
comparison should report validation and held-out test keypoint/bbox AP,
assignment cardinality, and loss components together. A second seed should be
added before treating any full-scale improvement as a general result.

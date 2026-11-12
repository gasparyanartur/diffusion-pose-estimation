# V26 latent-only: image-attention distribution

Pinned epoch046,47 completed epochs, same checkpoint/fixed32 images as the larger-call sweep. SHA256: `8763930455e9473b198a9bcb945d6b81f67f7fd39317f8efa451644c3d1c5c47`. Frozen training source, FP32/TF32off. No training/model changes.

## Method

Reconstructed all8 heads from the actual Q/K projections and padding mask in each of3 clean/denoiser image-attention modules. Reconstructed weighted-value/output projection was checked against native outputs480 times; maximum absolute discrepancy 3.170967102050781e-05 with atol3e-5/rtol3e-4. Hooks do not modify outputs.

Gaussian K4/K16 use seed42+imageID, identical noise starts, eta0. Attention captured first and final calls (K4 t999/t249; K16 t999/t61). Each measurement is at the incoming state of that decoder block; its spatial association is assessed against the completed call prediction. Clean control uses constructed original GT slots once, excluding duplicates/negatives; B1 negative construction may use synthetic fallback.

Gaussian tables summarize top10 existence-scored slots per image; all50 summaries also saved. Head averages are probability averages, not importance weights based on value/output magnitudes. Per-image summaries averaged equally across32images.

## Which scales?

| Condition | Layer | stride8 | stride16 | stride32 | stride64 |
|---|---:|---:|---:|---:|---:|
| K4 first | 1 | 13.04% | 74.49% | 0.65% | 11.82% |
| K4 first | 2 | 13.60% | 61.46% | 0.33% | 24.61% |
| K4 first | 3 | 17.82% | 80.83% | 0.95% | 0.40% |
| K4 final | 1 | 0.35% | 0.77% | 47.95% | 50.93% |
| K4 final | 2 | 1.69% | 11.07% | 46.86% | 40.38% |
| K4 final | 3 | 3.86% | 74.94% | 9.09% | 12.11% |
| K16 final | 1 | 0.33% | 0.64% | 48.70% | 50.33% |
| K16 final | 2 | 0.94% | 6.95% | 51.19% | 40.92% |
| K16 final | 3 | 3.79% | 70.98% | 14.47% | 10.76% |
| Clean GT control | 1 | 1.27% | 8.68% | 22.48% | 67.57% |
| Clean GT control | 2 | 26.82% | 72.98% | 0.16% | 0.04% |
| Clean GT control | 3 | 0.56% | 78.98% | 18.42% | 2.04% |

Uniform attention over valid tokens would allocate about74.99/18.94/4.82/1.25% to strides8/16/32/64. Thus coarse dominance is learned, not explained by coarse levels having more tokens. K4-final L1 stride64 receives~41times its uniform-token mass. This is attention preference, not causal contribution.

Head specialization: K4-final L1 has4heads predominantly stride32 and4 predominantly stride64. L2 has4 predominantly stride32,3 stride64,1 stride16. L3 has7 predominantly stride16 and1 stride64. In L3, zero-based head5 puts~100% of its mass at stride16; head1 puts83.7% at stride64.

## Spatial localization

| K4 final layer | Scale | Attention inside predicted box, conditional on scale | Uniform-token baseline | Attention inside associated GT box | GT baseline |
|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 24.1% | 24.7% | 35.9% | 32.2% |
| 1 | 16 | 18.9% | 24.8% | 30.2% | 32.8% |
| 1 | 32 | 23.7% | 24.9% | 35.7% | 32.9% |
| 1 | 64 | 13.1% | 23.6% | 28.6% | 32.5% |
| 2 | 8 | 52.5% | 24.7% | 59.3% | 32.2% |
| 2 | 16 | 55.3% | 24.8% | 62.5% | 32.8% |
| 2 | 32 | 54.2% | 24.9% | 61.3% | 32.9% |
| 2 | 64 | 35.5% | 23.6% | 46.7% | 32.5% |
| 3 | 8 | 72.8% | 24.7% | 73.8% | 32.2% |
| 3 | 16 | 77.9% | 24.8% | 77.3% | 32.8% |
| 3 | 32 | 57.3% | 24.9% | 64.9% | 32.9% |
| 3 | 64 | 31.2% | 23.6% | 50.6% | 32.5% |

Layer1 is a broad/coarse read: localization is weak or below uniform for some levels. Layer2 becomes more person-localized. Layer3 stride16, which carries74.94% of attention, puts77.9% of its within-level mass inside the predicted box versus24.8% under uniform attention;77.3% inside the retrospectively associated GT box versus32.8% uniform. Its attention centroid is3.6% of the image diagonal from the predicted box center. At K16 final these become79.7/28.9% predicted-box and75.3/32.7% GT-box. This supports person-region localization in this branch; it does not establish precise joint localization.

GT association uses the best-OKS person for each completed-call pose, not a known generated-slot identity. Broad or overlapping boxes can inflate overlap. GT never enters Gaussian sampling. This is a descriptive association and not a correct-person assignment score.

## Do scales look at the same place?

There is partial broad spatial agreement, especially in layer2. K4-final layer2 stride16/32 peak separation averages6.6% of the padded-image diagonal (common32x32 raster); layer1 averages7.2%. In layer3 the stride16/32 peaks are separated by33.5%, and stride8/16 by30.8%. So this is not consistent sampling of the same body location across every scale. It may read different body regions or another person/background; those roles are not proven by peaks. Fine stride8 has only3.9% of final-layer mass.

Within-level maps are bilinearly resized to32x32 for pairwise Jensen-Shannon divergence/peak distances. Peak distance is unstable for diffuse or nearly tied maps. Center/overlap metrics use normalized feature-cell centers relative to resized unpadded pixels. A backbone cell has a receptive field wider than its center, especially stride64; this is not exact pixel support.

## Figures

![Scale mass](../../tmp/v26_attention/scale_mass.png)

Example images are the first3 fixed-manifest images, each shown for highest-confidence and retrospectively best-pose slot; they were not chosen by visual outcome. Heatmaps average heads, normalize each scale separately and use an independent maximum per panel. Brightness across panels does not indicate total mass; read the percentage labels. Cyan points are predicted joints; green box is best-OKS GT person.

- [example_0_confidence.png](/home/gasparyanartur/dev/pose-graph-diff/tmp/v26_attention/example_0_confidence.png)
- [example_0_best_pose.png](/home/gasparyanartur/dev/pose-graph-diff/tmp/v26_attention/example_0_best_pose.png)
- [example_1_confidence.png](/home/gasparyanartur/dev/pose-graph-diff/tmp/v26_attention/example_1_confidence.png)
- [example_1_best_pose.png](/home/gasparyanartur/dev/pose-graph-diff/tmp/v26_attention/example_1_best_pose.png)
- [example_2_confidence.png](/home/gasparyanartur/dev/pose-graph-diff/tmp/v26_attention/example_2_confidence.png)
- [example_2_best_pose.png](/home/gasparyanartur/dev/pose-graph-diff/tmp/v26_attention/example_2_best_pose.png)

## Interpretation and limits

The model uses image content and develops scale- and head-specific spatial preferences. Poor AP cannot be explained simply as uniformly attending everywhere or ignoring the person. The attention does not consistently align the same fine body location across scales, and fine stride8 is heavily downweighted. This remains an observational audit: ablating/replacing actual values and measuring downstream prediction changes would be needed to establish causal utility or harm.

Artifacts: `/home/gasparyanartur/dev/pose-graph-diff/tmp/v26_attention/metrics.json` (all32 per-image and aggregate measures), `probe.py`, `console.log`, figures. No AP or full-test evaluation is claimed here.

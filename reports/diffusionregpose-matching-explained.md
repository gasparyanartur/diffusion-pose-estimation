# Hungarian matching and OTA in DiffusionRegPose

During training, the model predicts a *set* of person candidates. The ground
truth contains a set of people, but it does not say which prediction slot should
learn which person. Matching creates that correspondence for the loss. It is
computed separately for each image and each supervised clean output.

This report explains the two available strategies in this repository:
one-to-one Hungarian matching and pose-aware Dynamic-K **OTA** (optimal
transport assignment, named after the
[OTA paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Ge_OTA_Optimal_Transport_Assignment_for_Object_Detection_CVPR_2021_paper.pdf)).
The current DiffusionRegPose
[preset](../conf/pipeline/diffusionregpose.yaml) defaults to Hungarian;
[PoseFlow](../conf/pipeline/poseflow.yaml) defaults to OTA. Both strategies
operate on complete person predictions, even though pose decoder layers contain
separate box and joint tokens internally.

## The shared cost matrix

For `Q` clean predictions and `T` labeled people in one image, both matchers
build a `Q × T` matrix. Entry `C[q,t]` measures how expensive it would be to
assign query `q` to person `t`. Lower is better. The implementation is
[`_diffusionregpose_pairwise_matching_cost`](../src/models/pipelines/pose_estimators/diffusionregpose/training.py#L530).

The cost always includes a sigmoid focal **classification** term for the
person class. The geometric terms depend on the supervised output:

| Output | Assignment geometry | Default class / geometry cost weights |
| --- | --- | --- |
| Box-only decoder or encoder proposal | L1 distance between normalized `cx,cy,w,h` boxes and negative generalized IoU | `2 / 5 / 2` (class / box / GIoU) |
| Pose-capable decoder | L1 distance over labeled joint XY coordinates and `1 − OKS`, using target area and CrowdPose joint sigmas | `2 / 10 / 4` (class / joints / OKS) |

For pose-capable outputs, the box L1 and GIoU **assignment costs** are zero;
their box predictions still receive box losses after matching. A joint is
labeled when its target visibility is greater than zero, including labeled but
occluded joints. The exact default weights are in the
[pipeline config](../conf/pipeline/diffusionregpose.yaml#L44). The matcher
runs without gradients; choosing pairs is distinct from the differentiable
classification, box, keypoint, and OKS losses applied to those pairs by the
[criterion](../src/models/pipelines/pose_estimators/diffusionregpose/training.py#L868).

## Hungarian: one prediction per person

The [Hungarian matcher](../src/models/pipelines/pose_estimators/diffusionregpose/training.py#L606)
passes the cost matrix to SciPy's
[`linear_sum_assignment`](https://scipy.github.io/devdocs/reference/generated/scipy.optimize.linear_sum_assignment.html).
It chooses the lowest-total-cost set of pairs subject to **at most one target per query and
at most one query per target**. With at least as many queries as people, each
person gets one prediction; all remaining queries are classification
negatives. The algorithm optimizes the assignment *across the whole image*,
so taking each person's cheapest query independently would be wrong when two
people prefer the same query. There is no cost or IoU threshold that rejects a
poor match. If `Q < T`, only `Q` people can be matched.

In symbols, Hungarian minimizes `sum(C[q,t] for chosen (q,t))` under those
one-to-one constraints. The clean loss is normalized by the number of
ground-truth people in the batch, clamped to at least one. This is the
reference-compatible default
[`normalization_count`](../src/models/pipelines/pose_estimators/diffusionregpose/training.py#L623).

## OTA: several predictions can learn one person

The [OTA matcher](../src/models/pipelines/pose_estimators/diffusionregpose/training.py#L660)
starts with the *same* class and task-geometry cost. It then uses predicted
person boxes for candidate preference and for a Dynamic-K positive count:

1. For each query–person pair, test whether the predicted box center lies
   inside the target box and an expanded central region. A pair outside the
   intersection gets a finite cost penalty of `100`. A query outside every
   target's box-or-center region gets an additional finite `10,000` penalty.
   These penalties strongly discourage distant pairs but do not prohibit them.
2. Compute box IoU for every query–person pair, even at a pose-capable layer
   whose ordinary assignment cost uses joints instead of boxes. For each
   person, sum its highest `min(ota_candidate_topk, Q)` IoUs, convert that sum
   to an integer, and clamp it to at least one. In short,
   `dynamic_k(t) = max(1, int(sum(top_k IoU(q,t))))`. The default top-K is `5`.
   This gives that person's desired number of positive queries.
3. For each person, select its `dynamic_k` lowest *penalized-cost* queries.
   A query selected for two people is kept only for the lower-cost person.
   If that leaves a person with no query, move an available query or one from
   a person with multiple matches to the missing person. The matcher requires
   `Q ≥ T`, so it can preserve at least one match per person.

After conflict resolution, **one query belongs to at most one person**, but a
person may have several queries. Those repeated target indices refer to the
entire person record: class, box, keypoints, visibility, and area. This is not
joint-by-joint matching. The clean loss is normalized by the *number of
positive query assignments*, rather than by the number of people; see
[`normalization_count`](../src/models/pipelines/pose_estimators/diffusionregpose/training.py#L679).
This implementation uses Dynamic-K selection and conflict repair; its name
does not mean it runs a separate transport solver.

## A small assignment example

Suppose there are three queries and two people. These are illustrative
shared base costs; assume OTA's spatial penalties do not change their ranking:

| Query | Person A | Person B |
| --- | ---: | ---: |
| `q1` | 1 | 8 |
| `q2` | 2 | 7 |
| `q3` | 9 | 1 |

Hungarian chooses `q1 → A` and `q3 → B`, total cost `2`. Query `q2` is
unmatched. Suppose the box IoUs for `q1, q2, q3` are `0.9, 0.8, 0.4` against
A and `0.05, 0.05, 0.95` against B. Summing the top three gives `2.1` and
`1.05`, so integer conversion gives `dynamic_k(A)=2` and `dynamic_k(B)=1`.
OTA chooses `q1 → A`, `q2 → A`, and `q3 → B`. Both `q1` and `q2` now receive
full-person supervision for A. The example isolates the cardinality
difference; actual `dynamic_k` values come from predicted box
IoUs, not from the cost table.

| Property | Hungarian | Dynamic-K OTA here |
| --- | --- | --- |
| Matches per person | One when `Q ≥ T` | One or more after conflict repair |
| Matches per query | At most one | At most one |
| Selection | Global minimum-cost one-to-one assignment | Spatially penalized, per-person Dynamic-K selection, then conflict and coverage repair |
| Clean-loss denominator | Ground-truth person count | Positive query-assignment count |

## Where matching applies

The [criterion](../src/models/pipelines/pose_estimators/diffusionregpose/training.py#L1009)
rematches the final clean decoder output, each auxiliary clean decoder output,
and the selected encoder proposal output independently. A default box-only
decoder layer and the encoder proposal use box-aware assignment costs;
pose-capable decoder layers use pose-aware costs. The chosen pairs supervise
the clean classification and relevant geometry losses. Unmatched clean
queries still contribute classification negatives, including on an image
with no labeled people. For an empty target image, both matchers return no
positive pairs.

Training also creates denoising queries from known targets. Their
correspondence is constructed directly, so they bypass both matchers and have
separate classification and box losses. Matching does not assign the noisy
keypoint proposals used to condition the diffusion or flow path; the criterion
matches *predicted people* to labeled people.

## What the existing experiment shows

The [selected-layer comparison](diffusionregpose-ota-comparison.md) holds its
resolved architecture, data, initialization, schedule, seed, and evaluator
fixed while changing clean-query assignment. In that one-seed, reduced-depth
256/427-pixel experiment, OTA had higher held-out CrowdPose test keypoint and
bbox AP. The result supports OTA for that measured setting; it does not yet
establish the effect for a full-scale model. That report contains the precise
checkpoint, split, evaluator, and AP provenance.

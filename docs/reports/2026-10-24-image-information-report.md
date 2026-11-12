# DiffusionRegPose investigation: system properties, image representations and h

**Consolidated research report · 23–24 October 2026**

This report brings together the probe designs, construction pilots, image
interventions, native DiffusionRegPose notebook and diagnostics, V24 representation
probes, and completed matched training comparison. It is the single reference
for these related investigations. The notebook remains available for interactive
inspection; the original standalone reports have been merged and removed.

The goal is to identify **properties of the current systems**: which inputs affect
predictions, where image information enters, what individual stages contribute,
which distinctions are accessible in features, and whether extra calls or h help.
An observed sensitivity or inactive computation route is a useful finding without
having to establish a complete theory of geometric understanding.

**Main result for native DiffusionRegPose:** large changes in diffusion state
barely change joint predictions. The first pose stage has an inactive joint
image-value read but necessary feature processing; a later joint image read
contributes strongly to localization. A coordinate clamp removes variation on
one geometric path before refinement.

**Main result for V24:** the h-trained system makes a hidden-arm distinction easier
to classify from frozen image features, but neither model correctly localizes
the hidden wrists in that construction. At the final checkpoint, h improves
full-test pose AP by 0.3808 points in one matched training pair. These experiments
measure different properties and do not establish the cause of the AP gain.

## Outline

1. [Findings and their scope](#findings)
2. [Models, checkpoints and datasets](#models)
3. [Probe design, scoring and execution](#methods)
4. [Native DiffusionRegPose: performance, geometry and active computation](#native)
5. [V24: hidden joints, feature accessibility and depth labels](#representations)
6. [Preliminary real-image interventions](#interventions)
7. [Final matched h/control training result](#training)
8. [Interpretation and research decisions](#interpretation)
9. [Construction history, corrections and verification](#history)
10. [Reproducibility, source inventory and complete contrasts](#evidence)
11. [Live multi-step h/control comparison](#multistep)

<a id="findings"></a>
## 1. Findings and their scope

| System property | Evidence | Supported conclusion |
|---|---|---|
| Native DiffusionRegPose has weak sensitivity to the tested diffusion states | At fixed image/time, state differences of 48.6–96.0 pixel-equivalent units produce mean final joint movements of 0.0116–0.0156 pixels. | Different native seeded trajectories scarcely change the predicted joints on this cohort. |
| A coordinate boundary operation collapses geometric variation | All 336,000 observed first-pose joint references have negative X and Y; clamping maps them to the same boundary coordinates. | This coordinate residual path discards differences between those references. |
| The first pose stage's joint image-value route is inactive | Its image-attention output equals projection bias in all 480 native calls; replacing it with that bias changes no outputs. | That particular route supplies no image values on the tested inputs. |
| The first pose stage still performs necessary feature updates | Removing its joint self-attention and feed-forward residual updates reduces pose AP to zero. | The stage is not redundant despite its inactive joint image read. |
| The final pose stage uses joint image evidence | Removing its joint image values changes final-call AP from 50.42 to 18.45. | Useful joint image values enter through this later route. |
| V24 image features preserve a hidden-arm distinction | A newly supervised linear encoder-feature classifier labels both alternatives correctly in 86.3% of control pairs and 96.9% of h pairs. | The distinction is accessible to a small reader; this is not native coordinate accuracy. |
| The V24 hidden-wrist behavior fails despite visible competence | Both models detect all people but localize 0/1,024 hidden wrists; visible correctness is 1,023/1,024 and 1,024/1,024. | This construction exposes an occluded-joint localization failure in both arms. |
| The h-trained model has a small full-test advantage | Final pose AP is 54.9141 control versus 55.2949 h over 8,000 test images. | h gains 0.3808 AP points for this paired training seed. |

The native findings concern a **32-image diagnostic cohort and one reduced
DiffusionRegPose checkpoint**. The V24 feature percentages above use all 512
synthetic test pairs in an explicitly post-hoc coverage check. Full-test AP uses
separate final V24 checkpoints. These boundaries prevent transferring a finding
from one model, checkpoint or metric to another.

<a id="models"></a>
## 2. Models, checkpoints and datasets

### 2.1 Native DiffusionRegPose

The notebook uses the reduced OTA epoch-79 checkpoint from the September training
run. It has three encoder and three decoder layers; the decoder stages are a
box-only layer followed by two joint-refinement layers. They were initialized
from ED-Pose decoder layers 2/4/6, but the executed order is labelled 1/2/3.
Each diffusion call executes the whole model. Only joint XY proposals are
diffusion variables in this inspection; incoming references, predicted joints
and image-attention sample locations are distinct quantities.

The fixed cohort comprises 32 CrowdPose test images, sampled without replacement
with subset seed 42 from eligible images with labelled noncrowd people. Initial
proposals use `1729 + image_id`, reset for each step count. Sampling is native
deterministic DDIM, eta=0, resize 256/427 and the native CUDA attention kernel.
The notebook evaluates one, two and three calls. The later state-sensitivity
study adds five proposal seeds at the three-call schedule 999 → 665 → 332.

### 2.2 Matched V24 control and h

V24 is a separate endpoint-flow family. Both arms share spatial encoder
proposals, 200 initial candidates, top-50 selection, the original geometry path,
losses, optimizer schedule and corrected per-image coordinate layout. The h arm
adds a person-level residual writer/reader in the two pose stages. h initializes
from an image read, updates between those stages, and resets on each forward.
It is **not carried between sampling calls**. Extra modules start behind zero
residual gates; the initial shared computation and RNG were checked for parity.

Both models start from matched weights containing ED-Pose pretraining and then
train separately. Thus adding h can change the backbone and encoder through
training. A difference between the two trained systems does not isolate h's
inference-time route. Parameter counts are 40,511,752 control and 42,353,674 h.

| Evidence | Model age | Data |
|---|---|---|
| Preliminary image changes | V24 control epoch 63 | 64 selected real test images |
| Synthetic representation suite | V24 control/h epoch 65, step 99,000 | Newly rendered people; fixed weights |
| Initialization reference | Exact shared starting weights, already pretrained | Same depth-order renders |
| Final h comparison | V24 control/h epoch 79, step 120,000 | All 8,000 held-out CrowdPose test images |
| Native inspection and state diagnostics | Reduced DiffusionRegPose epoch 79 | 32 real test images |

The synthetic probes stayed at epoch 65 as training continued. They do not
measure final epoch-79 representations. Four unstarted seed-43/44 training runs
were cancelled; the matched training experiment retains only the seed-42 pair.

<a id="methods"></a>
## 3. Probe design, scoring and execution

The investigation used three complementary types of measurement:

- **Observation and controlled replay:** record native states, reference geometry
  and attention, then change a specific input or computation route while holding
  the image, timestep and other inputs fixed.
- **Native behavior:** ask the unchanged estimator to localize a joint in paired
  scenes with a known changed answer. Model outputs receive no ground-truth joint
  alignment or list of candidate answers.
- **Frozen-feature classification:** train a small new classifier on fixed image
  or decoder features to measure whether a particular distinction is accessible.
  New labels and any known sampling location are part of that classifier's task.

### 3.1 Synthetic scene and admission protocol

Scenes use SMPL-X bodies, Blender 4.4.3 and orthographic 384×256 renders. Depth,
segmentation, joints and camera rays are for construction audits; native models
receive RGB only. A scene family contains the related paired configurations and
appearance repeats, and stays wholly within a reader split.

Each construction gets a 32-family pilot and at most one repair using a fresh
pilot. Required controls include geometry/label validity, identical quantities
where promised, ≥85% native visible competence in both models, and ≥85% RGB
label discrimination under whole-family cross-validation. The RGB reader is
PCA16 plus a small logistic classifier; it checks distinguishability, not native
pose competence or a particular semantic cue. Main cases are selected by
physical checks before model predictions, never by retaining scenes a model solves.

Native synthetic scoring selects the highest-confidence qualifying person after
native NMS: class 1, score ≥0.5 and box IoU ≥0.5. Wrist correctness requires
`error ≤ 0.04 × sqrt(shared person-box area)`. Alternative targets are separated
by at least three times that tolerance. Missing detections are failures;
duplicates and strict single-candidate paired success are also recorded.

### 3.2 Reader contracts and different scores

Ordinary readers use all 256 channels, train-only standardization and L2 logistic
regression. Regularization is selected on validation from
`{0.01, 0.1, 1, 10, 100}`; ties prefer stronger regularization. Main appearance
alone is used for fitting and selection; appearance repeats test transfer.

**Hidden-arm classification** requires both short/long labels to be correct in a
pair. A constant answer fails every pair. It uses 256/64/512 training/validation/
test families and samples maps at the shared visible elbow, never the hidden
true wrist. Decoder readers use the associated elbow feature.

**Depth ordering** asks whether the “wrist nearer” score is higher in the correct
member of a depth-opposite pair; ties count as half. This uses 1,024/128/512
families, maps at the shared wrist, and held-out camera yaws and lighting. Pair
ordering can be high despite poor absolute per-image classification. It must not
be compared directly with the hidden-arm both-labels score.

Intervals use 10,000 paired bootstrap resamples by physical family. Repeated
renders do not replicate model training seeds. Zero-event bootstrap intervals
can degenerate and do not prove a population probability of exactly zero.

### 3.3 Predeclared useful effects

The registered margins were a 10-point native h advantage on the paired ambiguity
tasks, 50% native paired capability given ≥85% unambiguous competence, depth
accessibility 10 points above its 50% shared-2D bound, and incremental h depth
accessibility 5 points above both capacity-matched controls. Nuisance preservation
allowed at most a 10-point drop. The protocol also required ≥90% correspondence
coverage before generalizing feature comparisons to the whole synthetic cohort.
These margins belong to their specified tasks, not to the native 32-image masks.

<a id="native"></a>
## 4. Native DiffusionRegPose results

### 4.1 More calls barely change subset AP

| Native sampling calls | Pose AP | Bounding-box AP |
|---|---:|---:|
| 1 | 50.5718 | 47.7583 |
| 2 | 50.7381 | 47.7054 |
| 3 | 50.4237 | 47.8515 |

On this cohort, two calls gain 0.1663 AP points over one, while three lose 0.1481.
This is a small subset diagnostic, not a full-test scaling result. Individual
images can improve or worsen while pooled AP remains nearly flat.

![Native subset performance](/home/gasparyanartur/dev/pose-graph-diff/docs/reports/assets/diffusionregpose-investigation/native-step-performance.png)

*Figure 1. Notebook output: pooled 32-image pose AP in blue and per-image AP in
gray. The original plot uses the 0–1 AP scale; tables use 0–100. Common initial
noise is reset at each step count.*

The same best and worst three images are selected at every call count. Best:
111057, 115025, 100537. Worst: 100656, 102109, 104488. Ranking uses native
per-image CrowdPose AP, with image-ID tie-breaking and no undefined scores.
The gallery's confidence cutoff is display-only; all native detections enter AP.

![Best and worst native predictions](/home/gasparyanartur/dev/pose-graph-diff/docs/reports/assets/diffusionregpose-investigation/native-best-worst-three-steps.png)

*Figure 2. Three-call gallery from the notebook: annotations and predictions for
the three best and three worst images by the disclosed ranking. White rings
identify annotated occluded joints. These are selected examples, not a random
visual sample or additional aggregate evidence.*

### 4.2 Incoming references and predicted poses are different

The notebook traces actual box-to-joint expansion, incoming decoder references,
auxiliary/final heads and image sample locations separately. In the six selected
images across all traced settings/calls, the first pose stage has 25,200/25,200
joint-reference centers outside the image; the final pose stage has 0/25,200.
The preceding box-only layer has 0/7,200 outside reference centers.

![Decoder geometry and predictions](/home/gasparyanartur/dev/pose-graph-diff/docs/reports/assets/diffusionregpose-investigation/native-decoder-geometry.png)

*Figure 3. Image 111057, third of three calls. The left column is incoming
reference geometry; middle/right columns are actual predictions. The first-pose
reference panel expands its axes to show coordinates outside the image, whereas
later predictions lie on the person. A local query-group number is not a tracked
person identity across calls.*

This is real checkpoint geometry, not a plotting conversion. Coordinate
refinement is unbounded in the inspected implementation and the external
reference path also permits that pattern; this alone does not establish a local
implementation discrepancy or an AP cause.

### 4.3 Structural connectivity and image sample strength

![Structural query masks](/home/gasparyanartur/dev/pose-graph-diff/docs/reports/assets/diffusionregpose-investigation/native-query-mask.png)

*Figure 4. Actual structural query masks: white allows an interaction and black
blocks it. Box-only queries are unrestricted. After expansion, joints interact
within their person group; box queries can also read other boxes. This shows
allowed connectivity, not learned attention strength.*

![Weighted left-wrist image sampling](/home/gasparyanartur/dev/pose-graph-diff/docs/reports/assets/diffusionregpose-investigation/native-wrist-attention.png)

*Figure 5. Final-stage left-wrist attention in image 111057. Star: incoming
reference; yellow ring: prediction; green cross: annotation. Arrows show spatial
offsets; thickness/opacity and dot size encode original sample weight. Only the
strongest feature level for head 0 is displayed, without renormalizing that
level's weights.*

In this example, the nearby two samples carry 45.66% and 19.42% of all 16 sample
weights in that head. Distant samples carry 0.40% and 0.51%. Treating all arrows
as equally strong would misrepresent the computation. For the three opening
examples, head-0 weight inside the associated person box is 83.87%, approximately
100%, and 57.22%. A person box is only a spatial proxy, not foreground segmentation.
Encoded features can contain nonlocal context, so sample positions are not a
pixel-level causal saliency map.

Independent reconstruction on image 111057's three calls gives sample-coordinate
error zero, weight error ≤1.79×10⁻⁷ and native attention-output error ≤2.48×10⁻⁶.
The sampled full predictions match exactly. The visual interpretation is thus
checked against the actual native computation.

### 4.4 Large state differences produce tiny joint differences

For each fixed image/time, compare five native three-call trajectories seeded
`1729 + image_id + 100000 × seed_index`. Later states are native trajectory states,
not arbitrary noise at later timesteps. Selected box-query slots stay identical
across these seed comparisons.

| Call / timestep | Input-state difference, pixel equivalent | Incoming-reference movement, pixels | First-pose joint movement, pixels | Final joint movement, pixels |
|---|---:|---:|---:|---:|
| 1 / 999 | 96.0023 | 13.0105 | 0.0159 | 0.0116 |
| 2 / 665 | 83.2875 | 16.5752 | 0.0203 | 0.0156 |
| 3 / 332 | 48.6365 | 12.3134 | 0.0156 | 0.0116 |

Averages cover corresponding raw slots, 50 groups × 14 joints × 32 images ×
four alternative seeds. Pixel-equivalent state uses `xy = (state / 6 + 1) / 2`;
those noisy coordinates need not form valid poses. Final-call AP across the five
seeds ranges from 50.42108 to 50.43030.

**Property:** native seeded state variation has very weak influence on predicted
joint positions here. This is not exact state independence, nor a measurement of
every possible perturbation or local state derivative.

### 4.5 Clamping collapses a geometric path

Across the expanded diagnostic, all 336,000 first-pose incoming joint-reference
observations have negative X and Y. Raw coordinates vary between seeds, but
clamping them to `[0,1]` always produces `(0,0)`. The inverse-sigmoid residual
path therefore starts from the same boundary logit.

**Property:** this coordinate path loses variation before applying the learned
correction. Raw references still enter positional embeddings, allowing a small
remaining state effect. This identifies one concrete mechanism; it does not
establish the full cause of weak sensitivity or why training reached this state.

### 4.6 Useful computation survives an inactive first joint image read

The first-pose joint cross-attention output equals its learned output-projection
bias in all 480 native calls. Image values contribute exactly zero. A replay
replacing only that output with its bias leaves logits, boxes and joints exactly
unchanged. Box rows and the bias are preserved.

A separate replay zeros only first-pose joint self-attention and feed-forward
residual contributions, retaining inputs, normalizations, coordinate heads,
image attention and box updates. Pose AP collapses to zero. Removing final-stage
joint image values instead leaves AP near 18.45.

| Conditional single-call replay | t=999 | t=665 | t=332 |
|---|---:|---:|---:|
| Baseline pose AP | 50.5718 | 50.6722 | 50.4237 |
| First-pose joint image values removed | 50.5718 | 50.6722 | 50.4237 |
| Final-pose joint image values removed | 18.2031 | 18.3252 | 18.4477 |
| First-pose joint feature updates removed | 0.0000 | 0.0000 | 0.0000 |
| First updates and final joint image values removed | 0.0000 | 0.0000 | 0.0000 |

![State sensitivity and computation-route probes](/home/gasparyanartur/dev/pose-graph-diff/docs/reports/assets/diffusionregpose-investigation/native-state-and-route-probes.png)

*Figure 6. Verified native diagnostic: log-scale state/reference/prediction
movement at left, conditional third-call AP at right. The inactive first joint
image read and necessary first-stage feature updates are different findings.*

**Interpretation:** image-conditioned queries enter a stage doing necessary
feature/geometry processing despite an inactive joint image-value read; a later
stage supplies a useful joint image read. This characterizes the computation
more precisely than calling the entire intermediate stage ineffective.

Masks replay one call at its baseline image/time/state; they do not rerun complete
masked trajectories. Downstream box interactions and pose NMS can still change
retained detections. Abrupt masks measure dependence of the current computation,
not the performance achievable after retraining. Both first-update removal and
its combination with final-read removal hit zero AP, so the interaction cannot
quantify compensation. No learned compensation mechanism has been established.

<a id="representations"></a>
## 5. V24 representation and synthetic behavior results

### 5.1 Hidden wrists: a native localization failure

Paired rendered arms share visible joint positions, projected forearm direction
and physical bone length, but their hidden wrist projects a shorter or longer
distance from the elbow. A shared foreground panel hides the hand/distal forearm,
leaving some proximal arm surface visible. The actual forearm length is 35 render
pixels; projected short/long lengths are 0.35 and 0.985 of that length.

![Hidden-wrist paired construction](/home/gasparyanartur/dev/pose-graph-diff/docs/reports/assets/diffusionregpose-investigation/hidden-wrist-pair.png)

*Figure 7. First retained repaired pilot family, 330001. Red crosses are true
wrist annotations added for this report, never model inputs. The hidden targets
are 22.225 pixels apart and the allowed error is 6.840 pixels in this example.
The panel-free controls show the two physical configurations. Pilot examples
illustrate construction; aggregate results below use the main test set.*

Final pilots pass visible-wrist competence 64/64 in both models and RGB
discrimination 63/64. Main images use the same per-family source across arms.

| Native behavior over 512 test families | Control | h |
|---|---:|---:|
| Person detected, hidden condition | 1,024/1,024 | 1,024/1,024 |
| Wrist correct, panel removed | 1,023/1,024 | 1,024/1,024 |
| Wrist correct, panel present | 0/1,024 | 0/1,024 |
| Both hidden alternatives correct | 0/512 | 0/512 |
| Extra qualifying detections, hidden condition | 0 | 0 |

The appearance-change repeat also has zero hidden-wrist successes. This is a
specific occluded-joint failure despite visible competence and person detection.
It does not describe every natural occlusion or identify its internal cause.

### 5.2 The short/long distinction remains accessible in features

The pose estimators remain frozen while a linear classifier learns the short/
long decision from new task labels. Image maps are sampled at the known shared
visible elbow; providing the hidden wrist as a sampling location would leak the
answer. A successful pair requires both class labels correct, not coordinates.

The registered stage comparison needs corresponding detected person tokens in
both models and images. It retains only 156/512 test families (30.5%), 113/256
training and 31/64 validation families. This fails the 90% coverage condition
for generalizing that representation comparison to the whole cohort.

| Input source, restricted common cohort | Control: both correct, 95% interval | h: both correct, 95% interval |
|---|---:|---:|
| Projected backbone map | 42.3% [34.6, 50.0] | 85.3% [79.5, 90.4] |
| Encoded image map | 70.5% [63.5, 77.6] | 95.5% [92.3, 98.7] |
| First-pose elbow feature | 47.4% [39.7, 55.1] | 70.5% [63.5, 77.6] |
| Final-pose elbow feature | 48.1% [40.4, 55.8] | 90.4% [85.3, 94.9] |
| Native predicted pose/box/score reader | 31.4% [24.4, 39.1] | 44.2% [36.5, 51.9] |

We then fitted four **post-hoc map readers on all original families**, avoiding
the detector-token filter. This uses the unchanged training/validation protocol
and preserves the primary results above.

| Input source, all 512 test families | Control | h |
|---|---:|---:|
| Backbone map | 36.7% [32.6, 41.0] | 83.4% [80.1, 86.5] |
| Encoded image map | 86.3% [83.4, 89.3] | 96.9% [95.3, 98.2] |
| Encoded map, changed appearance | 83.0% | 98.2% |

![Short-long classification from frozen image features](/home/gasparyanartur/dev/pose-graph-diff/docs/reports/assets/diffusionregpose-investigation/hidden-wrist-feature-classifiers.png)

*Figure 8. Post-hoc all-family classifier means and 95% scene-pair bootstrap
intervals. Each score requires both labels correct. The classifiers get new
supervision and known elbow alignment; these bars are not pose localization AP.*

The paired h-minus-control encoder difference is +10.5 points [7.6, 13.5]; the
backbone difference is +46.7 [42.4, 51.2]. The distinction survives image processing
and is easier for this reader in the h-trained system. Binary classification
is easier than accurate coordinates, and the visual cue is not isolated as
foreshortening. These are meaningful accessibility measurements without proving
that the native decoder has a complete geometric solution or is the sole
bottleneck. Because shared features train differently with h, the comparison also
does not isolate h's inference-time causal route.

### 5.3 Depth labels beyond identical complete 2D projections

A second construction fixes all 14 projected joint positions and bone length
while changing the sign of wrist-minus-elbow depth. Geometrically,
`dz = ±sqrt(L² − projected_length²)`. Half the families expose surface cues and
half torso-occlusion cues, checked using actual camera rays. A function receiving
the literal shared 2D skeleton must tie within every pair.

Reader splits contain 1,024/128/512 physical families, four images per family
including appearance repeats. Camera yaws are −20/0/20° in training, −10/+10° in
validation and −30/+30° at test; test lighting directions are also held out.
All 512 test families survive common model correspondence.

| Reader input | Pretrained initialization | Trained control | Trained h |
|---|---:|---:|---:|
| Backbone map | 100.0% | 100.0% | 100.0% |
| Encoded map | 100.0% | 100.0% | 100.0% |
| First-pose wrist feature | 95.5% [93.6, 97.3] | 97.3% [95.7, 98.6] | 99.8% [99.4, 100.0] |
| Final-pose wrist feature | 94.5% [92.4, 96.5] | 100.0% | 100.0% |
| Native predicted pose/box/score reader | 66.8% [62.7, 70.7] | 97.7% [96.3, 98.8] | 100.0% |

These are within-pair depth-order scores. Initialization already contains ED-Pose
weights; it is not a random network. Both trained final features gain 5.5 points
[3.5, 7.6] over initialization. Final-stage strata are:

| Checkpoint | Surface cues | Torso occlusion | Appearance repeat |
|---|---:|---:|---:|
| Pretrained initialization | 89.5% | 99.6% | 94.1% |
| Control | 100.0% | 100.0% | 100.0% |
| h | 100.0% | 100.0% | 100.0% |

This establishes a label-correlated image distinction absent from the literal
2D skeleton, including before V24 training. Predicted 2D outputs can carry that
label through error patterns; their readout success is not coherent 3D output.

### 5.4 The outside-arm control limits the depth interpretation

| Held-out input control | Pair ordering | Per-image classification |
|---|---:|---:|
| Identical/raw 2D geometry | 50.0% | 50.0% |
| Outside-arm head/leg pixels | 97.5% [96.1, 98.8] | 51.9% |

![Depth ordering and shortcut control](/home/gasparyanartur/dev/pose-graph-diff/docs/reports/assets/diffusionregpose-investigation/depth-order-and-shortcut.png)

*Figure 9. Encoder accessibility reaches ceiling even in pretrained features,
while unrelated head/leg crops nearly solve pair ordering. Weak relative image
proxies can order related images despite near-chance per-image classification.*

Raw-coordinate leakage was not detected. However, arm-specific or coherent-depth
interpretations are not isolated: pose/render/shadow differences remain possible.
Randomizing render order did not remove the warning. This does not prove model
readers use that exact shortcut; it means the assay cannot exclude it. Held-out
appearance and camera performance do not override the failed specificity control.

### 5.5 h does not show an incremental depth-reader advantage at ceiling

All three main combined readers expose 512 inputs and 513 classifier parameters:

| Reader inside h model | Depth pair ordering |
|---|---:|
| Final wrist + final elbow | 99.8% [99.4, 100.0] |
| Final wrist + h | 100.0% |
| Final wrist + shuffled h | 100.0% |

h alone supports 99.6% ordering. Real h beats elbow context by 0.2 points
[0.0, 0.6] and ties shuffled h, below the registered 5-point margin over both
controls. Shuffling uses label-independent individual rows from other families
within each split; it does not preserve the depth-label slot and does not alter
native inference. This saturated test provides no useful incremental h gain
and cannot establish a global absence of additional information in h.

### 5.6 Ownership construction stopped at admission

The ownership task intended to change limb connections while holding local joint
patches and the unordered joint positions fixed. Initial patch copying visibly
severed limb boundaries. A single repair introduced physical occluders, changing
the question to occluded-joint ownership.

Raw frontal visible controls were solved 32/32 by each arm, but back-facing
controls only 11/32 control and 3/32 h. The repaired construction passed scoped
physical/pixel audits and RGB discrimination 63/64, but native competence failed:

| Separated-person control | Original control/h | Corrected matching only, control/h |
|---|---|---|
| Same occluders as challenge | 6/64; 4/64 | 8/64; 5/64 |
| Occluders removed | 32/64; 32/64 | 47/64; 49/64 |
| Eligible person matches | 32/64; 32/64 | 64/64; 64/64 |

![Separated-person occluded controls](/home/gasparyanartur/dev/pose-graph-diff/docs/reports/assets/diffusionregpose-investigation/ownership-controls.png)

*Figure 10. First two repaired ownership families in generation order. Green:
true wrist; red: highest-score matched prediction. These show failures in simpler
separated-person controls before the ownership ambiguity is introduced.*

The original padded matching boxes were too wide, rejecting valid detections.
A post-pilot matching-only diagnostic used union mesh bounds while preserving
the original absolute wrist tolerance. This corrected scoring rejections, not
the localization failure. The required ≥85% competence still failed. Main
ownership inference was stopped under the one-repair rule; no ownership-specific
representation conclusion is available.

<a id="interventions"></a>
## 6. Preliminary real-image interventions

This earlier diagnostic uses V24 control epoch 63 and 64 seeded, annotation-
selected test images favouring larger visible arms. Twelve conditions per image
include localized masks with two fills, grayscale, outside-box masking and
reflection. There are 768 condition forwards plus one repeat. The same 63/64
people match in every condition; unmatched detections count as failures.

Errors below are percentages of the square root of annotated person-box area;
positive differences mean worse localization. Intervals are image-bootstrap,
2,000 draws. Raw-candidate matching and errors are not native postprocessed AP.

| Question / contrast | Measured error change, 95% interval | Property measured |
|---|---|---|
| Wrist mask vs unchanged image | +2.42 [1.61, 3.31] | Localization responds to the intervention. |
| Wrist mask vs equal-area distant-body mask | +1.22 [−1.22, 2.86] | Extra specificity of the local wrist corruption is uncertain. |
| Wrist + elbow + shoulder vs wrist + distant-body patches | +7.20 [4.09, 10.89] | Supporting-region removal has a larger behavioral effect here. |
| Same support contrast, alternate fill | +7.68 [4.56, 11.38] | The direction is retained with another fill. |
| Grayscale, all-joint mean | +0.16 [−1.17, 1.17] | Little net mean error change; predictions still move 3.73 units. |
| Outside-person-box masking | +0.65 [−0.59, 1.50] | Predictions move 4.57 units; average accuracy cost is uncertain. |
| Reflection and inverse reflection | −0.62 [−2.15, 0.57] | Similar mean accuracy, but 4.56-unit prediction disagreement. |
| Final vs first pose stage, all joints | −0.81 [−1.09, −0.53] | Later refinement improves these raw-coordinate errors. |
| Final vs first pose stage, selected wrist | −2.13 [−3.59, −0.94] | Refinement particularly helps the selected wrists here. |

![Preliminary image-intervention effects](/home/gasparyanartur/dev/pose-graph-diff/docs/reports/assets/diffusionregpose-investigation/preliminary-image-effects.png)

*Figure 11. Original preliminary diagnostic plot. It measures sensitivity to
image transformations on the stated cohort; it does not identify the encoded
semantic quantity or the internal reasoning mechanism.*

Mean joint error falls from 6.80 to 5.99, improving 52/63 matched people. Wrist
PCK@0.2 is 93.8% originally, 90.6% after wrist removal, 70.3% after wrist/support
removal and 89.1% after wrist/distant-body removal. Supporting-context effects
remain positive in post-specified checks excluding the three largest effects
(+4.76) and requiring higher box overlap (+5.11 over 54 people).

These sensitivity properties are useful but cannot distinguish anatomical
computation from spatial correlation or limb association. Support patches also
lie closer to the wrist than distant controls. Person boxes are not segmentation
masks, and interventions introduce artificial occlusions. Every returned joint
visibility value is the same 0.73105854, so that output does not provide joint-
specific uncertainty in this diagnostic. No calibration result was established.

<a id="training"></a>
## 7. Completed matched training and full-test h comparison

Both seed-42 models reach epoch 79 / step 120,000 after 80 epochs. Final native
single-call inference uses the same 8,000 unique test IDs, Gaussian source
`42 + image_id`, FP32, both TF32 flags off, resize 256/427 and corrected NMS.
All test IDs are absent from the 12,000-image trainval training pool.

| Final metric | Control | h | h minus control |
|---|---:|---:|---:|
| Pose AP | 54.9141 | 55.2949 | +0.3808 points |
| AP50 | 79.8570 | 80.0864 | +0.2294 points |
| AP75 | 58.6247 | 58.7238 | +0.0991 points |
| Pose AR | 62.7368 | 62.9864 | +0.2496 points |
| Bounding-box AP | 51.9010 | 52.0788 | +0.1778 points |
| Validation loss | 8.065090 | 8.058168 | −0.006922 |
| Training loss | 12.141640 | 12.117541 | −0.024098 |

**Result:** a small positive h difference for one paired seed. Scene-bootstrap
precision does not replicate training seeds, and no robust multi-seed advantage
has been established. The epoch-65 feature probes do not explain the final AP gain.

The fixed 32-image monitor favours control, approximately 51.00 versus 49.71 AP.
Rescoring the same IDs from cached final full-test predictions also favours
control, 51.23 versus 49.72. The full test reverses the ordering. That subset's
ordering does not generalize to the full test; the small numerical discrepancy
between the two subset evaluations remains unexplained.

Validation losses are close rather than identical, with minima at epoch 79 for
both. All 2,000 validation images occur in trainval training under the shared
recipe, so those curves are not an independent generalization/overfitting test.
Training and validation totals also have different augmentation and training-
only terms. The earlier observations do not establish that control generalizes
better; the held-out endpoint slightly favours h.

<a id="interpretation"></a>
## 8. Interpretation and research decisions

The native probes support a concrete description of the tested DiffusionRegPose
computation: image-conditioned queries go through useful first-pose feature
processing, despite an inactive joint image-value route; the next stage reads
useful joint image values. Geometric state variation is heavily attenuated, with
one observed loss of variation at the coordinate clamp. More sampling calls do
not automatically mean meaningful revision of distinct geometric hypotheses.

This prioritizes investigating reference construction and state preservation,
rather than treating the whole first pose stage as disposable. It does **not**
yet demonstrate a repair that improves AP, a local-code discrepancy relative to
the reference implementation, or a learned compensation strategy.

The V24 probes identify a different set of properties: failure on a specified
hidden-joint task, a task distinction accessible to newly supervised frozen-feature
readers, and more accessible features after h-enabled training. The task mismatch
between classification and coordinates limits causal interpretation; it does not
make either measurement meaningless. A precise follow-up would test coordinate
learning from frozen features across varied held-out scenes, with coordinate and
rendering-shortcut controls.

The depth test shows why accessibility and semantic interpretation must be kept
separate. Its label can be read beyond the shared 2D skeleton, but outside-arm
pixels nearly solve pair ordering. This prevents a specific claim of coherent
3D pose. The ownership test stopped earlier because native controls failed.
These limits concern those tests; they do not erase the properties established
by the native observations and replays.

### Questions still worth resolving

1. Where are the first-pose references made negative, and what preserves or
   destroys state variation before and after that point?
2. Does weak native state sensitivity persist on a broader cohort and under
   controlled local perturbations, beyond different seeded trajectories?
3. Does the final read recover from moderate earlier feature perturbations?
   Graded self-attention/FFN attenuation plus fixed-assignment coordinate errors
   would avoid the zero-AP floor of abrupt removal.
4. Do frozen V24 features support precise hidden-joint coordinates, rather than
   only the measured binary distinction?
5. Does h alter the benefit of additional inference calls? The requested full-test
   comparison is running in section 11, with h resetting on every call.

These are proposed directions or the explicitly identified running evaluation,
not authorization for new training or model changes. The investigation measures
system properties; it neither requires nor claims a general verdict about reasoning.

<a id="history"></a>
## 9. Construction history, corrections and verification

| Stage | What changed or was checked | Disposition |
|---|---|---|
| Preliminary real-image suite | Local masking, color/context/reflection and stage errors | Retained as sensitivity measurements, not semantic representation proof. |
| Conceptual study design | Separated native competence, accessibility and initialization/stage effects | Converted into fixed paired-scene protocol and small admission pilots. |
| Ownership first construction | Joint patch composition visibly severed limbs | Rejected; single repair used physical foreground occluders. |
| Ownership box audit | Padded reference boxes rejected valid detections | Matching-only correction retained original absolute wrist tolerance; controls still failed. |
| Hidden-arm first pilot | Visible competence passed; RGB 61/64 and outside-arm ordering warning | Fresh repair randomized render order and used shared rest-space texture; final RGB 63/64. |
| Depth first construction | Too few collision-free occlusion scenes | Fresh repair shortened occluded projections; camera-ray membership verified. |
| Provisional depth native check | Ran before final ray verification | Preserved separately; not used for final admission. |
| Final hidden/depth admission | Visible wrists 64/64 per arm; RGB 63/64 and 64/64; depth raw-2D ties | Both main studies executed without lowering thresholds. |
| Hidden-stage correspondence audit | Only 156/512 test families retained | Conditional table qualified; post-hoc all-family map analysis added. |
| Native notebook plots | Distinguished references/predictions; weighted arrows by actual strength | Numerical cache unchanged; visual sample strengths corrected. |
| Native frozen diagnostic | CPU postprocessing introduced score/tie differences | Repeated with original GPU postprocessing and exact cached-output parity; earlier run superseded. |

The native notebook evaluates 96 image/step settings; selected replays cover 18
settings, 36 calls and 108 decoder layers with exact raw-output parity. Initial
sources match across step counts, reconstructed weights sum correctly and hooks
are removed after use. The later verified diagnostic has 864 calls: 480 native
and 384 conditional masked calls. Raw and postprocessed final baseline outputs
match the original cache exactly; first joint image-value maximum is zero.
Twelve focused native diagnostic tests pass. The synthetic suite has strict
restore, hook/no-hook parity, geometry/scoring tests and finite-output checks.

The synthetic information suite consumes 32,117 calls including pilots/parity,
29,696 on main data, and about 7.22 GB under its 10 GB cap. It fits 45 registered
selected readers plus four post-hoc map readers, not 49 new pose estimators.
Selected meshes, original predictions, source snapshots and construction rejection
records remain. Generated unselected mesh intermediates were removed after
selection/assembly to meet the cap. No estimator is fine-tuned by these probes.

<a id="evidence"></a>
## 10. Reproducibility, source inventory and additional contrasts

### 10.1 Primary numerical evidence

- [Native notebook](/home/gasparyanartur/dev/pose-graph-diff/notebooks/investigations/diffusionregpose_diffusion_inspection.ipynb)
  and [inspection summary](/home/gasparyanartur/dev/pose-graph-diff/tmp/diffusionregpose_diffusion_inspection/summary.json).
- [Verified native replay results](/home/gasparyanartur/dev/pose-graph-diff/tmp/diffusionregpose_state_sensitivity_verified/results.json)
  and [native manifest](/home/gasparyanartur/dev/pose-graph-diff/tmp/diffusionregpose_state_sensitivity_verified/manifest.json).
- [Spatial attention audit](/home/gasparyanartur/dev/pose-graph-diff/tmp/diffusionregpose_attention_audit/spatial_summary.json)
  and [native attention reconstruction](/home/gasparyanartur/dev/pose-graph-diff/tmp/diffusionregpose_attention_audit/native_reconstruction.json).
- [Hidden-wrist native behavior](/storage/outputs/pose-graph-diff/diagnostics/h_control_information_20261023_epoch065/main_B_readouts/native_behavior.json),
  [registered readers](/storage/outputs/pose-graph-diff/diagnostics/h_control_information_20261023_epoch065/main_B_readouts/results.json)
  and [all-family map analysis](/storage/outputs/pose-graph-diff/diagnostics/h_control_information_20261023_epoch065/main_B_readouts/secondary_all_family_maps.json).
- [Depth readers](/storage/outputs/pose-graph-diff/diagnostics/h_control_information_20261023_epoch065/main_C_readouts/results.json)
  and [depth shortcut controls](/storage/outputs/pose-graph-diff/diagnostics/h_control_information_20261023_epoch065/main_C_readouts/image_audits.json).
- [Preliminary intervention summaries](/home/gasparyanartur/dev/pose-graph-diff/tmp/image_information_protocol/summary.json)
  and [robustness checks](/home/gasparyanartur/dev/pose-graph-diff/tmp/image_information_protocol/robustness.json).
- [Frozen V24 training experiment](/storage/outputs/pose-graph-diff/results/poseflow_v24_matched/20261022_192822/),
  [probe provenance](/home/gasparyanartur/dev/pose-graph-diff/tmp/h_control_information_suite/provenance.json)
  and [cached final 32-image audit](/home/gasparyanartur/dev/pose-graph-diff/tmp/h_control_information_suite/final_cached_fixed32_audit.json).

### 10.2 Checkpoint identity

| Checkpoint | SHA256 |
|---|---|
| Native reduced DiffusionRegPose epoch 79 | `97ee195be5f944f3cea424eb188a310277bc0bdee529e87be86fddbf483b5409` |
| V24 control, probe epoch 65 | `21e185e32cd069a4a839ad0a0d7d7924443318684a1c91dfa882d62043dfa715` |
| V24 h, probe epoch 65 | `3525f691a74bdb56a2de0f1335985f53588961646f30b1e4f3b6834fab6d4596` |
| V24 shared pretrained initialization | `8a5f23f90656d1eef09abd4b53a88a9ae8851bf696cea3cd21c408b6e41858cd` |
| V24 control, final epoch 79 | `e69a37efd6d205a855039cb81275db1c1706b48186b0dfe6648700b1f4d37e75` |
| V24 h, final epoch 79 | `d8760903c4cfb3807abf54bc30de702ccf20f808dd4bbb17ae16fa33ed297b0e` |

The native checkpoint/config, annotations, ordered IDs, source hashes and settings
are in its manifest. V24 source/configs remain frozen under the training root;
each final evaluation records its checkpoint/config/annotation hashes and IDs.
The exact pre-repair synthetic protocol remains in the raw execution evidence
under `tmp/h_control_information_suite/protocol.v1.md`; current construction,
source snapshots and audits remain alongside numerical artifacts. This is
reproducibility evidence, not another current report.

### 10.3 Complete registered depth contrasts

| Contrast | Difference, percentage points [95% interval] |
|---|---:|
| h final wrist vs control final wrist | 0.0 [0.0, 0.0] |
| Control backbone vs initialization | 0.0 [0.0, 0.0] |
| Control encoder vs initialization | 0.0 [0.0, 0.0] |
| Control first-pose wrist vs initialization | +1.8 [−0.6, 3.9] |
| Control final-pose wrist vs initialization | +5.5 [3.5, 7.6] |
| h backbone vs initialization | 0.0 [0.0, 0.0] |
| h encoder vs initialization | 0.0 [0.0, 0.0] |
| h first-pose wrist vs initialization | +4.3 [2.5, 6.2] |
| h final-pose wrist vs initialization | +5.5 [3.5, 7.6] |
| Wrist + h vs wrist + elbow, inside h model | +0.2 [0.0, 0.6] |
| Wrist + h vs wrist + shuffled h | 0.0 [0.0, 0.0] |

All ordinary features have 256 channels. Combined h/context controls have 512
inputs, with a separate 768-input shoulder/elbow/wrist comparison retained in
raw results. Initialization/stage contrasts are descriptive; selecting a best
cell does not create a new primary result. Held-out camera tests apply to depth;
hidden-arm yaw remains ±5° across all splits. Neck/head rig anchors complete the
14-joint projection check but are not claimed to match CrowdPose head semantics;
synthetic AP uses the common 12 limb joints only.

### 10.4 Source reports merged and removed

The following eight unique source reports, the previous version of this report,
and six duplicate archived copies are consolidated here. Former redirect stubs
and the obsolete archive index are also removed. The notebook, raw evidence,
checkpoints, source snapshots and generated result files are preserved.

| Original report | Material integrated here |
|---|---|
| `2026-10-23-discriminating-study-design.md` | Research questions, distinction between behavior/accessibility, construction and control logic; sections 3, 5 and 8. |
| `2026-10-23-h-control-test-suite.md` | Model contracts, geometry, splits, readers, admission gates, margins, repair history and reproducibility; sections 2–3, 5, 9–10. |
| `2026-10-23-image-information.md` | Real-image interventions, numerical contrasts and limits; section 6. |
| `2026-10-23-information-suite-pilot.md` | Ownership construction, failed controls and matching correction; sections 5.6 and 9. |
| `2026-10-24-diffusionregpose-state-sensitivity.md` | Native state sensitivity, clamp observations, route interventions, verification and compensation limits; sections 4 and 9–10. |
| `2026-10-24-information-probes.md` | Final admission, construction repairs, execution contracts and shortcut warning; sections 3, 5 and 9. |
| `2026-10-24-information-probes-results.md` | Completed hidden/depth tables, contrasts, coverage and controls; sections 5 and 10. |
| `2026-10-24-v24-completion-review.md` | Final training/full-test results, loss/split interpretation and all-family sensitivity analysis; sections 5, 7 and 10. |

[Merge provenance](/home/gasparyanartur/dev/pose-graph-diff/docs/reports/assets/diffusionregpose-investigation/merge-provenance.json) records source-report hashes and
the notebook identity. [Figure provenance](/home/gasparyanartur/dev/pose-graph-diff/docs/reports/assets/diffusionregpose-investigation/figure-provenance.json) records
original copied figures and the evidence for assembled/summary plots. Figures
are stored with this report rather than relying on temporary render exports.


### 10.5 Subsequent implementation: explicit DINO region supervision for h

The next approved iteration implements V25, an auxiliary decoder from final h to
7×7×384 frozen DINOv3 ViT-S/16 features of the pose-matched GT box. It uses the
full augmented image's teacher map, shared final pose assignments and mean
per-cell cosine loss (weight1.0). Teacher weights are fixed; the decoder has no
image-feature bypass. This adds341,632 student parameters (42,695,306 total).
Teacher and auxiliary decoder do not execute during ordinary pose sampling.

A finite wiring smoke imported all578 shared V24 h model tensors and left only
seven decoder state entries fresh. It used two augmented trainval images,
110262 and119927, with sizes373×256 and296×256, for three optimizer updates.
Raw auxiliary loss was1.0403,0.9452,0.8659, with11–12 positive assignments and
no skipped ROIs. Decoder, h writer and h image initializer received nonzero finite
gradients. FP32 and BF16 backward passes were finite; full-width checkpoint
restore reproduced pose predictions exactly. Teacher weights were absent from
optimizer and checkpoint. Warm steps took about0.40–0.42s at batch2 and peak
allocated memory was about1.36GB, under concurrent GPU use; these are smoke
measurements, not an isolated throughput benchmark.

Correct-versus-wrong-box similarity remained near zero in this tiny smoke, so
falling loss is not evidence that h has learned person-specific content. No AP
claim or full training run follows from these updates. Long-run initialization
and a matched training comparison remain a later decision.

[Smoke measurements](/home/gasparyanartur/dev/pose-graph-diff/tmp/v25_h_dino_smoke/result.json)
and [resolved smoke configuration](/home/gasparyanartur/dev/pose-graph-diff/tmp/v25_h_dino_smoke/config.yaml).



### 10.6 Training the explicit h target: one new matched run

The approved longer experiment started on 24 October. It asks whether supervising
h to predict its matched person's DINO grid improves pose estimation over the
same h model trained only with pose losses. There is **one new run: 80 epochs, seed 42**;
the completed V24 h seed 42 run supplies the comparison. V25 starts from the
original EDPose-derived initialization and original fresh h branch, rather than
continuing the epoch 79 model used in the wiring smoke.

Before launch, all 578 existing student tensors and the CPU random-generator state
matched the saved original initialization exactly. The new 341,632-parameter grid
decoder is the only fresh student component. The original config is reused with
only the V25 class, pinned teacher/grid/weight fields and output names changed.
Existing data/optimizer/sampler source matches the frozen V24 source except for
the reviewed optional context export, shared final-assignment interface and V25
factory hooks. No new control or seed sweep is queued.

| Setting | Fixed protocol |
|---|---|
| Training data | CrowdPose trainval, 12,000 images; same augmentation recipe |
| Training budget | 80 epochs, 120,000 optimizer updates, seed 42 |
| Batch/precision | Batch 4, accumulation 2; 32-bit recipe, original TF32 defaults preserved |
| Optimizer | Original AdamW; LR 0.0002/backbone 0.0001, drops at epochs 31/61, clip 0.1 |
| Auxiliary target | Frozen pinned DINOv3 ViT-S/16, GT-box 7×7×384 grid, weight 1 |
| Monitoring | Per-step auxiliary loss/cosine/variance/wrong-box gap; original fixed 32-image test AP |
| Primary comparison | Final epoch 79 K1 pose AP on all 8,000 held-out test images |
| Completed baseline | V24 h 55.2949 AP under the same final evaluator |

All 2,000 validation images are in the training set, so validation remains a fit
diagnostic. The fixed 32-image monitor does not select checkpoints, stop training or tune the loss weight. The
worker queues the unchanged canonical full-test evaluator after a successful fit;
it verifies identical test IDs, seeds, resize, precision and evaluator metadata
before writing the final AP difference. The one paired seed is pilot evidence.

The new loss can fall by predicting generic region features. Therefore the live
correct-versus-wrong-box similarity and prediction variance remain separate from
pose AP; a declining feature loss alone will not establish useful person content.
The existing multistep evaluation continues independently.

[Live comparison](http://127.0.0.1:33332/?darkMode=true#scalars&_smoothingWeight=0),
[run status](/storage/outputs/pose-graph-diff/results/poseflow_v25_h_dino/20261024_181523/status.json),
[persistent output](/storage/outputs/pose-graph-diff/results/poseflow_v25_h_dino/20261024_181523/launch.log),
[frozen protocol and hashes](/storage/outputs/pose-graph-diff/results/poseflow_v25_h_dino/20261024_181523/manifest.json),
and [resolved configuration](/storage/outputs/pose-graph-diff/results/poseflow_v25_h_dino/20261024_181523/h_dino_seed42/.hydra/config.yaml).


<a id="multistep"></a>

## 11. Follow-up: matched multi-step inference

**Complete.**

The original matched V24 evaluation used K1 only. This follow-up compares final
epoch-079 control/h checkpoints at K1/K2/K3/K5 on all 8,000 held-out test images.
K1 reuses the verified final baseline. Seeds, resize, precision and native NMS
match that baseline. h resets on each forward, as in the trained model; no
cross-call carry is added. Both estimators remain frozen.

| Steps | Control pose AP | h pose AP | h minus control, AP points |
|---|---:|---:|---:|
| 1 | 54.9141 | 55.2949 | +0.3808 |
| 2 | 54.9049 | 55.2969 | +0.3920 |
| 3 | 54.9044 | 55.2960 | +0.3915 |
| 5 | 54.9047 | 55.2987 | +0.3940 |

The diagnostic calls the frozen existing endpoint Euler sampler with an explicit
step count, leaving the original K1-only wrapper/source unchanged. On eight
preflight images per arm, explicit K1 raw outputs and final records matched
the native wrapper and cached full-test predictions exactly. K2/K3/K5 timesteps,
fixed sources, finite outputs and per-forward h reset passed checks.

Changes from K1 (AP points):

- control: K2: -0.0092; K3: -0.0097; K5: -0.0094.

- h: K2: +0.0020; K3: +0.0010; K5: +0.0038.

These models were trained single-call; this tests inference-time refinement
with that training, not learned cross-call memory or multi-step training. Equal
step counts also use different compute between arms. Single training seed only.

[Predictions, progress, timings and comparison](/storage/outputs/pose-graph-diff/diagnostics/v24_multistep_epoch079_20261024/).

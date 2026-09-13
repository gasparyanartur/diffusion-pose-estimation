# FlowDet → DiffusionRegPose: implementation reference

Last checked: 2026-09-13

## Scope and evidence status

This report specifies the smallest faithful conversion of DiffusionRegPose's diffusion process into the flow process defined by FlowDet. It separates:

- **Paper fact** — stated by FlowDet, DiffusionDet, or DiffusionRegPose.
- **Code fact** — present in the official public implementation.
- **Derived mapping** — follows mechanically after replacing FlowDet boxes with DiffusionRegPose pose coordinates.
- **Implementation choice** — not fixed by the papers and therefore must be made explicit in code/configuration.

There is no public FlowDet implementation linked by the paper as of the date above. Consequently, equations, training supervision, and stated inference behavior are authoritative; details absent from the paper cannot be recovered from official code.

Primary sources:

- [FlowDet paper, arXiv:2512.16771](https://arxiv.org/html/2512.16771v1)
- [DiffusionDet paper, arXiv:2211.09788](https://arxiv.org/abs/2211.09788)
- [DiffusionDet official repository](https://github.com/ShoufaChen/DiffusionDet)
- [DiffusionRegPose paper, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Tan_DiffusionRegPose_Enhancing_Multi-Person_Pose_Estimation_using_a_Diffusion-Based_End-to-End_Regression_CVPR_2024_paper.pdf)
- [DiffusionRegPose official repository](https://github.com/tjiiv-cprg/DiffusionRegPose) (audited at commit `9746abc8fe9e909ea6c91fdb448e9e14a79e836c`)

## Bottom line

FlowRegPose is not a new pose architecture. It retains the ED-Pose-derived image backbone, encoder, human decoder, human-to-keypoint expansion, keypoint attention, pose/box/class heads, matching, and task losses. The replaced subsystem is:

> pose corruption + time semantics + clean-pose-to-update conversion + iterative sampler

The exact process is:

Training:

\[
x_0\sim\mathcal N(0,I),\qquad t\sim U[0,1],\qquad
x_t=(1-t)x_0+t x_1,
\]

\[
(\hat x_1,\hat b,\hat c)=f_\theta(I,x_t,t).
\]

The decoder is supervised using the existing pose, box, and classification losses on its clean endpoint/task predictions. FlowDet does **not** add a velocity-MSE loss to the detector.

Inference:

\[
\hat u_t=\hat x_1-x_0,
\qquad
x_{t+\Delta}=x_t+\Delta\hat u_t.
\]

The initial source `x0` is fixed and retained across all Euler steps. It is not replaced by the current state `xt`, and the update is not `(x1_hat - xt)/(1-t)`. This is FlowDet Eq. 7.

## Notation and state space

| Symbol | Detection meaning | Pose mapping | Shape |
|---|---|---|---|
| `I` | image | image | `[B,C,H,W]` |
| `N` | proposal count | candidate-pose count | scalar |
| `K` | — | body-keypoint count | scalar |
| `x0` | source boxes from prior | source pose coordinates from prior | `[B,N,2K]` |
| `x1` | padded clean target boxes | padded clean target pose coordinates | `[B,N,2K]` |
| `xt` | interpolated boxes | interpolated pose coordinates | `[B,N,2K]` |
| `t` | continuous flow time | continuous flow time | `[B]`, float |
| `x1_hat` | predicted clean boxes | predicted clean pose XY | `[B,N,2K]` |
| `b_hat` | same as transported state in FlowDet | separately predicted human box | `[B,N,4]` |
| `c_hat` | class logits | person/class logits | `[B,N,C]` |

DiffusionRegPose's keypoint output also contains visibility/confidence components. Only the first `2K` normalized XY coordinates are the current generative state; visibility and human boxes remain ordinary decoder outputs.

Coordinate-domain invariant:

- Task space: normalized coordinates, normally `[0,1]`.
- Transport space: scaled coordinates used by the generative process.
- `x0`, `x1`, `xt`, `x1_hat`, and velocity must all be in the **same transport space**.
- Conversion to `[0,1]` happens only at the transformer boundary; task outputs are converted back into transport space before the Euler update.

## Exact DiffusionDet → FlowDet delta

| Component | DiffusionDet | FlowDet |
|---|---|---|
| Source | Gaussian noise in the diffusion construction | explicit source `x0 ~ N(0,I)` |
| Target notation | clean state usually written `x0` | clean endpoint `x1` |
| Time | discrete index | continuous `t ~ U[0,1]` |
| Training path | `sqrt(alpha_bar_t) x_clean + sqrt(1-alpha_bar_t) eps` | `(1-t)x0 + t*x1` |
| Schedule | cosine beta/alpha schedule | none |
| Network output | clean endpoint | clean endpoint |
| Detector loss | standard set-prediction loss | standard set-prediction loss |
| Sampler conversion | infer noise from predicted clean state | `u_hat = x1_hat - x0` |
| Update | DDIM reverse step | forward Euler step |
| Direction | high diffusion time → zero | `0 → 1` |
| Default solver | DDIM | Euler |
| Multi-step extras | renewal and ensemble | renewal and ensemble retained |

FlowDet uses endpoint prediction specifically so the detector remains supervised by task-aligned set losses. Its decoder stages predict endpoint boxes and class labels; Eq. 7 algebraically recovers velocity for sampling. The paper's Appendix A specifies focal, L1, and gIoU set losses and no additional CFM velocity loss.

## FlowDet facts relevant to the port

### Path and endpoint parameterization

FlowDet Eqs. 4–7 define all essential mathematics:

\[
x_t=(1-t)x_0+t x_1,
\qquad
u_t=x_1-x_0,
\]

\[
\hat x_1=v_\theta(x_t,t),
\qquad
\hat u_t=\hat x_1-x_0,
\]

\[
x_{t+\Delta}=x_t+\Delta\hat u_t.
\]

Although the paper introduces conventional CFM velocity regression in Eq. 1, the detector itself is trained through endpoint/task losses. Do not add `MSE(u_hat, x1-x0)` when reproducing FlowDet's detector formulation.

### Prior, target padding, and pairing

The main FlowDet configuration uses:

- standard Gaussian prior;
- target-set padding with random samples from the source distribution;
- random source-target pairing;
- one independently sampled continuous time per image/sample;
- Euler integration.

The source and padded target arrays must have the same length `N_train`. Padding constructs transport endpoints; real object/pose annotations remain the semantic targets used by set matching and losses.

FlowDet tested Hungarian source-target coupling with center-distance, box-distance + gIoU, and InterpIoU costs. All were worse than random pairing in its detector experiments. This result concerns **transport-pair construction**, not the separate prediction-to-GT assignment used by the task loss.

### Time conditioning

FlowDet uses sinusoidal time embedding followed by FiLM-style conditioning in the decoder. Training samples continuous `t`; inference uses a linear schedule over `[0,1]`.

### Inference behavior

FlowDet uses Euler in its main experiments. With 100 proposals and a ResNet-50 backbone, it reports AP 43.2/44.7/45.5 for 1/2/3 steps versus DiffusionDet's 43.1/43.0/45.1. Higher-order solvers were not interchangeable drop-ins in the reported setup: Heun underperformed, while RK4 and capped DOPRI5 were numerically ineffective. This is empirical evidence for the published detector, not a theorem about pose transport.

During multi-step inference, FlowDet also states that it discards weak proposals, replenishes them from the prior, aggregates predictions from steps, and applies NMS. These are sampler/post-processing features rather than part of the defining flow equations.

## DiffusionRegPose implementation audit

The official code already has the correct endpoint-prediction interface for the port:

1. `model_predictions()` converts the current transport state to normalized keypoints and calls the transformer.
2. The decoder emits `pred_keypoints`, `pred_boxes`, and `pred_logits`.
3. The first `2K` predicted keypoint values are converted back to scaled transport coordinates as the clean-state estimate.
4. Diffusion-specific code then converts that estimate to predicted noise for DDIM.

The relevant source is [`models/diffusionregpose/diffusionregpose.py`](https://github.com/tjiiv-cprg/DiffusionRegPose/blob/9746abc8fe9e909ea6c91fdb448e9e14a79e836c/models/diffusionregpose/diffusionregpose.py): `q_sample` and target construction at lines 484–535, endpoint-to-noise conversion at 572–576 and 687–693, DDIM sampling at 748–837, and diffusion-buffer construction at 964–1010.

The transformer time embedding is already algebraically compatible with floating-point time: it multiplies the supplied tensor by sinusoidal frequencies without an integer lookup. See [`transformer.py`, lines 28–40 and 210–220](https://github.com/tjiiv-cprg/DiffusionRegPose/blob/9746abc8fe9e909ea6c91fdb448e9e14a79e836c/models/diffusionregpose/transformer.py#L28-L40). Its semantics and numerical range still change when moving from diffusion indices to `[0,1]`, so checkpoint compatibility must not be assumed.

### Material paper/code discrepancies

These must be resolved before treating the public repository as a faithful experimental baseline:

1. **Training time is forced to zero in code.** `prepare_diffusion_concat()` samples a random integer timestep at line 501, but line 527 overwrites it with `tensor([0])` before `q_sample()`. Thus the audited commit does not actually train over the advertised diffusion-time range.
2. **Padding differs.** The paper reports empty-pose padding as its default. The code concatenates `torch.randn` placeholders when `num_gt < num_proposals` (lines 508–513).
3. **Configured values are overwritten.** `box_renewal` and `use_ensemble` exist in config, but the model assigns both to `True` internally at lines 981–982.
4. **Renewal branch polarity appears inverted.** Filtering and replenishment are guarded by `if not self.box_renewal` in the sampler (lines 778 and 806), so the hard-coded `True` disables the shown renewal block.

For a flow port, do not preserve these behaviors accidentally. Continuous random time is required by FlowDet; padding and renewal must be explicit named choices.

## Function-level replacement map

| Current DiffusionRegPose component | Required flow form | Status |
|---|---|---|
| `cosine_beta_schedule()` | remove | diffusion-only |
| beta/alpha/posterior buffers | remove | diffusion-only |
| `q_sample()` | remove | replaced by linear interpolation |
| `prepare_diffusion_concat()` | `prepare_flow_concat()` | rewrite |
| `prepare_targets()` | return `xt`, `x0`, continuous `t` | rewrite interface |
| integer `t` | floating `t in [0,1]` | semantic/type change |
| transformer time MLP | retain module | accepts float; retrain under new range |
| `model_predictions()` | return scaled `x1_hat` and task outputs | simplify |
| `ModelPrediction(pred_noise, pred_x_start)` | `FlowPrediction(pred_x1)` or direct tensor | simplify |
| `predict_noise_from_start()` | remove | replaced by `x1_hat - x0` in sampler |
| `ddim_sample()` | `flow_sample()` | rewrite |
| reverse DDIM schedule | forward linear Euler schedule | rewrite |
| backbone/encoder | unchanged | retain |
| human and pose decoders | structurally unchanged | retain |
| pose/box/class heads | structurally unchanged | retain |
| matcher and task losses | unchanged initially | FlowDet retains task supervision |
| DN machinery | independent of flow state | retain unless separately refactored |
| pose NMS/post-processing | independent of path | retain |

## Reference training algorithm

The following is the direct pose-state substitution of FlowDet.

```python
def prepare_flow_concat(gt_pose_xy, N, K, scale, device):
    # gt_pose_xy: normalized real GT coordinates [M, 2K]
    # Build N endpoint slots in normalized or transport space according to
    # one explicit padding policy. Never mix coordinate domains.
    x1_norm = pad_target_set(gt_pose_xy, N)          # [N, 2K]
    x1 = (2.0 * x1_norm - 1.0) * scale              # transport space

    x0 = torch.randn(N, 2 * K, device=device)        # transport space
    t = torch.rand((), device=device)                # one time per image
    xt = (1.0 - t) * x0 + t * x1

    xt_norm = (xt / scale + 1.0) / 2.0
    return xt_norm, x0, t
```

Batch-level forward:

```python
features = image_encoder(images)
xt_norm, x0, t = prepare_flow_targets(targets)

out = transformer_and_heads(features, xt_norm, t)
# out: pred_keypoints, pred_boxes, pred_logits, auxiliary outputs

loss = existing_diffusionregpose_task_criterion(out, real_targets)
```

Required invariants:

- `t.dtype` is floating point.
- `t.shape == [B]` at the transformer boundary.
- each batch item gets its own sampled `t`; all `N` proposals in that image share it, matching both FlowDet and current DiffusionRegPose batching semantics;
- `x0` is stored only if needed for diagnostics during training; no velocity target is required by the published endpoint-loss formulation;
- padded transport slots do not become fake positive pose annotations in the task criterion;
- existing auxiliary decoder-stage losses stay active.

### Padding policy

Two different source facts exist:

- FlowDet pads target boxes with samples from the source distribution.
- DiffusionRegPose paper selects empty-pose padding; its public code instead uses random placeholders.

Therefore padding is not a mechanical no-op. A strict **FlowDet-method port** uses prior-sampled target padding. A strict **DiffusionRegPose-controlled comparison** keeps the verified baseline padding policy fixed. Record the choice in configuration and report it; do not bury it inside `prepare_flow_concat()`.

When using prior padding, sample directly in transport space and convert once. The public DiffusionRegPose code currently creates Gaussian placeholders before applying its `[0,1] → [-scale,scale]` transform, which does not yield a standard-Gaussian endpoint in transport space.

## Reference inference algorithm

For `S` equal Euler steps:

```python
@torch.no_grad()
def flow_sample(features, B, N, K, S, scale):
    x0 = torch.randn(B, N, 2 * K, device=features.device)
    x = x0.clone()
    dt = 1.0 / S
    step_outputs = []

    for i in range(S):
        t = torch.full((B,), i / S, device=x.device, dtype=x.dtype)
        x_norm = ((x.clamp(-scale, scale) / scale) + 1.0) / 2.0

        out = transformer_and_heads(features, x_norm, t)
        x1_hat_norm = out["pred_keypoints"][..., :2 * K]
        x1_hat = (2.0 * x1_hat_norm - 1.0) * scale

        velocity = x1_hat - x0
        x = x + dt * velocity
        step_outputs.append(out)

    return x, step_outputs
```

This loop contains the non-negotiable FlowDet detail: `velocity = x1_hat - x0`, where `x0` is the original source tensor. Using `x1_hat - x` instead silently implements a different flow.

At one step, `x = x0 + (x1_hat-x0) = x1_hat`, which is a useful unit test.

### Renewal bookkeeping

If proposal renewal is enabled, proposal identity includes both `(x, x0)`. When a weak slot is removed, remove the same indices from both tensors. A replenished slot receives a new prior sample and the corresponding new fixed source anchor. Failing to update `x0` corrupts Eq. 7.

The FlowDet paper states that renewal occurs but does not specify in enough detail how newly injected samples are time-aligned at an intermediate `t`, nor does it publish code. Treat late-step injection semantics as an explicit implementation choice and isolate renewal in the first correctness tests. The base flow equations themselves are fully specified without renewal.

### Which pose tensor is returned?

The ODE state after the last update is `x(t=1)`. The decoder also emits an endpoint estimate at every call. FlowDet states that predictions across steps are ensembled, but does not provide code-level pseudocode fixing whether a non-ensemble API should expose the integrated terminal state, the last endpoint estimate, or both.

Expose both internally:

- `terminal_state = x` for verifying the integrator;
- `last_endpoint = x1_hat` and `step_outputs` for task heads and ensemble/post-processing.

Do not make this distinction implicit. For `S=1` the two are identical; for `S>1` they need not be.

## What remains unchanged

The following components do not depend on diffusion-versus-flow path mathematics:

- image normalization and augmentation;
- backbone and multi-scale feature extraction;
- deformable encoder;
- human-box decoder;
- human-to-keypoint query expansion;
- K-SA/K-CA pose-image interaction;
- pose, visibility, human-box, and class prediction heads;
- prediction-to-GT matching;
- focal, box, and keypoint losses;
- auxiliary decoder-stage supervision;
- output coordinate rescaling and pose NMS.

Human boxes should not be added to the transported state merely because FlowDet transports boxes. In DiffusionRegPose, pose XY is the generative state and human boxes are jointly predicted conditioning outputs. Flow transports the variable that diffusion previously transported.

## Delete list

After the flow path is verified, remove:

- `cosine_beta_schedule`;
- `extract` if unused elsewhere;
- `q_sample`;
- `predict_noise_from_start`;
- beta, alpha-cumulative-product, reciprocal-alpha, and posterior buffers;
- DDIM `alpha`, `alpha_next`, `sigma`, `c`, and inferred-noise calculations;
- diffusion-only config fields `timesteps`, `ddim_sampling_eta`, and `objective`.

Retain a single `sampling_steps`/`flow_steps` field. Training has no discrete total-timestep count.

## Verification checklist

### Mathematical unit tests

1. `t=0` gives `xt == x0`.
2. `t=1` gives `xt == x1`.
3. With an oracle endpoint predictor returning fixed `x1`, any number of equal Euler steps reaches `x1` to numerical tolerance.
4. With `S=1`, terminal state equals predicted endpoint.
5. Shuffling target slots changes only the source-target coupling, not the semantic GT set passed to the loss.
6. Filtering/renewal applies identical indices to `x`, `x0`, and proposal-aligned metadata.

### Interface tests

1. transport tensors are `[B,N,2K]`;
2. transformer input coordinates are in the expected normalized range;
3. timestep tensor is float `[B]`;
4. `pred_keypoints[..., :2K]` is converted back to the same transport domain as `x0` before subtraction;
5. visibility channels are never included in the velocity;
6. box/class/keypoint losses match the baseline on a fixed decoder output;
7. one-step inference has no unused terminal re-evaluation.

### Controlled comparison

To attribute differences to the generative path, hold fixed:

- ED-Pose initialization;
- backbone and decoder depth;
- query/proposal count;
- target completion and padding policy;
- matcher and all loss weights;
- training schedule and augmentation;
- post-processing and pose NMS;
- random seeds and evaluation split.

Compare diffusion versus flow at equal decoder-call counts: one DDIM step against one Euler step, then two against two, etc. Report both accuracy and wall-clock latency because the backbone is computed once while the decoder is repeated.

## Minimal implementation sequence

1. Add `prepare_flow_concat()` returning normalized `xt`, transport-space `x0`, and float `t`.
2. Make time input genuinely continuous; remove the forced-zero line.
3. Simplify `model_predictions()` to return transport-space `x1_hat` plus the existing output dictionary.
4. Add a no-renewal, no-ensemble Euler sampler that retains `x0`.
5. Pass the mathematical and interface tests above.
6. Run one-step parity/sanity evaluation.
7. Add multi-step output aggregation.
8. Add renewal only after defining and testing its intermediate-time semantics.
9. Delete the unused diffusion schedule and DDIM code.

## Compact architecture summary

```text
image ──► backbone/encoder ─────────────────────────────────────────────┐
                                                                      │
x0 ~ N(0,I) ─┐                                                       │
              ├─► xt=(1-t)x0+t*x1 ─► normalize ─► ED-Pose decoder ◄──┘
x1 padded GT ─┘                                  │
                                                 ├─► x1_hat pose XY
                                                 ├─► human boxes
                                                 └─► class/visibility

training:  existing task losses on endpoint/task outputs
inference: u_hat=x1_hat-x0;  x←x+Δt*u_hat; repeat decoder call
```

The defining implementation fact is simple: DiffusionRegPose already predicts a clean pose endpoint. FlowDet changes how the decoder input is constructed and how that endpoint is used between repeated calls; it does not require replacing ED-Pose or inventing a velocity head.

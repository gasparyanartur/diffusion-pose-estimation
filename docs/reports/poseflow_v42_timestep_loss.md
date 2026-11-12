# V42 box-center timestep-dependent geometry loss

The new arm is exactly corrected `benchmark_poseflow_v42_uniform_t` model and
source construction, with a training-objective change. It is not paired V42,
image-confidence V42, JEPA, or an ordered-query model. Fresh ED-Pose initialization,
seed42, full80 epochs, FP32/batch4/accum2, original optimizer/augmentation/scheduler
and native coordinate diffusion/DDIM remain unchanged.

For each image, u=t/(T-1), a_box=.1+.8u, a_pose=.9-.8u. The objective is
`L_classification + 2*a_box*L_native_box + 2*a_pose*L_native_pose`.
High t999 is the beginning of reverse sampling:90% box/10% pose emphasis.
Low t0 is its clean end:10% box/90% pose. Scale2 preserves native coefficients at
midpoint and average coefficients under uniform timestep sampling. Ratios are
multipliers of existing weighted terms, not a guarantee of their actual relative
magnitudes or gradient norms. The schedule is unrelated to training epoch.

Box geometry includes native L1/GIoU final, auxiliary, encoder proposal and DN
box terms, with existing native branch coefficients. Pose geometry includes all
supervised keypoint L1/OKS terms. All focal/classification terms, including DN,
remain unchanged. Native matching costs and Hungarian algorithm are unchanged.
No loss-weight object or matcher parameters are mutated.

`TimestepLossCriterion` computes native assignments once per output and reuses
them for an unweighted native diagnostic and the weighted objective. Per-image
geometry contributions are weighted before summing with the original global GT
normalizer. DN contributions retain their separate query-count denominator.
A scoped, cleared-after-call assignment cache avoids a second matching pass.
Native unit-weight loss/gradient parity and a mixed-timestep oracle check this
contract, including encoder, aux and DN branches with unequal image person counts.

Training passes actual sampled t[B] to the criterion. It logs unweighted native
total, weighted total, mean box/pose factors and low/mid/high-t factor diagnostics.
Validation includes the same scheduled objective and the comparable unweighted
native diagnostic. Model/inference APIs, parameter tensors and sampling remain
unchanged. Checkpoints add policy metadata and global Python/NumPy/Torch/CUDA RNG;
wrong-policy restoration is rejected. No dataset-prefetch replay claim is made.

The testable hypothesis is that box emphasis at high noise and pose emphasis at
low noise improve final geometry and multi-call behavior. Classification should
remain numerically identical for identical predictions, as should all Hungarian
assignments. More accurate generated trajectories are not assumed from loss
weighting alone. Observe K1/K4/K8 AP, raw and weighted losses, timestep coverage,
and box/pose quality separately.

## Usage and evaluation

Hydra preset: `benchmark_poseflow_v42_timestep_loss`.
Durable entrypoint: `scripts/training/train_poseflow_v42_timestep_loss.py --root
<prepared-root>`; the manifest pins source/config/input and initial-model hashes.
Full80 starts fresh, without a trained V42 warmstart or short pilot cutoff.

The intentionally retained historical protocol uses trainval training,
overlapping validation-loss diagnostics, fixed32 TEST development AP each epoch,
and final full8000 TEST at epoch79. This is development exposure, not an untouched
benchmark. Compare matched ages and final full-TEST results with the uniform-t
box-center baseline. Existing runs and six AP pins are retained in TensorBoard.

## Verified launch

2026-11-10: fresh full80 run `/home/gasparyanartur/outputs/pose-graph-diff/poseflow_v42_timestep_loss/20261110_222822`, tmux
`poseflow-v42-timestep-loss`, PID348105. Verified50 optimizer updates with finite
loss, actual uniform mixed timesteps and schedule factors.31 focused/historical
checks and9 final overlapping focused checks pass. Native CUDA restore predicts
exactly; controlled next-update max state difference1.19e-7. Initial model hash
exactly matches the original corrected V42 initialization.844 sources/5 inputs
verified; no trained checkpoint reused. TensorBoard stream
`V42-box-center-timestep-loss/.` has native and scheduled train scalars. First
learned AP follows epoch1; no learned improvement is claimed at launch.

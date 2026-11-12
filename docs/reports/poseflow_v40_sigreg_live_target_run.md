# V40-SigReg live-target: full training run

Fresh seed-42 training launched at 2026-11-03 15:22 Europe/Stockholm, following the user's
review of the completed two-epoch correction pilot and explicit request to start
a run. This is a fresh 80-epoch experiment, not a pilot resume.

Run: `/storage/outputs/pose-graph-diff/results/poseflow_v40_sigreg/20261103_151941`

- Trainer PID: `365331`; persistent tmux session: `poseflow-v40-sigreg-live-target`.
- Frozen model/configuration match the successful pilot; only the Trainer limit
  changes from 2 to 80. The scheduler horizon remains 80 epochs.
- Live corruption and clean-target gradients, SIGReg 0.05 on matched clean codes,
  identity final clean normalization, no EMA. Model/proposal/loss code unchanged.
- Original data/augmentation/AdamW/scheduler, FP32, resize 256/427, batch size 4,
  accumulation 2, gradient clipping 0.1, proposals 200 and latent slots 50.
- Fixed32 held-out bbox evaluation every epoch with K1/K4, all 50 outputs and no
  NMS. Full8000 K1/K4 evaluation follows the epoch-79 final checkpoint.
- Clean-code RMS >10 or a numerical failure stops training with a forensic
  checkpoint. Guard checks do not alter valid updates.

Startup verification passed: identical initial model hash and 342-tensor ED-Pose
initialization coverage 100%, 609 frozen source files and five input/configuration
hashes verified, live-target diagnostic zero, 66 scalar tags and finite ongoing
optimizer progress. At update 290, matched clean-code RMS was
0.3905 and batch loss 3.9441. These are launch-health
observations; this run has no evaluated learned result at verification.

TensorBoard stream: `V40-SigReg-live-target` at
[the detection dashboard](http://lynx.tail481642.ts.net:33330/). All 19 previous
streams, history and AP pins remain. The failed detached-target run and successful
pilot artifacts/pointers are preserved. A separate active-run pointer is stored
at `tmp/v40_sigreg/live_target/full_run/run_root.txt`.

Manifest, source, resolved configuration, initialization provenance, live status,
stdout, TensorBoard/CSV logs, per-epoch probes/evaluations and checkpoints are
under the run directory. Initial verification is saved as `live_verification.json`.
Existing 51 focused checks and native CUDA/strict-restore verification apply to
unchanged production code; the launch callback guard smoke also passes.

The [two-epoch pilot](poseflow_v40_sigreg_live_target.md) avoided early scale
explosion, with epoch-1 fixed32 AP 13.15/5.34 (K1/K4). The K4 gap remains unresolved,
and long-run stability/performance are not established by startup checks.


Subsequent observed status during V41 verification: the run reached epoch 13,
update 20,466, then stopped at the configured clean-code scale guard (RMS
13.35154 >10). Its forensic checkpoint and all prior artifacts are preserved.
No process was interrupted or restarted by the V41 implementation. The initial
launch checks above are historical health observations, not sustained stability.

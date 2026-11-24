# GroupPose on CrowdPose

The reduced preset uses ResNet-50, three encoder layers, three pose decoder
layers, 50 person groups, 14 joints, and the same 256/427 aspect-preserving
resize as reduced PoseFlow. It preserves GroupPose's native pose proposals,
group attention, Hungarian assignment, and classification, keypoint L1 and
OKS losses. The comparable PoseFlow preset uses one box stage and two pose
stages, and trains with OTA.

## Fixed32 TEST verification

All rows use the same [32-image TEST manifest](../../results/all_run_audit/20261116_203800/fixed32_test.json),
one CUDA forward pass per image, native GroupPose score selection, and
`crowdposetools` pose evaluation. Scores are in AP points.

| Checkpoint | Encoder / decoder | Resize | AP | AP50 | AP75 | AP easy / medium / hard |
|---|---:|---:|---:|---:|---:|---:|
| Released CrowdPose Swin-L | 6 / 6 | 800 / 1333 | 71.40 | 86.85 | 78.42 | 75.70 / 70.14 / 73.17 |
| Same released Swin-L | 6 / 6 | 256 / 427 | 47.04 | 64.52 | 47.30 | 50.58 / 45.02 / 55.93 |

The [released checkpoint](https://drive.google.com/file/d/1sz0QkHsL1LLWG0hx4mhgiDvB4_fr0LJu/view)
is the official 14-joint Swin-L model at epoch 79. All 752 checkpoint entries
loaded strictly. At 800/1333, every native prediction record and AP, AP50, and
AP75 matched a replay using the upstream data and evaluation code. Each ledger
contains the 32 expected image IDs and 1,600 native prediction records. See the
[parity check](../../results/grouppose_baseline/20261118/upstream_parity.json)
and [ledger verification](../../results/grouppose_baseline/20261118/ledger_verification.json)
for the exact values and provenance.

The reduced ResNet-50 model passed a one-update train, strict model restore,
and fixed32 inference smoke. The update used seed 42, random initialization,
and one TRAIN image (100000). Weighted loss was 120.7566 and the gradient norm
before clipping was 135.1559. This verifies the training and evaluation paths.
GroupPose's reported metrics are pose-only.

## Reproduce the smoke

Use a clean checkout of the
[official GroupPose source](https://github.com/Michel-liu/GroupPose/tree/127fe0e2e7b54d7a43420015a2e90e456a8ea1fb)
and the project Python/CUDA environment. The runner defaults to the local
ABI-compatible DiffusionRegPose CUDA deformable attention extension. Use
`--ops-root` to supply a different directory containing the upstream-compatible
extension. GroupPose uses its own attention modules around this kernel.

```bash
git clone https://github.com/Michel-liu/GroupPose.git tmp/grouppose/upstream
git -C tmp/grouppose/upstream checkout --detach 127fe0e2e7b54d7a43420015a2e90e456a8ea1fb

venv/bin/python -m scripts.evaluation.evaluate_grouppose smoke-train \
  --upstream tmp/grouppose/upstream \
  --annotations data/crowdpose/annotations/crowdpose_train.json \
  --image-root data/crowdpose/images \
  --output tmp/grouppose/example_smoke.pth

venv/bin/python -m scripts.evaluation.evaluate_grouppose evaluate \
  --upstream tmp/grouppose/upstream \
  --checkpoint tmp/grouppose/example_smoke.pth --checkpoint-role smoke \
  --annotations data/crowdpose/annotations/crowdpose_test.json \
  --image-root data/crowdpose/images \
  --subset-manifest results/all_run_audit/20261116_203800/fixed32_test.json \
  --output results/grouppose_baseline/example_smoke
```

The runner also accepts trained reduced checkpoints with the upstream
`model` state dictionary and `--checkpoint-role trained`. Omit the subset
manifest for full annotation coverage. For released Swin-L weights, use
`--preset conf/group_pose/released_crowdpose.yaml --checkpoint-role released`.
The `--shortest-side` and `--max-size` overrides affect preprocessing only.

## Approved TRAIN+VAL run

Run `20261118_004914` trains with all 12,000 official TRAIN+VAL images and
evaluates the shared fixed32 TEST subset at initialization and after every
epoch. Its evaluated IDs exactly match the PoseFlow V56 TEST continuation.
This is a development benchmark used for checkpoint selection.

The reduced model starts from ImageNet ResNet-50 with a fresh GroupPose
transformer. The recipe uses seed 42, native Hungarian matching, FP32,
effective batch 8, and native AdamW parameter groups. An 80-epoch schedule
drops the learning rate at epoch 70. The trainer saves the latest and
best fixed32 AP checkpoints at epoch boundaries and restores model, optimizer,
scheduler, and random state on resume.

- [Run configuration](../../results/grouppose_baseline/20261118_004914/run.json)
- [Launch log](../../results/grouppose_baseline/20261118_004914.log)
- [Split and TensorBoard verification](../../results/grouppose_baseline/20261118_004914/launch_verification.json)

The initialization quick-test recorded 0.00 AP before any training. The
first post-epoch quick-test is pending. The approved evaluation uses fixed32
TEST throughout. The [local comparison chart](http://127.0.0.1:33333/#scalars)
serves GroupPose and the PoseFlow V56 TEST continuation under
`quick-test/pose_AP`.

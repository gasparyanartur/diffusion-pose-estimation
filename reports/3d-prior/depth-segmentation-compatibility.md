# Depth and segmentation feature compatibility for a 3D pose prior

Status: source review, 2026-09-28. No checkpoints were run on CrowdPose for
this report. "Compatible" below describes a documented interface or a
plausible engineering route, **not** measured depth or mask quality.

## Decision the matrix must support

The first experiment needs visible scene geometry from one RGB image. A later
experiment may isolate each visible person before passing points to a 3D pose
model. Those are different outputs:

- A **depth map** gives one visible surface per pixel. A metric depth claim
  requires a scale convention and appropriate camera calibration.
- A **semantic person mask** labels person pixels, but merges adjacent people.
- An **instance mask** assigns separate visible masks to individual people.
  It does not reconstruct an occluded limb or back surface.
- A **shared backbone pass** means both heads consume features from the same
  weights, layer indices, image transform, and spatial grid. Merely calling
  two encoders "DINO" does not establish this.

## Depth and geometry models

| Model/checkpoint | Encoder and feature contract | Geometry output and camera information | Feature-reuse implication |
|---|---|---|---|
| [DA3-BASE](https://huggingface.co/depth-anything/DA3-BASE) | DA3's own DINOv2-derived ViT-B with selected intermediate layers, altered attention settings, and a DA3 checkpoint; [configuration](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/src/depth_anything_3/configs/da3-base.yaml). | Relative depth, confidence, estimated intrinsics and camera pose from the [DA3 API](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/docs/API.md). Intrinsics are estimated, not known ground truth. | `export_feat_layers` exposes activations from **this same DA3 pass**. No official pretrained person-segmentation head for them was found. |
| [DA3MONO-LARGE](https://huggingface.co/depth-anything/DA3MONO-LARGE) | Dedicated DA3 monocular checkpoint with a DINOv2-derived large encoder. | Direct **relative** depth; the [model zoo](https://github.com/ByteDance-Seed/Depth-Anything-3) does not list camera-pose or metric-depth capability for this variant. | Candidate depth-quality comparison. A separate intrinsics source is needed for pinhole back-projection; exported features still need a matching mask head. |
| [DA3METRIC-LARGE](https://huggingface.co/depth-anything/DA3METRIC-LARGE) | Separate monocular metric checkpoint in the DA3 family. | Metric depth requires the documented conversion `focal_px * net_output / 300`; without a measured focal length, scale inherits focal uncertainty. [DA3 FAQ](https://github.com/ByteDance-Seed/Depth-Anything-3) | It is not the DA3-BASE checkpoint or a plug-in segmentation head. Do not assume that BASE and METRIC features are interchangeable. |
| [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) Small/Base/Large | Fine-tuned DINOv2 ViT-S/B/L plus a DPT-style depth decoder; official relative and separately fine-tuned metric variants. | Relative depth by default; [indoor/outdoor metric checkpoints](https://github.com/DepthAnything/Depth-Anything-V2/blob/main/metric_depth/README.md) are separate. The standard relative model does not supply intrinsics. | Same *family* as Meta DINOv2, different trained checkpoint and decoder. Meta's DINOv2 segmentation head is not validated on DA2 features. |
| [MoGe-2](https://github.com/microsoft/MoGe) / [MoGe-3](https://github.com/microsoft/MoGe) | Own fine-tuned DINOv2-derived encoder, geometry neck and heads. V3 extends V2 with [sparse 3D refinement](https://github.com/microsoft/MoGe/blob/main/moge/model/v3.py). | The official models expose metric point/depth maps and camera intrinsics; V3 targets finer geometry. Their `mask` is a valid-geometry mask, **not** a person-instance mask. | Strong alternative for point-cloud quality, but no matching off-the-shelf person head. Backbone family similarity does not make Meta segmentation weights reusable. |
| [Depth Pro](https://github.com/apple-aiml-research/ml-depth-pro) | Its own multi-scale DINOv2-L/16-derived patch/image encoders and depth decoder; [default config](https://github.com/apple-aiml-research/ml-depth-pro/blob/main/src/depth_pro/depth_pro.py). | Returns depth in meters and estimated focal length in pixels if none is supplied. | Its multi-scale depth features are not the inputs to a released DINOv2 semantic or instance head. |
| [Meta DINOv2 dense depth heads](https://github.com/facebookresearch/dinov2) | A *matching* Meta DINOv2 ViT-S/B/L/g backbone with a NYU-Depth or KITTI linear/DPT head. Head variant, backbone size, register-token policy and preprocessing must match. | Dataset-trained depth; no joint intrinsics estimator documented in the head table. Cross-domain use on CrowdPose needs evaluation. | The clearest documented route to one shared **Meta DINOv2** encoder with its released semantic-segmentation heads. It does not yield individual person masks. |
| [Meta DINOv3 SYNTHMIX depther](https://github.com/facebookresearch/dinov3) | Released depther expects Meta's **ViT-7B/16** backbone checkpoint and its own head, not the project's DINOv3 ConvNeXt or ViT-S features. | Monocular depth; the official example does not state metric scale or provide intrinsics, so do not treat it as calibrated geometry. | Could pair conceptually with Meta's ViT-7B ADE20K segmentor, but exact weight sharing and CrowdPose behavior need verification; 7B is a large dependency. |

The DA3 distinction matters: [its configuration](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/src/depth_anything_3/configs/da3-base.yaml)
names a DINOv2 module but sets its own `out_layers`, `alt_start`,
`qknorm_start`, `rope_start`, and `cat_token`. Its [forward path](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/src/depth_anything_3/model/da3.py)
runs that backbone once for depth and can expose auxiliary features. A
feature tensor with the same channel count as Meta DINOv2 is not evidence
that a Meta-trained segmentation head will work on it.

## Segmentation models and the features they expect

| Model/checkpoint | Output relevant to people | Expected visual features or prompts | Directly consumes DA3 features? |
|---|---|---|---|
| [Meta DINOv2 ADE20K/VOC2012 heads](https://github.com/facebookresearch/dinov2) | Semantic classes, including `person` in VOC2012; all people share one class mask. Linear and multi-scale heads exist for ViT-S/B/L/g, and an ADE20K Mask2Former model is listed for ViT-g. | Matching **Meta DINOv2 checkpoint** and the head's exact backbone size, layer selection, patch grid and preprocessing; dense-task examples use mmcv/mmsegmentation. | **No verified direct reuse.** DA3's weights and feature path differ. A DA3-specific head needs training or at least a controlled transfer evaluation. |
| [Meta DINOv3 ADE20K segmentor](https://github.com/facebookresearch/dinov3) | ADE20K semantic segmentation; no separate person identities. | Released Mask2Former-style head with **Meta DINOv3 ViT-7B/16 LVD-1689M** backbone; official loading takes segmentor and backbone weights. | No. DINOv3 ViT-7B is neither DA3's DINOv2-derived ViT-B nor the repo's smaller DINOv3 encoders. |
| [SAM 3](https://github.com/facebookresearch/sam3) | Promptable concept **instance** masks; a `person` text prompt can request multiple people. Individual mask quality on CrowdPose is unmeasured here. | Its own [ViT visual backbone, neck, text encoder, detector and mask head](https://github.com/facebookresearch/sam3/blob/main/sam3/model_builder.py), with image/text or visual prompts. | No. Full pretrained model runs a separate visual encoder; replacing it with DA3 features requires model surgery and training. |
| [SAM 2.1](https://github.com/facebookresearch/sam2) | Prompted object masks. For automatic per-person use, boxes or points must come from another source, such as a person detector. | Its own [Hiera image encoder and FPN neck](https://github.com/facebookresearch/sam2/blob/main/sam2/configs/sam2/sam2_hiera_t.yaml), then prompt/mask modules. | No. DA3 features are not a supported input to the pretrained SAM 2 mask decoder. |
| [Mask2Former COCO instance models](https://github.com/open-mmlab/mmdetection/blob/main/configs/mask2former/README.md) | COCO class-labelled **instance** masks, including people. | A matching ResNet-50/101 or Swin feature pyramid and trained pixel/transformer decoder; the [R50 config](https://github.com/facebookresearch/Mask2Former/blob/main/configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml) uses `res2`–`res5`, 256-channel pixel-decoder features and learned object queries. | No direct reuse. DA3 patch tokens would need a feature-pyramid adapter and a trained mask decoder. |
| [Torchvision Mask R-CNN R50-FPN v2](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn_v2.html) | COCO person **instance** masks and boxes. | Own pretrained ResNet-50 plus FPN and RoI mask head; model weights include the backbone. | No. Useful independent mask baseline already within the project's PyTorch/torchvision stack. |

**Important distinction:** Mask2Former is a decoder family, not an indication
of feature compatibility. Meta's DINOv2 ViT-g ADE20K Mask2Former is a
semantic model with a Meta DINOv2 backbone; a COCO instance Mask2Former is a
different checkpoint with a ResNet or Swin backbone. [Meta DINOv2 release](https://github.com/facebookresearch/dinov2),
[COCO instance configurations](https://github.com/open-mmlab/mmdetection/blob/main/configs/mask2former/README.md).

## Pairwise compatibility matrix

Legend: **Direct** means published heads are designed for one exact backbone
family/checkpoint, subject to confirming load keys and preprocessing in a real
run. **Train** means a shared pass is architecturally possible but a matching
head/adapter must be trained. **Separate** means the existing checkpoints need
two visual encoders. This is a checkpoint/interface assessment, not a quality
ranking.

| Depth choice | Meta DINOv2 semantic head | Meta DINOv3 semantic head | SAM 2/3 masks | COCO Mask2Former or Mask R-CNN instance masks |
|---|---|---|---|---|
| DA3-BASE | **Train:** DA3 feature export enables one-pass head training; Meta head weights unverified. | Separate | Separate | Separate; a custom DA3 feature pyramid and instance head would be **Train**. |
| DA3MONO / DA3METRIC | **Train:** separate DA3 checkpoint per depth variant; intrinsics still need the variant-specific route. | Separate | Separate | Separate, or **Train** a variant-specific head. |
| Depth Anything V2 | **Train:** both use DINOv2-derived models, but DA2 fine-tuned weights are not Meta segmentation-head weights. | Separate | Separate | Separate. |
| MoGe-2/3 or Depth Pro | **Train:** DINOv2 lineage alone is insufficient for a Meta head; geometry-specific weights/features differ. | Separate | Separate | Separate. |
| Meta DINOv2 depth head | **Direct candidate** for matching Meta DINOv2 backbone/head variants; semantic masks only. | Separate | Separate | Separate. |
| Meta DINOv3 ViT-7B depther | Separate | **Direct candidate** for matching ViT-7B backbone/head variants; semantic masks only. | Separate | Separate. |

The direct candidates still need an integration check: verify that both heads
use the same frozen backbone weights, normalization, intermediate layers,
register-token handling and image resize. The official [DINOv2](https://github.com/facebookresearch/dinov2)
and [DINOv3](https://github.com/facebookresearch/dinov3) release tables list
the paired heads; they do not demonstrate a shared-forward multi-head runtime.

## Repository and data constraints

- [`src/models/features/dino.py`](../../src/models/features/dino.py) loads
  DINOv3 ConvNeXt Small by default; its ViT option is DINOv3 ViT-S/16.
  [`scripts/data/precompute_dino.py`](../../scripts/data/precompute_dino.py)
  also defaults to DINOv3 ConvNeXt Small. Neither is DA3's DINOv2-derived
  ViT-B checkpoint, Meta DINOv2 ViT-B/14, or the Meta DINOv3 ViT-7B/16 used
  by released dense heads.
- Local `data/crowdpose/annotations/crowdpose_val.json` annotations have
  `bbox`, `category_id`, `id`, `image_id`, `iscrowd`, `keypoints`, and
  `num_keypoints`; they have no `segmentation` field. A person-mask head
  cannot be trained or evaluated against masks from this file alone.
- The repo pins Torch 2.4.1 and torchvision 0.19.1 in
  [`requirements.txt`](../../requirements.txt). Meta's [DINOv3 evaluation
  setup](https://github.com/facebookresearch/dinov3) requests Torch 2.7.1 or
  newer. Runtime compatibility for DA3, MoGe-3, SAM 3, and the current repo
  environment was **not tested** in this report.
- Check each exact checkpoint license before packaging or redistribution:
  [DA3-BASE is Apache 2.0](https://huggingface.co/depth-anything/DA3-BASE),
  while DA3's [Large 1.1 model is CC BY-NC 4.0](https://github.com/ByteDance-Seed/Depth-Anything-3);
  [MoGe code is MIT with Apache-licensed DINOv2 code](https://github.com/microsoft/MoGe#%EF%B8%8F-license).
  [DINOv3](https://github.com/facebookresearch/dinov3#license) and
  [SAM 3](https://github.com/facebookresearch/sam3#license) each publish
  model-specific license terms. This report makes no license-compatibility
  judgment for a combined distribution.

## Recommendation for the next experiment

1. Keep **DA3-BASE** as the first one-image point-cloud probe: it returns
   relative depth and estimated intrinsics together, and its API exposes
   intermediate features. Record the exact checkpoint revision, image
   transform, intrinsics, depth convention and exported layer shapes. Compare
   one example with **MoGe-3** or **Depth Pro** only if geometry quality is
   insufficient; their metric output claims are model-specific and need
   visual or quantitative checking on CrowdPose. [DA3 API](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/docs/API.md),
   [MoGe model list](https://github.com/microsoft/MoGe),
   [Depth Pro API](https://github.com/apple-aiml-research/ml-depth-pro).
2. For **visible per-person clouds**, first establish an independent
   pretrained instance-mask baseline such as COCO Mask R-CNN R50-FPN v2; test
   SAM 3 separately if text-prompted crowd separation is useful. Report mask
   overlap, missing people and limb-boundary failures on selected images.
   [Torchvision weights](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn_v2.html),
   [SAM 3](https://github.com/facebookresearch/sam3).
3. Treat **one DA3 backbone for depth and segmentation** as a subsequent
   trained-head experiment. Export fixed DA3 layers, align mask targets to
   its processed image grid, obtain external person-instance labels or
   carefully audited pseudo-labels, and compare mask quality against the
   independent baseline. The current report supplies no evidence that a
   released Meta DINO segmentation head can be attached without training.

No model was downloaded or benchmarked for this document. Accuracy, memory,
throughput, feature tensor shape, and package conflicts remain empirical
questions for the approved visualization and mask experiments.

# DiffusionRegPose model size and hyperparameters

This report describes the current native DiffusionRegPose implementation in
this repository as of 2026-08-23. The user-requested path
src/pipelines/diffusionregpose does not exist in this checkout. The canonical
implementation is
[src/models/pipelines/pose_estimators/diffusionregpose/](../src/models/pipelines/pose_estimators/diffusionregpose/).

The report covers the model assembly, its imported model components, the
training wrapper and criterion, the Hydra presets, and the native CUDA
deformable-attention operator. Parameter totals were checked by instantiating
DiffusionRegPoseModel with backbone_pretrained=False; disabling pretrained
weight loading does not change any tensor shape or parameter count.

## Executive summary

The default pipeline=diffusionregpose model has:

| Quantity | Count |
|---|---:|
| Unique parameters | **49,527,324** |
| Trainable parameters with freeze_backbone_stem_and_layer1=true | **49,304,924** |
| Frozen parameters | **222,400** |
| Unique registered buffers | 118,240 elements |
| FP32 storage for unique parameters | about 198.11 MB / 188.93 MiB |
| FP32 storage for trainable parameters | about 197.22 MB / 188.08 MiB |

The trainable total is the repository’s configured/reference figure of
49,304,924. The default model is a ResNet-50 backbone followed by four
256-channel multi-scale feature levels, a six-layer deformable encoder, a
two-stage proposal selection path, and a six-layer deformable decoder. The
decoder starts with 900 proposal queries, expands 100 selected groups into one
box query plus 14 keypoint queries per group, and therefore produces 100
person slots with 14 keypoints each.

The model is large primarily because of the ResNet-50 backbone (23.45M
parameters), the six decoder layers (9.28M), the six encoder layers (7.69M),
and the fourth 3x3 feature projection (4.72M).

## Implementation scope and ownership

| File | Responsibility | Learnable parameters? |
|---|---|---:|
| [backbone.py](../src/models/pipelines/pose_estimators/diffusionregpose/backbone.py:15) | ResNet-50 hierarchy, frozen BatchNorm buffers, sine position encoding | ResNet only |
| [diffusion.py](../src/models/pipelines/pose_estimators/diffusionregpose/diffusion.py:12) | Cosine beta schedule and DDIM/q-sampling state | No; buffers only |
| [transformer.py](../src/models/pipelines/pose_estimators/diffusionregpose/transformer.py:75) | Multi-scale attention, encoder/decoder layers, time conditioning, prediction heads | Yes |
| [model.py](../src/models/pipelines/pose_estimators/diffusionregpose/model.py:149) | Backbone/projection/transformer assembly, proposals, denoising queries, sampling | Yes through assembled submodules |
| [training.py](../src/models/pipelines/pose_estimators/diffusionregpose/training.py:196) | Targets, Hungarian matching, losses, Lightning wrapper, Hydra config | Criterion has no trainable parameters |
| [evaluation.py](../src/models/pipelines/pose_estimators/diffusionregpose/evaluation.py:176) | Post-processing, pose NMS, CrowdPose result conversion and metrics | No |
| [deformable_attention_cuda.cu](../src/models/pipelines/pose_estimators/diffusionregpose/deformable_attention_cuda.cu:1) | Optional native forward/backward MSDA kernel | No |

__init__.py re-exports these components; it does not add a second model or
another set of parameters.

## Parameter accounting method

Counts in this report mean unique torch.nn.Parameter objects from an
instantiated default DiffusionRegPoseModel:

~~~
sum(parameter.numel() for parameter in model.parameters())
~~~

Trainable and frozen totals additionally filter on parameter.requires_grad.
This distinction matters because:

1. Decoder prediction-head lists are attached both at the model root and under
   transformer.decoder, so they are aliases, not additional copies.
2. pose_hw_embed repeats one nn.Sequential object across four list entries, so
   its weights are shared across pose-decoder stages.
3. FrozenBatchNorm2d stores affine/running-statistic tensors as buffers rather
   than parameters.

A raw state_dict() sum is not a unique-parameter count: the current model
contains 753 state-dict entries and 51,897,398 serialized tensor elements
because shared module references appear under multiple state-dict keys. The
unique parameter total above is the appropriate model-size number.

## End-to-end tensor flow

For input images and padding masks:

~~~
images       [B, 3, H, W]
padding_mask [B, H, W]
       |
       v
ResNet-50 layer2/layer3/layer4
       |
       v
four projected feature levels, each with 256 channels
       |
       v
flatten + sine positions + level embeddings
       |
       v
deformable encoder memory [B, S, 256]
       |
       v
two-stage proposal selection: 900 queries
       |
       v
six-layer decoder; after two box layers expand 100 groups x (1 box + 14 points)
       |
       v
pred_logits [B, 100, 2]
pred_boxes [B, 100, 4]                 normalized cxcywh in [0, 1]
pred_keypoints [B, 100, 42]            14 x (x, y), followed by 14 visibility values
~~~

The model-internal boxes use sigmoid-normalized (cx, cy, w, h) values in
[0, 1]. Keypoint coordinates entering the model are canonical [-1, 1]
coordinates; the diffusion proposal state is scaled to [-6, 6] and mapped
back to [0, 1] before decoder query expansion.

## Default architectural hyperparameters

The active full model is defined in
[conf/pipeline/diffusionregpose.yaml](../conf/pipeline/diffusionregpose.yaml:3)
and forwarded by
[DiffusionRegPosePipelineConfig.build](../src/models/pipelines/pose_estimators/diffusionregpose/training.py:633).

| Hyperparameter | Default |
|---|---:|
| Backbone | torchvision ResNet-50 with FrozenBatchNorm2d |
| Pretrained backbone | true |
| Frozen backbone regions | stem (conv1, bn1) and layer1 |
| Diffusion training timesteps | 1,000 |
| Sampling timesteps | 1 |
| Diffusion scale | 6.0, hard-coded in DiffusionRegPoseModel |
| DDIM eta | 0.0, deterministic, hard-coded in DiffusionRegPoseModel |
| Two-stage proposal queries (num_queries) | 900 |
| Diffusion proposals (num_proposals) | 100 |
| Decoder groups (num_group) | 100 |
| Classes | 2; CrowdPose person label is class 1 |
| Feature levels | 4 |
| Hidden/model dimension | 256 |
| Attention heads | 8 |
| Attention head dimension | 32 |
| Sampling points per level | 4 |
| Encoder layers | 6 |
| Decoder layers | 6 |
| Decoder FFN dimension | 2,048 |
| Body points | 14 |
| Box-only decoder layers | 2 |
| Denoising queries | 100 |
| Denoising box noise scale | 0.4 |
| Denoising label noise ratio | 0.5 |
| Denoising label loss coefficient | 0.3 |
| Denoising box/GIoU coefficient | 0.5 |
| Attention dropout in model assembly | 0.0 |
| Decoder/encoder layer dropout in direct class defaults | 0.1, overridden to 0.0 by assembly |
| Position features | 128 for height + 128 for width |
| Position temperatures | 20 for height and 20 for width |

num_queries controls the initial two-stage proposal tensor and does not
create a learned query embedding. num_group controls query expansion and
attention-mask sizes but does not itself add parameters. During sampling,
num_proposals must equal num_group.

## ResNet-50 backbone

The backbone is constructed by
[DiffusionRegPoseResNetBackbone](../src/models/pipelines/pose_estimators/diffusionregpose/backbone.py:88)
with return_interm_indices=(1, 2, 3), corresponding to torchvision layer2,
layer3, and layer4. The custom FrozenBatchNorm2d has no parameters; its
weight, bias, running mean, and running variance are buffers.

### Backbone parameter counts

| Component | Structure | Output channels | Parameters | Trainable by default |
|---|---|---:|---:|---:|
| Stem convolution | 7x7, 3 -> 64, stride 2 | 64 | 9,408 | 0 |
| layer1 | 3 bottleneck blocks, 64/64/256 | 256 | 212,992 | 0 |
| layer2 | 4 bottleneck blocks, 128/128/512 | 512 | 1,212,416 | 1,212,416 |
| layer3 | 6 bottleneck blocks, 256/256/1024 | 1,024 | 7,077,888 | 7,077,888 |
| layer4 | 3 bottleneck blocks, 512/512/2048 | 2,048 | 14,942,208 | 14,942,208 |
| **Backbone total** | ResNet-50 convolution weights | — | **23,454,912** | **23,232,512** |

The frozen count is 9,408 + 212,992 = 222,400. The backbone total excludes
BatchNorm buffers from parameter counts. If
freeze_backbone_stem_and_layer1 is set to false, the total model size is
unchanged but all 23,454,912 backbone parameters become trainable.

### Feature-map dimensions

The returned feature hierarchy has nominal strides 8, 16, and 32:

| Level | Source | Shape before projection |
|---|---|---|
| 0 | ResNet layer2 | [B, 512, ceil-ish(H/8), ceil-ish(W/8)] |
| 1 | ResNet layer3 | [B, 1024, H/16, W/16] |
| 2 | ResNet layer4 | [B, 2048, H/32, W/32] |

For square inputs the observed shapes are:

| Input | Level 0 | Level 1 | Level 2 |
|---|---|---|---|
| 512 x 512 | 512 x 64 x 64 | 1024 x 32 x 32 | 2048 x 16 x 16 |
| 800 x 800 | 512 x 100 x 100 | 1024 x 50 x 50 | 2048 x 25 x 25 |

## Sine positional encoding and feature projections

PositionEmbeddingSineHW produces a parameter-free 256-channel tensor: 128
height features concatenated with 128 width features. It uses normalized
cumulative valid-pixel coordinates, separate temperatures of 20, and the
default scale 2*pi
([backbone.py](../src/models/pipelines/pose_estimators/diffusionregpose/backbone.py:15)).

build_input_projections creates four 256-channel levels
([transformer.py](../src/models/pipelines/pose_estimators/diffusionregpose/transformer.py:220)):

| Level | Operation | Parameter calculation | Parameters |
|---|---|---:|---:|
| 0 | Conv2d(512, 256, 1) + GroupNorm(32, 256) | 512*256 + 256 + 256 + 256 | 131,840 |
| 1 | Conv2d(1024, 256, 1) + GN | 1024*256 + 256 + 512 | 262,912 |
| 2 | Conv2d(2048, 256, 1) + GN | 2048*256 + 256 + 512 | 525,056 |
| 3 | Conv2d(2048, 256, 3, stride=2, padding=1) + GN | 2048*256*9 + 256 + 512 | 4,719,360 |
| **Total** | — | — | **5,639,168** |

All projected features have shape [B, 256, h_l, w_l]. They are flattened
and concatenated into memory [B, S, 256], where
S = sum_l(h_l * w_l). Representative totals are:

| Input | Projected spatial sizes | Memory tokens S |
|---|---|---:|
| 512 x 512 | 64x64, 32x32, 16x16, 8x8 | 5,440 |
| 800 x 800 | 100x100, 50x50, 25x25, 13x13 | 13,294 |
| 800 x 1333 | 100x167, 50x84, 25x42, 13x21 | 22,223 |

The fourth level is created by applying a stride-2 3x3 convolution to the
last backbone feature. The exact non-square dimensions depend on the current
input and padding mask; runtime batching pads transformed images to the
largest height and width in that batch.

## Deformable encoder

The encoder is built from six independent deep copies of
DeformableTransformerEncoderLayer
([model.py](../src/models/pipelines/pose_estimators/diffusionregpose/model.py:109)).
Each layer has d_model=256, d_ffn=2048, four feature levels, eight heads,
four sampling points, ReLU activation, two LayerNorms, and zero dropout in the
assembled model.

### One encoder layer

| Subcomponent | Dimensions | Parameters |
|---|---|---:|
| MSDA sampling offsets | Linear(256, 256) | 65,792 |
| MSDA attention weights | Linear(256, 128) | 32,896 |
| MSDA value projection | Linear(256, 256) | 65,792 |
| MSDA output projection | Linear(256, 256) | 65,792 |
| **MSDA subtotal** | 4 levels x 8 heads x 4 points | **230,272** |
| FFN linear 1 | 256 -> 2048 | 526,336 |
| FFN linear 2 | 2048 -> 256 | 524,544 |
| LayerNorm x2 | 256 features each | 1,024 |
| **One encoder layer** | — | **1,282,176** |
| **Six encoder layers** | independent copies | **7,693,056** |

The native CUDA implementation and the PyTorch grid_sample fallback implement
the same attention interface and have no parameters. The CUDA source is an
operator implementation, not another model layer.

The encoder also computes reference points from spatial shapes and valid
ratios. These tensors are runtime state and do not affect parameter count.

## Time conditioning and two-stage proposal path

### Diffusion time conditioning

The sinusoidal embedding itself is parameter-free. With d_model=256, the
learned time path is:

| Layer | Dimensions | Parameters |
|---|---|---:|
| Sinusoidal embedding | 256 -> 256 | 0 |
| Linear + GELU | 256 -> 1024 | 263,168 |
| Linear | 1024 -> 1024 | 1,049,600 |
| **time_mlp** | — | **1,312,768** |
| SiLU + linear | 1024 -> 256 | 262,400 |
| **block_time_mlp** | — | **262,400** |

The time embedding is used by encode_features to modulate positional features
before the encoder
([model.py](../src/models/pipelines/pose_estimators/diffusionregpose/model.py:528)).

### Two-stage proposal heads

After encoder memory, the model applies:

| Component | Dimensions | Parameters |
|---|---|---:|
| enc_output | Linear(256, 256) | 65,792 |
| enc_output_norm | LayerNorm(256) | 512 |
| enc_out_bbox_embed | 3-layer MLP 256 -> 256 -> 256 -> 4 | 132,612 |
| enc_out_class_embed | Linear(256, 2) | 514 |

The proposal heads score encoder tokens and select the top 900 proposal
features/references. The selected feature/reference tensors are detached in
select_two_stage_queries; this affects gradient flow but not parameter size.

## Deformable decoder

The decoder has six independent copies of
DeformableTransformerDecoderLayer
([transformer.py](../src/models/pipelines/pose_estimators/diffusionregpose/transformer.py:266)).
Each layer has a deformable cross-attention block, standard eight-head
self-attention, a 256-to-2048-to-256 FFN, three LayerNorms, and zero dropout in
the full model assembly.

### One decoder layer

| Subcomponent | Dimensions | Parameters |
|---|---|---:|
| Deformable cross-attention | same MSDA as encoder | 230,272 |
| Standard self-attention | 8 heads, 256 model dim | 263,168 |
| FFN | 256 -> 2048 -> 256 | 1,050,880 |
| LayerNorm x3 | 256 features each | 1,536 |
| **One decoder layer** | — | **1,545,856** |
| **Six decoder layers** | independent copies | **9,275,136** |

The standard self-attention uses PyTorch’s packed in_proj_weight with shape
[768, 256], in_proj_bias with shape [768], and a 256 -> 256 output
projection.

### Decoder-only learned extras

These modules are outside the six repeated decoder layers:

| Component | Dimensions | Parameters |
|---|---|---:|
| Final decoder norm | LayerNorm(256) | 512 |
| Reference-point head | MLP 512 -> 256 -> 256 | 197,120 |
| emb_Q, emb_K, emb_SQ, emb_SK | four Linear(2, 256) layers | 3,072 |
| emb_V, emb_SV | two Linear(2, 2) layers | 12 |
| hw | Embedding(14, 2) | 28 |
| keypoint_embed | Embedding(14, 256) | 3,584 |
| **Decoder extras** | — | **204,328** |

The decoder-only total is therefore 9,275,136 + 204,328 = 9,479,464 when
prediction heads are excluded. The assembled transformer module reports
20,407,388 when inspected in isolation because its decoder contains references
to the root-registered prediction heads. The model-global partition below
counts those head objects only once.

### Query sequence dimensions

The decoder operates in sequence-first form internally, [Q, B, 256], and
returns batch-first intermediate outputs.

| Phase | Inference | Training |
|---|---:|---:|
| Initial two-stage queries | 900 | 900 normal + 100 denoising = 1,000 |
| First two decoder layers | 900 | 1,000 |
| After group expansion | 100 groups x 15 = 1,500 | 100 denoising + 1,500 normal = 1,600 |
| Remaining four decoder layers | 1,500 | 1,600 |
| Final normal box slots | 100 | 100 |
| Final normal keypoint coordinates | 100 x 14 x 2 = 2,800 | same |

The expansion occurs after the second box decoder layer. Each group consists of
one box query followed by 14 body-point queries. The point-query refinement
uses two small self/cross attention paths (emb_S* and emb_*) and the learned
body-point embeddings.

## Prediction heads and parameter sharing

DiffusionRegPosePredictionHeads is created with six decoder stages, two
box-only stages, 14 body points, and all sharing flags at their defaults
(false)
([transformer.py](../src/models/pipelines/pose_estimators/diffusionregpose/transformer.py:433)).

| Head family | Per-head architecture | Instances | Unique parameters |
|---|---|---:|---:|
| Box refinement | MLP 256 -> 256 -> 256 -> 4 | 6 independent | 795,672 |
| Classification | Linear(256, 2) | 6 independent | 3,084 |
| Pose xy refinement | MLP 256 -> 256 -> 256 -> 2 | 4 independent | 528,392 |
| Pose width/height refinement | MLP 256 -> 256 -> 256 -> 2 | one shared object repeated 4 times | 132,098 |
| **Decoder prediction heads** | — | — | **1,459,246** |

The four pose heads are num_decoder_layers - num_box_decoder_layers = 4.
The special +1 branch in the source applies only to a 17-keypoint model; it is
inactive for CrowdPose’s 14 points. The pose_hw_embed list is built from the
same pose_hw_head object four times, so it contributes 132,098 unique
parameters rather than 528,392.

The first-stage box and class heads are deep-copied into the two-stage encoder
proposal path, adding 132,612 + 514 = 133,126 parameters. Decoder head
aliases do not add parameters a second time.

## Complete unique-parameter partition

This partition sums to the unique model total without double-counting aliases:

| Model registration group | Unique parameters | Trainable |
|---|---:|---:|
| label_enc (Embedding(101, 256)) | 25,856 | 25,856 |
| ResNet-50 backbone | 23,454,912 | 23,232,512 |
| Four input projections | 5,639,168 | 5,639,168 |
| Box heads | 795,672 | 795,672 |
| Class heads | 3,084 | 3,084 |
| Pose xy heads | 528,392 | 528,392 |
| Shared pose width/height head | 132,098 | 132,098 |
| Transformer exclusive parameters | 18,948,142 | 18,948,142 |
| **Total** | **49,527,324** | **49,304,924** |

The transformer-exclusive row includes:

| Transformer-exclusive component | Parameters |
|---|---:|
| level_embed, shape [4, 256] | 1,024 |
| Six encoder layers | 7,693,056 |
| Six decoder layers plus decoder-only extras | 9,479,464 |
| enc_output | 65,792 |
| enc_output_norm | 512 |
| Encoder proposal box head copy | 132,612 |
| Encoder proposal class head copy | 514 |
| time_mlp | 1,312,768 |
| block_time_mlp | 262,400 |
| **Transformer-exclusive total** | **18,948,142** |

The assembled transformer module reports 20,407,388 when inspected in
isolation because its decoder contains references to the root-registered
prediction heads. The model-global total above counts those head objects only
once.

## Diffusion state and sampling hyperparameters

DiffusionRegPoseDiffusion constructs a cosine beta schedule with timesteps=1000
and s=0.008
([diffusion.py](../src/models/pipelines/pose_estimators/diffusionregpose/diffusion.py:12)).
It registers 12 one-dimensional coefficient buffers, each of length 1,000:

~~~
betas
alphas_cumprod
alphas_cumprod_prev
sqrt_alphas_cumprod
sqrt_one_minus_alphas_cumprod
log_one_minus_alphas_cumprod
sqrt_recip_alphas_cumprod
sqrt_recipm1_alphas_cumprod
posterior_variance
posterior_log_variance_clipped
posterior_mean_coef1
posterior_mean_coef2
~~~

These are float64 buffers and contribute 12,000 elements, not parameters. The
model uses:

| Setting | Value | Effect |
|---|---:|---|
| Training diffusion steps | 1,000 | Schedule length |
| Proposal state shape | [B, 100, 28] | 100 proposals x 14 points x 2 coordinates |
| Proposal scale | 6.0 | State is clamped to [-6, 6] |
| Training target timestep | 0 | build_diffusionregpose_targets uses timestep-zero q-sampling |
| Sampling steps | 1 | One DDIM pair for the default evaluator |
| DDIM eta | 0.0 | Deterministic sampling; step noise is zero |

Calling sample_keypoints with sampling_timesteps > 1 increases runtime
linearly with the number of DDIM pairs but does not change parameter count.

## Training targets, losses, and optimizer

The training wrapper and criterion are in
[training.py](../src/models/pipelines/pose_estimators/diffusionregpose/training.py:92).
The criterion and Hungarian matcher have no trainable weights. The matcher
stores a 14-element sigma buffer:

~~~
[0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107,
 0.087, 0.087, 0.089, 0.089, 0.079, 0.079]
~~~

The target builder:

- uses CrowdPose person label 1;
- retains all labeled joints (visibility > 0) in the criterion target;
- represents target boxes as normalized [0, 1] cxcywh;
- represents target keypoints as flattened xy coordinates followed by 14
  visibility values;
- initializes/pads the 100-proposal diffusion state with clean keypoints where
  available and Gaussian noise for remaining proposals;
- applies q_sample at timestep zero.

### Active loss weights

The full Hydra preset overrides the dataclass’s keypoints default of 1.0 to
10.0. The active values are:

| Loss/cost | Value |
|---|---:|
| Classification loss | 2.0 |
| Box L1 loss | 5.0 |
| GIoU loss | 2.0 |
| Keypoint L1 loss | 10.0 |
| OKS loss | 4.0 |
| Matcher class cost | 2.0 |
| Matcher box cost | 5.0 |
| Matcher GIoU cost | 2.0 |
| Matcher keypoint cost | 10.0 |
| Matcher OKS cost | 4.0 |
| Focal alpha | 0.25 |
| Denoising classification coefficient | 0.3 |
| Denoising box/GIoU coefficient | 0.5 |

Final and auxiliary decoder outputs are matched and supervised. Denoising
classification and box/GIoU losses are also calculated for the denoising
queries. Auxiliary keypoint losses begin after the configured two box-only
decoder layers.

### Active optimizer and schedule

The full pipeline uses build_adamw_with_schedule from
[src/training/optim.py](../src/training/optim.py:15):

| Setting | Value |
|---|---:|
| Optimizer | AdamW |
| Main learning rate | 2e-4 |
| Backbone learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Scheduler | MultiStepLR |
| Milestones | epochs 31 and 61 |
| Decay factor | 0.1 at each milestone |
| Warmup | 0 epochs |
| Split decay/no-decay groups | false in the active preset |
| Runtime max epochs | 80 |
| Gradient clipping | 0.1 |
| Precision | 32-bit |
| Gradient accumulation | 8 batches |
| Effective training batch | 8, from micro-batch 1 x accumulation 8 |
| Validation loss | every epoch |
| Validation AP | every 10 epochs and at the final epoch |

With split_weight_decay=false, the optimizer creates main and backbone
groups but combines each group’s decay/no-decay parameters before applying the
configured weight_decay; the source does not preserve a separate zero-decay
norm/bias group in this mode.

The paper-compatible initialization path is
external/DiffusionRegPose/pretrain_models/edpose_r50_crowdpose.pth. The
published DiffusionRegPose model-zoo checkpoint is reserved for evaluation and
parity rather than used as the training initialization in the active preset.

## Data and image preprocessing hyperparameters

The full data preset is
[conf/data/crowdpose_diffusionregpose.yaml](../conf/data/crowdpose_diffusionregpose.yaml:1).

| Setting | Full training preset |
|---|---|
| Dataset | CrowdPose trainval / val / test annotations |
| Target format | set |
| Keypoints | 14 |
| External/reference transforms | enabled |
| Train short-side scales | 480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800 |
| Train aspect-ratio max size | 1,333 |
| Crop-resize scales | 400, 500, 600 |
| Crop size range | 384 to 600 |
| Evaluation resize | short side 800, max size 1,333 |
| Micro-batch size | 1 |
| Data-loader workers | 4 |
| Prefetch factor | 2 |
| AP batch size | 32 |
| Dataset fraction | 1.0 |
| Image normalization | ImageNet mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225] |
| Batch padding | bottom/right zero padding to largest transformed H/W, with boolean mask |

The generic data field max_size: 512 remains present in the preset, but the
active external-transform path uses train_max_size=1333 and
eval_max_size=1333 for its resize decisions. Transformed images are then
batched by variable-size padding. Consequently, parameter count is resolution
independent, while memory-token and activation sizes depend on the actual
transformed/padded dimensions.

## Evaluation and post-processing hyperparameters

The evaluator defaults to a shortest side of 800 and max size 1,333
([evaluate_diffusionregpose.py](../scripts/evaluation/evaluate_diffusionregpose.py:27)).
The model’s final logits are sigmoid probabilities over two classes. The
postprocessor selects query_count candidates from the flattened query x class
probabilities, scales normalized boxes/keypoints to target image dimensions,
applies CrowdPose pose NMS by default, and serializes 14 keypoints as
(x, y, visibility) triples
([evaluation.py](../src/models/pipelines/pose_estimators/diffusionregpose/evaluation.py:176)).

Evaluation-specific facts that do not change model size:

- standalone evaluation defaults to sampling_timesteps=1, seed 42, and CUDA
  when available;
- torch.backends.cuda.matmul.allow_tf32 is disabled while cuDNN TF32 remains
  enabled;
- post-processing uses a pose-NMS threshold based on 0.04 * sqrt(pose_area)
  and suppresses candidates with more than 10 nearby joints when their score
  is above 0.05;
- result conversion uses batched box NMS with IoU threshold 1.0.

## Configuration variants

### Debug architecture

[conf/pipeline/diffusionregpose_debug.yaml](../conf/pipeline/diffusionregpose_debug.yaml:3)
changes architecture, so it is not a checkpoint-compatible submodel:

| Setting | Full | Debug |
|---|---:|---:|
| num_queries | 900 | 100 |
| num_proposals | 100 | 10 |
| num_feature_levels | 4 | 3 |
| Encoder layers | 6 | 2 |
| Decoder layers | 6 | 3 |
| Box decoder layers | 2 | 1 |
| Decoder groups | 100 | 10 |
| Denoising queries | 100 | 100 |
| Total parameters | 49,527,324 | **34,254,502** |
| Trainable parameters | 49,304,924 | **34,032,102** |
| Frozen parameters | 222,400 | 222,400 |

The debug count was also instantiated and checked with pretrained loading
disabled. Its reduced count comes from fewer feature levels, encoder layers,
decoder layers, and prediction-head copies; reducing query/group counts alone
would not materially reduce parameter count.

### Fast full-architecture feedback

benchmark_diffusionregpose_fast uses the full pipeline architecture and the
debug data/training runtime settings. It reduces data/resolution but keeps
the 900-query, 4-level, 6-encoder, 6-decoder, 100-group model and therefore
retains 49,527,324 total parameters.

### Low-resolution full-data learning gate

benchmark_diffusionregpose_lowres also keeps the full model and changes only
the data/runtime path:

| Setting | Low-resolution value |
|---|---:|
| Train/eval short side | 128 |
| Train/eval max size | 192 |
| Crop size | 64 to 128 |
| Batch size | 8 |
| Gradient accumulation | 1 |
| Max epochs | 80 |
| Effective batch | 8 |

It has the same parameter totals as the full model. Its lower image resolution
reduces activation and attention work but does not change layer dimensions or
checkpoint shape.

## Analytical FLOPs and MACs

### Counting convention and scope

The estimates below use:

- one multiply-accumulate (MAC) as one multiplication plus one addition;
- 1 MAC = 2 FLOPs;
- convolution and linear-layer MACs counted from actual output dimensions;
- the observed ResNet feature shapes and projected token counts for each input;
- the default model dimensions: d=256, 8 heads, 4 levels, 4 sampling points,
  FFN width f=2048, 6 encoder layers, and 6 decoder layers.

The primary totals are a **main matrix-operation floor**, not a profiler
measurement. They include convolution, projection, linear, FFN, and standard
self-attention score/value matrix products. They exclude or only approximate
deformable bilinear sampling, softmax, masking, positional encoding,
LayerNorm/GroupNorm, activation functions, top-k, attention-mask construction,
Hungarian matching, criterion/loss computation, data movement, and optimizer
kernels. Actual wall time cannot be inferred from FLOPs alone.

The estimates include the ResNet-50 convolution hierarchy, four input
projections, six encoder layers, the encoder proposal path, and six decoder
layers. Prediction/refinement heads and the keypoint-query refinement attention
are discussed separately because their tensor layouts change during the
decoder.

### Core formulas

For a convolution with output height H_o, width W_o, input channels C_i,
output channels C_o, and kernel area k²:

~~~text
MACs = H_o * W_o * C_i * C_o * k²
~~~

For a linear layer applied to N tokens:

~~~text
MACs = N * input_dim * output_dim
~~~

Let S be the total flattened feature-memory tokens across all four levels and
Q be the decoder query count for one decoder layer.

The default encoder-layer matrix floor is:

~~~text
MSDA projections =
  S * [d² + d*(heads*levels*points*2)
         + d*(heads*levels*points) + d²]

FFN = S * 2*d*f

one encoder layer = S * 1,277,952 MACs
six encoder layers = S * 7,667,712 MACs
~~~

The encoder proposal path contributes:

~~~text
S * [d² + d*2 + (d² + d² + d*4)]
= S * 198,144 MACs
~~~

For one decoder layer, the cross-attention value projection is applied to
memory S, while offsets/weights/output projections are applied to Q. Standard
self-attention includes its four d-by-d projections and the Q-by-Q score/value
products:

~~~text
decoder layer =
  S*d²
  + Q*[d*(heads*levels*points*2)
       + d*(heads*levels*points)
       + d²]
  + Q*4*d²
  + 2*Q²*d
  + Q*2*d*f
~~~

With the default dimensions this is:

~~~text
decoder layer = S*65,536 + Q*1,474,560 + Q²*512 MACs
~~~

Inference uses Q=900 for two box layers and Q=1,500 for four expanded layers.
Training uses Q=1,000 for the first two layers because of 100 denoising
queries, then Q=1,600 for the four expanded layers.

### Representative main matrix-operation floors

The following table reports MACs first and equivalent FLOPs second. The
training column uses the denoising-query layout but is still a forward
matrix-operation estimate; it does not include backward or criterion work.

| Input | Feature tokens S | Inference floor | Training-layout floor |
|---|---:|---:|---:|
| 512 x 512 | 5,440 | 84.463G MAC = 168.926 GFLOPs | 86.177G MAC = 172.354 GFLOPs |
| 800 x 800 | 13,294 | 181.958G MAC = 363.916 GFLOPs | 183.672G MAC = 367.344 GFLOPs |
| 800 x 1333 | 22,223 | 292.942G MAC = 585.884 GFLOPs | 294.656G MAC = 589.312 GFLOPs |
| 128 x 192 low-resolution example | 510 | 23.269G MAC = 46.538 GFLOPs | 24.984G MAC = 49.968 GFLOPs |

Component MACs for the default inference layout are:

| Input | ResNet-50 | Projections | Encoder | Proposal path | Decoder |
|---|---:|---:|---:|---:|---:|
| 512 x 512 | 21.353G | 1.242G | 41.712G | 1.078G | 19.078G |
| 800 x 800 | 52.132G | 3.091G | 101.935G | 2.634G | 22.166G |
| 800 x 1333 | 87.333G | 5.129G | 170.400G | 4.403G | 25.677G |
| 128 x 192 | 2.002G | 0.116G | 3.911G | 0.101G | 17.140G |

The decoder remains relatively expensive at low resolution because its standard
self-attention depends on Q² and its query counts do not shrink with image
resolution.

### Additional decoder work not in the floor

After the second box layer, the default model refines 100 groups x 14 points,
or 1,400 point tokens. The two point-refinement attention paths in
_refine_diffusion_keypoints contain approximately:

~~~text
2 * [N*(1,028) + N²*(258)]
N = 1,400
approximately 1.014G MAC = 2.028 GFLOPs
~~~

The projection term is small relative to the N² attention term; the formula
represents the two paths, each with two 2-to-256 projections and one 2-to-2
projection. It is not included in the main floor. Decoder reference-update and prediction
heads add roughly 3.0G MAC, or about 6 GFLOPs, for the default inference query
layout. Denoising heads add a smaller training-only term.

Deformable attention’s bilinear sampling, softmax, and weighted accumulation
also sit outside the matrix floor. Consequently, the table should be read as a
consistent architecture-comparison baseline rather than an exact hardware
FLOP count.

### Effect of reducing query/group counts

Reducing num_queries from 900 to 600 and num_group/num_proposals from 100 to
50 changes the inference query layout from (900, 1,500) to (600, 750). Keeping
100 denoising queries for the comparison changes training from (1,000, 1,600)
to (700, 850). At 800 x 800:

| Layout | Inference decoder | Training-layout decoder |
|---|---:|---:|
| Default | 22.166G MAC = 44.333 GFLOPs | 23.881G MAC = 47.761 GFLOPs |
| Reduced queries/groups | 12.941G MAC = 25.882 GFLOPs | 14.287G MAC = 28.573 GFLOPs |

This is a large decoder saving without parameter-count reduction, but it lowers
the number of proposal groups and therefore the maximum output capacity. It
must be treated as a model-behavior change, not an exact-parity optimization.

### Effect of reducing depth and FFN width

At 800 x 800, encoder 6 -> 4 removes two of six encoder-layer passes:
approximately 33% of encoder matrix work, or 33.978G MAC from the inference
floor. Decoder 6 -> 4, keeping two box layers, changes the inference decoder
floor from 22.166G to approximately 13.696G MAC and the training-layout
decoder floor from 23.881G to approximately 14.798G MAC.

Reducing FFN width from 2048 to 1024 saves 525,312 MAC per token per
encoder/decoder layer. Because the encoder processes S tokens, this is
especially effective at high image resolution. It reduces parameters by
6,303,744 across all 12 layers and should be evaluated together with the
query/layout changes rather than judged by parameter count alone.

### Training-step interpretation

The model forward is only one part of a training step. A backward pass for
dense linear/convolution operations commonly costs roughly another one to two
forward-equivalents, but the exact multiplier depends on saved activations,
kernel fusion, and the native deformable-attention backward implementation.
This report therefore does not claim an exact training FLOP total.

Training also adds:

- Hungarian matching for final and auxiliary outputs;
- focal, box, GIoU, keypoint, and OKS losses;
- denoising losses and auxiliary denoising losses;
- gradient accumulation and optimizer updates;
- periodic validation and AP post-processing.

For this reason, the FLOPs table is most reliable for comparing architectural
changes. Each proposed reduction still needs a representative benchmark that
measures data wait, forward, backward, matching/loss, optimizer, validation,
and AP time separately.

## Performance improvements derived from the FLOPs analysis

The FLOPs analysis identifies two different scaling regimes:

1. **Backbone and encoder cost scales with feature tokens.** ResNet-50 and the
   six deformable encoder layers dominate as image resolution grows. At
   800x1333, the encoder matrix floor is 170.400G MAC, compared with 41.712G
   at 512x512.
2. **Decoder cost scales with both memory tokens and query count.** Its
   deformable cross-attention depends on S, while standard self-attention
   depends on Q². At low resolution, the decoder remains expensive because
   its 900-to-1500 query layout does not shrink with the image.
3. **Training adds non-forward work.** Backward, Hungarian matching, auxiliary
   losses, denoising losses, optimizer updates, validation, and AP evaluation
   are not captured by the matrix-operation floor.

Therefore the highest-impact interventions should first target the dominant
FLOP term for the intended input resolution, then consider actual parameter
shrinkage and ED-Pose transfer risk.

The current training preset initializes the model from the ED-Pose CrowdPose
checkpoint. That creates an important distinction:

- A **shape-preserving change** can reuse the ED-Pose tensors, although it may
  still change output capacity or the optimization objective.
- An **architecture-changing reduction** can sometimes load a same-shaped subset
  with the current shape-filtered loader, but it is not full checkpoint
  compatibility. Missing/new tensors must be initialized and the resulting
  model requires fine-tuning and validation.
- A **new student backbone or substantially smaller hidden dimension** should be
  treated as a new model-transfer experiment, usually with ImageNet/student
  initialization, distillation, or a separately trained checkpoint.

### Reduction opportunities ranked by compute impact

| Change | Parameter-count effect | Training-time effect | ED-Pose transfer | Predicted performance effect |
|---|---:|---|---|---|
| Lower train/eval image resolution | None | Very high; reduces ResNet and encoder memory-token cost | Full tensor shapes remain loadable | Likely AP reduction from lost spatial detail; magnitude is data- and resolution-dependent |
| Reduce num_group and num_proposals together | None | High after query expansion; reduces quadratic decoder self-attention | Weight shapes remain loadable; maximum detections/keypoint groups decrease | Paper evidence supports 100 as the best trade-off; 50 loses about 1.3 AP, while 200 loses about 0.6 AP versus 100 |
| Reduce encoder depth | 2,564,352 fewer for 6 -> 4 | High; removes one-third of encoder-layer passes | Partial loading; new fine-tuning required | Likely AP reduction from weaker multi-scale feature refinement; must be measured after fine-tuning |
| Reduce FFN width 2048 -> 1024 | 6,303,744 fewer | High; reduces every encoder/decoder FFN | Requires code/config extension and partial loading | Likely capacity loss, especially for crowded scenes; distillation or fine-tuning may recover some AP |
| Reduce decoder depth | 3,622,160 fewer for 6 -> 4 | High; removes about 38% of default decoder matrix work | Partial loading; output/deep-supervision structure changes | Likely AP reduction from fewer iterative denoising/refinement steps |
| Replace or shrink ResNet-50 | Potentially the largest reduction | Very high | ED-Pose R50 backbone tensors are not directly compatible | Likely backbone-dependent AP loss unless the replacement has strong pretrained or distilled initialization |
| Reduce num_queries | None | Medium; reduces initial decoder self-attention | Weight shapes remain loadable; output proposal capacity changes | Uncertain proposal-recall effect; the paper's QueryNumber ablation does not directly test this setting |
| Freeze ResNet layer2 or later | Total unchanged; trainable count falls | Medium; removes backward/optimizer work, not backbone forward work | Full tensors remain loadable | Usually neutral to mildly negative if the target domain is close to ED-Pose; more negative under domain shift |
| Reduce dn_number | None | Small-to-medium training-only saving | Weight shapes remain loadable; denoising objective changes | Possible optimization/generalization loss; no direct AP estimate from the paper ablation |
| Reduce feature levels 4 -> 3 | 5,015,680 fewer | Medium/high; reduces projection and deformable-attention work | Partial loading; attention/projection shapes change | Likely AP reduction, particularly for scale variation and small persons |
| Reduce hidden dimension 256 -> 192 | 7,739,584 fewer; 41,787,740 total | Very high; reduces most transformer/projection/head compute | Broad shape incompatibility; head dimension 24 triggers a warning | Likely capacity reduction; requires a separately trained or distilled student |
| Reduce hidden dimension 256 -> 128 | 14,659,968 fewer; 34,867,356 total | Very high | Broad shape incompatibility; requires a student/fine-tuning plan | Higher risk of AP loss; not a safe ED-Pose checkpoint-preserving change |
| Reduce validation/AP cadence | None | Medium at AP epochs; does not accelerate training steps | No checkpoint impact | No direct model-performance effect; it only reduces evaluation frequency |
| Remove/reduce auxiliary losses | None | Reduces matching/loss work, not forward work | Objective changes; not parity-preserving | Likely optimization loss and potentially lower AP; needs an ablation |

The parameter totals in the table were obtained from the same instantiated model
accounting used elsewhere in this report. The FFN-width and layer-count savings
are analytical source-derived values; the hidden-dimension totals were
instantiated with the current assembly and pretrained loading disabled.

#### QueryNumber correspondence and paper evidence

The paper's ablation is the closest evidence for the
`num_group`/`num_proposals` reduction, but the names are not identical. In the
paper, **QueryNumber** is the number of instance queries used to represent
people. In this implementation, `num_group` controls the number of expanded
person groups, and `num_proposals` controls the diffusion proposal state; the
sampling code requires them to match. Therefore, the repository's default
`num_group=100` and `num_proposals=100` are the closest implementation of the
paper's 100-query setting. `num_queries=900` is separate: it controls the
initial two-stage encoder proposal tensor and should not be interpreted as the
paper's QueryNumber.

On CrowdPose, the paper reports the following QueryNumber ablation:

| Paper QueryNumber | AP | Relative to 100 queries | Interpretation for this report |
|---:|---:|---:|---|
| 50 | 71.4 | -1.3 AP | Lower decoder cost and candidate capacity, with a measurable accuracy penalty |
| 100 | 72.7 | Baseline/best | Recommended starting point and current repository default |
| 200 | 72.1 | -0.6 AP | More candidate capacity, but higher compute and no accuracy gain in this ablation |

These AP differences are observed paper results, not a guaranteed prediction
for this checkout: dataset preprocessing, training schedule, initialization,
and implementation details can shift the absolute values. They do support the
practical recommendation that reducing the matched pair from 100 to 50 is a
reasonable speed/accuracy experiment when the expected number of people per
image is low, while increasing it to 200 is unlikely to be worthwhile purely
for AP. See the [official DiffusionRegPose paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Tan_DiffusionRegPose_Enhancing_Multi-Person_Pose_Estimation_using_a_Diffusion-Based_End-to-End_Regression_CVPR_2024_paper.pdf), Table 5.

### Recommended sequence

If retaining the ED-Pose initialization is the priority, use a
shape-preserving runtime/training strategy first:

1. Use the existing full-architecture low-resolution preset. It keeps all
   checkpoint tensor shapes while reducing feature-map and encoder cost.
2. Reduce num_group and num_proposals together only if the expected number of
   people per image permits it. At 800x800, changing the query layout from
   (900, 1500) to (600, 750) reduces the decoder floor from 22.166G to
   12.941G MAC, but lowers output capacity.
3. Freeze layer2, and possibly layer3, if the target domain is close enough to
   the ED-Pose training domain. This reduces trainable parameters and optimizer
   state but does not reduce the forward FLOPs of the frozen backbone.
4. Reduce dn_number or AP cadence for faster experiments, understanding that
   denoising and evaluation behavior changes.

For a genuinely smaller checkpoint, the most reviewable first architecture
experiment is encoder 6 -> 4 plus decoder 6 -> 4. It removes 6,186,512
parameters in total while retaining the 256-dimensional representation and
ResNet-50 feature interface. The first four layer tensors and the retained
head shapes can generally be transferred, but the resulting model is not
equivalent to the original six-layer model and must be fine-tuned.

The next higher-impact experiment is reducing FFN width from 2048 to 1024.
This saves 6,303,744 parameters and removes substantial matrix work, but d_ffn
is currently hard-coded in the transformer assembly rather than exposed as a
Hydra field. It therefore needs an explicit reduced-model implementation and
partial ED-Pose transfer audit.

Changing hidden_dim, reducing feature levels, or replacing the backbone should
be treated as a separately named student architecture. Those changes affect
projections, attention weights, LayerNorms, time conditioning, proposal heads,
and prediction heads simultaneously. Loading only same-shaped ED-Pose tensors
with strict=False is useful initialization evidence, not proof of checkpoint
compatibility.

## Approximate runtime memory implications

At FP32, the unique parameter tensors occupy approximately 198.11 MB. The
default frozen parameters account for about 0.89 MB; freezing them mainly
reduces gradient/optimizer state and compute, not checkpoint tensor shape. The
unique registered buffers add about 0.52 MB, including the float64 diffusion
schedule and frozen BatchNorm state.

For a rough FP32 training lower bound, trainable parameters plus gradients plus
two AdamW state tensors require approximately:

~~~
4 x 49,304,924 x 4 bytes ~= 788.88 MB
~~~

This includes one parameter copy, one gradient copy, and two AdamW moment
copies. It excludes activations, temporary attention tensors, CUDA allocator
overhead, dataloader memory, and any checkpoint duplication. The actual peak
is therefore resolution-, batch-, padding-, and decoder-query-dependent.

## Source and verification notes

The main source anchors are:

- model defaults and assembly:
  [model.py:149](../src/models/pipelines/pose_estimators/diffusionregpose/model.py:149)
- transformer layer construction:
  [model.py:90](../src/models/pipelines/pose_estimators/diffusionregpose/model.py:90)
- deformable attention and layer definitions:
  [transformer.py:75](../src/models/pipelines/pose_estimators/diffusionregpose/transformer.py:75)
- prediction heads:
  [transformer.py:433](../src/models/pipelines/pose_estimators/diffusionregpose/transformer.py:433)
- diffusion buffers:
  [diffusion.py:37](../src/models/pipelines/pose_estimators/diffusionregpose/diffusion.py:37)
- training and loss configuration:
  [training.py:571](../src/models/pipelines/pose_estimators/diffusionregpose/training.py:571)
- full Hydra model preset:
  [conf/pipeline/diffusionregpose.yaml](../conf/pipeline/diffusionregpose.yaml:1)
- full Hydra training preset:
  [conf/training/diffusionregpose.yaml](../conf/training/diffusionregpose.yaml:1)

Verification performed in the current virtual environment:

~~~
DiffusionRegPoseModel(backbone_pretrained=False)
unique parameters: 49,527,324
trainable parameters: 49,304,924
frozen parameters: 222,400
~~~

Representative backbone/projection shapes were also instantiated for
512x512, 800x800, and 800x1333 inputs. No production source or Hydra
configuration was modified for this report.

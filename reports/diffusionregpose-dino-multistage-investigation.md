# DINO multi-stage feature investigation for DiffusionRegPose

Date: 2026-09-29  
Scope: architecture, implementation feasibility, and runtime/memory assessment. No model, training, or evaluator changes were made.

## Findings summary

The proposed feature family is available locally: DINOv3 ConvNeXt-Small returns four spatial stages at strides 4/8/16/32. The prior weak-head sweep is a significant warning: its frozen stage-4 ConvNeXt variant reached 0.0083 test AP, far below the trainable ResNets in that setup. That result is stage-, resolution-, and train-policy-specific, so it does not settle a multi-stage DINO branch. The attention stack can consume more than four levels in principle, but common-memory augmentation makes every encoder layer process the added DINO tokens and changes every deformable-attention module's level-dependent projections. With the paper-shaped input, adding DINO stages 2–4 approximately doubles flattened memory and yields an estimated 2.10× encoder MAC floor; adding all four stages yields about 5.38× encoder MACs and 10.00× encoder sample sites. Neither is a measured runtime multiplier.

A separate decoder-only DINO branch is an optional lower-cost, lower-checkpoint-disruption design; it is not required to use DINO. The direct design is to append DINO stages 2–4 to the shared memory, giving seven levels (the four existing ResNet-derived levels plus three DINO maps). That lets all encoder and decoder layers read DINO and lets DINO tokens participate in proposal ranking. It is the more direct test of whether a shared multi-stage pyramid helps. The feature normalization mismatch and transformed-image alignment must be solved for either design.

The sections below show the mechanics and assumptions behind those comparisons. The final section gives a conditional next experiment; no design has measured pose accuracy or DINO-specific runtime yet.

## What “multi-stage” means here

There are two distinct stage axes:

1. **DINO backbone stages** are spatial feature maps at different resolutions. The repository's DINOv3 ConvNeXt-Small exposes four maps: 96 channels at stride 4, 192 at stride 8, 384 at stride 16, and 768 at stride 32.
2. **DiffusionRegPose transformer stages** are six encoder layers and six decoder layers in the paper-shaped model. Each encoder layer updates all spatial memory tokens; each decoder layer self-attends its pose/detection queries and cross-attends to the same image memory.

“Another set of multi-stage features” most directly means adding DINO backbone stages to the image-conditioning path. It does not mean adding more decoder depth.

## Current image and attention path

`DiffusionRegPoseResNetBackbone` returns ResNet-50 `layer2`, `layer3`, and `layer4` (512/1024/2048 channels at strides 8/16/32). `build_input_projections` maps each to 256 channels and makes the fourth level by downsampling the last projected feature (nominal stride 64). Each level gets a sine position embedding, a padding mask, and a learned level embedding. `prepare_transformer_inputs` flattens the four maps and concatenates them into one memory with `spatial_shapes`, `level_start_index`, `valid_ratios`, positions, and masks.

The six encoder layers apply multi-scale deformable self-attention to that memory. Encoder positions provide query offsets; projected memory provides values. The resulting encoded tokens drive two-stage box/person proposal scoring and query selection. The decoder initially refines 900 normal proposal queries through two box layers (plus training denoising queries), selects 100 proposals, then expands each selected person into one box and 14 joint queries. The remaining four decoder layers refine those queries using self-attention and multi-scale deformable cross-attention over the *same* image memory. In the decoder cross-attention, current query/reference locations choose samples from the image pyramid; this is where a separate DINO branch can condition pose queries without changing the encoder memory.

The attention implementation uses 8 heads, 4 sample points per level, and 256 hidden dimensions in the paper preset. For each attention module, increasing the configured level count increases the offset/weight projection outputs as well as sampled values. Increasing flattened memory size also increases value projection and memory-processing work at every encoder layer, and value projection work in each decoder layer.

Useful implementation references:

- [`backbone.py`](../src/models/pipelines/pose_estimators/diffusionregpose/backbone.py): ResNet feature levels and sine positions.
- [`model.py`](../src/models/pipelines/pose_estimators/diffusionregpose/model.py): model assembly, four input projections, and `prepare_transformer_inputs`.
- [`transformer.py`](../src/models/pipelines/pose_estimators/diffusionregpose/transformer.py): `MultiScaleDeformableAttention`, encoder and decoder layers, and per-level sampling.
- [`reports/diffusionregpose-how-it-works.md`](diffusionregpose-how-it-works.md): proposal selection, query expansion, and diffusion state.

## Available DINO features and integration contracts

`DinoFeatureExtractor` in [`src/models/features/dino.py`](../src/models/features/dino.py) currently returns only one selected ConvNeXt stage under `"spatial"`; default is stage 4. It records patch centers but does not currently return a list of pyramid maps, masks, or spatial shapes. The Hugging Face model is constructed with `output_hidden_states=True`, so a forward already computes and returns all stage maps; exposing stages 2–4 does not require rerunning the backbone or another DINO pass. It does require retaining those tensors for later fusion. Its ConvNeXt-Small stage contract is:

| DINO stage | Channels | Stride | Relative spatial tokens |
|---|---:|---:|---:|
| 1 | 96 | 4 | 4× stage 2 |
| 2 | 192 | 8 | 4× stage 3 |
| 3 | 384 | 16 | 4× stage 4 |
| 4 | 768 | 32 | 1× |

These channel dimensions need per-stage projection to 256. A separate branch also needs the same transformed-image coordinate frame as the pose references; stage masks and positional/level identity must match feature shapes. The current DINO extractor accepts an unnormalized `[0,1]` image and applies ImageNet normalization itself, while DiffusionRegPose's data module applies ImageNet normalization before returning the batch. Directly passing the pipeline batch through the existing extractor would therefore normalize twice. The DINO path must reuse pre-normalized transformed pixels or explicitly invert/adjust normalization.

For variable-size batches, features must stay aligned with each image after resize/crop/flip and batch padding. The DINO cache utility currently extracts unaugmented source images, resizes them using its image processor, and writes per-image stage-1/stage-4 features (or ViT patch tokens); it does not encode the random training crop/flip applied by DiffusionRegPose. Such caches cannot be substituted into the current stochastic training path without transform-aware caching or a changed augmentation design. Online frozen extraction is the clean first experiment.

The repository's DINOv3 ViT extractor currently returns only the final patch-token grid. The ViT can expose intermediate transformer blocks from one forward, but these are *depth taps at the same patch stride*, not a native multi-resolution pyramid: every selected block has the same `H/16 × W/16` grid. The official DINOv3 ViT implementation has `get_intermediate_layers(..., n=[...], reshape=True)` for selected blocks; its ConvNeXt implementation instead exposes native stage maps with strides 4/8/16/32 ([ViT source](https://github.com/facebookresearch/dinov3/blob/main/dinov3/models/vision_transformer.py), [ConvNeXt source](https://github.com/facebookresearch/dinov3/blob/main/dinov3/models/convnext.py), [Transformers DINOv3 model docs](https://huggingface.co/docs/transformers/model_doc/dinov3)). Thus ViT can provide multi-layer features, and they can be fused in a residual branch or passed through a learned feature pyramid neck, but “stages 2/3/4” does not mean the same thing for ViT as for ConvNeXt. To give ViT true 3-scale maps, add explicit learned resampling/downsampling (an FPN-like neck); simple reshaping alone will not do that.

### ViT versus ConvNeXt DINO for this branch

| Property | DINOv3 ViT-S/16 in this repo | DINOv3 ConvNeXt-Small in this repo | Consequence for query-conditioned residual fusion |
|---|---|---|---|
| Feature structure | 12 transformer blocks, 384 channels, one 1/16-resolution patch grid; intermediate block outputs can be tapped | Four hierarchical stages, channels 96/192/384/768 at strides 4/8/16/32 | ConvNeXt is naturally multi-scale; ViT offers multiple semantic depths at one spatial scale unless a neck builds a pyramid. |
| Existing extractor | Uses final `last_hidden_state`, drops CLS/register tokens, reshapes to spatial map | Requests `output_hidden_states=True`, selects one stage; all stage maps are already returned | Both need interface work for multi-feature fusion; ViT needs intermediate-block API/taps, ConvNeXt needs return-list support. |
| Query sampling | A query can sample one or several same-grid block maps; concatenate/sum/project their samples | A query can sample taps at multiple spatial scales directly | A small MLP after sampling is a reasonable learned readout for either. For ViT it must learn across depth taps; for ConvNeXt it can also combine spatial scale. |
| Compute/memory shape | Global self-attention over patch tokens; keeping block taps adds activation retention, especially if trainable | Convolutional hierarchical stages; native maps have different token counts, dominated by stride 8 | Frozen backbones avoid backbone backward storage, but ViT's high-resolution global-attention forward cost can be substantial. Need measured latency at the actual padded input size; current data do not establish which is faster end-to-end. |
| Prior simple-head evidence | Frozen ViT-S/16 reached 0.037624 test AP | Frozen ConvNeXt-Small stage 4 reached 0.008311 | In that matched 128px probe, ViT outperformed ConvNeXt; it is a reason to include ViT in the branch comparison, not a guaranteed DiffusionRegPose outcome. |

For a residual branch, the useful abstraction is: decoder query/reference → deformable sample from one or more DINO maps → concatenate or weighted-sum sampled descriptors → compact MLP → gated residual into the pose-capable decoder state. The MLP is the task adapter that can turn generic DINO descriptors into query-specific pose evidence. A zero-initialized scalar/vector gate can preserve the pretrained decoder path initially. This means the branch need not expose DINO maps as global encoder levels to be useful. The experiment should compare DINO architecture and feature set with the same branch/readout, gate, decoder insertion points, preprocessing, trainability policy, and seed. ViT comparisons need clarify whether they test (a) multiple ViT depths at one stride or (b) a neck-generated spatial pyramid; those are different hypotheses.

## Prior simple-head backbone benchmark

The earlier backbone comparison is still present, but its results are embedded in the executed [backbone sweep visualization notebook](../notebooks/visualization/visualize_backbone_sweep.ipynb), rather than a standalone Markdown results report. The historical [`backbone-probe-design.md`](backbone-probe-design.md) describes the benchmark design, not its outcomes. Git history shows the visualization notebook was renamed from `notebook/visualize_backbone_sweep.ipynb` to `notebooks/visualization/visualize_backbone_sweep.ipynb` in the August 16 notebook reorganization; I found no report-deletion commit. The actual per-run configs, test metrics, and checkpoints remain under `results/sweeps/benchmark_backbone/`.

The query-probe runs compared pretrained ResNet-18/34, DINOv3 ConvNeXt-Small, and DINOv3 ViT-S/16 using the same non-backbone resolved config. The four single-backbone configs are identical after removing the `backbone` entry: seed 42, 100 epochs, batch 64, 128-pixel square input, BF16 mixed precision, the same `QueryBackboneProbe` (20 instance slots, 14 keypoints, 256-dimensional features), optimizer, losses, augmentations, data, and evaluator. The reported test checkpoints were selected from validation-loss checkpointing.

| Backbone run | Feature level and spatial grid at 128 input | Train policy | Best logged validation AP | Test AP |
|---|---|---|---:|---:|
| ResNet-18, task 1 | `layer3`, stride 16, 8×8 | ImageNet pretrained, trainable | 0.047019 | 0.048678 |
| ResNet-34, task 2 | `layer3`, stride 16, 8×8 | ImageNet pretrained, trainable | 0.055617 | 0.049862 |
| DINOv3 ConvNeXt-Small, task 3 | stage 4, stride 32, 4×4 | Frozen | 0.009344 | 0.008311 |
| DINOv3 ViT-S/16, task 4 | patch stride 16, 8×8 | Frozen | 0.040335 | 0.037624 |

Under this exact probe, ConvNeXt-Small scored **0.831 AP**, versus **4.868** and **4.986 AP** for ResNet-18/34: a deficit of 4.04–4.16 AP points (about 83% relative). The DINO ViT-S/16 result was much closer to ResNet at 3.762 AP. This is a substantial negative result for the **frozen stage-4 ConvNeXt feature path at 128-pixel input**, and it argues against assuming that adding DINO features will help merely because DINO is pretrained.

It is not evidence that every DINO ConvNeXt stage, multi-stage DINO, or DINO-conditioned DiffusionRegPose will fail. Two material confounds limit that generalization: the tested ConvNeXt used a 4×4 feature grid while the ResNets and DINO ViT used 8×8; and DINO was frozen while the pretrained ResNets were trainable. Thus the benchmark compared practical extractor/configuration combinations, not frozen feature representations under identical adaptation policy. It used one training seed. The available ConvNeXt stage-3 config at stride 16 was not part of these four runs.

The separate [DINO feature probe report](dino.md) supports a more specific hypothesis: ConvNeXt stage 3 has the strongest linearly-decodable joint identity among tested DINO stages, but raw per-token features do not recover sub-patch offsets, and stage1+stage4 concatenation did not improve the offset probe. Therefore DINO features should be evaluated with a query-conditioned learned readout (for example, sampled descriptors followed by an MLP), rather than expected to work from channel concatenation alone. The benchmark result is a warning and a useful baseline, not a direct prediction of the full DiffusionRegPose integration.

Run evidence:

- [ResNet-18 test metrics](../results/sweeps/benchmark_backbone/task_1/20260517_101310/test_results.json), [ResNet-34](../results/sweeps/benchmark_backbone/task_2/20260517_114830/test_results.json), [DINO ConvNeXt-Small](../results/sweeps/benchmark_backbone/task_3/20260517_132549/test_results.json), and [DINO ViT-S/16](../results/sweeps/benchmark_backbone/task_4/20260517_150502/test_results.json).
- The matching resolved configs are in each run's `.hydra/config.yaml`; DINO ConvNeXt uses the default `stage=4` from its config.

## Fusion choices

| Design | Expected behavior and cost | Checkpoint / feasibility | Assessment |
|---|---|---|---|
| Replace ResNet pyramid with DINO stages | Same nominal four levels if adapted; avoids a second memory pyramid. Changes backbone, feature statistics, input projections, proposal features, and all learned co-adaptation. | EDPose/ResNet checkpoint no longer describes the visual path. Requires broad retraining or transfer study. | Not a first step. |
| Append DINO stages to common encoder/decoder memory | Both encoders learn to mix both sources and all decoder layers can sample both. Stages 2–4 make 7 levels; appending all four DINO levels makes 8. | Changes `level_embed`, each deformable-attention offsets/weights output shape, encoder memory, and proposal selection features; requires explicit checkpoint remapping, new-level initialization, and proposal-scale semantics for duplicate strides. Highest runtime/memory risk. | Direct full-fusion experiment; stage 2–4 is the relevant first candidate. |
| Separate DINO query-conditioned branch | Keeps the ResNet encoder/proposals and checkpoint-shaped attention intact; a second deformable module reads selected DINO maps using decoder query references and adds a gated residual. No DINO tokens pass through the six-layer encoder. | New projections/attention weights and a new gate; load the existing model unchanged, initialize the branch separately. Any DINO path adds one DINO trunk forward versus current ResNet-only; tapping more stages in that same forward adds retained feature storage, not a second DINO pass. | Lower-cost diagnostic for decoder pose refinement; does not test encoder mixing or DINO-conditioned proposal selection. |
| Single pooled/global DINO vector | Small feature map and low cost, but discards spatial localization and cannot test multi-stage dense features. | Easy interface, weak match to the question. | Useful only as a later control. |

Zero gate initialization preserves the baseline decoder output at initialization and makes the DINO branch's influence learnable. The first backward pass should be checked explicitly: with a zero gate, branch weights receive no gradient on that first pass, while the gate can receive a gradient. The training test must verify this expected gradient path and that the branch begins affecting outputs after the gate changes. Freeze and eval mode are separate requirements for a deterministic pretrained feature path.

### Why not just set `num_feature_levels=7`?

At the attention-operator level, seven shared levels are supported and are a valid design. The separate branch is not a prerequisite; it was suggested to cap encoder cost and preserve the existing encoder/proposal path while testing whether query-conditioned DINO features help pose refinement. For a full shared-pyramid experiment, the direct comparison should be the current four-level baseline against four ResNet levels plus DINO stages 2–4.

There are two implementation details behind “just expand”:

1. In the current `prepare_transformer_inputs`, increasing `num_feature_levels` alone does not add DINO. The method projects the backbone's three ResNet maps and automatically creates any remaining levels by downsampling the last ResNet feature. The DINO stage maps must be extracted and projected explicitly, with their masks and positional encodings appended to the flattened memory.
2. The seven-level choice changes the six encoder and six decoder attention projections and `level_embed`, so old checkpoint tensors need deliberate remapping/initialization. Also, `gen_encoder_output_proposals` currently makes proposal width/height depend on the level's list index (`0.05 × 2**level`, or the learned base width times that factor). If DINO strides 8/16/32 are simply appended as levels 4/5/6, their default proposal scales become 0.8/1.6/3.2 before validity filtering, despite being high-resolution maps. A seven-level implementation must define proposal scales by intended stride/anchor semantics, including duplicate stride levels, rather than inheriting those indices blindly. This proposal-path adjustment matters because encoder proposals initialize the decoder queries.

The distinction is scope and cost: shared seven-level memory gives DINO access to encoder mixing and proposal selection, at the estimated 2.10× encoder MAC floor and 1.32× decoder floor for stages 2–4. The separate decoder branch keeps the original encoder cost and proposal selection, but tests only the decoder refinement contribution. If the research question is whether DINO stages improve the full multi-scale image memory, start with the seven-level shared design and handle level/proposal semantics explicitly. If the first question is whether DINO can help pose queries at bounded cost, the branch is a narrower diagnostic.

## Runtime and memory evidence

Existing measurements are for an RTX 3090, PyTorch 2.4.1+cu121, FP32, the native deformable-attention operator, three warmups, and ten measured updates. They provide baselines, not DINO-addition measurements.

| Existing profile | Padded batch input | Median update | Forward | Peak allocated |
|---|---:|---:|---:|---:|
| Paper architecture, batch 1 | 800×1216 | 380.9 ms | 118.3 ms | 4.04 GiB |
| Checkpoint-compatible 256 preset, batch 8 | 384×427 | 681.9 ms | 165.5 ms | 6.10 GiB |

The runtime report also measured the selected development model's batch-16 backbone-plus-position forward at 23.9 ms, but that is a different reduced architecture and should not be used to predict DINO latency.

### Token-count estimate

At the measured paper input 800×1216, current ResNet levels are approximately 100×152, 50×76, 25×38, and 13×19: about 20,197 tokens total. Four DINO ConvNeXt stages at strides 4/8/16/32 are 200×304, 100×152, 50×76, and 25×38: about 80,750 tokens. Appending all four maps yields about 100,947 total memory tokens, roughly **5.0×** the current memory length. The exact odd-size shapes follow convolution rounding, and these counts exclude masked padding differences.

At the low-resolution profile's padded 384×427 input, the same arithmetic gives roughly 3,450 current ResNet tokens and 13,500 four-stage DINO tokens. Appending the full DINO pyramid would produce about 4.9× current memory. The DINO stride-4 map alone contributes about 10,176 of those 13,500 DINO tokens at this shape; it is the main reason all-stage concatenation is unattractive.

This token ratio is **not** a wall-clock multiplier. Dense deformable attention samples a fixed number of points, but the flattened memory still flows through value projections, masks, encoder layers, and decoder value projections. DINO trunk latency, map projection cost, backward state, allocator peaks, and the native kernel's level scaling are unmeasured here. The current DINO extractor already asks Hugging Face for all hidden states, so using more stages from that DINO model does not require another trunk pass. Adding DINO to the ResNet model does add a DINO backbone forward, whose latency is unknown. A separate decoder branch avoids multiplying the six encoder's memory workload, but adds dense DINO map storage/projections and query-side work in each enabled decoder layer. Frozen DINO avoids backbone-gradient storage but not weights, forward compute, input/output activations, or downstream projection/attention gradients.

Do not infer GPU memory fit or per-step overhead from token counts alone. The current full paper profile peaks at 4.04 GiB allocated with batch 1, and the compatible 256 profile at 6.10 GiB with batch 8, but neither includes DINO.

### Attention and projection cost by fusion choice

For comparison, use the operation floor from the runtime audit with hidden width `d=256`, FFN width `f=2048`, eight heads, four sample points, six encoder layers, and six decoder layers. Let `S` be flattened memory length, `L` attention levels, and `Q` decoder queries in a layer. The audit's encoder floor is

```text
encoder MACs = 6 * S * [2d² + 3d(heads * L * points) + 2df]
```

Its decoder floor includes a per-layer memory value projection `S*d²`, deformable-attention query projections/sampling weights, query self-attention, and FFN. At paper resolution, training has approximately 1,000 decoder queries in each of the first two layers and 1,600 in each of the four expanded-pose layers (normal plus DN queries). Applying those formulae to the 800×1216 token counts above gives:

| Image memory design | `S` | `L` | Encoder floor, 6 layers | Relative encoder floor | Decoder floor, 6 layers | Relative decoder floor |
|---|---:|---:|---:|---:|---:|---:|
| Existing ResNet pyramid | 20,197 | 4 | 154.9 GMAC | 1.00× | 26.6 GMAC | 1.00× |
| Add DINO stages 2–4 to shared memory | 40,147 | 7 | 325.6 GMAC | 2.10× | 35.1 GMAC | 1.32× |
| Add all DINO stages to shared memory | 100,947 | 8 | 833.6 GMAC | 5.38× | 59.2 GMAC | 2.22× |

The decoder does not scale by the full memory-token ratio because query self-attention and FFNs are unchanged. Its memory value projections do scale with `S`, and the deformable query projections scale with `L`. These operation floors omit backbone compute, kernel launch and memory costs, deformable interpolation, normalization/activation, and the full backward pass. They are comparable estimates under one shape/layout, not latency predictions.

The native deformable operator evaluates `Q × heads × levels × points` sample sites per layer. Relative to the existing four-level model, encoder sampling sites increase by about **3.48×** for stages 2–4 added to memory and **10.00×** for all four stages added. Decoder sampling sites per layer increase by **1.75×** or **2×**, respectively. These site counts explain why level count matters even though the sampled point count is fixed per level.

For a separate DINO branch using stages 2–4 only in the four expanded-pose layers, the extra branch itself has an estimated **6.1 GMAC** decoder floor per 800×1216 training image, about 23% of the existing decoder floor. This counts its DINO value projections and query attention projections/output projections. Projecting the DINO stage maps to 256 channels adds about **1.31 GMAC** once per image; DINO ConvNeXt feature extraction is excluded and unmeasured. The branch's 614,400 sample sites across those four layers are about 57% of the existing decoder's 1,075,200 sites across all six layers. This does not mean 57% more decoder time: the branch has fewer layers than the full decoder but distinct memory traffic, module launches, and feature projection cost.

At batch size `B`, multiply these per-image operation counts by `B`; peak memory also depends on which activations and gradients are retained. A frozen DINO trunk still materializes its output maps for downstream training. The stride-4 DINO map alone is 60,800 tokens per paper-shape image (1.49 GMAC for a 96→256 1×1 projection); its fine spatial grid dominates the cost of the four-stage option. Stages 2–4 are close to the existing pyramid's combined token count, but add distinct learned representations at strides 8/16/32.

Feature-tensor storage can be bounded separately from model activation memory. At paper shape, raw FP32 DINO stages 2–4 contain about 5.11 million values (20.4 MB/image), and their projected 256-channel maps contain the same number; together that is about 40.9 MB/image or 327 MB for batch 8. All four raw stages contain 10.94 million values (43.8 MB/image), while their projected maps contain 20.67 million values (82.7 MB/image); raw plus projected is about 126.5 MB/image or 1.01 GB for batch 8. These bounds exclude masks, positional tensors, allocator overhead, and encoder/decoder activations. Mixed precision reduces storage bytes but not token counts.

The shared-memory MAC table excludes the one-time 1×1 DINO input projections. Those add about 1.31 GMAC/image for stages 2–4, plus 1.49 GMAC/image for stage 1 if all four are appended. The DINO ConvNeXt forward is also excluded and remains to be measured; it is common to all integration choices using that same second DINO network.

### Mechanism and checkpoint effects

The level count is encoded in learned tensor dimensions, not just metadata. Each `MultiScaleDeformableAttention` has `sampling_offsets` output width `heads × levels × points × 2` and `attention_weights` output width `heads × levels × points`. Changing 4 levels to 7/8 changes those layers in all six encoder and six decoder modules, and the learned `level_embed` changes from `[4,256]` to `[7,256]` or `[8,256]`. Existing per-level weights could be copied into preserved level slots, but added level slices and embeddings need explicit initialization. Shared-memory addition also changes what the encoder transforms and which flattened tokens feed two-stage proposal ranking; it is not a side branch that leaves proposal semantics fixed.

The separate-branch option adds new projections and new DINO deformable-attention module weights but leaves existing four-level offsets, weights, level embeddings, encoder memory, and encoder proposal ranking shapes intact. It therefore has the smallest checkpoint remapping surface. With a zero residual gate, the initial output path can match the existing decoder; the gate's first gradient and the branch's later gradient must be verified. If the DINO branch is restricted to the four pose-capable decoder layers, its result says nothing about benefits to the first two box-only layers or the encoder's proposal ranking. This tradeoff motivates the branch; it does not mean the seven-level shared design cannot be implemented.

| Option | Dense memory seen by encoder | Attention changes | Proposal path | Checkpoint loading | Main additional cost |
|---|---|---|---|---|---|
| Replace ResNet maps with DINO | Similar level/token budget if selected maps align | Existing modules still see different feature distributions; may alter input projection contracts | DINO features directly drive ranking | ResNet/EDPose feature path is replaced; broad transfer adaptation | DINO trunk; adaptation/retraining risk |
| Append DINO stages 2–4 | About 2× tokens, 7 total levels | All encoder/decoder attention layers expand 4→7 | Encoder proposals now rank mixed ResNet+DINO memory | Level embeddings and all 12 offset/weight projections need new slices | About 2.10× encoder MAC floor; about 1.32× decoder floor; +1.31 GMAC map projection/image; plus DINO trunk |
| Append DINO stages 1–4 | About 5× tokens, 8 total levels | All modules expand 4→8 | Same mixed-memory proposal change | Same remapping across all 12 modules, with more new levels | About 5.38× encoder MAC floor; about 2.22× decoder floor; +2.80 GMAC map projection/image; plus DINO trunk |
| Separate stages-2/3/4 decoder branch | Existing encoder remains at 4 ResNet levels | Four new 3-level query-side attention modules if used only after expansion | Encoder ranking remains unchanged | Existing EDPose tensors keep their shapes; branch is newly initialized | About 6.1 GMAC decoder floor + 1.31 GMAC map projection/image + unmeasured DINO trunk |

## Proposed next experiment

After a separate implementation plan is reviewed, use the checkpoint-compatible 256 preset as the controlled starting point and run a **runtime/contract pilot before AP training**:

1. Baseline: existing preset and one fixed real batch from the normal transformed input path.
2. Shared-pyramid candidate: preserve the four ResNet levels and append projected DINO stages 2–4, for seven levels total. Define and test proposal scales for duplicate stride-8/16/32 maps, remap the original four attention level slices, and initialize the three new slices and level embeddings. This is the primary experiment for full shared-memory DINO fusion.
3. Optional decoder-branch control: if encoder cost is prohibitive, compare stage-4-only and stages-2–4 in the separate query-conditioned branch. This measures the marginal value of multi-stage taps within that narrower design; it does not substitute for the shared-pyramid comparison.
4. Profile baseline and seven-level candidate with batch 1 first (to protect current GPU headroom), three warmups, and ten measured forward/backward updates. Record DINO extraction, projection, encoder, decoder, whole update, peak allocated/reserved memory, valid-token shapes, and proposal-scale/valid-proposal distributions. Synchronize CUDA around timed regions. Because the same DINO ConvNeXt forward produces all hidden states, multi-stage taps add retention/projection/fusion overhead beyond the common trunk-forward cost.
5. Gate continuation on exact image/feature coordinate alignment under a deterministic resize and a training crop/flip, finite outputs/loss/gradients, expected gate/branch gradient behavior, successful loading of all existing checkpoint tensors, and a batch-1 memory fit. If that passes, profile batch 4 at the same input size before selecting a training batch.
6. Only after the runtime gate, run a short, matched learning pilot from the same initialized checkpoint and seed: four-level baseline versus seven-level shared memory. Track validation keypoint AP, bbox AP, peak memory, update time, and encoder proposal validity/scale distribution. Treat this as a screening result; require a longer matched run and later seed replication for a positive conclusion. If that shared candidate is too costly, run the branch controls as a separate question.

The exact training duration, validation checkpoints, and GPU launch should be selected in that follow-up plan after checking active jobs and available memory. No profiling/training job was launched for this report.

## Risks and decisions to keep explicit

- **Preprocessing:** current data batches are already ImageNet-normalized. Avoid silently double-normalizing; pad masks, resize/crop/flip coordinates, and DINO patch/stage centers must agree.
- **Feature scale:** DINO's pretrained feature distributions and channels differ from EDPose ResNet features. New per-stage projections and the query fusion need controlled initialization; do not assume matching channels imply compatibility.
- **Checkpoint loading:** preserve and report exact loaded tensor coverage. Common-memory expansion changes deformable-attention parameter shapes; a separate branch is lower-risk but still adds new parameters.
- **Learning claim:** zero-initialized fusion protects the starting forward function but does not establish benefit. Evaluate AP on the same validation split and checkpoint protocol.
- **Resource claim:** all added DINO runtime and peak-memory impact remains unknown until matched profiling.
- **Model license:** the selected DINOv3 model is released under Meta's DINOv3 License Agreement, not a generic permissive code license. Preserve its terms and publication acknowledgment requirements if results are published.

## Sources

- Local source and measurements are linked inline above; especially [`runtime_paper_b1.json`](diffusionregpose_runtime/runtime_paper_b1.json) and [`runtime_development_compatible_256.json`](diffusionregpose_runtime/runtime_development_compatible_256.json).
- Tan et al., [DiffusionRegPose, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Tan_DiffusionRegPose_Enhancing_Multi-Person_Pose_Estimation_using_a_Diffusion-Based_End-to-End_Regression_CVPR_2024_paper.pdf).
- Meta, [DINOv3 project page](https://ai.meta.com/research/dinov3/) and [official ConvNeXt implementation](https://github.com/facebookresearch/dinov3/blob/main/dinov3/models/convnext.py).
- Meta, [DINOv3 license](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md); the selected [Hugging Face checkpoint card](https://huggingface.co/facebook/dinov3-convnext-small-pretrain-lvd1689m) identifies its license as `dinov3-license`.

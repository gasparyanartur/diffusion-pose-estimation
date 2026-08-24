#!/usr/bin/env python3
"""Compare the native DiffusionRegPose forward path with the reference model.

The reference repository imports a compiled CUDA operator unconditionally.  This
probe installs the operator's documented PyTorch fallback before importing the
reference model, which keeps the comparison independent of the local CUDA ABI.
"""

from __future__ import annotations

import argparse
import sys
import types
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn.functional as functional


ROOT = Path(__file__).resolve().parents[2]
EXTERNAL = ROOT / "external" / "DiffusionRegPose"
CHECKPOINT = EXTERNAL / "logs" / "crowdpose_r50_effective_batch8_40epoch" / "checkpoint.pth"


def build_inference_attention_mask(batch_size: int, group_count: int, body_point_count: int, device):
    group_size = body_point_count + 1
    query_count = group_count * group_size
    mask = torch.zeros(batch_size, 8, query_count, query_count, dtype=torch.bool, device=device)
    for query_index in range(query_count):
        group_start = (query_index // group_size) * group_size
        group_end = group_start + group_size
        mask[:, :, query_index, :group_start] = True
        mask[:, :, query_index, group_end:] = True
    box_indices = torch.arange(query_count, device=device)
    box_indices = box_indices[box_indices.remainder(group_size) == 0]
    mask[:, :, box_indices[:, None], box_indices[None, :]] = False
    return mask.flatten(0, 1)


def install_deformable_attention_fallback() -> None:
    operator = types.ModuleType("MultiScaleDeformableAttention")

    def forward(value, spatial_shapes, level_start_index, sampling_locations, attention_weights, _step):
        batch_size, _, num_heads, head_dim = value.shape
        _, query_count, _, level_count, point_count, _ = sampling_locations.shape
        values = value.split(tuple(height * width for height, width in spatial_shapes.tolist()), dim=1)
        sampled = []
        for level, (height, width) in enumerate(spatial_shapes.tolist()):
            level_value = values[level].flatten(2).transpose(1, 2).reshape(
                batch_size * num_heads, head_dim, height, width
            )
            grid = (2.0 * sampling_locations[:, :, :, level] - 1.0).transpose(1, 2).flatten(0, 1)
            sampled.append(
                functional.grid_sample(
                    level_value, grid, mode="bilinear", padding_mode="zeros", align_corners=False
                )
            )
        weights = attention_weights.transpose(1, 2).reshape(
            batch_size * num_heads, 1, query_count, level_count * point_count
        )
        output = (torch.stack(sampled, dim=-2).flatten(-2) * weights).sum(-1)
        return output.view(batch_size, num_heads * head_dim, query_count).transpose(1, 2).contiguous()

    operator.ms_deform_attn_forward = forward
    operator.ms_deform_attn_backward = lambda *args: (_ for _ in ()).throw(
        RuntimeError("backward is not supported by the parity probe")
    )
    sys.modules[operator.__name__] = operator


def load_reference_model(query_count: int, group_count: int):
    sys.path.insert(0, str(EXTERNAL))
    install_deformable_attention_fallback()

    from models.diffusionregpose.backbones import backbone as reference_backbone
    from models.diffusionregpose.diffusionregpose import build_diffusionregpose
    from util.config import Config

    reference_backbone.is_main_process = lambda: False
    config = Config.fromfile(str(EXTERNAL / "config" / "diffusionregpose.cfg.py"))
    values = config._cfg_dict.to_dict()
    values.update(
        device="cpu",
        eval=False,
        num_queries=query_count,
        num_group=group_count,
        num_proposals=group_count,
        use_dn=False,
        dn_number=0,
    )
    args = Namespace(**values)
    model, _, _ = build_diffusionregpose(args)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model_state = {
        key: value
        for key, value in checkpoint["model"].items()
        if key not in {
            "betas",
            "alphas_cumprod",
            "alphas_cumprod_prev",
            "sqrt_alphas_cumprod",
            "sqrt_one_minus_alphas_cumprod",
            "log_one_minus_alphas_cumprod",
            "sqrt_recip_alphas_cumprod",
            "sqrt_recipm1_alphas_cumprod",
            "posterior_variance",
            "posterior_log_variance_clipped",
            "posterior_mean_coef1",
            "posterior_mean_coef2",
        }
    }
    load_result = model.load_state_dict(model_state, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(load_result)
    model.eval()
    for name in (
        "betas",
        "alphas_cumprod",
        "alphas_cumprod_prev",
        "sqrt_alphas_cumprod",
        "sqrt_one_minus_alphas_cumprod",
        "log_one_minus_alphas_cumprod",
        "sqrt_recip_alphas_cumprod",
        "sqrt_recipm1_alphas_cumprod",
        "posterior_variance",
        "posterior_log_variance_clipped",
        "posterior_mean_coef1",
        "posterior_mean_coef2",
    ):
        setattr(model, name, checkpoint["model"][name])
    model.scale = 6.0
    model.num_proposals = group_count
    return model


def reference_prediction(
    model, images: torch.Tensor, diffusion_keypoints: torch.Tensor, timestep: int, diagnostics=None
):
    from util.misc import NestedTensor

    samples = NestedTensor(
        images, torch.zeros(images.shape[0], *images.shape[2:], dtype=torch.bool, device=images.device)
    )
    features, positions = model.backbone(samples)
    sources, masks = [], []
    for feature in features:
        source, mask = feature.decompose()
        sources.append(model.input_proj[len(sources)](source))
        masks.append(mask)
    while len(sources) < model.num_feature_levels:
        source = model.input_proj[len(sources)](
            features[-1].tensors if len(sources) == len(features) else sources[-1]
        )
        mask = functional.interpolate(samples.mask[:, None].float(), size=source.shape[-2:]).to(torch.bool)[:, 0]
        sources.append(source)
        masks.append(mask)
        positions.append(model.backbone[1](NestedTensor(source, mask)).to(source.dtype))
    model.transformer.eval()
    handles = []
    if diagnostics is not None:
        handles.append(
            model.transformer.encoder.register_forward_hook(
                lambda _module, _inputs, output: diagnostics.update(reference_memory=output[0].detach())
            )
        )
        handles.append(
            model.transformer.enc_output_norm.register_forward_hook(
                lambda _module, _inputs, output: diagnostics.update(reference_proposals=output.detach())
            )
        )
        handles.append(
            model.transformer.decoder.register_forward_hook(
                lambda _module, _inputs, output: diagnostics.update(reference_decoder=output)
            )
        )
        handles.append(
            model.transformer.decoder.ref_point_head.register_forward_hook(
                lambda _module, _inputs, output: (
                    diagnostics.setdefault("reference_query_pos", output.detach()), None
                )[1]
            )
        )
        handles.append(
            model.transformer.decoder.ref_point_head.register_forward_pre_hook(
                lambda _module, inputs: (
                    diagnostics.setdefault("reference_query_sine", inputs[0].detach()), None
                )[1]
            )
        )
        handles.append(
            model.transformer.decoder.layers[0].register_forward_pre_hook(
                lambda _module, _inputs, kwargs: (
                    diagnostics.setdefault("reference_decoder_reference", kwargs["tgt_reference_points"].detach()),
                    None,
                )[1],
                with_kwargs=True,
            )
        )
        handles.append(
            model.transformer.decoder.layers[2].register_forward_pre_hook(
                lambda _module, inputs, kwargs: (
                    diagnostics.setdefault("reference_expanded_query", kwargs["tgt"].detach()),
                    diagnostics.setdefault("reference_expanded_reference", kwargs["tgt_reference_points"].detach()),
                    diagnostics.setdefault("reference_expanded_query_pos", kwargs["tgt_query_pos"].detach()),
                    None,
                )[3],
                with_kwargs=True,
            )
        )
        handles.append(
            model.transformer.decoder.layers[2].self_attn.register_forward_hook(
                lambda _module, _inputs, output: diagnostics.update(reference_masked_self_attention=output[0].detach())
            )
        )
    inference_mask = build_inference_attention_mask(
        images.shape[0], diffusion_keypoints.shape[1], 14, images.device
    )
    _, output = model.model_predictions(
        torch.tensor(
            [[images.shape[-1], images.shape[-2], images.shape[-1], images.shape[-2]]],
            device=images.device,
        ),
        None,
        sources,
        masks,
        diffusion_keypoints * 2.0 - 1.0,
        torch.tensor([timestep], device=images.device),
        positions,
        None,
        None,
        inference_mask,
        torch.tensor([timestep], device=images.device),
        None,
    )
    for handle in handles:
        handle.remove()
    return output


def native_prediction(images, diffusion_keypoints, timestep, query_count, group_count, diagnostics=None):
    from src.models.pipelines.pose_estimators.diffusionregpose.model import DiffusionRegPoseModel

    model = DiffusionRegPoseModel(
        backbone_pretrained=False,
        num_queries=query_count,
        num_proposals=group_count,
        num_group=group_count,
    )
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(images.device).eval()
    handle = None
    if diagnostics is not None:
        handle = model.transformer.decoder.register_forward_hook(
            lambda _module, _inputs, output: diagnostics.update(native_decoder=output)
        )
        query_pos_handle = None
        if diagnostics is not None:
            query_pos_handle = model.transformer.decoder.ref_point_head.register_forward_hook(
                lambda _module, _inputs, output: (
                    diagnostics.setdefault("native_query_pos", output.detach()), None
                )[1]
            )
            query_sine_handle = model.transformer.decoder.ref_point_head.register_forward_pre_hook(
                lambda _module, inputs: (
                    diagnostics.setdefault("native_query_sine", inputs[0].detach()), None
                )[1]
            )
            decoder_reference_handle = model.transformer.decoder.layers[0].register_forward_pre_hook(
                lambda _module, _inputs, kwargs: (
                    diagnostics.setdefault("native_decoder_reference", kwargs["tgt_reference_points"].detach()),
                    None,
                )[1],
                with_kwargs=True,
            )
            expanded_query_handle = model.transformer.decoder.layers[2].register_forward_pre_hook(
                lambda _module, inputs, kwargs: (
                    diagnostics.setdefault("native_expanded_query", inputs[0].detach()),
                    diagnostics.setdefault("native_expanded_reference", kwargs["tgt_reference_points"].detach()),
                    diagnostics.setdefault("native_expanded_query_pos", kwargs["tgt_query_pos"].detach()),
                    None,
                )[3],
                with_kwargs=True,
            )
            masked_self_attention_handle = model.transformer.decoder.layers[2].self_attn.register_forward_hook(
                lambda _module, _inputs, output: diagnostics.update(native_masked_self_attention=output[0].detach())
            )
    with torch.no_grad():
        output = model(
            images,
            diffusion_keypoints * 2.0 - 1.0,
            torch.tensor([timestep], device=images.device),
        )
    if handle is not None:
        handle.remove()
    if diagnostics is not None and query_pos_handle is not None:
        query_pos_handle.remove()
        query_sine_handle.remove()
        decoder_reference_handle.remove()
        expanded_query_handle.remove()
        masked_self_attention_handle.remove()
    return output, model


def compare_frontend(native_model, reference_model, images: torch.Tensor) -> None:
    from util.misc import NestedTensor

    padding_mask = torch.zeros(
        images.shape[0], *images.shape[2:], dtype=torch.bool, device=images.device
    )
    native_features, native_masks, native_positions = native_model.backbone(images, padding_mask)
    reference_features, reference_positions = reference_model.backbone(
        NestedTensor(images, padding_mask)
    )

    for index, (native_feature, reference_feature) in enumerate(zip(native_features, reference_features)):
        reference_tensor = reference_feature.tensors
        difference = (native_feature - reference_tensor).abs()
        print(
            f"backbone[{index}]: max_abs={difference.max().item():.6g} "
            f"mean_abs={difference.mean().item():.6g}"
        )
        position_difference = (native_positions[index] - reference_positions[index]).abs()
        print(
            f"position[{index}]: max_abs={position_difference.max().item():.6g} "
            f"mean_abs={position_difference.mean().item():.6g}"
        )
        native_projection = native_model.input_proj[index](native_feature)
        reference_projection = reference_model.input_proj[index](reference_tensor)
        projection_difference = (native_projection - reference_projection).abs()
        print(
            f"projection[{index}]: max_abs={projection_difference.max().item():.6g} "
            f"mean_abs={projection_difference.mean().item():.6g}"
        )


def compare_attention_operator() -> None:
    from src.models.pipelines.pose_estimators.diffusionregpose.transformer import MultiScaleDeformableAttention

    torch.manual_seed(11)
    attention = MultiScaleDeformableAttention(
        d_model=32, num_levels=2, num_heads=4, num_points=3
    ).eval()
    query = torch.randn(1, 5, 32)
    shapes = torch.tensor([[2, 3], [1, 2]])
    starts = torch.tensor([0, 6])
    flattened = torch.randn(1, 8, 32)
    references = torch.rand(1, 5, 2, 4)
    value = attention.value_proj(flattened).view(1, 8, 4, 8)
    offsets = attention.sampling_offsets(query).view(1, 5, 4, 2, 3, 2)
    weights = attention.attention_weights(query).view(1, 5, 4, 2 * 3).softmax(-1)
    weights = weights.view(1, 5, 4, 2, 3)
    locations = references[:, :, None, :, None, :2] + offsets / 3.0 * references[:, :, None, :, None, 2:] * 0.5
    fallback_values = value.split((6, 2), dim=1)
    fallback_samples = []
    for level, (height, width) in enumerate(shapes.tolist()):
        level_value = fallback_values[level].flatten(2).transpose(1, 2).reshape(4, 8, height, width)
        grid = (2.0 * locations[:, :, :, level] - 1.0).transpose(1, 2).flatten(0, 1)
        fallback_samples.append(functional.grid_sample(level_value, grid, align_corners=False))
    fallback_weights = weights.transpose(1, 2).reshape(4, 1, 5, 6)
    fallback = (torch.stack(fallback_samples, dim=-2).flatten(-2) * fallback_weights).sum(-1)
    fallback = fallback.view(1, 32, 5).transpose(1, 2)
    native = attention(query, references, flattened, shapes, starts)
    difference = (native - attention.output_proj(fallback)).abs()
    print(
        f"attention_operator: max_abs={difference.max().item():.6g} "
        f"mean_abs={difference.mean().item():.6g}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--queries", type=int, default=32)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--timestep", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--assert-tolerance",
        type=float,
        default=None,
        help="Fail when any final output max error exceeds this absolute tolerance.",
    )
    args = parser.parse_args()

    torch.manual_seed(7)
    device = torch.device(args.device)
    images = torch.rand(1, 3, args.height, args.width, device=device)
    diffusion_keypoints = torch.rand(1, args.groups, 14 * 2, device=device)
    diagnostics = {}
    native, native_model = native_prediction(
        images, diffusion_keypoints, args.timestep, args.queries, args.groups, diagnostics
    )
    reference_model = load_reference_model(args.queries, args.groups)
    reference_model.to(device)
    compare_attention_operator()
    compare_frontend(native_model, reference_model, images)
    native_inputs = native_model.prepare_transformer_inputs(
        images, torch.zeros(1, args.height, args.width, dtype=torch.bool, device=device)
    )
    native_memory = native_model.encode_features(
        native_inputs, torch.tensor([args.timestep], device=device)
    )
    reference = reference_prediction(
        reference_model,
        images,
        diffusion_keypoints,
        args.timestep,
        diagnostics,
    )
    native_decoder_features, _ = diagnostics["native_decoder"]
    reference_decoder_features, _ = diagnostics["reference_decoder"]
    query_position_difference = (
        diagnostics["native_query_pos"].float() - diagnostics["reference_query_pos"].float()
    ).abs()
    print(
        f"decoder_query_position: max_abs={query_position_difference.max().item():.6g} "
        f"mean_abs={query_position_difference.mean().item():.6g}"
    )
    query_sine_difference = (
        diagnostics["native_query_sine"].float() - diagnostics["reference_query_sine"].float()
    ).abs()
    print(
        f"decoder_query_sine: max_abs={query_sine_difference.max().item():.6g} "
        f"mean_abs={query_sine_difference.mean().item():.6g}"
    )
    decoder_reference_difference = (
        diagnostics["native_decoder_reference"].float()
        - diagnostics["reference_decoder_reference"].float()
    ).abs()
    print(
        f"decoder_reference_input: max_abs={decoder_reference_difference.max().item():.6g} "
        f"mean_abs={decoder_reference_difference.mean().item():.6g}"
    )
    expanded_query_difference = (
        diagnostics["native_expanded_query"].float() - diagnostics["reference_expanded_query"].float()
    ).abs()
    expanded_reference_difference = (
        diagnostics["native_expanded_reference"].float()
        - diagnostics["reference_expanded_reference"].float()
    ).abs()
    print(
        f"expanded_query: max_abs={expanded_query_difference.max().item():.6g} "
        f"mean_abs={expanded_query_difference.mean().item():.6g}"
    )
    print(
        f"expanded_reference: max_abs={expanded_reference_difference.max().item():.6g} "
        f"mean_abs={expanded_reference_difference.mean().item():.6g}"
    )
    expanded_query_pos_difference = (
        diagnostics["native_expanded_query_pos"].float()
        - diagnostics["reference_expanded_query_pos"].float()
    ).abs()
    print(
        f"expanded_query_pos: max_abs={expanded_query_pos_difference.max().item():.6g} "
        f"mean_abs={expanded_query_pos_difference.mean().item():.6g}"
    )
    masked_self_attention_difference = (
        diagnostics["native_masked_self_attention"].float()
        - diagnostics["reference_masked_self_attention"].float()
    ).abs()
    print(
        f"masked_self_attention: max_abs={masked_self_attention_difference.max().item():.6g} "
        f"mean_abs={masked_self_attention_difference.mean().item():.6g}"
    )
    from src.models.pipelines.pose_estimators.diffusionregpose.transformer import sine_embedding_for_position

    reference_position = diagnostics["reference_decoder_reference"][:, :, 0, :]
    native_sine_on_reference = sine_embedding_for_position(reference_position)
    dim_t = 10000 ** (
        2 * (torch.arange(128, dtype=torch.float32, device=reference_position.device) // 2) / 128
    )
    external_parts = []
    for index in ([1, 0] if reference_position.shape[-1] == 2 else [1, 0, 2, 3]):
        coordinate = reference_position[..., index]
        values = coordinate[..., None] * (2 * torch.pi) / dim_t
        external_parts.append(torch.stack((values[..., 0::2].sin(), values[..., 1::2].cos()), dim=3).flatten(2))
    external_sine = torch.cat(external_parts, dim=2)
    native_formula_difference = (native_sine_on_reference - external_sine).abs()
    print(
        f"sine_formula: max_abs={native_formula_difference.max().item():.6g} "
        f"mean_abs={native_formula_difference.mean().item():.6g}"
    )
    native_query_sine_formula_difference = (
        diagnostics["native_query_sine"].float() - sine_embedding_for_position(
            diagnostics["native_decoder_reference"][:, :, 0, :]
        ).float()
    ).abs()
    reference_query_sine_formula_difference = (
        diagnostics["reference_query_sine"].float() - external_sine.float()
    ).abs()
    print(
        f"native_sine_input: max_abs={native_query_sine_formula_difference.max().item():.6g} "
        f"mean_abs={native_query_sine_formula_difference.mean().item():.6g}"
    )
    print(
        f"reference_sine_input: max_abs={reference_query_sine_formula_difference.max().item():.6g} "
        f"mean_abs={reference_query_sine_formula_difference.mean().item():.6g}"
    )
    for index, (native_layer, reference_layer) in enumerate(
        zip(native_decoder_features, reference_decoder_features)
    ):
        layer_difference = (native_layer.float() - reference_layer.float()).abs()
        print(
            f"decoder_features[{index}]: max_abs={layer_difference.max().item():.6g} "
            f"mean_abs={layer_difference.mean().item():.6g}"
        )
    memory_difference = (native_memory - diagnostics["reference_memory"]).abs()
    print(
        f"encoder_memory: max_abs={memory_difference.max().item():.6g} "
        f"mean_abs={memory_difference.mean().item():.6g}"
    )
    from src.models.pipelines.pose_estimators.diffusionregpose.model import gen_encoder_output_proposals

    native_proposal_memory, _ = gen_encoder_output_proposals(
        native_memory,
        native_inputs.padding_mask,
        native_inputs.spatial_shapes,
    )
    native_proposal_memory = native_model.transformer.enc_output_norm(
        native_model.transformer.enc_output(native_proposal_memory)
    )
    proposal_difference = (native_proposal_memory - diagnostics["reference_proposals"]).abs()
    print(
        f"proposal_memory: max_abs={proposal_difference.max().item():.6g} "
        f"mean_abs={proposal_difference.mean().item():.6g}"
    )
    native_queries, native_references, _, _ = native_model.select_two_stage_queries(
        native_memory, native_inputs
    )
    from models.diffusionregpose.utils import gen_encoder_output_proposals as reference_proposals

    _, reference_proposal_logits = reference_proposals(
        diagnostics["reference_memory"],
        native_inputs.padding_mask,
        native_inputs.spatial_shapes,
    )
    reference_class_logits = reference_model.transformer.enc_out_class_embed(
        diagnostics["reference_proposals"]
    )
    reference_box_logits = reference_model.transformer.enc_out_bbox_embed(
        diagnostics["reference_proposals"]
    ) + reference_proposal_logits
    reference_indices = torch.topk(
        reference_class_logits.max(-1).values, args.queries, dim=1
    ).indices
    reference_queries = torch.gather(
        diagnostics["reference_proposals"], 1, reference_indices[..., None].expand(-1, -1, 256)
    )
    reference_references = torch.gather(
        reference_box_logits, 1, reference_indices[..., None].expand(-1, -1, 4)
    )
    query_difference = (native_queries - reference_queries.detach()).abs()
    reference_difference = (native_references - reference_references.detach()).abs()
    finite_reference_difference = reference_difference[torch.isfinite(reference_difference)]
    print(
        f"selected_queries: max_abs={query_difference.max().item():.6g} "
        f"mean_abs={query_difference.mean().item():.6g}"
    )
    print(
        f"selected_references: max_abs={finite_reference_difference.max().item():.6g} "
        f"mean_abs={finite_reference_difference.mean().item():.6g}"
    )

    final_errors = {}
    for name in ("pred_logits", "pred_boxes", "pred_keypoints"):
        native_value = native[name].float()
        reference_value = reference[name].float()
        difference = (native_value - reference_value).abs()
        final_errors[name] = difference.max().item()
        print(
            f"{name}: native={tuple(native_value.shape)} reference={tuple(reference_value.shape)} "
            f"max_abs={difference.max().item():.6g} mean_abs={difference.mean().item():.6g}"
        )
    if args.assert_tolerance is not None:
        exceeded = {
            name: error for name, error in final_errors.items() if error > args.assert_tolerance
        }
        if exceeded:
            raise AssertionError(
                f"final parity errors exceeded tolerance {args.assert_tolerance}: {exceeded}"
            )


if __name__ == "__main__":
    main()

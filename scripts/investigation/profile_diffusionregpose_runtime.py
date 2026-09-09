from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import subprocess
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import torch
from torch import nn


@dataclass(frozen=True)
class DiffusionRegPoseDimensions:
    hidden_dim: int = 256
    feedforward_dim: int = 2048
    num_heads: int = 8
    num_feature_levels: int = 4
    num_sampling_points: int = 4
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    num_box_decoder_layers: int = 2
    num_queries: int = 900
    num_groups: int = 100
    num_proposals: int = 100
    num_body_points: int = 14
    num_denoising_queries: int = 100


def feature_map_shapes(
    image_height: int,
    image_width: int,
    *,
    num_feature_levels: int,
) -> tuple[tuple[int, int], ...]:
    if image_height <= 0 or image_width <= 0:
        raise ValueError("image dimensions must be positive")
    if num_feature_levels <= 0:
        raise ValueError("num_feature_levels must be positive")

    shapes = [
        (math.ceil(image_height / stride), math.ceil(image_width / stride))
        for stride in (8, 16, 32)
    ][:num_feature_levels]
    while len(shapes) < num_feature_levels:
        previous_height, previous_width = shapes[-1]
        shapes.append((math.ceil(previous_height / 2), math.ceil(previous_width / 2)))
    return tuple(shapes)


def _convolution_macs(
    output_height: int,
    output_width: int,
    input_channels: int,
    output_channels: int,
    kernel_size: int,
) -> int:
    return (
        output_height
        * output_width
        * input_channels
        * output_channels
        * kernel_size
        * kernel_size
    )


def resnet50_backbone_macs(image_height: int, image_width: int) -> int:
    height = math.ceil(image_height / 2)
    width = math.ceil(image_width / 2)
    total = _convolution_macs(height, width, 3, 64, 7)

    height = math.ceil(height / 2)
    width = math.ceil(width / 2)
    input_channels = 64
    for block_count, channels, stage_stride in zip(
        (3, 4, 6, 3),
        (64, 128, 256, 512),
        (1, 2, 2, 2),
    ):
        for block_index in range(block_count):
            stride = stage_stride if block_index == 0 else 1
            output_height = math.ceil(height / stride)
            output_width = math.ceil(width / stride)
            total += _convolution_macs(height, width, input_channels, channels, 1)
            total += _convolution_macs(
                output_height,
                output_width,
                channels,
                channels,
                3,
            )
            total += _convolution_macs(
                output_height,
                output_width,
                channels,
                channels * 4,
                1,
            )
            if block_index == 0:
                total += _convolution_macs(
                    output_height,
                    output_width,
                    input_channels,
                    channels * 4,
                    1,
                )
            height = output_height
            width = output_width
            input_channels = channels * 4
    return total


def input_projection_macs(
    feature_shapes: tuple[tuple[int, int], ...],
    hidden_dim: int,
) -> int:
    backbone_channels = (512, 1024, 2048)
    total = 0
    for level_index, (height, width) in enumerate(feature_shapes):
        if level_index < len(backbone_channels):
            input_channels = backbone_channels[level_index]
            kernel_size = 1
        else:
            input_channels = 2048 if level_index == 3 else hidden_dim
            kernel_size = 3
        total += _convolution_macs(
            height,
            width,
            input_channels,
            hidden_dim,
            kernel_size,
        )
    return total


def decoder_query_counts(
    dimensions: DiffusionRegPoseDimensions,
    *,
    training: bool,
) -> tuple[int, ...]:
    if dimensions.num_box_decoder_layers > dimensions.num_decoder_layers:
        raise ValueError("num_box_decoder_layers cannot exceed num_decoder_layers")
    denoising_queries = dimensions.num_denoising_queries if training else 0
    box_query_count = dimensions.num_queries + denoising_queries
    expanded_query_count = (
        dimensions.num_groups * (dimensions.num_body_points + 1)
        + denoising_queries
    )
    return (
        (box_query_count,) * dimensions.num_box_decoder_layers
        + (expanded_query_count,)
        * (dimensions.num_decoder_layers - dimensions.num_box_decoder_layers)
    )


def analytical_mac_breakdown(
    image_height: int,
    image_width: int,
    dimensions: DiffusionRegPoseDimensions,
    *,
    training: bool,
) -> dict[str, int]:
    shapes = feature_map_shapes(
        image_height,
        image_width,
        num_feature_levels=dimensions.num_feature_levels,
    )
    feature_tokens = sum(height * width for height, width in shapes)
    hidden_dim = dimensions.hidden_dim
    attention_width = (
        dimensions.num_heads
        * dimensions.num_feature_levels
        * dimensions.num_sampling_points
    )

    encoder_layer_macs = feature_tokens * (
        hidden_dim * hidden_dim
        + hidden_dim * attention_width * 2
        + hidden_dim * attention_width
        + hidden_dim * hidden_dim
        + 2 * hidden_dim * dimensions.feedforward_dim
    )
    encoder_macs = encoder_layer_macs * dimensions.num_encoder_layers
    proposal_macs = feature_tokens * (
        hidden_dim * hidden_dim
        + hidden_dim * 2
        + hidden_dim * hidden_dim
        + hidden_dim * hidden_dim
        + hidden_dim * 4
    )

    decoder_macs = 0
    for query_count in decoder_query_counts(dimensions, training=training):
        decoder_macs += (
            feature_tokens * hidden_dim * hidden_dim
            + query_count
            * (
                hidden_dim * attention_width * 2
                + hidden_dim * attention_width
                + hidden_dim * hidden_dim
            )
            + query_count * 4 * hidden_dim * hidden_dim
            + 2 * query_count * query_count * hidden_dim
            + query_count * 2 * hidden_dim * dimensions.feedforward_dim
        )

    components = {
        "backbone": resnet50_backbone_macs(image_height, image_width),
        "input_projections": input_projection_macs(shapes, hidden_dim),
        "encoder": encoder_macs,
        "proposal_path": proposal_macs,
        "decoder": decoder_macs,
    }
    return {
        "image_height": image_height,
        "image_width": image_width,
        "feature_tokens": feature_tokens,
        **components,
        "total": sum(components.values()),
    }


def _linear_percentile(sorted_values: list[float], fraction: float) -> float:
    position = (len(sorted_values) - 1) * fraction
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    upper_weight = position - lower_index
    return (
        sorted_values[lower_index] * (1.0 - upper_weight)
        + sorted_values[upper_index] * upper_weight
    )


def summarize_measurements(measurements_ms: Iterable[float]) -> dict[str, float | int]:
    values = sorted(float(value) for value in measurements_ms)
    if not values:
        raise ValueError("at least one measurement is required")
    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "p10_ms": _linear_percentile(values, 0.1),
        "p90_ms": _linear_percentile(values, 0.9),
        "min_ms": values[0],
        "max_ms": values[-1],
    }


def summarize_memory(measurements_gib: Iterable[float]) -> dict[str, float | int]:
    values = sorted(float(value) for value in measurements_gib)
    if not values:
        raise ValueError("at least one memory measurement is required")
    return {
        "count": len(values),
        "mean_gib": statistics.fmean(values),
        "median_gib": statistics.median(values),
        "p10_gib": _linear_percentile(values, 0.1),
        "p90_gib": _linear_percentile(values, 0.9),
        "min_gib": values[0],
        "max_gib": values[-1],
    }


def parse_nvidia_smi_memory(output: str) -> tuple[float, float]:
    fields = [field.strip() for field in output.strip().split(",")]
    if len(fields) != 2:
        raise ValueError(f"unexpected nvidia-smi memory output: {output!r}")
    free_mib, total_mib = (float(field) for field in fields)
    return free_mib / 1024.0, total_mib / 1024.0


def query_gpu_memory(gpu_index: int) -> tuple[float, float]:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu_index}",
            "--query-gpu=memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_nvidia_smi_memory(result.stdout)


def build_reduction_matrix() -> dict[str, dict[str, Any]]:
    paper = DiffusionRegPoseDimensions()
    query_reduced = DiffusionRegPoseDimensions(
        num_queries=300,
        num_groups=30,
        num_proposals=30,
        num_denoising_queries=30,
    )
    compact = DiffusionRegPoseDimensions(
        hidden_dim=192,
        feedforward_dim=768,
        num_heads=6,
        num_feature_levels=3,
        num_encoder_layers=3,
        num_decoder_layers=4,
        num_queries=300,
        num_groups=30,
        num_proposals=30,
        num_denoising_queries=30,
    )
    development = DiffusionRegPoseDimensions(
        feedforward_dim=1024,
        num_feature_levels=3,
        num_encoder_layers=2,
        num_decoder_layers=3,
        num_box_decoder_layers=2,
        num_queries=200,
        num_groups=20,
        num_proposals=20,
        num_denoising_queries=20,
    )
    tiny = DiffusionRegPoseDimensions(
        hidden_dim=128,
        feedforward_dim=512,
        num_heads=4,
        num_feature_levels=3,
        num_encoder_layers=2,
        num_decoder_layers=3,
        num_box_decoder_layers=1,
        num_queries=200,
        num_groups=20,
        num_proposals=20,
        num_denoising_queries=20,
    )
    candidates = {
        "paper_800x1333": ((800, 1333), paper, True),
        "paper_800x1216": ((800, 1216), paper, True),
        "paper_512x512": ((512, 512), paper, True),
        "paper_192x192": ((192, 192), paper, True),
        "query_reduced_192x192": ((192, 192), query_reduced, False),
        "compact_r50_192x192": ((192, 192), compact, False),
        "development_r50_192x192": ((192, 192), development, False),
        "tiny_r50_192x192": ((192, 192), tiny, False),
    }
    matrix: dict[str, dict[str, Any]] = {}
    for name, (image_shape, dimensions, checkpoint_compatible) in candidates.items():
        height, width = image_shape
        inference = analytical_mac_breakdown(
            height,
            width,
            dimensions,
            training=False,
        )
        training = analytical_mac_breakdown(
            height,
            width,
            dimensions,
            training=True,
        )
        matrix[name] = {
            "image_shape": [height, width],
            "dimensions": asdict(dimensions),
            "checkpoint_compatible": checkpoint_compatible,
            "inference_macs": inference["total"],
            "training_macs": training["total"],
            "training_components": {
                component: training[component]
                for component in (
                    "backbone",
                    "input_projections",
                    "encoder",
                    "proposal_path",
                    "decoder",
                )
            },
        }
    return matrix


def build_model_for_dimensions(dimensions: DiffusionRegPoseDimensions) -> nn.Module:
    from src.models.pipelines.pose_estimators.diffusionregpose.model import (
        DiffusionRegPoseModel,
    )

    return DiffusionRegPoseModel(
        backbone_pretrained=False,
        num_queries=dimensions.num_queries,
        num_proposals=dimensions.num_proposals,
        num_feature_levels=dimensions.num_feature_levels,
        num_encoder_layers=dimensions.num_encoder_layers,
        num_decoder_layers=dimensions.num_decoder_layers,
        hidden_dim=dimensions.hidden_dim,
        feedforward_dim=dimensions.feedforward_dim,
        num_heads=dimensions.num_heads,
        num_body_points=dimensions.num_body_points,
        num_box_decoder_layers=dimensions.num_box_decoder_layers,
        num_group=dimensions.num_groups,
        dn_number=dimensions.num_denoising_queries,
    )


def build_parameter_variants() -> dict[str, dict[str, int]]:
    variants = {
        "paper": DiffusionRegPoseDimensions(),
        "development": DiffusionRegPoseDimensions(
            feedforward_dim=1024,
            num_feature_levels=3,
            num_encoder_layers=2,
            num_decoder_layers=3,
            num_box_decoder_layers=2,
            num_queries=200,
            num_groups=20,
            num_proposals=20,
            num_denoising_queries=20,
        ),
        "compact_192": DiffusionRegPoseDimensions(
            hidden_dim=192,
            feedforward_dim=768,
            num_heads=6,
            num_feature_levels=3,
            num_encoder_layers=3,
            num_decoder_layers=4,
            num_queries=300,
            num_groups=30,
            num_proposals=30,
            num_denoising_queries=30,
        ),
        "tiny_128": DiffusionRegPoseDimensions(
            hidden_dim=128,
            feedforward_dim=512,
            num_heads=4,
            num_feature_levels=3,
            num_encoder_layers=2,
            num_decoder_layers=3,
            num_box_decoder_layers=1,
            num_queries=200,
            num_groups=20,
            num_proposals=20,
            num_denoising_queries=20,
        ),
    }
    profiles = {}
    for name, dimensions in variants.items():
        model = build_model_for_dimensions(dimensions)
        profiles[name] = parameter_summary(model)
        del model
        gc.collect()
    return profiles


def _unique_parameters(modules: Iterable[nn.Module]) -> dict[int, nn.Parameter]:
    return {
        id(parameter): parameter
        for module in modules
        for parameter in module.parameters()
    }


def parameter_summary(module: nn.Module) -> dict[str, int]:
    parameters = _unique_parameters((module,)).values()
    parameter_list = list(parameters)
    total = sum(parameter.numel() for parameter in parameter_list)
    trainable = sum(
        parameter.numel() for parameter in parameter_list if parameter.requires_grad
    )
    return {
        "parameter_tensors": len(parameter_list),
        "parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": total - trainable,
        "parameter_bytes": sum(
            parameter.numel() * parameter.element_size() for parameter in parameter_list
        ),
    }


def parameter_partition(model: nn.Module) -> dict[str, dict[str, int]]:
    owner_modules = (
        ("label_embedding", (model.label_enc,)),
        ("backbone", (model.backbone,)),
        ("input_projections", (model.input_proj,)),
        ("box_heads", (model.bbox_embed,)),
        ("class_heads", (model.class_embed,)),
        ("pose_heads", (model.pose_embed,)),
        ("pose_size_head", (model.pose_hw_embed,)),
        ("transformer_exclusive", (model.transformer,)),
    )
    claimed_parameter_ids: set[int] = set()
    partition: dict[str, dict[str, int]] = {}
    for owner_name, modules in owner_modules:
        owned_parameters = {
            parameter_id: parameter
            for parameter_id, parameter in _unique_parameters(modules).items()
            if parameter_id not in claimed_parameter_ids
        }
        claimed_parameter_ids.update(owned_parameters)
        parameter_count = sum(
            parameter.numel() for parameter in owned_parameters.values()
        )
        trainable_count = sum(
            parameter.numel()
            for parameter in owned_parameters.values()
            if parameter.requires_grad
        )
        partition[owner_name] = {
            "parameter_tensors": len(owned_parameters),
            "parameters": parameter_count,
            "trainable_parameters": trainable_count,
            "frozen_parameters": parameter_count - trainable_count,
        }

    all_parameter_ids = set(_unique_parameters((model,)))
    if claimed_parameter_ids != all_parameter_ids:
        missing_count = len(all_parameter_ids - claimed_parameter_ids)
        raise RuntimeError(f"parameter partition omitted {missing_count} tensors")
    return partition


def module_inventory(model: nn.Module) -> dict[str, int]:
    return dict(sorted(Counter(type(module).__name__ for module in model.modules()).items()))


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def _time_cuda_call(function, device: torch.device):
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    result = function()
    torch.cuda.synchronize(device)
    return result, (time.perf_counter() - start) * 1000.0


class CudaModuleTimer:
    def __init__(self, modules: dict[str, nn.Module]) -> None:
        self._measurements: dict[str, list[float]] = {
            name: [] for name in modules
        }
        self._start_events: dict[str, list[torch.cuda.Event]] = {
            name: [] for name in modules
        }
        self._end_events: dict[str, list[torch.cuda.Event]] = {
            name: [] for name in modules
        }
        self._handles = []
        for name, module in modules.items():
            self._handles.append(
                module.register_forward_pre_hook(self._pre_hook(name))
            )
            self._handles.append(module.register_forward_hook(self._post_hook(name)))

    def _pre_hook(self, name: str):
        def record_start(_module, _inputs) -> None:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self._start_events[name].append(event)

        return record_start

    def _post_hook(self, name: str):
        def record_end(_module, _inputs, _output) -> None:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self._end_events[name].append(event)

        return record_end

    def finish_iteration(self) -> None:
        for name in self._measurements:
            starts = self._start_events[name]
            ends = self._end_events[name]
            if len(starts) != len(ends):
                raise RuntimeError(f"unbalanced CUDA timing hooks for {name}")
            self._measurements[name].append(
                sum(start.elapsed_time(end) for start, end in zip(starts, ends))
            )
            starts.clear()
            ends.clear()

    def clear(self) -> None:
        for values in self._measurements.values():
            values.clear()
        for events in (*self._start_events.values(), *self._end_events.values()):
            events.clear()

    def summaries(self) -> dict[str, dict[str, float | int]]:
        return {
            name: summarize_measurements(values)
            for name, values in self._measurements.items()
            if values
        }

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()

    def __enter__(self) -> "CudaModuleTimer":
        return self

    def __exit__(self, *_args) -> None:
        self.close()


def _forward_modules(model: nn.Module) -> dict[str, nn.Module]:
    modules = {
        "backbone_and_position": model.backbone,
        "encoder": model.transformer.encoder,
        "decoder": model.transformer.decoder,
    }
    modules.update({
        f"input_projection_{index}": projection
        for index, projection in enumerate(model.input_proj)
    })
    return modules


def _optimizer_from_pipeline(pipeline: nn.Module):
    configured = pipeline.configure_optimizers()
    if isinstance(configured, dict):
        return configured["optimizer"]
    return configured


def _load_profile_components(
    config_name: str,
    batch_size: int | None,
    overrides: tuple[str, ...],
    device: torch.device,
):
    from hydra import compose, initialize_config_dir

    from src.data import DataConfig
    from src.models.pipelines.pose_estimators.diffusionregpose import (
        DiffusionRegPosePipelineConfig,
    )

    config_directory = Path(__file__).resolve().parents[2] / "conf"
    hydra_overrides = list(overrides)
    if batch_size is not None:
        hydra_overrides.append(f"data.batch_size={batch_size}")
    with initialize_config_dir(version_base="1.3", config_dir=str(config_directory)):
        config = compose(config_name=config_name, overrides=hydra_overrides)

    data_config = DataConfig.from_hydra_config(config.data)
    data_module = data_config.build(pin_memory=device.type == "cuda")
    data_module.setup()
    data_module._trainer = SimpleNamespace(training=True)

    pipeline_config = DiffusionRegPosePipelineConfig.from_hydra_config(config.pipeline)
    pipeline_config.pretrained_checkpoint = None
    pipeline_config.backbone_pretrained = False
    pipeline = pipeline_config.build(max_epochs=1).to(device).train()
    return data_module, pipeline


def _profile_data_pipeline(
    data_module,
    *,
    iterations: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    loader = data_module.train_dataloader()
    iterator = iter(loader)
    loading_times = []
    transfer_times = []
    augmentation_times = []
    transformed_batch = None
    for _ in range(iterations):
        start = time.perf_counter()
        try:
            raw_batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            raw_batch = next(iterator)
        loading_times.append((time.perf_counter() - start) * 1000.0)

        device_batch, transfer_time = _time_cuda_call(
            lambda: _move_to_device(raw_batch, device),
            device,
        )
        transformed_batch, augmentation_time = _time_cuda_call(
            lambda: data_module.on_after_batch_transfer(device_batch, 0),
            device,
        )
        transfer_times.append(transfer_time)
        augmentation_times.append(augmentation_time)

    assert transformed_batch is not None
    profile = {
        "loading_and_collate": summarize_measurements(loading_times),
        "host_to_device": summarize_measurements(transfer_times),
        "gpu_augmentation": summarize_measurements(augmentation_times),
    }
    return transformed_batch, profile


def _run_training_iteration(pipeline, optimizer, batch, device, module_timer):
    from src.models.pipelines.pose_estimators.diffusionregpose.training import (
        build_diffusionregpose_targets,
    )

    optimizer.zero_grad(set_to_none=True)
    target_result, target_time = _time_cuda_call(
        lambda: build_diffusionregpose_targets(
            batch,
            pipeline.model,
            num_proposals=pipeline.model.num_proposals,
            num_body_points=pipeline.model.transformer.decoder.num_body_points,
        ),
        device,
    )
    diffusion_keypoints, timesteps, targets = target_result
    outputs, forward_time = _time_cuda_call(
        lambda: pipeline.model(
            batch["image"],
            diffusion_keypoints,
            timesteps,
            targets,
            batch.get("padding_mask"),
        ),
        device,
    )
    module_timer.finish_iteration()
    losses, criterion_time = _time_cuda_call(
        lambda: pipeline.criterion(outputs, targets),
        device,
    )
    _, backward_time = _time_cuda_call(losses["total"].backward, device)
    _, optimizer_time = _time_cuda_call(optimizer.step, device)
    loss_value = float(losses["total"].detach().cpu())
    return {
        "target_preparation": target_time,
        "forward": forward_time,
        "criterion_and_matching": criterion_time,
        "backward": backward_time,
        "optimizer": optimizer_time,
        "loss": loss_value,
    }


def _profile_validation_sampling(
    pipeline,
    batch,
    *,
    warmup_iterations: int,
    iterations: int,
    device: torch.device,
) -> dict[str, Any]:
    from src.models.pipelines.pose_estimators.diffusionregpose.evaluation import (
        postprocess_crowdpose,
    )

    pipeline.model.eval()
    sampling_times = []
    postprocess_times = []
    with torch.no_grad():
        for iteration in range(warmup_iterations + iterations):
            outputs, sampling_time = _time_cuda_call(
                lambda: pipeline.model.sample_keypoints(
                    batch["image"], padding_mask=batch.get("padding_mask")
                ),
                device,
            )
            _, postprocess_time = _time_cuda_call(
                lambda: postprocess_crowdpose(
                    outputs,
                    batch.get("original_img_hw", batch["img_hw"]),
                    num_body_points=pipeline.model.transformer.decoder.num_body_points,
                    nms_target_sizes=batch["img_hw"],
                ),
                device,
            )
            if iteration >= warmup_iterations:
                sampling_times.append(sampling_time)
                postprocess_times.append(postprocess_time)
    pipeline.model.train()
    return {
        "sampling": summarize_measurements(sampling_times),
        "postprocess_and_pose_nms": summarize_measurements(postprocess_times),
    }


def profile_cuda_runtime(
    *,
    config_name: str,
    batch_size: int | None,
    warmup_iterations: int,
    iterations: int,
    minimum_free_gib: float,
    allow_busy_gpu: bool,
    gpu_index: int,
    overrides: tuple[str, ...] = (),
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA runtime profiling requires a CUDA device")
    if warmup_iterations < 1 or iterations < 1:
        raise ValueError("warmup_iterations and iterations must be positive")

    free_gib, total_gib = query_gpu_memory(gpu_index)
    if free_gib < minimum_free_gib and not allow_busy_gpu:
        raise RuntimeError(
            f"only {free_gib:.2f} GiB CUDA memory is free; require "
            f"{minimum_free_gib:.2f} GiB or pass --allow-busy-gpu"
        )
    device = torch.device("cuda", gpu_index)

    data_module, pipeline = _load_profile_components(
        config_name,
        batch_size,
        overrides,
        device,
    )
    batch, data_profile = _profile_data_pipeline(
        data_module,
        iterations=iterations,
        device=device,
    )
    optimizer = _optimizer_from_pipeline(pipeline)
    stage_measurements: dict[str, list[float]] = {
        "target_preparation": [],
        "forward": [],
        "criterion_and_matching": [],
        "backward": [],
        "optimizer": [],
    }
    losses = []
    peak_allocated = []
    peak_reserved = []
    with CudaModuleTimer(_forward_modules(pipeline.model)) as module_timer:
        for _ in range(warmup_iterations):
            _run_training_iteration(
                pipeline,
                optimizer,
                batch,
                device,
                module_timer,
            )
        module_timer.clear()
        for _ in range(iterations):
            torch.cuda.reset_peak_memory_stats(device)
            result = _run_training_iteration(
                pipeline,
                optimizer,
                batch,
                device,
                module_timer,
            )
            for stage_name in stage_measurements:
                stage_measurements[stage_name].append(result[stage_name])
            losses.append(result["loss"])
            peak_allocated.append(torch.cuda.max_memory_allocated(device) / 1024**3)
            peak_reserved.append(torch.cuda.max_memory_reserved(device) / 1024**3)

        module_profile = module_timer.summaries()

    validation_profile = _profile_validation_sampling(
        pipeline,
        batch,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
        device=device,
    )
    stage_profile = {
        name: summarize_measurements(values)
        for name, values in stage_measurements.items()
    }
    stage_profile["full_update"] = summarize_measurements([
        sum(stage_measurements[name][index] for name in stage_measurements)
        for index in range(iterations)
    ])
    return {
        "device": {
            "name": torch.cuda.get_device_name(device),
            "total_memory_gib": total_gib,
            "free_memory_before_profile_gib": free_gib,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
        "config_name": config_name,
        "overrides": list(overrides),
        "batch_size": int(batch["image"].shape[0]),
        "image_shape": list(batch["image"].shape),
        "warmup_iterations": warmup_iterations,
        "measured_iterations": iterations,
        "data": data_profile,
        "training_stages": stage_profile,
        "forward_modules_cuda_events": module_profile,
        "validation": validation_profile,
        "peak_memory": {
            "allocated": summarize_memory(peak_allocated),
            "reserved": summarize_memory(peak_reserved),
        },
        "loss_range": [min(losses), max(losses)],
    }


def build_static_profile() -> dict[str, Any]:
    from src.models.pipelines.pose_estimators.diffusionregpose.model import (
        DiffusionRegPoseModel,
    )

    dimensions = DiffusionRegPoseDimensions()
    model = DiffusionRegPoseModel(backbone_pretrained=False)
    representative_shapes = ((128, 192), (512, 512), (800, 800), (800, 1333))
    return {
        "schema_version": 1,
        "dimensions": asdict(dimensions),
        "parameters": parameter_summary(model),
        "parameter_partition": parameter_partition(model),
        "module_inventory": module_inventory(model),
        "analytical_macs": {
            f"{height}x{width}": {
                "inference": analytical_mac_breakdown(
                    height,
                    width,
                    dimensions,
                    training=False,
                ),
                "training_layout": analytical_mac_breakdown(
                    height,
                    width,
                    dimensions,
                    training=True,
                ),
            }
            for height, width in representative_shapes
        },
        "reduction_matrix": build_reduction_matrix(),
        "parameter_variants": build_parameter_variants(),
        "counting_scope": {
            "mac_to_flop_ratio": 2,
            "included": "convolution, linear, FFN, and dense attention matrix products",
            "excluded": (
                "normalization, activation, softmax, deformable bilinear sampling, "
                "prediction/refinement heads, matching, losses, backward, and optimizer"
            ),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile DiffusionRegPose architecture and analytical operation counts."
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path. Prints to stdout when omitted.",
    )
    parser.add_argument(
        "--runtime-config",
        help="Hydra config to profile on a real CrowdPose training batch.",
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--warmup-iterations", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--minimum-free-gib", type=float, default=18.0)
    parser.add_argument("--allow-busy-gpu", action="store_true")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Repeatable Hydra override for controlled architecture comparisons.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    profile = build_static_profile()
    if args.runtime_config:
        profile["runtime"] = profile_cuda_runtime(
            config_name=args.runtime_config,
            batch_size=args.batch_size,
            warmup_iterations=args.warmup_iterations,
            iterations=args.iterations,
            minimum_free_gib=args.minimum_free_gib,
            allow_busy_gpu=args.allow_busy_gpu,
            gpu_index=args.gpu_index,
            overrides=tuple(args.override),
        )
    serialized = json.dumps(profile, indent=2)
    if args.output is None:
        print(serialized)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(f"{serialized}\n")


if __name__ == "__main__":
    main()

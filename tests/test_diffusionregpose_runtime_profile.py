from __future__ import annotations

import torch
import pytest
from torch import nn

from scripts.investigation.profile_diffusionregpose_runtime import (
    DiffusionRegPoseDimensions,
    analytical_mac_breakdown,
    build_reduction_matrix,
    feature_map_shapes,
    parameter_partition,
    parameter_summary,
    parse_nvidia_smi_memory,
    summarize_memory,
    summarize_measurements,
)
from scripts.investigation.generate_diffusionregpose_runtime_report import (
    build_runtime_rows,
)
from src.models.pipelines.pose_estimators.diffusionregpose.model import (
    DiffusionRegPoseModel,
)
from src.models.pipelines.pose_estimators.diffusionregpose.transformer import (
    sine_embedding_for_position,
)


def test_feature_map_shapes_include_generated_fourth_level() -> None:
    assert feature_map_shapes(512, 512, num_feature_levels=4) == (
        (64, 64),
        (32, 32),
        (16, 16),
        (8, 8),
    )
    assert feature_map_shapes(800, 1333, num_feature_levels=4) == (
        (100, 167),
        (50, 84),
        (25, 42),
        (13, 21),
    )


def test_default_analytical_mac_floor_matches_audited_values() -> None:
    dimensions = DiffusionRegPoseDimensions()

    inference = analytical_mac_breakdown(800, 1333, dimensions, training=False)
    training = analytical_mac_breakdown(800, 1333, dimensions, training=True)

    assert inference["feature_tokens"] == 22_223
    assert inference["backbone"] == 87_333_299_200
    assert inference["input_projections"] == 5_128_585_216
    assert inference["encoder"] == 170_399_563_776
    assert inference["proposal_path"] == 4_403_354_112
    assert inference["decoder"] == 25_677_447_168
    assert inference["total"] == 292_942_249_472
    assert training["decoder"] == 27_391_623_168
    assert training["total"] == 294_656_425_472


def test_query_reduction_changes_decoder_cost_without_changing_dense_costs() -> None:
    full = analytical_mac_breakdown(
        800,
        800,
        DiffusionRegPoseDimensions(),
        training=False,
    )
    reduced = analytical_mac_breakdown(
        800,
        800,
        DiffusionRegPoseDimensions(num_queries=600, num_groups=50, num_proposals=50),
        training=False,
    )

    assert reduced["backbone"] == full["backbone"]
    assert reduced["encoder"] == full["encoder"]
    assert reduced["decoder"] == 12_941_205_504


def test_parameter_summary_deduplicates_shared_modules() -> None:
    shared = nn.Linear(4, 3)
    model = nn.ModuleDict({"first": shared, "second": shared})

    summary = parameter_summary(model)

    assert summary == {
        "parameter_tensors": 2,
        "parameters": 15,
        "trainable_parameters": 15,
        "frozen_parameters": 0,
        "parameter_bytes": 60,
    }


def test_default_parameter_partition_is_complete_and_unique() -> None:
    model = DiffusionRegPoseModel(backbone_pretrained=False)

    total = parameter_summary(model)
    partition = parameter_partition(model)

    assert total["parameters"] == 49_527_324
    assert total["trainable_parameters"] == 49_304_924
    assert sum(group["parameters"] for group in partition.values()) == total["parameters"]
    assert partition["backbone"]["parameters"] == 23_454_912
    assert partition["input_projections"]["parameters"] == 5_639_168
    assert partition["transformer_exclusive"]["parameters"] == 18_948_142


def test_reduction_matrix_separates_resolution_and_architecture_effects() -> None:
    matrix = build_reduction_matrix()

    paper = matrix["paper_800x1333"]
    low_resolution = matrix["paper_192x192"]
    compact = matrix["compact_r50_192x192"]

    assert low_resolution["dimensions"] == paper["dimensions"]
    assert low_resolution["training_macs"] < paper["training_macs"] / 10
    assert compact["dimensions"]["num_queries"] == 300
    assert compact["dimensions"]["num_groups"] == 30
    assert compact["training_macs"] < low_resolution["training_macs"] / 4
    assert compact["checkpoint_compatible"] is False


def test_measurement_summary_reports_distribution_and_sample_count() -> None:
    summary = summarize_measurements([1.0, 4.0, 2.0, 5.0, 3.0])

    assert summary == {
        "count": 5,
        "mean_ms": 3.0,
        "median_ms": 3.0,
        "p10_ms": 1.4,
        "p90_ms": 4.6,
        "min_ms": 1.0,
        "max_ms": 5.0,
    }

    memory = summarize_memory([2.0, 4.0])
    assert memory["median_gib"] == 3.0
    assert "median_ms" not in memory


def test_nvidia_smi_memory_parser_does_not_require_a_cuda_context() -> None:
    assert parse_nvidia_smi_memory("512, 24576\n") == (0.5, 24.0)


def test_compact_hidden_and_feedforward_dimensions_are_wired_consistently() -> None:
    model = DiffusionRegPoseModel(
        backbone_pretrained=False,
        hidden_dim=192,
        feedforward_dim=768,
        num_heads=6,
        num_feature_levels=3,
        num_encoder_layers=2,
        num_decoder_layers=3,
        num_box_decoder_layers=1,
        num_queries=100,
        num_proposals=10,
        num_group=10,
        dn_number=10,
    )

    decoder = model.transformer.decoder
    position = sine_embedding_for_position(
        torch.zeros(1, 1, 4),
        num_pos_feats=96,
    )

    assert position.shape == (1, 1, 384)
    assert decoder.ref_point_head.layers[0].in_features == 384
    assert model.transformer.encoder.layers[0].linear1.out_features == 768
    assert decoder.layers[0].linear1.out_features == 768


def test_runtime_report_extracts_measured_development_throughput() -> None:
    rows = {row["variant"]: row for row in build_runtime_rows()}

    development = rows["Development, batch 16"]
    assert development["batch_size"] == 16
    assert development["images_per_second"] == pytest.approx(38.58, rel=1e-3)
    assert development["peak_allocated_gib"] == pytest.approx(1.558, rel=1e-3)

from __future__ import annotations

import csv
import json
from dataclasses import fields
from pathlib import Path

import matplotlib.pyplot as plt

from scripts.investigation.profile_diffusionregpose_runtime import (
    DiffusionRegPoseDimensions,
    build_model_for_dimensions,
    parameter_partition,
)


REPORT_DIRECTORY = Path("reports/diffusionregpose_runtime")
RUNTIME_FILES = {
    "Paper, batch 1": "runtime_paper_b1.json",
    "Low-res full, batch 8": "runtime_full_lowres_b8.json",
    "Compact 256, batch 8": "runtime_compact256_r50_lowres_b8.json",
    "Compact 192, batch 8": "runtime_compact192_r50_lowres_b8.json",
    "Development, batch 16": "runtime_development_preset.json",
    "Tiny 128, batch 8": "runtime_tiny128_r50_lowres_b8.json",
}
STAGE_NAMES = (
    "target_preparation",
    "forward",
    "criterion_and_matching",
    "backward",
    "optimizer",
)
COMPONENT_NAMES = (
    "backbone",
    "input_projections",
    "encoder",
    "proposal_path",
    "decoder",
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _dimensions_from_dict(values: dict) -> DiffusionRegPoseDimensions:
    names = {field.name for field in fields(DiffusionRegPoseDimensions)}
    return DiffusionRegPoseDimensions(**{
        name: value for name, value in values.items() if name in names
    })


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_runtime_rows() -> list[dict]:
    rows = []
    for name, filename in RUNTIME_FILES.items():
        runtime = _read_json(REPORT_DIRECTORY / filename)["runtime"]
        update_ms = runtime["training_stages"]["full_update"]["median_ms"]
        batch_size = runtime["batch_size"]
        row = {
            "variant": name,
            "batch_size": batch_size,
            "image_shape": "x".join(str(value) for value in runtime["image_shape"][-2:]),
            "update_median_ms": update_ms,
            "images_per_second": batch_size * 1000.0 / update_ms,
            "peak_allocated_gib": runtime["peak_memory"]["allocated"]["median_gib"],
            "peak_reserved_gib": runtime["peak_memory"]["reserved"]["median_gib"],
        }
        for stage in STAGE_NAMES:
            row[f"{stage}_median_ms"] = runtime["training_stages"][stage]["median_ms"]
            row[f"{stage}_per_image_ms"] = row[f"{stage}_median_ms"] / batch_size
        rows.append(row)
    return rows


def build_parameter_rows(static_profile: dict) -> list[dict]:
    development_dimensions = _dimensions_from_dict(
        static_profile["reduction_matrix"]["development_r50_192x192"]["dimensions"]
    )
    development_model = build_model_for_dimensions(development_dimensions)
    development_partition = parameter_partition(development_model)
    partitions = {
        "Paper": static_profile["parameter_partition"],
        "Development": development_partition,
    }
    groups = {
        "backbone": ("backbone",),
        "projections": ("input_projections",),
        "prediction_heads": (
            "box_heads",
            "class_heads",
            "pose_heads",
            "pose_size_head",
        ),
        "transformer_and_conditioning": ("transformer_exclusive",),
        "label_embedding": ("label_embedding",),
    }
    rows = []
    for variant, partition in partitions.items():
        row = {"variant": variant}
        for group_name, owners in groups.items():
            row[group_name] = sum(partition[owner]["parameters"] for owner in owners)
        row["total"] = sum(part["parameters"] for part in partition.values())
        rows.append(row)
    return rows


def build_mac_rows(static_profile: dict) -> list[dict]:
    selections = {
        "Paper 800x1333": "paper_800x1333",
        "Full 192x192": "paper_192x192",
        "Development 192x192": "development_r50_192x192",
        "Tiny 192x192": "tiny_r50_192x192",
    }
    rows = []
    for label, key in selections.items():
        candidate = static_profile["reduction_matrix"][key]
        row = {"variant": label}
        for component in COMPONENT_NAMES:
            row[component] = candidate["training_components"][component] / 1e9
        row["total"] = candidate["training_macs"] / 1e9
        rows.append(row)
    return rows


def _stacked_bar(rows: list[dict], columns: tuple[str, ...], output: Path, ylabel: str) -> None:
    labels = [row["variant"] for row in rows]
    bottoms = [0.0] * len(rows)
    figure, axis = plt.subplots(figsize=(10, 5.5))
    for column in columns:
        values = [row[column] for row in rows]
        axis.bar(labels, values, bottom=bottoms, label=column.replace("_", " "))
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
    axis.set_ylabel(ylabel)
    axis.legend(frameon=False, ncol=2)
    axis.tick_params(axis="x", rotation=15)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _runtime_plots(rows: list[dict]) -> None:
    per_image_rows = [row for row in rows if row["variant"] != "Paper, batch 1"]
    labels = [row["variant"] for row in per_image_rows]
    bottoms = [0.0] * len(per_image_rows)
    figure, axis = plt.subplots(figsize=(11, 5.5))
    for stage in STAGE_NAMES:
        values = [row[f"{stage}_per_image_ms"] for row in per_image_rows]
        axis.bar(labels, values, bottom=bottoms, label=stage.replace("_", " "))
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
    axis.set_ylabel("Median synchronized update time per image (ms)")
    axis.tick_params(axis="x", rotation=18)
    axis.legend(frameon=False, ncol=2)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(REPORT_DIRECTORY / "runtime_stage_per_image.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5.5))
    for row in rows:
        axis.scatter(
            row["peak_allocated_gib"],
            row["images_per_second"],
            s=85,
        )
        axis.annotate(
            row["variant"],
            (row["peak_allocated_gib"], row["images_per_second"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_xlabel("Peak allocated CUDA memory (GiB)")
    axis.set_ylabel("Measured training throughput (images/s)")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(REPORT_DIRECTORY / "throughput_memory.png", dpi=180)
    plt.close(figure)


def main() -> None:
    static_profile = _read_json(REPORT_DIRECTORY / "static_profile.json")
    runtime_rows = build_runtime_rows()
    parameter_rows = build_parameter_rows(static_profile)
    mac_rows = build_mac_rows(static_profile)

    _write_csv(REPORT_DIRECTORY / "runtime_comparison.csv", runtime_rows)
    _write_csv(REPORT_DIRECTORY / "parameter_comparison.csv", parameter_rows)
    _write_csv(REPORT_DIRECTORY / "mac_comparison.csv", mac_rows)

    _stacked_bar(
        parameter_rows,
        (
            "backbone",
            "projections",
            "prediction_heads",
            "transformer_and_conditioning",
            "label_embedding",
        ),
        REPORT_DIRECTORY / "parameters_by_subsystem.png",
        "Unique parameters",
    )
    _stacked_bar(
        mac_rows,
        COMPONENT_NAMES,
        REPORT_DIRECTORY / "analytical_macs_by_subsystem.png",
        "Analytical forward matrix floor (G MAC)",
    )
    _stacked_bar(
        mac_rows[1:],
        COMPONENT_NAMES,
        REPORT_DIRECTORY / "analytical_macs_lowres.png",
        "Analytical forward matrix floor (G MAC)",
    )
    _runtime_plots(runtime_rows)


if __name__ == "__main__":
    main()

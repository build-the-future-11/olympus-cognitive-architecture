#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Run the frozen Olympus instruction-model engineering backtest matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import matplotlib  # type: ignore[import-not-found, unused-ignore]

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore[import-not-found, unused-ignore]
import numpy as np

from olympus.foundry.eval_suite import evaluate_checkpoint
from olympus.foundry.sft import SFTConfig, TinyModelConfig, run_sft


MATRIX_VERSION = "olympus-instruction-backtests-v1"
PAIRED_SEEDS = [20261101, 20261102, 20261103, 20261104, 20261105]
SHORT_SEEDS = PAIRED_SEEDS[:3]


@dataclass(frozen=True)
class Variant:
    name: str
    width: int
    layers: int
    heads: int
    pack_sequences: bool
    warmup_optimizer_steps: int
    minimum_learning_rate_ratio: float
    restore_best_validation: bool
    seeds: list[int]


VARIANTS = [
    Variant("cosine_large", 128, 4, 8, True, 10, 0.05, True, PAIRED_SEEDS),
    Variant("constant_large", 128, 4, 8, True, 0, 1.0, True, PAIRED_SEEDS),
    Variant("no_restore_large", 128, 4, 8, True, 10, 0.05, False, PAIRED_SEEDS),
    Variant("no_pack_large", 128, 4, 8, False, 10, 0.05, True, SHORT_SEEDS),
    Variant("cosine_medium", 64, 2, 4, True, 10, 0.05, True, SHORT_SEEDS),
    Variant("cosine_small", 32, 1, 4, True, 10, 0.05, True, SHORT_SEEDS),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(path)


def config_for(variant: Variant, seed: int) -> SFTConfig:
    return SFTConfig(
        mode="full",
        seed=seed,
        epochs=30,
        batch_size=4,
        learning_rate=1e-3,
        warmup_optimizer_steps=variant.warmup_optimizer_steps,
        minimum_learning_rate_ratio=variant.minimum_learning_rate_ratio,
        gradient_accumulation_steps=2,
        gradient_clip_norm=1.0,
        weight_decay=0.01,
        mixed_precision=False,
        pack_sequences=variant.pack_sequences,
        early_stopping_patience=0,
        early_stopping_min_delta=0.0,
        restore_best_validation=variant.restore_best_validation,
        device="cpu",
        model=TinyModelConfig(
            width=variant.width,
            layers=variant.layers,
            heads=variant.heads,
            max_sequence_tokens=192,
            dropout=0.0,
        ),
    )


def bootstrap_mean_ci(values: list[float], seed: int = 20261011) -> list[float]:
    if not values:
        return [math.nan, math.nan]
    generator = np.random.default_rng(seed)
    array = np.asarray(values, dtype=np.float64)
    draws = generator.choice(array, size=(20_000, len(array)), replace=True).mean(axis=1)
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return [float(lower), float(upper)]


def summarize(values: list[float]) -> dict[str, float | list[float]]:
    return {
        "n": float(len(values)),
        "mean": statistics.fmean(values),
        "sample_standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
        "median": statistics.median(values),
        "bootstrap_mean_95_ci": bootstrap_mean_ci(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def frozen_manifest(manifest_path: Path) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for seed in variant.seeds:
            config = config_for(variant, seed)
            runs.append(
                {
                    "variant": variant.name,
                    "seed": seed,
                    "config": config.model_dump(mode="json"),
                }
            )
    return {
        "schema_version": 1,
        "matrix_version": MATRIX_VERSION,
        "classification": "DEVELOPMENT_EXPLORATORY_TEST_ALREADY_OBSERVED",
        "frozen_before_execution": True,
        "dataset_manifest_path": str(manifest_path.resolve()),
        "dataset_manifest_file_sha256": sha256(manifest_path),
        "planned_runs": len(runs),
        "runs": runs,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_results(output_root: Path, rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    figures = output_root / "figures"
    figures.mkdir(exist_ok=True)
    completed = [row for row in rows if row["status"] == "COMPLETED"]
    names = [variant.name for variant in VARIANTS]
    colors = ["#204060", "#648fff", "#dc267f", "#ffb000", "#785ef0", "#009e73"]
    outputs: list[Path] = []

    if not completed:
        return []

    fig, axis = plt.subplots(figsize=(9.0, 4.8))
    values = [
        [float(row["test_loss"]) for row in completed if row["variant"] == name] for name in names
    ]
    axis.boxplot(values, tick_labels=names, showmeans=True)
    axis.set_ylabel("Observed smoke-test loss (lower is better)")
    axis.set_title("Exploratory test loss by frozen variant")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    outputs.append(figures / "test_loss_by_variant.png")
    fig.savefig(outputs[-1], dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    for name, color in zip(names, colors, strict=True):
        selected = [row for row in completed if row["variant"] == name]
        if selected:
            axis.scatter(
                [float(row["total_parameters"]) for row in selected],
                [float(row["test_loss"]) for row in selected],
                label=name,
                color=color,
                alpha=0.85,
            )
    axis.set_xscale("log")
    axis.set_xlabel("Parameters (log scale)")
    axis.set_ylabel("Observed smoke-test loss")
    axis.set_title("Parameter scale and exploratory loss")
    axis.grid(alpha=0.25)
    if axis.has_data():
        axis.legend(fontsize=8)
    fig.tight_layout()
    outputs.append(figures / "efficiency_tradeoff.png")
    fig.savefig(outputs[-1], dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9.0, 4.8))
    means = [
        statistics.fmean(
            float(row["format_compliance_rate"]) for row in completed if row["variant"] == name
        )
        for name in names
    ]
    axis.bar(names, means, color=colors)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Mean format-compliance rate")
    axis.set_title("Exploratory generation-format compliance")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    outputs.append(figures / "format_compliance.png")
    fig.savefig(outputs[-1], dpi=180)
    plt.close(fig)

    return [{"path": str(path), "sha256": sha256(path)} for path in outputs]


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row["status"] == "COMPLETED"]
    by_variant: dict[str, Any] = {}
    endpoints = [
        "validation_loss",
        "test_loss",
        "format_compliance_rate",
        "exact_match_rate",
        "elapsed_seconds",
        "peak_rss_bytes",
    ]
    for variant in VARIANTS:
        selected = [row for row in completed if row["variant"] == variant.name]
        by_variant[variant.name] = {
            "completed_runs": len(selected),
            "failed_runs": len(variant.seeds) - len(selected),
            "metrics": {
                endpoint: summarize([float(row[endpoint]) for row in selected])
                for endpoint in endpoints
                if selected
            },
        }

    paired: dict[str, Any] = {}
    baseline = {int(row["seed"]): row for row in completed if row["variant"] == "cosine_large"}
    for comparator in ("constant_large", "no_restore_large"):
        compare = {int(row["seed"]): row for row in completed if row["variant"] == comparator}
        common = sorted(set(baseline) & set(compare))
        differences = [
            float(baseline[seed]["test_loss"]) - float(compare[seed]["test_loss"])
            for seed in common
        ]
        paired[comparator] = {
            "definition": (
                "cosine_large test loss minus comparator test loss; negative favors cosine_large"
            ),
            "seeds": common,
            "differences": differences,
            "summary": summarize(differences) if differences else None,
        }
    return {
        "schema_version": 1,
        "matrix_version": MATRIX_VERSION,
        "classification": "DEVELOPMENT_EXPLORATORY_TEST_ALREADY_OBSERVED",
        "planned_runs": sum(len(variant.seeds) for variant in VARIANTS),
        "completed_runs": len(completed),
        "failed_runs": len(rows) - len(completed),
        "variants": by_variant,
        "paired_test_loss_differences": paired,
        "claim_boundary": (
            "These results audit tiny-model engineering on an already observed 36-record smoke "
            "corpus. They do not test Olympus fusion, JEPA quality, general capability, safety "
            "efficacy, or model-promotion readiness."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    output_root = args.output.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite backtest root: {output_root}")
    output_root.mkdir(parents=True)
    matrix_path = output_root / "matrix_manifest.json"
    atomic_json(matrix_path, frozen_manifest(manifest_path))
    runs_root = output_root / "runs"
    rows: list[dict[str, Any]] = []
    failures = output_root / "failures.jsonl"

    for variant in VARIANTS:
        for seed in variant.seeds:
            config = config_for(variant, seed)
            started = time.time()
            row: dict[str, Any] = {
                "variant": variant.name,
                "seed": seed,
                "status": "FAILED",
            }
            try:
                summary = run_sft(manifest_path, runs_root, config=config)
                run_directory = Path(summary.checkpoint_path).parent
                evaluation_path = run_directory / "exploratory_test_evaluation.json"
                evaluation = evaluate_checkpoint(
                    Path(summary.checkpoint_path),
                    manifest_path,
                    evaluation_path,
                    max_generation_tokens=48,
                )
                row.update(
                    {
                        "status": "COMPLETED",
                        "run_id": summary.run_id,
                        "training_config_sha256": summary.training_config_sha256,
                        "checkpoint_path": summary.checkpoint_path,
                        "checkpoint_sha256": summary.checkpoint_sha256,
                        "metrics_path": summary.log_path,
                        "metrics_sha256": sha256(Path(summary.log_path)),
                        "evaluation_path": str(evaluation_path),
                        "evaluation_sha256": sha256(evaluation_path),
                        "total_parameters": summary.total_parameters,
                        "trainable_parameters": summary.trainable_parameters,
                        "attempted_epochs": summary.attempted_epochs,
                        "restored_epoch": summary.completed_epochs,
                        "validation_loss": summary.final_validation_loss,
                        "test_loss": evaluation.overall_candidate_loss,
                        "format_compliance_rate": evaluation.format_compliance_rate,
                        "exact_match_rate": evaluation.exact_match_rate,
                        "regression_count": evaluation.regression_count,
                        "elapsed_seconds": summary.elapsed_seconds,
                        "peak_rss_bytes": evaluation.peak_rss_bytes,
                    }
                )
            except Exception as error:  # noqa: BLE001 - failures are retained as evidence
                failure = {
                    "variant": variant.name,
                    "seed": seed,
                    "started_unix": started,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
                with failures.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(failure, sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                row.update({"error_type": type(error).__name__, "error": str(error)})
            rows.append(row)
            atomic_json(output_root / "progress.json", rows)
            print(json.dumps(row, sort_keys=True), flush=True)

    write_csv(output_root / "runs.csv", rows)
    analysis = aggregate(rows)
    analysis["figures"] = plot_results(output_root, rows)
    analysis["matrix_manifest_sha256"] = sha256(matrix_path)
    atomic_json(output_root / "aggregate.json", analysis)
    if not failures.exists():
        failures.write_text("", encoding="utf-8")
    atomic_json(
        output_root / "completion_receipt.json",
        {
            "schema_version": 1,
            "status": "COMPLETED" if analysis["failed_runs"] == 0 else "COMPLETED_WITH_FAILURES",
            "aggregate_sha256": sha256(output_root / "aggregate.json"),
            "runs_csv_sha256": sha256(output_root / "runs.csv"),
            "failures_sha256": sha256(failures),
            "completed_runs": analysis["completed_runs"],
            "failed_runs": analysis["failed_runs"],
            "claim_boundary": analysis["claim_boundary"],
        },
    )
    return 0 if analysis["failed_runs"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Render tables and figures only from locked Olympus v2 confirmatory artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt  # type: ignore[import-not-found, unused-ignore]

VARIANTS = (
    "full",
    "uniform",
    "no_counterfactual",
    "no_disagreement_risk",
    "mechanism_off",
)
REGIMES = ("clean_test", "corrupted_test", "ood_test")
SEEDS = (20261301, 20261302, 20261303, 20261304, 20261305)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(run_root: Path) -> dict[str, dict[int, dict[str, dict[str, Any]]]]:
    return {
        variant: {
            seed: {
                regime: json.loads(
                    (
                        run_root
                        / "confirmatory"
                        / variant
                        / str(seed)
                        / f"{regime}.json"
                    ).read_text(encoding="utf-8")
                )
                for regime in REGIMES
            }
            for seed in SEEDS
        }
        for variant in VARIANTS
    }


def _mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)


def render(run_root: Path, output: Path) -> None:
    summary_path = run_root / "CONFIRMATORY_SUMMARY.json"
    freeze_path = run_root / "SCIENTIFIC_FREEZE.json"
    if not summary_path.is_file() or not freeze_path.is_file():
        raise RuntimeError("locked summary and scientific freeze are required")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("refusing to overwrite rendered evidence")
    output.mkdir(parents=True, exist_ok=True)
    data = _load(run_root)

    rows: list[dict[str, str | int | float]] = []
    for variant in VARIANTS:
        for regime in REGIMES:
            payloads = [data[variant][seed][regime] for seed in SEEDS]
            accuracy = [float(item["evaluation"]["decision_accuracy"]) for item in payloads]
            selective = [float(item["selective_accuracy_at_50_all_examples"]) for item in payloads]
            brier = [float(item["evaluation"]["risk"]["intrinsic_brier"]) for item in payloads]
            loss = [float(item["evaluation"]["total_loss"]) for item in payloads]
            accuracy_mean, accuracy_sd = _mean_sd(accuracy)
            selective_mean, selective_sd = _mean_sd(selective)
            brier_mean, brier_sd = _mean_sd(brier)
            loss_mean, loss_sd = _mean_sd(loss)
            rows.append(
                {
                    "variant": variant,
                    "regime": regime,
                    "seeds": len(SEEDS),
                    "accuracy_mean": accuracy_mean,
                    "accuracy_sd": accuracy_sd,
                    "selective_50_mean": selective_mean,
                    "selective_50_sd": selective_sd,
                    "intrinsic_brier_mean": brier_mean,
                    "intrinsic_brier_sd": brier_sd,
                    "total_loss_mean": loss_mean,
                    "total_loss_sd": loss_sd,
                }
            )
    table_path = output / "confirmatory_metrics.csv"
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    labels = ["Full", "Uniform", "No CF", "No risk Δ", "Mechanism off"]
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2"]
    figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=True)
    for axis, regime in zip(axes, REGIMES, strict=True):
        means = [
            statistics.mean(
                float(data[variant][seed][regime]["evaluation"]["decision_accuracy"])
                for seed in SEEDS
            )
            for variant in VARIANTS
        ]
        errors = [
            statistics.stdev(
                float(data[variant][seed][regime]["evaluation"]["decision_accuracy"])
                for seed in SEEDS
            )
            for variant in VARIANTS
        ]
        axis.bar(range(len(VARIANTS)), means, yerr=errors, color=colors, capsize=3)
        axis.set_title(regime.replace("_", " ").title())
        axis.set_xticks(range(len(VARIANTS)), labels, rotation=35, ha="right")
        axis.set_ylim(0.0, 1.05)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Decision accuracy (mean ± SD, n=5)")
    figure.tight_layout()
    accuracy_path = output / "confirmatory_accuracy.png"
    figure.savefig(accuracy_path, dpi=220, bbox_inches="tight")
    plt.close(figure)

    full_gates = {
        name: statistics.mean(
            float(
                data["full"][seed]["corrupted_test"]["evaluation"]["routing"][
                    "mean_gates"
                ][name]
            )
            for seed in SEEDS
        )
        for name in ("hermes", "prometheus", "perseus", "atlas", "kronos")
    }
    figure, axis = plt.subplots(figsize=(6.2, 3.5))
    axis.bar(list(full_gates), list(full_gates.values()), color="#4C78A8")
    axis.axhline(0.2, color="#F58518", linestyle="--", label="uniform weight")
    axis.set_ylim(0.0, 0.35)
    axis.set_ylabel("Mean gate weight")
    axis.set_title("Full model routing on corrupted protected data")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    gate_path = output / "corrupted_gate_weights.png"
    figure.savefig(gate_path, dpi=220, bbox_inches="tight")
    plt.close(figure)

    receipt = {
        "schema_version": 1,
        "status": "RENDERED_FROM_LOCKED_EVIDENCE",
        "confirmatory_summary_sha256": _sha256(summary_path),
        "scientific_freeze_sha256": _sha256(freeze_path),
        "outputs": {
            path.name: _sha256(path) for path in (table_path, accuracy_path, gate_path)
        },
    }
    (output / "render_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    render(
        Path("artifacts/research/olympus-controlled-fusion-v2/runs"),
        Path("research/olympus_irp_successor_v2/paper/figures"),
    )

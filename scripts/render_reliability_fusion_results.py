#!/usr/bin/env python3
"""Render descriptive v3 evidence without changing the frozen primary analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean, stdev
from tempfile import NamedTemporaryFile
from typing import Any

VARIANTS = (
    "full",
    "no_reliability_supervision",
    "semantic",
    "uniform",
    "reliability_only",
)
SEEDS = (20261601, 20261602, 20261603, 20261604, 20261605)
REGIMES = ("clean_test", "corrupted_test", "ood_test")
METRICS = (
    "decision_accuracy",
    "selective_accuracy_at_50_all_examples",
    "reliability_mae",
    "reliability_brier",
    "reliability_clean_minus_damaged",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(path)


def _summary(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": mean(values),
        "sample_sd": stdev(values) if len(values) > 1 else 0.0,
        "standard_error": stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0,
        "minimum": min(values),
        "maximum": max(values),
    }


def render(run_root: Path, output_root: Path) -> None:
    primary_path = run_root / "EVIDENCE_SUMMARY.json"
    freeze_path = run_root / "SCIENTIFIC_FREEZE.json"
    if not primary_path.is_file() or not freeze_path.is_file():
        raise RuntimeError("locked primary evidence and scientific freeze are required")

    rows: list[dict[str, Any]] = []
    aggregate: dict[str, dict[str, dict[str, dict[str, float | int] | None]]] = {}
    raw_hashes: dict[str, str] = {}
    risk_violations = 0
    transition_violations = 0
    checkpoint_bindings: set[tuple[str, int, str]] = set()
    for variant in VARIANTS:
        aggregate[variant] = {}
        for regime in REGIMES:
            observations: list[dict[str, Any]] = []
            for seed in SEEDS:
                path = run_root / "confirmatory" / variant / str(seed) / f"{regime}.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                raw_hashes[str(path.relative_to(run_root))] = _sha256(path)
                observations.append(payload)
                rows.append(payload)
                risk_violations += int(payload["risk_monotonicity_violations"])
                transition_violations += int(payload["transition_legality_violations"])
                checkpoint_bindings.add((variant, seed, payload["checkpoint_sha256"]))
            aggregate[variant][regime] = {}
            for metric in METRICS:
                values = [float(row[metric]) for row in observations if row[metric] is not None]
                aggregate[variant][regime][metric] = _summary(values) if values else None

    development_parameters: dict[str, list[int]] = {}
    development_elapsed: dict[str, dict[str, float | int]] = {}
    for variant in VARIANTS:
        counts: list[int] = []
        elapsed: list[float] = []
        for seed in SEEDS:
            path = run_root / "development" / variant / str(seed) / "report.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            counts.append(int(report["parameters"]))
            elapsed.append(float(report["elapsed_seconds"]))
        development_parameters[variant] = counts
        development_elapsed[variant] = _summary(elapsed)
    unique_parameter_counts = sorted(
        {count for counts in development_parameters.values() for count in counts}
    )

    primary = json.loads(primary_path.read_text(encoding="utf-8"))
    audit = {
        "schema_version": 1,
        "status": "DESCRIPTIVE_EVIDENCE_AUDIT_COMPLETE",
        "primary_analysis_sha256": _sha256(primary_path),
        "scientific_freeze_sha256": _sha256(freeze_path),
        "primary_disposition": primary["disposition"],
        "confirmatory_receipts": len(rows),
        "checkpoint_bindings": len(checkpoint_bindings),
        "risk_monotonicity_violations": risk_violations,
        "transition_legality_violations": transition_violations,
        "unique_parameter_counts": unique_parameter_counts,
        "parameter_matched": len(unique_parameter_counts) == 1,
        "development_elapsed_seconds": development_elapsed,
        "aggregate": aggregate,
        "raw_receipt_sha256": raw_hashes,
        "claim_to_evidence": [
            {
                "claim": "The v3 primary hypothesis is not supported.",
                "artifact": "EVIDENCE_SUMMARY.json",
                "analysis": "Frozen paired bootstrap over five common seeds.",
            },
            {
                "claim": "All controls are parameter matched.",
                "artifact": "development/*/*/report.json",
                "analysis": "Unique parameter-count audit across 25 reports.",
            },
            {
                "claim": "Host-risk monotonicity and transition legality were preserved.",
                "artifact": "confirmatory/*/*/*.json",
                "analysis": "Sum of violations across 75 protected receipts.",
            },
        ],
        "claim_boundary": (
            "Secondary aggregates are descriptive and cannot rescue the failed preregistered "
            "primary hypothesis. Synthetic mechanism scope only."
        ),
    }
    _atomic_write(
        output_root / "FULL_EVIDENCE_AUDIT.json",
        (json.dumps(audit, indent=2, sort_keys=True) + "\n").encode(),
    )

    lines = [
        "# Olympus v3 reliability-fusion results",
        "",
        f"Frozen disposition: **{primary['disposition']}**.",
        "",
        "Primary full-minus-semantic corrupted selective-accuracy delta: "
        f"{primary['paired_bootstrap']['mean_delta']:.4f} "
        f"(95% bootstrap CI {primary['paired_bootstrap']['ci95_low']:.4f} to "
        f"{primary['paired_bootstrap']['ci95_high']:.4f}; required effect "
        f"{primary['practical_effect_threshold']:.4f}).",
        "",
        "## Confirmatory means across five seeds",
        "",
        "| Variant | Regime | Accuracy | Selective@50 | Reliability MAE | Reliability separation |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        for regime in REGIMES:
            regime_metrics = aggregate[variant][regime]
            accuracy = regime_metrics["decision_accuracy"]
            selective = regime_metrics["selective_accuracy_at_50_all_examples"]
            reliability_mae = regime_metrics["reliability_mae"]
            separation = regime_metrics["reliability_clean_minus_damaged"]
            assert accuracy is not None
            assert selective is not None
            assert reliability_mae is not None
            separation_text = "n/a" if separation is None else f"{separation['mean']:.4f}"
            lines.append(
                f"| {variant} | {regime} | "
                f"{accuracy['mean']:.4f} | "
                f"{selective['mean']:.4f} | "
                f"{reliability_mae['mean']:.4f} | "
                f"{separation_text} |"
            )
    lines.extend(
        [
            "",
            "## Integrity audit",
            "",
            f"- Protected receipts: {len(rows)} of 75.",
            f"- Distinct checkpoint bindings: {len(checkpoint_bindings)} of 25.",
            f"- Parameter count: {unique_parameter_counts} "
            f"(matched={len(unique_parameter_counts) == 1}).",
            f"- Risk-monotonicity violations: {risk_violations}.",
            f"- Transition-legality violations: {transition_violations}.",
            "- Secondary results are descriptive and do not alter the negative disposition.",
            "- Scope is controlled synthetic evidence only.",
            "",
        ]
    )
    _atomic_write(output_root / "RESULTS.md", "\n".join(lines).encode())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root", type=Path, default=Path("artifacts/runs/olympus-reliability-v3")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("research/olympus_irp_successor_v3")
    )
    args = parser.parse_args()
    render(args.run_root, args.output_root)


if __name__ == "__main__":
    main()

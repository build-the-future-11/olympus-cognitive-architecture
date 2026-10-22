"""Independently verify retained multi-horizon dynamics evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def verify(
    root: Path,
    local_source: Path,
    local_protocol: Path,
    study_id: str,
    run_id: int,
) -> dict[str, Any]:
    output = root / study_id.replace("-", "_")
    retained_source = (
        root / f"multi_horizon_frozen_source/scripts/run_{study_id.replace('-', '_')}.py"
    )
    retained_protocol = (
        root
        / f"multi_horizon_frozen_source/research/{study_id.replace('-', '_')}/PROTOCOL.md"
    )
    manifest: dict[str, str] = load(output / "MANIFEST.json")
    freeze: dict[str, Any] = load(output / "SCIENTIFIC_FREEZE.json")
    admission: dict[str, Any] = load(output / "PROTECTED_ADMISSION.json")
    fits: list[dict[str, Any]] = load(output / "FIT_REPORTS.json")
    rows: list[dict[str, Any]] = load(output / "PROTECTED_ROWS.json")
    summary: dict[str, Any] = load(output / "SUMMARY.json")
    protocol: dict[str, Any] = freeze["protocol"]

    for relative, expected in manifest.items():
        artifact = output / relative
        if not artifact.is_file() or sha256(artifact) != expected:
            raise ValueError(f"manifest mismatch: {relative}")
    observed_manifest_files = {
        str(path.relative_to(output))
        for path in output.rglob("*")
        if path.is_file() and path.name != "MANIFEST.json"
    }
    if observed_manifest_files != set(manifest):
        raise ValueError("manifest denominator mismatch")

    source_hash = sha256(local_source)
    protocol_hash = sha256(local_protocol)
    if sha256(retained_source) != source_hash:
        raise ValueError("retained runner differs from frozen local runner")
    if sha256(retained_protocol) != protocol_hash:
        raise ValueError("retained protocol differs from frozen local protocol")
    identities = freeze["identities"]
    if identities != admission["identities"]:
        raise ValueError("freeze/admission identity mismatch")
    if identities["source_sha256"] != source_hash:
        raise ValueError("runner identity mismatch")
    if identities["protocol_sha256"] != protocol_hash:
        raise ValueError("protocol identity mismatch")
    if identities["protocol_object_sha256"] != hashlib.sha256(
        canonical(protocol)
    ).hexdigest():
        raise ValueError("protocol-object identity mismatch")
    if freeze["protected_generated"] is not False:
        raise ValueError("initial freeze was not pre-protected")
    if admission["protected_generated"] is not False:
        raise ValueError("admission receipt did not precede protected generation")
    if admission["resource_preflight"]["available_bytes"] < 2 * 1024**3:
        raise ValueError("resource floor not met")
    if study_id == "multi-horizon-dynamics-v2":
        network = freeze.get("network_preflight")
        if not isinstance(network, dict) or network.get("internet_disabled") is not True:
            raise ValueError("v2 Internet-off preflight was not satisfied")
        if admission.get("network_preflight") != network:
            raise ValueError("freeze/admission network preflight mismatch")

    dataset_seeds = protocol["dataset_seeds"]
    optimizer_seeds = protocol["optimizer_seeds"]
    models = protocol["models"]
    regimes = list(protocol["protected_offsets"])
    expected_fits = {
        (dataset_seed, optimizer_seed, mode)
        for dataset_seed in dataset_seeds
        for optimizer_seed in optimizer_seeds
        for mode in models
    }
    observed_fits = {
        (fit["dataset_seed"], fit["optimizer_seed"], fit["mode"]) for fit in fits
    }
    if observed_fits != expected_fits or len(fits) != len(expected_fits):
        raise ValueError("fit denominator mismatch")
    checkpoint_hashes = {
        f"checkpoints/{dataset_seed}-{optimizer_seed}-{mode}.pt": sha256(
            output / f"checkpoints/{dataset_seed}-{optimizer_seed}-{mode}.pt"
        )
        for dataset_seed, optimizer_seed, mode in sorted(expected_fits)
    }
    checkpoint_digest = hashlib.sha256(canonical(checkpoint_hashes)).hexdigest()
    if checkpoint_digest != admission["checkpoint_manifest_sha256"]:
        raise ValueError("checkpoint admission digest mismatch")
    for fit in fits:
        relative = (
            f"checkpoints/{fit['dataset_seed']}-{fit['optimizer_seed']}-{fit['mode']}.pt"
        )
        if fit["checkpoint_sha256"] != checkpoint_hashes[relative]:
            raise ValueError(f"fit checkpoint mismatch: {relative}")

    all_modes = [*models, "persistence"]
    expected_rows = {
        (dataset_seed, optimizer_seed, mode, regime)
        for dataset_seed in dataset_seeds
        for optimizer_seed in optimizer_seeds
        for mode in all_modes
        for regime in regimes
    }
    observed_rows = {
        (row["dataset_seed"], row["optimizer_seed"], row["mode"], row["regime"])
        for row in rows
    }
    if observed_rows != expected_rows or len(rows) != len(expected_rows):
        raise ValueError("protected-row denominator mismatch")
    for row in rows:
        horizon = row["horizon_mse"]
        values = [*horizon, row["mean_mse"], row["recovery_error"]]
        if len(horizon) != protocol["horizon"] or not all(
            math.isfinite(value) and value >= 0.0 for value in values
        ):
            raise ValueError("invalid protected metric")
        if row["mode"] == protocol["candidate"]:
            coverage = row.get("coverage90")
            if coverage is None or not 0.0 <= coverage <= 1.0:
                raise ValueError("invalid candidate coverage")
        elif "coverage90" in row:
            raise ValueError("unexpected control coverage")

    def clustered(mode: str, regime: str, key: str) -> list[float]:
        values: list[float] = []
        for dataset_seed in dataset_seeds:
            matched = [
                row
                for row in rows
                if row["dataset_seed"] == dataset_seed
                and row["mode"] == mode
                and row["regime"] == regime
            ]
            values.append(sum(float(row[key]) for row in matched) / len(matched))
        return values

    means = {
        mode: {
            regime: {
                "mean_mse": sum(clustered(mode, regime, "mean_mse"))
                / len(dataset_seeds),
                "horizon8_mse": sum(
                    float(row["horizon_mse"][-1])
                    for row in rows
                    if row["mode"] == mode and row["regime"] == regime
                )
                / (len(dataset_seeds) * len(optimizer_seeds)),
            }
            for regime in regimes
        }
        for mode in all_modes
    }
    candidate = protocol["candidate"]
    learned_controls = ["direct", "residual"]
    worst_regime = max(regimes, key=lambda item: means[candidate][item]["horizon8_mse"])
    best_control = min(
        learned_controls,
        key=lambda item: means[item][worst_regime]["horizon8_mse"],
    )
    candidate_clusters = clustered(candidate, worst_regime, "recovery_error")
    control_clusters = clustered(best_control, worst_regime, "recovery_error")
    deltas = [
        left - right
        for left, right in zip(candidate_clusters, control_clusters, strict=True)
    ]
    rng = random.Random(protocol["bootstrap_seed"])
    samples = sorted(
        sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas)
        for _ in range(protocol["bootstrap_samples"])
    )
    interval = [samples[250], samples[9749]]
    coverage = sum(
        float(row["coverage90"])
        for row in rows
        if row["mode"] == candidate and row["regime"] == worst_regime
    ) / (len(dataset_seeds) * len(optimizer_seeds))
    candidate_h8 = means[candidate][worst_regime]["horizon8_mse"]
    control_h8 = means[best_control][worst_regime]["horizon8_mse"]
    clean_regression = (
        means[candidate]["clean"]["horizon8_mse"]
        - means["residual"]["clean"]["horizon8_mse"]
    )
    criteria = {
        "worst_regime_ratio": candidate_h8 / control_h8,
        "paired_delta_ci95": interval,
        "clean_h8_regression": clean_regression,
        "coverage90": coverage,
    }
    primary = protocol["primary"]
    primary_supported = (
        criteria["worst_regime_ratio"] <= primary["worst_regime_h8_ratio_max"]
        and interval[1] < primary["paired_delta_ci_upper_max"]
        and clean_regression <= primary["clean_h8_regression_max"]
        and abs(coverage - 0.9) <= primary["coverage90_error_max"]
    )
    scalar_criteria = {
        "worst_regime_ratio": candidate_h8 / control_h8,
        "clean_h8_regression": clean_regression,
        "coverage90": coverage,
    }
    for key, value in scalar_criteria.items():
        if not close(value, summary["criteria"][key]):
            raise ValueError(f"summary criterion mismatch: {key}")
    if not all(
        close(left, right)
        for left, right in zip(
            interval, summary["criteria"]["paired_delta_ci95"], strict=True
        )
    ):
        raise ValueError("summary bootstrap interval mismatch")
    for mode in all_modes:
        for regime in regimes:
            for key in ("mean_mse", "horizon8_mse"):
                if not close(means[mode][regime][key], summary["means"][mode][regime][key]):
                    raise ValueError(f"summary mean mismatch: {mode}/{regime}/{key}")
    expected_status = "COMPLETED_SUPPORTED" if primary_supported else "COMPLETED_NEGATIVE"
    if summary["status"] != expected_status or summary["primary_supported"] != primary_supported:
        raise ValueError("summary disposition mismatch")
    if summary["promotion_authorized"] is not False:
        raise ValueError("unexpected promotion authorization")
    if summary["elapsed_seconds"] > protocol["max_wall_seconds"]:
        raise ValueError("wall budget exceeded")

    return {
        "status": "VERIFIED",
        "run_id": run_id,
        "scientific_status": expected_status,
        "primary_supported": primary_supported,
        "source_sha256": source_hash,
        "protocol_sha256": protocol_hash,
        "manifest_entries": len(manifest),
        "checkpoints": len(checkpoint_hashes),
        "fits": len(fits),
        "protected_rows": len(rows),
        "criteria": criteria,
        "claim_boundary": summary["claim_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--study-id", default="multi-horizon-dynamics-v1")
    parser.add_argument("--run-id", type=int, default=360757510)
    parser.add_argument("--receipt", type=Path)
    arguments = parser.parse_args()
    receipt = verify(
        arguments.root,
        arguments.source,
        arguments.protocol,
        arguments.study_id,
        arguments.run_id,
    )
    if arguments.receipt is not None:
        with arguments.receipt.open("x", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=2, sort_keys=True)
            handle.write("\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

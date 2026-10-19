"""Recompute the prespecified scientific analysis from retained raw forecasts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

FAMILIES = {"pendulum", "duffing"}
NEURAL_ARMS = {
    "full", "direct", "variance", "latent", "frozen_target", "shuffled_target",
    "direct_cpu_matched",
}
ARMS = NEURAL_ARMS | {"persistence", "linear_ridge", "cubic_ridge"}
CONDITIONS = {"id", "ood", "noise", "missing_velocity"}
HORIZONS = {"1", "10", "20", "50"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_analysis(contrasts: np.ndarray) -> dict[str, Any]:
    if contrasts.ndim != 1 or len(contrasts) != 8 or not np.isfinite(contrasts).all():
        raise ValueError("confirmation requires all eight finite paired block contrasts")
    rng = np.random.default_rng(93005)
    draws = rng.integers(0, 8, size=(10000, 8))
    effects = 1 - np.exp(-contrasts[draws].mean(axis=1))
    wins = int((contrasts > 0).sum())
    return {
        "block_log_ratios": contrasts.tolist(),
        "effect_fraction": float(1 - np.exp(-contrasts.mean())),
        "effect_ci95": np.quantile(effects, [0.025, 0.975]).tolist(),
        "wins": wins,
        "one_sided_sign_p": sum(math.comb(8, k) for k in range(wins, 9)) / 256,
        "independent_blocks": 8,
        "bootstrap_resamples": 10000,
        "bootstrap_seed": 93005,
    }


def validate_report_contract(
    report: dict[str, Any], frozen: dict[str, Any] | None = None,
    freeze_hash: str | None = None,
) -> None:
    if report["status"] != "COMPLETED":
        raise ValueError("incomplete attempt retained")
    if report["study_id"] != "olympus-observable-prediction-v1-20261019":
        raise ValueError("wrong study identity")
    phase = report["phase"]
    if phase not in {"dry", "development", "confirmation"}:
        raise ValueError("unknown study phase")
    if set(report["families"]) != FAMILIES:
        raise ValueError("incomplete or unexpected family matrix")
    expected_conditions = CONDITIONS if phase == "confirmation" else {"development"}
    for evidence in report["families"].values():
        if set(evidence["arms"]) != ARMS:
            raise ValueError("incomplete or unexpected arm matrix")
        for arm, record in evidence["arms"].items():
            if arm in NEURAL_ARMS and record["status"] != "COMPLETED":
                raise ValueError("a neural arm failed")
            if set(record["conditions"]) != expected_conditions:
                raise ValueError("incomplete or unexpected condition matrix")
            for metrics in record["conditions"].values():
                if set(metrics["horizons"]) != HORIZONS:
                    raise ValueError("incomplete or unexpected horizon matrix")
    if frozen is not None:
        if frozen["status"] != "FROZEN" or not freeze_hash:
            raise ValueError("actual frozen manifest is required")
        if phase != "confirmation" or report["freeze_sha256"] != freeze_hash:
            raise ValueError("report does not belong to the actual frozen confirmation")
        if report["configuration"]["id"] != frozen["selected_configuration"]:
            raise ValueError("configuration differs from frozen selection")
        if report["configuration"] != frozen["configuration"]:
            raise ValueError("configuration coefficients differ from frozen values")
        if report["protocol_sha256"] != frozen["protocol_sha256"]:
            raise ValueError("protocol differs from freeze")
        for source, expected in frozen["source_hashes"].items():
            if report["source"]["files_sha256"].get(source) != expected:
                raise ValueError("report source differs from freeze: " + source)
        if report["environment"] != frozen["environment"]:
            raise ValueError("environment differs from frozen runtime")


def cpu_matching_valid(candidate: dict[str, Any], control: dict[str, Any]) -> bool:
    budget = candidate["training_cpu_seconds"]
    elapsed = control["training_cpu_seconds"]
    return bool(
        candidate["status"] == control["status"] == "COMPLETED"
        and math.isfinite(budget) and math.isfinite(elapsed) and budget > 0
        and control["cpu_budget_seconds"] == budget
        and budget <= elapsed <= 1.05 * budget
        and control["maximum_steps"] == 3 * candidate["optimizer_steps"]
        and 0 < control["optimizer_steps"] < control["maximum_steps"]
        and not control["cap_reached"]
    )


def verify_report(
    path: Path, frozen: dict[str, Any] | None = None, freeze_hash: str | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = json.loads(path.read_text())
    validate_report_contract(report, frozen, freeze_hash)
    supervisor_path = path.parent.parent / (path.parent.name + "-supervisor.json")
    supervisor = json.loads(supervisor_path.read_text())
    if (
        supervisor["status"] != "COMPLETED" or supervisor["exit_code"] != 0
        or supervisor["monitor_failure"] is not None
        or supervisor["report_sha256"] != digest(path)
        or supervisor["protocol_sha256"] != report["protocol_sha256"]
        or supervisor["largest_child_peak_rss_bytes"] >= 768 * 1024**2
        or report["peak_rss_bytes"] >= 768 * 1024**2
    ):
        raise ValueError("supervisor did not admit this exact completed report")
    for family, evidence in report["families"].items():
        folder = path.parent / family
        if digest(folder / "data.npz") != evidence["data_sha256"]:
            raise ValueError("data artifact hash mismatch")
        if digest(folder / "forecasts.npz") != evidence["forecasts_sha256"]:
            raise ValueError("forecast artifact hash mismatch")
        data = np.load(folder / "data.npz", allow_pickle=False)
        arrays = np.load(folder / "forecasts.npz", allow_pickle=False)
        expected_data = {"train", "development", "mean", "std"}
        if report["phase"] == "confirmation":
            expected_data |= {"id", "ood"}
        if set(data.files) != expected_data:
            raise ValueError("unexpected data splits")
        if any(not np.isfinite(data[key]).all() for key in data.files):
            raise ValueError("nonfinite data or normalization")
        flat = data["train"].reshape(-1, 2)
        if not np.array_equal(flat.mean(axis=0), data["mean"]) or not np.array_equal(
            np.maximum(flat.std(axis=0), 1e-6), data["std"]
        ):
            raise ValueError("normalization is not training-only")
        starts: set[bytes] = set()
        for key in expected_data - {"mean", "std"}:
            values = data[key]
            expected_count = (8 if report["phase"] == "dry" else
                              24 if key == "development" else 64)
            if values.shape != (expected_count, 33 if key == "train" else 51, 2):
                raise ValueError("unexpected trajectory shape")
            initial = {row.tobytes() for row in values[:, 0]}
            if len(initial) != len(values) or initial & starts:
                raise ValueError("duplicate trajectory or split leakage")
            starts.update(initial)
        expected_arrays: set[str] = set()
        for arm, record in evidence["arms"].items():
            arm_path = folder / (arm + "-forecasts.npz")
            if digest(arm_path) != record["forecasts_sha256"]:
                raise ValueError("per-arm forecast artifact hash mismatch")
            arm_arrays = np.load(arm_path, allow_pickle=False)
            expected_arrays.update(arm_arrays.files)
            for key in arm_arrays.files:
                if not key.startswith(arm + "__") or not np.array_equal(
                    arrays[key], arm_arrays[key], equal_nan=True
                ):
                    raise ValueError("aggregate forecasts differ from immutable arm evidence")
            if "checkpoint_sha256" in record:
                if digest(folder / arm / "checkpoint.pt") != record["checkpoint_sha256"]:
                    raise ValueError("checkpoint artifact hash mismatch")
                if digest(folder / arm / "inference.pt") != record["inference_checkpoint_sha256"]:
                    raise ValueError("inference artifact hash mismatch")
                losses_path = folder / arm / "losses.jsonl"
                if digest(losses_path) != record["losses_sha256"]:
                    raise ValueError("training loss artifact hash mismatch")
                losses = [json.loads(row) for row in losses_path.read_text().splitlines()]
                if [row["step"] for row in losses] != list(
                    range(1, record["optimizer_steps"] + 1)
                ):
                    raise ValueError("optimizer step receipt differs from loss history")
                if arm != "direct_cpu_matched" and len(losses) != (
                    10 if report["phase"] == "dry" else 500
                ):
                    raise ValueError("fixed-update arm is incomplete")
            for condition, metrics in record["conditions"].items():
                name = "id" if condition in ("noise", "missing_velocity") else condition
                targets = (data[name][:, 1:] - data["mean"]) / data["std"]
                predictions = arrays[arm + "__" + condition + "__predictions"].astype(float)
                errors = arrays[arm + "__" + condition + "__errors"]
                if predictions.shape != targets.shape:
                    raise ValueError("forecast shape mismatch")
                failed = np.logical_or.accumulate(
                    ~np.isfinite(predictions).all(axis=2)
                    | (np.abs(predictions) > 1e6).any(axis=2), axis=1,
                )
                if int(failed[:, -1].sum()) != metrics["divergent_trajectories"]:
                    raise ValueError("divergence receipt differs from raw predictions")
                safe = np.where(failed[:, :, None], 0, predictions)
                recomputed = ((safe - targets) ** 2).mean(axis=2)
                recomputed[failed] = 1e12
                if not np.allclose(recomputed, errors, rtol=1e-12, atol=1e-12):
                    raise ValueError("retained errors do not match raw predictions and targets")
                for horizon, recorded in metrics["horizons"].items():
                    measured = float(recomputed[:, :int(horizon)].mean())
                    if not math.isclose(measured, recorded, rel_tol=1e-12, abs_tol=1e-12):
                        raise ValueError("reported horizon metric does not match raw evidence")
        if set(arrays.files) != expected_arrays:
            raise ValueError("aggregate array matrix is incomplete or unexpected")
    return report


def analyze(paths: list[Path], freeze_path: Path) -> dict[str, Any]:
    frozen: dict[str, Any] = json.loads(freeze_path.read_text())
    freeze_hash = digest(freeze_path)
    reports = [verify_report(path, frozen, freeze_hash) for path in paths]
    if len(reports) != 8 or sorted(r["seed"] for r in reports) != list(range(1701, 1709)):
        raise ValueError("all eight declared confirmation seeds are required exactly once")
    if any(r["phase"] != "confirmation" for r in reports):
        raise ValueError("development outcomes cannot enter confirmation")
    if len({r["freeze_sha256"] for r in reports}) != 1 or not reports[0]["freeze_sha256"]:
        raise ValueError("confirmation reports must share the same freeze")
    reports.sort(key=lambda r: r["seed"])
    families = ["pendulum", "duffing"]
    arms = list(reports[0]["families"][families[0]]["arms"])
    summary: dict[str, Any] = {"families": {}, "source_reports": {
        str(p): digest(p) for p in paths
    }, "freeze_sha256": reports[0]["freeze_sha256"]}
    contrasts: list[float] = []
    id_contrasts: list[float] = []
    per_family: dict[str, list[float]] = {f: [] for f in families}
    matching_valid, no_divergence = True, True
    for report in reports:
        block: list[float] = []
        id_block: list[float] = []
        for family in families:
            records = report["families"][family]["arms"]
            candidate, control = records["full"], records["direct_cpu_matched"]
            for condition, destination in [("ood", block), ("id", id_block)]:
                before = control["conditions"][condition]["horizons"]["20"]
                after = candidate["conditions"][condition]["horizons"]["20"]
                destination.append(math.log((before + 1e-12) / (after + 1e-12)))
            per_family[family].append(block[-1])
            measured_match = cpu_matching_valid(candidate, control)
            if measured_match != control["cpu_match_valid"]:
                raise ValueError("CPU matching receipt disagrees with measured times and steps")
            matching_valid &= measured_match
            no_divergence &= all(c["divergent_trajectories"] == 0
                                 for c in candidate["conditions"].values())
        contrasts.append(sum(block) / 2)
        id_contrasts.append(sum(id_block) / 2)
    primary = paired_analysis(np.array(contrasts))
    primary["per_family_effect_fraction"] = {
        f: 1 - math.exp(-sum(v) / len(v)) for f, v in per_family.items()
    }
    primary["id_error_ratio"] = math.exp(-sum(id_contrasts) / len(id_contrasts))
    primary["cpu_matching_valid"] = matching_valid
    primary["candidate_no_divergence"] = no_divergence
    lower, upper = primary["effect_ci95"]
    main_pass = lower >= 0.1 and primary["one_sided_sign_p"] <= 0.05
    guards = (primary["id_error_ratio"] <= 1.05 and no_divergence
              and all(e > 0 for e in primary["per_family_effect_fraction"].values()))
    if not matching_valid:
        disposition = "INCONCLUSIVE"
    elif main_pass and guards:
        disposition = "SUPPORTED"
    elif main_pass:
        disposition = "MIXED"
    elif upper < 0.1:
        disposition = "NEGATIVE"
    else:
        disposition = "INCONCLUSIVE"
    summary.update({"primary": primary, "disposition": disposition})
    for family in families:
        summary["families"][family] = {}
        for arm in arms:
            records = [r["families"][family]["arms"][arm] for r in reports]
            summary["families"][family][arm] = {
                "conditions": {condition: {h: float(np.mean([
                    r["conditions"][condition]["horizons"][h] for r in records
                ])) for h in ["1", "10", "20", "50"]}
                for condition in ["id", "ood", "noise", "missing_velocity"]},
                "mean_training_wall_seconds": float(np.mean([
                    r["training_wall_seconds"] for r in records
                ])),
                "trainable_parameters": records[0]["trainable_parameters"],
                "replicate_ood20": [r["conditions"]["ood"]["horizons"]["20"] for r in records],
            }
    summary["execution"] = {
        "blocks": len(reports), "families": len(families), "arms": len(arms),
        "neural_fits": sum(
            arm in NEURAL_ARMS for r in reports
            for f in r["families"].values() for arm in f["arms"]
        ),
        "total_block_wall_seconds": sum(r["elapsed_seconds"] for r in reports),
        "largest_process_peak_rss_bytes": max(r["peak_rss_bytes"] for r in reports),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("analysis output must be fresh")
    summary = analyze(args.reports, args.freeze)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        "disposition": summary["disposition"], "primary": summary["primary"],
    }, indent=2))


if __name__ == "__main__":
    main()

"""Standard-library provenance and independent report-arithmetic checks for v5."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(root: Path) -> dict[str, Any]:
    run = root / "artifacts/robust-fusion-v5"
    contract, freeze, summary = (
        read(run / p) for p in ("CONTRACT.json", "SCIENTIFIC_FREEZE.json", "SUMMARY.json")
    )
    protocol = contract["protocol"]
    assert protocol == summary["protocol"]
    assert sha(run / "CONTRACT.json") == freeze["contract_sha256"]
    assert sha(run / "SCIENTIFIC_FREEZE.json") == summary["freeze_sha256"]
    assert freeze["sources"] == contract["sources"]
    for name, digest in freeze["sources"].items():
        assert sha(root / name) == digest, name
    for name, digest in freeze["bindings"].items():
        assert sha(run / name) == digest, name
    rows = [read(p) for p in sorted((run / "protected").glob("*/report.json"))]
    expected = {
        (ds, seed, coverage)
        for ds in protocol["dataset_seeds"]
        for seed in protocol["training_seeds"]
        for coverage in protocol["coverage"]
    }
    assert len(rows) == summary["cells"] == len(expected) == 20
    assert {(r["dataset"], r["training_seed"], r["coverage"]) for r in rows} == expected
    assert summary["protected_reports"] == 80
    data: dict[tuple[int, str], Any] = {}
    parameters, prediction_count = set(), 0
    for row in rows:
        name = f"d{row['dataset']}-s{row['training_seed']}-{row['coverage']}"
        dev = read(run / "development" / name / "report.json")
        parameters.add((dev["parameters_bank"], dev["parameters_heads"]))

        def score(route: str, development: Any = dev) -> tuple[float, float]:
            acc = [
                development["development"][regime]["routing"][route]["accuracy"]
                for regime in protocol["development_regimes"]
            ]
            return min(acc), statistics.mean(acc)

        selected = max(protocol["candidate_routes"], key=score)
        assert selected == dev["selected_route"] == row["selected_route"]
        assert (
            selected
            == summary["selected_routes"][
                row["coverage"] + "/" + str(row["dataset"]) + "/" + str(row["training_seed"])
            ]
        )
        fit_sets = []
        for fold in dev["folds"]:
            fit, held = set(fold["fit_ids"]), set(fold["router_target_ids"])
            assert fit.isdisjoint(held)
            assert len(fit) == len(held) == 512 and len(fit | held) == 1024
            fit_sets.append(fit)
        assert fit_sets[0].isdisjoint(fit_sets[1])
        assert set(dev["development"]) == set(protocol["development_regimes"])
        assert "novel_shift" not in dev["development"]
        assert set(row["regimes"]) == set(protocol["protected_regimes"])
        for regime, report in row["regimes"].items():
            key = (row["dataset"], regime)
            if key in data:
                assert report["data_hashes"] == data[key]
            data[key] = report["data_hashes"]
            assert sha(run / "protected" / name / f"{regime}.pt") == report["prediction_sha256"]
            prediction_count += 1
        for regime in protocol["development_regimes"]:
            assert (
                dev["train_hashes"]["source_ids"] != dev["development_hashes"][regime]["source_ids"]
            )
        for report in row["regimes"].values():
            assert dev["train_hashes"]["source_ids"] != report["data_hashes"]["source_ids"]
            assert all(
                v["source_ids"] != report["data_hashes"]["source_ids"]
                for v in dev["development_hashes"].values()
            )
    assert len(parameters) == 1 and prediction_count == 80 and len(data) == 20

    def metric(row: Any, regime: str, kind: str, mode: str) -> float:
        if kind == "routing":
            mode = row["selected_route"] if mode == "selected" else mode
            return float(row["regimes"][regime][kind][mode]["accuracy"])
        return float(row["regimes"][regime][kind][mode])

    for coverage, regimes in summary["means"].items():
        selected = [r for r in rows if r["coverage"] == coverage]
        for regime, kinds in regimes.items():
            for kind, modes in kinds.items():
                for mode, actual in modes.items():
                    assert math.isclose(
                        actual,
                        statistics.mean(metric(r, regime, kind, mode) for r in selected),
                        abs_tol=1e-12,
                    )
    for name, actual in summary["comparisons"].items():
        regime, kind, contrast = name.split("/")
        left, right = contrast.split("-minus-")
        lc, lm = left.split(":")
        rc, rm = right.split(":")
        deltas = []
        for ds in protocol["dataset_seeds"]:
            left_values = [
                metric(r, regime, kind, lm)
                for r in rows
                if r["dataset"] == ds and r["coverage"] == lc
            ]
            right_values = [
                metric(r, regime, kind, rm)
                for r in rows
                if r["dataset"] == ds and r["coverage"] == rc
            ]
            deltas.append(statistics.mean(left_values) - statistics.mean(right_values))
        assert all(
            math.isclose(a, b, abs_tol=1e-12)
            for a, b in zip(deltas, actual["dataset_deltas"], strict=True)
        ), name
        assert math.isclose(statistics.mean(deltas), actual["mean"], abs_tol=1e-12), name
    comparisons, means = summary["comparisons"], summary["means"]
    route = comparisons["novel_shift/routing/broad:selected-minus-core:semantic"]
    guard = comparisons["corrupted/routing/broad:selected-minus-core:semantic"]
    temporal = comparisons["clean/temporal/broad:residual-minus-broad:persistence"]
    route_ok = (
        route["mean"] >= 0.02
        and route["ci95"][0] > 0
        and guard["mean"] >= -0.01
        and guard["ci95"][0] >= -0.02
    )
    temporal_ok = (
        means["broad"]["clean"]["temporal"]["residual"]
        <= 0.9
        * min(
            means["core"]["clean"]["temporal"]["v4"],
            means["broad"]["clean"]["temporal"]["persistence"],
        )
        and temporal["ci95"][1] < 0
        and means["broad"]["corrupted"]["temporal"]["residual"]
        <= means["core"]["corrupted"]["temporal"]["v4"] + 0.02
    )
    assert route_ok == summary["routing_primary_supported"]
    assert temporal_ok == summary["temporal_primary_supported"]
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    development = [e for e in events if e["phase"] == "development_complete"]
    protected = [e for e in events if e["phase"] == "protected_complete"]
    assert len(development) == len(protected) == 20
    assert max(e["time"] for e in development) <= freeze["time"] < min(e["time"] for e in protected)
    assert summary["promotion_authorized"] is False
    return {
        "status": "PASS",
        "cells": 20,
        "bank_fits": 40,
        "protected_prediction_hashes_verified": prediction_count,
        "matched_parameters": list(parameters)[0],
        "all_aggregate_means_recomputed": True,
        "comparison_cluster_deltas_recomputed": len(summary["comparisons"]),
        "development_only_selection_verified": True,
        "fold_overlap": 0,
        "common_protected_tensors_verified": True,
        "source_checkpoint_freeze_and_chronology_verified": True,
        "primary_predicates_verified": True,
        "routing_primary_supported": route_ok,
        "temporal_primary_supported": temporal_ok,
        "limitation": (
            "Report arithmetic and provenance; not independent model retraining "
            "or raw tensor inference verification"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.root.resolve()), indent=2))


if __name__ == "__main__":
    main()

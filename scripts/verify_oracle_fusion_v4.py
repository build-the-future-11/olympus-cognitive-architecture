"""Independent, standard-library verification of downloaded v4 evidence.

This checks provenance, complete denominators, selection and aggregate arithmetic.
It does not claim to re-execute Torch inference from the checkpoint.
"""

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
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(root: Path) -> dict[str, Any]:
    run = root / "artifacts/oracle-fusion-v4"
    contract = read(run / "CONTRACT.json")
    freeze = read(run / "SCIENTIFIC_FREEZE.json")
    summary = read(run / "SUMMARY.json")
    assert sha(run / "CONTRACT.json") == freeze["contract_sha256"]
    assert sha(run / "SCIENTIFIC_FREEZE.json") == summary["freeze_sha256"]
    assert freeze["sources"] == contract["sources"]
    for name, expected in freeze["sources"].items():
        assert sha(root / name) == expected, name
    for name, expected in freeze["bindings"].items():
        assert sha(run / name) == expected, name
    protocol = contract["protocol"]
    rows = [read(p) for p in sorted((run / "protected").glob("*/report.json"))]
    expected_cells = (
        len(protocol["dataset_seeds"])
        * len(protocol["training_seeds"])
        * len(protocol["coverage"])
        * len(protocol["counterfactual_weights"])
    )
    assert len(rows) == expected_cells == summary["cells"]
    assert summary["protected_reports"] == expected_cells * 3
    conditions = {(r["dataset"], r["training_seed"], r["coverage"], r["cf"]) for r in rows}
    assert len(conditions) == expected_cells
    expected_conditions = {
        (ds, seed, coverage, cf)
        for ds in protocol["dataset_seeds"]
        for seed in protocol["training_seeds"]
        for coverage in protocol["coverage"]
        for cf in protocol["counterfactual_weights"]
    }
    assert conditions == expected_conditions
    parameters = set()
    prediction_files = 0
    data_hashes: dict[tuple[int, str], Any] = {}
    for row in rows:
        dev = read(run / "development" / row["cell"] / "report.json")
        parameters.add((dev["parameters_bank"], dev["parameters_routers"]))
        for fold in dev["folds"]:
            assert set(fold["fit_ids"]).isdisjoint(fold["router_target_ids"])
            assert fold["overlap"] == 0
        expected_control = max(
            protocol["strong_controls"], key=lambda m: dev["development"][m]["accuracy"]
        )
        assert expected_control == dev["selected_control"] == row["selected_control"]
        for regime, report in row["regimes"].items():
            prediction = run / "protected" / row["cell"] / f"{regime}-predictions.pt"
            assert sha(prediction) == report["prediction_sha256"]
            prediction_files += 1
            key = (row["dataset"], regime)
            if key in data_hashes:
                assert data_hashes[key] == report["data_hashes"]
            data_hashes[key] = report["data_hashes"]
    assert len(parameters) == 1
    assert len(data_hashes) == len(protocol["dataset_seeds"]) * 3
    for key, modes in summary["means"].items():
        coverage, cf_string, regime = key.split("/")
        cf = float(cf_string[2:])
        matched = [r for r in rows if r["coverage"] == coverage and r["cf"] == cf]
        for mode, values in modes.items():
            for metric, observed in values.items():
                if mode == "diagnostics":
                    expected = statistics.mean(r["regimes"][regime][metric] for r in matched)
                else:
                    expected = statistics.mean(r["regimes"][regime][mode][metric] for r in matched)
                assert math.isclose(expected, observed, abs_tol=1e-10), (key, mode, metric)
    primary_deltas = []
    for ds in protocol["dataset_seeds"]:
        matched = [
            r for r in rows if r["dataset"] == ds and r["coverage"] == "mixed" and r["cf"] == 0
        ]
        primary_deltas.append(
            statistics.mean(
                r["regimes"]["corrupted"]["utility"]["accuracy"]
                - r["regimes"]["corrupted"][r["selected_control"]]["accuracy"]
                for r in matched
            )
        )
    assert primary_deltas == summary["primary"]["dataset_deltas"]
    assert math.isclose(statistics.mean(primary_deltas), summary["primary"]["mean"], abs_tol=1e-12)
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    development = [e for e in events if e["phase"] == "development_complete"]
    protected = [e for e in events if e["phase"] == "protected_complete"]
    assert len(development) == len(protected) == expected_cells
    assert max(e["time"] for e in development) <= freeze["time"]
    assert min(e["time"] for e in protected) > freeze["time"]
    return {
        "status": "PASS",
        "cells": len(rows),
        "prediction_files_hash_verified": prediction_files,
        "parameters": list(parameters)[0],
        "development_only_selection": True,
        "crossfit_overlap": 0,
        "common_protected_data_verified": True,
        "all_reported_means_recomputed": True,
        "primary_cluster_deltas_recomputed": primary_deltas,
        "freeze_precedes_protected_evaluation": True,
        "limitation": "Provenance and aggregate verification; not independent model retraining",
    }


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.root.resolve()), indent=2))


if __name__ == "__main__":
    main()

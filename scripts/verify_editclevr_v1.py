"""Standard-library provenance and arithmetic verifier for EditCLEVR v1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def interval(values: list[float], *, samples: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    bootstraps = sorted(
        statistics.mean(rng.choices(values, k=len(values))) for _ in range(samples)
    )
    return [bootstraps[250], bootstraps[9749]]


def verify(root: Path) -> dict[str, Any]:
    run = root / "artifacts/editclevr-v1"
    contract = read(run / "CONTRACT.json")
    freeze = read(run / "SCIENTIFIC_FREEZE.json")
    summary = read(run / "SUMMARY.json")
    protocol = contract["protocol"]
    assert protocol == summary["protocol"]
    assert sha(run / "CONTRACT.json") == freeze["contract_sha256"]
    assert contract["sources"] == freeze["sources"]
    for path, digest in contract["sources"].items():
        assert sha(root / path) == digest, path
    for path, digest in freeze["bindings"].items():
        assert sha(run / path) == digest, path

    rows = []
    parameter_counts = set()
    protected_id_hashes: dict[str, str] = {}
    for seed in protocol["training_seeds"]:
        development = read(run / "development" / f"seed-{seed}.json")
        protected = read(run / "protected" / f"seed-{seed}.json")
        assert protected["time"] > freeze["time"]
        assert protected["selected_control"] == development["selected_control"]
        assert development["selected_control"] == max(
            ("pair_only", "uniform", "delta_only"),
            key=lambda mode: development["report"][mode]["accuracy"],
        )
        assert 1 <= development["best_epoch"] <= protocol["epochs"]
        assert len(development["history"]) == protocol["epochs"]
        parameter_counts.add(development["parameters"])
        assert set(protected["reports"]) == set(protocol["protected_splits"])
        for split in protocol["protected_splits"]:
            path = run / "protected" / f"seed-{seed}-{split}.pt"
            assert sha(path) == protected["prediction_hashes"][split]
            if split in protected_id_hashes:
                assert protected_id_hashes[split] == protected["pair_ids_sha256"][split]
            protected_id_hashes[split] = protected["pair_ids_sha256"][split]
            assert protected["reports"][split]["learned"]["examples"] > 0
        rows.append(protected)
    assert len(parameter_counts) == 1
    assert len(set(protected_id_hashes.values())) == len(protocol["protected_splits"])

    shifted, in_distribution = [], []
    for row in rows:
        control = row["selected_control"]
        shifted.append(
            statistics.mean(
                row["reports"][split]["learned"]["accuracy"]
                - row["reports"][split][control]["accuracy"]
                for split in ("test_hard", "test_cogent")
            )
        )
        in_distribution.append(
            row["reports"]["test_id"]["learned"]["accuracy"]
            - row["reports"]["test_id"][control]["accuracy"]
        )
    assert all(
        math.isclose(left, right, abs_tol=1e-12)
        for left, right in zip(
            shifted, summary["shifted_learned_minus_control"]["seed_deltas"], strict=True
        )
    )
    assert all(
        math.isclose(left, right, abs_tol=1e-12)
        for left, right in zip(
            in_distribution, summary["id_learned_minus_control"]["seed_deltas"], strict=True
        )
    )
    assert math.isclose(
        statistics.mean(shifted), summary["shifted_learned_minus_control"]["mean"], abs_tol=1e-12
    )
    assert math.isclose(
        statistics.mean(in_distribution), summary["id_learned_minus_control"]["mean"], abs_tol=1e-12
    )
    shifted_ci = interval(
        shifted, samples=protocol["bootstrap_samples"], seed=protocol["bootstrap_seed"]
    )
    id_ci = interval(
        in_distribution,
        samples=protocol["bootstrap_samples"],
        seed=protocol["bootstrap_seed"],
    )
    assert all(
        math.isclose(left, right, abs_tol=1e-12)
        for left, right in zip(
            shifted_ci, summary["shifted_learned_minus_control"]["ci95"], strict=True
        )
    )
    assert all(
        math.isclose(left, right, abs_tol=1e-12)
        for left, right in zip(
            id_ci, summary["id_learned_minus_control"]["ci95"], strict=True
        )
    )
    supported = (
        statistics.mean(shifted) >= 0.02
        and shifted_ci[0] > 0
        and statistics.mean(in_distribution) >= -0.01
        and id_ci[0] >= -0.02
    )
    assert supported == summary["primary_supported"]
    assert summary["promotion_authorized"] is False
    return {
        "status": "PASS",
        "seeds": len(rows),
        "protected_prediction_hashes": len(rows) * len(protocol["protected_splits"]),
        "parameter_count": parameter_counts.pop(),
        "development_only_control_selection": True,
        "common_split_identities_across_seeds": True,
        "source_checkpoint_freeze_and_chronology_verified": True,
        "aggregate_deltas_and_intervals_recomputed": True,
        "primary_supported": supported,
        "limitation": "Provenance and report arithmetic; not independent retraining",
    }


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.root.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

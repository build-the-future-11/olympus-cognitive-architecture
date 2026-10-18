"""Fresh, immutable v5 routing and temporal matched-ablation campaign."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

from scripts import run_oracle_fusion_v4 as v4

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research/olympus_irp_successor_v5"
PROTOCOL: dict[str, Any] = {
    "id": "olympus-robust-fusion-v5.0.0",
    "dataset_seeds": [20265100, 20265200, 20265300, 20265400, 20265500],
    "training_seeds": [20265601, 20265602],
    "coverage": ["core", "broad"],
    "folds": 2,
    "train_examples": 1024,
    "development_examples_per_regime": 256,
    "test_examples_per_regime": 1024,
    "expert_steps": 250,
    "router_steps": 300,
    "temporal_steps": 500,
    "batch_size": 64,
    "lr": 0.001,
    "weight_decay": 0.01,
    "cf_weight": 0,
    "development_regimes": ["clean", "corrupted", "known_shift"],
    "protected_regimes": ["clean", "corrupted", "known_shift", "novel_shift"],
    "candidate_routes": ["semantic", "prediction_only", "utility", "blend", "fixed", "corruption"],
    "selection": (
        "maximize minimum development accuracy across 3 regimes, then mean "
        "accuracy, then declared order"
    ),
    "bootstrap_samples": 10000,
    "bootstrap_seed": 20265999,
    "routing_primary": (
        "broad selected vs core semantic on fresh novel_shift; >=2pp, CI lower >0; "
        "corrupted mean loss <=1pp and CI lower >=-2pp"
    ),
    "temporal_primary": (
        "broad residual clean MSE <=90% of core v4 and persistence; CI for "
        "residual-minus-persistence below zero; corrupted MSE <= core v4 +0.02"
    ),
    "max_wall_seconds": 3600,
    "scope": "synthetic mechanism study; no SOTA or deployment claim",
}


def sources() -> dict[str, str]:
    paths = sorted((ROOT / "olympus").rglob("*.py"))
    paths += [
        Path(__file__),
        ROOT / "scripts/run_oracle_fusion_v4.py",
        ROOT / "tests/test_robust_fusion_v5.py",
        RESEARCH / "PROTOCOL.md",
    ]
    return {str(p.relative_to(ROOT)): v4.sha(p) for p in paths}


def verify_prior() -> dict[str, Any]:
    remote = ROOT / "V5_REMOTE_PRESERVATION.json"
    if remote.exists():
        receipt = v4.read(remote)
        assert receipt["inventory_sha256"] == v4.sha(RESEARCH / "PRIOR_INVENTORY.json")
        assert receipt["local_verification"]["immutable_evidence_unchanged"]
        return {
            "unchanged": True,
            "scope": "original host pre-upload only; recheck after retrieval",
            "receipt": receipt,
        }
    files = v4.read(RESEARCH / "PRIOR_INVENTORY.json")["files"]
    changed = [
        p
        for p, value in files.items()
        if not (ROOT / p).is_file() or v4.sha(ROOT / p) != value["sha256"]
    ]
    immutable = [path for path in changed if path.startswith("artifacts/")]
    source_changes = [path for path in changed if path not in immutable]
    return {
        "files_checked": len(files),
        "immutable_evidence_changed_or_missing": immutable,
        "mutable_source_changed_or_missing": source_changes,
        "immutable_evidence_unchanged": not immutable,
        "unchanged": not immutable,
    }


def specs() -> Any:
    for ds in PROTOCOL["dataset_seeds"]:
        for seed in PROTOCOL["training_seeds"]:
            for coverage in PROTOCOL["coverage"]:
                yield ds, seed, coverage


def name(ds: int, seed: int, coverage: str) -> str:
    return f"d{ds}-s{seed}-{coverage}"


def fit_heads(oof: dict[str, Any], seed: int, deadline: float) -> Any:
    import torch
    from torch.nn import functional as F

    from olympus.models.robust_fusion_v5 import Heads, routed

    torch.manual_seed(seed + 70000)
    heads = Heads()
    # Separate optimizers/clipping keep temporal gradients from changing routing.
    routing_params = [heads.fixed, *heads.routing.parameters()]
    router_opt = torch.optim.AdamW(
        routing_params, lr=PROTOCOL["lr"], weight_decay=PROTOCOL["weight_decay"]
    )
    temporal_opts = {
        k: torch.optim.AdamW(
            net.parameters(), lr=PROTOCOL["lr"], weight_decay=PROTOCOL["weight_decay"]
        )
        for k, net in heads.temporal.items()
    }
    sampler = torch.Generator().manual_seed(seed + 80000)
    for step in range(PROTOCOL["temporal_steps"]):
        ix = torch.randint(len(oof["labels"]), (PROTOCOL["batch_size"],), generator=sampler)
        z, logits, states, actions = (oof[k][ix] for k in ("z", "logits", "states", "actions"))
        scores, temporal = heads(z, logits, states, actions)
        if step < PROTOCOL["router_steps"]:
            predictions = routed(logits, scores)
            loss = sum(
                F.nll_loss(predictions[k].clamp_min(1e-8).log(), oof["labels"][ix])
                for k in ("fixed", "semantic", "prediction_only")
            )
            loss += F.binary_cross_entropy_with_logits(
                scores["utility"], logits.argmax(-1).eq(oof["labels"][ix, None]).float()
            )
            loss += F.binary_cross_entropy_with_logits(scores["corruption"], oof["quality"][ix])
            assert torch.isfinite(loss)
            router_opt.zero_grad(set_to_none=True)
            torch.autograd.backward(loss)
            torch.nn.utils.clip_grad_norm_(routing_params, 1.0)
            router_opt.step()
        for k, opt in temporal_opts.items():
            loss = F.mse_loss(temporal[k], oof["target"][ix])
            assert torch.isfinite(loss)
            opt.zero_grad(set_to_none=True)
            torch.autograd.backward(loss)
            torch.nn.utils.clip_grad_norm_(heads.temporal[k].parameters(), 1.0)
            opt.step()
        if (step + 1) % 50 == 0:
            v4.resource_check()
            if time.monotonic() > deadline:
                raise RuntimeError("fixed development wall budget exhausted")
    return heads.eval().requires_grad_(False)


def evaluate(banks: Any, heads: Any, batch: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    from olympus.models.robust_fusion_v5 import ROUTES, TEMPORAL, routed

    routes: dict[str, list[Any]] = {k: [] for k in ROUTES}
    times: dict[str, list[Any]] = {k: [] for k in TEMPORAL}
    with torch.no_grad():
        for bank in banks:
            z, logits, old_time = v4.predict(bank, batch)
            scores, temporal = heads(z, logits, batch["states"], batch["actions"])
            for key, p in routed(logits, scores, batch["decision_targets"]).items():
                routes[key].append(p)
            for key, p in {**temporal, "v4": old_time, "persistence": batch["states"]}.items():
                times[key].append(p)
    probabilities = {k: torch.stack(v).mean(0) for k, v in routes.items()}
    temporal_predictions = {k: torch.stack(v).mean(0) for k, v in times.items()}
    report = {
        "routing": {k: v4.metrics(p, batch["decision_targets"]) for k, p in probabilities.items()},
        "temporal": {
            k: float((p - batch["next_state_targets"]).square().mean())
            for k, p in temporal_predictions.items()
        },
    }
    archive = {
        "probabilities": probabilities,
        "temporal_predictions": temporal_predictions,
        "labels": batch["decision_targets"],
        "temporal_target": batch["next_state_targets"],
        "source_ids": batch["source_ids"],
    }
    return report, archive


def select_route(reports: dict[str, Any]) -> str:
    def score(mode: str) -> tuple[float, float]:
        values = [reports[r]["routing"][mode]["accuracy"] for r in PROTOCOL["development_regimes"]]
        return min(values), sum(values) / len(values)

    return str(max(PROTOCOL["candidate_routes"], key=score))


def develop(output: Path, deadline: float) -> None:
    import torch

    from olympus.data.oracle_fusion_v4 import subset
    from olympus.data.robust_fusion_v5 import generate

    v4.write(
        output / "CONTRACT.json",
        {
            "protocol": PROTOCOL,
            "sources": sources(),
            "torch": torch.__version__,
            "python": sys.version,
            "platform": platform.platform(),
            "threads": 1,
        },
    )
    for ds, seed, coverage in specs():
        cell = output / "development" / name(ds, seed, coverage)
        cell.mkdir(parents=True, exist_ok=False)
        start = time.monotonic()
        train = generate(ds + 1, PROTOCOL["train_examples"], coverage)
        banks, folds = [], []
        accum: dict[str, list[Any]] = {
            k: [] for k in ("z", "logits", "states", "actions", "labels", "quality", "target")
        }
        for fold in range(2):
            fit = subset(train, torch.arange(fold, len(train["tokens"]), 2))
            held = subset(train, torch.arange(1 - fold, len(train["tokens"]), 2))
            bank = v4.fit_bank(fit, seed + fold * 100, 0, output, deadline)
            banks.append(bank)
            z, logits, _ = v4.predict(bank, held)
            values = {
                "z": z,
                "logits": logits,
                "states": held["states"],
                "actions": held["actions"],
                "labels": held["decision_targets"],
                "quality": held["component_quality"],
                "target": held["next_state_targets"],
            }
            for k, value in values.items():
                accum[k].append(value)
            folds.append(
                {
                    "fit_ids": fit["source_ids"].tolist(),
                    "router_target_ids": held["source_ids"].tolist(),
                }
            )
            v4.save_tensor(cell / f"bank-{fold}.pt", bank.state_dict())
        heads = fit_heads({k: torch.cat(v) for k, v in accum.items()}, seed, deadline)
        v4.save_tensor(cell / "heads.pt", heads.state_dict())
        development, dev_hashes = {}, {}
        for offset, regime in enumerate(PROTOCOL["development_regimes"]):
            batch = generate(ds + 2 + offset, PROTOCOL["development_examples_per_regime"], regime)
            development[regime], _ = evaluate(banks, heads, batch)
            dev_hashes[regime] = v4.tensors_hash(batch)
        selected = select_route(development)
        v4.write(
            cell / "report.json",
            {
                "dataset": ds,
                "training_seed": seed,
                "coverage": coverage,
                "selected_route": selected,
                "development": development,
                "folds": folds,
                "train_hashes": v4.tensors_hash(train),
                "development_hashes": dev_hashes,
                "checkpoints": {p.name: v4.sha(p) for p in cell.glob("*.pt")},
                "parameters_bank": sum(p.numel() for p in banks[0].parameters()),
                "parameters_heads": sum(p.numel() for p in heads.parameters()),
                "elapsed_seconds": time.monotonic() - start,
            },
        )
        v4.event(
            output,
            phase="development_complete",
            cell=cell.name,
            selected_route=selected,
            elapsed=time.monotonic() - start,
        )


def freeze(output: Path) -> None:
    contract = v4.read(output / "CONTRACT.json")
    assert contract["sources"] == sources() and contract["protocol"] == PROTOCOL
    bindings = {}
    for args in specs():
        cell = output / "development" / name(*args)
        report = v4.read(cell / "report.json")
        for file, digest in report["checkpoints"].items():
            assert v4.sha(cell / file) == digest
            bindings[str((cell / file).relative_to(output))] = digest
        bindings[str((cell / "report.json").relative_to(output))] = v4.sha(cell / "report.json")
    prior = verify_prior()
    assert prior["unchanged"], prior
    v4.write(
        output / "SCIENTIFIC_FREEZE.json",
        {
            "time": time.time(),
            "sources": sources(),
            "contract_sha256": v4.sha(output / "CONTRACT.json"),
            "bindings": bindings,
            "prior_preservation": prior,
            "protected_data_generated": False,
        },
    )


def confirm(output: Path) -> None:
    import torch

    from olympus.data.robust_fusion_v5 import generate
    from olympus.models.oracle_fusion_v4 import ExpertBank
    from olympus.models.robust_fusion_v5 import Heads

    frozen = v4.read(output / "SCIENTIFIC_FREEZE.json")
    assert frozen["sources"] == sources()
    assert frozen["contract_sha256"] == v4.sha(output / "CONTRACT.json")
    for p, digest in frozen["bindings"].items():
        assert v4.sha(output / p) == digest
    for ds, seed, coverage in specs():
        v4.resource_check()
        cell = output / "development" / name(ds, seed, coverage)
        dest = output / "protected" / cell.name
        dest.mkdir(parents=True, exist_ok=False)
        banks = []
        for fold in range(2):
            bank = ExpertBank()
            bank.load_state_dict(torch.load(cell / f"bank-{fold}.pt", weights_only=True))
            banks.append(bank.eval().requires_grad_(False))
        heads = Heads()
        heads.load_state_dict(torch.load(cell / "heads.pt", weights_only=True))
        heads.eval().requires_grad_(False)
        reports = {}
        for offset, regime in enumerate(PROTOCOL["protected_regimes"]):
            batch = generate(ds + 11 + offset, PROTOCOL["test_examples_per_regime"], regime)
            reports[regime], archive = evaluate(banks, heads, batch)
            v4.save_tensor(dest / f"{regime}.pt", archive)
            reports[regime]["data_hashes"] = v4.tensors_hash(batch)
            reports[regime]["prediction_sha256"] = v4.sha(dest / f"{regime}.pt")
        v4.write(
            dest / "report.json",
            {
                "dataset": ds,
                "training_seed": seed,
                "coverage": coverage,
                "selected_route": v4.read(cell / "report.json")["selected_route"],
                "regimes": reports,
            },
        )
        v4.event(output, phase="protected_complete", cell=cell.name)


def summarize(output: Path) -> dict[str, Any]:
    import numpy as np

    from olympus.models.robust_fusion_v5 import ROUTES, TEMPORAL

    rows = [v4.read(output / "protected" / name(*args) / "report.json") for args in specs()]
    rng = np.random.default_rng(PROTOCOL["bootstrap_seed"])

    def value(row: Any, regime: str, kind: str, mode: str) -> float:
        if kind == "routing":
            return float(
                row["regimes"][regime][kind][row["selected_route"] if mode == "selected" else mode][
                    "accuracy"
                ]
            )
        return float(row["regimes"][regime][kind][mode])

    means = {
        coverage: {
            regime: {
                kind: {
                    mode: float(
                        np.mean(
                            [
                                value(r, regime, kind, mode)
                                for r in rows
                                if r["coverage"] == coverage
                            ]
                        )
                    )
                    for mode in modes
                }
                for kind, modes in [("routing", (*ROUTES, "selected")), ("temporal", TEMPORAL)]
            }
            for regime in PROTOCOL["protected_regimes"]
        }
        for coverage in PROTOCOL["coverage"]
    }
    comparisons = {}
    for regime in PROTOCOL["protected_regimes"]:
        contrasts = [
            ("routing", "broad", "selected", "core", "semantic"),
            ("routing", "broad", "semantic", "core", "semantic"),
            ("routing", "broad", "prediction_only", "broad", "semantic"),
            ("routing", "broad", "blend", "broad", "semantic"),
            ("routing", "broad", "utility", "broad", "semantic"),
            ("temporal", "broad", "residual", "core", "v4"),
            ("temporal", "broad", "residual", "broad", "direct"),
            ("temporal", "broad", "residual", "broad", "persistence"),
            ("temporal", "broad", "direct", "core", "v4"),
            ("temporal", "broad", "residual", "core", "residual"),
        ]
        for kind, lc, lm, rc, rm in contrasts:
            deltas = []
            for ds in PROTOCOL["dataset_seeds"]:
                left = [
                    value(r, regime, kind, lm)
                    for r in rows
                    if r["dataset"] == ds and r["coverage"] == lc
                ]
                right = [
                    value(r, regime, kind, rm)
                    for r in rows
                    if r["dataset"] == ds and r["coverage"] == rc
                ]
                deltas.append(float(np.mean(left) - np.mean(right)))
            sampled = rng.choice(deltas, size=(PROTOCOL["bootstrap_samples"], len(deltas))).mean(1)
            comparisons[f"{regime}/{kind}/{lc}:{lm}-minus-{rc}:{rm}"] = {
                "mean": float(np.mean(deltas)),
                "ci95": np.quantile(sampled, [0.025, 0.975]).tolist(),
                "dataset_deltas": deltas,
                "descriptive": True,
                "clusters": 5,
            }
    routing = comparisons["novel_shift/routing/broad:selected-minus-core:semantic"]
    guard = comparisons["corrupted/routing/broad:selected-minus-core:semantic"]
    temporal = comparisons["clean/temporal/broad:residual-minus-broad:persistence"]
    routing_ok = (
        routing["mean"] >= 0.02
        and routing["ci95"][0] > 0
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
    result = {
        "protocol": PROTOCOL,
        "cells": len(rows),
        "protected_reports": len(rows) * 4,
        "means": means,
        "comparisons": comparisons,
        "routing_primary_supported": routing_ok,
        "temporal_primary_supported": temporal_ok,
        "selected_routes": {
            r["coverage"] + "/" + str(r["dataset"]) + "/" + str(r["training_seed"]): r[
                "selected_route"
            ]
            for r in rows
        },
        "freeze_sha256": v4.sha(output / "SCIENTIFIC_FREEZE.json"),
        "prior_preservation": verify_prior(),
        "promotion_authorized": False,
    }
    v4.write(output / "SUMMARY.json", result)
    lines = [
        "# Olympus v5 results",
        "",
        f"Routing primary supported: **{routing_ok}**. "
        f"Temporal primary supported: **{temporal_ok}**.",
        "",
        "All intervals are descriptive, five dataset clusters and two paired training seeds.",
        "",
        "| Coverage / route | Clean | Corrupted | Known shift | Novel shift |",
        "|---|---:|---:|---:|---:|",
    ]
    for coverage in PROTOCOL["coverage"]:
        for route in (*ROUTES, "selected"):
            lines.append(
                "| "
                + coverage
                + "/"
                + route
                + " | "
                + " | ".join(
                    f"{means[coverage][r]['routing'][route]:.4f}"
                    for r in PROTOCOL["protected_regimes"]
                )
                + " |"
            )
    lines += [
        "",
        "| Coverage / temporal | Clean | Corrupted | Known shift | Novel shift |",
        "|---|---:|---:|---:|---:|",
    ]
    for coverage in PROTOCOL["coverage"]:
        for mode in TEMPORAL:
            lines.append(
                "| "
                + coverage
                + "/"
                + mode
                + " | "
                + " | ".join(
                    f"{means[coverage][r]['temporal'][mode]:.6f}"
                    for r in PROTOCOL["protected_regimes"]
                )
                + " |"
            )
    lines += ["", "| Paired difference | Mean | 95% descriptive interval |", "|---|---:|---:|"]
    for key, v in comparisons.items():
        lines.append(f"| {key} | {v['mean']:.6f} | [{v['ci95'][0]:.6f}, {v['ci95'][1]:.6f}] |")
    lines += [
        "",
        "Oracle uses labels and cannot be deployed. Synthetic reference models; "
        "no natural-data/SOTA validation. All prior results preserved.",
    ]
    with (output / "RESULTS.md").open("x") as f:
        f.write("\n".join(lines) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("command", choices=["run", "check", "test"])
    args = parser.parse_args()
    output = args.output.resolve()
    assert ROOT in output.parents and "artifacts" in output.relative_to(ROOT).parts
    output.mkdir(parents=True, exist_ok=True)
    if args.command == "check":
        receipt = verify_prior()
        print(json.dumps(receipt))
        assert receipt["unchanged"]
        return
    from olympus.foundry.resources import ResourceGovernor

    try:
        before = v4.pageouts()
        time.sleep(2)
        after = v4.pageouts()
        if after > before:
            raise RuntimeError("pageouts increasing; workload not admitted")
        with ResourceGovernor(ROOT / "artifacts/locks/campaign-validation.lock") as governor:
            assert governor.preflight is not None
            v4.event(
                output,
                phase="admitted",
                snapshot=governor.preflight.as_dict(),
                pageouts=[before, after],
                pid=os.getpid(),
            )
            import torch

            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            torch.use_deterministic_algorithms(True)
            if args.command == "test":
                import pytest

                code = pytest.main(
                    [
                        "-q",
                        "tests/test_robust_fusion_v5.py",
                        "--override-ini=addopts=",
                        "--junitxml=" + str(output / "tests-v5.xml"),
                    ]
                )
                if code:
                    raise RuntimeError(f"qualification failed {code}")
                return
            assert verify_prior()["unchanged"]
            develop(output, time.monotonic() + PROTOCOL["max_wall_seconds"])
            freeze(output)
            confirm(output)
            summarize(output)
    except BaseException as error:
        v4.write(
            output / f"FAILURE-{time.time_ns()}.json",
            {
                "error": repr(error),
                "time": time.time(),
                "torch_imported": "torch" in sys.modules,
                "sources": sources(),
            },
        )
        raise


if __name__ == "__main__":
    main()

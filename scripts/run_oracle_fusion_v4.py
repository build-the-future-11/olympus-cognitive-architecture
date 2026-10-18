"""Immutable, resource-guarded v4 development -> freeze -> protected evaluation.

Use `python -m scripts.run_oracle_fusion_v4 --output artifacts/oracle-fusion-v4 run`.
No Torch import occurs before resource admission. No historical file is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research/olympus_irp_successor_v4"
PROTOCOL: dict[str, Any] = {
    "id": "olympus-oracle-fusion-v4.0.0",
    "scope": "controlled synthetic diagnostic using Olympus reference encoders; no SOTA claim",
    "dataset_seeds": [20264100, 20264200, 20264300, 20264400, 20264500],
    "training_seeds": [20264601, 20264602],
    "coverage": ["missing_only", "mixed"],
    "counterfactual_weights": [0.0, 0.05],
    "folds": 2,
    "train_examples": 1024,
    "development_examples": 256,
    "test_examples_per_regime": 1024,
    "expert_steps": 250,
    "router_steps": 300,
    "batch_size": 64,
    "expert_lr": 0.001,
    "router_lr": 0.001,
    "weight_decay": 0.01,
    "temporal_weight": 0.25,
    "gradient_clip": 1.0,
    "selection": "fixed steps; best single expert and strongest control selected on development",
    "primary": "mixed/cf0 utility minus development-selected strongest control; corrupted accuracy",
    "strong_controls": ["uniform", "fixed", "semantic", "best_single", "semantic_train_mean"],
    "required_effect": 0.02,
    "bootstrap": {
        "unit": "dataset mean across two paired training seeds",
        "samples": 10000,
        "seed": 20264999,
        "interval": "percentile 95%; five clusters, descriptive",
    },
    "minimum_available_bytes": 2147483648,
    "maximum_swap_fraction": 0.90,
    "max_wall_seconds": 3600,
    "protected_regimes": ["clean", "corrupted", "ood"],
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path: Path, value: Any) -> None:
    """Exclusive publication: never overwrite a result, failure, or freeze."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def sources() -> dict[str, str]:
    paths = sorted((ROOT / "olympus").rglob("*.py"))
    paths += [Path(__file__), ROOT / "tests/test_oracle_fusion_v4.py", RESEARCH / "PROTOCOL.md"]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def verify_prior() -> dict[str, Any]:
    remote = ROOT / "V4_REMOTE_PRESERVATION.json"
    if remote.exists():
        receipt = read(remote)
        if receipt["inventory_sha256"] != sha(RESEARCH / "PRIOR_INVENTORY.json"):
            raise RuntimeError("remote preservation receipt is bound to a different inventory")
        if not receipt["local_verification"]["unchanged"]:
            raise RuntimeError("pre-upload local preservation did not pass")
        return {
            "unchanged": True,
            "scope": "local pre-upload verification only; reverify original host after retrieval",
            "remote_receipt_sha256": sha(remote),
            "local_verification": receipt["local_verification"],
        }
    prior = read(RESEARCH / "PRIOR_INVENTORY.json")["files"]
    changed = [
        p
        for p, record in prior.items()
        if not (ROOT / p).is_file() or sha(ROOT / p) != record["sha256"]
    ]
    return {"files_checked": len(prior), "changed_or_missing": changed, "unchanged": not changed}


def pageouts() -> int:
    if platform.system() != "Darwin":
        return 0
    value = subprocess.check_output(["vm_stat"], text=True)
    match = re.search(r"Pageouts:\s+(\d+)", value)
    if match is None:
        raise RuntimeError("cannot inspect pageout guard")
    return int(match.group(1))


def resource_check() -> dict[str, Any]:
    from olympus.foundry.resources import memory_snapshot

    snapshot = memory_snapshot()
    if snapshot.available_bytes < PROTOCOL["minimum_available_bytes"]:
        raise RuntimeError("2 GiB resource floor rejected")
    if snapshot.swap_fraction > PROTOCOL["maximum_swap_fraction"]:
        raise RuntimeError("90 percent swap ceiling rejected")
    return snapshot.as_dict()


def event(output: Path, **values: Any) -> None:
    record = {"time": time.time(), **values}
    with (output / "events.jsonl").open("a") as f:
        f.write(json.dumps(record, allow_nan=False) + "\n")
        f.flush()
    print(json.dumps(record), flush=True)


def tensors_hash(batch: Any) -> dict[str, str]:
    return {
        key: hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()
        for key, value in batch.items()
    }


def save_tensor(path: Path, value: Any) -> None:
    import torch

    with path.open("xb") as f:
        torch.save(value, f)


def predict(bank: Any, batch: Any) -> tuple[Any, Any, Any]:
    import torch

    from olympus.data.oracle_fusion_v4 import subset

    bank.eval()
    result: list[list[Any]] = [[], [], []]
    with torch.no_grad():
        for start in range(0, len(batch["tokens"]), 128):
            values = bank(
                subset(batch, torch.arange(start, min(start + 128, len(batch["tokens"]))))
            )
            for target, value in zip(result, values, strict=True):
                target.append(value)
    return torch.cat(result[0]), torch.cat(result[1]), torch.cat(result[2])


def fit_bank(train: Any, seed: int, cf: float, output: Path, deadline: float) -> Any:
    import torch

    from olympus.data.oracle_fusion_v4 import subset
    from olympus.models.oracle_fusion_v4 import ExpertBank, expert_loss

    torch.manual_seed(seed)
    bank = ExpertBank()
    optimizer = torch.optim.AdamW(
        bank.parameters(), lr=PROTOCOL["expert_lr"], weight_decay=PROTOCOL["weight_decay"]
    )
    sampler = torch.Generator().manual_seed(seed + 10000)
    bank.train()
    for step in range(PROTOCOL["expert_steps"]):
        ids = torch.randint(len(train["tokens"]), (PROTOCOL["batch_size"],), generator=sampler)
        batch = subset(train, ids)
        _, logits, temporal = bank(batch)
        loss = expert_loss(logits, batch["decision_targets"], cf)
        loss = (
            loss
            + PROTOCOL["temporal_weight"] * (temporal - batch["next_state_targets"]).square().mean()
        )
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite expert loss")
        optimizer.zero_grad(set_to_none=True)
        torch.autograd.backward(loss)
        torch.nn.utils.clip_grad_norm_(bank.parameters(), PROTOCOL["gradient_clip"])
        optimizer.step()
        if (step + 1) % 50 == 0:
            resource_check()
            if time.monotonic() > deadline:
                raise RuntimeError("fixed wall budget exhausted")
            event(
                output, phase="expert", step=step + 1, seed=seed, cf=cf, loss=float(loss.detach())
            )
    bank.eval()
    bank.requires_grad_(False)
    return bank


def fit_routers(features: Any, logits: Any, quality: Any, labels: Any, seed: int) -> Any:
    import torch
    from torch import nn

    from olympus.models.oracle_fusion_v4 import Routers, probabilities

    torch.manual_seed(seed + 70000)
    router = Routers()
    optimizer = torch.optim.AdamW(
        router.parameters(), lr=PROTOCOL["router_lr"], weight_decay=PROTOCOL["weight_decay"]
    )
    sampler = torch.Generator().manual_seed(seed + 80000)
    utility = logits.argmax(-1).eq(labels[:, None]).float()
    for _ in range(PROTOCOL["router_steps"]):
        ix = torch.randint(len(labels), (PROTOCOL["batch_size"],), generator=sampler)
        scores = router(features[ix])
        loss = sum(
            nn.functional.nll_loss(
                probabilities(logits[ix], scores, mode)[0].clamp_min(1e-8).log(), labels[ix]
            )
            for mode in ("fixed", "semantic")
        )
        loss = loss + nn.functional.binary_cross_entropy_with_logits(scores["utility"], utility[ix])
        loss = loss + nn.functional.binary_cross_entropy_with_logits(
            scores["corruption"], quality[ix]
        )
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite router loss")
        optimizer.zero_grad(set_to_none=True)
        torch.autograd.backward(loss)
        torch.nn.utils.clip_grad_norm_(router.parameters(), PROTOCOL["gradient_clip"])
        optimizer.step()
    router.eval()
    with torch.no_grad():
        router.training_mean_gates.copy_(router(features)["semantic"].softmax(-1).mean(0))
    router.requires_grad_(False)
    return router


def metrics(p: Any, y: Any) -> dict[str, float]:
    import torch

    confidence, prediction = p.max(-1)
    correct = prediction.eq(y).float()
    chosen = torch.argsort(confidence, descending=True, stable=True)[: len(y) // 2]
    ece = 0.0
    for low in range(10):
        mask = (confidence > low / 10) & (confidence <= (low + 1) / 10)
        if mask.any():
            ece += float(
                mask.float().mean() * (correct[mask].mean() - confidence[mask].mean()).abs()
            )
    return {
        "accuracy": float(correct.mean()),
        "nll": float(-p[torch.arange(len(y)), y].clamp_min(1e-8).log().mean()),
        "selective_accuracy_50": float(correct[chosen].mean()),
        "brier": float((p - torch.nn.functional.one_hot(y, 4)).square().sum(-1).mean()),
        "ece": ece,
    }


def evaluate(banks: Any, router: Any, batch: Any, best_single: int) -> tuple[Any, Any]:
    import torch

    from olympus.models.oracle_fusion_v4 import MODES, probabilities, router_features

    combined: dict[str, list[Any]] = {mode: [] for mode in MODES}
    gates: dict[str, list[Any]] = {mode: [] for mode in MODES}
    expert_predictions, temporal_predictions, utility_predictions = [], [], []
    for bank in banks:
        z, logits, temporal = predict(bank, batch)
        with torch.no_grad():
            scores = router(router_features(z, logits))
            for mode in MODES:
                p, g = probabilities(
                    logits,
                    scores,
                    mode,
                    quality=batch["component_quality"] if mode == "oracle_corruption" else None,
                    labels=batch["decision_targets"] if mode == "oracle_task" else None,
                    best_single=best_single,
                )
                combined[mode].append(p)
                gates[mode].append(g)
            utility_predictions.append(scores["utility"].sigmoid())
        expert_predictions.append(logits.softmax(-1))
        temporal_predictions.append(temporal)
    predictions = {mode: torch.stack(values).mean(0) for mode, values in combined.items()}
    y = batch["decision_targets"]
    report: dict[str, Any] = {mode: metrics(p, y) for mode, p in predictions.items()}
    experts = torch.stack(expert_predictions).mean(0)
    report["expert_accuracy"] = [
        float(experts[:, i].argmax(-1).eq(y).float().mean()) for i in range(5)
    ]
    report["mean_gates"] = {
        mode: torch.stack(values).mean((0, 1)).tolist() for mode, values in gates.items()
    }
    actual = torch.stack([p.argmax(-1).eq(y[:, None]).float() for p in expert_predictions])
    report["utility_brier"] = float((torch.stack(utility_predictions) - actual).square().mean())
    temporal = torch.stack(temporal_predictions).mean(0)
    target = batch["next_state_targets"]
    report["temporal_mse"] = float((temporal - target).square().mean())
    report["persistence_mse"] = float((batch["states"] - target).square().mean())
    legacy = batch["states"] + 0.1 * torch.nn.functional.pad(batch["actions"], (0, 2))
    report["legacy_formula_mse"] = float((legacy - target).square().mean())
    # The generator's clean-state dynamics is privileged and is not a learned comparator.
    archive = {
        "probabilities": predictions,
        "labels": y,
        "expert_probabilities": experts,
        "temporal_prediction": temporal,
        "temporal_target": target,
        "source_ids": batch["source_ids"],
    }
    return report, archive


def cell_name(dataset: int, seed: int, coverage: str, cf: float) -> str:
    return f"d{dataset}-s{seed}-{coverage}-cf{cf:g}"


def cell_specs() -> Any:
    for ds in PROTOCOL["dataset_seeds"]:
        for seed in PROTOCOL["training_seeds"]:
            for coverage in PROTOCOL["coverage"]:
                for cf in PROTOCOL["counterfactual_weights"]:
                    yield ds, seed, coverage, cf


def develop(output: Path, deadline: float) -> None:
    import torch

    from olympus.data.oracle_fusion_v4 import generate, subset
    from olympus.models.oracle_fusion_v4 import router_features

    contract_path = output / "CONTRACT.json"
    contract = {
        "protocol": PROTOCOL,
        "sources": sources(),
        "torch": torch.__version__,
        "python": sys.version,
        "platform": platform.platform(),
        "threads": 1,
    }
    if contract_path.exists():
        if read(contract_path) != contract:
            raise RuntimeError("contract changed; use a new version, never overwrite")
    else:
        write(contract_path, contract)
    for ds, seed, coverage, cf in cell_specs():
        name = cell_name(ds, seed, coverage, cf)
        cell = output / "development" / name
        if (cell / "report.json").exists():
            report = read(cell / "report.json")
            if any(sha(cell / p) != h for p, h in report["checkpoints"].items()):
                raise RuntimeError("completed checkpoint changed")
            continue
        cell.mkdir(parents=True, exist_ok=False)
        resource_check()
        if time.monotonic() > deadline:
            raise RuntimeError("fixed wall budget exhausted")
        start = time.monotonic()
        train = generate(ds + 1, PROTOCOL["train_examples"], coverage)
        dev = generate(ds + 2, PROTOCOL["development_examples"], "mixed")
        banks, features, logits_oof, qualities, labels, fold_audit = [], [], [], [], [], []
        for fold in range(2):
            fit_ids = torch.arange(fold, PROTOCOL["train_examples"], 2)
            held_ids = torch.arange(1 - fold, PROTOCOL["train_examples"], 2)
            fit, held = subset(train, fit_ids), subset(train, held_ids)
            bank = fit_bank(fit, seed + fold * 100, cf, output, deadline)
            z, logits, _ = predict(bank, held)
            banks.append(bank)
            features.append(router_features(z, logits))
            logits_oof.append(logits)
            qualities.append(held["component_quality"])
            labels.append(held["decision_targets"])
            fold_audit.append(
                {
                    "fit_ids": fit["source_ids"].tolist(),
                    "router_target_ids": held["source_ids"].tolist(),
                    "overlap": len(
                        set(fit["source_ids"].tolist()) & set(held["source_ids"].tolist())
                    ),
                }
            )
            save_tensor(cell / f"bank-{fold}.pt", bank.state_dict())
        router = fit_routers(
            torch.cat(features),
            torch.cat(logits_oof),
            torch.cat(qualities),
            torch.cat(labels),
            seed,
        )
        save_tensor(cell / "routers.pt", router.state_dict())
        # Select expert using development labels only, before test generation.
        dev_probs = torch.stack([predict(bank, dev)[1].softmax(-1) for bank in banks]).mean(0)
        acc = dev_probs.argmax(-1).eq(dev["decision_targets"][:, None]).float().mean(0)
        best = int(acc.argmax())
        diagnostics, _ = evaluate(banks, router, dev, best)
        strongest = max(PROTOCOL["strong_controls"], key=lambda m: diagnostics[m]["accuracy"])
        report = {
            "dataset": ds,
            "training_seed": seed,
            "coverage": coverage,
            "cf": cf,
            "best_single": best,
            "selected_control": strongest,
            "development": diagnostics,
            "folds": fold_audit,
            "data_hashes": {"train": tensors_hash(train), "development": tensors_hash(dev)},
            "checkpoints": {p.name: sha(p) for p in cell.glob("*.pt")},
            "parameters_bank": sum(p.numel() for p in banks[0].parameters()),
            "parameters_routers": sum(p.numel() for p in router.parameters()),
            "elapsed_seconds": time.monotonic() - start,
            "expert_updates": 2 * PROTOCOL["expert_steps"],
            "router_updates": PROTOCOL["router_steps"],
        }
        write(cell / "report.json", report)
        event(
            output,
            phase="development_complete",
            cell=name,
            seconds=report["elapsed_seconds"],
            selected_control=strongest,
        )


def freeze(output: Path) -> None:
    contract = read(output / "CONTRACT.json")
    if contract["sources"] != sources() or contract["protocol"] != PROTOCOL:
        raise RuntimeError("source or protocol drift")
    bindings = {}
    for args in cell_specs():
        cell = output / "development" / cell_name(*args)
        report = read(cell / "report.json")
        for p, digest in report["checkpoints"].items():
            if sha(cell / p) != digest:
                raise RuntimeError("checkpoint drift")
            bindings[str((cell / p).relative_to(output))] = digest
        bindings[str((cell / "report.json").relative_to(output))] = sha(cell / "report.json")
    preservation = verify_prior()
    if not preservation["unchanged"]:
        raise RuntimeError(f"prior files changed: {preservation['changed_or_missing']}")
    write(
        output / "SCIENTIFIC_FREEZE.json",
        {
            "time": time.time(),
            "sources": sources(),
            "contract_sha256": sha(output / "CONTRACT.json"),
            "bindings": bindings,
            "prior_preservation": preservation,
            "protected_data_generated": False,
        },
    )


def confirm(output: Path) -> None:
    import torch

    from olympus.data.oracle_fusion_v4 import generate
    from olympus.models.oracle_fusion_v4 import ExpertBank, Routers

    frozen = read(output / "SCIENTIFIC_FREEZE.json")
    if frozen["sources"] != sources() or sha(output / "CONTRACT.json") != frozen["contract_sha256"]:
        raise RuntimeError("freeze source/contract mismatch")
    for p, h in frozen["bindings"].items():
        if sha(output / p) != h:
            raise RuntimeError(f"freeze artifact mismatch: {p}")
    for args in cell_specs():
        resource_check()
        name = cell_name(*args)
        cell = output / "development" / name
        destination = output / "protected" / name
        if destination.exists():
            raise FileExistsError("protected results cannot be overwritten")
        destination.mkdir(parents=True)
        development = read(cell / "report.json")
        banks = []
        for fold in range(2):
            bank = ExpertBank()
            bank.load_state_dict(torch.load(cell / f"bank-{fold}.pt", weights_only=True))
            bank.requires_grad_(False)
            banks.append(bank)
        router = Routers()
        router.load_state_dict(torch.load(cell / "routers.pt", weights_only=True))
        router.eval().requires_grad_(False)
        reports = {}
        for offset, regime in enumerate(PROTOCOL["protected_regimes"]):
            batch = generate(args[0] + 11 + offset, PROTOCOL["test_examples_per_regime"], regime)
            report, predictions = evaluate(banks, router, batch, development["best_single"])
            report["data_hashes"] = tensors_hash(batch)
            # Domain-only lookup and constant-mask majority use training labels only.
            train = generate(args[0] + 1, PROTOCOL["train_examples"], args[2])
            domain_map = torch.tensor(
                [
                    int(
                        torch.bincount(
                            train["decision_targets"][train["domain_ids"] == d], minlength=4
                        ).argmax()
                    )
                    for d in range(3)
                ]
            )
            report["domain_only_accuracy"] = float(
                domain_map[batch["domain_ids"]].eq(batch["decision_targets"]).float().mean()
            )
            majority = int(torch.bincount(train["decision_targets"], minlength=4).argmax())
            report["mask_only_accuracy"] = float(
                batch["decision_targets"].eq(majority).float().mean()
            )
            save_tensor(destination / f"{regime}-predictions.pt", predictions)
            report["prediction_sha256"] = sha(destination / f"{regime}-predictions.pt")
            reports[regime] = report
        write(
            destination / "report.json",
            {
                "cell": name,
                "dataset": args[0],
                "training_seed": args[1],
                "coverage": args[2],
                "cf": args[3],
                "selected_control": development["selected_control"],
                "regimes": reports,
            },
        )
        event(output, phase="protected_complete", cell=name)


def summarize(output: Path) -> None:
    import numpy as np

    from olympus.models.oracle_fusion_v4 import MODES

    rows = [read(output / "protected" / cell_name(*args) / "report.json") for args in cell_specs()]
    rng = np.random.default_rng(PROTOCOL["bootstrap"]["seed"])

    def interval(values: list[float]) -> dict[str, Any]:
        array = np.asarray(values)
        sampled = rng.choice(
            array, size=(PROTOCOL["bootstrap"]["samples"], len(array)), replace=True
        ).mean(1)
        return {
            "mean": float(array.mean()),
            "ci95": np.quantile(sampled, [0.025, 0.975]).tolist(),
            "dataset_deltas": values,
            "clusters": len(values),
            "descriptive": True,
        }

    means = {}
    for coverage in PROTOCOL["coverage"]:
        for cf in PROTOCOL["counterfactual_weights"]:
            selected = [r for r in rows if r["coverage"] == coverage and r["cf"] == cf]
            for regime in PROTOCOL["protected_regimes"]:
                means[f"{coverage}/cf{cf:g}/{regime}"] = {
                    mode: {
                        metric: float(
                            np.mean([r["regimes"][regime][mode][metric] for r in selected])
                        )
                        for metric in ("accuracy", "nll", "selective_accuracy_50", "brier", "ece")
                    }
                    for mode in MODES
                }
                means[f"{coverage}/cf{cf:g}/{regime}"]["diagnostics"] = {
                    metric: float(np.mean([r["regimes"][regime][metric] for r in selected]))
                    for metric in (
                        "temporal_mse",
                        "persistence_mse",
                        "legacy_formula_mse",
                        "domain_only_accuracy",
                        "mask_only_accuracy",
                        "utility_brier",
                    )
                }
    comparisons = {}
    main = [r for r in rows if r["coverage"] == "mixed" and r["cf"] == 0]
    for regime in PROTOCOL["protected_regimes"]:
        for left, right in (
            ("utility", "selected_control"),
            ("utility", "semantic"),
            ("utility", "corruption"),
            ("oracle_corruption", "uniform"),
            ("oracle_task", "uniform"),
            ("semantic", "fixed"),
            ("semantic", "semantic_train_mean"),
        ):
            deltas = []
            for ds in PROTOCOL["dataset_seeds"]:
                matched = [r for r in main if r["dataset"] == ds]
                deltas.append(
                    float(
                        np.mean(
                            [
                                r["regimes"][regime][left]["accuracy"]
                                - r["regimes"][regime][
                                    r["selected_control"] if right == "selected_control" else right
                                ]["accuracy"]
                                for r in matched
                            ]
                        )
                    )
                )
            comparisons[f"{regime}/{left}-minus-{right}"] = interval(deltas)
        for mode in (
            "uniform",
            "semantic",
            "utility",
            "corruption",
            "oracle_corruption",
            "oracle_task",
        ):
            for ablation in ("coverage", "counterfactual"):
                deltas = []
                for ds in PROTOCOL["dataset_seeds"]:
                    paired = []
                    for seed in PROTOCOL["training_seeds"]:

                        def lookup(
                            coverage: str,
                            cf: float,
                            ds: int = ds,
                            seed: int = seed,
                            regime: str = regime,
                            mode: str = mode,
                        ) -> float:
                            row = next(
                                r
                                for r in rows
                                if r["dataset"] == ds
                                and r["training_seed"] == seed
                                and r["coverage"] == coverage
                                and r["cf"] == cf
                            )
                            return float(row["regimes"][regime][mode]["accuracy"])

                        paired.append(
                            lookup("mixed", 0)
                            - (
                                lookup("missing_only", 0)
                                if ablation == "coverage"
                                else lookup("mixed", 0.05)
                            )
                        )
                    deltas.append(float(np.mean(paired)))
                comparisons[f"{regime}/{mode}/{ablation}"] = interval(deltas)
    primary = comparisons["corrupted/utility-minus-selected_control"]
    supported = primary["mean"] >= PROTOCOL["required_effect"] and primary["ci95"][0] > 0
    result = {
        "protocol": PROTOCOL["id"],
        "cells": len(rows),
        "protected_reports": len(rows) * 3,
        "primary": primary,
        "primary_disposition": "SUPPORTED_SYNTHETIC_ONLY" if supported else "NOT_SUPPORTED",
        "means": means,
        "comparisons": comparisons,
        "prior_preservation": verify_prior(),
        "scope": PROTOCOL["scope"],
        "promotion_authorized": False,
        "freeze_sha256": sha(output / "SCIENTIFIC_FREEZE.json"),
    }
    write(output / "SUMMARY.json", result)
    lines = [
        "# Olympus v4 oracle-routing results",
        "",
        f"Primary: **{result['primary_disposition']}**.",
        "",
        "All intervals are descriptive, clustered across five independently generated datasets; "
        "two training seeds per dataset.",
        "",
        "## Mixed training, no counterfactual penalty",
        "",
        "| Route | Clean accuracy | Corrupted accuracy | OOD accuracy |",
        "|---|---:|---:|---:|",
    ]
    for mode in MODES:
        values = [
            means[f"mixed/cf0/{regime}"][mode]["accuracy"]
            for regime in PROTOCOL["protected_regimes"]
        ]
        lines.append(f"| {mode} | " + " | ".join(f"{v:.4f}" for v in values) + " |")
    lines += [
        "",
        "## Paired corrupted accuracy differences",
        "",
        "| Contrast | Delta | Descriptive 95% CI |",
        "|---|---:|---|",
    ]
    for key, value in comparisons.items():
        if key.startswith("corrupted/"):
            lines.append(
                f"| {key} | {value['mean']:.4f} | "
                f"[{value['ci95'][0]:.4f}, {value['ci95'][1]:.4f}] |"
            )
    lines += [
        "",
        "Oracle routes are privileged diagnostics. Neither is deployable.",
        "Matched routing shares expert banks and executes all learned heads. "
        "Training time is not equated between privileged oracles and learned heads.",
        "The counterfactual ablation tests the analogous penalty in v4 probability fusion; "
        "it is not a causal reanalysis of v3 latent fusion.",
        "No public benchmark, natural-data validation, independent replication, "
        "or SOTA result is established.",
        "All old results remain separate; "
        "this successor changes data, objective and fusion representation.",
        "",
    ]
    with (output / "RESULTS.md").open("x") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "command", choices=("run", "develop", "freeze", "confirm", "summarize", "check")
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if (
        output == ROOT
        or ROOT not in output.parents
        or "artifacts" not in output.relative_to(ROOT).parts
    ):
        raise ValueError("output must be a dedicated artifact directory in this checkout")
    output.mkdir(parents=True, exist_ok=True)
    if args.command == "check":
        result = verify_prior()
        print(json.dumps(result))
        if not result["unchanged"]:
            raise RuntimeError("historical file preservation failed")
        return
    from olympus.foundry.resources import ResourceGovernor

    try:
        before = pageouts()
        time.sleep(2)
        after = pageouts()
        if after > before:
            raise RuntimeError("pageouts increasing; workload not admitted")
        with ResourceGovernor(ROOT / "artifacts/locks/campaign-validation.lock") as governor:
            assert governor.preflight is not None
            event(
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
            deadline = time.monotonic() + PROTOCOL["max_wall_seconds"]
            if args.command in {"run", "develop"}:
                develop(output, deadline)
            if args.command in {"run", "freeze"}:
                freeze(output)
            if args.command in {"run", "confirm"}:
                confirm(output)
            if args.command in {"run", "summarize"}:
                summarize(output)
    except BaseException as error:
        write(
            output / f"FAILURE-{time.time_ns()}.json",
            {
                "error": repr(error),
                "command": args.command,
                "torch_imported": "torch" in sys.modules,
                "time": time.time(),
                "source": sources(),
            },
        )
        raise


if __name__ == "__main__":
    main()

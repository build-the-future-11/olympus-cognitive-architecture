"""Bounded disjoint-stacked Olympus campaign; no Torch before resource admission.

Commands: qualify (artificial tests), run (one development/freeze/confirmation),
verify (hashes plus raw prediction arithmetic). Output directories are exclusive.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from scripts import run_oracle_fusion_v4 as v4

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research/olympus_routing_v6"
PROTOCOL = json.loads((RESEARCH / "protocol.json").read_text())
REGIMES = PROTOCOL["development_regimes"]
VARIANTS = PROTOCOL["variants"]


def sources() -> dict[str, str]:
    paths = list((ROOT / "olympus").rglob("*.py")) + [
        Path(__file__),
        ROOT / "scripts/run_oracle_fusion_v4.py",
        ROOT / "scripts/qualify_flagship_v2.py",
        RESEARCH / "protocol.json",
        ROOT / "tests/test_conservative_routing_v6.py",
    ]
    return {str(p.relative_to(ROOT)): v4.sha(p) for p in sorted(paths)}


def check_deadline(deadline: float) -> None:
    if time.monotonic() > deadline:
        raise TimeoutError("frozen campaign wall budget exhausted")
    v4.resource_check()
    if platform.system() == "Darwin":
        from scripts.qualify_flagship_v2 import local_tensor_admission

        local_tensor_admission()


def ensure_disjoint(parts: dict[str, list[int]]) -> None:
    seen: set[int] = set()
    for name, ids in parts.items():
        if len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError(f"source identity overlap in {name}")
        seen.update(ids)


def ensemble(banks: list[Any], batch: Any) -> tuple[Any, Any]:
    import torch

    from olympus.models.conservative_routing_v6 import features

    values = [v4.predict(bank, batch) for bank in banks]
    z = torch.stack([v[0][:, :5] for v in values]).mean(0)
    p = torch.stack([v[1].softmax(-1) for v in values]).mean(0)
    return features(z, p), p


def meta_split(banks: list[Any], ds: int, stage: str) -> dict[str, Any]:
    import torch

    from olympus.data.conservative_routing_v6 import generate

    acc: dict[str, list[Any]] = {k: [] for k in ("x", "p", "y", "groups", "ids")}
    hashes = {}
    for i, regime in enumerate(REGIMES):
        batch = generate(
            ds + PROTOCOL["split_offsets"][stage] + i, PROTOCOL["meta_examples_per_regime"], regime
        )
        x, p = ensemble(banks, batch)
        values = (x, p, batch["decision_targets"], torch.full((len(x),), i), batch["source_ids"])
        for key, value in zip(acc, values, strict=True):
            acc[key].append(value)
        hashes[regime] = v4.tensors_hash(batch)
    return {**{k: torch.cat(v) for k, v in acc.items()}, "hashes": hashes}


def indices(data: dict[str, Any], sampler: Any) -> Any:
    import torch

    selected = []
    for group in range(len(REGIMES)):
        eligible = torch.where(data["groups"] == group)[0]
        selected.append(
            eligible[
                torch.randint(len(eligible), (PROTOCOL["batch_per_group"],), generator=sampler)
            ]
        )
    return torch.cat(selected)


def fit_bank(train: Any, seed: int, output: Path, deadline: float) -> Any:
    import torch

    from olympus.data.oracle_fusion_v4 import subset
    from olympus.models.oracle_fusion_v4 import ExpertBank, expert_loss

    torch.manual_seed(seed)
    model = ExpertBank()
    opt = torch.optim.AdamW(
        model.parameters(), lr=PROTOCOL["lr"], weight_decay=PROTOCOL["weight_decay"]
    )
    sampler = torch.Generator().manual_seed(seed + 10000)
    model.train()
    for step in range(PROTOCOL["bank_steps"]):
        ids = torch.randint(len(train["tokens"]), (PROTOCOL["bank_batch_size"],), generator=sampler)
        batch = subset(train, ids)
        _, logits, temporal = model(batch)
        loss = expert_loss(logits, batch["decision_targets"], 0.0)
        loss += (
            PROTOCOL["bank_temporal_weight"]
            * (temporal - batch["next_state_targets"]).square().mean()
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite bank loss")
        opt.zero_grad(set_to_none=True)
        torch.autograd.backward(loss)
        # Unused reference-system heads have no gradient by design; every
        # admitted gradient and all stored parameters must remain finite.
        if any(
            p.grad is not None and not bool(torch.isfinite(p.grad).all())
            for p in model.parameters()
        ):
            raise FloatingPointError("nonfinite bank gradient")
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), PROTOCOL["gradient_clip"], error_if_nonfinite=True
        )
        opt.step()
        if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise FloatingPointError("nonfinite bank parameters")
        if step % 50 == 0:
            check_deadline(deadline)
            v4.event(output, phase="bank_step", step=step, seed=seed)
    return model.eval().requires_grad_(False)


def fit_reference(data: dict[str, Any], seed: int, deadline: float) -> Any:
    import torch

    from olympus.models.conservative_routing_v6 import ReferenceRoutes

    torch.manual_seed(seed)
    model = ReferenceRoutes()
    opt = torch.optim.AdamW(
        model.parameters(), lr=PROTOCOL["lr"], weight_decay=PROTOCOL["weight_decay"]
    )
    sampler = torch.Generator().manual_seed(seed + 1)
    for step in range(PROTOCOL["reference_steps"]):
        ix = indices(data, sampler)
        pred = model(data["x"][ix], data["p"][ix])
        loss = sum(
            torch.nn.functional.nll_loss(pred[k].clamp_min(1e-8).log(), data["y"][ix])
            for k in ("fixed", "semantic")
        )
        update(model, opt, loss)
        if step % 50 == 0:
            check_deadline(deadline)
    return model.eval().requires_grad_(False)


def update(model: Any, optimizer: Any, loss: Any) -> None:
    import torch

    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("nonfinite training objective")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    for param in model.parameters():
        if param.requires_grad and (
            param.grad is None or not bool(torch.isfinite(param.grad).all())
        ):
            raise FloatingPointError("missing/nonfinite trainable gradient")
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), PROTOCOL["gradient_clip"], error_if_nonfinite=True
    )
    optimizer.step()
    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise FloatingPointError("nonfinite parameter after update")


def select(reports: dict[str, Any], modes: list[str]) -> str:
    def key(mode: str) -> tuple[float, float]:
        values = [reports[r][mode]["accuracy"] for r in REGIMES]
        return min(values), sum(values) / len(values)

    return str(max(modes, key=key))


def fit_corrections(
    data: dict[str, Any], reference: Any, ref_mode: str, seed: int, deadline: float
) -> dict[str, Any]:
    import torch

    from olympus.models.conservative_routing_v6 import Correction, correction_loss

    reference_p = reference(data["x"], data["p"])[ref_mode].detach()
    result = {}
    for name in VARIANTS:
        torch.manual_seed(seed + 2)
        model = Correction(reference.semantic, name.split("_")[0], PROTOCOL["alpha_initial"])
        opt = torch.optim.AdamW(
            model.parameters(), lr=PROTOCOL["lr"], weight_decay=PROTOCOL["weight_decay"]
        )
        sampler = torch.Generator().manual_seed(seed + 3)
        penalty = (
            PROTOCOL["anchor_penalty"]
            if name.startswith(("anchored", "constant")) and not name.endswith("no_penalty")
            else 0.0
        )
        for step in range(PROTOCOL["correction_steps"]):
            ix = indices(data, sampler)
            pred, alpha = model(data["x"][ix], data["p"][ix], reference_p[ix])
            loss = correction_loss(
                pred,
                data["y"][ix],
                data["groups"][ix],
                alpha,
                robust="robust" in name,
                penalty=penalty,
            )
            update(model, opt, loss)
            if step % 50 == 0:
                check_deadline(deadline)
        result[name] = model.eval().requires_grad_(False)
    return result


def predictions(
    reference: Any, corrections: dict[str, Any], ref_mode: str, x: Any, p: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = reference(x, p)
    result["reference"] = result[ref_mode]
    alphas = {}
    for name, model in corrections.items():
        result[name], alphas[name] = model(x, p, result["reference"])
    return result, alphas


def describe(pred: dict[str, Any], alpha: dict[str, Any], y: Any) -> dict[str, Any]:
    from olympus.models.conservative_routing_v6 import intervention_diagnostics

    out: dict[str, Any] = {k: v4.metrics(p, y) for k, p in pred.items()}
    for k in pred:
        out[k]["intervention"] = intervention_diagnostics(pred[k], pred["reference"], y)
        if k in alpha:
            out[k]["mean_alpha"] = float(alpha[k].mean())
    return out


def describe_meta(
    reference: Any, corrections: dict[str, Any], ref_mode: str, data: Any
) -> dict[str, Any]:
    out = {}
    for group, regime in enumerate(REGIMES):
        ix = data["groups"] == group
        pred, alpha = predictions(reference, corrections, ref_mode, data["x"][ix], data["p"][ix])
        out[regime] = describe(pred, alpha, data["y"][ix])
    return out


def develop(output: Path, deadline: float) -> None:
    import torch

    from olympus.data.conservative_routing_v6 import generate
    from olympus.data.oracle_fusion_v4 import subset

    for ds in PROTOCOL["dataset_seeds"]:
        for seed in PROTOCOL["training_seeds"]:
            cell = output / "development" / f"d{ds}-s{seed}"
            cell.mkdir(parents=True, exist_ok=False)
            started = time.monotonic()
            train = generate(
                ds + PROTOCOL["split_offsets"]["bank"], PROTOCOL["bank_examples"], "broad"
            )
            banks = [
                fit_bank(
                    subset(train, torch.arange(i, len(train["tokens"]), 2)),
                    seed + i * 100,
                    output,
                    deadline,
                )
                for i in range(2)
            ]
            data = {
                stage: meta_split(banks, ds, stage)
                for stage in ("reference_fit", "reference_select", "correction_fit", "selection")
            }
            ids = {
                "bank": train["source_ids"].tolist(),
                **{k: v["ids"].tolist() for k, v in data.items()},
            }
            ensure_disjoint(ids)
            reference = fit_reference(data["reference_fit"], seed, deadline)
            ref_reports = describe_meta(reference, {}, "semantic", data["reference_select"])
            ref_mode = select(ref_reports, PROTOCOL["reference_candidates"])
            v4.write(
                cell / "REFERENCE_FREEZE.json",
                {"mode": ref_mode, "reports": ref_reports, "ids": ids, "time": time.time()},
            )
            corrections = fit_corrections(
                data["correction_fit"], reference, ref_mode, seed, deadline
            )
            report = describe_meta(reference, corrections, ref_mode, data["selection"])
            selected_control = select(report, PROTOCOL["control_candidates"])
            checkpoint = {
                "banks": [b.state_dict() for b in banks],
                "reference": reference.state_dict(),
                "corrections": {k: m.state_dict() for k, m in corrections.items()},
                "ref_mode": ref_mode,
            }
            v4.save_tensor(cell / "checkpoint.pt", checkpoint)
            # Fresh reconstructed modules must produce exactly the same final predictions.
            restored = load_models(cell / "checkpoint.pt")
            replay_data = meta_split(restored[0], ds, "selection")
            if not torch.equal(replay_data["x"], data["selection"]["x"]) or not torch.equal(
                replay_data["p"], data["selection"]["p"]
            ):
                raise RuntimeError("checkpoint bank replay mismatch")
            original, original_alpha = predictions(
                reference, corrections, ref_mode, data["selection"]["x"], data["selection"]["p"]
            )
            replay, replay_alpha = predictions(
                restored[1],
                restored[2],
                restored[3],
                replay_data["x"],
                replay_data["p"],
            )
            if any(not torch.equal(original[k], replay[k]) for k in original) or any(
                not torch.equal(original_alpha[k], replay_alpha[k]) for k in original_alpha
            ):
                raise RuntimeError("checkpoint output replay mismatch")
            v4.write(
                cell / "report.json",
                {
                    "dataset": ds,
                    "training_seed": seed,
                    "reference_mode": ref_mode,
                    "selected_control": selected_control,
                    "selection": report,
                    "source_ids": ids,
                    "split_hashes": {
                        "bank": v4.tensors_hash(train),
                        **{k: v["hashes"] for k, v in data.items()},
                    },
                    "checkpoint_sha256": v4.sha(cell / "checkpoint.pt"),
                    "checkpoint_replay": "EXACT",
                    "elapsed_seconds": time.monotonic() - started,
                    "parameters": {
                        k: sum(p.numel() for p in m.parameters()) for k, m in corrections.items()
                    },
                },
            )
            v4.event(
                output,
                phase="development_cell_complete",
                dataset=ds,
                seed=seed,
                control=selected_control,
            )
            check_deadline(deadline)


def load_models(path: Path) -> tuple[Any, Any, Any, str]:
    import torch

    from olympus.models.conservative_routing_v6 import Correction, ReferenceRoutes
    from olympus.models.oracle_fusion_v4 import ExpertBank

    state = torch.load(path, map_location="cpu", weights_only=True)
    banks = [ExpertBank() for _ in range(2)]
    for b, s in zip(banks, state["banks"], strict=True):
        b.load_state_dict(s)
        b.eval().requires_grad_(False)
    ref = ReferenceRoutes()
    ref.load_state_dict(state["reference"])
    ref.eval().requires_grad_(False)
    corrections = {}
    for name in VARIANTS:
        m = Correction(ref.semantic, name.split("_")[0], PROTOCOL["alpha_initial"])
        m.load_state_dict(state["corrections"][name])
        m.eval().requires_grad_(False)
        corrections[name] = m
    return banks, ref, corrections, state["ref_mode"]


def development_manifest(output: Path) -> dict[str, str]:
    expected = {
        f"d{d}-s{t}": (d, t) for d in PROTOCOL["dataset_seeds"] for t in PROTOCOL["training_seeds"]
    }
    directory = output / "development"
    if {p.name for p in directory.iterdir()} != set(expected):
        raise RuntimeError("development cell matrix mismatch")
    files = {}
    for name, (dataset, seed) in expected.items():
        cell = directory / name
        required = {"checkpoint.pt", "report.json", "REFERENCE_FREEZE.json"}
        if cell.is_symlink() or {p.name for p in cell.iterdir()} != required:
            raise RuntimeError("development file matrix mismatch")
        for filename in required:
            path = cell / filename
            if path.is_symlink() or not path.is_file():
                raise RuntimeError("nonregular frozen artifact")
            files[str(path.relative_to(output))] = v4.sha(path)
        report = v4.read(cell / "report.json")
        reference = v4.read(cell / "REFERENCE_FREEZE.json")
        if (report["dataset"], report["training_seed"]) != (dataset, seed):
            raise RuntimeError("development cell identity mismatch")
        if report["checkpoint_sha256"] != v4.sha(cell / "checkpoint.pt"):
            raise RuntimeError("development checkpoint hash mismatch")
        if reference["mode"] != report["reference_mode"] or reference["mode"] != select(
            reference["reports"], PROTOCOL["reference_candidates"]
        ):
            raise RuntimeError("reference selection mismatch")
        if report["selected_control"] != select(
            report["selection"], PROTOCOL["control_candidates"]
        ):
            raise RuntimeError("control selection mismatch")
        ensure_disjoint(report["source_ids"])
    return files


def freeze(output: Path) -> None:
    files = development_manifest(output)
    if sources() != v4.read(output / "CONTRACT.json")["sources"]:
        raise RuntimeError("source changed during development")
    v4.write(
        output / "SCIENTIFIC_FREEZE.json",
        {
            "protocol": PROTOCOL,
            "sources": sources(),
            "time": time.time(),
            "files": files,
        },
    )


def validate_freeze(output: Path) -> dict[str, Any]:
    f = v4.read(output / "SCIENTIFIC_FREEZE.json")
    if f["protocol"] != PROTOCOL or f["sources"] != sources():
        raise RuntimeError("frozen source/protocol differs")
    actual = development_manifest(output)
    if actual != f["files"]:
        raise RuntimeError("changed frozen file or manifest membership")
    return dict(f)


def confirm(output: Path, deadline: float) -> None:
    from olympus.data.conservative_routing_v6 import generate

    validate_freeze(output)
    for cell in sorted((output / "development").iterdir()):
        info = v4.read(cell / "report.json")
        banks, reference, corrections, ref_mode = load_models(cell / "checkpoint.pt")
        destination = output / "protected" / cell.name
        destination.mkdir(parents=True, exist_ok=False)
        partitions = dict(info["source_ids"])
        for i, regime in enumerate(PROTOCOL["protected_regimes"]):
            check_deadline(deadline)
            batch = generate(
                info["dataset"] + PROTOCOL["split_offsets"]["protected"] + i,
                PROTOCOL["test_examples_per_regime"],
                regime,
            )
            partitions[regime] = batch["source_ids"].tolist()
            ensure_disjoint(partitions)
            x, p = ensemble(banks, batch)
            pred, alpha = predictions(reference, corrections, ref_mode, x, p)
            pred["selected_control"] = pred[info["selected_control"]]
            raw = destination / f"{regime}.pt"
            v4.save_tensor(
                raw,
                {
                    "probabilities": pred,
                    "alpha": alpha,
                    "labels": batch["decision_targets"],
                    "source_ids": batch["source_ids"],
                    "experts": p,
                },
            )
            v4.write(
                destination / f"{regime}.json",
                {
                    "dataset": info["dataset"],
                    "training_seed": info["training_seed"],
                    "regime": regime,
                    "metrics": describe(pred, alpha, batch["decision_targets"]),
                    "prediction_sha256": v4.sha(raw),
                    "checkpoint_sha256": info["checkpoint_sha256"],
                    "tensor_hashes": v4.tensors_hash(batch),
                    "freeze_sha256": v4.sha(output / "SCIENTIFIC_FREEZE.json"),
                },
            )
        v4.event(output, phase="protected_cell_complete", cell=cell.name)


def summarize(output: Path) -> dict[str, Any]:
    import numpy as np

    rows = [v4.read(p) for p in sorted((output / "protected").glob("*/*.json"))]
    expected = {
        (d, s, r)
        for d in PROTOCOL["dataset_seeds"]
        for s in PROTOCOL["training_seeds"]
        for r in PROTOCOL["protected_regimes"]
    }
    if {(r["dataset"], r["training_seed"], r["regime"]) for r in rows} != expected or len(
        rows
    ) != len(expected):
        raise RuntimeError("missing/duplicate protected cells")
    modes = list(rows[0]["metrics"])
    means = {
        regime: {
            mode: {
                metric: float(
                    np.mean([r["metrics"][mode][metric] for r in rows if r["regime"] == regime])
                )
                for metric in ("accuracy", "nll", "brier", "ece")
            }
            for mode in modes
        }
        for regime in PROTOCOL["protected_regimes"]
    }
    comparisons = {}
    for regime in PROTOCOL["protected_regimes"]:
        for comparator in ("selected_control", "reference"):
            ds_deltas = [
                float(
                    np.mean(
                        [
                            r["metrics"][PROTOCOL["primary_candidate"]]["accuracy"]
                            - r["metrics"][comparator]["accuracy"]
                            for r in rows
                            if r["regime"] == regime and r["dataset"] == ds
                        ]
                    )
                )
                for ds in PROTOCOL["dataset_seeds"]
            ]
            rng = np.random.default_rng(PROTOCOL["bootstrap_seed"])
            sampled = rng.choice(
                ds_deltas, size=(PROTOCOL["bootstrap_samples"], len(ds_deltas))
            ).mean(1)
            comparisons[f"{regime}/{comparator}"] = {
                "mean": float(np.mean(ds_deltas)),
                "ci95": np.quantile(sampled, [0.025, 0.975]).tolist(),
                "dataset_deltas": ds_deltas,
                "descriptive": True,
            }
    primary = comparisons[PROTOCOL["primary_regime"] + "/selected_control"]
    supported = primary["mean"] >= PROTOCOL["minimum_effect"] and primary["ci95"][0] > 0
    for regime in REGIMES:
        for comparator in ("selected_control", "reference"):
            c = comparisons[f"{regime}/{comparator}"]
            supported &= (
                c["mean"] >= PROTOCOL["nonregression_mean"]
                and c["ci95"][0] >= PROTOCOL["nonregression_interval_lower"]
            )
    return {
        "protocol": PROTOCOL["id"],
        "disposition": "SUPPORTED" if supported else "NEGATIVE",
        "scope": PROTOCOL["scope"],
        "cells": len(rows),
        "means": means,
        "comparisons": comparisons,
        "independent_retraining": False,
        "promotion_authorized": False,
    }


def verify(output: Path) -> dict[str, Any]:
    import torch

    from olympus.data.conservative_routing_v6 import generate

    validate_freeze(output)
    count = 0
    expected_cells = {
        f"d{d}-s{t}" for d in PROTOCOL["dataset_seeds"] for t in PROTOCOL["training_seeds"]
    }
    if {p.name for p in (output / "protected").iterdir()} != expected_cells:
        raise RuntimeError("protected cell matrix mismatch")
    for cell_name in sorted(expected_cells):
        cell = output / "protected" / cell_name
        expected_files = {
            f"{r}.{ext}" for r in PROTOCOL["protected_regimes"] for ext in ("pt", "json")
        }
        if cell.is_symlink() or {p.name for p in cell.iterdir()} != expected_files:
            raise RuntimeError("protected file matrix mismatch")
        development = v4.read(output / "development" / cell_name / "report.json")
        banks, reference, corrections, ref_mode = load_models(
            output / "development" / cell_name / "checkpoint.pt"
        )
        partitions = dict(development["source_ids"])
        for offset, regime in enumerate(PROTOCOL["protected_regimes"]):
            path = cell / f"{regime}.json"
            raw_path = path.with_suffix(".pt")
            if path.is_symlink() or raw_path.is_symlink():
                raise RuntimeError("symlink protected artifact")
            report = v4.read(path)
            if (report["dataset"], report["training_seed"], report["regime"]) != (
                development["dataset"],
                development["training_seed"],
                regime,
            ):
                raise RuntimeError("protected identity mismatch")
            if v4.sha(raw_path) != report["prediction_sha256"] or report["freeze_sha256"] != v4.sha(
                output / "SCIENTIFIC_FREEZE.json"
            ):
                raise RuntimeError("prediction/freeze hash mismatch")
            if report["checkpoint_sha256"] != development["checkpoint_sha256"]:
                raise RuntimeError("checkpoint binding mismatch")
            batch = generate(
                development["dataset"] + PROTOCOL["split_offsets"]["protected"] + offset,
                PROTOCOL["test_examples_per_regime"],
                regime,
            )
            if v4.tensors_hash(batch) != report["tensor_hashes"]:
                raise RuntimeError("regenerated data hash mismatch")
            raw = torch.load(raw_path, map_location="cpu", weights_only=True)
            partitions[regime] = raw["source_ids"].tolist()
            ensure_disjoint(partitions)
            if not torch.equal(raw["source_ids"], batch["source_ids"]) or not torch.equal(
                raw["labels"], batch["decision_targets"]
            ):
                raise RuntimeError("regenerated source IDs or labels differ")
            x, experts = ensemble(banks, batch)
            pred, alpha = predictions(reference, corrections, ref_mode, x, experts)
            pred["selected_control"] = pred[development["selected_control"]]
            for key, expected in (("probabilities", pred), ("alpha", alpha)):
                if set(raw[key]) != set(expected) or any(
                    not torch.equal(raw[key][k], expected[k]) for k in expected
                ):
                    raise RuntimeError("checkpoint prediction replay mismatch")
            if not torch.equal(experts, raw["experts"]):
                raise RuntimeError("checkpoint expert replay mismatch")
            if describe(pred, alpha, raw["labels"]) != report["metrics"]:
                raise RuntimeError("metric or intervention arithmetic mismatch")
            count += 1
    result = summarize(output)
    if result != v4.read(output / "SUMMARY.json"):
        raise RuntimeError("summary arithmetic mismatch")
    return {
        "status": "PASS",
        "raw_reports_recomputed": count,
        "checkpoint_prediction_replay": "EXACT",
        "independent_retraining": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["run", "qualify", "verify"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output.resolve()
    if not args.worker:
        timeout = (
            PROTOCOL["max_qualification_seconds"]
            if args.command == "qualify"
            else PROTOCOL["max_campaign_seconds"]
        )
        command = [
            sys.executable,
            "-m",
            "scripts.run_conservative_routing_v6",
            *sys.argv[1:],
            "--worker",
        ]
        try:
            result = subprocess.run(command, timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            output.mkdir(parents=True, exist_ok=True)
            v4.write(
                output / f"SUPERVISOR_TIMEOUT-{time.time_ns()}.json",
                {"timeout_seconds": timeout, "command": args.command},
            )
            raise
        raise SystemExit(result.returncode)
    if args.command == "run":
        if args.qualification is None:
            raise ValueError("run requires source-bound --qualification directory")
        qualified = v4.read(args.qualification / "QUALIFIED.json")
        if qualified["sources"] != sources() or qualified["tests_sha256"] != v4.sha(
            args.qualification / "tests.xml"
        ):
            raise RuntimeError("qualification is stale or altered")
    if args.command != "verify":
        output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        from olympus.foundry.resources import ResourceGovernor
        from scripts.qualify_flagship_v2 import local_tensor_admission

        admission = local_tensor_admission()
        with ResourceGovernor(ROOT / "artifacts/locks/campaign-validation.lock") as governor:
            v4.event(
                output,
                phase="admitted",
                local=admission,
                memory=governor.preflight.as_dict() if governor.preflight else None,
            )
            import torch

            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            torch.use_deterministic_algorithms(True)
            if args.command == "qualify":
                import pytest

                code = pytest.main(
                    [
                        "-q",
                        "tests/test_conservative_routing_v6.py",
                        "--override-ini=addopts=",
                        "--junitxml=" + str(output / "tests.xml"),
                    ]
                )
                if code or time.monotonic() - started > PROTOCOL["max_qualification_seconds"]:
                    raise RuntimeError(f"qualification failed or exceeded budget: {code}")
                v4.write(
                    output / "QUALIFIED.json",
                    {
                        "sources": sources(),
                        "tests_sha256": v4.sha(output / "tests.xml"),
                        "elapsed_seconds": time.monotonic() - started,
                    },
                )
            elif args.command == "verify":
                print(json.dumps(verify(output)))
            else:
                v4.write(
                    output / "CONTRACT.json",
                    {
                        "protocol": PROTOCOL,
                        "sources": sources(),
                        "python": sys.version,
                        "torch": str(torch.__version__),
                        "platform": platform.platform(),
                    },
                )
                deadline = started + PROTOCOL["max_campaign_seconds"]
                develop(output, deadline)
                freeze(output)
                confirm(output, deadline)
                v4.write(output / "SUMMARY.json", summarize(output))
                v4.write(output / "VERIFIED.json", verify(output))
                v4.event(output, phase="completed", elapsed_seconds=time.monotonic() - started)
    except BaseException as error:
        v4.write(
            output / f"FAILURE-{time.time_ns()}.json",
            {
                "error": repr(error),
                "torch_imported": "torch" in sys.modules,
                "elapsed_seconds": time.monotonic() - started,
                "sources": sources(),
            },
        )
        raise


if __name__ == "__main__":
    main()

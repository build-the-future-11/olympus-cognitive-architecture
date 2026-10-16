#!/usr/bin/env python3
"""Execute the frozen Olympus controlled-fusion study without protected-set tuning."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import torch

from olympus.data.controlled_fusion import generate_controlled_fusion_dataset
from olympus.data.olympus_dataset import load_olympus_dataset
from olympus.evaluation.olympus_system import (
    evaluate_olympus_system,
    olympus_batch_inputs_targets,
)
from olympus.models.olympus_system import OlympusSystemConfig
from olympus.models.reference_families import (
    AionReferenceConfig,
    AtlasReferenceConfig,
    HermesReferenceConfig,
    KronosReferenceConfig,
    PerseusReferenceConfig,
    PrometheusReferenceConfig,
    ReferenceTransformerConfig,
)
from olympus.models.runtime import load_olympus_checkpoint
from olympus.training.olympus_system import (
    OlympusSystemTrainingConfig,
    train_olympus_system,
)

SEEDS = (20261301, 20261302, 20261303, 20261304, 20261305)
VARIANTS = (
    "full",
    "uniform",
    "no_counterfactual",
    "no_disagreement_risk",
    "mechanism_off",
)
PROTECTED_REGIMES = ("clean_test", "corrupted_test", "ood_test")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(path)


def _model_config(variant: str) -> OlympusSystemConfig:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    backbone = ReferenceTransformerConfig(
        vocab_size=32,
        model_dim=16,
        layers=1,
        heads=4,
        feedforward_dim=32,
        max_sequence_length=8,
    )
    return OlympusSystemConfig(
        hermes=HermesReferenceConfig(backbone=backbone, tool_count=4),
        prometheus=PrometheusReferenceConfig(
            backbone=backbone,
            domains=3,
            adapter_dim=8,
            evidence_classes=4,
        ),
        perseus=PerseusReferenceConfig(
            backbone=backbone,
            experts=4,
            active_experts=2,
            expert_dim=32,
        ),
        atlas=AtlasReferenceConfig(
            backbone=backbone,
            image_channels=3,
            patch_size=2,
            retrieval_dim=12,
        ),
        kronos=KronosReferenceConfig(
            state_dim=5,
            action_dim=3,
            hidden_dim=12,
            discrete_actions=4,
            planning_horizon=4,
        ),
        aion=AionReferenceConfig(
            state_dim=8,
            hidden_dim=12,
            research_stages=4,
            stop_stage=3,
            kill_threshold=0.85,
        ),
        fusion_dim=16,
        decision_classes=4,
        fusion_strategy="uniform" if variant in {"uniform", "mechanism_off"} else "learned",
        disagreement_risk=variant not in {"no_disagreement_risk", "mechanism_off"},
    )


def _training_config(variant: str, seed: int) -> OlympusSystemTrainingConfig:
    loss = {
        "transition_weight": 0.5,
        "temporal_weight": 0.25,
        "uncertainty_weight": 0.1,
        "risk_weight": 0.1,
        "gate_balance_weight": 0.01,
        "counterfactual_weight": (
            0.0 if variant in {"no_counterfactual", "mechanism_off"} else 0.05
        ),
    }
    return OlympusSystemTrainingConfig.model_validate(
        {
            "seed": seed,
            "steps": 300,
            "batch_size": 32,
            "learning_rate": 5e-4,
            "minimum_learning_rate_ratio": 0.1,
            "warmup_steps": 20,
            "weight_decay": 0.01,
            "gradient_clip_norm": 1.0,
            "device": "cpu",
            "deterministic": True,
            "resource_preflight": True,
            "min_available_bytes": 2 * 1024**3,
            "max_swap_fraction": 0.90,
            "workload_lock": "artifacts/locks/olympus-controlled-fusion.lock",
            "model": _model_config(variant).model_dump(mode="json"),
            "loss": loss,
        }
    )


def _campaign_contract() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol": "research/olympus_irp_successor_v2/PROTOCOL.md",
        "variants": list(VARIANTS),
        "seeds": list(SEEDS),
        "protected_regimes": list(PROTECTED_REGIMES),
        "training_configs": {
            f"{variant}:{seed}": _training_config(variant, seed).model_dump(mode="json")
            for variant in VARIANTS
            for seed in SEEDS
        },
    }


def prepare_data(dataset_root: Path) -> None:
    receipt = generate_controlled_fusion_dataset(dataset_root)
    print(json.dumps(receipt.model_dump(mode="json"), indent=2, sort_keys=True))


def develop(dataset_root: Path, run_root: Path) -> None:
    contract_path = run_root / "campaign_contract.json"
    contract = _campaign_contract()
    if contract_path.exists():
        if json.loads(contract_path.read_text(encoding="utf-8")) != contract:
            raise RuntimeError("existing campaign contract differs from the frozen contract")
    else:
        _atomic_json(contract_path, contract)

    manifest = dataset_root / "manifest_development.json"
    for variant in VARIANTS:
        for seed in SEEDS:
            output = run_root / "development" / variant / str(seed)
            if (output / "report.json").is_file():
                continue
            try:
                started = time.time()
                report = train_olympus_system(
                    manifest,
                    output,
                    config=_training_config(variant, seed),
                )
                print(
                    f"completed variant={variant} seed={seed} "
                    f"loss={report.final_evaluation.total_loss:.6f} "
                    f"elapsed={time.time() - started:.1f}s"
                )
            except RuntimeError as error:
                message = str(error)
                if (
                    "available memory" in message
                    or "swap utilization" in message
                    or "another model-loading or training workload holds the lock" in message
                ):
                    _atomic_json(
                        run_root / f"campaign_block_{int(time.time())}.json",
                        {
                            "status": "BLOCKED_CAMPAIGN_PREFLIGHT",
                            "variant": variant,
                            "seed": seed,
                            "error": message,
                            "resource_guard_weakened": False,
                        },
                    )
                    print(f"resource blocked before training: {message}")
                    return
                _atomic_json(
                    output / "failure.json",
                    {
                        "status": "FAILED_RETAINED",
                        "variant": variant,
                        "seed": seed,
                        "error": message,
                    },
                )
                print(f"failed variant={variant} seed={seed}: {message}")


def freeze(dataset_root: Path, run_root: Path, repo_root: Path) -> None:
    reports = [
        run_root / "development" / variant / str(seed) / "report.json"
        for variant in VARIANTS
        for seed in SEEDS
    ]
    missing = [str(path) for path in reports if not path.is_file()]
    if missing:
        raise RuntimeError(f"cannot freeze; {len(missing)} development reports are missing")
    tracked_sources = (
        repo_root / "olympus/models/olympus_system.py",
        repo_root / "olympus/data/controlled_fusion.py",
        repo_root / "olympus/evaluation/olympus_system.py",
        repo_root / "olympus/training/olympus_system.py",
        repo_root / "scripts/run_controlled_fusion_study.py",
        repo_root / "research/olympus_irp_successor_v2/PROTOCOL.md",
    )
    dataset_files = tuple(sorted(dataset_root.glob("*")))
    receipt = {
        "schema_version": 1,
        "status": "CONFIRMATORY_PROTOCOL_FROZEN",
        "frozen_at_unix": time.time(),
        "contract_sha256": _sha256(run_root / "campaign_contract.json"),
        "source_sha256": {
            str(path.relative_to(repo_root)): _sha256(path) for path in tracked_sources
        },
        "dataset_sha256": {path.name: _sha256(path) for path in dataset_files if path.is_file()},
        "checkpoint_sha256": {},
    }
    # Build the checkpoint map explicitly; the verbose form above is intentionally
    # avoided because variant/seed identity is scientific metadata.
    receipt["checkpoint_sha256"] = {
        f"{variant}:{seed}": _sha256(
            run_root / "development" / variant / str(seed) / "checkpoint.pt"
        )
        for variant in VARIANTS
        for seed in SEEDS
    }
    path = run_root / "SCIENTIFIC_FREEZE.json"
    if path.exists():
        raise FileExistsError("refusing to overwrite the scientific freeze receipt")
    _atomic_json(path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


def _evaluate_checkpoint(
    checkpoint: Path,
    manifest: Path,
    expected_checkpoint_sha256: str,
) -> dict[str, Any]:
    model, _ = load_olympus_checkpoint(
        checkpoint,
        expected_sha256=expected_checkpoint_sha256,
    )
    dataset = load_olympus_dataset(manifest)
    batch = dataset.evaluation
    evaluation = evaluate_olympus_system(model, batch, dataset.manifest.extensions)
    inputs, _ = olympus_batch_inputs_targets(batch, dataset.manifest.extensions)
    model.eval()
    with torch.inference_mode():
        output = model(**inputs)
    retained = max(1, math.ceil(batch["decision_targets"].shape[0] * 0.5))
    lowest_risk = output.effective_risk.argsort()[:retained]
    raw_correct = output.raw_decision_logits.argmax(dim=-1).eq(batch["decision_targets"])
    selective_accuracy_at_50 = float(raw_correct[lowest_risk].float().mean().item())
    risk_violations = int((output.effective_risk + 1e-7 < batch["risk_score"]).sum().item())
    transition_allowed = batch["allowed_transitions"].gather(
        1, output.aion.selected_stage.unsqueeze(1)
    )
    transition_violations = int((~transition_allowed).sum().item())
    return {
        "evaluation": evaluation.model_dump(mode="json"),
        "selective_accuracy_at_50_all_examples": selective_accuracy_at_50,
        "selective_retained_examples": retained,
        "risk_monotonicity_violations": risk_violations,
        "transition_legality_violations": transition_violations,
    }


def confirm(dataset_root: Path, run_root: Path) -> None:
    freeze_path = run_root / "SCIENTIFIC_FREEZE.json"
    if not freeze_path.is_file():
        raise RuntimeError("confirmatory evaluation requires SCIENTIFIC_FREEZE.json")
    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
    for variant in VARIANTS:
        for seed in SEEDS:
            checkpoint = run_root / "development" / variant / str(seed) / "checkpoint.pt"
            expected = frozen["checkpoint_sha256"][f"{variant}:{seed}"]
            for regime in PROTECTED_REGIMES:
                destination = run_root / "confirmatory" / variant / str(seed) / f"{regime}.json"
                if destination.exists():
                    continue
                result = _evaluate_checkpoint(
                    checkpoint,
                    dataset_root / f"manifest_{regime}.json",
                    expected,
                )
                _atomic_json(
                    destination,
                    {
                        "schema_version": 1,
                        "status": "CONFIRMATORY_EVALUATION_COMPLETE",
                        "variant": variant,
                        "seed": seed,
                        "regime": regime,
                        **result,
                    },
                )
                print(f"evaluated variant={variant} seed={seed} regime={regime}")


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _paired_bootstrap(differences: list[float]) -> dict[str, float]:
    rng = random.Random(20261401)
    means = [
        sum(rng.choice(differences) for _ in differences) / len(differences)
        for _ in range(20_000)
    ]
    return {
        "mean": sum(differences) / len(differences),
        "ci95_low": _percentile(means, 0.025),
        "ci95_high": _percentile(means, 0.975),
    }


def _sign_flip_pvalue(differences: list[float]) -> float:
    observed = abs(sum(differences) / len(differences))
    permuted = [
        abs(
            sum(sign * value for sign, value in zip(signs, differences, strict=True))
            / len(differences)
        )
        for signs in itertools.product((-1.0, 1.0), repeat=len(differences))
    ]
    return sum(value >= observed - 1e-15 for value in permuted) / len(permuted)


def _holm_adjust(raw: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw, key=raw.get)  # type: ignore[arg-type]
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, name in enumerate(ordered):
        running = max(running, (count - rank) * raw[name])
        adjusted[name] = min(1.0, running)
    return adjusted


def analyze(run_root: Path) -> None:
    results: dict[str, dict[int, dict[str, dict[str, Any]]]] = {}
    for variant in VARIANTS:
        results[variant] = {}
        for seed in SEEDS:
            results[variant][seed] = {
                regime: json.loads(
                    (
                        run_root
                        / "confirmatory"
                        / variant
                        / str(seed)
                        / f"{regime}.json"
                    ).read_text(encoding="utf-8")
                )
                for regime in PROTECTED_REGIMES
            }

    full_clean = [
        float(results["full"][seed]["clean_test"]["evaluation"]["decision_accuracy"])
        for seed in SEEDS
    ]
    uniform_clean = [
        float(results["uniform"][seed]["clean_test"]["evaluation"]["decision_accuracy"])
        for seed in SEEDS
    ]
    full_selective = [
        float(
            results["full"][seed]["corrupted_test"][
                "selective_accuracy_at_50_all_examples"
            ]
        )
        for seed in SEEDS
    ]
    uniform_selective = [
        float(
            results["uniform"][seed]["corrupted_test"][
                "selective_accuracy_at_50_all_examples"
            ]
        )
        for seed in SEEDS
    ]
    clean_difference = [a - b for a, b in zip(full_clean, uniform_clean, strict=True)]
    selective_difference = [
        a - b for a, b in zip(full_selective, uniform_selective, strict=True)
    ]
    clean_stats = _paired_bootstrap(clean_difference)
    selective_stats = _paired_bootstrap(selective_difference)

    invariant_violations = sum(
        int(results[variant][seed][regime][name])
        for variant in VARIANTS
        for seed in SEEDS
        for regime in PROTECTED_REGIMES
        for name in ("risk_monotonicity_violations", "transition_legality_violations")
    )
    supported = (
        clean_stats["mean"] >= 0.02
        and clean_stats["ci95_low"] > 0.0
        and selective_stats["mean"] >= 0.03
        and selective_stats["ci95_low"] > 0.0
        and invariant_violations == 0
    )
    positive_signal = clean_stats["mean"] > 0.0 or selective_stats["mean"] > 0.0

    secondary_raw: dict[str, dict[str, float]] = {"clean": {}, "selective": {}}
    for variant in ("no_counterfactual", "no_disagreement_risk", "mechanism_off"):
        other_clean = [
            float(results[variant][seed]["clean_test"]["evaluation"]["decision_accuracy"])
            for seed in SEEDS
        ]
        other_selective = [
            float(
                results[variant][seed]["corrupted_test"][
                    "selective_accuracy_at_50_all_examples"
                ]
            )
            for seed in SEEDS
        ]
        secondary_raw["clean"][variant] = _sign_flip_pvalue(
            [a - b for a, b in zip(full_clean, other_clean, strict=True)]
        )
        secondary_raw["selective"][variant] = _sign_flip_pvalue(
            [a - b for a, b in zip(full_selective, other_selective, strict=True)]
        )

    summary = {
        "schema_version": 1,
        "status": "EVIDENCE_LOCKED",
        "disposition": "SUPPORTED" if supported else ("MIXED" if positive_signal else "NEGATIVE"),
        "primary": {
            "clean_accuracy_full_minus_uniform": clean_stats,
            "corrupted_selective_accuracy_full_minus_uniform": selective_stats,
            "required_clean_difference": 0.02,
            "required_selective_difference": 0.03,
            "invariant_violations": invariant_violations,
        },
        "secondary_sign_flip_pvalues": secondary_raw,
        "secondary_holm_adjusted_pvalues": {
            family: _holm_adjust(values) for family, values in secondary_raw.items()
        },
        "claim_boundary": (
            "This disposition applies only to the registered controlled synthetic generator."
        ),
    }
    path = run_root / "CONFIRMATORY_SUMMARY.json"
    if path.exists():
        raise FileExistsError("refusing to overwrite locked confirmatory evidence")
    _atomic_json(path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("prepare-data", "develop", "freeze", "confirm", "analyze"),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("artifacts/research/olympus-controlled-fusion-v2/dataset"),
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("artifacts/research/olympus-controlled-fusion-v2/runs"),
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    if args.command == "prepare-data":
        prepare_data(args.dataset_root)
    elif args.command == "develop":
        develop(args.dataset_root, args.run_root)
    elif args.command == "freeze":
        freeze(args.dataset_root, args.run_root, repo_root)
    elif args.command == "confirm":
        confirm(args.dataset_root, args.run_root)
    else:
        analyze(args.run_root)


if __name__ == "__main__":
    main()

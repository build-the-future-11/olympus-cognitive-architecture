#!/usr/bin/env python3
"""Run the preregistered Olympus v3 reliability-routing successor study."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
import time
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Literal

import torch
from pydantic import Field, model_validator
from torch import Tensor

from olympus.core.schemas import StrictModel
from olympus.data.olympus_dataset import load_olympus_dataset
from olympus.data.reliability_fusion import generate_reliability_fusion_dataset
from olympus.foundry.resources import ResourceGovernor
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
from olympus.models.reliability_fusion import (
    ReliabilityAwareOlympusSystem,
    ReliabilityFusionConfig,
    ReliabilityLossConfig,
    reliability_fusion_loss,
)
from olympus.models.runtime import RuntimeDevice, resolve_olympus_device

SEEDS = (20261601, 20261602, 20261603, 20261604, 20261605)
VARIANTS = (
    "full",
    "no_reliability_supervision",
    "semantic",
    "uniform",
    "reliability_only",
)
PROTECTED_REGIMES = ("clean_test", "corrupted_test", "ood_test")
INPUT_NAMES = frozenset(
    {
        "tokens",
        "image",
        "states",
        "actions",
        "allowed_transitions",
        "risk_score",
        "attention_mask",
        "domain_ids",
        "temporal_mask",
        "timestamp",
    }
)


class V3TrainingConfig(StrictModel):
    seed: int = Field(ge=0, le=2**32 - 1)
    steps: int = Field(default=300, ge=1, le=10_000_000)
    batch_size: int = Field(default=32, ge=1, le=4_096)
    learning_rate: float = Field(default=5e-4, gt=0.0, le=0.1)
    minimum_learning_rate_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    warmup_steps: int = Field(default=20, ge=0)
    weight_decay: float = Field(default=0.01, ge=0.0, le=1.0)
    gradient_clip_norm: float = Field(default=1.0, gt=0.0, le=100.0)
    device: RuntimeDevice = "cpu"
    deterministic: bool = True
    resource_preflight: bool = True
    min_available_bytes: int = Field(default=2 * 1024**3, ge=256 * 1024**2)
    max_swap_fraction: float = Field(default=0.90, gt=0.0, le=1.0)
    workload_lock: Path = Path("artifacts/locks/olympus-reliability-fusion-v3.lock")
    model: ReliabilityFusionConfig
    loss: ReliabilityLossConfig

    @model_validator(mode="after")
    def valid_schedule(self) -> V3TrainingConfig:
        if self.warmup_steps > self.steps:
            raise ValueError("warmup steps must not exceed total steps")
        return self


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(path)


def _atomic_json(path: Path, payload: object) -> None:
    _atomic_bytes(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())


def _base_system_config() -> OlympusSystemConfig:
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
        fusion_strategy="learned",
        disagreement_risk=True,
    )


def _training_config(variant: str, seed: int) -> V3TrainingConfig:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    routing_modes = {
        "full": "reliability_shrinkage",
        "no_reliability_supervision": "reliability_shrinkage",
        "semantic": "semantic",
        "uniform": "uniform",
        "reliability_only": "reliability_only",
    }
    reliability_weight = (
        0.0
        if variant
        in {
            "no_reliability_supervision",
            "semantic",
            "uniform",
        }
        else 0.25
    )
    return V3TrainingConfig(
        seed=seed,
        model=ReliabilityFusionConfig(
            system=_base_system_config(),
            routing_mode=routing_modes[variant],  # type: ignore[arg-type]
            reliability_logit_strength=1.0,
            shrinkage_to_adaptive=0.75,
        ),
        loss=ReliabilityLossConfig(reliability_weight=reliability_weight),
    )


def _campaign_contract() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol": "research/olympus_irp_successor_v3/PROTOCOL.md",
        "variants": list(VARIANTS),
        "seeds": list(SEEDS),
        "protected_regimes": list(PROTECTED_REGIMES),
        "primary_comparison": {
            "variant": "full",
            "control": "semantic",
            "regime": "corrupted_test",
            "endpoint": "selective_accuracy_at_50_all_examples",
            "practical_effect_threshold": 0.02,
        },
        "training_configs": {
            f"{variant}:{seed}": _training_config(variant, seed).model_dump(mode="json")
            for variant in VARIANTS
            for seed in SEEDS
        },
    }


def _split_batch(batch: Mapping[str, Tensor]) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    inputs = {name: value for name, value in batch.items() if name in INPUT_NAMES}
    targets = {
        name: batch[name]
        for name in (
            "decision_targets",
            "transition_targets",
            "next_state_targets",
            "component_reliability_targets",
        )
    }
    if "intrinsic_risk_targets" in batch:
        targets["intrinsic_risk_targets"] = batch["intrinsic_risk_targets"]
    return inputs, targets


def _move_batch(batch: Mapping[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {name: tensor.to(device=device) for name, tensor in batch.items()}


def _learning_rate(config: V3TrainingConfig, step: int) -> float:
    completed = step + 1
    if config.warmup_steps and completed <= config.warmup_steps:
        scale = completed / config.warmup_steps
    elif config.steps == config.warmup_steps:
        scale = 1.0
    else:
        progress = (completed - config.warmup_steps) / (config.steps - config.warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        floor = config.minimum_learning_rate_ratio
        scale = floor + (1.0 - floor) * cosine
    return config.learning_rate * scale


def _evaluate(
    model: ReliabilityAwareOlympusSystem,
    batch: Mapping[str, Tensor],
    loss_config: ReliabilityLossConfig,
) -> dict[str, Any]:
    inputs, targets = _split_batch(batch)
    model.eval()
    with torch.inference_mode():
        output = model(**inputs)
        loss = reliability_fusion_loss(
            output,
            **targets,
            temporal_mask=inputs.get("temporal_mask"),
            config=loss_config,
        )
    system = output.system
    predictions = system.raw_decision_logits.argmax(dim=-1)
    correct = predictions.eq(targets["decision_targets"])
    retained = max(1, math.ceil(correct.numel() * 0.5))
    lowest_risk = system.effective_risk.argsort()[:retained]
    reliability_targets = targets["component_reliability_targets"].to(output.component_reliability)
    damaged = reliability_targets < 1.0
    clean = ~damaged
    discrimination = None
    if bool(damaged.any()) and bool(clean.any()):
        discrimination = float(
            (
                output.component_reliability[clean].mean()
                - output.component_reliability[damaged].mean()
            ).item()
        )
    transition_allowed = batch["allowed_transitions"].gather(
        1, system.aion.selected_stage.unsqueeze(1)
    )
    return {
        "total_loss": float(loss.total.item()),
        "system_loss": float(loss.system.total.item()),
        "reliability_calibration_loss": float(loss.reliability_calibration.item()),
        "decision_accuracy": float(correct.float().mean().item()),
        "selective_accuracy_at_50_all_examples": float(correct[lowest_risk].float().mean().item()),
        "selective_retained_examples": retained,
        "transition_accuracy": float(
            system.aion.selected_stage.eq(targets["transition_targets"]).float().mean().item()
        ),
        "reliability_mae": float(
            (output.component_reliability - reliability_targets).abs().mean().item()
        ),
        "reliability_brier": float(
            (output.component_reliability - reliability_targets).square().mean().item()
        ),
        "reliability_clean_minus_damaged": discrimination,
        "mean_semantic_gates": {
            name: float(output.semantic_gates[:, index].mean().item())
            for index, name in enumerate(model.component_names)
        },
        "mean_final_gates": {
            name: float(system.family_gates[:, index].mean().item())
            for index, name in enumerate(model.component_names)
        },
        "risk_monotonicity_violations": int(
            (system.effective_risk + 1e-7 < batch["risk_score"]).sum().item()
        ),
        "transition_legality_violations": int((~transition_allowed).sum().item()),
    }


def _save_checkpoint(
    model: ReliabilityAwareOlympusSystem,
    config: V3TrainingConfig,
    path: Path,
) -> str:
    buffer = io.BytesIO()
    torch.save(
        {
            "schema_version": 1,
            "model_config": config.model.model_dump(mode="json"),
            "state_dict": model.state_dict(),
        },
        buffer,
    )
    payload = buffer.getvalue()
    _atomic_bytes(path, payload)
    return _sha256_bytes(payload)


def _load_checkpoint(path: Path, expected_sha256: str) -> ReliabilityAwareOlympusSystem:
    payload = path.read_bytes()
    if _sha256_bytes(payload) != expected_sha256:
        raise ValueError("v3 checkpoint hash mismatch")
    checkpoint = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    config = ReliabilityFusionConfig.model_validate(checkpoint["model_config"])
    model = ReliabilityAwareOlympusSystem(config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model


def prepare_data(dataset_root: Path, run_root: Path) -> None:
    governor = ResourceGovernor(
        run_root / "olympus-reliability-fusion-v3.lock",
        min_available_bytes=2 * 1024**3,
        max_swap_fraction=0.90,
    )
    with governor:
        receipt = generate_reliability_fusion_dataset(dataset_root)
    print(json.dumps(receipt.model_dump(mode="json"), indent=2, sort_keys=True))


def _train_one(
    manifest: Path,
    output_directory: Path,
    config: V3TrainingConfig,
) -> dict[str, Any]:
    report_path = output_directory / "report.json"
    checkpoint_path = output_directory / "checkpoint.pt"
    if report_path.exists() or checkpoint_path.exists():
        raise FileExistsError("refusing to overwrite an existing v3 run")
    dataset = load_olympus_dataset(manifest)
    device = resolve_olympus_device(config.device)
    governor = (
        ResourceGovernor(
            config.workload_lock,
            min_available_bytes=config.min_available_bytes,
            max_swap_fraction=config.max_swap_fraction,
        )
        if config.resource_preflight
        else nullcontext()
    )
    torch.manual_seed(config.seed)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    model = ReliabilityAwareOlympusSystem(config.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    training_batch = dataset.train
    evaluation_batch = _move_batch(dataset.evaluation, device)

    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    first_loss = math.nan
    final_loss = math.nan
    maximum_gradient_norm = 0.0
    started = time.time()
    try:
        if config.deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
        with governor:
            initial = _evaluate(model, evaluation_batch, config.loss)
            for step in range(config.steps):
                indices = torch.randperm(training_batch["tokens"].shape[0], generator=generator)[
                    : config.batch_size
                ]
                selected = {
                    name: tensor.index_select(0, indices).to(device=device)
                    for name, tensor in training_batch.items()
                }
                inputs, targets = _split_batch(selected)
                learning_rate = _learning_rate(config, step)
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
                model.train()
                optimizer.zero_grad(set_to_none=True)
                output = model(**inputs)
                loss = reliability_fusion_loss(
                    output,
                    **targets,
                    temporal_mask=inputs.get("temporal_mask"),
                    config=config.loss,
                )
                if not bool(torch.isfinite(loss.total)):
                    raise FloatingPointError("v3 training produced a non-finite loss")
                loss.total.backward()  # type: ignore[no-untyped-call]
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.gradient_clip_norm
                )
                if not bool(torch.isfinite(gradient_norm)):
                    raise FloatingPointError("v3 training produced a non-finite gradient")
                optimizer.step()
                value = float(loss.total.detach().item())
                if step == 0:
                    first_loss = value
                final_loss = value
                maximum_gradient_norm = max(maximum_gradient_norm, float(gradient_norm.item()))
            final = _evaluate(model, evaluation_batch, config.loss)
    finally:
        torch.use_deterministic_algorithms(previous_deterministic, warn_only=True)

    model.cpu()
    checkpoint_sha256 = _save_checkpoint(model, config, checkpoint_path)
    manifest_sha256 = dataset.manifest.manifest_sha256
    assert manifest_sha256 is not None
    serialized_config = json.dumps(
        config.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode()
    report = {
        "schema_version": 1,
        "status": "COMPLETED_DEVELOPMENT_NOT_PROMOTED",
        "dataset_manifest_sha256": manifest_sha256,
        "training_config_sha256": _sha256_bytes(serialized_config),
        "training_config": config.model_dump(mode="json"),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "first_loss": first_loss,
        "final_loss": final_loss,
        "maximum_gradient_norm": maximum_gradient_norm,
        "elapsed_seconds": time.time() - started,
        "initial_evaluation": initial,
        "final_evaluation": final,
        "checkpoint_sha256": checkpoint_sha256,
        "promotion_authorized": False,
        "claim_boundary": "Synthetic v3 development evidence only.",
    }
    _atomic_json(report_path, report)
    return report


def develop(dataset_root: Path, run_root: Path) -> None:
    contract = _campaign_contract()
    contract_path = run_root / "campaign_contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text(encoding="utf-8")) != contract:
            raise RuntimeError("existing v3 campaign contract differs from frozen contract")
    else:
        _atomic_json(contract_path, contract)
    for variant in VARIANTS:
        for seed in SEEDS:
            destination = run_root / "development" / variant / str(seed)
            if (destination / "report.json").is_file():
                continue
            try:
                report = _train_one(
                    dataset_root / "manifest_development.json",
                    destination,
                    _training_config(variant, seed),
                )
                print(f"completed variant={variant} seed={seed} loss={report['final_loss']:.6f}")
            except RuntimeError as error:
                message = str(error)
                if any(
                    marker in message
                    for marker in ("available memory", "swap utilization", "holds the lock")
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
                raise


def freeze(dataset_root: Path, run_root: Path, repo_root: Path) -> None:
    reports = [
        run_root / "development" / variant / str(seed) / "report.json"
        for variant in VARIANTS
        for seed in SEEDS
    ]
    missing = [str(path) for path in reports if not path.is_file()]
    if missing:
        raise RuntimeError(f"cannot freeze v3; {len(missing)} development reports are missing")
    tracked_sources = (
        repo_root / "olympus/models/reliability_fusion.py",
        repo_root / "olympus/data/reliability_fusion.py",
        repo_root / "olympus/data/olympus_dataset.py",
        repo_root / "scripts/run_reliability_fusion_study.py",
        repo_root / "research/olympus_irp_successor_v3/PROTOCOL.md",
    )
    receipt = {
        "schema_version": 1,
        "status": "CONFIRMATORY_PROTOCOL_FROZEN",
        "frozen_at_unix": time.time(),
        "contract_sha256": _sha256(run_root / "campaign_contract.json"),
        "source_sha256": {
            str(path.relative_to(repo_root)): _sha256(path) for path in tracked_sources
        },
        "dataset_sha256": {
            path.name: _sha256(path) for path in sorted(dataset_root.glob("*")) if path.is_file()
        },
        "checkpoint_sha256": {
            f"{variant}:{seed}": _sha256(
                run_root / "development" / variant / str(seed) / "checkpoint.pt"
            )
            for variant in VARIANTS
            for seed in SEEDS
        },
    }
    path = run_root / "SCIENTIFIC_FREEZE.json"
    if path.exists():
        raise FileExistsError("refusing to overwrite the v3 scientific freeze")
    _atomic_json(path, receipt)


def confirm(dataset_root: Path, run_root: Path) -> None:
    freeze_path = run_root / "SCIENTIFIC_FREEZE.json"
    if not freeze_path.is_file():
        raise RuntimeError("v3 confirmatory evaluation requires SCIENTIFIC_FREEZE.json")
    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
    for variant in VARIANTS:
        for seed in SEEDS:
            checkpoint = run_root / "development" / variant / str(seed) / "checkpoint.pt"
            model = _load_checkpoint(checkpoint, frozen["checkpoint_sha256"][f"{variant}:{seed}"])
            config = _training_config(variant, seed)
            for regime in PROTECTED_REGIMES:
                destination = run_root / "confirmatory" / variant / str(seed) / f"{regime}.json"
                if destination.exists():
                    continue
                dataset = load_olympus_dataset(dataset_root / f"manifest_{regime}.json")
                result = _evaluate(model, dataset.evaluation, config.loss)
                _atomic_json(
                    destination,
                    {
                        "schema_version": 1,
                        "status": "CONFIRMATORY_EVALUATION_COMPLETE",
                        "variant": variant,
                        "seed": seed,
                        "regime": regime,
                        "checkpoint_sha256": frozen["checkpoint_sha256"][f"{variant}:{seed}"],
                        **result,
                    },
                )


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
    rng = random.Random(20261701)
    bootstrapped = [
        sum(rng.choice(differences) for _ in differences) / len(differences) for _ in range(10_000)
    ]
    return {
        "mean_delta": sum(differences) / len(differences),
        "ci95_low": _percentile(bootstrapped, 0.025),
        "ci95_high": _percentile(bootstrapped, 0.975),
    }


def analyze(run_root: Path) -> None:
    endpoint = "selective_accuracy_at_50_all_examples"
    differences: list[float] = []
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        full_path = run_root / "confirmatory" / "full" / str(seed) / "corrupted_test.json"
        control_path = run_root / "confirmatory" / "semantic" / str(seed) / "corrupted_test.json"
        if not full_path.is_file() or not control_path.is_file():
            raise RuntimeError("confirmatory matrix is incomplete")
        full = json.loads(full_path.read_text(encoding="utf-8"))
        control = json.loads(control_path.read_text(encoding="utf-8"))
        delta = float(full[endpoint]) - float(control[endpoint])
        differences.append(delta)
        rows.append(
            {"seed": seed, "full": full[endpoint], "semantic": control[endpoint], "delta": delta}
        )
    inference = _paired_bootstrap(differences)
    threshold = 0.02
    if inference["ci95_low"] > 0.0 and inference["mean_delta"] >= threshold:
        disposition: Literal["SUPPORTED", "MIXED", "NEGATIVE", "INCONCLUSIVE"] = "SUPPORTED"
    elif inference["ci95_high"] < threshold:
        disposition = "NEGATIVE"
    elif inference["mean_delta"] > 0.0:
        disposition = "MIXED"
    else:
        disposition = "INCONCLUSIVE"
    result = {
        "schema_version": 1,
        "status": "EVIDENCE_LOCKED",
        "disposition": disposition,
        "primary_endpoint": endpoint,
        "practical_effect_threshold": threshold,
        "paired_rows": rows,
        "paired_bootstrap": inference,
        "claim_boundary": "Synthetic reliability-routing evidence only.",
    }
    _atomic_json(run_root / "EVIDENCE_SUMMARY.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=("prepare", "develop", "freeze", "confirm", "analyze", "all")
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("artifacts/data/olympus-reliability-v3")
    )
    parser.add_argument(
        "--run-root", type=Path, default=Path("artifacts/runs/olympus-reliability-v3")
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    if args.stage in {"prepare", "all"} and not (args.dataset_root / "receipt.json").exists():
        prepare_data(args.dataset_root, args.run_root)
    if args.stage in {"develop", "all"}:
        develop(args.dataset_root, args.run_root)
    if args.stage in {"freeze", "all"}:
        freeze(args.dataset_root, args.run_root, repo_root)
    if args.stage in {"confirm", "all"}:
        confirm(args.dataset_root, args.run_root)
    if args.stage in {"analyze", "all"}:
        analyze(args.run_root)


if __name__ == "__main__":
    main()

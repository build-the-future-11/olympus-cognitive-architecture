"""One-shot frozen multi-horizon synthetic scientific campaign v2."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import socket
import sys
import time
from pathlib import Path
from typing import Any

DATASET_SEEDS = (20268100, 20268200, 20268300)
OPTIMIZER_SEEDS = (20268601, 20268602)
PROTECTED_OFFSETS = {"clean": 11, "corrupted": 12, "shifted": 13}
TRAIN_EXAMPLES = 512
DEVELOPMENT_EXAMPLES = 128
PROTECTED_EXAMPLES = 256
HISTORY = 6
HORIZON = 8
STATE_DIM = 4
ACTION_DIM = 2
HIDDEN_DIM = 32
UPDATES = 200
BATCH_SIZE = 64
LEARNING_RATE = 0.002
MODEL_NAMES = ("direct", "residual", "uncertainty_residual")
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20268999
MAX_WALL_SECONDS = 3_600
WORST_REGIME_H8_RATIO_MAX = 0.90
PAIRED_DELTA_CI_UPPER_MAX = 0.0
CLEAN_H8_REGRESSION_MAX = 0.02
COVERAGE90_ERROR_MAX = 0.10

PROTOCOL = {
    "id": "multi-horizon-dynamics-v2",
    "date": "2026-10-21",
    "dataset_seeds": list(DATASET_SEEDS),
    "optimizer_seeds": list(OPTIMIZER_SEEDS),
    "protected_offsets": PROTECTED_OFFSETS,
    "train_examples": TRAIN_EXAMPLES,
    "development_examples": DEVELOPMENT_EXAMPLES,
    "protected_examples": PROTECTED_EXAMPLES,
    "history": HISTORY,
    "horizon": HORIZON,
    "state_dim": STATE_DIM,
    "action_dim": ACTION_DIM,
    "hidden_dim": HIDDEN_DIM,
    "updates": UPDATES,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "models": list(MODEL_NAMES),
    "controls": ["direct", "residual", "persistence"],
    "candidate": "uncertainty_residual",
    "bootstrap_samples": BOOTSTRAP_SAMPLES,
    "bootstrap_seed": BOOTSTRAP_SEED,
    "max_wall_seconds": MAX_WALL_SECONDS,
    "primary": {
        "worst_regime_h8_ratio_max": WORST_REGIME_H8_RATIO_MAX,
        "paired_delta_ci_upper_max": PAIRED_DELTA_CI_UPPER_MAX,
        "clean_h8_regression_max": CLEAN_H8_REGRESSION_MAX,
        "coverage90_error_max": COVERAGE90_ERROR_MAX,
    },
    "prior_protected_evidence_reused": False,
    "paid_compute": False,
    "promotion_authorized": False,
}


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def resource_preflight() -> dict[str, Any]:
    import psutil  # type: ignore[import-untyped]

    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    if memory.available < 2 * 1024**3:
        raise RuntimeError("resource refusal: less than 2 GiB available")
    if swap.total and swap.percent > 90.0:
        raise RuntimeError("resource refusal: swap exceeds 90 percent")
    return {
        "available_bytes": memory.available,
        "total_bytes": memory.total,
        "swap_total_bytes": swap.total,
        "swap_used_bytes": swap.used,
        "swap_percent": swap.percent,
    }


def internet_preflight() -> dict[str, object]:
    try:
        connection = socket.create_connection(("1.1.1.1", 443), timeout=2.0)
    except OSError as error:
        return {
            "internet_disabled": True,
            "probe": "1.1.1.1:443",
            "refusal": type(error).__name__,
        }
    connection.close()
    raise RuntimeError("configuration refusal: outbound Internet is enabled")


def source_identity() -> dict[str, str]:
    source = Path(__file__).resolve()
    protocol = source.parents[1] / "research/multi_horizon_dynamics_v2/PROTOCOL.md"
    return {
        "source_sha256": file_sha(source),
        "protocol_sha256": file_sha(protocol),
        "protocol_object_sha256": digest(PROTOCOL),
    }


def main(output: Path) -> None:
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    identities = source_identity()
    preflight = resource_preflight()
    network = internet_preflight()
    freeze = {
        "status": "FROZEN_BEFORE_PROTECTED_GENERATION",
        "protocol": PROTOCOL,
        "identities": identities,
        "resource_preflight": preflight,
        "network_preflight": network,
        "protected_generated": False,
    }
    write_new(output / "SCIENTIFIC_FREEZE.json", freeze)

    import torch
    from torch import Tensor, nn

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    class Dynamics(nn.Module):
        def __init__(self, mode: str) -> None:
            super().__init__()
            self.mode = mode
            width = HIDDEN_DIM
            self.encoder = nn.GRU(STATE_DIM + ACTION_DIM, width, batch_first=True)
            self.cell = nn.GRUCell(STATE_DIM + ACTION_DIM, width)
            self.mean = nn.Linear(width, STATE_DIM)
            self.log_variance = (
                nn.Linear(width, STATE_DIM)
                if mode == "uncertainty_residual"
                else None
            )

        def forward(self, history: Tensor, actions: Tensor) -> tuple[Tensor, Tensor | None]:
            _, hidden = self.encoder(history)
            cursor = hidden[-1]
            state = history[:, -1, :STATE_DIM]
            means: list[Tensor] = []
            variances: list[Tensor] = []
            for step in range(HORIZON):
                cursor = self.cell(torch.cat((state, actions[:, step]), -1), cursor)
                predicted = self.mean(cursor)
                state = state + predicted if self.mode != "direct" else predicted
                means.append(state)
                if self.log_variance is not None:
                    variances.append(self.log_variance(cursor).clamp(-8.0, 5.0))
            return torch.stack(means, 1), torch.stack(variances, 1) if variances else None

    def generate(seed: int, count: int, regime: str) -> dict[str, Tensor]:
        generator = torch.Generator().manual_seed(seed)
        actions = torch.randn(
            count, HISTORY + HORIZON, ACTION_DIM, generator=generator
        ) * 0.7
        state = torch.randn(count, STATE_DIM, generator=generator)
        states = [state]
        coupling = torch.tensor(
            [[0.0, 0.25, -0.10, 0.05], [-0.2, 0.0, 0.15, 0.1],
             [0.12, -0.08, 0.0, 0.2], [0.05, 0.1, -0.18, 0.0]]
        )
        action_map = torch.tensor([[0.24, -0.05, 0.12, 0.08], [-0.08, 0.2, 0.04, -0.14]])
        damping = 0.76 if regime != "shifted" else 0.64
        nonlinear = 0.16 if regime != "shifted" else 0.24
        for step in range(HISTORY + HORIZON - 1):
            noise = torch.randn(count, STATE_DIM, generator=generator) * 0.015
            drift = 0.025 * torch.sin(state * (1.0 + step / 10.0))
            state = (
                damping * state
                + nonlinear * torch.tanh(state @ coupling)
                + actions[:, step] @ action_map
                + drift
                + noise
            )
            states.append(state)
        full = torch.stack(states, 1)
        observed = full[:, :HISTORY].clone()
        if regime == "corrupted":
            observed[:, -2:] += torch.randn(
                observed[:, -2:].shape, generator=generator
            ) * 0.25
            observed[:, -1, 1] = 0.0
        history_actions = actions[:, :HISTORY]
        return {
            "history": torch.cat((observed, history_actions), -1),
            "future_actions": actions[:, HISTORY - 1 : HISTORY - 1 + HORIZON],
            "targets": full[:, HISTORY : HISTORY + HORIZON],
            "last_clean": full[:, HISTORY - 1],
        }

    def train_model(
        mode: str, dataset_seed: int, optimizer_seed: int
    ) -> tuple[nn.Module, dict[str, Any]]:
        torch.manual_seed(optimizer_seed)
        random.seed(optimizer_seed)
        model = Dynamics(mode)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=0.01
        )
        train = generate(dataset_seed + 1, TRAIN_EXAMPLES, "clean")
        development = generate(dataset_seed + 2, DEVELOPMENT_EXAMPLES, "clean")
        losses: list[float] = []
        fit_started = time.monotonic()
        for update in range(UPDATES):
            index_generator = torch.Generator().manual_seed(optimizer_seed * 1000 + update)
            indices = torch.randperm(TRAIN_EXAMPLES, generator=index_generator)[:BATCH_SIZE]
            mean, log_variance = model(
                train["history"][indices], train["future_actions"][indices]
            )
            error = mean - train["targets"][indices]
            if log_variance is None:
                loss = error.square().mean()
            else:
                loss = 0.5 * (error.square() * (-log_variance).exp() + log_variance).mean()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("non-finite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        with torch.no_grad():
            prediction, _ = model(development["history"], development["future_actions"])
            development_mse = float((prediction - development["targets"]).square().mean().item())
        return model, {
            "mode": mode,
            "dataset_seed": dataset_seed,
            "optimizer_seed": optimizer_seed,
            "updates": UPDATES,
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "development_mse": development_mse,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "fit_seconds": time.monotonic() - fit_started,
        }

    checkpoints: dict[tuple[int, int, str], Path] = {}
    fit_reports: list[dict[str, Any]] = []
    for dataset_seed in DATASET_SEEDS:
        for optimizer_seed in OPTIMIZER_SEEDS:
            for mode in MODEL_NAMES:
                if time.monotonic() - started > MAX_WALL_SECONDS:
                    raise TimeoutError("campaign exceeded frozen wall budget")
                model, report = train_model(mode, dataset_seed, optimizer_seed)
                checkpoint = output / "checkpoints" / f"{dataset_seed}-{optimizer_seed}-{mode}.pt"
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint)
                report["checkpoint_sha256"] = file_sha(checkpoint)
                checkpoints[(dataset_seed, optimizer_seed, mode)] = checkpoint
                fit_reports.append(report)
    write_new(output / "FIT_REPORTS.json", fit_reports)

    freeze["protected_generated"] = False
    freeze["checkpoint_manifest_sha256"] = digest(
        {str(path.relative_to(output)): file_sha(path) for path in checkpoints.values()}
    )
    write_new(output / "PROTECTED_ADMISSION.json", freeze)

    rows: list[dict[str, Any]] = []
    for dataset_seed in DATASET_SEEDS:
        protected = {
            regime: generate(
                dataset_seed + PROTECTED_OFFSETS[regime],
                PROTECTED_EXAMPLES,
                regime,
            )
            for regime in PROTECTED_OFFSETS
        }
        for optimizer_seed in OPTIMIZER_SEEDS:
            for mode in MODEL_NAMES:
                model = Dynamics(mode)
                state = torch.load(
                    checkpoints[(dataset_seed, optimizer_seed, mode)],
                    map_location="cpu",
                    weights_only=True,
                )
                model.load_state_dict(state)
                model.eval()
                for regime, data in protected.items():
                    with torch.no_grad():
                        prediction, log_variance = model(
                            data["history"], data["future_actions"]
                        )
                    squared = (prediction - data["targets"]).square()
                    horizon_mse = squared.mean((0, 2)).tolist()
                    row: dict[str, Any] = {
                        "dataset_seed": dataset_seed,
                        "optimizer_seed": optimizer_seed,
                        "mode": mode,
                        "regime": regime,
                        "horizon_mse": horizon_mse,
                        "mean_mse": float(squared.mean().item()),
                        "recovery_error": float(squared[:, -1].mean().item()),
                        "prediction_sha256": hashlib.sha256(
                            prediction.contiguous().numpy().tobytes()
                        ).hexdigest(),
                    }
                    if log_variance is not None:
                        standard = (0.5 * log_variance).exp()
                        inside = (prediction - data["targets"]).abs() <= 1.644854 * standard
                        row["coverage90"] = float(inside.float().mean().item())
                    rows.append(row)
            for regime, data in protected.items():
                prediction = data["last_clean"][:, None].expand(-1, HORIZON, -1)
                squared = (prediction - data["targets"]).square()
                rows.append(
                    {
                        "dataset_seed": dataset_seed,
                        "optimizer_seed": optimizer_seed,
                        "mode": "persistence",
                        "regime": regime,
                        "horizon_mse": squared.mean((0, 2)).tolist(),
                        "mean_mse": float(squared.mean().item()),
                        "recovery_error": float(squared[:, -1].mean().item()),
                        "prediction_sha256": hashlib.sha256(
                            prediction.contiguous().numpy().tobytes()
                        ).hexdigest(),
                    }
                )
    write_new(output / "PROTECTED_ROWS.json", rows)

    def clustered(mode: str, regime: str, key: str) -> list[float]:
        values = []
        for dataset_seed in DATASET_SEEDS:
            matched = [
                row for row in rows
                if row["dataset_seed"] == dataset_seed
                and row["mode"] == mode
                and row["regime"] == regime
            ]
            values.append(sum(float(row[key]) for row in matched) / len(matched))
        return values

    regimes = list(PROTECTED_OFFSETS)
    models = [*MODEL_NAMES, "persistence"]
    means = {
        mode: {
            regime: {
                "mean_mse": sum(clustered(mode, regime, "mean_mse")) / 3,
                "horizon8_mse": sum(
                    float(row["horizon_mse"][-1]) for row in rows
                    if row["mode"] == mode and row["regime"] == regime
                ) / 6,
            }
            for regime in regimes
        }
        for mode in models
    }
    candidate = "uncertainty_residual"
    learned_controls = ["direct", "residual"]
    worst_regime = max(regimes, key=lambda value: means[candidate][value]["horizon8_mse"])
    best_control = min(
        learned_controls,
        key=lambda value: means[value][worst_regime]["horizon8_mse"],
    )
    candidate_clusters = clustered(candidate, worst_regime, "recovery_error")
    control_clusters = clustered(best_control, worst_regime, "recovery_error")
    deltas = [candidate_clusters[index] - control_clusters[index] for index in range(3)]
    rng = random.Random(BOOTSTRAP_SEED)
    samples = []
    for _ in range(BOOTSTRAP_SAMPLES):
        selected = [deltas[rng.randrange(3)] for _ in range(3)]
        samples.append(sum(selected) / 3)
    samples.sort()
    interval = [samples[250], samples[9749]]
    coverage = sum(
        float(row["coverage90"]) for row in rows
        if row["mode"] == candidate and row["regime"] == worst_regime
    ) / 6
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
    primary_supported = (
        candidate_h8 / control_h8 <= WORST_REGIME_H8_RATIO_MAX
        and interval[1] < PAIRED_DELTA_CI_UPPER_MAX
        and clean_regression <= CLEAN_H8_REGRESSION_MAX
        and abs(coverage - 0.9) <= COVERAGE90_ERROR_MAX
    )
    summary = {
        "status": "COMPLETED_SUPPORTED" if primary_supported else "COMPLETED_NEGATIVE",
        "primary_supported": primary_supported,
        "worst_regime": worst_regime,
        "best_learned_control": best_control,
        "criteria": criteria,
        "means": means,
        "fits": len(fit_reports),
        "protected_rows": len(rows),
        "protocol_sha256": identities["protocol_sha256"],
        "source_sha256": identities["source_sha256"],
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
        },
        "elapsed_seconds": time.monotonic() - started,
        "promotion_authorized": False,
        "claim_boundary": "fresh bounded synthetic dynamics only",
    }
    write_new(output / "SUMMARY.json", summary)
    write_new(
        output / "MANIFEST.json",
        {
            str(path.relative_to(output)): file_sha(path)
            for path in sorted(output.rglob("*"))
            if path.is_file()
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    target = Path(
        os.environ.get(
            "MULTI_HORIZON_OUTPUT", "/kaggle/working/multi_horizon_dynamics_v2"
        )
    )
    try:
        main(target)
    except BaseException as error:
        target.mkdir(parents=True, exist_ok=True)
        failure = target / f"FAILURE-{time.time_ns()}.json"
        if not failure.exists():
            write_new(
                failure,
                {
                    "status": "FAILED_RETAINED",
                    "error": repr(error),
                    "torch_imported": "torch" in sys.modules,
                    "protocol_sha256": digest(PROTOCOL),
                },
            )
        raise

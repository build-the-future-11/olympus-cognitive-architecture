"""Heavy frozen confirmatory successor for multi-horizon dynamics v3."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

DATASET_SEEDS = (20269100, 20269200, 20269300)
OPTIMIZER_SEEDS = (20269601, 20269602)
PROTECTED_OFFSETS = {"clean": 21, "corrupted": 22, "shifted": 23}
TRAIN_EXAMPLES = 4_096
DEVELOPMENT_EXAMPLES = 1_024
PROTECTED_EXAMPLES = 2_048
HISTORY = 6
HORIZON = 8
STATE_DIM = 4
ACTION_DIM = 2
HIDDEN_DIM = 32
UPDATES = 3_000
BATCH_SIZE = 256
LEARNING_RATE = 0.002
MODEL_NAMES = ("direct", "residual", "uncertainty_residual")
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20269999
MAX_WALL_SECONDS = 21_600
WORST_REGIME_H8_RATIO_MAX = 0.90
PAIRED_DELTA_CI_UPPER_MAX = 0.0
CLEAN_H8_REGRESSION_MAX = 0.02
COVERAGE90_ERROR_MAX = 0.10
BASE_SOURCE_SHA256 = "45a0375eb5297e8d3071487b61ee44df194f765e4254e9cd1cacc2be6844b636"

PROTOCOL = {
    "id": "multi-horizon-dynamics-v3-heavy",
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
    "base_source_sha256": BASE_SOURCE_SHA256,
    "confirmatory_after_v2_negative": True,
    "prior_protected_evidence_reused": False,
    "paid_compute": False,
    "promotion_authorized": False,
}


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def load_base(path: Path) -> ModuleType:
    if file_sha(path) != BASE_SOURCE_SHA256:
        raise RuntimeError("base-source refusal: v2 runner identity mismatch")
    spec = importlib.util.spec_from_file_location("multi_horizon_v3_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("base-source refusal: runner cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure(base: Any) -> None:
    for name, value in {
        "DATASET_SEEDS": DATASET_SEEDS,
        "OPTIMIZER_SEEDS": OPTIMIZER_SEEDS,
        "PROTECTED_OFFSETS": PROTECTED_OFFSETS,
        "TRAIN_EXAMPLES": TRAIN_EXAMPLES,
        "DEVELOPMENT_EXAMPLES": DEVELOPMENT_EXAMPLES,
        "PROTECTED_EXAMPLES": PROTECTED_EXAMPLES,
        "HISTORY": HISTORY,
        "HORIZON": HORIZON,
        "STATE_DIM": STATE_DIM,
        "ACTION_DIM": ACTION_DIM,
        "HIDDEN_DIM": HIDDEN_DIM,
        "UPDATES": UPDATES,
        "BATCH_SIZE": BATCH_SIZE,
        "LEARNING_RATE": LEARNING_RATE,
        "MODEL_NAMES": MODEL_NAMES,
        "BOOTSTRAP_SAMPLES": BOOTSTRAP_SAMPLES,
        "BOOTSTRAP_SEED": BOOTSTRAP_SEED,
        "MAX_WALL_SECONDS": MAX_WALL_SECONDS,
        "WORST_REGIME_H8_RATIO_MAX": WORST_REGIME_H8_RATIO_MAX,
        "PAIRED_DELTA_CI_UPPER_MAX": PAIRED_DELTA_CI_UPPER_MAX,
        "CLEAN_H8_REGRESSION_MAX": CLEAN_H8_REGRESSION_MAX,
        "COVERAGE90_ERROR_MAX": COVERAGE90_ERROR_MAX,
        "PROTOCOL": PROTOCOL,
    }.items():
        setattr(base, name, value)


def source_identity() -> dict[str, str]:
    source = Path(__file__).resolve()
    protocol = (
        source.parents[1]
        / "research/multi_horizon_dynamics_v3_heavy/PROTOCOL.md"
    )
    return {
        "source_sha256": file_sha(source),
        "protocol_sha256": file_sha(protocol),
        "protocol_object_sha256": digest(PROTOCOL),
        "base_source_sha256": BASE_SOURCE_SHA256,
    }


def main(output: Path) -> None:
    source = Path(__file__).resolve()
    base = load_base(source.with_name("run_multi_horizon_dynamics_v2.py"))
    configure(base)
    dynamic: Any = base
    dynamic.source_identity = source_identity
    dynamic.main(output)


if __name__ == "__main__":
    target = Path(
        os.environ.get(
            "MULTI_HORIZON_OUTPUT",
            "/kaggle/working/multi_horizon_dynamics_v3_heavy",
        )
    )
    try:
        main(target)
    except BaseException as error:
        target.mkdir(parents=True, exist_ok=True)
        failure = target / f"FAILURE-{time.time_ns()}.json"
        if not failure.exists():
            failure.write_text(
                json.dumps(
                    {
                        "status": "FAILED_RETAINED",
                        "error": repr(error),
                        "torch_imported": "torch" in sys.modules,
                        "protocol_sha256": digest(PROTOCOL),
                        "base_source_sha256": BASE_SOURCE_SHA256,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        raise

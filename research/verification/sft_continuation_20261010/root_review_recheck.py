"""Independent artificial CPU trajectory and repaired checkpoint admission checks."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import types
from pathlib import Path

import torch

root = Path(__file__).resolve().parents[3]
source = root / "olympus/foundry/sft.py"
source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
# Import the actual Foundry modules without the unrelated optional web-API entrypoint.
namespace = types.ModuleType("olympus")
namespace.__path__ = [str(root / "olympus")]
namespace.__package__ = "olympus"
sys.modules["olympus"] = namespace
sys.path.insert(0, str(root))
from olympus.foundry import sft  # noqa: E402
from olympus.foundry.data_pipeline import (  # noqa: E402
    REQUIRED_CATEGORIES,
    prepare_instruction_dataset,
)


def same(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for first, second in zip(left, right, strict=True):
            same(first, second)
    else:
        assert left == right


checks = []
with tempfile.TemporaryDirectory(prefix="root-sft-recheck-") as directory:
    tmp = Path(directory)
    rows = [
        {
            "id": f"{split}-{category}",
            "category": category,
            "prompt": f"Independent {split} {category} input",
            "response": f"Independent {split} {category} answer",
            "license": "MIT",
            "source": "artificial://root-sft-recheck",
            "split": split,
        }
        for split in ("train", "validation", "test")
        for category in REQUIRED_CATEGORIES
    ]
    raw = tmp / "fixture.jsonl"
    raw.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    prepare_instruction_dataset(raw, tmp / "prepared")
    manifest = tmp / "prepared/manifest.json"
    config = sft.SFTConfig(
        seed=347,
        epochs=1,
        batch_size=2,
        learning_rate=0.0025,
        gradient_accumulation_steps=3,
        pack_sequences=False,
        mixed_precision=False,
        loss_normalization="legacy_batch_mean_v1",
        model=sft.TinyModelConfig(
            width=16, layers=1, heads=2, max_sequence_tokens=128, dropout=0.4
        ),
    )
    extended = config.model_copy(update={"epochs": 2})
    first = sft.run_sft(manifest, tmp / "split", config=config)
    before = Path(first.checkpoint_path).read_bytes()
    full = sft.run_sft(manifest, tmp / "full", config=extended)
    torch.rand(61)
    resumed = sft.run_sft(
        manifest, tmp / "split", config=extended, resume_checkpoint=Path(first.checkpoint_path)
    )
    def load(path):
        return torch.load(path, map_location="cpu", weights_only=True)
    actual, expected = load(resumed.checkpoint_path), load(full.checkpoint_path)
    for key in (
        "model_state",
        "optimizer_state",
        "generator_state",
        "rng_state",
        "validation_loss",
        "optimizer_steps",
        "completed_epochs",
    ):
        same(actual[key], expected[key])
    assert Path(first.checkpoint_path).read_bytes() == before
    full_metrics = Path(full.log_path).read_text().splitlines()
    assert Path(resumed.log_path).read_text().splitlines() == full_metrics[1:]
    checks.append(
        {
            "check": "independent legacy-objective dropout continuation",
            "status": "PASS",
            "scope": "bitwise model/optimizer/RNG/metrics; earlier checkpoint preserved",
        }
    )
    parent = load(first.checkpoint_path)
    for name in (
        "nonfinite_model",
        "fractional_epochs",
        "boolean_steps",
        "epochs_above_saved_schedule",
        "nonfinite_optimizer",
    ):
        changed = copy.deepcopy(parent)
        if name == "nonfinite_model":
            value = next(v for v in changed["model_state"].values() if v.is_floating_point())
            value.reshape(-1)[0] = float("nan")
        elif name == "fractional_epochs":
            changed["completed_epochs"] = 1.5
        elif name == "boolean_steps":
            changed["optimizer_steps"] = True
        elif name == "epochs_above_saved_schedule":
            changed["completed_epochs"] = 2
        else:
            value = next(iter(changed["optimizer_state"]["state"].values()))["exp_avg"]
            value.reshape(-1)[0] = float("inf")
        damaged = tmp / f"{name}.pt"
        torch.save(changed, damaged)
        output = tmp / name
        try:
            sft.run_sft(manifest, output, config=extended, resume_checkpoint=damaged)
        except ValueError as error:
            assert not list(output.rglob("checkpoint.pt"))
            assert not list(output.rglob("attempt.json"))
            checks.append({"check": name, "status": "REJECTED_BEFORE_ATTEMPT", "error": str(error)})
        else:
            raise AssertionError(f"invalid parent was admitted: {name}")

assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
receipt = {
    "reviewer": "root independent review",
    "status": "PASS",
    "source_sha256": source_hash,
    "source_unchanged_during_execution": True,
    "checks": checks,
    "scope": (
        "Real CPU Foundry modules using namespace-only import harness; artificial data; "
        "no API integration, GPU or scientific efficacy claim."
    ),
    "prior_failures": (
        "root_review_findings.json retains the earlier NaN COMPLETED "
        "and fractional epoch admission witnesses."
    ),
}
Path(__file__).with_name("root_review_recheck.json").write_text(
    json.dumps(receipt, indent=2) + "\n"
)
print(json.dumps(receipt, indent=2))

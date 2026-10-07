#!/usr/bin/env python3
"""Reproduce a bounded CPU engineering probe using the repository's smoke fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import torch

from olympus.foundry.data_pipeline import prepare_instruction_dataset
from olympus.foundry.quantization import quantize_checkpoint
from olympus.foundry.sft import SFTConfig, TinyModelConfig, run_sft


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="A new directory; existing paths are rejected")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    source = root / "datasets/hermes-smoke/source.jsonl"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    prepared = output / "prepared"
    manifest = prepare_instruction_dataset(source, prepared)
    config = SFTConfig(
        mode="full",
        seed=31,
        epochs=1,
        batch_size=3,
        learning_rate=0.002,
        gradient_accumulation_steps=2,
        mixed_precision=False,
        model=TinyModelConfig(width=32, layers=1, heads=4, max_sequence_tokens=160),
    )
    training = run_sft(prepared / "manifest.json", output / "training", config=config)
    checkpoint = Path(training.checkpoint_path)
    reports = [
        quantize_checkpoint(
            checkpoint, prepared / "manifest.json", output / f"int{bits}", bits=bits
        )
        for bits in (4, 8)
    ]
    code_files = [
        "olympus/foundry/latency.py",
        "olympus/foundry/quantization.py",
        "olympus/foundry/sft.py",
        "olympus/foundry/data_pipeline.py",
        "scripts/quantization_timing_probe.py",
    ]
    receipt = {
        "schema": "olympus.quantization_timing_probe.v1",
        "purpose": "engineering fixture; quantized storage with dequantized float32 CPU inference",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "source_data_sha256": digest(source),
        "source_data_records": manifest.quality.accepted_records,
        "configuration": config.model_dump(mode="json"),
        "checkpoint_sha256": digest(checkpoint),
        "source_code_sha256": {name: digest(root / name) for name in code_files},
        "reports": [report.model_dump(mode="json") for report in reports],
        "model_promotion_authorized": False,
        "integer_kernel_speedup_claim_authorized": False,
        "scientific_superiority_claim_authorized": False,
    }
    files = sorted(path for path in output.rglob("*") if path.is_file())
    receipt["artifact_sha256"] = {
        path.relative_to(output).as_posix(): digest(path) for path in files
    }
    (output / "probe.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(output / "probe.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

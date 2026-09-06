#!/usr/bin/env python3
"""Run the frozen, resource-bounded multi-seed packing ablation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from olympus.evaluation.comparison import (
    ComparisonProtocol,
    ConditionObservation,
    compare_conditions,
)
from olympus.foundry.data_pipeline import prepare_instruction_dataset
from olympus.foundry.sft import SFTConfig, TinyModelConfig, run_sft


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path, default=Path("datasets/hermes-smoke/source.jsonl")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/stage-execution/research-smoke")
    )
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    manifest = prepare_instruction_dataset(arguments.source.resolve(), output / "dataset")
    manifest_path = output / "dataset/manifest.json"
    seeds = [17, 31, 47]
    protocol = ComparisonProtocol(
        protocol_id="olympus-packing-ablation-v1",
        primary_metric="negative_validation_loss",
        higher_is_better=True,
        minimum_effect=0.01,
        seeds=seeds,
        reference_condition="olympus_reference",
        baseline_condition="simple_sft",
        ablation_conditions=["without_sequence_packing"],
        budget_tolerance_fraction=0.05,
    )
    atomic_json(output / "protocol.json", protocol.model_dump(mode="json"))
    observations: list[ConditionObservation] = []
    conditions = {
        "olympus_reference": True,
        "simple_sft": False,
        "without_sequence_packing": False,
    }
    for condition, pack_sequences in conditions.items():
        for seed in seeds:
            config = SFTConfig(
                seed=seed,
                epochs=2,
                batch_size=3,
                learning_rate=0.002,
                gradient_accumulation_steps=2,
                mixed_precision=False,
                pack_sequences=pack_sequences,
                model=TinyModelConfig(
                    width=32,
                    layers=1,
                    heads=4,
                    max_sequence_tokens=160,
                ),
            )
            summary = run_sft(
                manifest_path,
                output / "runs" / condition / f"seed-{seed}",
                config=config,
            )
            observations.append(
                ConditionObservation(
                    condition=condition,
                    seed=seed,
                    primary_metric=-summary.final_validation_loss,
                    optimizer_steps=summary.optimizer_steps,
                    trainable_parameters=summary.trainable_parameters,
                    elapsed_seconds=summary.elapsed_seconds,
                    peak_rss_bytes=int(summary.final_memory["process_peak_rss_bytes"]),
                )
            )
    report = compare_conditions(protocol, observations)
    atomic_json(
        output / "observations.json",
        {
            "schema_version": 1,
            "dataset_manifest_sha256": manifest.manifest_sha256,
            "observations": [item.model_dump(mode="json") for item in observations],
        },
    )
    atomic_json(output / "comparison.json", report.model_dump(mode="json"))
    print(report.model_dump_json(indent=2))
    if not report.passed_integrity_gate:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

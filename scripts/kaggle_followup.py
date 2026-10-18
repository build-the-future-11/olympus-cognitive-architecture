"""Frozen synthetic follow-up: fresh evaluation, replication and fixed-gate ablation."""

from __future__ import annotations

import argparse
import copy
import json
import random
import statistics
from pathlib import Path
from typing import Any

from olympus.training.campaign import _admission, analyze, execute_campaign
from olympus.training.provenance import atomic_json, file_sha256

PROTOCOL: dict[str, Any] = {
    "id": "olympus-followup-20261017-v1",
    "purpose": "synthetic development; no scientific promotion",
    "original_recovery_sha256": "db8fe659bc3b68b2c897523e40a6d7741c1dedf0d11ca04c4aee2a000124012b",
    "fresh_test": {"seed": 2026101731, "examples": 2048},
    "dataset_seeds": [2026101740, 2026101750, 2026101760, 2026101770, 2026101780],
    "training_seeds": [2026101721, 2026101722, 2026101723],
    "splits": {"train": 512, "development": 128, "test": 1024},
    "steps": 300,
    "primary_metric": "test total_loss; paired learned minus uniform",
    "secondary_metrics": [
        "loss_components",
        "next_state_mse",
        "decision_accuracy",
        "transition_accuracy",
        "risk.intrinsic_brier",
        "risk.intrinsic_ece",
    ],
    "selection": "minimum development loss; test never selects checkpoints",
    "ablation": "original learned checkpoints; fixed training-mean gates; no retraining",
    "bootstrap": {
        "unit": "dataset mean of three paired training seeds",
        "samples": 10000,
        "seed": 2026101790,
        "interval": "percentile 95%; descriptive with only five datasets",
    },
}


def bind_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Frozen input differs: {path}")
    else:
        atomic_json(path, value)


def generate(path: Path, *, name: str, examples: int, seed: int) -> tuple[Any, list[str]]:
    import torch

    from olympus.data.controlled_fusion import ControlledSplitSpec, generate_controlled_split

    spec = ControlledSplitSpec(name=name, examples=examples, seed=seed, regime="mixed")
    bind_json(path.with_suffix(".spec.json"), spec.model_dump(mode="json"))
    if path.exists():
        receipt = json.loads(path.with_suffix(".sha.json").read_text())
        if file_sha256(path) != receipt["sha256"]:
            raise ValueError("Generated data checksum mismatch")
        return torch.load(path, weights_only=True), receipt["ids"]
    batch, ids = generate_controlled_split(spec)
    torch.save(batch, path)
    atomic_json(path.with_suffix(".sha.json"), {"sha256": file_sha256(path), "ids": ids})
    return batch, ids


def evaluate_checkpoint(run: Path, batch: Any, *, train: Any | None = None) -> dict[str, Any]:
    import torch

    from olympus.evaluation.olympus_system import evaluate_olympus_system
    from olympus.models.olympus_system import OlympusSystemLossConfig
    from olympus.models.runtime import load_olympus_checkpoint

    report = json.loads((run / "report.json").read_text())
    model, _ = load_olympus_checkpoint(
        run / "checkpoint.pt", expected_sha256=report["checkpoint_sha256"]
    )
    model = model.to("cuda")
    loss = OlympusSystemLossConfig.model_validate(report["training_config"]["loss"])

    def evaluate(data: Any) -> dict[str, Any]:
        return evaluate_olympus_system(
            model, data, model.extension_names, loss_config=loss, batch_size=32
        ).model_dump(mode="json")

    normal = evaluate(batch)
    result = {"checkpoint_sha256": report["checkpoint_sha256"], "normal": normal}
    if train is not None:
        if model.config.fusion_strategy != "learned":
            raise ValueError("Fixed-gate ablation requires learned routing")
        means = evaluate(train)["routing"]["mean_gates"]
        gates = torch.tensor(list(means.values()), device="cuda")
        gates = gates / gates.sum()
        logits = gates.log() * model.config.gate_temperature
        handle = model.gate_router.register_forward_hook(
            lambda module, inputs, output: logits.to(output).expand_as(output)
        )
        try:
            fixed = evaluate(batch)
        finally:
            handle.remove()
        for observed, expected in zip(
            fixed["routing"]["mean_gates"].values(), gates.tolist(), strict=True
        ):
            if abs(observed - expected) > 1e-6:
                raise AssertionError("Fixed gates were not applied")
        restored = evaluate(batch)
        if restored != normal:
            raise AssertionError("Fixed-gate hook changed the original model")
        result.update(training_mean_gates=means, fixed=fixed, restored_normal_matches=True)
    return result


def run(root: Path, original: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    bind_json(root / "protocol.json", PROTOCOL)
    # Validate original artifact digests without changing the original results.
    old = analyze(original / "development-cuda/runs")
    assert len(old["cells"]) == 6 and all(c["completed_steps"] == 300 for c in old["cells"])
    with _admission(root / "evaluation-lock"):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; no fallback")
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        fresh, _ = generate(
            root / "fresh-test.pt", name="followup_fresh_test", **PROTOCOL["fresh_test"]
        )
        train = torch.load(original / "development-cuda/data/train.pt", weights_only=True)
        fresh_results = {}
        for cell in old["cells"]:
            cell_id = cell["id"]
            fresh_results[cell_id] = evaluate_checkpoint(
                original / "development-cuda/runs/cells" / cell_id,
                fresh,
                train=train if cell_id.startswith("learned-") else None,
            )
            atomic_json(root / "fresh-and-ablation.json", fresh_results)
            print("FRESH_AND_ABLATION_COMPLETED", cell_id, flush=True)

    template = json.loads(
        (Path(__file__).resolve().parents[1] / "configs/kaggle/development-cuda.json").read_text()
    )
    matrix = copy.deepcopy(template)
    matrix["campaign_id"] = PROTOCOL["id"]
    matrix["cells"] = []
    with _admission(root / "data-lock"):
        from olympus.data.olympus_dataset import write_olympus_dataset_manifest

        for seed in PROTOCOL["dataset_seeds"]:
            data = root / "data" / str(seed)
            data.mkdir(parents=True, exist_ok=True)
            ids = {}
            for offset, (split, count) in enumerate(PROTOCOL["splits"].items()):
                _, ids[split] = generate(
                    data / f"{split}.pt",
                    name=f"followup_{seed}_{split}",
                    examples=count,
                    seed=seed + offset,
                )
            manifest = data / "manifest.json"
            if not manifest.exists():
                write_olympus_dataset_manifest(
                    manifest,
                    dataset_id=f"followup-{seed}",
                    version="1.0.0",
                    license_name="CC-BY-4.0",
                    train_batch=data / "train.pt",
                    evaluation_batch=data / "development.pt",
                    train_source_ids=ids["train"],
                    evaluation_source_ids=ids["development"],
                )
            for training_seed in PROTOCOL["training_seeds"]:
                for mode in ("learned", "uniform"):
                    cell = copy.deepcopy(template["cells"][0])
                    cell["id"] = f"d{seed}-{mode}-{training_seed}"
                    cell["dataset"] = f"data/{seed}/manifest.json"
                    cell["training_config"]["seed"] = training_seed
                    cell["training_config"]["model"]["fusion_strategy"] = mode
                    matrix["cells"].append(cell)
    bind_json(root / "campaign.json", matrix)
    execute_campaign(root / "campaign.json", root / "runs")
    analysis = analyze(root / "runs")
    assert len(analysis["cells"]) == 30 and all(
        c.get("completed_steps") == 300 for c in analysis["cells"]
    )
    atomic_json(root / "replication-analysis.json", analysis)
    test_results = {}
    with _admission(root / "evaluation-lock"):
        for cell in matrix["cells"]:
            seed = cell["id"].split("-")[0][1:]
            batch = torch.load(root / "data" / seed / "test.pt", weights_only=True)
            test_results[cell["id"]] = evaluate_checkpoint(root / "runs/cells" / cell["id"], batch)
            atomic_json(root / "replication-test.json", test_results)
            print("REPLICATION_TEST_COMPLETED", cell["id"], flush=True)
    pairs, dataset_means = [], []
    for seed in PROTOCOL["dataset_seeds"]:
        deltas = []
        for training_seed in PROTOCOL["training_seeds"]:
            learned = test_results[f"d{seed}-learned-{training_seed}"]["normal"]["total_loss"]
            uniform = test_results[f"d{seed}-uniform-{training_seed}"]["normal"]["total_loss"]
            deltas.append(learned - uniform)
            pairs.append(
                {
                    "dataset_seed": seed,
                    "training_seed": training_seed,
                    "learned": learned,
                    "uniform": uniform,
                    "delta": learned - uniform,
                }
            )
        dataset_means.append(statistics.mean(deltas))
    rng = random.Random(PROTOCOL["bootstrap"]["seed"])
    boot = sorted(statistics.mean(rng.choices(dataset_means, k=5)) for _ in range(10000))
    atomic_json(
        root / "summary.json",
        {
            "status": "ALL_THREE_EXPERIMENTS_COMPLETED",
            "promotion_authorized": False,
            "fresh_checkpoints": len(fresh_results),
            "fixed_gate_ablations": 3,
            "replication_fits": len(analysis["cells"]),
            "test_pairs": pairs,
            "dataset_mean_deltas": dataset_means,
            "mean_delta": statistics.mean(dataset_means),
            "descriptive_dataset_bootstrap_95": [boot[250], boot[9749]],
            "learned_test_wins": sum(p["delta"] < 0 for p in pairs),
        },
    )
    print("ALL_THREE_EXPERIMENTS_COMPLETED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.original.resolve())

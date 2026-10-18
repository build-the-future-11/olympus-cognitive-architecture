"""Frozen EditCLEVR paired-image development and protected evaluation.

The module performs resource admission before importing Torch. It evaluates a
narrow image-edit classifier and does not claim to test the full Olympus system.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from scripts import run_oracle_fusion_v4 as evidence

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research/olympus_editclevr_v1"
PROTOCOL: dict[str, Any] = {
    "id": "olympus-editclevr-v1.0.0",
    "dataset": "torux/EditCLEVR",
    "dataset_revision": "734efa9b0164ba742cd2c39b9c2945c494497c23",
    "license": "CC-BY-4.0",
    "task": "classify the edited object factor from paired RGB images",
    "classes": ["none", "color", "material", "size", "shape"],
    "train_split": "train",
    "development_split": "val",
    "protected_splits": ["test_id", "test_hard", "test_cogent", "test_noop"],
    "training_seeds": [20266101, 20266102, 20266103, 20266104, 20266105],
    "epochs": 15,
    "batch_size": 64,
    "learning_rate": 0.0005,
    "weight_decay": 0.01,
    "image_size": [96, 128],
    "derived_noop_fraction": 0.25,
    "selection": "maximum development learned accuracy; test is never consulted",
    "control_selection": "best of pair_only, uniform, delta_only on development accuracy",
    "primary": (
        "learned minus development-selected control on mean test_hard/test_cogent accuracy; "
        ">=2pp and descriptive seed-bootstrap CI lower >0"
    ),
    "id_nonregression": "test_id learned-control mean >= -1pp and CI lower >= -2pp",
    "bootstrap_samples": 10000,
    "bootstrap_seed": 20266999,
    "minimum_available_bytes": 2147483648,
    "maximum_swap_fraction": 0.90,
    "max_wall_seconds": 7200,
    "promotion_authorized": False,
}

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def sources() -> dict[str, str]:
    paths = [
        ROOT / "olympus/data/editclevr.py",
        ROOT / "olympus/models/editclevr.py",
        Path(__file__),
        ROOT / "tests/test_editclevr_model.py",
        ROOT / "tests/test_editclevr_dataset.py",
        ROOT / "scripts/verify_editclevr_v1.py",
        RESEARCH / "PROTOCOL.md",
    ]
    return {str(path.relative_to(ROOT)): evidence.sha(path) for path in paths}


def _paths() -> tuple[Path, Path]:
    base = ROOT / "datasets/external/editclevr"
    return base / "payload-734efa9", base / "splits.json"


def _ids_sha256(values: list[str]) -> str:
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _resource_check() -> dict[str, Any]:
    snapshot = evidence.resource_check()
    if snapshot["available_bytes"] < PROTOCOL["minimum_available_bytes"]:
        raise RuntimeError("EditCLEVR 2 GiB admission floor rejected")
    if snapshot["swap_fraction"] > PROTOCOL["maximum_swap_fraction"]:
        raise RuntimeError("EditCLEVR 90 percent swap ceiling rejected")
    return snapshot


def _dataset(split: str, *, include_derived_noop: bool) -> Any:
    import torch
    from torch.nn import functional as F
    from torch.utils.data import Dataset

    from olympus.data.editclevr import EditCLEVRDataset
    from olympus.models.editclevr import EDIT_FACTORS

    payload, registry = _paths()
    base = EditCLEVRDataset(payload, registry, split, verify_files=False)
    class Wrapped(Dataset[Any]):
        def __len__(self) -> int:
            extra = round(len(base) * PROTOCOL["derived_noop_fraction"])
            return len(base) + (extra if include_derived_noop else 0)

        def __getitem__(self, index: int) -> tuple[Any, Any, int, str]:
            derived = index >= len(base)
            source_index = index % len(base)
            before, after = base.load_rgb_pair(source_index)
            if derived:
                after = before.clone()
                factor = "none"
                pair_id = base[source_index].pair.pair_id + ":derived-noop"
            else:
                factor = base[source_index].pair.edit_factor
                pair_id = base[source_index].pair.pair_id
            size = tuple(PROTOCOL["image_size"])
            images = F.interpolate(
                torch.stack((before, after)), size=size, mode="bilinear", align_corners=False
            )
            images = (images - 0.5) / 0.5
            return images[0], images[1], EDIT_FACTORS.index(factor), pair_id

    return Wrapped()


def _loader(dataset: Any, *, shuffle: bool, seed: int) -> Any:
    import torch
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=PROTOCOL["batch_size"],
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        num_workers=2,
        persistent_workers=True,
        pin_memory=torch.cuda.is_available(),
    )


def _metrics(probabilities: Any, labels: Any) -> dict[str, Any]:
    import torch

    from olympus.models.editclevr import EDIT_FACTORS

    predicted = probabilities.argmax(-1)
    per_class = {}
    for index, name in enumerate(EDIT_FACTORS):
        mask = labels == index
        per_class[name] = (
            None if not bool(mask.any()) else float(predicted[mask].eq(index).float().mean())
        )
    confidence, _ = probabilities.max(-1)
    correct = predicted.eq(labels).float()
    ece = torch.zeros((), dtype=torch.float64)
    for low in torch.linspace(0, 0.9, 10):
        mask = (confidence >= low) & (confidence < low + 0.1)
        if bool(mask.any()):
            ece += mask.double().mean() * (correct[mask].mean() - confidence[mask].mean()).abs()
    return {
        "accuracy": float(correct.mean()),
        "nll": float(
            -probabilities[torch.arange(len(labels)), labels].clamp_min(1e-8).log().mean()
        ),
        "ece": float(ece),
        "per_class_accuracy": per_class,
        "examples": int(len(labels)),
    }


def _evaluate(model: Any, loader: Any, device: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    from olympus.models.editclevr import FUSION_MODES

    model.eval()
    predictions: dict[str, list[Any]] = {mode: [] for mode in FUSION_MODES}
    labels: list[Any] = []
    pair_ids: list[str] = []
    gates: list[Any] = []
    with torch.no_grad():
        for before, after, target, ids in loader:
            output = model(before.to(device), after.to(device))
            for mode in FUSION_MODES:
                predictions[mode].append(output.probabilities[mode].cpu())
            labels.append(target)
            gates.append(output.gates.cpu())
            pair_ids.extend(ids)
    joined_labels = torch.cat(labels)
    joined = {mode: torch.cat(values) for mode, values in predictions.items()}
    report: dict[str, Any] = {
        mode: _metrics(values, joined_labels) for mode, values in joined.items()
    }
    report["mean_gates"] = torch.cat(gates).mean(0).tolist()
    archive = {"probabilities": joined, "labels": joined_labels, "pair_ids": pair_ids}
    return report, archive


def develop(output: Path, deadline: float) -> None:
    import torch

    from olympus.models.editclevr import OlympusEditCLEVRModel, editclevr_loss

    if not torch.cuda.is_available():
        raise RuntimeError("EditCLEVR development requires CUDA; CPU fallback is disabled")
    device = torch.device("cuda")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    evidence.write(
        output / "CONTRACT.json",
        {
            "protocol": PROTOCOL,
            "sources": sources(),
            "torch": torch.__version__,
            "python": sys.version,
        },
    )
    train = _dataset("train", include_derived_noop=True)
    development = _dataset("val", include_derived_noop=True)
    for seed in PROTOCOL["training_seeds"]:
        _resource_check()
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = OlympusEditCLEVRModel().to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=PROTOCOL["learning_rate"], weight_decay=PROTOCOL["weight_decay"]
        )
        best_accuracy, best_epoch = -1.0, -1
        checkpoint = output / "development" / f"seed-{seed}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        history = []
        for epoch in range(PROTOCOL["epochs"]):
            model.train()
            total, batches = 0.0, 0
            for before, after, labels, _ids in _loader(train, shuffle=True, seed=seed + epoch):
                result = model(before.to(device), after.to(device))
                loss = editclevr_loss(result, labels.to(device))
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite EditCLEVR loss")
                optimizer.zero_grad(set_to_none=True)
                torch.autograd.backward(loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total += float(loss.detach())
                batches += 1
            report, _archive = _evaluate(
                model, _loader(development, shuffle=False, seed=seed), device
            )
            accuracy = report["learned"]["accuracy"]
            history.append({"epoch": epoch + 1, "loss": total / batches, "development": report})
            if accuracy > best_accuracy:
                best_accuracy, best_epoch = accuracy, epoch + 1
                with checkpoint.open("wb") as handle:
                    torch.save({"model": model.state_dict(), "epoch": best_epoch}, handle)
            if time.monotonic() > deadline:
                raise RuntimeError("EditCLEVR development wall budget exhausted")
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(payload["model"])
        report, archive = _evaluate(model, _loader(development, shuffle=False, seed=seed), device)
        selected_control = max(
            ("pair_only", "uniform", "delta_only"), key=lambda mode: report[mode]["accuracy"]
        )
        prediction_path = output / "development" / f"seed-{seed}-predictions.pt"
        with prediction_path.open("wb") as handle:
            torch.save(archive, handle)
        evidence.write(
            output / "development" / f"seed-{seed}.json",
            {
                "seed": seed,
                "best_epoch": best_epoch,
                "selected_control": selected_control,
                "report": report,
                "history": history,
                "checkpoint_sha256": evidence.sha(checkpoint),
                "predictions_sha256": evidence.sha(prediction_path),
                "pair_ids_sha256": _ids_sha256(archive["pair_ids"]),
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
            },
        )


def freeze(output: Path) -> None:
    contract = evidence.read(output / "CONTRACT.json")
    if contract["sources"] != sources() or contract["protocol"] != PROTOCOL:
        raise RuntimeError("EditCLEVR source or protocol changed before freeze")
    bindings = {}
    for seed in PROTOCOL["training_seeds"]:
        for suffix in (".pt", "-predictions.pt", ".json"):
            path = output / "development" / f"seed-{seed}{suffix}"
            bindings[str(path.relative_to(output))] = evidence.sha(path)
    evidence.write(
        output / "SCIENTIFIC_FREEZE.json",
        {
            "time": time.time(),
            "contract_sha256": evidence.sha(output / "CONTRACT.json"),
            "sources": sources(),
            "bindings": bindings,
            "protected_evaluation_started": False,
        },
    )


def confirm(output: Path) -> None:
    import torch

    from olympus.models.editclevr import OlympusEditCLEVRModel

    frozen = evidence.read(output / "SCIENTIFIC_FREEZE.json")
    if frozen["sources"] != sources():
        raise RuntimeError("EditCLEVR source changed after freeze")
    for path, digest in frozen["bindings"].items():
        if evidence.sha(output / path) != digest:
            raise RuntimeError(f"EditCLEVR frozen binding changed: {path}")
    device = torch.device("cuda")
    protected = {
        split: _dataset(split, include_derived_noop=False) for split in PROTOCOL["protected_splits"]
    }
    for seed in PROTOCOL["training_seeds"]:
        _resource_check()
        model = OlympusEditCLEVRModel().to(device)
        checkpoint = torch.load(
            output / "development" / f"seed-{seed}.pt", map_location=device, weights_only=True
        )
        model.load_state_dict(checkpoint["model"])
        reports, prediction_hashes, pair_id_hashes = {}, {}, {}
        for split, dataset in protected.items():
            report, archive = _evaluate(model, _loader(dataset, shuffle=False, seed=seed), device)
            path = output / "protected" / f"seed-{seed}-{split}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                torch.save(archive, handle)
            reports[split] = report
            prediction_hashes[split] = evidence.sha(path)
            pair_id_hashes[split] = _ids_sha256(archive["pair_ids"])
        development = evidence.read(output / "development" / f"seed-{seed}.json")
        evidence.write(
            output / "protected" / f"seed-{seed}.json",
            {
                "seed": seed,
                "selected_control": development["selected_control"],
                "reports": reports,
                "prediction_hashes": prediction_hashes,
                "pair_ids_sha256": pair_id_hashes,
                "time": time.time(),
            },
        )


def _interval(values: list[float]) -> list[float]:
    rng = random.Random(PROTOCOL["bootstrap_seed"])
    samples = sorted(
        statistics.mean(rng.choices(values, k=len(values)))
        for _ in range(PROTOCOL["bootstrap_samples"])
    )
    return [samples[250], samples[9749]]


def summarize(output: Path) -> dict[str, Any]:
    rows = [
        evidence.read(output / "protected" / f"seed-{seed}.json")
        for seed in PROTOCOL["training_seeds"]
    ]
    hard_deltas, id_deltas = [], []
    for row in rows:
        control = row["selected_control"]
        shifted = statistics.mean(
            row["reports"][split]["learned"]["accuracy"]
            - row["reports"][split][control]["accuracy"]
            for split in ("test_hard", "test_cogent")
        )
        hard_deltas.append(shifted)
        id_deltas.append(
            row["reports"]["test_id"]["learned"]["accuracy"]
            - row["reports"]["test_id"][control]["accuracy"]
        )
    shifted_ci, id_ci = _interval(hard_deltas), _interval(id_deltas)
    shifted_mean, id_mean = statistics.mean(hard_deltas), statistics.mean(id_deltas)
    supported = (
        shifted_mean >= 0.02
        and shifted_ci[0] > 0
        and id_mean >= -0.01
        and id_ci[0] >= -0.02
    )
    summary = {
        "protocol": PROTOCOL,
        "seeds": len(rows),
        "protected_reports": len(rows) * len(PROTOCOL["protected_splits"]),
        "shifted_learned_minus_control": {
            "mean": shifted_mean,
            "ci95": shifted_ci,
            "seed_deltas": hard_deltas,
        },
        "id_learned_minus_control": {"mean": id_mean, "ci95": id_ci, "seed_deltas": id_deltas},
        "primary_supported": supported,
        "promotion_authorized": False,
        "limitation": (
            "One synthetic public image benchmark and five optimizer seeds; no independent "
            "dataset, temporal claim, full Olympus claim, or independent replication"
        ),
    }
    evidence.write(output / "SUMMARY.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/editclevr-v1"))
    parser.add_argument("command", choices=("develop", "freeze", "confirm", "summarize", "run"))
    args = parser.parse_args()
    output = args.output.resolve()
    _resource_check()
    deadline = time.monotonic() + PROTOCOL["max_wall_seconds"]
    if args.command in {"develop", "run"}:
        output.mkdir(parents=True, exist_ok=False)
        develop(output, deadline)
    if args.command in {"freeze", "run"}:
        freeze(output)
    if args.command in {"confirm", "run"}:
        confirm(output)
    if args.command in {"summarize", "run"}:
        print(json.dumps(summarize(output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

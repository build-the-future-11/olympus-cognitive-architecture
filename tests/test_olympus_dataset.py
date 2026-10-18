from __future__ import annotations

from pathlib import Path

import pytest
import torch

from olympus.data.olympus_dataset import (
    load_olympus_dataset,
    write_olympus_dataset_manifest,
)


def _batch(batch: int = 2) -> dict[str, torch.Tensor]:
    return {
        "tokens": torch.ones((batch, 4), dtype=torch.long),
        "attention_mask": torch.ones((batch, 4), dtype=torch.bool),
        "image": torch.rand(batch, 3, 8, 8),
        "states": torch.rand(batch, 4, 5),
        "actions": torch.rand(batch, 4, 3),
        "temporal_mask": torch.ones((batch, 4), dtype=torch.bool),
        "allowed_transitions": torch.ones((batch, 4), dtype=torch.bool),
        "risk_score": torch.rand(batch),
        "decision_targets": torch.zeros(batch, dtype=torch.long),
        "transition_targets": torch.zeros(batch, dtype=torch.long),
        "next_state_targets": torch.rand(batch, 4, 5),
        "intrinsic_risk_targets": torch.rand(batch),
        "extension__sensor": torch.rand(batch, 7),
    }


def test_olympus_dataset_is_content_addressed_and_source_disjoint(tmp_path: Path) -> None:
    train = tmp_path / "train.pt"
    evaluation = tmp_path / "evaluation.pt"
    torch.save(_batch(), train)
    torch.save(_batch(), evaluation)
    manifest_path = tmp_path / "manifest.json"
    manifest = write_olympus_dataset_manifest(
        manifest_path,
        dataset_id="olympus-test",
        version="1.0.0",
        license_name="LicenseRef-Proprietary",
        train_batch=train,
        evaluation_batch=evaluation,
        train_source_ids=["train-0", "train-1"],
        evaluation_source_ids=["evaluation-0", "evaluation-1"],
        extensions=["sensor"],
    )
    loaded = load_olympus_dataset(manifest_path)
    assert loaded.manifest == manifest
    assert loaded.train["image"].shape == (2, 3, 8, 8)
    assert loaded.evaluation["extension__sensor"].shape == (2, 7)

    evaluation.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="digest changed"):
        load_olympus_dataset(manifest_path)


def test_olympus_dataset_rejects_missing_fields_and_source_overlap(tmp_path: Path) -> None:
    train = tmp_path / "train.pt"
    evaluation = tmp_path / "evaluation.pt"
    incomplete = _batch()
    del incomplete["image"]
    torch.save(incomplete, train)
    torch.save(_batch(), evaluation)
    with pytest.raises(ValueError, match=r"missing=\['image'\]"):
        write_olympus_dataset_manifest(
            tmp_path / "missing.json",
            dataset_id="olympus-test",
            version="1.0.0",
            license_name="LicenseRef-Proprietary",
            train_batch=train,
            evaluation_batch=evaluation,
            train_source_ids=["train-0", "train-1"],
            evaluation_source_ids=["evaluation-0", "evaluation-1"],
            extensions=["sensor"],
        )


def test_olympus_dataset_rejects_forbidden_transition_target(tmp_path: Path) -> None:
    train = tmp_path / "train.pt"
    evaluation = tmp_path / "evaluation.pt"
    invalid = _batch()
    invalid["allowed_transitions"][0, 0] = False
    torch.save(invalid, train)
    torch.save(_batch(), evaluation)
    with pytest.raises(ValueError, match="must be allowed"):
        write_olympus_dataset_manifest(
            tmp_path / "manifest.json",
            dataset_id="olympus-test",
            version="1.0.0",
            license_name="LicenseRef-Proprietary",
            train_batch=train,
            evaluation_batch=evaluation,
            train_source_ids=["train-0", "train-1"],
            evaluation_source_ids=["evaluation-0", "evaluation-1"],
            extensions=["sensor"],
        )

    torch.save(_batch(), train)
    with pytest.raises(ValueError, match="source identities must be disjoint"):
        write_olympus_dataset_manifest(
            tmp_path / "overlap.json",
            dataset_id="olympus-test",
            version="1.0.0",
            license_name="LicenseRef-Proprietary",
            train_batch=train,
            evaluation_batch=evaluation,
            train_source_ids=["shared", "train-1"],
            evaluation_source_ids=["shared", "evaluation-1"],
            extensions=["sensor"],
        )

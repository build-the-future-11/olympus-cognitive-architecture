from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from olympus.data.controlled_fusion import (
    ControlledSplitSpec,
    generate_controlled_fusion_dataset,
    generate_controlled_split,
)
from olympus.data.olympus_dataset import load_olympus_dataset


def test_controlled_split_is_deterministic_and_regime_sensitive() -> None:
    spec = ControlledSplitSpec(name="tiny", examples=12, seed=7, regime="corrupted")
    first, first_ids = generate_controlled_split(spec)
    second, second_ids = generate_controlled_split(spec)
    assert first_ids == second_ids
    assert all(torch.equal(first[name], second[name]) for name in first)
    assert torch.equal(first["intrinsic_risk_targets"], torch.ones(12))
    assert torch.allclose(first["risk_score"], torch.full((12,), 0.35))
    assert bool(
        first["allowed_transitions"]
        .gather(1, first["transition_targets"].unsqueeze(1))
        .all()
    )


def test_controlled_dataset_materialization_is_hash_bound_and_immutable(tmp_path: Path) -> None:
    root = tmp_path / "controlled"
    receipt = generate_controlled_fusion_dataset(root)
    assert receipt.source_overlap_count == 0
    assert set(receipt.bundle_sha256) == {
        "train",
        "development",
        "clean_test",
        "corrupted_test",
        "ood_test",
    }
    loaded = load_olympus_dataset(root / "manifest_development.json")
    assert loaded.train["tokens"].shape == (512, 8)
    assert loaded.evaluation["tokens"].shape == (128, 8)
    raw_receipt = json.loads((root / "receipt.json").read_text(encoding="utf-8"))
    assert raw_receipt["protected_splits"] == ["clean_test", "corrupted_test", "ood_test"]
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        generate_controlled_fusion_dataset(root)

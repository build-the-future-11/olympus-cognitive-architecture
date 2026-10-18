from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from olympus.data.editclevr import EditCLEVRDataset, EditCLEVRPair


def _materialized_pair(root: Path) -> dict[str, object]:
    base = Path("atomic_id/case")
    relative = {
        "before_image": base / "before.png",
        "after_image": base / "after.png",
        "before_scene_json": base / "before.json",
        "after_scene_json": base / "after.json",
        "instance_masks_before": base / "before.npz",
        "instance_masks_after": base / "after.npz",
    }
    for name in ("before_image", "after_image"):
        path = root / relative[name]
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (3, 2), color=(10, 20, 30)).save(path)
    for name in ("before_scene_json", "after_scene_json"):
        path = root / relative[name]
        path.write_text(json.dumps({"objects": [{"id": 0}]}), encoding="utf-8")
    for name in ("instance_masks_before", "instance_masks_after"):
        np.savez(root / relative[name], masks=np.ones((1, 2, 3), dtype=np.uint8))
    return {
        "pair_id": "pair-0",
        "suite": "atomic_id",
        "edit_factor": "color",
        **{name: path.as_posix() for name, path in relative.items()},
    }


def test_editclevr_dataset_streams_and_decodes_materialized_pairs(tmp_path: Path) -> None:
    payload = tmp_path / "payload"
    payload.mkdir()
    record = _materialized_pair(payload)
    registry = tmp_path / "splits.json"
    registry.write_text(json.dumps({"train": [record], "test_id": [record]}), encoding="utf-8")

    dataset = EditCLEVRDataset(payload, registry, "train")

    assert len(dataset) == 1
    assert dataset[0].pair.pair_id == "pair-0"
    before, after = dataset.load_rgb_pair(0)
    assert before.shape == after.shape == (3, 2, 3)
    assert before.dtype is torch.float32
    assert bool((before >= 0).all() and (before <= 1).all())
    scenes = dataset.load_scenes(0)
    assert scenes[0]["objects"][0]["id"] == 0
    masks = dataset.load_masks(0)
    assert masks[0]["masks"].shape == (1, 2, 3)


def test_editclevr_dataset_rejects_unknown_split_missing_files_and_unsafe_paths(
    tmp_path: Path,
) -> None:
    payload = tmp_path / "payload"
    payload.mkdir()
    record = _materialized_pair(payload)
    registry = tmp_path / "splits.json"
    registry.write_text(json.dumps({"train": [record]}), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown EditCLEVR split"):
        EditCLEVRDataset(payload, registry, "validation")

    (payload / str(record["before_image"])).unlink()
    with pytest.raises(ValueError, match="missing or escapes"):
        EditCLEVRDataset(payload, registry, "train")

    invalid = dict(record)
    invalid["before_image"] = "../escape.png"
    with pytest.raises(ValueError, match="unsafe EditCLEVR path"):
        EditCLEVRPair.model_validate(invalid)

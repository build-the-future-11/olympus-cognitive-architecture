from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from scripts.build_multi_horizon_dynamics_notebook import build
from scripts.build_multi_horizon_dynamics_v2_notebook import build as build_v2
from scripts.build_multi_horizon_dynamics_v3_heavy_notebook import build as build_v3
from scripts.run_multi_horizon_dynamics_v1 import PROTOCOL, digest
from scripts.run_multi_horizon_dynamics_v2 import (
    PROTOCOL as PROTOCOL_V2,
)
from scripts.run_multi_horizon_dynamics_v2 import (
    internet_preflight,
)
from scripts.run_multi_horizon_dynamics_v3_heavy import PROTOCOL as PROTOCOL_V3

ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "scripts/run_multi_horizon_dynamics_v1.py"
PROTOCOL_PATH = ROOT / "research/multi_horizon_dynamics_v1/PROTOCOL.md"
SOURCE_V2 = ROOT / "scripts/run_multi_horizon_dynamics_v2.py"
PROTOCOL_PATH_V2 = ROOT / "research/multi_horizon_dynamics_v2/PROTOCOL.md"
SOURCE_V3 = ROOT / "scripts/run_multi_horizon_dynamics_v3_heavy.py"
PROTOCOL_PATH_V3 = ROOT / "research/multi_horizon_dynamics_v3_heavy/PROTOCOL.md"


def _integers(value: object) -> tuple[int, ...]:
    assert isinstance(value, (list, dict))
    values = value.values() if isinstance(value, dict) else value
    integers: list[int] = []
    for item in values:
        assert isinstance(item, int) and not isinstance(item, bool)
        integers.append(item)
    return tuple(integers)


def test_frozen_protocol_has_disjoint_data_seeds_and_finite_budget() -> None:
    training = {
        seed + offset
        for seed in _integers(PROTOCOL["dataset_seeds"])
        for offset in (1, 2)
    }
    protected = {
        seed + offset
        for seed in _integers(PROTOCOL["dataset_seeds"])
        for offset in _integers(PROTOCOL["protected_offsets"])
    }

    assert training.isdisjoint(protected)
    assert len(_integers(PROTOCOL["dataset_seeds"])) == 3
    assert len(_integers(PROTOCOL["optimizer_seeds"])) == 2
    assert PROTOCOL["max_wall_seconds"] == 3_600
    assert PROTOCOL["paid_compute"] is False
    assert PROTOCOL["promotion_authorized"] is False
    assert len(digest(PROTOCOL)) == 64


def test_notebook_builder_embeds_exact_frozen_inputs(tmp_path: Path) -> None:
    notebook_path = tmp_path / "campaign.ipynb"
    build(SOURCE, PROTOCOL_PATH, notebook_path)
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cell = "".join(notebook["cells"][1]["source"])

    assert repr(SOURCE.read_text(encoding="utf-8")) in cell
    assert repr(PROTOCOL_PATH.read_text(encoding="utf-8")) in cell
    assert "runpy.run_path" in cell
    assert "KRONOS" not in cell


def test_prospective_artifacts_do_not_claim_family_identity() -> None:
    assert "Kronos" not in SOURCE.read_text(encoding="utf-8")
    assert "Kronos" not in PROTOCOL_PATH.read_text(encoding="utf-8")


def test_v2_uses_fresh_seeds_and_fails_closed_when_internet_is_enabled(
    tmp_path: Path,
) -> None:
    assert set(_integers(PROTOCOL["dataset_seeds"])).isdisjoint(
        _integers(PROTOCOL_V2["dataset_seeds"])
    )
    assert set(_integers(PROTOCOL["optimizer_seeds"])).isdisjoint(
        _integers(PROTOCOL_V2["optimizer_seeds"])
    )
    with patch(
        "scripts.run_multi_horizon_dynamics_v2.socket.create_connection",
        side_effect=OSError("blocked"),
    ):
        assert internet_preflight()["internet_disabled"] is True
    connection = Mock()
    with (
        patch(
            "scripts.run_multi_horizon_dynamics_v2.socket.create_connection",
            return_value=connection,
        ),
        pytest.raises(RuntimeError, match="Internet is enabled"),
    ):
        internet_preflight()
    connection.close.assert_called_once_with()

    notebook_path = tmp_path / "campaign-v2.ipynb"
    build_v2(SOURCE_V2, PROTOCOL_PATH_V2, notebook_path)
    cell = "".join(json.loads(notebook_path.read_text())["cells"][1]["source"])
    assert repr(SOURCE_V2.read_text(encoding="utf-8")) in cell
    assert repr(PROTOCOL_PATH_V2.read_text(encoding="utf-8")) in cell


def test_v3_heavy_freeze_is_disjoint_scaled_and_exactly_embedded(
    tmp_path: Path,
) -> None:
    prior_dataset_seeds = {
        *_integers(PROTOCOL["dataset_seeds"]),
        *_integers(PROTOCOL_V2["dataset_seeds"]),
    }
    prior_optimizer_seeds = {
        *_integers(PROTOCOL["optimizer_seeds"]),
        *_integers(PROTOCOL_V2["optimizer_seeds"]),
    }
    training = {
        seed + offset
        for seed in _integers(PROTOCOL_V3["dataset_seeds"])
        for offset in (1, 2)
    }
    protected = {
        seed + offset
        for seed in _integers(PROTOCOL_V3["dataset_seeds"])
        for offset in _integers(PROTOCOL_V3["protected_offsets"])
    }

    assert set(_integers(PROTOCOL_V3["dataset_seeds"])).isdisjoint(prior_dataset_seeds)
    assert set(_integers(PROTOCOL_V3["optimizer_seeds"])).isdisjoint(prior_optimizer_seeds)
    assert training.isdisjoint(protected)
    assert PROTOCOL_V3["train_examples"] == 4_096
    assert PROTOCOL_V3["development_examples"] == 1_024
    assert PROTOCOL_V3["protected_examples"] == 2_048
    assert PROTOCOL_V3["updates"] == 3_000
    assert PROTOCOL_V3["batch_size"] == 256
    assert PROTOCOL_V3["max_wall_seconds"] == 21_600
    assert PROTOCOL_V3["prior_protected_evidence_reused"] is False
    assert PROTOCOL_V3["paid_compute"] is False

    notebook_path = tmp_path / "campaign-v3-heavy.ipynb"
    build_v3(SOURCE_V3, SOURCE_V2, PROTOCOL_PATH_V3, notebook_path)
    cell = "".join(json.loads(notebook_path.read_text())["cells"][1]["source"])
    assert repr(SOURCE_V3.read_text(encoding="utf-8")) in cell
    assert repr(SOURCE_V2.read_text(encoding="utf-8")) in cell
    assert repr(PROTOCOL_PATH_V3.read_text(encoding="utf-8")) in cell
    assert "KRONOS" not in cell

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from olympus.models.smoke import FamilySmokeManifest, run_family_smoke


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_all_family_smoke_executes_and_persists_truthful_artifacts(tmp_path: Path) -> None:
    manifest = run_family_smoke(tmp_path / "run", seed=2026, steps=3)

    assert manifest.passed
    assert manifest.promotion_authorized is False
    assert manifest.scientific_claim == "execution_smoke_only"
    assert manifest.source_paths == [
        "core/schemas.py",
        "models/substrate.py",
        "models/hermes.py",
        "models/prometheus.py",
        "models/perseus.py",
        "models/atlas.py",
        "models/kronos.py",
        "models/aion.py",
        "models/smoke.py",
    ]
    assert manifest.substrate.passed
    assert [result.family for result in manifest.families] == [
        "Hermes",
        "Prometheus",
        "Perseus",
        "Olympus-Atlas",
        "Kronos",
        "Aion",
    ]

    for result in manifest.families:
        assert result.passed
        assert result.status == "EXPERIMENTAL_SMOKE_NOT_PROMOTED"
        assert result.training.gradient_update_observed
        assert result.training.final_loss < result.training.initial_loss
        assert all(result.deterministic_checks.values())
        path = tmp_path / "run" / result.checkpoint.relative_path
        assert path.is_file()
        assert _sha256(path) == result.checkpoint.sha256
        payload = torch.load(path, map_location="cpu", weights_only=True)
        assert payload["family"] == result.family
        assert payload["qualifying_checkpoint"] is False
        assert payload["promoted"] is False

    stored = FamilySmokeManifest.model_validate_json(
        (tmp_path / "run/manifest.json").read_text(encoding="utf-8")
    )
    assert json.loads(stored.model_dump_json()) == json.loads(manifest.model_dump_json())

    inconsistent = manifest.model_dump(mode="python")
    inconsistent["passed"] = False
    with pytest.raises(ValidationError, match="pass flag"):
        FamilySmokeManifest.model_validate(inconsistent)

    duplicate = manifest.model_dump(mode="python")
    duplicate["families"][1]["family"] = duplicate["families"][0]["family"]
    with pytest.raises(ValidationError, match="names must be unique"):
        FamilySmokeManifest.model_validate(duplicate)

    bad_family = manifest.model_dump(mode="python")
    bad_family["families"][0]["passed"] = False
    with pytest.raises(ValidationError, match="family smoke pass flag"):
        FamilySmokeManifest.model_validate(bad_family)


def test_family_smoke_requires_an_optimization_step(tmp_path: Path) -> None:
    try:
        run_family_smoke(tmp_path, steps=0)
    except ValueError as error:
        assert "at least one" in str(error)
    else:
        raise AssertionError("zero-step smoke unexpectedly succeeded")

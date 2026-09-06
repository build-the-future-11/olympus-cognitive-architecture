from __future__ import annotations

import importlib
import json

import pytest
from typer.testing import CliRunner

from olympus.cli import app
from olympus.models import MODEL_FAMILIES, get_family, model_family_status


def test_registry_names_real_classes_and_no_qualifying_checkpoints() -> None:
    assert [family.public_name for family in MODEL_FAMILIES] == [
        "Hermes",
        "Prometheus",
        "Perseus",
        "Olympus-Atlas",
        "Kronos",
        "Aion",
    ]

    for family in MODEL_FAMILIES:
        module = importlib.import_module(family.module)
        assert family.qualified_class_names == tuple(
            f"{family.module}.{name}"
            for name in (*family.learned_classes, *family.deterministic_classes)
        )
        for class_name in family.learned_classes + family.deterministic_classes:
            assert isinstance(getattr(module, class_name), type)
        assert family.implementation == "IMPLEMENTED"
        assert family.validation == "EXPERIMENTAL_SMOKE"
        assert family.promotion == "NOT_PROMOTED"
        assert family.qualifying_checkpoint is None
    assert {family.adapter_name for family in MODEL_FAMILIES} == {
        "hermes",
        "prometheus",
        "perseus",
        "olympus_atlas",
        "kronos",
        "aion",
    }


def test_registry_dependencies_and_lookup_are_explicit() -> None:
    assert get_family("Olympus-Atlas").promotion_dependencies == ("Hermes", "Perseus")
    assert get_family("kronos").promotion_dependencies == ("Olympus-Atlas", "Perseus")
    assert get_family("AION").promotion_dependencies == (
        "Hermes",
        "Prometheus",
        "Perseus",
        "Olympus-Atlas",
        "Kronos",
    )
    with pytest.raises(KeyError, match="unknown Olympus model family"):
        get_family("zeus")


def test_status_snapshot_and_cli_preserve_claim_boundary() -> None:
    status = model_family_status()
    assert status["summary"] == {
        "implementations": "IMPLEMENTED",
        "strongest_validation": "EXPERIMENTAL_SMOKE",
        "promotion": "NOT_PROMOTED",
        "qualifying_checkpoints": 0,
    }

    result = CliRunner().invoke(app, ["models", "status"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["summary"] == status["summary"]
    assert len(payload["families"]) == 6
    assert all(family["qualifying_checkpoint"] is None for family in payload["families"])
    assert "not qualifying evaluations" in payload["claim_boundary"]

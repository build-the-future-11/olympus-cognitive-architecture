"""Lightweight source/serialization controls that do not import Torch."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from olympus.training.provenance import atomic_json, digest_json, file_sha256


def test_campaign_import_has_no_torch_or_training_side_effects() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import olympus.training.campaign; assert 'torch' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_campaign_source_parses_with_python_311_grammar() -> None:
    root = Path(__file__).resolve().parents[1]
    for path in (root / "olympus").rglob("*.py"):
        ast.parse(path.read_text(), filename=str(path), feature_version=(3, 11))


def test_atomic_json_rejects_nonfinite_values_without_replacing_evidence(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    atomic_json(path, {"step": 2})
    original = file_sha256(path)
    with pytest.raises(ValueError):
        atomic_json(path, {"loss": float("nan")})
    assert file_sha256(path) == original
    assert json.loads(path.read_text()) == {"step": 2}
    assert not list(tmp_path.glob(".state.json.*"))


def test_configuration_digest_is_canonical_and_sensitive() -> None:
    assert digest_json({"a": 1, "b": 2}) == digest_json({"b": 2, "a": 1})
    assert digest_json({"batch_size": 4}) != digest_json({"batch_size": 8})

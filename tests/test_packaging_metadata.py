from __future__ import annotations

import sys
import tomllib
from pathlib import Path
from subprocess import run

from olympus import __version__


def test_pyproject_declares_complete_release_metadata() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    project = payload["project"]

    assert project["version"] == __version__
    assert project["license"] == "LicenseRef-Proprietary"
    assert project["license-files"] == ["LICENSE"]
    assert project["authors"] == [{"name": "Ryan"}]
    assert len(project["dependencies"]) == 11
    assert all(">=" in dependency for dependency in project["dependencies"])
    assert any(dependency.startswith("numpy>=") for dependency in project["dependencies"])
    assert any(dependency.startswith("cryptography>=") for dependency in project["dependencies"])
    assert project["urls"]["Source"].endswith("/olympus-cognitive-architecture")
    assert (root / "LICENSE").is_file()


def test_source_distribution_excludes_external_payloads_and_includes_validators() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = (root / "MANIFEST.in").read_text(encoding="utf-8").splitlines()

    assert "recursive-include datasets/hermes-smoke *.jsonl" in manifest
    assert "prune datasets/external" in manifest
    assert "recursive-include scripts *.py" in manifest
    assert "include uv.lock" in manifest
    assert (root / "uv.lock").is_file()


def test_top_level_import_does_not_eagerly_initialize_api() -> None:
    result = run(
        [
            sys.executable,
            "-c",
            "import sys, olympus; print(olympus.__version__, 'olympus.api' in sys.modules)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "0.2.0 False"


def test_lazy_app_export_remains_public() -> None:
    result = run(
        [sys.executable, "-c", "from olympus import app; print(app.title)"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "Olympus Forge API"

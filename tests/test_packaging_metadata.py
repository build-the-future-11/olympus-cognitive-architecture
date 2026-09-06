from __future__ import annotations

import tomllib
from pathlib import Path

from olympus import __version__


def test_pyproject_declares_complete_release_metadata() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    project = payload["project"]

    assert project["version"] == __version__
    assert project["license"] == "LicenseRef-Proprietary"
    assert project["license-files"] == ["LICENSE"]
    assert project["authors"] == [{"name": "Ryan"}]
    assert len(project["dependencies"]) == 9
    assert all(">=" in dependency for dependency in project["dependencies"])
    assert any(dependency.startswith("numpy>=") for dependency in project["dependencies"])
    assert project["urls"]["Source"].endswith("/olympus-cognitive-architecture")
    assert (root / "LICENSE").is_file()


def test_release_manifest_excludes_internal_outreach_material() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = (root / "MANIFEST.in").read_text(encoding="utf-8")

    assert "recursive-include docs *.md" in manifest
    assert "recursive-exclude docs *OUTREACH*.md" in manifest

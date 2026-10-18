from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_root_research_state_matches_version_and_successor_states() -> None:
    state = _read(ROOT / "RESEARCH_STATE.json")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert state["software"]["version"] == project["version"]
    assert state["software"]["release_ready"] is False
    assert state["software"]["model_promoted"] is False

    studies = {study["id"]: study for study in state["studies"]}
    v4 = _read(ROOT / studies["olympus-oracle-fusion-v4"]["path"])
    v5 = _read(ROOT / studies["olympus-robust-fusion-v5"]["path"])
    assert studies["olympus-oracle-fusion-v4"]["phase"] == v4["phase"]
    assert v4["scientific_result"].startswith("PRIMARY_NOT_SUPPORTED")
    assert studies["olympus-robust-fusion-v5"]["phase"] == v5["phase"]
    assert v5["promotion_authorized"] is False


def test_security_policy_tracks_current_minor_version() -> None:
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "version"
    ]
    supported = ".".join(version.split(".")[:2]) + ".x"
    assert f"`{supported}`" in (ROOT / "SECURITY.md").read_text(encoding="utf-8")

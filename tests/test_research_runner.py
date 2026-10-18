"""Static and source-identity checks for frozen research notebooks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from notebooks.build_research_runner import build


@pytest.mark.parametrize("campaign,gpu", [("v5", False), ("editclevr", True)])
def test_research_runner_is_self_contained_and_compilable(
    tmp_path: Path, campaign: str, gpu: bool
) -> None:
    notebook = build(tmp_path / campaign, campaign=campaign)
    value = json.loads(notebook.read_text(encoding="utf-8"))
    assert value["metadata"]["kaggle"]["isGpuEnabled"] is gpu
    source = "\n".join(cell["source"] for cell in value["cells"] if cell["cell_type"] == "code")
    assert "EXPECTED_SHA256" in source
    assert "verify_restored" in source
    assert "verify_archive" in source
    assert f"scripts.run_{'robust_fusion_v5' if campaign == 'v5' else 'editclevr_v1'}" in source
    for cell in value["cells"]:
        if cell["cell_type"] == "code":
            compile(cell["source"], "<generated-research-cell>", "exec")


def test_research_runner_rejects_invalid_campaign_without_creating_output(tmp_path: Path) -> None:
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="campaign"):
        build(output, campaign="unknown")
    assert not output.exists()

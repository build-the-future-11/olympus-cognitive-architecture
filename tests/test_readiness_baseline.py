from __future__ import annotations

from pathlib import Path

from olympus.foundry.readiness import (
    evaluate_portfolio_readiness,
    readiness_evidence_schema_bytes,
)

ROOT = Path(__file__).parents[1]


def test_checked_in_readiness_schema_is_current() -> None:
    schema_path = ROOT / "configs/llm-readiness-evidence-v1.schema.json"
    assert schema_path.read_bytes() == readiness_evidence_schema_bytes()


def test_checked_in_readiness_report_is_reproducible(tmp_path: Path) -> None:
    generated_path = tmp_path / "CURRENT_REPORT.json"
    evaluate_portfolio_readiness(
        ROOT / "research/llm_readiness_v1/CURRENT_EVIDENCE.json",
        ROOT / "configs/llm-readiness-v1.json",
        generated_path,
    )
    expected_path = ROOT / "research/llm_readiness_v1/CURRENT_REPORT.json"
    assert generated_path.read_bytes() == expected_path.read_bytes()

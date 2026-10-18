from __future__ import annotations

from pathlib import Path

from olympus.foundry.readiness import (
    DatasetEvidence,
    audit_instruction_dataset,
    blocker_counts,
    evaluate_portfolio_readiness,
    readiness_evidence_schema_bytes,
    readiness_report_schema_bytes,
)

ROOT = Path(__file__).parents[1]


def test_checked_in_readiness_schema_is_current() -> None:
    schema_path = ROOT / "configs/llm-readiness-evidence-v1.schema.json"
    assert schema_path.read_bytes() == readiness_evidence_schema_bytes()

    report_schema_path = ROOT / "configs/llm-readiness-report-v1.schema.json"
    assert report_schema_path.read_bytes() == readiness_report_schema_bytes()


def test_checked_in_readiness_report_is_reproducible(tmp_path: Path) -> None:
    generated_path = tmp_path / "CURRENT_REPORT.json"
    evaluate_portfolio_readiness(
        ROOT / "research/llm_readiness_v1/CURRENT_EVIDENCE.json",
        ROOT / "configs/llm-readiness-v1.json",
        generated_path,
    )
    expected_path = ROOT / "research/llm_readiness_v1/CURRENT_REPORT.json"
    assert generated_path.read_bytes() == expected_path.read_bytes()


def test_current_dataset_audit_and_blocker_totals_are_reproducible(tmp_path: Path) -> None:
    audit_path = tmp_path / "dataset-audit.json"
    audit = audit_instruction_dataset(
        ROOT / "research/llm_readiness_v1/hermes_smoke_dataset/manifest.json",
        audit_path,
    )
    assert audit.training_records == 12
    assert audit.protected_records == 12
    assert audit.minimum_protected_records_per_category == 1
    assert DatasetEvidence.model_validate_json(audit_path.read_bytes()) == audit

    report = evaluate_portfolio_readiness(
        ROOT / "research/llm_readiness_v1/CURRENT_EVIDENCE.json",
        ROOT / "configs/llm-readiness-v1.json",
        tmp_path / "portfolio-report.json",
    )
    counts = blocker_counts(report)
    assert counts["data_receipt_integrity"] == 5
    assert counts["data_scale"] == 6
    assert counts["release_documentation"] == 6

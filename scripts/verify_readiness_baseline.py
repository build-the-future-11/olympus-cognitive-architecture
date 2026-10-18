"""Verify the checked-in readiness schemas and current decision baseline."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from olympus.foundry.readiness import (
    PortfolioReadinessReport,
    evaluate_portfolio_readiness,
    readiness_evidence_schema_bytes,
    readiness_report_schema_bytes,
)

ROOT = Path(__file__).parents[1]


def main() -> None:
    evidence_schema = ROOT / "configs/llm-readiness-evidence-v1.schema.json"
    report_schema = ROOT / "configs/llm-readiness-report-v1.schema.json"
    expected_report = ROOT / "research/llm_readiness_v1/CURRENT_REPORT.json"

    if evidence_schema.read_bytes() != readiness_evidence_schema_bytes():
        raise SystemExit("checked-in readiness evidence schema is stale")
    if report_schema.read_bytes() != readiness_report_schema_bytes():
        raise SystemExit("checked-in readiness report schema is stale")

    PortfolioReadinessReport.model_validate_json(expected_report.read_bytes())
    with TemporaryDirectory(prefix="olympus-readiness-") as directory:
        generated_report = Path(directory) / "CURRENT_REPORT.json"
        report = evaluate_portfolio_readiness(
            ROOT / "research/llm_readiness_v1/CURRENT_EVIDENCE.json",
            ROOT / "configs/llm-readiness-v1.json",
            generated_report,
        )
        if generated_report.read_bytes() != expected_report.read_bytes():
            raise SystemExit("checked-in readiness report is stale")
        if report.ready_families != 0 or report.total_families != 6:
            raise SystemExit("current readiness disposition changed unexpectedly")

    print("readiness schemas and 0/6 baseline are exact")


if __name__ == "__main__":
    main()

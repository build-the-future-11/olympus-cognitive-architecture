"""Verify the checked-in readiness schemas and current decision baseline."""

from __future__ import annotations

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

from olympus.foundry.readiness import (
    FROZEN_POLICY_SHA256,
    POLICY_FILENAMES,
    TRUST_POLICY_FILENAME,
    PortfolioReadinessReport,
    ReceiptTrustPolicy,
    evaluate_portfolio_readiness,
    readiness_evidence_schema_bytes,
    readiness_receipt_schema_bytes,
    readiness_report_schema_bytes,
    readiness_trust_policy_schema_bytes,
)

ROOT = Path(__file__).parents[1]


def main() -> None:
    evidence_schema = ROOT / "configs/llm-readiness-evidence-v1.schema.json"
    report_schema = ROOT / "configs/llm-readiness-report-v1.schema.json"
    receipt_schema = ROOT / "configs/llm-readiness-receipt-v1.schema.json"
    trust_schema = ROOT / "configs/llm-readiness-trust-v1.schema.json"
    expected_report = ROOT / "research/llm_readiness_v1/CURRENT_REPORT.json"
    trust_policy = ROOT / "configs" / TRUST_POLICY_FILENAME

    if evidence_schema.read_bytes() != readiness_evidence_schema_bytes():
        raise SystemExit("checked-in readiness evidence schema is stale")
    if report_schema.read_bytes() != readiness_report_schema_bytes():
        raise SystemExit("checked-in readiness report schema is stale")
    if receipt_schema.read_bytes() != readiness_receipt_schema_bytes():
        raise SystemExit("checked-in readiness receipt schema is stale")
    if trust_schema.read_bytes() != readiness_trust_policy_schema_bytes():
        raise SystemExit("checked-in readiness trust policy schema is stale")
    legacy_policy_names = {
        "capability": "CAPABILITY_SUITE.json",
        "safety": "SAFETY_SUITE.json",
        "production": "PRODUCTION_QUALIFICATION.json",
    }
    for name, filename in POLICY_FILENAMES.items():
        payload = (ROOT / "configs" / filename).read_bytes()
        if hashlib.sha256(payload).hexdigest() != FROZEN_POLICY_SHA256[name]:
            raise SystemExit(f"checked-in {name} readiness policy is stale")
        legacy = ROOT / "research/llm_readiness_v1" / legacy_policy_names[name]
        if legacy.read_bytes() != payload:
            raise SystemExit(f"research copy of {name} readiness policy has drifted")
    ReceiptTrustPolicy.model_validate_json(trust_policy.read_bytes())

    PortfolioReadinessReport.model_validate_json(expected_report.read_bytes())
    with TemporaryDirectory(prefix="olympus-readiness-") as directory:
        generated_report = Path(directory) / "CURRENT_REPORT.json"
        report = evaluate_portfolio_readiness(
            ROOT / "research/llm_readiness_v1/CURRENT_EVIDENCE.json",
            ROOT / "configs/llm-readiness-v1.json",
            generated_report,
            trust_policy,
        )
        if generated_report.read_bytes() != expected_report.read_bytes():
            raise SystemExit("checked-in readiness report is stale")
        if report.ready_families != 0 or report.total_families != 6:
            raise SystemExit("current readiness disposition changed unexpectedly")

    print("readiness schemas and 0/6 baseline are exact")


if __name__ == "__main__":
    main()

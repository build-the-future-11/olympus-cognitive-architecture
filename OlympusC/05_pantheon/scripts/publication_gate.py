#!/usr/bin/env python3
"""Evaluate current artifact completeness without making a publication claim.

The top-level ``PUBLICATION_READINESS.json`` is the frozen V4 machine artifact;
``PUBLICATION_READINESS.md`` is its current explanatory wrapper.  This
compatibility-named script must never overwrite either file.  It writes a
current, explicitly structural assessment under ``poststudy/``; the separate
scientific-readiness gate is authoritative for interpretation and publication
blocking.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_JSON = ROOT / "poststudy" / "STRUCTURAL_READINESS.json"
OUTPUT_MD = OUTPUT_JSON.with_suffix(".md")


def _command_check(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> tuple[bool, str]:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, env=env, check=False)
    detail = stable_detail((result.stdout + result.stderr).strip()[-1000:])
    return result.returncode == 0, detail


def stable_detail(detail: str) -> str:
    """Remove non-semantic timing noise before persisting command evidence."""
    normalized = re.sub(r" in \d+(?:\.\d+)?s\b", "", detail)
    return "\n".join(line for line in normalized.splitlines() if "[100%]" not in line)


def assess_structure(root: Path = ROOT, python: str = sys.executable) -> dict[str, object]:
    checks: list[dict[str, object]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "pass": bool(passed), "detail": detail})

    passed, detail = _command_check([python, "scripts/validate_evidence.py"], cwd=root)
    add("local_evidence_validator", passed, detail)

    env = {**os.environ, "PYTHONPATH": str((root / "src").resolve())}
    passed, detail = _command_check([python, "-m", "pytest", "-q"], cwd=root, env=env)
    add("tests", passed, detail)

    results_path = root / "results" / "external" / "external_task_results.csv"
    manifest_path = root / "external" / "corebench" / "manifest.jsonl"
    if results_path.exists() and manifest_path.exists():
        frame = pd.read_csv(results_path)
        tasks = [
            json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()
        ]
        task_metadata = {task["task_id"]: task for task in tasks}
        visibility = {
            task_metadata.get(task_id, {}).get("benchmark_visibility", "unknown")
            for task_id in frame.task_id
        }
        fields = {
            task_metadata.get(task_id, {}).get("field")
            for task_id in frame.task_id
            if task_metadata.get(task_id, {}).get("field")
        }
        independent = all(
            (row.agent_a_provider, row.agent_a_model) != (row.agent_b_provider, row.agent_b_model)
            for _, row in frame.iterrows()
        )
        add("external_tasks", len(frame) >= 15, f"{len(frame)} tasks; threshold 15")
        question_count = int(frame.question_count.sum())
        add(
            "external_questions",
            question_count >= 20,
            f"{question_count} scored questions; threshold 20",
        )
        add("agent_identity_difference", independent, "agent identities differ on every task")
        add("cross_domain", len(fields) >= 2, f"{len(fields)} fields: {sorted(fields)}")
        add(
            "heldout_or_ood",
            bool(visibility & {"heldout", "ood"}),
            f"visibility={sorted(visibility)}; public_train is pilot evidence only",
        )
        numeric_columns = [
            "agent_a_accuracy",
            "agent_b_accuracy",
            "pairwise_agreement_rate",
            "false_consensus_rate",
        ]
        numeric_complete = not frame[numeric_columns].isna().any().any()
        report_columns = {"agent_a_report_exists", "agent_b_report_exists"}
        reports_complete = report_columns.issubset(frame.columns) and bool(
            frame[sorted(report_columns)].all().all()
        )
        error_columns = {"agent_a_provider_error", "agent_b_provider_error"}
        providers_clean = error_columns.issubset(frame.columns) and bool(
            frame[sorted(error_columns)].isna().all().all()
        )
        add(
            "normalized_artifacts_present",
            numeric_complete and reports_complete and providers_clean,
            (
                f"numeric={numeric_complete}; normalized_reports={reports_complete}; "
                f"provider_errors_absent={providers_clean}; this does not establish "
                "agent authorship"
            ),
        )
    else:
        for name in [
            "external_tasks",
            "external_questions",
            "agent_identity_difference",
            "cross_domain",
            "heldout_or_ood",
            "normalized_artifacts_present",
        ]:
            add(name, False, "external study not yet run")

    artifact_complete = all(bool(check["pass"]) for check in checks)
    return {
        "schema_version": 1,
        "gate_scope": "artifact_completeness_only",
        "verdict": "ARTIFACT_COMPLETE" if artifact_complete else "ARTIFACT_INCOMPLETE",
        "artifact_complete": artifact_complete,
        "scientific_readiness_evaluated": False,
        "publication_ready": False,
        "checks": checks,
        "interpretation": (
            "Structural completeness is necessary but insufficient. "
            "Run poststudy/scientific_readiness_gate.py for the authoritative scientific verdict."
        ),
    }


def write_assessment(result: dict[str, object], output_json: Path = OUTPUT_JSON) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    output_md = output_json.with_suffix(".md")
    checks = result["checks"]
    assert isinstance(checks, list)
    output_md.write_text(
        "# Structural Readiness Gate\n\n"
        f"**Verdict: {result['verdict']}**\n\n"
        "> This gate checks artifact completeness only. It cannot emit a "
        "publication-ready verdict, "
        "and it does not establish model capability or scientific interpretability.\n\n"
        + "\n".join(
            f"- {'PASS' if check['pass'] else 'BLOCK'} — **{check['check']}**: {check['detail']}"
            for check in checks
        )
        + f"\n\n{result['interpretation']}\n"
    )


def main() -> int:
    result = assess_structure()
    write_assessment(result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["artifact_complete"] else 3


if __name__ == "__main__":
    raise SystemExit(main())

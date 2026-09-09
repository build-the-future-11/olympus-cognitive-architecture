#!/usr/bin/env python3
"""Separate artifact completeness from scientific interpretability.

The frozen V4 protocol's legacy publication gate is retained byte-for-byte for
reproducibility. This post-study gate prevents that structural verdict from
being mistaken for evidence that the evaluated agents crossed a capability
floor suitable for estimating naturally occurring false consensus.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def assess(
    question_rows: pd.DataFrame,
    task_rows: pd.DataFrame,
    legacy_gate: dict,
    *,
    minimum_answered_per_agent: int = 5,
    minimum_joint_answered: int = 5,
    minimum_authored_reports_per_agent: int = 5,
) -> dict:
    required_q = {"agent_a_answered", "agent_b_answered"}
    required_t = {"agent_a_submitted_report", "agent_b_submitted_report"}
    if not required_q.issubset(question_rows.columns) or not required_t.issubset(task_rows.columns):
        missing = sorted(
            (required_q - set(question_rows.columns)) | (required_t - set(task_rows.columns))
        )
        raise ValueError(f"required capability columns missing: {missing}")
    a_answered = int(question_rows.agent_a_answered.sum())
    b_answered = int(question_rows.agent_b_answered.sum())
    joint_answered = int(
        (
            question_rows.agent_a_answered.astype(bool)
            & question_rows.agent_b_answered.astype(bool)
        ).sum()
    )
    a_reports = int(task_rows.agent_a_submitted_report.astype(bool).sum())
    b_reports = int(task_rows.agent_b_submitted_report.astype(bool).sum())
    artifact_complete = (
        legacy_gate.get("artifact_complete") is True
        if "artifact_complete" in legacy_gate
        else legacy_gate.get("verdict") == "PUBLICATION-READY"
    )
    capability_checks = {
        "agent_a_answered_floor": a_answered >= minimum_answered_per_agent,
        "agent_b_answered_floor": b_answered >= minimum_answered_per_agent,
        "joint_answered_floor": joint_answered >= minimum_joint_answered,
        "agent_a_authored_report_floor": a_reports >= minimum_authored_reports_per_agent,
        "agent_b_authored_report_floor": b_reports >= minimum_authored_reports_per_agent,
    }
    capability_floor_passed = all(capability_checks.values())
    if not artifact_complete:
        verdict = "ARTIFACT_INCOMPLETE"
    elif not capability_floor_passed:
        verdict = "NEGATIVE_RESULT_ARTIFACT_COMPLETE_CAPABILITY_FLOOR_FAILED"
    else:
        verdict = "ARTIFACT_COMPLETE_CAPABILITY_FLOOR_PASSED"
    return {
        "schema_version": 1,
        "verdict": verdict,
        "artifact_complete": artifact_complete,
        "capability_floor_passed": capability_floor_passed,
        "thresholds": {
            "minimum_answered_per_agent": minimum_answered_per_agent,
            "minimum_joint_answered": minimum_joint_answered,
            "minimum_authored_reports_per_agent": minimum_authored_reports_per_agent,
        },
        "observed": {
            "questions": len(question_rows),
            "tasks": len(task_rows),
            "agent_a_answered": a_answered,
            "agent_b_answered": b_answered,
            "joint_answered": joint_answered,
            "agent_a_authored_reports": a_reports,
            "agent_b_authored_reports": b_reports,
        },
        "capability_checks": capability_checks,
        "interpretation": (
            "The complete artifact supports a bounded negative capability result. "
            "It does not identify false-consensus prevalence among capable agents."
            if artifact_complete and not capability_floor_passed
            else "Capability-floor status is reported separately from artifact completeness."
        ),
        "gate_timing": (
            "Post-study reporting safeguard; thresholds are not presented as "
            "preregistered hypothesis tests."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions", type=Path, default=Path("results/external/question_level.csv")
    )
    parser.add_argument("--tasks", type=Path, default=Path("results/external/task_diagnostics.csv"))
    parser.add_argument(
        "--legacy-gate",
        type=Path,
        default=Path("poststudy/STRUCTURAL_READINESS.json"),
        help=(
            "current structural gate (the historical PUBLICATION_READINESS.json remains "
            "accepted explicitly)"
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("poststudy/SCIENTIFIC_READINESS.json"))
    parser.add_argument(
        "--require-capability-floor",
        action="store_true",
        help="return nonzero unless both artifact completeness and the capability floor pass",
    )
    args = parser.parse_args()
    result = assess(
        pd.read_csv(args.questions),
        pd.read_csv(args.tasks),
        json.loads(args.legacy_gate.read_text()),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    md_path = args.output.with_suffix(".md")
    observed = result["observed"]
    md_path.write_text(
        "# Scientific Readiness Gate\n\n"
        f"**Verdict: {result['verdict']}**\n\n"
        f"- Artifact complete: **{result['artifact_complete']}**\n"
        f"- Capability floor passed: **{result['capability_floor_passed']}**\n"
        f"- Answered questions: agent A {observed['agent_a_answered']}/"
        f"{observed['questions']}; agent B {observed['agent_b_answered']}/"
        f"{observed['questions']}; jointly answered {observed['joint_answered']}.\n"
        f"- Agent-authored valid reports: A {observed['agent_a_authored_reports']}/"
        f"{observed['tasks']}; "
        f"B {observed['agent_b_authored_reports']}/{observed['tasks']}.\n\n"
        f"{result['interpretation']}\n\n{result['gate_timing']}\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["artifact_complete"]:
        return 3
    if args.require_capability_floor and not result["capability_floor_passed"]:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

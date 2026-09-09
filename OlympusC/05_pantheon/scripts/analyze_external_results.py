#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0, c - h), min(1, c + h)]


p = Path("results/external/external_task_results.csv")
if not p.exists():
    raise SystemExit("No external results found. Run scripts/run_external_study.py first.")
df = pd.read_csv(p)
q = int(df.question_count.sum())
fc = int(df.false_consensus_count.sum())
question_path = Path("results/external/question_level.csv")
question_rows = pd.read_csv(question_path) if question_path.exists() else None
a_answered = int(question_rows.agent_a_answered.sum()) if question_rows is not None else None
b_answered = int(question_rows.agent_b_answered.sum()) if question_rows is not None else None
joint_answered = (
    int(
        (
            question_rows.agent_a_answered.astype(bool)
            & question_rows.agent_b_answered.astype(bool)
        ).sum()
    )
    if question_rows is not None
    else None
)
a_reports = (
    int(df.agent_a_submitted_report.astype(bool).sum())
    if "agent_a_submitted_report" in df
    else None
)
b_reports = (
    int(df.agent_b_submitted_report.astype(bool).sum())
    if "agent_b_submitted_report" in df
    else None
)
summary = {
    "n_tasks": int(len(df)),
    "n_questions": q,
    "agent_a_mean_task_accuracy": float(df.agent_a_accuracy.mean()),
    "agent_b_mean_task_accuracy": float(df.agent_b_accuracy.mean()),
    "mean_pairwise_agreement_rate": float(df.pairwise_agreement_rate.mean()),
    "false_consensus_count": fc,
    "false_consensus_question_rate": float(fc / q) if q else 0.0,
    "false_consensus_rate_wilson95": wilson(fc, q),
    "agent_pairs": df[["agent_a_provider", "agent_a_model", "agent_b_provider", "agent_b_model"]]
    .drop_duplicates()
    .to_dict("records"),
    "capability_observed": {
        "agent_a_answered": a_answered,
        "agent_b_answered": b_answered,
        "joint_answered": joint_answered,
        "agent_a_authored_reports": a_reports,
        "agent_b_authored_reports": b_reports,
    },
    "interpretation": (
        "False-consensus prevalence is not identified for capable agents because "
        "no evaluated agent returned a non-null answer."
        if a_answered == 0 and b_answered == 0
        else "Interpret false consensus together with the post-study capability gate."
    ),
}
Path("results/external/external_aggregate.json").write_text(json.dumps(summary, indent=2))
interval = summary["false_consensus_rate_wilson95"]
md_lines = [
    "# External Agent Results",
    "",
    f"- Tasks: **{summary['n_tasks']}**",
    f"- Scored questions: **{q}**",
    f"- Agent A mean task accuracy: **{summary['agent_a_mean_task_accuracy']:.3f}**",
    f"- Agent B mean task accuracy: **{summary['agent_b_mean_task_accuracy']:.3f}**",
    f"- Agent A non-null answers: **{a_answered}/{q}**",
    f"- Agent B non-null answers: **{b_answered}/{q}**",
    f"- Jointly answered questions: **{joint_answered}/{q}**",
    f"- Agent-authored reports: **A {a_reports}/{len(df)}; B {b_reports}/{len(df)}**",
    (f"- Mean substantive pairwise agreement: **{summary['mean_pairwise_agreement_rate']:.3f}**"),
    (
        f"- Observed false-consensus questions: **{fc}/{q} "
        f"({summary['false_consensus_question_rate']:.3f})**"
    ),
    (
        "- Wilson 95% interval for this observed event rate: "
        f"**[{interval[0]:.3f}, {interval[1]:.3f}]**"
    ),
    "",
    (
        "False consensus requires mutually matching non-null answers outside the accepted "
        "canonical answer set; joint nulls are abstentions, not agreement. "
        f"**{summary['interpretation']}** These values are generated from raw run artifacts "
        "and are not prefilled. See `poststudy/SCIENTIFIC_READINESS.md` for the authoritative "
        "interpretation gate."
    ),
]
md = "\n".join(md_lines) + "\n"
Path("paper/generated/external_results.md").write_text(md)
print(json.dumps(summary, indent=2))

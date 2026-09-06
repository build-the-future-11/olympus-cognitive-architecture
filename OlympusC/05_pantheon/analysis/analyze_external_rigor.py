#!/usr/bin/env python3
"""Question-level and task-cluster inference for the frozen OOD agent study."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "external"
OUT = ROOT / "results" / "external"


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float]:
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def cluster_interval(frame: pd.DataFrame, column: str, *, repetitions: int = 10_000) -> list[float]:
    task_ids = sorted(frame.task_id.unique())
    if not task_ids:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(20260903)
    task_sum = frame.groupby("task_id")[column].sum().reindex(task_ids).to_numpy(dtype=float)
    task_n = frame.groupby("task_id")[column].size().reindex(task_ids).to_numpy(dtype=float)
    draws = rng.integers(0, len(task_ids), size=(repetitions, len(task_ids)))
    values = task_sum[draws].sum(axis=1) / task_n[draws].sum(axis=1)
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def main() -> None:
    manifest = {
        row["task_id"]: row
        for row in (json.loads(line) for line in (ROOT / "external/corebench/manifest.jsonl").read_text().splitlines() if line.strip())
    }
    rows: list[dict[str, object]] = []
    task_rows: list[dict[str, object]] = []
    for comparison_path in sorted(RUNS.glob("*/comparison.json")):
        comparison = json.loads(comparison_path.read_text())
        task_id = comparison["task_id"]
        agent_a = json.loads((comparison_path.parent / "agent_a/agent_result.json").read_text())
        agent_b = json.loads((comparison_path.parent / "agent_b/agent_result.json").read_text())
        questions = list(comparison["agent_a"]["per_question"])
        for question in questions:
            rows.append({
                "task_id": task_id,
                "field": manifest[task_id]["field"],
                "language": manifest[task_id]["language"],
                "question": question,
                "agent_a_correct": int(comparison["agent_a"]["per_question"][question]),
                "agent_b_correct": int(comparison["agent_b"]["per_question"][question]),
                "agent_a_answered": int(agent_a["answers"].get(question) is not None),
                "agent_b_answered": int(agent_b["answers"].get(question) is not None),
                "agreement": int(comparison["agreement"][question]),
                "false_consensus": int(comparison["false_consensus"][question]),
            })
        task_rows.append({
            "task_id": task_id,
            "field": manifest[task_id]["field"],
            "questions": len(questions),
            "agent_a_accuracy": comparison["agent_a"]["accuracy"],
            "agent_b_accuracy": comparison["agent_b"]["accuracy"],
            "agent_a_submitted_report": int(comparison["agent_a_submitted_report"]),
            "agent_b_submitted_report": int(comparison["agent_b_submitted_report"]),
            "agent_a_turns": agent_a["turns"],
            "agent_b_turns": agent_b["turns"],
            "agent_a_runtime_sec": agent_a["runtime_sec"],
            "agent_b_runtime_sec": agent_b["runtime_sec"],
            "agent_a_tool_calls": sum(len(turn.get("tool_calls", [])) for turn in agent_a["transcript"]),
            "agent_b_tool_calls": sum(len(turn.get("tool_calls", [])) for turn in agent_b["transcript"]),
            "agent_a_provider_error": agent_a.get("provider_error"),
            "agent_b_provider_error": agent_b.get("provider_error"),
        })
    questions = pd.DataFrame(rows)
    tasks = pd.DataFrame(task_rows)
    if len(tasks) != len(manifest):
        raise SystemExit(f"incomplete study: {len(tasks)}/{len(manifest)} tasks")
    OUT.mkdir(parents=True, exist_ok=True)
    questions.to_csv(OUT / "question_level.csv", index=False)
    tasks.to_csv(OUT / "task_diagnostics.csv", index=False)

    n = len(questions)
    discordant_a = int(((questions.agent_a_correct == 1) & (questions.agent_b_correct == 0)).sum())
    discordant_b = int(((questions.agent_a_correct == 0) & (questions.agent_b_correct == 1)).sum())
    discordant = discordant_a + discordant_b
    mcnemar_p = float(binomtest(discordant_a, discordant, 0.5).pvalue) if discordant else 1.0
    metrics = {}
    for column in ["agent_a_correct", "agent_b_correct", "agent_a_answered", "agent_b_answered", "agreement", "false_consensus"]:
        k = int(questions[column].sum())
        metrics[column] = {
            "count": k,
            "n": n,
            "rate": float(k / n),
            "wilson95": wilson(k, n),
            "task_cluster_bootstrap95": cluster_interval(questions, column),
        }
    summary = {
        "tasks": len(tasks),
        "questions": n,
        "fields": sorted(questions.field.unique()),
        "languages": sorted(questions.language.unique()),
        "question_weighted_metrics": metrics,
        "task_macro_accuracy": {
            "agent_a": float(tasks.agent_a_accuracy.mean()),
            "agent_b": float(tasks.agent_b_accuracy.mean()),
        },
        "agent_authored_report_rate": {
            "agent_a": float(tasks.agent_a_submitted_report.mean()),
            "agent_b": float(tasks.agent_b_submitted_report.mean()),
        },
        "trajectory": {
            "agent_a_total_tool_calls": int(tasks.agent_a_tool_calls.sum()),
            "agent_b_total_tool_calls": int(tasks.agent_b_tool_calls.sum()),
            "agent_a_median_turns": float(tasks.agent_a_turns.median()),
            "agent_b_median_turns": float(tasks.agent_b_turns.median()),
            "agent_a_max_turn_saturation": int((tasks.agent_a_turns == 20).sum()),
            "agent_b_max_turn_saturation": int((tasks.agent_b_turns == 20).sum()),
            "agent_a_total_runtime_sec": float(tasks.agent_a_runtime_sec.sum()),
            "agent_b_total_runtime_sec": float(tasks.agent_b_runtime_sec.sum()),
        },
        "paired_accuracy": {
            "agent_a_only_correct": discordant_a,
            "agent_b_only_correct": discordant_b,
            "exact_mcnemar_p": mcnemar_p,
        },
        "inference_note": "Wilson intervals treat questions as Bernoulli trials; task-cluster bootstrap intervals preserve within-task dependence over the 17 observed tasks. Neither supports population-wide agent claims.",
    }
    (OUT / "rigorous_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Outcome & Count & Rate & Wilson 95\% CI & Task-bootstrap 95\% CI \\",
        r"\midrule",
    ]
    labels = {
        "agent_a_correct": "Qwen correct",
        "agent_b_correct": "Llama correct",
        "agent_a_answered": "Qwen non-null",
        "agent_b_answered": "Llama non-null",
        "agreement": "Substantive agreement",
        "false_consensus": "False consensus",
    }
    for key, label in labels.items():
        value = metrics[key]
        lines.append(f"{label} & {value['count']}/{n} & {value['rate']:.3f} & [{value['wilson95'][0]:.3f}, {value['wilson95'][1]:.3f}] & [{value['task_cluster_bootstrap95'][0]:.3f}, {value['task_cluster_bootstrap95'][1]:.3f}]" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "table_external_rigorous.tex").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

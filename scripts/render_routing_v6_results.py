"""Render verified v6 JSON evidence to LaTeX tables and PGFPlots figures.

No Torch imports or training. This independently checks JSON aggregation and
artifact hashes; checkpoint/data replay remains the scientific verifier's job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def label(name: str) -> str:
    return name.replace("anchored", "adaptive").replace("_", " ")


def render(evidence: Path, output: Path) -> None:
    summary = read(evidence / "SUMMARY.json")
    verified = read(evidence / "VERIFIED.json")
    freeze = read(evidence / "SCIENTIFIC_FREEZE.json")
    if verified["status"] != "PASS" or verified["raw_reports_recomputed"] != 40:
        raise ValueError("complete verified campaign required")
    for name, digest in freeze["files"].items():
        if sha(evidence / name) != digest:
            raise ValueError(f"frozen file changed: {name}")
    rows = [read(p) for p in sorted((evidence / "protected").glob("*/*.json"))]
    for p in (evidence / "protected").glob("*/*.json"):
        row = read(p)
        if sha(p.with_suffix(".pt")) != row["prediction_sha256"]:
            raise ValueError("raw prediction hash changed")
        if row["freeze_sha256"] != sha(evidence / "SCIENTIFIC_FREEZE.json"):
            raise ValueError("freeze binding changed")
        for values in row["metrics"].values():
            d = values["intervention"]
            delta = (d["rescues"] - d["harms"]) / d["examples"]
            observed = values["accuracy"] - row["metrics"]["reference"]["accuracy"]
            if not math.isclose(delta, observed, abs_tol=1e-12):
                raise ValueError("rescue/harm arithmetic mismatch")
    regimes = freeze["protocol"]["protected_regimes"]
    modes = list(summary["means"][regimes[0]])
    for regime in regimes:
        for mode in modes:
            for metric, value in summary["means"][regime][mode].items():
                computed = fmean(r["metrics"][mode][metric] for r in rows if r["regime"] == regime)
                if not math.isclose(computed, value, abs_tol=1e-12):
                    raise ValueError("summary mean differs from reports")
    for key, comparison in summary["comparisons"].items():
        regime, comparator = key.split("/")
        deltas = [
            fmean(
                r["metrics"]["anchored_robust"]["accuracy"] - r["metrics"][comparator]["accuracy"]
                for r in rows
                if r["dataset"] == ds and r["regime"] == regime
            )
            for ds in freeze["protocol"]["dataset_seeds"]
        ]
        if deltas != comparison["dataset_deltas"]:
            raise ValueError("dataset-cluster deltas differ")
    output.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table}[ht]\centering\small",
        r"\caption{Protected accuracy (percent), averaged across ten cells. "
        r"Selection aliases vary by development cell; all arms share bank predictions.}",
        r"\begin{tabular}{lrrrr}\toprule",
        r"Arm & Clean & Corrupted & Known shift & Composite \\\midrule",
    ]
    for mode in modes:
        values = " & ".join(f"{100 * summary['means'][r][mode]['accuracy']:.2f}" for r in regimes)
        lines.append(label(mode) + " & " + values + r" \\")
    lines.extend([r"\bottomrule\end{tabular}\end{table}"])
    (output / "accuracy.tex").write_text("\n".join(lines) + "\n")
    lines = [
        r"\begin{table}[ht]\centering\small",
        r"\caption{Adaptive robust minus selected control, in percentage points. "
        r"Intervals are paired five-dataset descriptive bootstrap intervals.}",
        r"\begin{tabular}{lrrr}\toprule Regime & Mean & Lower & Upper \\\midrule",
    ]
    for regime in regimes:
        c = summary["comparisons"][regime + "/selected_control"]
        lines.append(
            label(regime)
            + " & "
            + " & ".join(f"{100 * v:.3f}" for v in [c["mean"], *c["ci95"]])
            + r" \\"
        )
    lines.append(r"\bottomrule\end{tabular}\end{table}")
    (output / "contrasts.tex").write_text("\n".join(lines) + "\n")
    diagnostics = {}
    for regime in regimes:
        rr = [r["metrics"]["anchored_robust"] for r in rows if r["regime"] == regime]
        diagnostics[regime] = {
            "mean_alpha": fmean(r["mean_alpha"] for r in rr),
            **{
                k: sum(r["intervention"][k] for r in rr)
                for k in ("rescues", "harms", "examples", "prediction_changes")
            },
        }
    dev = [read(p) for p in (evidence / "development").glob("*/report.json")]
    analysis = {
        "summary_sha256": sha(evidence / "SUMMARY.json"),
        "verified_sha256": sha(evidence / "VERIFIED.json"),
        "json_aggregation_and_hashes": "PASS",
        "independent_checkpoint_replay": False,
        "diagnostics": diagnostics,
        "references": dict(Counter(r["reference_mode"] for r in dev)),
        "controls": dict(Counter(r["selected_control"] for r in dev)),
        "development_seconds": [r["elapsed_seconds"] for r in dev],
        "parameters": dev[0]["parameters"],
        "clusters": summary["comparisons"]["composite_shift/selected_control"]["dataset_deltas"],
    }
    (output / "ANALYSIS.json").write_text(json.dumps(analysis, indent=2) + "\n")
    points = []
    for i, regime in enumerate(regimes):
        c = summary["comparisons"][regime + "/selected_control"]
        points.append(
            f"({i},{100 * c['mean']}) += (0,{100 * (c['ci95'][1] - c['mean'])}) "
            f"-= (0,{100 * (c['mean'] - c['ci95'][0])})"
        )
    plot = (
        r"""\begin{figure}[ht]\centering
\begin{tikzpicture}\begin{axis}[width=.88\linewidth,height=5cm,
ylabel={Accuracy difference (pp)},xtick={0,1,2,3},
xticklabels={Clean,Corrupted,Known shift,Composite},xmin=-.4,xmax=3.4,
grid=major,axis line style={gray},tick label style={font=\small}]
\addplot+[only marks,mark=*,error bars/.cd,y dir=both,y explicit] coordinates {
"""
        + "\n".join(points)
        + r"""};
\addplot[dashed,gray,domain=-.4:3.4] {0};
\end{axis}\end{tikzpicture}
\caption{Adaptive robust minus the development-selected control. Error bars
show descriptive 95\% paired dataset-cluster bootstrap intervals. Only the
composite contrast is the improvement primary; its practical threshold is +2 pp.}
\end{figure}
"""
    )
    (output / "contrast-plot.tex").write_text(plot)
    lines = [
        r"\begin{table}[ht]\centering\small",
        r"\caption{Adaptive-robust intervention against the frozen reference. "
        r"Counts sum ten cells (two initializations share each dataset).}",
        r"\begin{tabular}{lrrrr}\toprule "
        r"Regime & Mean $\alpha$ & Rescues & Harms & Changes \\\midrule",
    ]
    for regime, d in diagnostics.items():
        lines.append(
            f"{label(regime)} & {d['mean_alpha']:.4f} & {d['rescues']} & "
            f"{d['harms']} & {d['prediction_changes']}" + r" \\"
        )
    lines.append(r"\bottomrule\end{tabular}\end{table}")
    (output / "interventions.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.evidence, args.output)

"""Select exactly one of the two prospectively declared auxiliary-loss settings."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_observable_prediction import verify_report  # noqa: E402
from scripts.execute_observable_study import EXECUTION_SOURCES, publish_json  # noqa: E402


def select(paths: list[Path]) -> dict[str, Any]:
    reports = [verify_report(path) for path in paths]
    expected = {(c, seed) for c in ("a", "b") for seed in (701, 702)}
    if len(reports) != 4 or {(r["configuration"]["id"], r["seed"]) for r in reports} != expected:
        raise ValueError("selection requires exactly the four declared development blocks")
    protocol_path = ROOT / "research/studies/observable_prediction_v1/protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    configurations = {c["id"]: c for c in protocol["development"]["configurations"]}
    for report in reports:
        if report["phase"] != "development" or report["freeze_sha256"] is not None:
            raise ValueError("only unprotected development outcomes can select the configuration")
        if report["protocol_sha256"] != protocol_hash:
            raise ValueError("development protocol mismatch")
        if report["configuration"] != configurations[report["configuration"]["id"]]:
            raise ValueError("undeclared development coefficients")
        for source in EXECUTION_SOURCES:
            if report["source"]["files_sha256"][source] != hashlib.sha256(
                (ROOT / source).read_bytes()
            ).hexdigest():
                raise ValueError("scientific implementation changed during development")
    scores: dict[str, float] = {}
    raw: dict[str, list[dict[str, Any]]] = {}
    for configuration in ("a", "b"):
        rows = [{"seed": r["seed"], "family": family,
                 "nmse20": r["families"][family]["arms"]["full"]["conditions"][
                     "development"
                 ]["horizons"]["20"]}
                for r in sorted(reports, key=lambda r: r["seed"])
                if r["configuration"]["id"] == configuration
                for family in ("pendulum", "duffing")]
        values = [row["nmse20"] for row in rows]
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("invalid development error")
        scores[configuration] = (
            0.0 if 0 in values else math.exp(sum(math.log(v) for v in values) / len(values))
        )
        raw[configuration] = rows
    chosen = min(("a", "b"), key=lambda c: (scores[c], c))
    return {
        "study_id": protocol["study_id"], "selected_configuration": chosen,
        "configuration": configurations[chosen], "geometric_mean_development20": scores,
        "selection_inputs": raw, "protocol_sha256": protocol_hash,
        "source_reports": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in paths},
        "protected_outcomes_used": False, "additional_tuning_permitted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("selection artifact must be fresh")
    selected = select(args.reports)
    publish_json(args.output, selected)
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()

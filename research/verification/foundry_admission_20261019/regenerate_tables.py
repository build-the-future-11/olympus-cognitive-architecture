#!/usr/bin/env python3
"""Verify retained evidence and regenerate tables without loading a model.

Run from any directory: python regenerate_tables.py [--output RESULTS.md]
Archive members are read directly; this program never extracts or executes them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
AUDIT = ROOT.parents[1] / "olympus_audit_20261019"


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_archive(path: Path, identity: dict[str, Any]) -> dict[str, bytes]:
    """Read only regular members and verify their exact retained byte identities."""
    payload = path.read_bytes()
    if digest(payload) != identity["sha256"] or len(payload) != identity["byte_count"]:
        raise ValueError(f"Archive identity mismatch: {path.name}")
    contents: dict[str, bytes] = {}
    with tarfile.open(path, "r:gz") as archive:
        for item in archive.getmembers():
            if not item.isfile() or item.name in contents:
                raise ValueError(f"Invalid or duplicate archive member: {item.name}")
            stream = archive.extractfile(item)
            if stream is None:
                raise ValueError(f"Missing member: {item.name}")
            contents[item.name] = stream.read()
    manifest = json.loads(contents.pop("ARCHIVE_MANIFEST.json"))
    if set(manifest) != set(contents):
        raise ValueError("Archive file set differs from its manifest")
    for name, data in contents.items():
        if digest(data) != manifest[name]["sha256"] or len(data) != manifest[name]["byte_count"]:
            raise ValueError(f"Archived evidence changed: {name}")
    if len(contents) != identity["member_files"]:
        raise ValueError("Archive member count differs from outer identity")
    return contents


def snapshots(receipt: dict[str, Any]) -> list[dict[str, Any]]:
    values = [receipt[key] for key in ("memory_before", "memory_after") if key in receipt]
    for command in receipt["commands"]:
        values.extend(command.get("memory_samples", []))
        values.extend(command[key] for key in ("memory_before", "memory_after") if key in command)
    return values


def verify_memory(receipt: dict[str, Any]) -> int:
    values = snapshots(receipt)
    for value in values:
        candidates = [value["available_bytes"]]
        candidates.extend(
            max(0, item["limit_bytes"] - item["used_bytes"] + item["cache_credit_bytes"])
            for item in value["cgroups"]
            if item["limit_bytes"] is not None
        )
        if value["effective_available_bytes"] != min(candidates):
            raise ValueError("Recorded memory arithmetic is inconsistent")
    return min(value["effective_available_bytes"] for value in values)


def render() -> str:
    identities = json.loads((ROOT / "archive-identities.json").read_text())
    evidence = read_archive(ROOT / "evidence.tar.gz", identities["evidence.tar.gz"])
    diagnostic = read_archive(
        AUDIT / "residency-diagnostic-v2.tar.gz", identities["residency-diagnostic-v2.tar.gz"]
    )
    run = json.loads(evidence["workflow_v4/receipt.json"])
    if run != json.loads((ROOT / "workflow-receipt.json").read_text()):
        raise ValueError("Readable workflow receipt differs from archived original")
    if run["status"] != "PASSED_ENGINEERING_ONLY" or run["git_status"]:
        raise ValueError("Workflow disposition or clean-source evidence changed")
    source = json.loads((ROOT / "source-manifest.json").read_text())
    if run["git_head"] != source["commit"]:
        raise ValueError("Source commit mismatch")
    for path, expected in run["source_sha256"].items():
        if source["sha256"][path] != expected:
            raise ValueError(f"Source digest mismatch: {path}")
    for path, expected in run["artifact_sha256"].items():
        if digest(evidence["workflow_v4/" + path]) != expected:
            raise ValueError(f"Workflow artifact mismatch: {path}")
    for filename, key in (
        ("pipeline", "pipeline"),
        ("status", "status_after_reload"),
        ("models", "models_after_reload"),
        ("generate", "generation"),
        ("export", "export"),
        ("asgi", "asgi"),
    ):
        if json.loads(evidence[f"workflow_v4/{filename}.stdout"]) != run[key]:
            raise ValueError(f"Stdout and receipt disagree: {filename}")
    checkpoint = run["pipeline"]["checkpoint"]
    model = evidence["workflow_v4/portable-model.json"]
    if model != (ROOT / "verification-model.json").read_bytes():
        raise ValueError("Portable model differs from archived bytes")
    if digest(model) != checkpoint["sha256"] or len(model) != checkpoint["byte_count"]:
        raise ValueError("Model identity mismatch")
    if (
        run["generation"]["content"]
        != run["asgi"]["completion"]["choices"][0]["message"]["content"]
    ):
        raise ValueError("CLI/API generation differs")
    heldout = run["pipeline"]["evaluation"]
    for name in ("candidate_metrics", "baseline_metrics"):
        metrics = heldout[name]
        if not math.isclose(
            math.exp(metrics["negative_log_likelihood"]),
            metrics["perplexity"],
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("Perplexity and NLL disagree")
    minimum = verify_memory(run)
    if minimum < 2 * 1024**3 or any(c["exit_status"] != 0 for c in run["commands"]):
        raise ValueError("A completed workflow violates the recorded gate")
    blocked = json.loads(diagnostic["receipt.json"])
    if blocked["outcome"] != "BLOCKED" or len(blocked["commands"]) != 1:
        raise ValueError("Residency outcome differs from the retained single blocked attempt")
    if any(
        name.startswith("fixture/")
        or name in {"fixture.json", "persistent_module.json", "recreated_module.json"}
        for name in diagnostic
    ):
        raise ValueError("Unexpected residency outcome files")
    blocked_minimum = verify_memory(blocked)
    if blocked_minimum >= 2 * 1024**3:
        raise ValueError("Residency receipt lacks the recorded resource shortfall")
    suites = ET.fromstring(evidence["test_logs/expanded_tests.xml"]).findall("testsuite")
    counts = {
        key: sum(int(s.get(key, "0")) for s in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    coverage = json.loads(evidence["test_logs/expanded_coverage.json"])["totals"]
    gates = json.loads(evidence["test_logs/final_gate_receipt.json"])
    if any(command["exit_status"] != 0 for command in gates["commands"]):
        raise ValueError("Packaging or web gate failed")
    rows = [
        "# Results regenerated from retained evidence",
        "",
        f"Source: `{run['git_head']}`. Both archives and their members were verified.",
        "",
        "## Complete Python suite",
        "",
        "| Measurement | Result |",
        "|---|---:|",
        f"| Test cases | {counts['tests']} |",
        f"| Failures | {counts['failures']} |",
        f"| Errors | {counts['errors']} |",
        f"| Skips | {counts['skipped']} |",
        f"| Combined statement/branch coverage | {coverage['percent_covered']:.2f}% |",
        f"| Statement coverage | {coverage['percent_statements_covered']:.2f}% |",
        f"| Branch-only coverage | {coverage['percent_branches_covered']:.2f}% |",
        "",
        "## Held-out character-bigram lifecycle",
        "",
        "| Metric | Uniform baseline | Candidate |",
        "|---|---:|---:|",
    ]
    for key in ("negative_log_likelihood", "perplexity", "top1_accuracy", "evaluated_transitions"):
        rows.append(
            f"| {key} | {heldout['baseline_metrics'][key]:.12g} | "
            f"{heldout['candidate_metrics'][key]:.12g} |"
        )
    rows += [
        "",
        "## Measured execution bounds",
        "",
        "| Measurement | Result |",
        "|---|---:|",
        f"| Total workflow seconds | {run['elapsed_seconds']:.6f} |",
        f"| Maximum child RSS, bytes | {max(c['child_peak_rss_bytes'] for c in run['commands'])} |",
        f"| Minimum estimated headroom, bytes | {minimum} |",
        f"| Swap-use change, bytes | {run['swap_used_change_bytes']} |",
        f"| Checkpoint/export bytes | {len(model)} |",
        "",
        "## Preserved lifecycle attempts",
        "",
        "| Attempt | Status | Children launched | Seconds |",
        "|---|---|---:|---:|",
    ]
    for version in ("v1", "v2", "v3", "v4"):
        attempt = json.loads(evidence[f"workflow_{version}/receipt.json"])
        rows.append(
            f"| {version} | {attempt['status']} | {len(attempt['commands'])} | "
            f"{attempt['elapsed_seconds']:.6f} |"
        )
    rows += [
        "",
        "## Residency diagnostic",
        "",
        f"Disposition: **{blocked['outcome']}**. One fixture worker; no arm or primary metric.",
        f"Minimum estimated headroom: {blocked_minimum} bytes; "
        f"shortfall: {2 * 1024**3 - blocked_minimum} bytes.",
        "",
        "This regeneration checks retained bytes and arithmetic. It does not rerun a model, "
        "reproduce a scientific experiment, or reconstruct unrecorded memory.stat inputs.",
        "",
    ]
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = render()
    if args.output is None:
        print(result, end="")
    else:
        args.output.write_text(result, encoding="utf-8")
        print(f"Verified evidence; wrote {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

from olympus.evaluation.o1 import (
    O1Score,
    load_frozen_protocol,
    paired_arm_effects,
    parse_and_score,
    render_prompt,
    sha256_file,
    summarize_scores,
    validate_score_keys,
)
from olympus.foundry.ollama import OllamaClient

ROOT = Path(__file__).resolve().parents[1]


def _append_jsonl(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute a hash-frozen Olympus O1 protocol.")
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "research/o1/protocol_v1/protocol.json"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()
    protocol, tasks, arms = load_frozen_protocol(args.protocol)
    client = OllamaClient(base_url=args.base_url, timeout_seconds=args.timeout)
    runtime_version = client.version()
    runtime_digest = client.model_digest(protocol.model)
    if runtime_version != protocol.provider_version:
        raise SystemExit(
            f"provider version drift: frozen={protocol.provider_version} runtime={runtime_version}"
        )
    if runtime_digest != protocol.model_digest:
        raise SystemExit(
            f"model digest drift: frozen={protocol.model_digest} runtime={runtime_digest}"
        )

    run_dir = (ROOT / protocol.raw_output_directory).resolve()
    if not run_dir.is_relative_to(ROOT.resolve()):
        raise SystemExit("protocol output directory escapes the repository root")
    run_dir.mkdir(parents=True, exist_ok=True)
    protocol_hash = sha256_file(args.protocol)
    manifest_path = run_dir / "RUN_MANIFEST.json"
    manifest = {
        "schema_version": 1,
        "protocol_path": str(args.protocol.relative_to(ROOT)),
        "protocol_sha256": protocol_hash,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "provider_version": runtime_version,
        "model": protocol.model,
        "model_digest": runtime_digest,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "expected_observations": len(tasks) * len(arms) * len(protocol.seeds),
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing["protocol_sha256"] != protocol_hash:
            raise SystemExit("existing run directory is bound to a different protocol")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    raw_path = run_dir / "raw_responses.jsonl"
    score_path = run_dir / "scores.jsonl"
    completed: set[tuple[str, str, int]] = set()
    scores: list[O1Score] = []
    if score_path.exists():
        for line in score_path.read_text().splitlines():
            score = O1Score.model_validate_json(line)
            scores.append(score)
        try:
            completed = validate_score_keys(scores, tasks, arms, protocol.seeds)
        except ValueError as error:
            raise SystemExit(f"invalid retained scores: {error}") from error

    for seed in protocol.seeds:
        for task_index, task in enumerate(tasks):
            shuffled_context = tasks[(task_index + 17) % len(tasks)].retrieval_context
            for arm in arms:
                key = (task.task_id, arm.arm_id, seed)
                if key in completed:
                    continue
                system, prompt = render_prompt(task, arm, shuffled_context=shuffled_context)
                start = time.monotonic()
                error = None
                content = ""
                evidence: dict[str, object] = {}
                try:
                    response = client.generate(
                        model=protocol.model,
                        prompt=prompt,
                        system=system,
                        max_tokens=protocol.maximum_output_tokens,
                        context_tokens=protocol.context_tokens,
                        temperature=protocol.temperature,
                        seed=seed,
                    )
                    content = response.content
                    evidence = response.evidence
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                elapsed = time.monotonic() - start
                _append_jsonl(
                    raw_path,
                    {
                        "protocol_sha256": protocol_hash,
                        "task_id": task.task_id,
                        "arm_id": arm.arm_id,
                        "seed": seed,
                        "system_sha256": hashlib.sha256(system.encode()).hexdigest(),
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "raw_response": content,
                        "error": error,
                        "elapsed_seconds": elapsed,
                        "provider_evidence": evidence,
                        "completed_at_utc": datetime.now(UTC).isoformat(),
                    },
                )
                score = parse_and_score(content, task, arm.arm_id, seed)
                if error is not None:
                    score = score.model_copy(update={"parse_error": error})
                _append_jsonl(score_path, score.model_dump())
                scores.append(score)
                completed.add(key)

    try:
        validate_score_keys(scores, tasks, arms, protocol.seeds, require_complete=True)
    except ValueError as error:
        raise SystemExit(f"incomplete O1 run: {error}") from error
    summary = {
        "schema_version": 1,
        "protocol_sha256": protocol_hash,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "observations": len(scores),
        "arms": summarize_scores(scores, tasks, resamples=protocol.bootstrap_resamples),
        "paired_effects_vs_general": paired_arm_effects(
            scores, reference_arm="general_direct", resamples=protocol.bootstrap_resamples
        ),
    }
    (run_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Supervise one declared study block and account for all attempts in one lane."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import resource
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts/observable-prediction-v1"
PROTOCOL = ROOT / "research/studies/observable_prediction_v1/protocol.json"
EXECUTION_SOURCES = (
    "olympus/core/forecasting.py", "olympus/foundry/resources.py",
    "scripts/run_observable_prediction.py", "scripts/analyze_observable_prediction.py",
    "scripts/verify_foundry_integrity.py", "scripts/execute_observable_study.py",
)


def publish_json(path: Path, value: Any) -> None:
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def memory_sample(pid: int | None = None) -> dict[str, Any]:
    mem = {line.split(":")[0]: int(line.split()[1]) * 1024
           for line in Path("/proc/meminfo").read_text().splitlines()}
    cg = Path("/sys/fs/cgroup")
    limit_text = (cg / "memory.max").read_text().strip()
    limit = int(limit_text) if limit_text != "max" else mem["MemTotal"]
    current = int((cg / "memory.current").read_text())
    stat = {k: int(v) for k, v in (line.split() for line in
                                 (cg / "memory.stat").read_text().splitlines())}
    reclaim = max(0, stat.get("inactive_file", 0) - stat.get("file_dirty", 0)
                  - stat.get("file_writeback", 0))
    rss = None
    if pid is not None:
        try:
            status = Path(f"/proc/{pid}/status").read_text()
            entry = next((line for line in status.splitlines() if line.startswith("VmRSS:")), None)
            rss = int(entry.split()[1]) * 1024 if entry else None
        except FileNotFoundError:
            pass
    return {
        "monotonic_seconds": time.perf_counter(), "child_rss_bytes": rss,
        "effective_available_estimate_bytes": min(mem["MemAvailable"],
                                                   max(0, limit - current + reclaim)),
        "cgroup_limit_bytes": limit, "cgroup_current_bytes": current,
        "cgroup_clean_inactive_estimate_bytes": reclaim,
        "cgroup_swap_bytes": int((cg / "memory.swap.current").read_text()),
        "host_swap_fraction": (mem["SwapTotal"] - mem["SwapFree"]) / mem["SwapTotal"]
        if mem["SwapTotal"] else 0,
        "cgroup_events": {k: int(v) for k, v in (line.split() for line in
                                                (cg / "memory.events").read_text().splitlines())},
    }


def terminate_child(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=4)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=1)


def execute(args: argparse.Namespace) -> dict[str, Any]:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,70}", args.attempt):
        raise ValueError("attempt identifier must be a simple fresh name")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(PROTOCOL.read_text())
    source_hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                     for p in EXECUTION_SOURCES}
    budget = protocol["budget"]
    ledger_path = ARTIFACTS / "budget-ledger.json"
    with (ARTIFACTS / "supervisor.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {
            "study_id": protocol["study_id"], "scientific_wall_limit_seconds": 900,
            "original_dry_preflight_seconds": 1.017155267,
            "original_dry_report": "dry-v1/report.json", "attempts": [],
        }
        if any(a["status"] == "RUNNING" for a in ledger["attempts"]):
            raise RuntimeError("an interrupted attempt needs explicit retained reconciliation")
        if any(a["status"] == "FAILED" for a in ledger["attempts"]):
            raise RuntimeError("lane stopped after a failed attempt; recorded review required")
        if args.phase != "dry" and not any(
            a["phase"] == "dry" and a["status"] == "COMPLETED"
            and a["protocol_sha256"] == hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
            and a["execution_source_hashes"] == source_hashes
            for a in ledger["attempts"]
        ):
            raise RuntimeError("development/confirmation requires a completed monitored dry pass")
        elapsed = ledger["original_dry_preflight_seconds"] + sum(
            a["elapsed_seconds"] for a in ledger["attempts"]
        )
        if elapsed + budget["per_process_seconds"] + 5 > budget["scientific_wall_seconds"]:
            raise RuntimeError("insufficient aggregate budget for another bounded attempt")
        if args.phase != "dry" and any(
            a["phase"] == args.phase and a["seed"] == args.seed
            and a["configuration"] == args.configuration for a in ledger["attempts"]
        ):
            raise RuntimeError("declared development/confirmation block already consumed")
        output = ARTIFACTS / args.attempt
        receipt_path = ARTIFACTS / (args.attempt + "-supervisor.json")
        log_path = ARTIFACTS / (args.attempt + ".log")
        if any(p.exists() for p in (output, receipt_path, log_path)):
            raise FileExistsError("attempt outputs must be fresh")
        before = memory_sample()
        if before["effective_available_estimate_bytes"] < budget["min_estimated_available_bytes"]:
            publish_json(receipt_path, {"status": "BLOCKED_BEFORE_LAUNCH", "preflight": before})
            raise RuntimeError("resource-v2 headroom floor is unavailable before launch")
        command = [sys.executable, str(ROOT / "scripts/run_observable_prediction.py"),
                   "--phase", args.phase, "--seed", str(args.seed),
                   "--configuration", args.configuration, "--output", str(output),
                   "--supervisor-pid", str(os.getpid())]
        if args.freeze is not None:
            command += ["--freeze", str(args.freeze.resolve())]
        record: dict[str, Any] = {
            "attempt": args.attempt, "phase": args.phase, "seed": args.seed,
            "configuration": args.configuration, "status": "RUNNING", "command": command,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
            "execution_source_hashes": source_hashes,
            "budget_used_before_seconds": elapsed, "preflight": before,
        }
        ledger["attempts"].append(record)
        publish_json(ledger_path, ledger)
        publish_json(receipt_path, record)
        environment = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                           OPENBLAS_NUM_THREADS="2")
        started = time.perf_counter()
        samples: list[dict[str, Any]] = []
        failure = None
        process: subprocess.Popen[str] | None = None
        with log_path.open("x") as log:
            try:
                process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True,
                                           text=True)
                while process.poll() is None:
                    sample = memory_sample(process.pid)
                    samples.append(sample)
                    if time.perf_counter() - started >= budget["per_process_seconds"]:
                        failure = "100-second child deadline reached"
                    elif sample["effective_available_estimate_bytes"] < budget[
                        "min_estimated_available_bytes"
                    ]:
                        failure = "resource-v2 remaining headroom floor crossed"
                    elif (sample["child_rss_bytes"] or 0) >= budget["max_child_rss_bytes"]:
                        failure = "sampled768MiB child RSS ceiling crossed"
                    elif sample["host_swap_fraction"] > 0.9:
                        failure = "host swap ceiling crossed"
                    elif any(sample["cgroup_events"].get(k, 0) > before["cgroup_events"].get(k, 0)
                             for k in ("oom", "oom_kill")):
                        failure = "cgroup OOM event observed"
                    if failure:
                        break
                    time.sleep(budget["monitor_interval_seconds"])
            except BaseException as exc:
                failure = "supervisor interrupted or failed: " + repr(exc)
            finally:
                if process is not None:
                    terminate_child(process)
        child_peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024
        if child_peak >= budget["max_child_rss_bytes"]:
            failure = "retained child high-water RSS exceeded768MiB"
        exit_code = process.returncode if process is not None else None
        record.update({
            "status": "COMPLETED" if exit_code == 0 and failure is None else "FAILED",
            "exit_code": exit_code, "monitor_failure": failure,
            "elapsed_seconds": time.perf_counter() - started, "samples": samples,
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
            "supervisor_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "largest_child_peak_rss_bytes": child_peak,
        })
        report = output / "report.json"
        record["report_sha256"] = (
            hashlib.sha256(report.read_bytes()).hexdigest() if report.exists() else None
        )
        publish_json(receipt_path, record)
        publish_json(ledger_path, ledger)
        return record


def main() -> None:
    def interrupted(signum: int, _: Any) -> None:
        raise InterruptedError(f"supervisor received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["dry", "development", "confirmation"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--configuration", choices=["a", "b"], required=True)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--freeze", type=Path)
    record = execute(parser.parse_args())
    print(json.dumps({k: record[k] for k in ["attempt", "status", "exit_code", "monitor_failure",
                                          "elapsed_seconds"]}, indent=2))
    if record["status"] != "COMPLETED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

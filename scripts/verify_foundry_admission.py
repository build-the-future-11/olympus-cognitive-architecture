#!/usr/bin/env python3
"""Run from the installed Olympus checkout: python this_script.py --output /tmp/fresh-run.

Engineering evidence only. No network requests, GPU work, external data or model-family
promotion. Checkpoint provenance includes a timestamp: hashes bind this run, not reruns.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

FLOOR = 2 * 1024**3
CORPUS_SHA = "6f4823e0503426bda11aa4bffe8357963f3bc97a62ea40c2b69700af166a8eab"


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise RuntimeError(reason)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def resources(repo: Path) -> Any:
    # Normal package import loads the API first; admit resources before that import.
    resource_path = repo / "olympus/foundry/resources.py"
    spec = importlib.util.spec_from_file_location("workflow_resources", resource_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("missing ResourceGovernor source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def cache_credit(current: Path, version: int) -> int:
    """Conservative available-memory estimate; keep raw cgroup usage in the receipt.

    Linux container working-set reporting discounts inactive file cache. We further
    subtract all potentially dirty, mapped, or shared bytes from that credit. Missing counters get
    no credit. This is an estimate, not a guarantee that reclamation is immediate.
    """
    path = current / "memory.stat"
    if not path.is_file():
        return 0
    stats = {
        key: int(value) for key, value in (line.split() for line in path.read_text().splitlines())
    }
    names = ("inactive_file", "file", "shmem", "file_mapped", "file_dirty", "file_writeback")
    if version == 1:
        names = (
            "total_inactive_file",
            "total_cache",
            "total_shmem",
            "total_mapped_file",
            "total_dirty",
            "total_writeback",
        )
    if not set(names).issubset(stats) or any(stats[key] < 0 for key in names):
        return 0
    inactive, cached, *unavailable = (stats[key] for key in names)
    return max(0, min(inactive, cached) - sum(unavailable))


def observe(module: Any) -> dict[str, Any]:
    result = module.memory_snapshot().as_dict()
    groups = []
    if platform.system() == "Linux":
        lines = Path("/proc/self/cgroup").read_text().splitlines()
        membership = [line.split(":", 2) for line in lines]
        for line in Path("/proc/self/mountinfo").read_text().splitlines():
            before, after = line.split(" - ", 1)
            fields, filesystem = before.split(), after.split()
            version = 2 if filesystem[0] == "cgroup2" else 1
            if filesystem[0] not in {"cgroup", "cgroup2"}:
                continue
            for _, controllers, location in membership:
                wrong_v1 = version == 1 and "memory" not in controllers.split(",")
                if (version == 2 and controllers) or wrong_v1:
                    continue
                mount = Path(fields[4])
                try:
                    current = mount / Path(location).relative_to(Path(fields[3]))
                except ValueError:
                    continue
                while current == mount or mount in current.parents:
                    names = {
                        2: ("memory.max", "memory.current"),
                        1: ("memory.limit_in_bytes", "memory.usage_in_bytes"),
                    }
                    limit_name, usage_name = names[version]
                    if (current / limit_name).is_file():
                        raw = (current / limit_name).read_text().strip()
                        used = int((current / usage_name).read_text())
                        limit = None if raw == "max" else int(raw)
                        groups.append(
                            {
                                "path": str(current),
                                "limit_bytes": limit,
                                "used_bytes": used,
                                "raw_headroom_bytes": None if limit is None else limit - used,
                                "cache_credit_bytes": min(used, cache_credit(current, version)),
                            }
                        )
                    if current == mount:
                        break
                    current = current.parent
        require(bool(groups), "Linux cgroup memory limits could not be observed")
    available = [result["available_bytes"]]
    available.extend(
        max(0, g["limit_bytes"] - g["used_bytes"] + g["cache_credit_bytes"])
        for g in groups
        if g["limit_bytes"] is not None
    )
    return {**result, "cgroups": groups, "effective_available_bytes": min(available)}


def admit(snapshot: dict[str, Any]) -> None:
    require(snapshot["effective_available_bytes"] >= FLOOR, "less than 2 GiB effective free memory")
    require(snapshot["swap_fraction"] <= 0.90, "swap utilization exceeds ResourceGovernor ceiling")


def execute(
    command: list[str],
    name: str,
    report: dict[str, Any],
    out: Path,
    repo: Path,
    env: dict[str, str],
    module: Any,
    deadline: float,
) -> str:
    before = observe(module)
    admit(before)
    start = time.monotonic()
    stdout, stderr = out / f"{name}.stdout", out / f"{name}.stderr"
    samples, failure = [before], None
    require(time.monotonic() < deadline, "workflow time budget exhausted before child launch")
    with stdout.open("wb") as stream, stderr.open("wb") as errors:
        proc = subprocess.Popen(
            command, cwd=repo, env=env, stdout=stream, stderr=errors, start_new_session=True
        )
        try:
            while True:
                pid, status, usage = os.wait4(proc.pid, os.WNOHANG)
                if pid:
                    break
                require(time.monotonic() < min(deadline, start + 30), "child time budget exhausted")
                samples.append(observe(module))
                admit(samples[-1])
                time.sleep(0.2)
        except BaseException as error:
            failure = f"{type(error).__name__}: {error}"
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            _, status, usage = os.wait4(proc.pid, 0)
        proc.returncode = os.waitstatus_to_exitcode(status)
    rss_factor = 1 if platform.system() == "Darwin" else 1024
    record = {
        "command": command,
        "cwd": str(repo),
        "exit_status": proc.returncode,
        "elapsed_seconds": time.monotonic() - start,
        "abort_reason": failure,
        "stdout": stdout.name,
        "stderr": stderr.name,
        "memory_samples": samples,
        "memory_after": observe(module),
        "child_user_seconds": usage.ru_utime,
        "child_system_seconds": usage.ru_stime,
        "child_major_page_faults": usage.ru_majflt,
        "child_peak_rss_bytes": int(usage.ru_maxrss * rss_factor),
    }
    report["commands"].append(record)
    save(out / "receipt.json", report)
    require(proc.returncode == 0 and failure is None, f"{name} failed; retained stdout/stderr")
    require(time.monotonic() < min(deadline, start + 30), "command time budget exhausted")
    return stdout.read_text()


def worker(arguments: list[str]) -> None:
    kind, root, model_id, checkpoint_id, expected, destination = arguments
    if kind == "export":
        from olympus.foundry.service import FoundryService

        with FoundryService(Path(root)) as service:
            health = service.model_health(model_id)
            exported = service.export_checkpoint(model_id, Path(destination))
            value = {"health": health, "export_path": str(exported), "sha256": digest(exported)}
        print(json.dumps(value))
        require(
            health.get("sha256") == expected and value["sha256"] == expected,
            "export identity mismatch",
        )
        require(health.get("checkpoint_id") == checkpoint_id, "reloaded checkpoint ID mismatch")
    elif kind == "asgi":
        import httpx

        from olympus.api import app

        async def requests() -> None:
            async with app.router.lifespan_context(app):
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(
                    transport=transport, trust_env=False, base_url="http://asgi.invalid"
                ) as client:
                    listing = await client.get("/v1/models")
                    completion = await client.post(
                        "/v1/chat/completions",
                        json={
                            "model": model_id,
                            "messages": [{"role": "user", "content": "verify evidence"}],
                            "max_tokens": 24,
                            "temperature": 0,
                            "seed": 7,
                        },
                    )
                    value = {
                        "models_status": listing.status_code,
                        "models": listing.json(),
                        "completion_status": completion.status_code,
                        "completion": completion.json(),
                    }
                    print(json.dumps(value))
                    require(
                        listing.status_code == completion.status_code == 200, "ASGI request failed"
                    )
                    entry = next(row for row in value["models"]["data"] if row["id"] == model_id)
                    health = entry["health"]
                    require(
                        health["status"] == "ok" and health["sha256"] == expected,
                        "ASGI listing did not admit the exact checkpoint",
                    )
                    evidence = value["completion"]["evidence"]
                    require(
                        evidence["checkpoint_id"] == checkpoint_id
                        and evidence["checkpoint_sha256"] == expected,
                        "ASGI completion checkpoint mismatch",
                    )

        asyncio.run(requests())
    else:
        raise ValueError("unknown worker mode")


def main() -> int:
    workflow_start = time.monotonic()
    deadline = workflow_start + 120
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    shared_lock = Path(tempfile.gettempdir()) / f"olympus-{os.getuid()}.workload.lock"
    parser.add_argument("--lock", type=Path, default=shared_lock)
    args = parser.parse_args()
    out, repo = args.output.absolute(), args.repository.resolve()
    out.mkdir(parents=True, exist_ok=False)  # An existing directory or symlink is never reused.
    report: dict[str, Any] = {
        "status": "RUNNING",
        "started_utc": datetime.now(UTC).isoformat(),
        "scope": "CPU bigram engineering; no capability or scientific claim",
        "checkpoint_identity": "timestamp-bound; hashes vary across reruns",
        "commands": [],
        "lock_path": str(args.lock),
        "total_budget_seconds": 120,
        "child_budget_seconds": 30,
        "headroom_policy": "v2: raw cgroup headroom plus clean inactive cache",
        "budget_enforcement": "polled deadlines; probe latency applies",
    }
    try:
        module = resources(repo)
        corpus = repo / "olympus/foundry/foundry_verification.txt"
        require(
            corpus.is_file() and not corpus.is_symlink() and corpus.stat().st_size <= 100_000,
            "only the bounded repository corpus is admitted",
        )
        require(digest(corpus) == CORPUS_SHA, "built-in corpus differs from the reviewed fixture")
        source_paths = sorted([*repo.glob("olympus/**/*.py"), repo / "pyproject.toml"])
        source_hashes = {str(p.relative_to(repo)): digest(p) for p in source_paths}
        report.update(
            {
                "source_sha256": source_hashes,
                "script_sha256": digest(Path(__file__)),
                "corpus_sha256": CORPUS_SHA,
                "corpus_bytes": corpus.stat().st_size,
                "python": sys.version,
                "executable": sys.executable,
                "platform": platform.platform(),
                "packages": {
                    p: importlib.metadata.version(p)
                    for p in ("torch", "fastapi", "httpx", "pydantic")
                },
                "device": "CPU character-bigram; CUDA/HIP masked; no tensor training",
                "memory_before": observe(module),
                "disk_free_before": shutil.disk_usage(out).free,
            }
        )
        admit(report["memory_before"])
        require(report["disk_free_before"] >= 64 * 1024**2, "less than 64 MiB free disk headroom")
        env = {
            **os.environ,
            "PYTHONPATH": str(repo),
            "PYTHONDONTWRITEBYTECODE": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "HIP_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "NO_COLOR": "1",
            "TERM": "dumb",
            "OLYMPUS_FOUNDRY_ROOT": str(out / "registry"),
        }
        with module.ResourceGovernor(args.lock, min_available_bytes=FLOOR):

            def run(command: list[str], name: str) -> str:
                return execute(command, name, report, out, repo, env, module, deadline)

            report["git_head"] = run(["git", "rev-parse", "HEAD"], "git-head").strip()
            report["git_status"] = run(
                ["git", "status", "--porcelain", "--untracked-files=normal"], "git-status"
            )
            cli = [sys.executable, "-m", "olympus.cli", "foundry"]
            root_arg = ["--root", str(out / "registry")]
            result = json.loads(run([*cli, "verify-pipeline", *root_arg], "pipeline"))
            report["pipeline"] = result
            model_id = result["model"]["model_id"]
            checkpoint_id = result["checkpoint"]["checkpoint_id"]
            evaluation = result["evaluation"]
            expected = result["checkpoint"]["sha256"]
            require(
                result["model"]["family"] == "FoundryVerificationBigram" and evaluation["passed"],
                "verification model was not admitted by its quality gate",
            )
            for key in ("negative_log_likelihood", "perplexity"):
                require(
                    evaluation["candidate_metrics"][key] < evaluation["baseline_metrics"][key],
                    "candidate failed the frozen held-out comparison",
                )
            for path in (result["checkpoint"]["path"], result["export_path"]):
                require(digest(Path(path)) == expected, "pipeline artifact digest mismatch")
            require(result["dataset"]["sha256"] == CORPUS_SHA, "consumed dataset mismatch")
            report["status_after_reload"] = json.loads(run([*cli, "status", *root_arg], "status"))
            models = json.loads(run([*cli, "list-models", *root_arg], "models"))
            report["models_after_reload"] = models
            require(
                len(models) == 1 and models[0]["checkpoint_id"] == checkpoint_id,
                "reloaded registry does not identify the admitted checkpoint",
            )
            require(report["status_after_reload"]["integrity"] == "ok", "SQLite integrity failed")
            generate = [
                *cli,
                "generate",
                model_id,
                "user: verify evidence",
                *root_arg,
                "--max-characters",
                "24",
                "--temperature",
                "0",
                "--seed",
                "7",
            ]
            generated = json.loads(run(generate, "generate"))
            report["generation"] = generated
            require(
                generated["evidence"]["checkpoint_sha256"] == expected,
                "CLI serving identity mismatch",
            )
            for mode in ("export", "asgi"):
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    mode,
                    str(out / "registry"),
                    model_id,
                    checkpoint_id,
                    expected,
                    str(out / "portable-model.json"),
                ]
                report[mode] = json.loads(run(command, mode))
            api_text = report["asgi"]["completion"]["choices"][0]["message"]["content"]
            require(
                api_text == generated["content"],
                "fresh API and CLI outputs disagree for identical prompt/settings",
            )
            report["memory_after"] = observe(module)
            report["swap_used_change_bytes"] = (
                report["memory_after"]["swap_used_bytes"]
                - report["memory_before"]["swap_used_bytes"]
            )
            report["disk_free_after"] = shutil.disk_usage(out).free
            require(
                source_hashes == {str(p.relative_to(repo)): digest(p) for p in source_paths},
                "source changed during the workflow",
            )
            require(time.monotonic() < deadline, "workflow time budget exhausted")
            report["status"] = "PASSED_ENGINEERING_ONLY"
    except BaseException as error:
        report.update(status="FAILED", failure=f"{type(error).__name__}: {error}")
    finally:
        report["finished_utc"] = datetime.now(UTC).isoformat()
        report["elapsed_seconds"] = time.monotonic() - workflow_start
        report["artifact_sha256"] = {
            str(p.relative_to(out)): digest(p)
            for p in out.rglob("*")
            if p.is_file() and p.name != "receipt.json"
        }
        save(out / "receipt.json", report)
    print(json.dumps({"status": report["status"], "receipt": str(out / "receipt.json")}))
    return 0 if report["status"] == "PASSED_ENGINEERING_ONLY" else 1


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    if sys.argv[1:2] == ["--worker"]:
        worker(sys.argv[2:])
    else:
        raise SystemExit(main())

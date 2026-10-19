"""Bounded engineering verification on fresh local artifacts; no capability campaign.

Run with the installed Python environment from any directory:
    python scripts/verify_foundry_integrity.py --output artifacts/integrity-attempt-1
Every attempt requires a new directory. The API binds only to loopback and is stopped
before exit. Linux and macOS are supported; only the recorded host is a measured result.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
PROMPT = "user: request verify evidence"
GENERATION = {"max_characters": 80, "temperature": 0.7, "seed": 7}
GIB = 1024**3


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rss_bytes(usage: resource.struct_rusage) -> int:
    return int(usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024))


def cgroup_number(name: str) -> int | None:
    path = Path("/sys/fs/cgroup") / name
    if not path.is_file():
        return None
    raw = path.read_text().strip()
    return int(raw) if raw.isdecimal() else None


def host_snapshot(output: Path = REPOSITORY) -> dict[str, Any]:
    from olympus.foundry.resources import memory_snapshot

    memory = memory_snapshot().as_dict()
    limit, current = cgroup_number("memory.max"), cgroup_number("memory.current")
    stat_path = Path("/sys/fs/cgroup/memory.stat")
    stat = (
        {
            key: int(value)
            for key, value in (line.split() for line in stat_path.read_text().splitlines())
        }
        if stat_path.is_file()
        else {}
    )
    # A working-set estimate, not a guaranteed allocation reservation. Credit only
    # inactive file cache, conservatively excluding all dirty/writeback pages.
    # https://docs.docker.com/reference/cli/docker/container/stats/
    reclaimable = max(
        0, stat.get("inactive_file", 0) - stat.get("file_dirty", 0) - stat.get("file_writeback", 0)
    )
    available = int(memory["available_bytes"])
    if limit is not None and current is not None:
        available = min(available, max(0, limit - current + reclaimable))
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "visible_cpus": os.cpu_count(),
        "memory": memory,
        "cgroup_memory_limit_bytes": limit,
        "cgroup_memory_current_bytes": current,
        "cgroup_unallocated_bytes": max(0, limit - current)
        if limit is not None and current is not None
        else None,
        "cgroup_inactive_file_bytes": stat.get("inactive_file"),
        "cgroup_reclaimable_estimate_bytes": reclaimable,
        "cgroup_swap_current_bytes": cgroup_number("memory.swap.current"),
        "effective_available_estimate_bytes": available,
        "memory_accounting": "host-and-cgroup-working-set-v2",
        "disk_path": str(output),
        "disk_free_bytes": shutil.disk_usage(output).free,
        "device": "CPU",
        "numeric_library_thread_request": 2,
    }


def source_identity() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=REPOSITORY, text=True, timeout=5).strip()

    paths = [REPOSITORY / "pyproject.toml", REPOSITORY / "apps/forge-web/package-lock.json"]
    for directory in ("olympus", "tests", "scripts", "apps/forge-web/src"):
        paths.extend(
            path
            for path in (REPOSITORY / directory).rglob("*")
            if path.is_file() and path.suffix in {".py", ".txt", ".ts", ".tsx", ".css"}
        )
    hashes = {str(path.relative_to(REPOSITORY)): digest(path) for path in sorted(paths)}
    return {
        "repository": "build-the-future-11/olympus-cognitive-architecture",
        "branch": git("branch", "--show-current"),
        "base_commit": git("rev-parse", "HEAD"),
        "working_tree": git("status", "--porcelain", "--untracked-files=normal"),
        "files_sha256": hashes,
        "files_manifest_sha256": hashlib.sha256(
            json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def recover(root: Path, export: Path, *, corrupt: bool) -> None:
    from olympus.foundry.service import FoundryService

    with FoundryService(root) as service:
        model = service.list_models()[0]
        health = service.model_health(model.model_id)
        result: dict[str, Any] = {"health": health, "registry": service.status()}
        assert result["registry"]["integrity"] == "ok"
        if corrupt:
            assert health["status"] == "corrupt"
            before = export.read_bytes()
            try:
                service.export_checkpoint(model.model_id, export)
            except RuntimeError as error:
                result["export_error"] = str(error)
            else:
                raise AssertionError("corrupt checkpoint was exported")
            assert export.read_bytes() == before
            result["existing_destination_preserved"] = True
        else:
            assert health["status"] == "ok"
            result["generation"] = service.generate(
                model.model_id, PROMPT, **GENERATION
            ).model_dump(mode="json")
            service.export_checkpoint(model.model_id, export)
            assert digest(export) == health["sha256"]
            result["export_sha256"] = digest(export)
        result["process_peak_rss_bytes"] = rss_bytes(resource.getrusage(resource.RUSAGE_SELF))
        print(json.dumps(result))


def serve(root: Path, ready: Path) -> None:
    import uvicorn

    os.environ["OLYMPUS_FOUNDRY_ROOT"] = str(root)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(128)
        write_json(ready, {"pid": os.getpid(), "port": listener.getsockname()[1]})
        server = uvicorn.Server(uvicorn.Config("olympus.api:app", log_level="warning"))
        server.run(sockets=[listener])


class Workflow:
    def __init__(self, output: Path) -> None:
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.started = time.monotonic()
        self.deadline = self.started + 120
        self.environment = {
            **os.environ,
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "NO_COLOR": "1",
        }
        self.report: dict[str, Any] = {
            "schema_version": 1,
            "study_id": "foundry-artifact-admission-v1-20261019",
            "started_utc": datetime.now(UTC).isoformat(),
            "scope": "Infrastructure verification; not Hermes or capability evidence.",
            "commands": [],
            "http": [],
            "checks": {},
            "budget": {"seconds": 120, "compute_lanes": 1, "threads": 2},
            "generation_config": {"prompt": PROMPT, **GENERATION},
        }

    def wait(self, process: subprocess.Popen[bytes], deadline: float) -> dict[str, Any]:
        timed_out = False
        while True:
            pid, status, usage = os.wait4(process.pid, os.WNOHANG)
            if pid:
                process.returncode = os.waitstatus_to_exitcode(status)
                return {
                    "exit_code": process.returncode,
                    "timed_out": timed_out,
                    "peak_process_rss_bytes": rss_bytes(usage),
                    "user_cpu_seconds": usage.ru_utime,
                    "system_cpu_seconds": usage.ru_stime,
                }
            if time.monotonic() >= deadline:
                timed_out = True
                os.killpg(process.pid, signal.SIGKILL)
            time.sleep(0.02)

    def command(self, name: str, command: list[str], expected: int = 0) -> Any:
        self.preflight(name)
        start = time.monotonic()
        stdout, stderr = self.output / f"{name}.stdout", self.output / f"{name}.stderr"
        with stdout.open("wb") as out, stderr.open("wb") as err:
            process = subprocess.Popen(
                command,
                cwd=REPOSITORY,
                env=self.environment,
                stdout=out,
                stderr=err,
                start_new_session=True,
            )
            observed = self.wait(process, min(self.deadline, start + 30))
        self.report["commands"].append(
            {
                "name": name,
                "argv": command,
                "expected_exit_code": expected,
                "elapsed_seconds": time.monotonic() - start,
                "stdout": stdout.name,
                "stderr": stderr.name,
                **observed,
            }
        )
        assert observed["exit_code"] == expected and not observed["timed_out"], name
        return json.loads(stdout.read_bytes()) if expected == 0 else None

    def request(self, port: int, path: str, payload: Any = None) -> tuple[int, Any]:
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}",
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Content-Type": "application/json"},
        )
        start = time.monotonic()
        try:
            response = urllib.request.urlopen(request, timeout=3)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            status, body = response.status, json.loads(response.read())
        self.report["http"].append(
            {
                "path": path,
                "request": payload,
                "status": status,
                "body": body,
                "elapsed_seconds": time.monotonic() - start,
            }
        )
        return status, body

    def api(self, name: str, root: Path, model: str, expected: dict[str, Any] | None) -> None:
        self.preflight(name)
        ready = self.output / f"{name}-ready.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--serve",
            str(root),
            "--ready",
            str(ready),
        ]
        start = time.monotonic()
        with (
            (self.output / f"{name}.stdout").open("wb") as out,
            (self.output / f"{name}.stderr").open("wb") as err,
        ):
            process = subprocess.Popen(
                command,
                cwd=REPOSITORY,
                env=self.environment,
                stdout=out,
                stderr=err,
                start_new_session=True,
            )
            try:
                startup_deadline = min(self.deadline, start + 15)
                while True:
                    if time.monotonic() >= startup_deadline:
                        raise TimeoutError("local API did not become ready within its budget")
                    try:
                        port = json.loads(ready.read_bytes())["port"]
                        if self.request(port, "/health")[0] == 200:
                            break
                    except OSError, ValueError:
                        time.sleep(0.05)
                status, models = self.request(port, "/v1/models")
                assert status == 200 and len(models["data"]) == 1
                row = models["data"][0]
                assert row["id"] == model and row["available"] == (expected is not None)
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": "request verify evidence"}],
                    "max_tokens": GENERATION["max_characters"],
                    "temperature": GENERATION["temperature"],
                    "seed": GENERATION["seed"],
                    "stream": False,
                }
                status, body = self.request(port, "/v1/chat/completions", payload)
                if expected is None:
                    assert row["health"]["status"] == "corrupt" and status == 422
                else:
                    assert status == 200 and body["model"] == model
                    assert body["choices"][0]["message"]["content"] == expected["content"]
                    assert body["evidence"] == expected["evidence"]
                    assert body["usage"]["unit"] == "characters"
                unknown = {**payload, "model": "nonexistent-integrity-fixture"}
                assert self.request(port, "/v1/chat/completions", unknown)[0] == 404
            finally:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                observed = self.wait(process, min(self.deadline, time.monotonic() + 5))
                self.report["commands"].append(
                    {
                        "name": name,
                        "argv": command,
                        "shutdown": "parent requested SIGTERM",
                        "elapsed_seconds": time.monotonic() - start,
                        **observed,
                    }
                )
                assert observed["exit_code"] in {0, -signal.SIGTERM}
                assert not observed["timed_out"]

    def preflight(self, stage: str) -> dict[str, Any]:
        snapshot = host_snapshot(self.output)
        self.report.setdefault("resource_preflights", []).append({"stage": stage, **snapshot})
        assert snapshot["effective_available_estimate_bytes"] >= 2 * GIB, "2 GiB headroom required"
        assert snapshot["disk_free_bytes"] >= GIB, "1 GiB disk headroom required"
        assert snapshot["memory"]["swap_fraction"] < 0.9, "swap budget exceeded"
        return snapshot

    def execute(self) -> None:
        before = self.preflight("workflow")
        self.report["host_before"] = before
        self.report["source"] = source_identity()
        self.report["versions"] = {
            package: importlib.metadata.version(package)
            for package in ("torch", "numpy", "pydantic", "fastapi", "uvicorn", "httpx")
        }
        corpus = REPOSITORY / "olympus/foundry/foundry_verification.txt"
        assert 2 * len(corpus.read_text()) <= 2000, "fixture character budget exceeded"
        prefix = [sys.executable, "-m", "olympus.cli", "foundry"]
        healthy_root = self.output / "healthy"
        result = self.command("pipeline", [*prefix, "verify-pipeline", "--root", str(healthy_root)])
        write_json(self.output / "pipeline.json", result)
        self.report["pipeline"] = result
        assert result["evaluation"]["passed"] is True
        assert result["model"]["family"] == "FoundryVerificationBigram"
        model = result["model"]["model_id"]
        checkpoint = Path(result["checkpoint"]["path"])
        assert digest(checkpoint) == result["checkpoint"]["sha256"]
        assert digest(Path(result["export_path"])) == digest(checkpoint)
        text = Path(result["dataset"]["materialized_path"]).read_text()
        split = result["experiment"]["config"]["training_characters"]
        assert set(text[split:]).issubset(set(text[:split]))
        self.report["split"] = {
            "training_character_range": [0, split],
            "training_target_positions": [1, split],
            "held_out_target_positions": [split, len(text)],
            "range_convention": "zero-based, end exclusive",
            "held_out_unknown_characters": 0,
            "frozen_baseline": "uniform distribution over training alphabet",
        }
        generated = self.command(
            "cli-generate",
            [
                *prefix,
                "generate",
                "--root",
                str(healthy_root),
                model,
                PROMPT,
                "--max-characters",
                "80",
                "--temperature",
                "0.7",
                "--seed",
                "7",
            ],
        )
        assert generated["model"] == model
        assert generated["evidence"]["checkpoint_id"] == result["checkpoint"]["checkpoint_id"]
        assert generated["evidence"]["checkpoint_sha256"] == result["checkpoint"]["sha256"]
        recovered = self.command(
            "recover-export",
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--recover",
                str(healthy_root),
                "--export",
                str(self.output / "verified-model.json"),
            ],
        )
        assert recovered["generation"] == generated
        assert recovered["export_sha256"] == result["checkpoint"]["sha256"]
        self.api("healthy-api", healthy_root, model, generated)
        self.report["checks"].update(
            {
                "actual_training_and_held_out_gate": True,
                "checkpoint_internal_and_portable_exports_identical": True,
                "fresh_process_registry_recovery": True,
                "cli_and_fresh_api_generation_identical": True,
            }
        )
        failure_root = self.output / "failure-fixture"
        failure = self.command(
            "failure-pipeline",
            [
                *prefix,
                "verify-pipeline",
                "--root",
                str(failure_root),
            ],
        )
        failure_checkpoint = Path(failure["checkpoint"]["path"])
        shutil.copyfile(failure_checkpoint, self.output / "failure-before-corruption.json")
        failure_checkpoint.write_bytes(
            failure_checkpoint.read_bytes() + b"\nintentional-corruption"
        )
        sentinel = self.output / "preserved-export.json"
        sentinel.write_bytes(b"Existing destination must survive rejected export.\n")
        self.command(
            "reject-corrupt-export",
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--recover",
                str(failure_root),
                "--export",
                str(sentinel),
                "--corrupt",
            ],
        )
        self.command(
            "reject-corrupt-cli",
            [
                *prefix,
                "generate",
                "--root",
                str(failure_root),
                failure["model"]["model_id"],
                PROMPT,
            ],
            expected=1,
        )
        self.api("corrupt-api", failure_root, failure["model"]["model_id"], None)
        assert digest(checkpoint) == result["checkpoint"]["sha256"]
        self.report["checks"].update(
            {
                "corrupt_fixture_unavailable": True,
                "corrupt_api_generation_http_422": True,
                "unknown_model_http_404": True,
                "corrupt_cli_generation_exit_1": True,
                "rejected_export_preserves_destination": True,
                "healthy_checkpoint_preserved": True,
            }
        )
        self.report["model_memory_scope"] = {
            "serialized_checkpoint_bytes": checkpoint.stat().st_size,
            "optimizer": "Not applicable: exact bigram counts; no gradient optimizer.",
            "activations": "No neural tensors; Python counts and probabilities only.",
            "kv_cache": "Not applicable to the character-bigram runtime.",
            "total_memory": "Process wait4 peak RSS includes interpreter and imported libraries.",
            "limitation": "Not a Transformer or 16 GiB Mac memory-efficiency measurement.",
        }


def main() -> int:
    import olympus

    assert Path(olympus.__file__).resolve() == REPOSITORY / "olympus/__init__.py"
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--recover", type=Path)
    mode.add_argument("--serve", type=Path)
    parser.add_argument("--export", type=Path)
    parser.add_argument("--ready", type=Path)
    parser.add_argument("--corrupt", action="store_true")
    args = parser.parse_args()
    if args.recover:
        if args.export is None:
            parser.error("--recover requires --export")
        recover(args.recover, args.export, corrupt=args.corrupt)
        return 0
    if args.serve:
        if args.ready is None:
            parser.error("--serve requires --ready")
        serve(args.serve, args.ready)
        return 0
    workflow = Workflow(args.output)
    lock = REPOSITORY / "artifacts/foundry-integrity-workload.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    passed = False
    try:
        with lock.open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            workflow.execute()
            passed = True
    except Exception as error:
        workflow.report["failure"] = {"type": type(error).__name__, "message": str(error)}
        (workflow.output / "failure.log").write_text(traceback.format_exc())
    finally:
        workflow.report.update(
            {
                "passed": passed,
                "elapsed_seconds": time.monotonic() - workflow.started,
                "finished_utc": datetime.now(UTC).isoformat(),
                "host_after": host_snapshot(workflow.output),
                "orchestrator_peak_rss_bytes": rss_bytes(resource.getrusage(resource.RUSAGE_SELF)),
            }
        )
        workflow.report["artifact_inventory"] = {
            str(path.relative_to(workflow.output)): {
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
            for path in sorted(workflow.output.rglob("*"))
            if path.is_file()
        }
        write_json(workflow.output / "workflow.json", workflow.report)
        print(json.dumps({"passed": passed, "report": str(workflow.output / "workflow.json")}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

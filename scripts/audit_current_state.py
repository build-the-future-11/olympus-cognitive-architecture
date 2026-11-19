"""Bounded, offline artifact inventory; it never loads or promotes a model.

Run with Python 3.11+ from any directory:
    python scripts/audit_current_state.py --repo /path/to/Olympus

Only the new report directory is written. Review reports before sharing them:
local paths and artifact identities may be private even though contents are omitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import selectors
import stat
import subprocess
import time
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

ARTIFACT_DIRS = ("artifacts", "checkpoints", "models", "exports", "runs", "releases")
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".cache"}
WEIGHT_SUFFIXES = {".safetensors", ".gguf", ".pt", ".pth", ".ckpt", ".bin", ".onnx"}
JSON_NAMES = {
    "manifest.json", "config.json", "adapter_config.json", "release-manifest.json",
    "model.safetensors.index.json", "pytorch_model.bin.index.json", "promotion.json",
    "evaluation.json", "quantization-report.json",
}
HASH_KEYS = {
    "checkpoint_sha256", "dataset_manifest_sha256", "manifest_sha256",
    "source_checkpoint_sha256", "evaluation_sha256", "model_card_sha256",
}
MAX_JSON_BYTES = 256 * 1024
MAX_DEPTH = 8
MAX_ENTRIES = 20_000
MAX_ARTIFACTS = 2_000
MAX_GIT_BYTES = 128 * 1024
MAX_METADATA_BYTES = 8 * 1024 * 1024
MAX_SHARDS = 512
IST = timezone(timedelta(hours=5, minutes=30))


def bounded_command(command: list[str], environment: dict[str, str]) -> dict[str, Any]:
    """Bound command time/output, killing only this audit-owned child if necessary."""
    process = None
    try:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=environment,
        )
        if process.stdout is None:
            raise RuntimeError("missing command pipe")
        deadline = time.monotonic() + 5
        output = bytearray()
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return {"status": "NOT_CHECKED", "reason": "TIMEOUT"}
                for key, _ in selector.select(timeout=remaining):
                    chunk = os.read(key.fd, min(8192, MAX_GIT_BYTES + 1 - len(output)))
                    if not chunk:
                        selector.unregister(key.fd)
                        continue
                    output.extend(chunk)
                    if len(output) > MAX_GIT_BYTES:
                        return {"status": "NOT_CHECKED", "reason": "OUTPUT_LIMIT"}
        code = process.wait(timeout=max(0.01, deadline - time.monotonic()))
        if code:
            return {"status": "NOT_CHECKED", "exit_code": code}
        return {"status": "VERIFIED_READ", "output": output.decode("utf-8", "replace").strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"status": "NOT_CHECKED", "reason": type(error).__name__}
    finally:
        if process is not None:
            if process.poll() is None:
                process.kill()
            process.wait()
            if process.stdout is not None:
                process.stdout.close()


def git_read(repo: Path, arguments: list[str]) -> dict[str, Any]:
    """Run only caller-selected read commands; never fetch or refresh the index."""
    environment = dict(os.environ, GIT_OPTIONAL_LOCKS="0", GIT_TERMINAL_PROMPT="0")
    command = [
        "git", "-c", "core.fsmonitor=false", "-c", "core.hooksPath=/dev/null",
        "-C", str(repo), *arguments,
    ]
    return bounded_command(command, environment)


def safe_remote(raw: str) -> str:
    """Retain host/repository identity, not credentials, queries, or fragments."""
    raw = raw.strip()
    if "://" not in raw:
        match = re.fullmatch(r"(?:[^@\s]+@)?([\w.-]+):([\w./-]+)", raw)
        return f"{match[1]}:{match[2]}" if match else "[local or unrecognized remote omitted]"
    try:
        parts = urlsplit(raw)
        if parts.scheme not in {"https", "http", "ssh", "git"} or not parts.hostname:
            return "[local or unrecognized remote omitted]"
        path = parts.path if re.fullmatch(r"/[\w./-]*", parts.path) else "/[path omitted]"
        return f"{parts.scheme}://{parts.hostname}{path}"
    except ValueError:
        return "[unrecognized remote omitted]"


def read_prefix(path: Path, limit: int) -> bytes:
    """Bound reads and reject final-component symlinks and non-regular files."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("not a regular file")
        return stream.read(limit)


def reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite JSON number")
    return number


def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def shard_state(base: Path, name: str) -> str:
    candidate = Path(name)
    if candidate.is_absolute() or not candidate.parts or ".." in candidate.parts:
        return "UNSAFE_PATH"
    current = base
    try:
        for part in candidate.parts:
            current /= part
            if current.is_symlink():
                return "SYMLINK_NOT_FOLLOWED"
        info = current.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_size == 0:
            return "EMPTY_OR_NOT_REGULAR"
        if read_prefix(current, 128).startswith(b"version https://git-lfs.github.com/spec/v1"):
            return "LFS_POINTER_ONLY"
        return "PRESENT_NOT_LOADED"
    except FileNotFoundError:
        return "MISSING"
    except (OSError, ValueError):
        return "UNREADABLE"


def inspect_json(path: Path) -> dict[str, Any]:
    raw = read_prefix(path, MAX_JSON_BYTES + 1)
    if len(raw) > MAX_JSON_BYTES:
        return {"json_status": "SKIPPED_SIZE_LIMIT"}
    try:
        value = json.loads(
            raw, parse_constant=reject_constant, parse_float=finite_float,
            object_pairs_hook=unique_pairs,
        )
    except (ValueError, UnicodeError, RecursionError):
        return {"json_status": "INVALID_JSON"}
    if not isinstance(value, dict):
        return {"json_status": "NOT_AN_OBJECT"}
    result: dict[str, Any] = {
        "json_status": "PARSED_NOT_VERIFIED", "file_sha256": hashlib.sha256(raw).hexdigest(),
        "hash_claims_not_verified": {
            key: item for key, item in value.items()
            if key in HASH_KEYS and isinstance(item, str) and re.fullmatch(r"[0-9a-f]{64}", item)
        },
    }
    # Only retain a narrowly constrained status claim, never arbitrary prompt/config data.
    status_claim = value.get("status")
    if status_claim in ("PROMOTED", "NOT_PROMOTED", "FAILED", "BLOCKED", "COMPLETE"):
        result["status_claim_not_verified"] = status_claim
    weight_map = value.get("weight_map")
    if weight_map is not None:
        if not isinstance(weight_map, dict) or not weight_map or not all(
            isinstance(item, str) for item in weight_map.values()
        ):
            result["shard_index_status"] = "INVALID_WEIGHT_MAP"
        else:
            names = sorted(set(weight_map.values()))
            if len(names) > MAX_SHARDS:
                result["shard_index_status"] = "SKIPPED_SHARD_LIMIT"
                return result
            result["shards"] = [
                {"path": name, "state": shard_state(path.parent, name)} for name in names
            ]
            result["shard_index_status"] = (
                "FILES_PRESENT_NOT_LOAD_VERIFIED"
                if all(item["state"] == "PRESENT_NOT_LOADED" for item in result["shards"])
                else "INCOMPLETE_OR_UNSAFE"
            )
    return result


def scan_artifacts(repo: Path, max_entries: int = MAX_ENTRIES) -> dict[str, Any]:
    if max_entries < 1:
        raise ValueError("max_entries must be positive")
    entries = 0
    metadata_bytes = 0
    found: list[dict[str, Any]] = []
    limits: list[str] = []
    errors: list[str] = []
    stack = [(repo / name, 0) for name in reversed(ARTIFACT_DIRS)]
    while stack and entries < max_entries and len(found) < MAX_ARTIFACTS:
        directory, depth = stack.pop()
        if directory.is_symlink():
            errors.append(f"symlink not followed: {directory.relative_to(repo)}")
            continue
        if not directory.exists():
            continue
        try:
            with os.scandir(directory) as iterator:
                for entry in iterator:
                    entries += 1
                    if entries > max_entries or len(found) >= MAX_ARTIFACTS:
                        limits.append("entry or artifact limit reached")
                        break
                    path = Path(entry.path)
                    relative = str(path.relative_to(repo))
                    if entry.is_symlink():
                        errors.append(f"symlink not followed: {relative}")
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        if entry.name not in SKIP_DIRS:
                            if depth < MAX_DEPTH:
                                stack.append((path, depth + 1))
                            else:
                                limits.append(f"depth limit: {relative}")
                        continue
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    if path.suffix.lower() not in WEIGHT_SUFFIXES and path.name not in JSON_NAMES:
                        continue
                    try:
                        info = entry.stat(follow_symlinks=False)
                        item: dict[str, Any] = {
                            "path": relative, "bytes": info.st_size,
                            "mtime_utc": datetime.fromtimestamp(info.st_mtime, UTC).isoformat(),
                            "state": "PRESENT_NOT_LOADED",
                        }
                        prefix = read_prefix(path, 128)
                        if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
                            item["state"] = "LFS_POINTER_ONLY"
                        elif info.st_size == 0:
                            item["state"] = "EMPTY_FILE"
                        elif path.name in JSON_NAMES:
                            cost = min(info.st_size, MAX_JSON_BYTES + 1)
                            if metadata_bytes + cost > MAX_METADATA_BYTES:
                                item["json_status"] = "SKIPPED_TOTAL_METADATA_LIMIT"
                                limits.append("total metadata read budget reached")
                            else:
                                metadata_bytes += cost
                                item.update(inspect_json(path))
                        found.append(item)
                    except (OSError, ValueError) as error:
                        errors.append(f"{relative}: {type(error).__name__}")
        except OSError as error:
            errors.append(f"{directory.relative_to(repo)}: {type(error).__name__}")
    if stack or entries >= max_entries or len(found) >= MAX_ARTIFACTS:
        limits.append("scan limit reached; remaining paths not inspected")
    return {
        "roots": list(ARTIFACT_DIRS), "entries_examined": min(entries, max_entries),
        "artifacts": sorted(found, key=lambda item: item["path"]),
        "limitations": sorted(set(limits)), "errors": errors,
        "scope_complete": not limits and not errors,
    }


def collect(repo: Path) -> dict[str, Any]:
    repo = repo.resolve(strict=True)
    root_check = git_read(repo, ["rev-parse", "--show-toplevel"])
    if root_check.get("output") != str(repo):
        raise ValueError("--repo must name the actual Git working-tree root")
    now = datetime.now(UTC)
    remotes = git_read(repo, ["config", "--get-regexp", r"^remote\..*\.url$"])
    remote_rows = []
    for line in remotes.get("output", "").splitlines():
        key, _, value = line.partition(" ")
        remote_rows.append({"name": key, "url": safe_remote(value)})
    # Do not retain raw config output, including credential-bearing URLs.
    return {
        "schema_version": 1, "audit_time_utc": now.isoformat(),
        "audit_time_ist": now.astimezone(IST).isoformat(), "workspace": str(repo),
        "head": git_read(repo, ["rev-parse", "HEAD"]),
        "branch": git_read(repo, ["branch", "--show-current"]),
        "worktree_status": git_read(repo, ["status", "--porcelain=v1"]),
        "remotes": remote_rows,
        "remote_config_read": {key: value for key, value in remotes.items() if key != "output"},
        "recent_commits": git_read(repo, ["log", "-10", "--format=%H %cI"]),
        "artifact_inventory": scan_artifacts(repo),
        "verification": {
            "remote_current_head": "NOT_CHECKED_OFFLINE",
            "training_completed": "NOT_ESTABLISHED_BY_FILE_PRESENCE",
            "runtime_generation": "NOT_RUN",
            "capability_evaluation": "NOT_RUN",
            "ghost_implementation": "REQUIRES_SOURCE_AND_RUNTIME_REVIEW",
            "model_release_readiness": "NOT_ESTABLISHED",
        },
        "limitations": [
            "Only this Git workspace and six named artifact directories were inspected.",
            "Unpushed files may exist; Git status is a snapshot, not a transaction.",
            "External caches, other worktrees, cloud runs and remote branches were not inspected.",
            "Weight files were not loaded, deserialized, fully hashed or functionally verified.",
            "JSON hashes identify the read files; embedded hashes/statuses are unverified claims.",
            "File timestamps are not training dates. Old ledgers are not current run evidence.",
            "This command does not establish a hard system-wide RAM limit.",
        ],
    }


def write_report(repo: Path, report: dict[str, Any]) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    parent = repo / "reports" / "olympus-audit"
    # Refuse symlinked report parents rather than writing outside the workspace.
    for directory in (repo / "reports", parent):
        if directory.is_symlink():
            raise ValueError("report parent must not be a symlink")
        directory.mkdir(exist_ok=True)
    destination = parent / stamp
    destination.mkdir(mode=0o700, exist_ok=False)
    evidence = destination / "EVIDENCE.json"
    with evidence.open("x", encoding="utf-8") as stream:
        os.chmod(evidence, 0o600)
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    count = len(report["artifact_inventory"]["artifacts"])
    summary = destination / "CURRENT_STATUS.md"
    with summary.open("x", encoding="utf-8") as stream:
        os.chmod(summary, 0o600)
        stream.write(
            "# Olympus current-state evidence snapshot\n\n"
            f"Audit time (IST): {report['audit_time_ist']}\n\n"
            f"Candidate artifact files found in scoped directories: **{count}**.\n\n"
            "**File presence does not prove training, successful loading, model quality, "
            "Ghost implementation, or release readiness.**\n\n"
            "Read `EVIDENCE.json` for exact source identity, dirty-worktree state, artifact "
            "inventory, unresolved shards, LFS pointers, scan limits and errors.\n\n"
            "## Required follow-through\n\n"
            "Bind each candidate to its training run and evaluation; inspect changed code; "
            "compare authenticated remote HEAD; then perform only explicitly admitted, "
            "bounded runtime checks on the target hardware. Keep old ledgers historical.\n"
        )
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    arguments = parser.parse_args()
    if os.name != "posix":
        print("Audit stopped: this collector supports macOS/Linux only.")
        return 2
    try:
        repo = arguments.repo.resolve(strict=True)
        report = collect(repo)
        destination = write_report(repo, report)
    except (OSError, ValueError) as error:
        print(f"Audit stopped: {type(error).__name__}: {error}")
        return 2
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

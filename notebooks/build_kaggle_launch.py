"""Preserve the superseded Olympus notebook draft; execution is explicitly blocked."""

# ruff: noqa: E501 -- Embedded notebook cell source is formatted for notebook readability.

from __future__ import annotations

import argparse
import hashlib
import json
import textwrap
from pathlib import Path
from typing import Any


def build(bundle: Path, output: Path) -> None:
    cells: list[dict[str, Any]] = []

    def md(value: str) -> None:
        cells.append(
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": textwrap.dedent(value).strip() + "\n",
            }
        )

    def code(value: str) -> None:
        value = textwrap.dedent(value).strip() + "\n"
        compile(value, "<notebook-cell>", "exec")
        cells.append(
            {
                "cell_type": "code",
                "metadata": {},
                "source": value,
                "execution_count": None,
                "outputs": [],
            }
        )

    bundle_hash = hashlib.sha256(bundle.read_bytes()).hexdigest()
    md(r'''
    # Superseded offline notebook draft — Olympus, 16 October 2026
    
    **SUPERSEDED_PROJECT_SWITCH.** The user selected FIM and offline preparation
    only. This Olympus draft is retained as historical engineering work and cannot
    execute. Its fresh-session workflow revision stopped before completion; it is
    not a launch-ready notebook and cannot train FIM.
    
    The only supported outcome here is **engineering/development evidence**. The
    small synthetic task is intentionally a pipeline/throughput check; it cannot
    support architecture superiority, a memory-mechanism claim or a conference result.
    Historical v2/v3 negatives and O1's unfrozen boundary remain unchanged.
    
    Use private interactive sessions in three stages: **CPU_PREPARE**, a fresh
    CPU **RESTORE** session, then one fresh GPU **GPU_EXECUTE** session. Changing
    accelerator may restart the kernel; no Python variables survive between stages.
    Never click Save & Run All for this workflow: it starts another execution.
    Use Quick Save with output-file saving explicitly enabled. Current account/UI
    behavior and successful private retrieval still require observation.
    
    The notebook intentionally stops at missing account observations and manual
    persistence gates. Do not use Run All to bypass them. No secrets are needed.
    ''')
    md(r'''
    ## 1. Verified source snapshot
    
    After confirming Olympus, upload the supplied source archive as a **private**
    Kaggle input and attach it here. Enter its exact mounted path, not a Mac path.
    All attached inputs are read-only. The archive contains an explicit source
    allowlist, tests, configurations and control helpers; no credentials or old data.
    ''')
    code(r'''
    raise RuntimeError(
        
            "SUPERSEDED_PROJECT_SWITCH: user selected FIM, offline "
            "preparation only. This incomplete Olympus draft must not execute."
        
    )
    ''')
    code(r'''
    import hashlib
    import json
    import os
    import shutil
    import subprocess
    import sys
    import tarfile
    import threading
    import time
    from datetime import UTC, datetime
    from pathlib import Path
    
    CONFIRMED_PROJECT = None  # Set to "olympus" only after the project choice is resolved.
    MODE = "CPU_PREPARE"  # Fresh CPU retrieval: RESTORE. Fresh production GPU: GPU_EXECUTE.
    DEADLINE = "2026-10-16T22:00:00+05:30"
    EXPECTED_BUNDLE_SHA256 = "1ee365cc4fcb1e586b415ac1c6973f95c3b852be9514f3d14c8e69660d66ce94"
    if CONFIRMED_PROJECT != "olympus":
        raise RuntimeError("Resolve the requested project first. This candidate is Olympus only.")
    if MODE not in {"CPU_PREPARE", "RESTORE", "GPU_EXECUTE"}:
        raise ValueError("Unknown mode")
    if datetime.now(UTC) >= datetime.fromisoformat(DEADLINE):
        raise RuntimeError("Tonight's planning deadline has passed; do not allocate GPU.")
    if MODE != "GPU_EXECUTE":
        smi = shutil.which("nvidia-smi")
        if smi:
            device_probe = subprocess.run([smi, "-L"], capture_output=True, text=True, timeout=10)
            if device_probe.returncode == 0 and "GPU " in device_probe.stdout:
                raise RuntimeError(
                    "GPU exposed in a CPU mode. Stop allocation; use a fresh CPU session."
                )
        if input("Confirm Kaggle session Accelerator is None; type CPU: ").strip() != "CPU":
            raise RuntimeError("CPU-only platform state must be observed before setup")
    SOURCE_ARCHIVE = Path(input("Exact /kaggle/input/... source archive path: ").strip())
    if not SOURCE_ARCHIVE.resolve().is_relative_to(Path("/kaggle/input")):
        raise ValueError("Source must be a private, read-only attached Kaggle input")
    
    
    def sha(path):
        h = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()
    
    
    if sha(SOURCE_ARCHIVE) != EXPECTED_BUNDLE_SHA256:
        raise RuntimeError("Source archive checksum mismatch")
    BASE = Path("/kaggle/working/olympus-launch-20261016")
    SOURCE, RESULTS = BASE / "source", BASE / "evidence"
    BASE.mkdir(parents=True, exist_ok=True)
    # Bootstrap only the hash-pinned stdlib archive verifier; it verifies all members.
    with tarfile.open(SOURCE_ARCHIVE, "r:gz") as archive:
        member = archive.getmember("files/notebooks/kaggle_support/artifacts.py")
        if not member.isfile() or member.size > 100000:
            raise ValueError("Invalid verifier member")
        bootstrap = archive.extractfile(member).read().decode()
    '''.replace('1ee365cc4fcb1e586b415ac1c6973f95c3b852be9514f3d14c8e69660d66ce94', bundle_hash))
    code(r'''
    import types
    
    verifier = types.ModuleType("pinned_archive_verifier")
    sys.modules[verifier.__name__] = verifier
    exec(compile(bootstrap, "pinned_archive_verifier", "exec"), verifier.__dict__)
    inventory = verifier.verify_archive(SOURCE_ARCHIVE, expected_sha256=EXPECTED_BUNDLE_SHA256)
    if not SOURCE.exists():
        verifier.restore_archive(SOURCE_ARCHIVE, SOURCE, expected_sha256=EXPECTED_BUNDLE_SHA256)
    sys.path.insert(0, str(SOURCE))
    ''')
    code(r'''
    from notebooks.kaggle_support import account_guard, artifacts
    
    # Check every source byte on setup re-entry, allowing only generated Python caches.
    with tarfile.open(SOURCE_ARCHIVE, "r:gz") as archive:
        source_manifest = json.load(archive.extractfile("BUNDLE_MANIFEST.json"))
    for item in source_manifest["files"]:
        if sha(SOURCE / item["path"]) != item["sha256"]:
            raise RuntimeError("Changed extracted source: " + item["path"])
    print("Verified source archive", EXPECTED_BUNDLE_SHA256)
    ''')
    md(r'''
    ### Fresh GPU session bootstrap (GPU_EXECUTE only)
    
    Complete CPU_PREPARE and fresh CPU RESTORE first. RESTORE creates a second,
    **gpu-ready** archive containing the verified CPU receipts and shared account
    ledger. Quick Save with outputs, inspect that private saved version, stop the
    CPU session, then attach **both source and gpu-ready inputs** to one fresh
    private interactive GPU session. Select GPU_EXECUTE above. Never Save & Run All.
    
    Immediately before starting that allocation, record the actual quota shown,
    its timezone-qualified observation time, allocation start time, and physical
    GPU count from the platform UI. Enter those original observations below, not
    a new timestamp. Cold start and setup consume the same 10% preflight allowance.
    If observations are too old for the guard, stop the allocation; do not invent
    a recent preallocation observation. This cell records accounting; it cannot
    enable or stop a platform accelerator.
    ''')
    code(r'''
    def write_json(path, value):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        temporary.replace(path)
    
    
    LEDGER = Path("/kaggle/working/shared-account-budget-20261016.json")
    if MODE == "GPU_EXECUTE":
        GPU_READY_ARCHIVE = Path(
            input(
                "Exact /kaggle/input/... gpu-ready archive from the saved private CPU RESTORE version: "
            ).strip()
        )
        EXPECTED_GPU_READY_SHA = input("External SHA256 printed by that CPU RESTORE session: ").strip()
        if not GPU_READY_ARCHIVE.resolve().is_relative_to(Path("/kaggle/input")):
            raise ValueError("GPU-ready evidence must be an attached saved private input")
        artifacts.verify_archive(GPU_READY_ARCHIVE, expected_sha256=EXPECTED_GPU_READY_SHA)
        if not RESULTS.exists():
            artifacts.restore_archive(
                GPU_READY_ARCHIVE, RESULTS, expected_sha256=EXPECTED_GPU_READY_SHA
            )
        artifacts.verify_restored(RESULTS, expected_archive_sha256=EXPECTED_GPU_READY_SHA)
        gpu_ready = json.loads((RESULTS / "gpu-ready.json").read_text())
        ready = json.loads((RESULTS / "cpu-ready.json").read_text())
        persistence = json.loads((RESULTS / "persistence-retrieval.json").read_text())
        if (
            gpu_ready.get("status") != "CPU_VERIFIED_READY_FOR_FRESH_GPU_SESSION"
            or ready.get("status") != "CPU_REGRESSIONS_PASSED"
            or ready.get("source_archive_sha256") != EXPECTED_BUNDLE_SHA256
            or gpu_ready.get("source_archive_sha256") != EXPECTED_BUNDLE_SHA256
            or gpu_ready.get("deadline") != DEADLINE
            or persistence.get("archive_hashes") != "verified_by_code"
            or persistence.get("fresh_session") != "operator_confirmed"
            or not persistence.get("private_version")
        ):
            raise RuntimeError(
                "Verified CPU readiness and fresh private persistence receipts are required"
            )
        for name, expected in gpu_ready["required_file_sha256"].items():
            if (
                name
                not in {
                    "cpu-ready.json",
                    "persistence-retrieval.json",
                    "environment.json",
                    "shared-account-budget.json",
                }
                or sha(RESULTS / name) != expected
            ):
                raise RuntimeError("GPU-ready receipt binding mismatch: " + name)
        if set(gpu_ready["required_file_sha256"]) != {
            "cpu-ready.json",
            "persistence-retrieval.json",
            "environment.json",
            "shared-account-budget.json",
        }:
            raise RuntimeError("Incomplete GPU-ready receipt bindings")
        saved_ledger = RESULTS / "shared-account-budget.json"
        if LEDGER.exists() and sha(LEDGER) != sha(saved_ledger):
            raise RuntimeError(
                "Existing shared ledger changed or has an active allocation; never overwrite it"
            )
        if not LEDGER.exists():
            with LEDGER.open("xb") as destination:
                destination.write(saved_ledger.read_bytes())
        prior_budget = account_guard.snapshot(LEDGER)
        if prior_budget["active_session_id"]:
            raise RuntimeError("Saved ledger has an unresolved active allocation; stop and reconcile")
        Q0 = float(input("ACTUAL remaining quota-hours observed immediately BEFORE this allocation: "))
        OBSERVED_AT = input(
            "Original preallocation quota observation ISO timestamp with timezone: "
        ).strip()
        ALLOCATION_STARTED_AT = input(
            "Actual platform allocation-start ISO timestamp with timezone: "
        ).strip()
        OTHER_ACTIVE = int(input("Observed OTHER account GPU allocations at start (must be 0): "))
        ALLOCATED_DEVICE_COUNT = int(
            input("Physical GPUs assigned by platform UI (1, or 2 for T4x2): ")
        )
        if ALLOCATED_DEVICE_COUNT not in (1, 2):
            raise RuntimeError("Unsupported physical GPU allocation")
        session = account_guard.start_session(
            LEDGER,
            project="Olympus-development",
            quota_hours=Q0,
            observed_at=OBSERVED_AT,
            other_active_sessions=OTHER_ACTIVE,
            allocation_started_at=ALLOCATION_STARTED_AT,
            device_count=ALLOCATED_DEVICE_COUNT,
        )
        # Include kernel cold start and bootstrap time; never reset this clock in later cells.
        GPU_PHASE_STARTED = (
            time.monotonic()
            - (datetime.now(UTC) - datetime.fromisoformat(ALLOCATION_STARTED_AT)).total_seconds()
        )
        PILOT_LIMIT_SECONDS = min(
            600.0, session["target_budget_hours"] * 3600 * 0.10 / ALLOCATED_DEVICE_COUNT
        )
        PILOT_PHASE = True
        shutil.copy2(LEDGER, RESULTS / "shared-account-budget.json")
        write_json(
            RESULTS / "gpu-bootstrap.json",
            {
                "source_archive_sha256": EXPECTED_BUNDLE_SHA256,
                "gpu_ready_archive_sha256": EXPECTED_GPU_READY_SHA,
                "session_id": session["active_session_id"],
                "allocation_started_at": ALLOCATION_STARTED_AT,
                "preallocation_observed_at": OBSERVED_AT,
                "physical_device_count": ALLOCATED_DEVICE_COUNT,
                "pilot_limit_seconds": PILOT_LIMIT_SECONDS,
            },
        )
    
    
    def guarded_command(label, command, estimate, timeout):
        if MODE != "GPU_EXECUTE":
            raise RuntimeError("GPU commands require the fresh GPU_EXECUTE session")
        if account_guard.snapshot(LEDGER)["active_session_id"] != session["active_session_id"]:
            raise RuntimeError("GPU session identity changed")
        if PILOT_PHASE and timeout + 10 > PILOT_LIMIT_SECONDS - (time.monotonic() - GPU_PHASE_STARTED):
            raise RuntimeError(
                "Setup/preflight/profile allowance exhausted; preserve evidence and STOP SESSION"
            )
        result = account_guard.run_command(
            LEDGER,
            list(map(str, command)),
            estimated_seconds=estimate,
            timeout_seconds=timeout,
            cwd=SOURCE,
            log_path=RESULTS / "gpu-logs" / (label + "-" + str(time.time_ns()) + ".log"),
        )
        shutil.copy2(LEDGER, RESULTS / "shared-account-budget.json")
        return result
    ''')
    md(r'''
    ## 2. Minimal dependencies and resolved environment
    
    Keep the Kaggle image's CUDA-enabled Torch. The isolated environment inherits
    it; the requirement file has **no Torch entry** and installation pins the
    inherited Torch version. CPU modes install with GPU disabled. A fresh GPU
    session repeats only this bounded setup through the account guard, after
    recording allocation start. A pip conflict blocks execution.
    The resolved package-name/version inventory is retained without environment
    variables, credential paths or private package URLs. If a dependency is missing,
    resolve installation issues during CPU preparation. Fresh GPU setup must
    reproduce the exact saved CPU package inventory; a mismatch blocks execution.
    GPU setup has no repair/retry loop and consumes the preflight allowance.
    ''')
    code(r'''
    import importlib.metadata as distribution_metadata
    
    VENV = Path("/tmp/olympus-kaggle-venv")
    torch_version = distribution_metadata.version("torch")  # Metadata only; does not import Torch.
    TORCH_CONSTRAINT = BASE / "inherited-torch-constraint.txt"
    TORCH_CONSTRAINT.write_text("torch==" + torch_version + "\n")
    
    
    def setup_command(label, command, timeout):
        if MODE == "GPU_EXECUTE":
            result = guarded_command("setup-" + label, command, timeout, timeout)
            if result["status"] != "SUCCEEDED":
                raise RuntimeError("GPU setup failed; preserve logs and STOP SESSION")
        else:
            subprocess.run(command, cwd=SOURCE, check=True, timeout=timeout)
    
    
    if not (VENV / "bin/python").exists():
        setup_command("venv", [sys.executable, "-m", "venv", "--system-site-packages", str(VENV)], 30)
    PY = str(VENV / "bin/python")
    setup_command(
        "install",
        [
            PY,
            "-m",
            "pip",
            "install",
            "--upgrade-strategy",
            "only-if-needed",
            "--constraint",
            str(TORCH_CONSTRAINT),
            "-r",
            str(SOURCE / "requirements-kaggle.txt"),
        ],
        90,
    )
    setup_command("pip-check", [PY, "-m", "pip", "check"], 30)
    os.environ.update(
        PYTHONUNBUFFERED="1",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        CUDA_VISIBLE_DEVICES="0",
    )
    
    
    def read_env():
        environment_path = BASE / "resolved-environment.json"
        command = (
            "import importlib.metadata as m,json,platform; from pathlib import Path; Path("
            + repr(str(environment_path))
            + (
                ").write_text(json.dumps({'python':platform.python_version(),'pack"
                "ages':dict(sorted((d.metadata['Name'],d.version) for d in "
                "m.distributions() if d.metadata.get('Name')))}))"
            )
        )
        setup_command("environment", [PY, "-c", command], 10)
        return json.loads(environment_path.read_text())
    
    
    ENVIRONMENT = read_env()
    if ENVIRONMENT["packages"].get("torch") != torch_version:
        raise RuntimeError("Inherited Torch changed; stop and retain setup logs")
    if MODE == "GPU_EXECUTE" and (
        ENVIRONMENT != ready["environment"]
        or ENVIRONMENT != json.loads((RESULTS / "environment.json").read_text())
    ):
        raise RuntimeError("Fresh GPU environment differs from verified CPU environment; STOP SESSION")
    print(
        {k: ENVIRONMENT["packages"].get(k) for k in ("torch", "numpy", "pydantic", "pillow", "pytest")}
    )
    ''')
    md(r'''
    ## 3. CPU preflight and numerical regressions
    
    Run this section only with the platform accelerator disabled. These are the
    maintained CPU correctness/resume checks, not the entire repository suite.
    Resource rejection or a test failure blocks GPU work. Rerunning successful
    preparation checks its receipt; it does not silently generate fresh evidence.
    ''')
    code(r'''
    def run_cpu(label, arguments, timeout=1800):
        logs = RESULTS / "cpu-logs"
        logs.mkdir(parents=True, exist_ok=True)
        log = logs / (label + ".log")
        if log.exists():
            raise RuntimeError(
                "Preserve the prior attempt; inspect it before any diagnosed retry: " + str(log)
            )
        started = time.monotonic()
        with log.open("w") as stream:
            proc = subprocess.Popen(
                [PY, "-u", *arguments],
                cwd=SOURCE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
    
            def pump():
                for line in proc.stdout:
                    print(line, end="", flush=True)
                    stream.write(line)
                    stream.flush()
    
            thread = threading.Thread(target=pump)
            thread.start()
            try:
                returncode = proc.wait(timeout=timeout)
            except BaseException:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                raise
            finally:
                thread.join()
                write_json(
                    log.with_suffix(".json"),
                    {
                        "returncode": proc.returncode,
                        "elapsed_seconds": time.monotonic() - started,
                        "command": arguments,
                    },
                )
        if returncode:
            raise RuntimeError(f"{label} failed with exit {returncode}; see {log}")
    
    
    def campaign_args(command, directory, extra=()):
        return [
            "-m",
            "olympus.training.campaign",
            command,
            "--run-root",
            str(directory / "runs"),
            *map(str, extra),
        ]
    
    
    if MODE == "CPU_PREPARE":
        RESULTS.mkdir(parents=True, exist_ok=True)
        if not (RESULTS / "cpu-ready.json").exists():
            run_cpu("preflight", campaign_args("preflight", RESULTS / "cpu", ["--device", "cpu"]))
            run_cpu(
                "regressions",
                [
                    "-m",
                    "scripts.validate_olympus_campaign",
                    "--device",
                    "cpu",
                    "--lock",
                    str(RESULTS / "cpu-validation.lock"),
                ],
            )
        else:
            ready = json.loads((RESULTS / "cpu-ready.json").read_text())
            assert ready["source_archive_sha256"] == EXPECTED_BUNDLE_SHA256
            assert ready["environment"] == ENVIRONMENT, (
                "Environment changed; use a new protocol/version"
            )
    ''')
    md(r'''
    ## 4. Data preparation and integrity
    
    These explicit synthetic development splits are **512 train / 128 validation**;
    smoke is 8/4. Fixed generator seeds and all artifact/source identities are
    recorded. No protected test is generated or consulted. There is no fitted
    normalization. A dataset directory is reused only after integrity validation.
    ''')
    code(r'''
    if MODE == "CPU_PREPARE":
        for name, command in [("smoke", "prepare-smoke"), ("development", "prepare-development")]:
            directory = RESULTS / name
            if not (directory / "data").exists():
                run_cpu(
                    "data-" + name, campaign_args(command, directory, ["--output", directory / "data"])
                )
            check = (
                (
                    "from pathlib import Path; from olympus.training.campaign import "
                    "_dataset_identity; import json; "
                    "print(json.dumps(_dataset_identity(Path("
                )
                + repr(str(directory / "data/manifest_development.json"))
                + "))))"
            )
            # Metadata digest check; numerical split/range checks run in the CPU regression and trainer.
            subprocess.run([PY, "-c", check], cwd=SOURCE, check=True)
        write_json(RESULTS / "environment.json", ENVIRONMENT)
        shutil.copy2(SOURCE_ARCHIVE, RESULTS / "source-snapshot.tar.gz")
        write_json(
            RESULTS / "cpu-ready.json",
            {
                "source_archive_sha256": EXPECTED_BUNDLE_SHA256,
                "environment": ENVIRONMENT,
                "status": "CPU_REGRESSIONS_PASSED",
                "created_at": datetime.now(UTC).isoformat(),
            },
        )
    ''')
    md(r'''
    ## 5. Credentials and privacy
    
    This generated-data workflow needs no credentials, W&B account, remote model
    download or Kaggle API token. Do not paste tokens into this notebook. Keep the
    source input, notebook and saved outputs private. External tracking is off.
    A future authorized service must use Kaggle Secrets and a separately reviewed
    destination; no secret is loaded or exported by this notebook.
    
    ## 6. Persistence round trip — mandatory before GPU
    
    In CPU_PREPARE mode, create the archive below. Quick Save with output files
    enabled, inspect the saved version's Output tab, then explicitly Stop Session.
    Start a **fresh CPU session**, attach that private saved version as an input,
    set MODE="RESTORE", and rerun setup. Retrieve and verify the archive below.
    A local tar file or successful save dialog alone is not persistence proof.
    ''')
    code(r'''
    if MODE == "CPU_PREPARE":
        PREPARATION = BASE / "preparation.tar.gz"
        if not PREPARATION.exists():
            preparation_receipt = artifacts.archive_tree(RESULTS, PREPARATION)
            write_json(BASE / "preparation-receipt.json", preparation_receipt)
        else:
            artifacts.verify_archive(PREPARATION)
        print(
            "Quick Save outputs, stop CPU session, attach this saved version in a fresh CPU session:",
            PREPARATION,
            sha(PREPARATION),
        )
    if MODE == "RESTORE":
        RETRIEVED = Path(
            input(
                "Exact /kaggle/input/.../preparation.tar.gz path from saved private version: "
            ).strip()
        )
        EXPECTED_PREPARATION_SHA = input("SHA256 printed by the original CPU session: ").strip()
        if not RETRIEVED.resolve().is_relative_to(Path("/kaggle/input")):
            raise ValueError("Retrieval must be from an attached saved private version")
        retrieval = artifacts.verify_archive(RETRIEVED, expected_sha256=EXPECTED_PREPARATION_SHA)
        if not RESULTS.exists():
            artifacts.restore_archive(RETRIEVED, RESULTS, expected_sha256=EXPECTED_PREPARATION_SHA)
        artifacts.verify_restored(RESULTS, expected_archive_sha256=EXPECTED_PREPARATION_SHA)
        ready = json.loads((RESULTS / "cpu-ready.json").read_text())
        assert ready["source_archive_sha256"] == EXPECTED_BUNDLE_SHA256
        assert ready["environment"] == ENVIRONMENT, (
            "Runtime changed; repeat CPU validation under a new version"
        )
        version = input("Saved private notebook version identifier (owner/slug/version): ").strip()
        if not version or input("Confirm this is a FRESH CPU session; type FRESH: ").strip() != "FRESH":
            raise RuntimeError("Fresh-session retrieval confirmation missing")
        write_json(
            RESULTS / "persistence-retrieval.json",
            {
                "sha256": EXPECTED_PREPARATION_SHA,
                "input_path": str(RETRIEVED),
                "private_version": version,
                "verified_at": datetime.now(UTC).isoformat(),
                "fresh_session": "operator_confirmed",
                "archive_hashes": "verified_by_code",
            },
        )
    ''')
    md(r'''
    ## 7. Actual account budget and GPU preflight
    
    Before changing the accelerator, read the **displayed remaining quota** and
    account-wide active sessions. Import the latest shared account ledger if any
    APEN/FIM/Wharton/quant work already used this allocation window. Never initialize
    another allowance for a different project. The local file lock does not lock
    other Kaggle notebooks: no other active allocation is allowed by this package.
    
    Preferred=max(0,min(6,Q0−24)); expanded=max(0,min(10,Q0−20)) is informational only.
    Target=min(4,preferred). Keep at least 10% plus two minutes for export/shutdown.
    **If preferred is zero, do not enable a GPU; request a revised allocation.**
    ''')
    code(r'''
    if MODE != "RESTORE" or not (RESULTS / "persistence-retrieval.json").exists():
        raise RuntimeError("Complete the fresh-session CPU persistence round trip first")
    LEDGER = Path("/kaggle/working/shared-account-budget-20261016.json")
    # If a prior account ledger exists in saved outputs, restore that exact file here first.
    Q0 = float(input("Actual remaining Kaggle quota-hours shown immediately now: "))
    OBSERVED_AT = datetime.now(UTC).isoformat()
    OTHER_ACTIVE = int(
        input("Observed OTHER active GPU sessions across the entire account (must be 0): ")
    )
    if not LEDGER.exists():
        if (
            input(
                "Confirm no earlier ledger exists for this account allocation window; type FIRST: "
            ).strip()
            != "FIRST"
        ):
            raise RuntimeError("Restore the shared account ledger before continuing")
        budget = account_guard.initialize(
            LEDGER,
            quota_hours=Q0,
            observed_at=OBSERVED_AT,
            deadline=DEADLINE,
            other_active_sessions=OTHER_ACTIVE,
        )
    else:
        budget = account_guard.observe_quota(
            LEDGER, quota_hours=Q0, observed_at=OBSERVED_AT, other_active_sessions=OTHER_ACTIVE
        )
    if (
        OTHER_ACTIVE != 0
        or budget["remaining_budget_hours"] <= budget["reserved_hours"]
        or Q0 <= 24
        or budget["seconds_to_deadline"] <= 600
    ):
        raise RuntimeError("No authorized preferred allocation is available; leave GPU off")
    if budget["active_session_id"]:
        raise RuntimeError(
            "An allocation is already recorded. Reconcile platform state; do not start another."
        )
    PLANNED_DEVICE_COUNT = int(
        input("Device count in the accelerator option you plan to select (1 or 2): ")
    )
    if PLANNED_DEVICE_COUNT not in (1, 2):
        raise RuntimeError("Unsupported allocation plan")
    if min(600, budget["target_budget_hours"] * 3600 * 0.10 / PLANNED_DEVICE_COUNT) < 180:
        raise RuntimeError(
            "The bounded preflight plan does not fit 10% of this allocation. Leave GPU off."
        )
    print(json.dumps(budget, indent=2))
    print(
        
            "Record UTC time immediately BEFORE enabling GPU. Use one GPU; "
            "T4x2 is charged conservatively as two devices until verified."
        
    )
    ''')
    code(r'''
    # Run immediately after enabling ONE platform GPU allocation; never launch Save & Run All.
    ALLOCATION_STARTED_AT = input(
        "UTC ISO timestamp recorded immediately before enabling GPU: "
    ).strip()
    ALLOCATED_DEVICE_COUNT = int(input("Physical devices assigned by platform (e.g. 2 for T4x2): "))
    session = account_guard.start_session(
        LEDGER,
        project="Olympus-development",
        quota_hours=Q0,
        observed_at=OBSERVED_AT,
        other_active_sessions=OTHER_ACTIVE,
        allocation_started_at=ALLOCATION_STARTED_AT,
        device_count=ALLOCATED_DEVICE_COUNT,
    )
    GPU_PHASE_STARTED = time.monotonic()
    PILOT_LIMIT_SECONDS = min(
        600.0, session["target_budget_hours"] * 3600 * 0.10 / ALLOCATED_DEVICE_COUNT
    )
    PILOT_PHASE = True
    
    
    def gpu_command(label, args, estimate, timeout):
        if PILOT_PHASE and timeout + 10 > PILOT_LIMIT_SECONDS - (time.monotonic() - GPU_PHASE_STARTED):
            raise RuntimeError(
                "Preflight/profile allowance exhausted; export and explicitly stop the session"
            )
        return account_guard.run_command(
            LEDGER,
            [PY, "-u", *map(str, args)],
            estimated_seconds=estimate,
            timeout_seconds=timeout,
            cwd=SOURCE,
            log_path=RESULTS / "gpu-logs" / (label + "-" + str(time.time_ns()) + ".log"),
        )
    
    
    result = gpu_command(
        "cuda-preflight", campaign_args("preflight", RESULTS / "cuda", ["--device", "cuda"]), 30, 30
    )
    if result["status"] != "SUCCEEDED":
        raise RuntimeError("CUDA preflight failed; preserve logs and stop allocation")
    result = gpu_command(
        "cuda-regressions",
        [
            "-m",
            "scripts.validate_olympus_campaign",
            "--device",
            "cuda",
            "--lock",
            RESULTS / "cuda-validation.lock",
        ],
        90,
        90,
    )
    if result["status"] != "SUCCEEDED":
        raise RuntimeError("CUDA correctness/resume failed; no long run")
    ''')
    md(r'''
    ## 8. Representative DEVELOPMENT profiles and finite freeze
    
    Profile both conditions with the actual 300-step/batch-32 shapes, full 128-record
    validation and checkpoint/export path. Pilot seed differs from the matrix seeds.
    This is representative only of this small synthetic engineering question.
    Precision stays FP32; no AMP, accumulation, distributed training or shape change.
    Inclusive timing phases must not be summed. Discard warmup samples for summaries.
    A failed or incomplete pilot blocks expansion. The entire GPU preflight/profile
    phase is limited to 10% of the account target, capped at ten minutes.
    ''')
    code(r'''
    template = json.loads((SOURCE / "configs/kaggle/development-cuda.json").read_text())
    development = RESULTS / "development"
    pilot = json.loads(json.dumps(template))
    pilot["campaign_id"] = "olympus-development-profile-v1"
    pilot["cells"] = [pilot["cells"][0], pilot["cells"][3]]
    for cell in pilot["cells"]:
        strategy = cell["training_config"]["model"]["fusion_strategy"]
        cell["id"] = "profile-" + strategy
        cell["training_config"]["seed"] = 2026101691
    pilot["max_wall_seconds"] = max(120, PILOT_LIMIT_SECONDS)
    pilot["stop_reserve_seconds"] = 60
    pilot_path = development / "pilot.json"
    if pilot_path.exists() and json.loads(pilot_path.read_text()) != pilot:
        raise RuntimeError("Existing pilot identity differs")
    write_json(pilot_path, pilot)
    profiles = []
    for cell in pilot["cells"]:
        remaining = PILOT_LIMIT_SECONDS - (time.monotonic() - GPU_PHASE_STARTED)
        if remaining < 30:
            raise RuntimeError("Pilot allowance exhausted; stop and preserve artifacts")
        output = development / (cell["id"] + ".json")
        if output.exists():
            raise RuntimeError("Prior pilot exists; inspect/reuse its validated receipt explicitly")
        timeout = min(180, remaining - 10)
        result = gpu_command(
            cell["id"],
            [
                "-m",
                "notebooks.kaggle_support.profile_run",
                "--manifest",
                pilot_path,
                "--run-root",
                development / "pilot-runs",
                "--cell",
                cell["id"],
                "--output",
                output,
                "--device",
                "cuda",
            ],
            timeout,
            timeout,
        )
        if result["status"] != "SUCCEEDED":
            raise RuntimeError("Representative pilot failed; no matrix")
        profiles.append(json.loads(output.read_text()))
    print(
        
            "Profiles retained. Inspect phase timings, device placement, "
            "memory and complete status before freeze."
        
    )
    ''')
    code(r'''
    # Freeze only this finite DEVELOPMENT matrix. No confirmatory hypothesis/test exists here.
    if len(profiles) != 2 or any(p["status"] != "COMPLETED_REPRESENTATIVE_PROFILE" for p in profiles):
        raise RuntimeError(
            "Both actual profiles must pass placement, update-count and completion checks"
        )
    completion = [
        json.loads(p.read_text()) for p in (development / "pilot-runs/cells").glob("*/completion.json")
    ]
    if len(completion) != 2:
        raise RuntimeError("Both completed representative profiles required")
    account_guard.admit(LEDGER, estimated_seconds=60)
    export_started = time.monotonic()
    pilot_export = BASE / ("pilot-recovery-" + str(time.time_ns()) + ".tar.gz")
    artifacts.archive_tree(RESULTS, pilot_export)
    export_elapsed = time.monotonic() - export_started
    write_json(
        RESULTS / "export-profile.json",
        {
            "elapsed_seconds": export_elapsed,
            "bytes": pilot_export.stat().st_size,
            "sha256": sha(pilot_export),
        },
    )
    estimate = (max(x["complete_elapsed_seconds"] for x in profiles) + export_elapsed) * 1.5 + 30
    budget = account_guard.snapshot(LEDGER)
    account_guard.admit(LEDGER, estimated_seconds=6 * estimate)
    frozen = json.loads(json.dumps(template))
    frozen["campaign_id"] = "olympus-development-six-cells-20261016"
    frozen["max_wall_seconds"] = min(21600, 6 * estimate / 0.85 + 120)
    frozen["stop_reserve_seconds"] = max(120, 0.10 * frozen["max_wall_seconds"])
    for cell in frozen["cells"]:
        cell["estimated_seconds"] = estimate
    MANIFEST = development / "campaign.json"
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != frozen:
        raise RuntimeError("Frozen manifest already exists with a different identity")
    write_json(MANIFEST, frozen)
    protocol = {
        "status": "FROZEN_DEVELOPMENT_ONLY",
        "manifest_sha256": sha(MANIFEST),
        "source_archive_sha256": EXPECTED_BUNDLE_SHA256,
        "environment_sha256": sha(RESULTS / "environment.json"),
        "data_manifest_sha256": sha(development / "data/manifest_development.json"),
        "question": (
            "Can learned and uniform composed-system runs complete "
            "reproducibly under identical fixed budgets?"
        ),
        "primary_comparison": (
            "Descriptive learned-minus-uniform development total loss; lower is better"
        ),
        "replication_unit": (
            "Three paired training initialization/sampling seeds on one fixed synthetic split"
        ),
        "selection": "Best development total loss, including step zero; no protected test",
        "normalization": "Generator-native; no fitted normalization",
        "precision": "fp32",
        "tuning_allowance": "None beyond the declared fixed configuration",
        "early_stopping": "None; 300 optimizer steps",
        "failures": (
            "Retain every mandatory cell; missing comparison blocks a "
            "complete matrix; no automatic retry"
        ),
        "analysis": (
            "All paired seed differences, mean and sample SD; descriptive "
            "only, no p-values/equivalence claim"
        ),
        "multiple_comparisons": "One descriptive primary contrast; all other metrics exploratory",
        "ablations": "Uniform routing retains router compute; not a disabled-memory control",
        "stress_test": "None; no scientifically justified OOD evaluation supplied",
        "execution_order": [c["id"] for c in frozen["cells"]],
        "estimated_seconds_per_cell": estimate,
        "quota": "Conservative estimate; manual platform reconciliation required",
        "deadline": DEADLINE,
    }
    write_json(development / "protocol.json", protocol)
    PILOT_PHASE = False
    print(
        "Frozen development protocol SHA256",
        sha(development / "protocol.json"),
        "estimated matrix seconds",
        6 * estimate,
    )
    ''')
    md(r'''
    ## 9. Execution, live status and immediate persistence
    
    Real training events stream to console and JSONL files. Budget heartbeats include
    charged allocation time and estimated balance. Every mandatory cell starts as
    PENDING. No automatic retries. Failed runs retain logs, configuration and partial
    checkpoints. Completed cells are validated by the existing controller on re-entry.
    
    After each cell, save the complete recovery archive using Quick Save outputs.
    Inspect/download that saved version before acknowledging. This is an attended
    manual persistence step; idle time remains charged. There is no unsupported
    automatic save API here. Acknowledgement is recorded as operator evidence.
    ''')
    code(r'''
    import signal
    
    
    def bounded_acknowledgement(prompt):
        # Linux notebook main thread only: a blocked persistence prompt cannot wait forever.
        state = account_guard.snapshot(LEDGER)
        margin = (
            (state["remaining_budget_hours"] - state["reserved_hours"]) * 3600 / ALLOCATED_DEVICE_COUNT
        )
        limit = min(120, margin, state["seconds_to_deadline"] - 120)
        if limit <= 0:
            raise RuntimeError("No persistence interaction margin; explicitly stop the GPU")
    
        def timeout_handler(signum, frame):
            raise TimeoutError(
                "Persistence acknowledgement timed out. Explicitly STOP SESSION in Kaggle now."
            )
    
        previous = signal.signal(signal.SIGALRM, timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, limit)
        try:
            return input(prompt).strip()
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
    
    
    status_path = RESULTS / "run-ledger.json"
    statuses = (
        json.loads(status_path.read_text())
        if status_path.exists()
        else {cell["id"]: {"status": "PENDING"} for cell in frozen["cells"]}
    )
    write_json(status_path, statuses)
    for cell in frozen["cells"]:
        identifier = cell["id"]
        if (development / "runs/cells" / identifier / "completion.json").exists():
            account_guard.admit(LEDGER, estimated_seconds=10)
            validated = json.loads(
                subprocess.check_output(
                    [
                        PY,
                        "-m",
                        "olympus.training.campaign",
                        "analyze",
                        "--run-root",
                        str(development / "runs"),
                    ],
                    cwd=SOURCE,
                    text=True,
                    timeout=10,
                )
            )
            observed = next(row for row in validated["cells"] if row["id"] == identifier)
            if observed["status"] != "COMPLETED_DEVELOPMENT_NOT_PROMOTED":
                raise RuntimeError("Existing completion failed validation")
            statuses[identifier] = {
                **statuses.get(identifier, {}),
                "status": "SUCCEEDED",
                "reused_validated": True,
            }
            write_json(status_path, statuses)
            continue
        if statuses[identifier]["status"] in {"FAILED", "INTERRUPTED", "RUNNING"}:
            raise RuntimeError(
                "Prior failed/interrupted cell needs diagnosed, bounded resume; no automatic retry"
            )
        try:
            account_guard.admit(LEDGER, estimated_seconds=estimate)
        except account_guard.BudgetError as error:
            statuses[identifier] = {"status": "SKIPPED_BUDGET", "reason": str(error)}
            write_json(status_path, statuses)
            continue
        statuses[identifier] = {"status": "RUNNING"}
        write_json(status_path, statuses)
        result = gpu_command(
            identifier,
            campaign_args("run", development, ["--manifest", MANIFEST, "--cell", identifier]),
            estimate,
            estimate,
        )
        statuses[identifier] = {"status": result["status"], "account_job": result}
        write_json(status_path, statuses)
        shutil.copy2(LEDGER, RESULTS / "shared-account-budget.json")
        if shutil.disk_usage(BASE).free < 2 * 1024**3:
            raise RuntimeError(
                "Insufficient free disk for safe export; stop allocation and preserve existing files"
            )
        archive = BASE / ("recovery-" + identifier + "-" + str(time.time_ns()) + ".tar.gz")
        receipt = artifacts.archive_tree(RESULTS, archive)
        write_json(archive.with_suffix(".receipt.json"), receipt)
        print("Save outputs and verify private saved-version retrieval:", archive, sha(archive))
        version = bounded_acknowledgement(
            "Saved private version identifier after file verification (blank stops): "
        )
        if not version:
            raise RuntimeError("Persistence not confirmed; stop GPU session with margin")
        statuses[identifier]["persistence"] = {
            "version": version,
            "sha256": sha(archive),
            "verification": "operator_reported",
        }
        write_json(status_path, statuses)
        if result["status"] != "SUCCEEDED":
            raise RuntimeError(
                "Failure retained; diagnose before any explicitly bounded retry. Stop allocation."
            )
    ''')
    md(r'''
    ## 10. Resume from persisted outputs
    
    In a fresh CPU session attach the latest private saved version. Verify its archive
    using the printed SHA256, restore into a **new** writable evidence directory, and
    use the same source archive. Restore the shared account ledger too. Do not reset
    budgets, rename away failures, change the total steps or patch a running protocol.
    Changed source, data, configuration or runtime requires a new version.
    
    The existing controller resumes optimizer-boundary checkpoints automatically:
    `python -m olympus.training.campaign run --manifest <restored>/development/campaign.json
    --run-root <restored>/development/runs --cell <frozen-id>`.
    Run that command only through the account guard after fresh quota/stop checks.
    Latest/best pointers, optimizer/scheduler/RNG, raw metrics, source and generated
    data are included in the recovery archive. Hard-kill offline intervals are
    conservatively charged by the trainer controller; do not edit its ledger to
    force admission. Cross-runtime exact continuation is unsupported.
    
    ## 11. Analysis and final export
    
    Analysis validates each completed cell against its frozen source/configuration/
    data and artifact digests. Incomplete cells stay in the denominator. CPU-only
    aggregation can be done after GPU shutdown; don't allocate GPU just for a table.
    ''')
    code(r'''
    # Pure-control analysis: no model inference or new tuning.
    analyzed = json.loads(
        subprocess.check_output(
            [PY, "-m", "olympus.training.campaign", "analyze", "--run-root", str(development / "runs")],
            cwd=SOURCE,
            text=True,
        )
    )
    write_json(RESULTS / "analysis.json", analyzed)
    ''')
    code(r'''
    import csv
    import statistics
    
    rows = []
    for row in analyzed["cells"]:
        condition, seed = row["id"].rsplit("-", 1)
        rows.append(
            {
                "id": row["id"],
                "condition": condition,
                "seed": seed,
                "status": row["status"],
                "development_loss": row.get("final_evaluation", {}).get("total_loss"),
            }
        )
    with (RESULTS / "per-cell.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    paired = []
    for seed in sorted({r["seed"] for r in rows}):
        pair = {r["condition"]: r["development_loss"] for r in rows if r["seed"] == seed}
        if set(pair) == {"learned", "uniform"} and all(v is not None for v in pair.values()):
            paired.append({"seed": seed, "learned_minus_uniform": pair["learned"] - pair["uniform"]})
    differences = [p["learned_minus_uniform"] for p in paired]
    summary = {
        "expected_cells": 6,
        "completed_cells": sum(r["development_loss"] is not None for r in rows),
        "paired_effects": paired,
        "mean_difference": statistics.mean(differences) if differences else None,
        "sample_sd": statistics.stdev(differences) if len(differences) > 1 else None,
        "inference": "Descriptive training-seed variability on one fixed synthetic split only",
    }
    write_json(RESULTS / "summary.json", summary)
    try:
        import matplotlib.pyplot as plt
    
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for condition in ("learned", "uniform"):
            subset = [
                r for r in rows if r["condition"] == condition and r["development_loss"] is not None
            ]
            ax.plot(
                [r["seed"] for r in subset],
                [r["development_loss"] for r in subset],
                "o-",
                label=condition,
            )
        ax.set(
            xlabel="Training seed (paired)",
            ylabel="Development total loss",
            title="DEVELOPMENT ONLY — all valid cells",
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(RESULTS / "development-loss.png", dpi=160)
        plt.show()
    except ImportError:
        print(
            
                "Matplotlib unavailable; retained CSV/JSON are the authoritative "
                "data. No fabricated figure."
            
        )
    print(json.dumps(summary, indent=2))
    ''')
    md(r'''
    ## 12. Export, stop the platform session, reconcile quota
    
    Before the reserve is consumed, create the final recovery archive and Quick Save
    **with output files enabled**. Verify the saved version, download it and compare
    SHA256. Then explicitly **Stop Session** in Kaggle. Check account session state
    and displayed final quota. Closing the tab or a Python process exiting is not
    proof of deallocation. No code here calls a platform-stop API.
    
    Record the final observation below from a **CPU-only** session or on the local
    machine. Never turn GPU back on just to run this bookkeeping. Retain the latest
    shared ledger with the final evidence. Unknown stop status means billing must be
    treated as continuing; the guard cannot guarantee a hard account cap.
    ''')
    code(r'''
    shutil.copy2(LEDGER, RESULTS / "shared-account-budget.json")
    final_archive = BASE / ("final-evidence-" + str(time.time_ns()) + ".tar.gz")
    final_receipt = artifacts.archive_tree(RESULTS, final_archive)
    write_json(final_archive.with_suffix(".receipt.json"), final_receipt)
    print("Verify private persistence, then STOP SESSION:", final_archive, sha(final_archive))
    print(json.dumps(account_guard.snapshot(LEDGER), indent=2))
    ''')
    code(r'''
    # Execute only after independently observing the GPU session STOPPED, on CPU.
    stopped = input("Observed platform stopped timestamp (ISO with timezone): ").strip()
    balance = float(input("Actual final displayed remaining quota-hours: "))
    evidence = input("Platform stop observation and saved private output version reference: ").strip()
    final_budget = account_guard.record_platform_stop(
        LEDGER,
        observed_stopped_at=stopped,
        final_quota_hours=balance,
        final_quota_observed_at=datetime.now(UTC).isoformat(),
        evidence=evidence,
    )
    shutil.copy2(LEDGER, RESULTS / "shared-account-budget.json")
    write_json(RESULTS / "final-budget-observation.json", final_budget)
    print(json.dumps(final_budget, indent=2))
    reconciled_archive = BASE / ("reconciled-evidence-" + str(time.time_ns()) + ".tar.gz")
    artifacts.archive_tree(RESULTS, reconciled_archive)
    print(
        "Save this CPU bookkeeping update and verify retrieval:",
        reconciled_archive,
        sha(reconciled_archive),
    )
    print("Engineering evidence alone does not support a scientific preprint result.")
    ''')
    md(r'''
    ### Official platform references checked during preparation
    
    - [Notebook save modes, outputs and limits](https://www.kaggle.com/docs/notebooks)
    - [Kaggle Staff: Quick Save output option](https://www.kaggle.com/product-feedback/533289)
    - [Fresh-session saved-output attachment](https://www.kaggle.com/product-feedback/296805)
    - [Quota and allocation guidance](https://www.kaggle.com/docs/efficient-gpu-usage)
    - [Session stop and allocated time](https://www.kaggle.com/page/GPU-tips-and-tricks)
    - [T4x2 announcement; charging multiplier remains unverified](https://www.kaggle.com/product-feedback/361104)
    
    Documentation describes a 20 GB working-output limit; this package imposes a
    stricter 1 GiB archive limit. Verify actual account limits before expansion.
    Published idle timeout descriptions conflict; unattended interactive survival
    is not promised. The preserved checkpoints and explicit stop procedure matter.
    ''')
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "kaggle": {"isGpuEnabled": False, "isInternetEnabled": False},
        },
        "cells": cells,
    }
    for index, cell in enumerate(cells):
        cell["id"] = f"cell-{index:03d}"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.bundle, args.output)

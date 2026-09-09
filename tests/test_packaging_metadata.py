from __future__ import annotations

import io
import os
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path

import pytest

from olympus import __version__
from scripts.build_release_artifacts import normalize_source_distribution
from scripts.validate_distribution_payload import validate_distribution_payload
from scripts.validate_installed_wheel import _run


def test_installed_wheel_commands_have_enforced_deadlines(tmp_path: Path) -> None:
    with pytest.raises(subprocess.TimeoutExpired):
        _run(
            [sys.executable, "-I", "-c", "import time; time.sleep(60)"],
            cwd=tmp_path, environment=os.environ, timeout_seconds=0.1,
        )
    assert _run(
        [sys.executable, "-I", "-c", "print('verified')"],
        cwd=tmp_path, environment=os.environ,
    ).strip() == "verified"


def test_installed_wheel_commands_do_not_hide_failures(tmp_path: Path) -> None:
    with pytest.raises(subprocess.CalledProcessError):
        _run(
            [sys.executable, "-I", "-c", "raise SystemExit(7)"],
            cwd=tmp_path, environment=os.environ,
        )


def _write_distribution_fixtures(
    tmp_path: Path, *, wheel_paths: set[str], sdist_paths: set[str]
) -> tuple[Path, Path]:
    wheel = tmp_path / "fixture.whl"
    with zipfile.ZipFile(wheel, mode="w") as archive:
        for path in sorted(wheel_paths):
            archive.writestr(path, b"# fixture\n")

    sdist = tmp_path / "fixture.tar.gz"
    with tarfile.open(sdist, mode="w:gz") as archive:
        for path in sorted(sdist_paths):
            payload = b"# fixture\n"
            member = tarfile.TarInfo(name=f"fixture-0.0.0/{path}")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    return wheel, sdist


def test_pyproject_declares_complete_release_metadata() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    project = payload["project"]

    assert project["version"] == __version__
    assert project["license"] == "LicenseRef-Proprietary"
    assert project["license-files"] == ["LICENSE"]
    assert project["authors"] == [{"name": "Ryan"}]
    assert len(project["dependencies"]) == 10
    assert all(">=" in dependency for dependency in project["dependencies"])
    assert any(dependency.startswith("numpy>=") for dependency in project["dependencies"])
    assert "cryptography>=49.0.0,<51" in project["dependencies"]
    assert project["urls"]["Source"].endswith("/olympus-cognitive-architecture")
    assert payload["tool"]["pytest"]["ini_options"]["pythonpath"] == ["."]
    assert (root / "LICENSE").is_file()


def test_release_manifest_includes_all_release_scripts_without_stale_exclusion() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = (root / "MANIFEST.in").read_text(encoding="utf-8")

    assert "recursive-include docs *.md" in manifest
    assert "recursive-include scripts *.py" in manifest
    assert "recursive-exclude docs *OUTREACH*.md" not in manifest


def test_environment_example_contains_only_supported_runtime_settings() -> None:
    root = Path(__file__).resolve().parents[1]
    environment = (root / ".env.example").read_text(encoding="utf-8")

    assert "OLYMPUS_FOUNDRY_ROOT=" in environment
    assert "OLYMPUS_OLLAMA_URL=" in environment
    assert "OLYMPUS_API_TOKEN=" in environment
    assert "POSTGRES" not in environment
    assert "REDIS" not in environment


def test_distribution_payload_validator_accepts_exact_tracked_sources(tmp_path: Path) -> None:
    tracked = {
        "olympus/__init__.py",
        "olympus/models/example.py",
        "tests/test_example.py",
        "examples/example.py",
        "scripts/release_check.py",
    }
    wheel, sdist = _write_distribution_fixtures(
        tmp_path,
        wheel_paths={path for path in tracked if path.startswith("olympus/")},
        sdist_paths=tracked | {
            "olympus_cognitive_architecture.egg-info/PKG-INFO",
            "olympus_cognitive_architecture.egg-info/SOURCES.txt",
        },
    )

    validate_distribution_payload(
        wheel,
        sdist,
        repository_root=tmp_path,
        tracked_python_paths=tracked,
    )


@pytest.mark.parametrize("extra", [
    "olympus_cognitive_architecture.egg-info/evil.py",
    "olympus_cognitive_architecture-attacker.egg-info/PKG-INFO",
])
def test_distribution_metadata_allowlist_is_exact(tmp_path: Path, extra: str) -> None:
    tracked = {"olympus/__init__.py"}
    wheel, sdist = _write_distribution_fixtures(
        tmp_path, wheel_paths=tracked, sdist_paths=tracked | {extra}
    )
    with pytest.raises(ValueError, match="outside approved roots|untracked source-distribution"):
        validate_distribution_payload(wheel, sdist, repository_root=tmp_path,
                                      tracked_python_paths=tracked)


def test_distribution_payload_validator_rejects_untracked_python_and_tests(
    tmp_path: Path,
) -> None:
    tracked = {
        "olympus/__init__.py",
        "olympus/models/expected.py",
        "tests/test_expected.py",
    }
    wheel, sdist = _write_distribution_fixtures(
        tmp_path,
        wheel_paths={"olympus/__init__.py", "olympus/models/untracked.py"},
        sdist_paths={
            "olympus/__init__.py",
            "olympus/models/untracked.py",
            "tests/test_expected.py",
            "tests/test_untracked.py",
        },
    )

    with pytest.raises(ValueError) as caught:
        validate_distribution_payload(
            wheel,
            sdist,
            repository_root=tmp_path,
            tracked_python_paths=tracked,
        )

    message = str(caught.value)
    assert "unexpected wheel Python payload: olympus/models/untracked.py" in message
    assert "missing wheel Python payload: olympus/models/expected.py" in message
    assert "unexpected sdist Python payload" in message
    assert "tests/test_untracked.py" in message


def test_distribution_payload_validator_rejects_unsafe_archive_members(
    tmp_path: Path,
) -> None:
    tracked = {"olympus/__init__.py"}
    wheel, sdist = _write_distribution_fixtures(
        tmp_path,
        wheel_paths=tracked,
        sdist_paths=tracked,
    )
    with tarfile.open(sdist, mode="w:gz") as archive:
        regular = b"# fixture\n"
        member = tarfile.TarInfo(name="fixture-0.0.0/olympus/__init__.py")
        member.size = len(regular)
        archive.addfile(member, io.BytesIO(regular))
        link = tarfile.TarInfo(name="fixture-0.0.0/olympus/escape.py")
        link.type = tarfile.SYMTYPE
        link.linkname = "../../outside.py"
        archive.addfile(link)

    with pytest.raises(ValueError, match="regular files or directories"):
        validate_distribution_payload(
            wheel,
            sdist,
            repository_root=tmp_path,
            tracked_python_paths=tracked,
        )


def test_distribution_payload_validator_rejects_internal_outreach_document(
    tmp_path: Path,
) -> None:
    tracked = {"olympus/__init__.py"}
    wheel, sdist = _write_distribution_fixtures(
        tmp_path,
        wheel_paths=tracked,
        sdist_paths={*tracked, "docs/TEAM_OUTREACH_NOTES.md"},
    )

    with pytest.raises(ValueError, match="internal outreach document"):
        validate_distribution_payload(
            wheel,
            sdist,
            repository_root=tmp_path,
            tracked_python_paths=tracked,
        )


def test_distribution_payload_validator_rejects_executable_wheel_injection(
    tmp_path: Path,
) -> None:
    tracked = {"olympus/__init__.py"}
    wheel, sdist = _write_distribution_fixtures(
        tmp_path,
        wheel_paths={*tracked, "bootstrap.pth"},
        sdist_paths=tracked,
    )

    with pytest.raises(ValueError, match="forbidden payload"):
        validate_distribution_payload(
            wheel,
            sdist,
            repository_root=tmp_path,
            tracked_python_paths=tracked,
        )


def test_source_distribution_normalization_is_reproducible(tmp_path: Path) -> None:
    paths = ("fixture-0.0.0/PKG-INFO", "fixture-0.0.0/olympus/__init__.py")
    archives: list[Path] = []
    for index, member_paths in enumerate((paths, tuple(reversed(paths)))):
        archive_path = tmp_path / f"fixture-{index}.tar.gz"
        with tarfile.open(archive_path, mode="w:gz") as archive:
            for path in member_paths:
                payload = path.encode("utf-8")
                member = tarfile.TarInfo(name=path)
                member.mode = 0o600 if index == 0 else 0o640
                member.mtime = 1_700_000_000 + index
                member.uid = index + 100
                member.gid = index + 200
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
        normalize_source_distribution(archive_path, epoch=1_800_000_000)
        archives.append(archive_path)

    assert archives[0].read_bytes() == archives[1].read_bytes()


def test_release_workflows_enforce_clean_sources_and_distribution_validation() -> None:
    root = Path(__file__).resolve().parents[1]
    release_workflow = (root / ".github" / "workflows" / "release.yml").read_text()
    ci_workflow = (root / ".github" / "workflows" / "ci.yml").read_text()
    audit_script = (root / "scripts" / "audit_locked_dependencies.py").read_text()

    assert 'test -z "$(git status --porcelain)"' in release_workflow
    assert "--untracked-files=no" not in release_workflow
    assert "dist/SHA256SUMS" not in release_workflow
    assert "release-metadata/SHA256SUMS" in release_workflow
    assert 'git cat-file -t "${GITHUB_REF_NAME}"' in release_workflow
    for required_flag in (
        '"--locked"',
        '"--no-dev"',
        '"--no-emit-project"',
        '"--strict"',
        '"-r"',
        '"--require-hashes"',
        '"--disable-pip"',
    ):
        assert required_flag in audit_script
    for workflow in (release_workflow, ci_workflow):
        assert "scripts/audit_locked_dependencies.py" in workflow
        assert "scripts/build_release_artifacts.py" in workflow
        assert "--verify-reproducible" in workflow
        assert "pip_audit --strict ." not in workflow
        assert "scripts/validate_distribution_payload.py" in workflow
        assert "scripts/validate_installed_wheel.py" in workflow
        assert 'temporary-parent "${RUNNER_TEMP}"' in workflow
        assert "scripts/install_tectonic.sh" in workflow
        assert "make verify PYTHON=.venv/bin/python" in workflow
    installer = (root / "scripts/install_tectonic.sh").read_text()
    assert 'tectonic_version="0.17.0"' in installer
    assert 'shasum -a 256 --check' in installer
    assert 'test "$("${temporary_directory}/tectonic" --version)"' in installer

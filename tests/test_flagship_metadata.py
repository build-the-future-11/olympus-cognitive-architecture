"""Tensor-free admission and notebook contracts, runnable without model allocation."""

from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path

import pytest

from notebooks.build_flagship_runner import build
from olympus.models.capacity import estimate_capacity
from olympus.models.specifications import model_spec
from scripts import qualify_flagship_v2


def test_family_target_range_is_not_a_trained_parameter_count() -> None:
    prometheus = model_spec("prometheus")
    assert prometheus.target_parameters is None
    assert prometheus.target_parameters_min == 12_000_000_000
    assert prometheus.target_parameters_max == 20_000_000_000
    assert model_spec("kronos").target_parameters == 8_000_000_000_000
    assert not prometheus.capability_claimed


def test_sparse_kronos_still_requires_four_terabytes_of_raw_backing() -> None:
    estimate = estimate_capacity(
        total_parameters=8_000_000_000_000,
        active_parameters=1_000_000,
        resident_parameters=2_000_000,
        weight_bits=4,
        device_budget_bytes=16 * 1024**3,
        backing_storage_bytes=100 * 1024**3,
    )
    assert estimate.backing_weight_bytes == 4_000_000_000_000
    assert estimate.fits_declared_device_budget
    assert not estimate.backing_storage_available and not estimate.runnable


def test_capacity_admits_training_state_and_reserve_not_just_active_weights() -> None:
    estimate = estimate_capacity(
        total_parameters=1000,
        active_parameters=10,
        resident_parameters=100,
        weight_bits=16,
        trainable_parameters=20,
        trainable_bytes_per_parameter=16,
        activation_and_kv_reserve_bytes=80,
        device_budget_bytes=599,
        backing_storage_bytes=2000,
    )
    assert estimate.required_device_bytes == 600
    assert not estimate.runnable


@pytest.mark.parametrize("case", ["negative", "bool", "active", "trainable", "bits"])
def test_capacity_refuses_invalid_declarations(case: str) -> None:
    values = dict(
        total_parameters=100,
        active_parameters=10,
        resident_parameters=20,
        device_budget_bytes=1000,
        backing_storage_bytes=1000,
    )
    if case == "negative":
        values["device_budget_bytes"] = -1
    elif case == "bool":
        values["active_parameters"] = True
    elif case == "active":
        values["active_parameters"] = 21
    elif case == "trainable":
        values["trainable_parameters"] = 21
    else:
        values["weight_bits"] = 3
    with pytest.raises(ValueError):
        estimate_capacity(**values)


def test_metadata_and_qualifier_imports_do_not_load_torch() -> None:
    code = """
import importlib.abc, sys
class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise RuntimeError('metadata attempted to load Torch before admission')
sys.meta_path.insert(0, NoTorch())
from olympus.models import estimate_capacity
from olympus.models.specifications import model_spec
from olympus.foundry.readiness import PortfolioReadinessReport
from scripts.qualify_flagship_v2 import qualify
assert 'torch' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=20)


@pytest.mark.parametrize("case", ["low_free", "pageouts", "admitted"])
def test_local_tensor_admission_preserves_memory_floor_and_pageouts(
    monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    snapshots = iter(
        [(512 * 1024**2, 100), (512 * 1024**2, 101 if case == "pageouts" else 100)]
        if case != "low_free"
        else [(511 * 1024**2, 100), (512 * 1024**2, 100)]
    )
    monkeypatch.setattr("scripts.qualify_flagship_v2.platform.system", lambda: "Darwin")
    monkeypatch.setattr(qualify_flagship_v2, "_macos_vm_counters", lambda: next(snapshots))
    monkeypatch.setattr("scripts.qualify_flagship_v2.time.sleep", lambda _: None)
    if case == "admitted":
        assert qualify_flagship_v2.local_tensor_admission()["pageouts"] == [100, 100]
    else:
        with pytest.raises(RuntimeError, match="admission refused"):
            qualify_flagship_v2.local_tensor_admission()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("embed_source", [False, True])
def test_flagship_notebook_pins_source_device_and_offline_scope(
    tmp_path: Path, device: str, embed_source: bool
) -> None:
    notebook = build(tmp_path / device, device=device, embed_source=embed_source)  # type: ignore[arg-type]
    value = json.loads(notebook.read_text())
    receipt = json.loads((notebook.parent / "source-receipt.json").read_text())
    assert len(value["cells"]) == 3
    assert value["metadata"]["kaggle"] == {
        "isGpuEnabled": device == "cuda",
        "isInternetEnabled": False,
    }
    assert f'DEVICE = "{device}"' in value["cells"][1]["source"]
    assert ("ENV['CUDA_VISIBLE_DEVICES'] = '0'" in value["cells"][1]["source"]) == (
        device == "cuda"
    )
    assert (
        f"/kaggle/working/olympus-flagship-{receipt['archive_sha256'][:12]}-{device}"
        in value["cells"][1]["source"]
    )
    if embed_source:
        encoded = base64.b64encode((notebook.parent / "olympus-source.tar.gz").read_bytes())
        assert repr(encoded.decode("ascii")) in value["cells"][1]["source"]
        assert (
            "SOURCE_ARCHIVE = '/kaggle/working/olympus-source.tar.gz'"
            in value["cells"][1]["source"]
        )
        assert "embedded and hash-verified" in value["cells"][0]["source"]
    else:
        assert "SOURCE_ARCHIVE = None" in value["cells"][1]["source"]
    assert "scripts.qualify_flagship_v2" in value["cells"][2]["source"]
    assert "finally:" in value["cells"][2]["source"]
    assert "regressions.log" in value["cells"][2]["source"]
    assert "QUALIFICATION_RECEIPT" in value["cells"][2]["source"]
    assert "persist()" in value["cells"][2]["source"]
    for cell in value["cells"]:
        if cell["cell_type"] == "code":
            compile(cell["source"], "<flagship-qualification>", "exec")
    with pytest.raises(FileExistsError):
        build(tmp_path / device)

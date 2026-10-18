"""Deterministic aligned data for the registered Olympus controlled-mechanism study."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

import torch
from pydantic import Field
from torch import Tensor

from olympus.core.schemas import StrictModel
from olympus.data.olympus_dataset import write_olympus_dataset_manifest

ControlledRegime = Literal["mixed", "clean", "corrupted", "ood"]


class ControlledSplitSpec(StrictModel):
    name: str = Field(min_length=1)
    examples: int = Field(ge=1)
    seed: int = Field(ge=0, le=2**63 - 1)
    regime: ControlledRegime


class ControlledDatasetReceipt(StrictModel):
    schema_version: Literal[1] = 1
    dataset_id: Literal["olympus-controlled-fusion"] = "olympus-controlled-fusion"
    version: Literal["2.0.0"] = "2.0.0"
    license: Literal["CC-BY-4.0"] = "CC-BY-4.0"
    generator: Literal["olympus.data.controlled_fusion"] = "olympus.data.controlled_fusion"
    splits: list[ControlledSplitSpec]
    bundle_sha256: dict[str, str]
    manifest_sha256: dict[str, str]
    source_id_sha256: dict[str, str]
    source_overlap_count: int = Field(ge=0)
    protected_splits: list[str]
    claim_boundary: str


REGISTERED_SPLITS = (
    ControlledSplitSpec(name="train", examples=512, seed=20261201, regime="mixed"),
    ControlledSplitSpec(name="development", examples=128, seed=20261202, regime="mixed"),
    ControlledSplitSpec(name="clean_test", examples=256, seed=20261203, regime="clean"),
    ControlledSplitSpec(
        name="corrupted_test", examples=256, seed=20261204, regime="corrupted"
    ),
    ControlledSplitSpec(name="ood_test", examples=256, seed=20261205, regime="ood"),
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(path)


def _source_ids(spec: ControlledSplitSpec) -> list[str]:
    return [
        f"{spec.name}-{_sha256(f'{spec.seed}:{index}'.encode())[:20]}"
        for index in range(spec.examples)
    ]


def _corruption_plan(
    spec: ControlledSplitSpec, generator: torch.Generator
) -> tuple[Tensor, Tensor]:
    """Return missing and perturbed modality indices; -1 means no intervention."""

    missing = torch.full((spec.examples,), -1, dtype=torch.long)
    perturbed = torch.full((spec.examples,), -1, dtype=torch.long)
    if spec.regime == "mixed":
        selected = torch.rand(spec.examples, generator=generator) < 0.70
        missing[selected] = torch.randint(0, 3, (int(selected.sum()),), generator=generator)
    elif spec.regime in {"corrupted", "ood"}:
        missing = torch.randint(0, 3, (spec.examples,), generator=generator)
        offset = torch.randint(1, 3, (spec.examples,), generator=generator)
        perturbed = (missing + offset) % 3
    return missing, perturbed


def generate_controlled_split(spec: ControlledSplitSpec) -> tuple[dict[str, Tensor], list[str]]:
    """Generate one deterministic split without reading any other split or global RNG."""

    generator = torch.Generator(device="cpu").manual_seed(spec.seed)
    count = spec.examples
    labels = torch.randint(0, 4, (count,), generator=generator)
    missing, perturbed = _corruption_plan(spec, generator)
    noise_scale = 0.12 if spec.regime != "ood" else 0.24

    tokens = torch.randint(8, 31, (count, 8), generator=generator, dtype=torch.long)
    class_tokens = labels + 1
    if spec.regime == "ood":
        class_tokens = torch.tensor([3, 1, 4, 2], dtype=torch.long)[labels]
    tokens[:, 0] = class_tokens
    tokens[:, 1] = 5 + labels.remainder(2)
    tokens[:, 2] = 7 + labels.div(2, rounding_mode="floor")
    attention_mask = torch.ones_like(tokens, dtype=torch.bool)

    image = torch.rand((count, 3, 8, 8), generator=generator) * noise_scale
    for index in range(count):
        label = int(labels[index])
        channel = label % 3
        top = 0 if label < 2 else 4
        left = 0 if label % 2 == 0 else 4
        if spec.regime == "ood":
            channel = (channel + 1) % 3
            left = 4 - left
        image[index, channel, top : top + 4, left : left + 4] += 0.75
    image.clamp_(0.0, 1.0)

    actions = torch.rand((count, 4, 3), generator=generator) * 2.0 - 1.0
    time = torch.arange(4, dtype=torch.float32).view(1, 4, 1)
    label_value = labels.float().view(count, 1, 1) / 3.0
    state_noise = torch.randn((count, 4, 5), generator=generator) * noise_scale
    state_template = torch.cat(
        (
            label_value.expand(-1, 4, 1),
            (label_value + 0.08 * time).expand(-1, -1, 1),
            (labels.remainder(2).float().view(count, 1, 1)).expand(-1, 4, 1),
            (labels.div(2, rounding_mode="floor").float().view(count, 1, 1)).expand(
                -1, 4, 1
            ),
            torch.sin(label_value * torch.pi + time * 0.3),
        ),
        dim=-1,
    )
    states = state_template + state_noise

    for modality, value in enumerate((tokens, image, states)):
        selected_missing = missing == modality
        selected_perturbed = perturbed == modality
        if bool(selected_missing.any()):
            value[selected_missing] = 0
        if bool(selected_perturbed.any()):
            if modality == 0:
                tokens[selected_perturbed, :3] = torch.randint(
                    1,
                    12,
                    (int(selected_perturbed.sum()), 3),
                    generator=generator,
                )
            elif modality == 1:
                image[selected_perturbed] = torch.rand(
                    (int(selected_perturbed.sum()), 3, 8, 8), generator=generator
                )
            else:
                states[selected_perturbed] = torch.randn(
                    (int(selected_perturbed.sum()), 4, 5), generator=generator
                )

    temporal_mask = torch.ones((count, 4), dtype=torch.bool)
    allowed_transitions = torch.ones((count, 4), dtype=torch.bool)
    transition_targets = labels.remainder(3)
    forbidden = (transition_targets + 1).remainder(3)
    allowed_transitions[torch.arange(count), forbidden] = False
    allowed_transitions[:, 3] = True

    corruption_count = missing.ge(0).float() + perturbed.ge(0).float()
    risk_score = 0.05 + corruption_count * 0.15
    intrinsic_risk_targets = corruption_count / 2.0
    padded_actions = torch.nn.functional.pad(actions, (0, 2))
    next_state_targets = states + 0.1 * padded_actions

    bundle = {
        "tokens": tokens,
        "attention_mask": attention_mask,
        "image": image,
        "states": states,
        "actions": actions,
        "temporal_mask": temporal_mask,
        "allowed_transitions": allowed_transitions,
        "risk_score": risk_score,
        "domain_ids": labels.remainder(3),
        "timestamp": torch.arange(count, dtype=torch.float32) / max(1, count - 1),
        "decision_targets": labels,
        "transition_targets": transition_targets,
        "next_state_targets": next_state_targets,
        "intrinsic_risk_targets": intrinsic_risk_targets,
    }
    return bundle, _source_ids(spec)


def _serialize_bundle(bundle: dict[str, Tensor]) -> bytes:
    buffer = io.BytesIO()
    torch.save(bundle, buffer)
    return buffer.getvalue()


def generate_controlled_fusion_dataset(output: Path) -> ControlledDatasetReceipt:
    """Materialize the registered splits and hash-bound two-split manifests once."""

    output = output.resolve()
    receipt_path = output / "receipt.json"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("refusing to overwrite a controlled dataset directory")
    output.mkdir(parents=True, exist_ok=True)

    source_ids: dict[str, list[str]] = {}
    bundle_hashes: dict[str, str] = {}
    for spec in REGISTERED_SPLITS:
        bundle, identifiers = generate_controlled_split(spec)
        payload = _serialize_bundle(bundle)
        path = output / f"{spec.name}.pt"
        _atomic_write(path, payload)
        source_ids[spec.name] = identifiers
        bundle_hashes[spec.name] = _sha256(payload)

    all_ids = [identifier for values in source_ids.values() for identifier in values]
    overlap_count = len(all_ids) - len(set(all_ids))
    if overlap_count:
        raise RuntimeError("controlled split source identities overlap")

    manifest_hashes: dict[str, str] = {}
    for evaluation_name in ("development", "clean_test", "corrupted_test", "ood_test"):
        manifest = write_olympus_dataset_manifest(
            output / f"manifest_{evaluation_name}.json",
            dataset_id="olympus-controlled-fusion",
            version="2.0.0",
            license_name="CC-BY-4.0",
            train_batch=output / "train.pt",
            evaluation_batch=output / f"{evaluation_name}.pt",
            train_source_ids=source_ids["train"],
            evaluation_source_ids=source_ids[evaluation_name],
        )
        assert manifest.manifest_sha256 is not None
        manifest_hashes[evaluation_name] = manifest.manifest_sha256

    receipt = ControlledDatasetReceipt(
        splits=list(REGISTERED_SPLITS),
        bundle_sha256=bundle_hashes,
        manifest_sha256=manifest_hashes,
        source_id_sha256={
            name: _sha256(("\n".join(values) + "\n").encode())
            for name, values in source_ids.items()
        },
        source_overlap_count=overlap_count,
        protected_splits=["clean_test", "corrupted_test", "ood_test"],
        claim_boundary=(
            "Controlled synthetic mechanism evidence only; no natural-data, deployment, "
            "target-scale, or JEPA claim is licensed by this dataset."
        ),
    )
    _atomic_write(
        receipt_path,
        (json.dumps(receipt.model_dump(mode="json"), indent=2, sort_keys=True) + "\n").encode(),
    )
    return receipt


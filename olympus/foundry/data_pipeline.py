from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal, cast

from pydantic import Field, model_validator

from olympus.core.schemas import StrictModel

CapabilityCategory = Literal[
    "instruction",
    "reasoning",
    "mathematics",
    "code",
    "science",
    "tool_use",
    "extraction",
    "planning",
    "agent_behavior",
    "self_correction",
    "research",
    "safety",
]
SplitName = Literal["train", "validation", "test"]

REQUIRED_CATEGORIES: tuple[str, ...] = (
    "instruction",
    "reasoning",
    "mathematics",
    "code",
    "science",
    "tool_use",
    "extraction",
    "planning",
    "agent_behavior",
    "self_correction",
    "research",
    "safety",
)
ALLOWED_LICENSES = {"LicenseRef-Proprietary", "Apache-2.0", "MIT", "CC-BY-4.0"}

_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE = re.compile(r"(?<!\d)(?:\+?\d[\d .()-]{7,}\d)(?!\d)")
_SECRET = re.compile(
    r"(?:api[_-]?key|secret|password|token)\s*[:=]\s*['\"]?[A-Za-z0-9_./+-]{8,}",
    re.IGNORECASE,
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: object) -> bytes:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"{serialized}\n".encode()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(path)


class InstructionExample(StrictModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{2,63}$")
    category: CapabilityCategory
    prompt: str = Field(min_length=8, max_length=4_000)
    response: str = Field(min_length=1, max_length=8_000)
    license: str = Field(min_length=1, max_length=100)
    source: str = Field(min_length=1, max_length=1_000)
    split: SplitName

    @model_validator(mode="after")
    def validate_rights_and_text(self) -> InstructionExample:
        if self.license not in ALLOWED_LICENSES:
            raise ValueError(f"unsupported training license: {self.license}")
        combined = f"{self.prompt}\n{self.response}"
        if "\x00" in combined:
            raise ValueError("training text contains NUL bytes")
        if _EMAIL.search(combined) or _PHONE.search(combined) or _SECRET.search(combined):
            raise ValueError(f"possible PII or secret in example {self.id}")
        return self

    def training_text(self) -> str:
        return f"<|user|>\n{self.prompt.strip()}\n<|assistant|>\n{self.response.strip()}\n<|end|>"


class SourceDescriptor(StrictModel):
    uri: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    license: str
    owner: str
    record_count: int = Field(gt=0)


class SplitDescriptor(StrictModel):
    name: SplitName
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    records: int = Field(gt=0)
    categories: dict[str, int]


class QualityReport(StrictModel):
    input_records: int = Field(gt=0)
    accepted_records: int = Field(gt=0)
    duplicate_ids: int = Field(ge=0)
    normalized_duplicates: int = Field(ge=0)
    pii_or_secret_hits: int = Field(ge=0)
    split_leakage_hits: int = Field(ge=0)
    missing_category_split_pairs: list[str]
    mean_prompt_characters: float = Field(gt=0)
    mean_response_characters: float = Field(gt=0)


class DatasetManifestV2(StrictModel):
    schema_version: Literal[2] = 2
    dataset_id: str
    version: str
    source: SourceDescriptor
    transformations: list[str]
    split_policy: str
    deduplication: str
    quality_policy: str
    contamination_policy: str
    pii_policy: str
    splits: list[SplitDescriptor]
    quality: QualityReport
    manifest_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    def canonical_bytes(self, *, include_hash: bool = True) -> bytes:
        payload = self.model_dump(mode="json")
        if not include_hash:
            payload["manifest_sha256"] = None
        return _canonical_json(payload)


def _normalized_example(example: InstructionExample) -> str:
    return " ".join(f"{example.prompt} {example.response}".casefold().split())


def _ngrams(text: str, size: int = 8) -> set[tuple[str, ...]]:
    tokens = re.findall(r"[a-z0-9_]+", text.casefold())
    return {tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)}


def _assert_no_cross_split_contamination(examples: list[InstructionExample]) -> int:
    seen: dict[tuple[str, ...], tuple[str, str]] = {}
    hits = 0
    for example in examples:
        for gram in _ngrams(f"{example.prompt} {example.response}"):
            previous = seen.get(gram)
            if previous is not None and previous[0] != example.split:
                hits += 1
                raise ValueError(
                    "cross-split 8-token contamination between "
                    f"{previous[1]} and {example.id}"
                )
            seen[gram] = (example.split, example.id)
    return hits


def load_source_examples(source_path: Path) -> list[InstructionExample]:
    examples: list[InstructionExample] = []
    for line_number, line in enumerate(source_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            examples.append(InstructionExample.model_validate_json(line))
        except Exception as error:
            raise ValueError(f"invalid dataset record at line {line_number}: {error}") from error
    if not examples:
        raise ValueError("dataset source contains no records")
    return examples


def prepare_instruction_dataset(
    source_path: Path,
    output_root: Path,
    *,
    dataset_id: str = "olympus-foundry-instructions",
    version: str = "1.0.0",
    source_uri: str = "repository://datasets/hermes-smoke/source.jsonl",
) -> DatasetManifestV2:
    source_payload = source_path.read_bytes()
    examples = load_source_examples(source_path)
    ids = [example.id for example in examples]
    duplicate_ids = len(ids) - len(set(ids))
    if duplicate_ids:
        raise ValueError(f"dataset contains {duplicate_ids} duplicate record IDs")
    normalized = [_normalized_example(example) for example in examples]
    normalized_duplicates = len(normalized) - len(set(normalized))
    if normalized_duplicates:
        raise ValueError(f"dataset contains {normalized_duplicates} normalized duplicates")
    leakage_hits = _assert_no_cross_split_contamination(examples)

    missing: list[str] = []
    for split in ("train", "validation", "test"):
        present = {example.category for example in examples if example.split == split}
        missing.extend(
            f"{split}:{category}" for category in REQUIRED_CATEGORIES if category not in present
        )
    if missing:
        raise ValueError(f"required category/split coverage missing: {', '.join(missing)}")

    output_root = output_root.resolve()
    descriptors: list[SplitDescriptor] = []
    for split in ("train", "validation", "test"):
        selected = sorted(
            (example for example in examples if example.split == split),
            key=lambda item: item.id,
        )
        payload = b"".join(_canonical_json(example.model_dump(mode="json")) for example in selected)
        path = output_root / "splits" / f"{split}.jsonl"
        _atomic_write(path, payload)
        descriptors.append(
            SplitDescriptor(
                name=cast(SplitName, split),
                path=f"splits/{split}.jsonl",
                sha256=sha256_bytes(payload),
                records=len(selected),
                categories=dict(sorted(Counter(item.category for item in selected).items())),
            )
        )

    report = QualityReport(
        input_records=len(examples),
        accepted_records=len(examples),
        duplicate_ids=duplicate_ids,
        normalized_duplicates=normalized_duplicates,
        pii_or_secret_hits=0,
        split_leakage_hits=leakage_hits,
        missing_category_split_pairs=missing,
        mean_prompt_characters=sum(len(item.prompt) for item in examples) / len(examples),
        mean_response_characters=sum(len(item.response) for item in examples) / len(examples),
    )
    manifest = DatasetManifestV2(
        dataset_id=dataset_id,
        version=version,
        source=SourceDescriptor(
            uri=source_uri,
            sha256=sha256_bytes(source_payload),
            license="LicenseRef-Proprietary",
            owner="BU1LD Olympus",
            record_count=len(examples),
        ),
        transformations=[
            "UTF-8 decode and strict schema validation",
            "Unicode-preserving whitespace trim during training serialization",
            "Stable lexical ordering by record ID within explicit splits",
        ],
        split_policy=(
            "Human-authored disjoint train/validation/test assignments; every capability "
            "category occurs in every split. Test records are never used for optimization."
        ),
        deduplication="Exact record-ID and normalized prompt-plus-response rejection.",
        quality_policy=(
            "Reject empty, oversized, malformed, NUL-bearing, unlicensed, PII-bearing, "
            "secret-bearing, duplicate, or category-incomplete records."
        ),
        contamination_policy=(
            "Reject any exact normalized duplicate or shared contiguous 8-token sequence "
            "across split boundaries."
        ),
        pii_policy="Reject email, phone-like, API key, password, secret, and token patterns.",
        splits=descriptors,
        quality=report,
    )
    manifest.manifest_sha256 = sha256_bytes(manifest.canonical_bytes(include_hash=False))
    _atomic_write(output_root / "manifest.json", manifest.canonical_bytes())
    return manifest


def verify_dataset_manifest(manifest_path: Path) -> DatasetManifestV2:
    manifest = DatasetManifestV2.model_validate_json(manifest_path.read_bytes())
    expected = sha256_bytes(manifest.canonical_bytes(include_hash=False))
    if manifest.manifest_sha256 != expected:
        raise ValueError("dataset manifest hash mismatch")
    for split in manifest.splits:
        path = Path(split.path)
        if not path.is_absolute():
            path = manifest_path.parent / path
        if not path.is_file() or sha256_bytes(path.read_bytes()) != split.sha256:
            raise ValueError(f"dataset split hash mismatch: {split.name}")
    return manifest

"""Bounded-memory access to a validated, materialized EditCLEVR snapshot."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Self, overload

import numpy as np
import torch
from PIL import Image
from pydantic import Field, model_validator
from torch import Tensor

from olympus.core.schemas import StrictModel

PAYLOAD_FIELDS = (
    "before_image",
    "after_image",
    "before_scene_json",
    "after_scene_json",
    "instance_masks_before",
    "instance_masks_after",
)
SUITES = frozenset({"atomic_id", "cogent_ood", "hard_distractor", "no_edit"})


class _StreamingJsonReader:
    def __init__(self, path: Path, *, chunk_size: int = 1024 * 1024) -> None:
        self._handle = path.open("r", encoding="utf-8")
        self._chunk_size = chunk_size
        self._buffer = ""
        self._position = 0
        self._eof = False
        self._decoder = json.JSONDecoder()

    def close(self) -> None:
        self._handle.close()

    def _compact(self) -> None:
        if self._position:
            self._buffer = self._buffer[self._position :]
            self._position = 0

    def _fill(self) -> bool:
        if self._eof:
            return False
        self._compact()
        chunk = self._handle.read(self._chunk_size)
        if not chunk:
            self._eof = True
            return False
        self._buffer += chunk
        return True

    def skip_whitespace(self) -> None:
        while True:
            while self._position < len(self._buffer) and self._buffer[self._position].isspace():
                self._position += 1
            if self._position < len(self._buffer) or not self._fill():
                return

    def peek(self) -> str:
        self.skip_whitespace()
        if self._position >= len(self._buffer):
            raise ValueError("unexpected end of JSON input")
        return self._buffer[self._position]

    def expect(self, token: str) -> None:
        if self.peek() != token:
            raise ValueError(f"expected {token!r} in JSON input")
        self._position += 1

    def decode(self) -> Any:
        self.skip_whitespace()
        while True:
            try:
                value, end = self._decoder.raw_decode(self._buffer, self._position)
            except json.JSONDecodeError as error:
                if self._fill():
                    continue
                raise ValueError("invalid or truncated JSON input") from error
            self._position = end
            return value

    def ensure_end(self) -> None:
        self.skip_whitespace()
        if self._position < len(self._buffer) or self._fill():
            raise ValueError("unexpected content after top-level JSON object")


def iter_split_records(
    path: Path,
    *,
    chunk_size: int = 1024 * 1024,
) -> Iterator[tuple[str, int, Any]]:
    """Stream records from the large top-level split object without loading it all."""

    reader = _StreamingJsonReader(path, chunk_size=chunk_size)
    try:
        reader.expect("{")
        if reader.peek() == "}":
            reader.expect("}")
            return
        while True:
            split_name = reader.decode()
            if not isinstance(split_name, str):
                raise ValueError("split registry keys must be strings")
            reader.expect(":")
            reader.expect("[")
            index = 0
            if reader.peek() != "]":
                while True:
                    yield split_name, index, reader.decode()
                    index += 1
                    delimiter = reader.peek()
                    if delimiter == "]":
                        break
                    reader.expect(",")
            reader.expect("]")
            delimiter = reader.peek()
            if delimiter == "}":
                reader.expect("}")
                break
            reader.expect(",")
        reader.ensure_end()
    finally:
        reader.close()


class EditCLEVRPair(StrictModel):
    pair_id: str = Field(min_length=1)
    suite: str = Field(min_length=1)
    edit_factor: str = Field(min_length=1)
    before_image: str
    after_image: str
    before_scene_json: str
    after_scene_json: str
    instance_masks_before: str
    instance_masks_after: str

    @model_validator(mode="after")
    def valid_paths(self) -> Self:
        if self.suite not in SUITES:
            raise ValueError(f"unknown EditCLEVR suite: {self.suite}")
        for field in PAYLOAD_FIELDS:
            raw = getattr(self, field)
            path = PurePosixPath(raw)
            if path.is_absolute() or ".." in path.parts or not path.parts:
                raise ValueError(f"unsafe EditCLEVR path in {field}")
            if path.parts[0] != self.suite:
                raise ValueError(f"EditCLEVR path in {field} does not match suite")
        return self


@dataclass(frozen=True, slots=True)
class EditCLEVRPairPaths:
    pair: EditCLEVRPair
    before_image: Path
    after_image: Path
    before_scene_json: Path
    after_scene_json: Path
    instance_masks_before: Path
    instance_masks_after: Path


def _image_tensor(path: Path) -> Tensor:
    with Image.open(path) as image:
        rgb = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
    return torch.from_numpy(rgb).permute(2, 0, 1).to(dtype=torch.float32).div_(255.0)


class EditCLEVRDataset(Sequence[EditCLEVRPairPaths]):
    """Split-scoped paths plus deterministic RGB, scene, and mask decoders."""

    def __init__(
        self,
        payload_root: Path,
        split_registry: Path,
        split: str,
        *,
        verify_files: bool = True,
    ) -> None:
        self.payload_root = payload_root.resolve()
        if payload_root.is_symlink() or not self.payload_root.is_dir():
            raise ValueError("EditCLEVR payload root must be a regular directory")
        self.split_registry = split_registry.resolve()
        if split_registry.is_symlink() or not self.split_registry.is_file():
            raise ValueError("EditCLEVR split registry must be a regular file")
        self.split = split
        pairs: list[EditCLEVRPair] = []
        known_splits: set[str] = set()
        for split_name, index, raw in iter_split_records(self.split_registry):
            known_splits.add(split_name)
            if split_name != split:
                continue
            if not isinstance(raw, dict):
                raise ValueError(f"{split_name}[{index}] is not an object")
            fields = ("pair_id", "suite", "edit_factor", *PAYLOAD_FIELDS)
            selected = {name: raw.get(name) for name in fields}
            pairs.append(EditCLEVRPair.model_validate(selected))
        if split not in known_splits:
            raise ValueError(f"unknown EditCLEVR split: {split}")
        if not pairs:
            raise ValueError(f"EditCLEVR split is empty: {split}")
        self._pairs = tuple(pairs)
        if verify_files:
            for index in range(len(self._pairs)):
                self._resolve(index)

    def __len__(self) -> int:
        return len(self._pairs)

    @overload
    def __getitem__(self, index: int) -> EditCLEVRPairPaths: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[EditCLEVRPairPaths]: ...

    def __getitem__(self, index: int | slice) -> EditCLEVRPairPaths | Sequence[EditCLEVRPairPaths]:
        if isinstance(index, slice):
            return tuple(self._resolve(item) for item in range(*index.indices(len(self))))
        if index < 0:
            index += len(self._pairs)
        if index < 0 or index >= len(self._pairs):
            raise IndexError(index)
        return self._resolve(index)

    def _path(self, raw: str) -> Path:
        candidate = self.payload_root.joinpath(*PurePosixPath(raw).parts)
        if candidate.is_symlink():
            raise ValueError(f"EditCLEVR payload member is a symlink: {raw}")
        resolved = candidate.resolve()
        if not resolved.is_relative_to(self.payload_root) or not resolved.is_file():
            raise ValueError(f"EditCLEVR payload member is missing or escapes the root: {raw}")
        return resolved

    def _resolve(self, index: int) -> EditCLEVRPairPaths:
        pair = self._pairs[index]
        return EditCLEVRPairPaths(
            pair=pair,
            before_image=self._path(pair.before_image),
            after_image=self._path(pair.after_image),
            before_scene_json=self._path(pair.before_scene_json),
            after_scene_json=self._path(pair.after_scene_json),
            instance_masks_before=self._path(pair.instance_masks_before),
            instance_masks_after=self._path(pair.instance_masks_after),
        )

    def load_rgb_pair(self, index: int) -> tuple[Tensor, Tensor]:
        paths = self[index]
        return _image_tensor(paths.before_image), _image_tensor(paths.after_image)

    def load_scenes(self, index: int) -> tuple[dict[str, Any], dict[str, Any]]:
        paths = self[index]
        values: list[dict[str, Any]] = []
        for path in (paths.before_scene_json, paths.after_scene_json):
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError(f"EditCLEVR scene must be an object: {path}")
            values.append(value)
        return values[0], values[1]

    def load_masks(self, index: int) -> tuple[dict[str, np.ndarray[Any, Any]], ...]:
        paths = self[index]
        results: list[dict[str, np.ndarray[Any, Any]]] = []
        for path in (paths.instance_masks_before, paths.instance_masks_after):
            with np.load(path, allow_pickle=False) as archive:
                results.append({name: np.array(archive[name], copy=True) for name in archive.files})
        return tuple(results)

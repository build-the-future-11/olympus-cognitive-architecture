from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def hash_tree(root: str | Path, include: list[str] | None = None) -> dict[str, str]:
    root = Path(root)
    paths = [root / p for p in include] if include else [p for p in root.rglob("*") if p.is_file()]
    out: dict[str, str] = {}
    for p in sorted(paths):
        if p.exists() and p.is_file():
            out[str(p.relative_to(root))] = sha256_file(p)
    return out

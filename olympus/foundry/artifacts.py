from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


def path_aliases(path: Path) -> set[Path]:
    """Include both the symlink target and the directory entry atomic writes replace."""
    return {path.resolve(), path.parent.resolve() / path.name}


def validate_output_paths(
    outputs: Sequence[Path], inputs: Sequence[Path], *, require_fresh: bool = True
) -> None:
    """Reject input/output aliases and preserve evidence from earlier attempts."""
    protected = {alias for path in inputs for alias in path_aliases(path)}
    for output in outputs:
        aliases = path_aliases(output)
        if aliases & protected:
            raise ValueError("output paths must be distinct from input artifacts and each other")
        if require_fresh and (output.exists() or output.is_symlink()):
            raise ValueError("output already exists; use a fresh output path for this attempt")
        protected.update(aliases)

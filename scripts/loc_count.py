from __future__ import annotations

from pathlib import Path

EXCLUDED = {
    ".git",
    ".venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "dist",
    "coverage",
}
EXTENSIONS = {".py", ".ts", ".tsx", ".md", ".yml", ".yaml", ".json"}


def meaningful_lines(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        if (
            any(part in EXCLUDED for part in path.parts)
            or path.suffix not in EXTENSIONS
            or not path.is_file()
        ):
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith(("#", "//")):
                total += 1
    return total


if __name__ == "__main__":
    print(meaningful_lines(Path.cwd()))

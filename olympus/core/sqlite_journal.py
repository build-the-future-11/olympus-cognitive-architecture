"""SQLite journal-mode helpers for portable local stores."""

from __future__ import annotations

import sqlite3


def apply_durable_journal_mode(connection: sqlite3.Connection) -> str:
    """Prefer WAL; fall back to DELETE when WAL cannot be enabled.

    Some filesystems (network mounts, restricted sandboxes) reject WAL sidecar
    files. Callers still get a durable ``FULL`` synchronous store either way.
    """
    try:
        row = connection.execute("PRAGMA journal_mode = WAL").fetchone()
        mode = str(row[0]).lower() if row else ""
        if mode == "wal":
            return "wal"
    except sqlite3.Error:
        mode = ""

    row = connection.execute("PRAGMA journal_mode = DELETE").fetchone()
    mode = str(row[0]).lower() if row else "delete"
    if mode not in {"delete", "truncate", "persist", "memory", "off"}:
        # Unexpected mode — still return what SQLite reported.
        return mode
    return mode

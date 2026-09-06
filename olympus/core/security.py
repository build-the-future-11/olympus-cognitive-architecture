from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PermissionPolicy:
    allow_network: bool = False
    allow_shell: bool = False
    allow_filesystem_read: bool = True
    allow_filesystem_write: bool = False
    allowed_read_roots: tuple[str, ...] = ()
    allowed_write_roots: tuple[str, ...] = ()
    allow_http_domains: tuple[str, ...] = ()

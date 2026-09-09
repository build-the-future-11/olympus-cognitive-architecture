from __future__ import annotations

import os
import socket
import ssl
import urllib.request
from email.message import Message
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from olympus.core.security import PermissionPolicy
from olympus.data.ingestion import IngestionPipeline, _PinnedHTTPSConnection


def test_https_pins_address_but_preserves_tls_hostname() -> None:
    context = MagicMock(spec=ssl.SSLContext)
    connection = _PinnedHTTPSConnection(
        "data.example.org", address="93.184.216.34", context=context, timeout=5
    )
    with patch("socket.socket") as transport, patch("socket.getaddrinfo") as dns:
        connection.connect()
        transport.return_value.connect.assert_called_once_with(("93.184.216.34", 443))
        dns.assert_not_called()
        context.wrap_socket.assert_called_once_with(
            transport.return_value, server_hostname="data.example.org"
        )
    connection.close()


def test_ingestion_disables_environment_proxies() -> None:
    addresses = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]
    with (
        patch("socket.getaddrinfo", return_value=addresses),
        patch("urllib.request.build_opener", return_value=_Opener(_Response(b"safe"))) as build,
    ):
        IngestionPipeline(_policy()).ingest_http("https://data.example.org/source")
    handlers = build.call_args.args
    proxy = next(
        handler for handler in handlers if isinstance(handler, urllib.request.ProxyHandler)
    )
    assert vars(proxy)["proxies"] == {}


def test_shard_parent_swap_cannot_escape_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    allowed = tmp_path / "allowed"
    nested = allowed / "nested"
    nested.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    original = os.open

    def racing_open(
        path: str | os.PathLike[str], flags: int, mode: int = 0o777,
        *, dir_fd: int | None = None,
    ) -> int:
        if dir_fd is not None and str(path) == "nested":
            nested.rmdir()
            nested.symlink_to(outside, target_is_directory=True)
        return original(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", racing_open)
    policy = PermissionPolicy(allow_filesystem_write=True, allowed_write_roots=(str(allowed),))
    with pytest.raises(OSError):
        IngestionPipeline(policy).shard([], nested / "shard.json")
    assert list(outside.iterdir()) == []


class _Response:
    def __init__(self, payload: bytes, content_type: str = "text/plain") -> None:
        self.payload = payload
        self.headers = Message()
        self.headers["Content-Type"] = content_type
        self.headers["Content-Length"] = str(len(payload))

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, limit: int) -> bytes:
        return self.payload[:limit]


class _Opener:
    def __init__(self, response: _Response) -> None:
        self.response = response

    def open(self, request: object, timeout: int) -> _Response:
        assert timeout == 5
        return self.response


def _policy() -> PermissionPolicy:
    return PermissionPolicy(allow_network=True, allow_http_domains=("data.example.org",))


def test_network_is_denied_by_default() -> None:
    with pytest.raises(PermissionError, match="disabled"):
        IngestionPipeline().ingest_http("https://data.example.org/file.txt")


def test_file_ingestion_is_bounded_and_can_be_root_restricted(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    sample = allowed / "sample.txt"
    sample.write_text("bounded", encoding="utf-8")
    policy = PermissionPolicy(allowed_read_roots=(str(allowed),))
    assert IngestionPipeline(policy, max_file_bytes=7).ingest_file(sample).text == "bounded"

    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    with pytest.raises(PermissionError, match="allowed roots"):
        IngestionPipeline(policy).ingest_file(outside)
    with pytest.raises(ValueError, match="byte limit"):
        IngestionPipeline(policy, max_file_bytes=6).ingest_file(sample)


def test_file_ingestion_rejects_a_symlink_swapped_after_root_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    sample = allowed / "sample.txt"
    sample.write_text("safe", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("private", encoding="utf-8")
    original_open = os.open
    swapped = False

    def racing_open(
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if dir_fd is not None and Path(path) == Path("sample.txt") and not swapped:
            sample.unlink()
            sample.symlink_to(outside)
            swapped = True
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", racing_open)
    policy = PermissionPolicy(allowed_read_roots=(str(allowed),))
    with pytest.raises(OSError):
        IngestionPipeline(policy).ingest_file(sample)
    assert swapped is True


def test_shard_writes_are_denied_by_default(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="writes are disabled"):
        IngestionPipeline().shard([], tmp_path / "shard.json")

    policy = PermissionPolicy(
        allow_filesystem_write=True,
        allowed_write_roots=(str(tmp_path),),
    )
    output = tmp_path / "shard.json"
    IngestionPipeline(policy).shard([], output)
    assert output.read_text(encoding="utf-8") == "[]"


@pytest.mark.parametrize(
    "url, error",
    [
        ("http://data.example.org/file.txt", ValueError),
        ("https://data.example.org:8443/file.txt", ValueError),
        ("https://other.example.org/file.txt", PermissionError),
        ("https://user:secret@data.example.org/file.txt", ValueError),
    ],
)
def test_url_policy_rejects_unsafe_targets(url: str, error: type[Exception]) -> None:
    with pytest.raises(error):
        IngestionPipeline(_policy()).ingest_http(url)


def test_private_resolution_is_rejected() -> None:
    address = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))]
    with patch("socket.getaddrinfo", return_value=address):
        with pytest.raises(PermissionError, match="non-public"):
            IngestionPipeline(_policy()).ingest_http("https://data.example.org/file.txt")


def test_response_is_content_typed_and_bounded() -> None:
    address = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]
    pipeline = IngestionPipeline(_policy(), max_response_bytes=12)
    with (
        patch("socket.getaddrinfo", return_value=address),
        patch("urllib.request.build_opener", return_value=_Opener(_Response(b"hello world"))),
    ):
        document = pipeline.ingest_http("https://data.example.org/file.txt")
    assert document.text == "hello world"

    with (
        patch("socket.getaddrinfo", return_value=address),
        patch(
            "urllib.request.build_opener",
            return_value=_Opener(_Response(b"not plain", "application/octet-stream")),
        ),
    ):
        with pytest.raises(ValueError, match="content type"):
            pipeline.ingest_http("https://data.example.org/file.bin")

    with (
        patch("socket.getaddrinfo", return_value=address),
        patch(
            "urllib.request.build_opener",
            return_value=_Opener(_Response(b"this payload is much too large")),
        ),
    ):
        with pytest.raises(ValueError, match="byte limit"):
            pipeline.ingest_http("https://data.example.org/file.txt")


def test_chunk_sizes_must_be_positive_and_nested_shards_are_created(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("bounded", encoding="utf-8")
    with pytest.raises(ValueError, match="chunk_size"):
        IngestionPipeline().ingest_file(source, chunk_size=0)

    policy = PermissionPolicy(
        allow_filesystem_write=True,
        allowed_write_roots=(str(tmp_path),),
    )
    output = tmp_path / "nested" / "shard.json"
    IngestionPipeline(policy).shard([], output)
    assert output.read_text(encoding="utf-8") == "[]"

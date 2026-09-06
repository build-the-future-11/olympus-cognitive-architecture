from __future__ import annotations

import socket
from email.message import Message
from pathlib import Path
from unittest.mock import patch

import pytest

from olympus.core.security import PermissionPolicy
from olympus.data.ingestion import IngestionPipeline


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

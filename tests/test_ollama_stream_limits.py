import json
from collections.abc import Iterator
from contextlib import closing

import httpx
import pytest

from olympus.foundry.ollama import OllamaClient


class Chunks(httpx.SyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks

    def __iter__(self) -> Iterator[bytes]:
        yield from self.chunks


def decode(chunks: list[bytes]) -> list[object]:
    with closing(httpx.Response(200, stream=Chunks(chunks))) as response:
        return list(OllamaClient._stream_objects(response))


def test_split_unicode_and_final_event() -> None:
    raw = json.dumps({"text": "é"}, ensure_ascii=False).encode()
    index = raw.index(b"\xc3") + 1
    assert decode([raw[:index], raw[index:], b"\n", b"{}", b"\n"]) == [{"text": "é"}, {}]
    assert decode([b"{}"]) == [{}]


@pytest.mark.parametrize("chunks, reason", [
    ([b"x" * 65537], "size limit"),
    ([b"x" * 65537 + b"\n"], "size limit"),
    ([b"\n" * 65536] * 33, "wire limit"),
    ([b"{}\n" * 8193], "event limit"),
    ([b"{}\n" * 8192, b"{}"], "event limit"),
])
def test_rejects_unbounded_provider_input(chunks: list[bytes], reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        decode(chunks)


def test_invalid_json_fails_instead_of_being_skipped() -> None:
    with pytest.raises(ValueError):
        decode([b"not json\n"])


def test_model_listing_is_bounded_before_json_decoding() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=Chunks([b" " * 65536] * 33))

    client = OllamaClient(transport=httpx.MockTransport(handle))
    with pytest.raises(ValueError, match="response exceeds size"):
        client.list_models()

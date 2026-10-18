"""Bounded ASGI load probe for engineering qualification.

The probe is intentionally labelled as a smoke test.  Its output cannot satisfy
the 24-hour production gate in ``readiness.py``.
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import httpx
from pydantic import Field

from olympus.core.schemas import StrictModel


class LoadSmokeReport(StrictModel):
    schema_version: int = 1
    endpoint: str
    requests: int = Field(gt=0)
    concurrency: int = Field(gt=0)
    successful_requests: int = Field(ge=0)
    failed_requests: int = Field(ge=0)
    error_rate: float = Field(ge=0, le=1)
    elapsed_seconds: float = Field(gt=0)
    throughput_requests_per_second: float = Field(ge=0)
    p50_latency_ms: float = Field(ge=0)
    p95_latency_ms: float = Field(ge=0)
    p99_latency_ms: float = Field(ge=0)
    status_counts: dict[str, int]
    qualification: str = "BOUNDED_ENGINEERING_SMOKE_NOT_PRODUCTION_EVIDENCE"


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * quantile)))
    return ordered[index]


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(path)


async def probe_asgi_app(
    app: Any,
    *,
    endpoint: str = "/health",
    requests: int = 200,
    concurrency: int = 16,
) -> LoadSmokeReport:
    if requests < 1:
        raise ValueError("requests must be positive")
    if concurrency < 1 or concurrency > requests:
        raise ValueError("concurrency must be in [1, requests]")
    semaphore = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    statuses: dict[str, int] = {}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://olympus.test"
    ) as client:

        async def one() -> None:
            async with semaphore:
                started = time.perf_counter()
                try:
                    response = await client.get(endpoint)
                    key = str(response.status_code)
                except httpx.HTTPError as error:
                    key = f"error:{type(error).__name__}"
                latencies.append((time.perf_counter() - started) * 1_000)
                statuses[key] = statuses.get(key, 0) + 1

        started = time.perf_counter()
        await asyncio.gather(*(one() for _ in range(requests)))
        elapsed = time.perf_counter() - started

    successful = sum(count for status, count in statuses.items() if status.startswith("2"))
    failed = requests - successful
    return LoadSmokeReport(
        endpoint=endpoint,
        requests=requests,
        concurrency=concurrency,
        successful_requests=successful,
        failed_requests=failed,
        error_rate=failed / requests,
        elapsed_seconds=elapsed,
        throughput_requests_per_second=requests / elapsed,
        p50_latency_ms=statistics.median(latencies),
        p95_latency_ms=_percentile(latencies, 0.95),
        p99_latency_ms=_percentile(latencies, 0.99),
        status_counts=dict(sorted(statuses.items())),
    )


def write_load_smoke_report(report: LoadSmokeReport, output_path: Path) -> None:
    _atomic_json(output_path, report.model_dump(mode="json"))

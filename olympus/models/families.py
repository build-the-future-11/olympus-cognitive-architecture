"""Legacy deterministic demo fixture; not a registered Olympus model family.

``HermesNano`` predates the governed family contracts and writes directly to
the supplied in-memory store. New model/runtime work must use
``HermesWorkspaceRuntime`` and proposal-only memory behavior instead.
"""

from __future__ import annotations

import hashlib
from typing import TypedDict

from olympus.core.interpretation import InterpretiveSuperpositionNetwork
from olympus.memory.store import MemoryRecord, MemoryStore


class HermesResponse(TypedDict):
    response: str
    confidence: float
    category: str


class HermesNano:
    """Backward-compatible demo helper, intentionally excluded from the registry."""
    def __init__(self, memory: MemoryStore) -> None:
        self.memory = memory
        self.interpreter = InterpretiveSuperpositionNetwork()

    def respond(self, prompt: str) -> HermesResponse:
        branches = self.interpreter.analyze(prompt)
        best = branches[0]
        self.memory.put(
            MemoryRecord(
                kind="episodic",
                key=f"prompt:{hashlib.sha256(prompt.encode('utf-8')).hexdigest()}",
                value=best.statement,
                salience=best.confidence,
                tags=["hermes", best.category.value],
            )
        )
        return {
            "response": best.statement,
            "confidence": best.confidence,
            "category": best.category.value,
        }

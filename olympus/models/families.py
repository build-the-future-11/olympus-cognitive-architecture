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

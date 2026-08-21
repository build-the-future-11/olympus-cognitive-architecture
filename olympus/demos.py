from __future__ import annotations

from pathlib import Path

from olympus.core.division import NeuralDivisionReassembly, SpecialistResult, SpecialistTask
from olympus.core.interpretation import InterpretiveSuperpositionNetwork
from olympus.core.representation import DynamicRepresentationTheory, RepresentationFamily
from olympus.core.retrodiction import RetrodictiveIdentityNetwork, code_bug_generators
from olympus.core.transform import train_learned_transform
from olympus.forge.compiler import NaturalLanguageBehaviorCompiler
from olympus.forge.runtime import ForgeRuntime
from olympus.memory.store import MemoryStore
from olympus.models.families import HermesNano, HermesResponse
from olympus.training.synthetic import transform_signals


def run_ambiguous_interpretation_demo() -> dict[str, dict[str, object]]:
    network = InterpretiveSuperpositionNetwork()
    branches = network.analyze("The cloud was a dragon above the city.")
    top_category = branches[0].category.value
    return {
        "ambiguous_interpretation": {
            "passed": len(branches) >= 3 and top_category in {"literal", "concrete_resemblance"},
            "detail": f"Top interpretation category: {top_category}",
        }
    }


def run_retrodiction_demo() -> dict[str, dict[str, object]]:
    current = "def average(total, count):\n    return total / count\n"
    network = RetrodictiveIdentityNetwork()
    results = network.infer(
        current_state=current,
        generators=code_bug_generators(),
        forward_simulator=lambda candidate: (
            candidate.replace("return total / (count - 1)", "return total / count")
            .replace("return total / count if count else 0", "return total / count")
            .replace("total =+ value", "total += value")
        ),
        invariant_checker=lambda candidate: "return" in candidate and "def average" in candidate,
    )
    return {
        "retrodiction": {
            "passed": results[0].forward_match,
            "detail": results[0].candidate_state.splitlines()[-1],
        }
    }


def run_transform_demo() -> dict[str, dict[str, object]]:
    metrics = train_learned_transform(transform_signals())
    return {
        "learned_transform": {
            "passed": metrics.reconstruction_loss < 0.2,
            "detail": f"reconstruction_loss={metrics.reconstruction_loss}",
        }
    }


def run_representation_demo() -> dict[str, dict[str, object]]:
    theory = DynamicRepresentationTheory()
    descriptor = theory.select({"edges": [("a", "b")], "hierarchy_depth": 3})
    return {
        "representation_selection": {
            "passed": descriptor.family == RepresentationFamily.TREE,
            "detail": descriptor.family.value,
        }
    }


def run_division_demo() -> dict[str, dict[str, object]]:
    division = NeuralDivisionReassembly()
    tasks = [
        SpecialistTask(name="research", payload="collect evidence"),
        SpecialistTask(name="planning", payload="sequence the work"),
    ]
    specialists = {
        "research": lambda payload: SpecialistResult(
            name="research", output=f"researched {payload}", confidence=0.82
        ),
        "planning": lambda payload: SpecialistResult(
            name="planning", output=f"planned {payload}", confidence=0.78
        ),
    }
    _, final = division.execute(tasks, specialists)
    return {
        "division_reassembly": {
            "passed": "research" in final.output and "planning" in final.output,
            "detail": final.output,
        }
    }


def run_forge_demo() -> dict[str, dict[str, object]]:
    compiler = NaturalLanguageBehaviorCompiler()
    runtime = ForgeRuntime()
    compilation = compiler.compile(
        "Maintain several possible interpretations, give each a confidence "
        "score, use tools to test predictions, and merge the remaining conclusions."
    )
    result = runtime.execute(compilation.spec, "A dependency graph looked like a forest canopy.")
    return {
        "forge": {
            "passed": "output" in result.outputs and len(result.events) >= 4,
            "detail": result.outputs["output"]["statement"],
        }
    }


def run_hermes_demo(memory_path: Path) -> HermesResponse:
    with MemoryStore(memory_path) as memory:
        hermes = HermesNano(memory)
        return hermes.respond("Summarize the project status and keep it private.")

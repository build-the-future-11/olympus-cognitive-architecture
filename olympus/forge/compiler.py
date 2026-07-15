from __future__ import annotations

from pydantic import Field

from olympus.core.schemas import StrictModel
from olympus.forge.language import BehaviorNode, BehaviorSpec, NodeKind


class CompilerIssue(StrictModel):
    level: str
    detail: str


class NaturalLanguageCompilation(StrictModel):
    spec: BehaviorSpec
    issues: list[CompilerIssue] = Field(default_factory=list)


class NaturalLanguageBehaviorCompiler:
    def compile(
        self, description: str, name: str = "compiled-behavior"
    ) -> NaturalLanguageCompilation:
        lowered = description.lower()
        nodes: list[BehaviorNode] = [
            BehaviorNode(
                id="interpret", kind=NodeKind.INTERPRET, description="Generate interpretations"
            ),
        ]
        issues: list[CompilerIssue] = []
        previous_node = "interpret"

        if "confidence" in lowered or "score" in lowered or "verify" in lowered:
            nodes.append(
                BehaviorNode(
                    id="verify",
                    kind=NodeKind.VERIFY,
                    description="Verify branch support",
                    depends_on=[previous_node],
                )
            )
            previous_node = "verify"
        if "tool" in lowered or "test" in lowered:
            nodes.append(
                BehaviorNode(
                    id="tool",
                    kind=NodeKind.TOOL,
                    description="Execute a deterministic tool check",
                    depends_on=[previous_node],
                )
            )
            previous_node = "tool"
        if "memory" in lowered:
            nodes.append(
                BehaviorNode(
                    id="memory",
                    kind=NodeKind.MEMORY_WRITE,
                    description="Persist useful findings",
                    depends_on=[previous_node],
                )
            )
            previous_node = "memory"

        nodes.append(
            BehaviorNode(
                id="merge",
                kind=NodeKind.MERGE,
                description="Merge surviving branches",
                depends_on=[previous_node],
            )
        )
        nodes.append(
            BehaviorNode(
                id="output",
                kind=NodeKind.OUTPUT,
                description="Return final result",
                depends_on=["merge"],
            )
        )

        if "creative" in lowered and "low-confidence" not in lowered:
            issues.append(
                CompilerIssue(
                    level="warning",
                    detail=(
                        "Creative exploration requested without explicit low-confidence allowance."
                    ),
                )
            )

        spec = BehaviorSpec(
            name=name,
            description=description,
            objective=description,
            nodes=nodes,
            outputs=["output"],
        )
        return NaturalLanguageCompilation(spec=spec, issues=issues)

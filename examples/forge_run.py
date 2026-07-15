from olympus.forge.compiler import NaturalLanguageBehaviorCompiler
from olympus.forge.runtime import ForgeRuntime

compiler = NaturalLanguageBehaviorCompiler()
runtime = ForgeRuntime()
compilation = compiler.compile(
    "Maintain several possible interpretations, give each a confidence "
    "score, use tools to test predictions, and merge the remaining conclusions."
)
result = runtime.execute(compilation.spec, "The cloud was a dragon above the city.")
print(result.outputs["output"])

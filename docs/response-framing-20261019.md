# Versioned response framing correction

This is a correction to generated-answer extraction and evaluation metadata.
It does not change the training objective, tokenizer, checkpoint format, model
weights, numerical promotion thresholds, or any historical dataset or result.

## Source-level defect

The byte-model training target is the trimmed answer followed by the eight
literal bytes `\n<|end|>` and the EOS token (ID 2). The original generation loop
stopped at EOS and then used the raw byte tokenizer to decode its output. EOS
was excluded from the text, but the literal framing suffix remained. A perfect
serialized target such as `{"answer":7}\n<|end|>` therefore failed exact-answer
and JSON scoring. The training prefix also trimmed the prompt while generation
previously preserved its leading and trailing whitespace.

This finding does not determine what any historical trained checkpoint would
score under corrected evaluation. No protected checkpoint has been reevaluated
and no historical metric or promotion outcome is reclassified.

## Output protocols

| Protocol | Prompt | Answer extraction |
| --- | --- | --- |
| `legacy_raw_bytes_v1` | Preserve caller whitespace | Original byte decoding, including a generated literal framing suffix |
| `framed_answer_eos_v2` | Apply the same `strip()` as training serialization | Remove exactly one terminal token sequence for `\n<|end|>` followed immediately by an emitted EOS |

New generation and evaluation calls default to `framed_answer_eos_v2`.
`generate_text`, `evaluate_checkpoint`, and `quantize_checkpoint` accept an
explicit `output_protocol` argument. The evaluation and quantization CLI
commands expose the same choice through `--output-protocol`.

The new protocol checks generated tokens before decoding. It does not stop at
the first textual marker, globally replace markers, or interpret answer content
as transport instructions. Interior markers, JSON strings containing markers,
and an answer that itself ends in a literal marker are preserved. In the last
case, the training target has two consecutive markers and only its final frame
is removed. A marker separated from EOS by another token is not removed.

EOS still terminates generation. If no exact terminal frame precedes EOS, the
remaining decoded text is returned as before. This repair does not add a new
completion-quality gate. A token-limited output without EOS retains a complete
or partial marker, and malformed or truncated UTF-8 retains the tokenizer's
replacement-character behavior.

## Evidence versions and compatibility

New held-out reports use schema 2 and suite
`olympus-held-out-capabilities-v2`. New quantization reports use schema 2 and
suite `olympus-quantization-evaluation-v2`. Both record `output_protocol` and
`max_generation_tokens`. Their loss-normalization metadata remains independent.
The quantized weight artifact itself retains its existing schema 1 and
`olympus-symmetric-int4-v1` or `olympus-symmetric-int8-v1` format.

Historical reports without output metadata are interpreted as
`legacy_raw_bytes_v1`, schema 1, with an unknown generation budget. Explicit
legacy evaluation writes the corresponding v1 suite and preserves legacy
decoding. Historical gate interpretation remains available; missing metadata is
never inferred to mean corrected semantics.

Promotion requires a matching schema, suite and output protocol. New v2
evaluation and quantization evidence must also declare the same positive
generation budget. Mixed protocols, contradictory versions and missing v2
budget metadata cannot satisfy evidence admission. Existing numerical gates are
unchanged. The quantization CLI now accepts `--max-generation-tokens` so both
reports can use the same declared budget.

## Oracle regressions and limits

`tests/test_response_framing.py` supplies deterministic next-token logits to the
actual autoregressive generation loop. Its expected tokens follow a separate
literal byte specification, independent of the training serializer and
tokenizer implementation. It covers plain and JSON targets, literal markers,
EOS boundaries, truncation, Unicode, legacy interpretation, report writers and
mixed-protocol rejection. Its report integration uses fresh artificial
checkpoints and fixtures; there is no training or capability measurement.

The token budget includes answer bytes plus the eight marker bytes and EOS.
A complete framed answer therefore needs its UTF-8 byte length plus nine
generation tokens. The default 48-token budget can complete at most 39 answer
bytes. This correction records the budget; it does not enlarge it based on a
held-out answer or silently repair a truncated training target. Exact-match
normalization and the existing JSON-format predicate are otherwise unchanged.

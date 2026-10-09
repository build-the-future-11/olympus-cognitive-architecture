# Ollama terminal-response contract — 9 October 2026

## Source and decision

This change is based on Olympus PR #18 at
`1351b1fc2b1d690ab7f06d20bd7773762a39cf53` in
[`build-the-future-11/olympus-cognitive-architecture`](https://github.com/build-the-future-11/olympus-cognitive-architecture/pull/18).
It closes the missing-reason exception identified by the independent
completion-contract review.

The nonstreaming `/api/chat` adapter now requires both JSON `done: true` and an
explicit `done_reason` of `stop` or `length`. An omitted or empty reason goes
through the existing provider-error boundary and returns HTTP 503. No normal
stop or generated-content success is published without the provider's reason.
Explicit stop, length-limited completion, request options, response content and
the original reason in successful evidence are preserved.

This is a deliberate strict response contract, not an inferred provider-version
range. Historical examples omit the field, but do not establish that an omitted
reason means a normal stop. Servers or proxies that omit it must supply explicit
termination evidence before their response is accepted. No compatibility flag,
new endpoint or unqualified legacy fallback is introduced.

## Authoritative source binding

Sources checked on 9 October 2026:

- [Official chat endpoint documentation](https://docs.ollama.com/api/chat)
  describes the terminal reason and includes it in the response example.
- [Pinned upstream chat integration test](https://github.com/ollama/ollama/blob/c2b7368d4156656ddb9a23b43f721a841b9c23e0/integration/api_test.go)
  requires a nonempty `ChatResponse.DoneReason` for terminal responses in both
  streaming and nonstreaming cases. The separate `GenerateResponse` test has a
  legacy exception; that is not the chat endpoint used here.
- [Pinned upstream reason serialization](https://github.com/ollama/ollama/blob/c2b7368d4156656ddb9a23b43f721a841b9c23e0/llm/server.go)
  distinguishes ordinary stop from the token limit. Its empty default is not
  converted into a successful normal stop by this adapter.

## Verification

The revised tests ran first against the unchanged PR #18 adapter: **four failed
and 20 passed**. The failing cases were omitted and empty reasons at both the
adapter and actual ASGI route; the route returned HTTP 200 instead of 503.

After the repair, the same focused command passed **all 24 cases**:

```bash
python -m pytest tests/test_ollama_completion_receipts.py \
  tests/test_foundry.py::test_ollama_client_validates_and_generates -q --tb=short
```

The set preserves explicit stop/length and malformed-response controls, adds a
normal-stop API control, and checks that provider errors expose neither a
successful finish reason nor generated content. Scoped Ruff, formatting,
strict Mypy and patch-whitespace checks pass on native CPython 3.14.7.
Independent source review accepted the exact adapter and test blobs.

The maintained full Python suite passed **183 tests** with **87.44% branch-aware
coverage**, exceeding its unchanged 85% gate. The 24 focused cases are included
in that total, not additional. The full suite includes the repository's existing
small synthetic Foundry fixtures; it does not use a live Ollama model.

Responses use HTTPX MockTransport; the API checks use the application's real
ASGI route. No live Ollama model or external inference endpoint was used. These
checks establish response-admission behavior, not model quality, scientific
outcomes, promotion or production acceptance. The parent web lockfile and all
scientific/release controls remain unchanged.

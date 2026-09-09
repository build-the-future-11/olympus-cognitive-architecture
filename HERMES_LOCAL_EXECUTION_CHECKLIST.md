# Hermes local execution checklist — 2026-09-08

Scope: make the existing experimental Hermes terminal integration demonstrably
usable with a real third-party checkpoint. This is not Hermes 12B qualification.

- [x] Diagnose local startup: normal `~/.ollama/models` is a symlink to missing
  `/Volumes/PRO-BLADE/Percy-Storage/ollama/models`. Preserve it rather than delete it.
- [x] Start isolated Ollama 0.33.0 on loopback port 11435 with a separate temporary
  model store. Server reports Apple M4 Metal acceleration, 11.8 GiB available GPU
  memory at startup. This is not measured model peak memory.
- [x] Add `hermes doctor` with bounded connectivity checks, exact selected-model
  identity, JSON output and nonzero failure status. Availability is not qualification.
- [x] Test empty stores, missing selected models and failure-detail redaction.
- [x] Run repository Ruff and strict mypy: PASS, 107 files.
- [x] Run targeted chat/Foundry/CLI tests: PASS, 36 tests.
- [x] Run full Python suite: PASS, 308 tests, 85.17% coverage, unchanged 85% gate.
- [x] Finish checkpoint acquisition: Ollama pull verified SHA-256 and succeeded.
  `qwen3:0.6b` manifest identity:
  `sha256:7df6b6e09427a769808717c0a93cadc4ae99ed4eb8bf5ca557c90846becea435`.
- [x] Verify real two-turn streamed generation and actual terminal CLI (exit 0).
  Prompt: "Remember this word for our conversation: cedar. Reply with ACK."
  Answer: "ACK." (2 chunks, 5.463 seconds wall time including first load).
  Follow-up: "What word did I ask you to remember? Reply with only that word."
  Answer: "cedar" (2 chunks, 0.475 seconds wall time).
  Both finished normally; retained history contains four messages.
  CLI answered "Hello! How can I assist you today?", accepted `/clear` and `/exit`.
  These are two observations, not a benchmark or independent capability evaluation.
- [x] Record exact reusable run commands and operational limitations in
  `docs/hermes-chat.md`. The loopback test service is left running for immediate use;
  persistence across session/reboot is not promised. Weights are in a temporary store.

## Run now

```sh
.venv/bin/python -m olympus.cli hermes chat --base-url http://127.0.0.1:11435 --model qwen3:0.6b
```

If the service has stopped, first start the isolated server using the command in
`docs/hermes-chat.md`. No normal Ollama configuration, broken symlink or existing
model storage was modified. Approximately 523 MB was downloaded to the temporary
store. No model weights were committed or uploaded.

Outside this bounded checklist: streaming backend cancellation certification,
generative document citation validation, independent capability/safety evaluation,
durable model installation, clean release packaging and other family qualification.
These remain unfinished work; none is implied complete by an integration smoke run.

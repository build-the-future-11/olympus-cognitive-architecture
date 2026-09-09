# Experimental Hermes chat

This is an explicit-base Ollama integration, not a released Hermes 12B checkpoint.
It does not download weights, silently choose a model, save conversations, execute
tools or certify factual accuracy. Document-grounded extraction remains the separate
`HermesGroundedService`; this chat command does not claim verified citations.

With the repository environment installed, start your existing Ollama service and
inspect its installed models with `ollama list`. Then run:

```sh
.venv/bin/python -m olympus.cli hermes doctor --model EXACT_INSTALLED_MODEL_NAME
.venv/bin/python -m olympus.cli hermes chat --model EXACT_INSTALLED_MODEL_NAME
```

`doctor` returns JSON and a nonzero exit status for an unavailable backend, missing
selected model or empty model list. Availability is explicitly unqualified and
does not claim successful generation. It does not download weights or run inference.

Provider responses are bounded to 2 MiB decoded wire data; streamed events have
64 KiB/event and 8192-event limits, with a separate 1 MiB generated-text cap.
Oversized/malformed responses fail instead of silently truncating into success.
Network timeouts are inactivity limits, not a hard whole-generation deadline.

If the model store is a broken external-drive symlink, restore that drive or choose
an explicit writable model store with `OLLAMA_MODELS` when starting Ollama. Do not
delete the symlink or original data blindly. For a separate local test instance:

```sh
OLLAMA_HOST=127.0.0.1:11435 OLLAMA_MODELS=/private/tmp/olympus-hermes-models ollama serve
# In another terminal; downloads a third-party test model, not Hermes weights:
OLLAMA_HOST=127.0.0.1:11435 ollama pull qwen3:0.6b
.venv/bin/python -m olympus.cli hermes chat --base-url http://127.0.0.1:11435 --model qwen3:0.6b
```

This temporary store may be removed by OS cleanup; it is not a durable installation.
The Qwen3 tag is mutable; record the digest printed by Hermes for the actual run.
See the [official model listing](https://ollama.com/library/qwen3:0.6b) for format,
download size and license. This small base is for integration verification, not a
qualified replacement for the proposed Hermes 12B model.

The exact installed model name is required. The startup banner identifies its
digest. `/clear` clears session history; `/exit`, EOF or Ctrl-C closes the session.
Responses stream incrementally and are provisional until successful completion.
Malformed or incomplete streams fail without adding the turn to history. Discard
any displayed partial output when a failure is reported. Ctrl-C closes the client session, but
immediate cancellation of server-side computation is not guaranteed.

Defaults are 2048 context tokens and 256 output tokens. `--context-tokens` and
`--max-tokens` change these bounds. Input uses a conservative UTF-8 byte allowance,
not an exact tokenizer measurement; oldest complete turns are dropped when needed,
with a visible notice. Overlong individual prompts are rejected. Model digest
changes before/after generation reject the turn. These checks are not an immutable
server-side snapshot or attestation.

Only the current in-memory conversation is sent to the configured backend. Remote
endpoints require HTTPS; choosing a remote endpoint sends conversation content there.
The application does not save chat, but backend logging/persistence is outside this
guarantee. Backend errors are redacted and failed turns are not added to history.

## Remaining release requirements

- Choose and qualify a licensed checkpoint on the target hardware; record weights,
  tokenizer/template, rights and measured performance. No checkpoint is bundled.
- Verified backend cancellation (client streaming and disconnect are implemented).
- Generative grounded answers with citation validation and adversarial evaluations.
- Explicit model acquisition, integrity verification and clean-machine packaging.
- Frozen capability evaluation, independent evidence and release qualification.

The other model families remain experimental reference components. Adding this CLI
does not complete their training or evaluation requirements.

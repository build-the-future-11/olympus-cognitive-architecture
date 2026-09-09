# API

The supported development deployment binds to `127.0.0.1`. Mutating and model
generation endpoints accept loopback clients without credentials. For any
non-loopback or proxied deployment, configure `OLYMPUS_API_TOKEN` and send
`Authorization: Bearer <token>`; proxied mutations are otherwise denied.
Without a token, both the socket peer and HTTP `Host` must identify loopback;
this prevents an attacker-controlled hostname from using DNS rebinding to gain
local mutation authority.

## Endpoints

- `GET /health`
- `POST /workspace/analyze`
- `POST /forge/compile`
- `POST /forge/run`
- `GET /demos`
- `GET /foundry/status`
- `GET /foundry/overview`
- `GET /foundry/jobs/current`
- `GET /foundry/jobs/history?limit=50&offset=0` (same authentication as current job)
- `POST /foundry/jobs/start`
- `POST /foundry/jobs/cancel`
- `POST /foundry/jobs/retry`
- `POST /foundry/verify`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `GET /foundry/providers/ollama`
- `POST /foundry/providers/ollama/generate`

Built-in background jobs run in a separate local process with a five-minute
wall-clock deadline. Cancellation escalates from SIGTERM to SIGKILL when necessary;
timeouts persist as FAILED with an explicit execution-deadline message. Retrying
starts a new job. This boundary is for trusted built-in verification, not arbitrary
code execution. Foreground `/foundry/verify` is synchronous; resource conflicts
return HTTP 409 without starting the pipeline.

## Example

```bash
curl -s http://127.0.0.1:8000/forge/compile \
  -H 'content-type: application/json' \
  -d '{"behavior":"Maintain several possible interpretations, use tools to test predictions, and merge the result."}'
```

## Foundry verification

Run the bounded real training and evaluation path:

```bash
curl -s -X POST http://127.0.0.1:8000/foundry/verify
```

List registered verified artifacts (including the infrastructure-only bigram;
this is not a list of qualified Hermes/other family releases):

```bash
curl -s http://127.0.0.1:8000/v1/models
```

Generate with the first verified model returned by the registry:

```bash
MODEL_ID="$(curl -s http://127.0.0.1:8000/v1/models | jq -er '.data[0].id')"
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d "{\"model\":\"${MODEL_ID}\",\"messages\":[{\"role\":\"user\",\"content\":\"request verify evidence\"}],\"max_tokens\":80,\"temperature\":0,\"stream\":false}"
```

Usage units are explicitly reported as characters for the verification runtime
rather than mislabeled as tokenizer tokens.

## Python SDK authentication

The SDK accepts the same bearer token. It refuses to send a token over plaintext
HTTP unless the target is loopback:

```python
from olympus.sdk import OlympusSDK

client = OlympusSDK("https://olympus.example", api_token="<token>")
models = client.list_models()
```

Treat the token as a server-side secret. The Forge browser client intentionally
does not persist or embed it; an authenticated remote web deployment needs a
trusted same-origin backend or identity-aware proxy.

# API

## Endpoints

- `GET /health`
- `POST /workspace/analyze`
- `POST /forge/compile`
- `POST /forge/run`
- `GET /demos`
- `GET /foundry/status`
- `POST /foundry/verify`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `GET /foundry/providers/ollama`
- `POST /foundry/providers/ollama/generate`

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

List only promoted model artifacts:

```bash
curl -s http://127.0.0.1:8000/v1/models
```

Generate with the first promoted model returned by the registry:

```bash
MODEL_ID="$(curl -s http://127.0.0.1:8000/v1/models | jq -er '.data[0].id')"
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d "{\"model\":\"${MODEL_ID}\",\"messages\":[{\"role\":\"user\",\"content\":\"request verify evidence\"}],\"max_tokens\":80,\"temperature\":0,\"stream\":false}"
```

Usage units are explicitly reported as characters for the verification runtime
rather than mislabeled as tokenizer tokens.

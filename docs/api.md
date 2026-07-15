# API

## Endpoints

- `GET /health`
- `POST /workspace/analyze`
- `POST /forge/compile`
- `POST /forge/run`
- `GET /demos`

## Example

```bash
curl -s http://127.0.0.1:8000/forge/compile \
  -H 'content-type: application/json' \
  -d '{"behavior":"Maintain several possible interpretations, use tools to test predictions, and merge the result."}'
```


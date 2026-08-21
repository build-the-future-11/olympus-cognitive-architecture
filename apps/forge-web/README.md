# Olympus Forge Web Console

The console renders live Foundry registry counts, verification receipts,
promoted models, checkpoint-backed generation, local-provider discovery, and
cognitive demo results from the Olympus API. It never substitutes sample or
fabricated pass states when the API is unavailable.

## Local development

Start the API from the repository root:

```bash
.venv/bin/uvicorn olympus.api:app --host 127.0.0.1 --port 8000
```

Then start Vite in a second terminal:

```bash
cd apps/forge-web
npm ci
npm run dev
```

Vite proxies `/api/*` to `http://127.0.0.1:8000/*`, so the browser remains on a
single origin.

The **Run verified Foundry pipeline** control performs an actual bounded
training/evaluation/export run. The generated model is an infrastructure
verification artifact and is explicitly labeled as neither Hermes nor a
reasoning model. Ollama discovery is displayed separately from successful model
generation.

## Production build

```bash
cd apps/forge-web
npm ci
npm test
npm run build
```

The deployable static output is written to `apps/forge-web/dist`. The production
web server must serve that directory and reverse-proxy `/api/*` to the Olympus
API after removing the `/api` prefix. This is the same route contract exercised
by the local Vite proxy and the automated tests.

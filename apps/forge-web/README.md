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

The browser bundle deliberately does not accept or embed `OLYMPUS_API_TOKEN`.
A remote deployment that exposes mutation controls therefore needs a trusted,
same-origin backend or identity-aware proxy that authenticates the user and
adds the bearer token server-side. Never place that token in a `VITE_*`
variable: Vite variables are public bundle content. If the gateway uses cookie
authentication, it must also enforce an Origin/CSRF policy. Without that
trusted gateway, static views remain usable but the API correctly rejects
mutation and generation requests.

Serve the production response with a restrictive Content Security Policy
(`default-src 'self'; connect-src 'self'; img-src 'self' data:; style-src
'self'; script-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors
'none'`) and `X-Content-Type-Options: nosniff`. Security headers belong on the
HTTP response rather than in the JavaScript bundle.

While a background workload is active, the console polls only
`/api/foundry/jobs/current`; it refreshes the remaining dashboard state once
the job reaches a terminal state. Polling backs off after consecutive failures
instead of retrying a disconnected API once per second indefinitely.

Foreground verification and background start/retry controls require an
explicit confirmation before they create Foundry artifacts. The background
path remains cancellable at artifact-safe boundaries; the foreground request
is explicitly identified as non-cancellable. Generation results display the
validated checkpoint identifier, checkpoint SHA-256, and runtime rather than
showing model text without its provenance.

# Olympus Model Foundry

The Foundry is the durable boundary between research inputs and model artifacts.
It is operational today for one intentionally small verification model. That
model proves the lifecycle; it does not prove Hermes capability.

## Executable lifecycle

```text
UTF-8 corpus
  → content-addressed materialization
  → immutable dataset registration
  → frozen experiment manifest
  → real character-bigram maximum-likelihood training
  → atomic checkpoint write and SHA-256 identity
  → held-out evaluation against a uniform baseline
  → portable export
  → verified model promotion
  → OpenAI-compatible local generation
```

Run it:

```bash
.venv/bin/python -m olympus.cli foundry verify-pipeline
.venv/bin/python -m olympus.cli foundry status
.venv/bin/python -m olympus.cli foundry list-models
```

Generate through the stable API using the first promoted model:

```bash
MODEL_ID="$(curl -s http://127.0.0.1:8000/v1/models | jq -er '.data[0].id')"
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d "{\"model\":\"${MODEL_ID}\",\"messages\":[{\"role\":\"user\",\"content\":\"request verify evidence\"}],\"temperature\":0,\"seed\":7,\"stream\":false}"
```

## Registry invariants

- A dataset ID and version cannot be rewritten to different content.
- Every experiment references the exact dataset SHA-256 and code commit.
- Every checkpoint has an immutable content hash and owning experiment.
- Evaluation records preserve baseline and candidate metrics together.
- Only a passing checkpoint can enter the model registry.
- Model health re-hashes the checkpoint before reporting `ok`.
- Generation refuses a modified checkpoint.
- Evidence is append-only and sequence ordered.
- SQLite foreign keys, WAL journaling, full synchronization, and explicit
  transactions are enabled.

## Artifact layout

The default runtime root is `artifacts/foundry/` and is intentionally untracked:

```text
registry.sqlite3
datasets/<dataset>/<version>-<sha256>.txt
checkpoints/ckpt_<sha-prefix>.json
exports/ckpt_<sha-prefix>/model.json
exports/ckpt_<sha-prefix>/manifest.json
```

Set `OLYMPUS_FOUNDRY_ROOT` to use another root. The API never accepts an
arbitrary filesystem path from a remote request; its verification endpoint uses
the reviewed repository corpus.

## Stable serving contract

The API exposes `GET /v1/models` and non-streaming
`POST /v1/chat/completions`. The bounded verification runtime counts characters,
not model tokens, and labels usage accordingly. Ollama uses a separate provider
adapter and health endpoint so an installed external model cannot be mistaken
for a Foundry-trained Olympus checkpoint.

## Promotion truth boundary

The built-in model family is `FoundryVerificationBigram`. Its limitations are
stored in the registry:

- character-bigram model;
- not an assistant or reasoning model;
- not Hermes;
- intended only for low-cost infrastructure verification.

Hermes will require a licensed base decision, baseline suite, actual
post-training evidence, quantization measurements, stable local serving, and
Percy integration before promotion under a Hermes name.

## Deep model-development commands

The second Foundry path is a real, bounded transformer-development pipeline. It
exists to validate data, training, adapter, evaluation, quantization, and
promotion controls on hardware that cannot safely train a large model.

```bash
.venv/bin/olympus foundry resource-status
.venv/bin/olympus foundry prepare-dataset \
  --source datasets/hermes-smoke/source.jsonl \
  --output artifacts/foundry/deep/dataset
.venv/bin/olympus foundry train-sft \
  artifacts/foundry/deep/dataset/manifest.json \
  --output artifacts/foundry/deep/training \
  --mode full --epochs 4 --width 48 --layers 2 --heads 4
```

`train-sft` also accepts `--mode lora` and `--mode qlora` with an exact
`--base-checkpoint`. QLoRA uses frozen symmetric 4-bit bases stored as packed
nibbles, not an int8 substitute. `--resume-checkpoint` restores model,
optimizer, completed epoch/step counters, and deterministic generator state.

```bash
.venv/bin/olympus foundry evaluate-checkpoint CHECKPOINT MANIFEST REPORT
.venv/bin/olympus foundry quantize-checkpoint CHECKPOINT MANIFEST OUTPUT --bits 4
.venv/bin/olympus foundry promotion-check hermes-alpha \
  CHECKPOINT MANIFEST EVALUATION QUANTIZATION MODEL_CARD OUTPUT \
  APPROVED_BASE_LICENSE
```

These commands do not promote by naming. The promotion check evaluates
checkpoint/dataset identities, rights, data scale, held-out quality, category
regressions, quantized tool reliability, fresh-process serving, and model-card
candidate evidence. Report decisions and required category/workflow coverage are
recomputed and validated. Detached Ed25519 signatures must bind the exact evidence
hashes, runner identity/source, issuer and validity interval. Configure the reviewed
trust JSON's SHA-256 as `OLYMPUS_PROMOTION_TRUST_SHA256` in operator-controlled
release configuration; pass the bundle and trust store using the CLI's
`--attestation-bundle` and `--attestation-trust-store` options. Training must not
control that configuration or signer credentials. Local self-signing is not
independent scientific validation. Production runners/signers are still unprovisioned.

Promotion writes immutable-by-convention, uniquely named run directories under
`<output-name>.runs/`, each with its own report and (only on success) release
manifest. The requested output is a latest-decision convenience copy, not an
authoritative mutable release record. Consumers must verify the exact run report,
artifact hashes and signature policy. Rechecking a failed candidate does not erase
historical release records. A parsing exception does not produce a new decision;
never treat an older output as the result of a command that failed. The CLI writes
invalid-input attempt records separately under the same runs directory. Exit codes
are 0 for promoted, 1 for valid evidence failing qualification, and 2 for invalid
or unreadable inputs. If output storage itself is unavailable, the CLI explicitly
reports that the diagnostic could not be persisted.

## Job persistence and concurrency

The verification service uses a root-scoped kernel lease and memory/swap preflight
for both synchronous and background entrypoints. Background jobs also use a
root-scoped ownership lease and `jobs.sqlite3` journal. Multiple API workers
sharing the same local root can inspect state and request cancellation. On manager
startup, state/history inspection, and before a new run, a running or
cancel-requested record without a lease owner is marked failed/interrupted.
Existing observers therefore recover after another worker dies without restarting
the API. A new run preserves the abandoned entry as failed in history;
retry is explicit and starts a new run, not a checkpoint continuation. Kernel
leases assume a local POSIX filesystem, not a distributed/network job queue.

`GET /foundry/jobs/history?limit=50&offset=0` provides authenticated, newest-first
history (maximum 100 per page). Shutdown stops admission and requests cancellation
with a finite wait. Uncooperative threads cannot be force-terminated safely;
the trusted custom-runner extension still requires cooperative cancellation.
Built-in background verification now runs in a separate process/session with its
own registry connection. The parent enforces a 300-second deadline (constructor
option `worker_timeout_seconds`, finite and at most 3,600 seconds), terminates the
worker group on cancellation, escalates to SIGKILL after a 0.5-second grace period,
and reaps the worker. A worker watchdog also monitors supervisor disappearance.
Children use isolated Python startup and the parent's package location, not a
package resolved from the working directory or inherited `PYTHONPATH`.
This is lifecycle isolation for trusted built-in code, not a security sandbox for
arbitrary programs. Foreground verification remains synchronous and admitted.

## Deep dataset and resource invariants

Evaluation and quantization reports use schema version 3 and explicitly
record `supervised-token-mean-v2` and `all-records-once-unpacked-v1`. The earlier
implementation averaged batch means, over-weighting short final batches. The
corrected loss sums negative log likelihood
over all unmasked next-token targets and divides by their count. Existing v1/v2
reports remain historical evidence; regenerate evaluation and quantization
reports and obtain new signatures before promotion. Do not relabel old JSON as v3
or compare old and corrected metrics as a measured model improvement. This does
not change the frozen Pantheon studies or fabricate new capability results.

SFT v2 also weights accumulated gradients by supervised-token counts, including
the last partial accumulation group. Training identity includes the objective
version, so v2 runs do not reuse v1 output identities. Resume requires v2 objective
metadata and the same device type and restores Python, Torch CPU and device RNG
states in addition to optimizer/data-order state. CPU dropout resume is tested
against uninterrupted training bit-for-bit. CUDA/MPS resume parity has not been
verified here; this is not a cross-hardware determinism claim. Old checkpoints
remain loadable for evaluation, but cannot silently resume under the new objective.

Version 3 also separates sampling: training category weights and sequence packing
never apply to validation, held-out evaluation or quantization quality checks.
Every evaluation record is visited once without cross-record packing. Resume
requires matching validation-policy metadata; old curves must not silently acquire
a different evaluation population. Quantization checks dataset identity before
writing weights and records the dataset hash in its report; promotion verifies it.
Training mixture keys must be declared categories and weights finite in [0, 64].
The current training sampler still uses rounded repeat counts (positive values
have at least one copy); this is not continuous probability-weighted sampling.

- Every JSONL record carries an ID, category, prompt, response, license, source,
  and explicit split.
- Every capability category must occur in train, validation, and untouched test.
- Exact IDs, normalized examples, and shared eight-token cross-split sequences
  are rejected.
- Email, phone-like, API-key, password, secret, and token patterns are rejected.
- Split and manifest hashes are verified before training, resume, or evaluation.
- Only one model-loading or training workload can hold the exclusive Foundry
  lock. Preflight requires 2 GiB available memory and less than 90% swap use.
- CPU mixed precision requests are recorded as ineffective; the pipeline never
  claims an optimization the backend did not apply.

The measured run and its negative capability result are recorded in
`OLYMPUS_MODEL_FOUNDRY_LEDGER.md`. Base selection, post-training admission, and
future-family plans live under `research/`.

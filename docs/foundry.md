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

These commands do not promote by naming. The promotion check emits a release
manifest only when checkpoint/dataset identities, rights, data scale, held-out
quality, category regressions, quantized tool reliability, fresh-process
serving, and model-card gates all pass.

## Deep dataset and resource invariants

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

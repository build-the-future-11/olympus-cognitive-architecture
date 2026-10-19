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
MODEL_ID="$(curl -s http://127.0.0.1:8000/v1/models | jq -er '.data[] | select(.available) | .id' | head -n 1)"
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d "{\"model\":\"${MODEL_ID}\",\"messages\":[{\"role\":\"user\",\"content\":\"request verify evidence\"}],\"temperature\":0,\"seed\":7,\"stream\":false}"
```

## Registry invariants

- A dataset ID and version cannot be rewritten to different content.
- Every experiment references the exact dataset SHA-256 and code commit.
- Every checkpoint has an immutable content hash and owning experiment.
- Evaluation records preserve baseline and candidate metrics together.
- Initial model admission validates the checkpoint against its immutable experiment
  and dataset identity, reconstructs the exact bigram training-prefix counts, and
  recomputes the held-out comparison. A status field or fabricated passing receipt
  alone cannot admit a model; a true baseline tie still fails.
- Model health and generation use the same single-read checkpoint loader to check
  hash, size, format, provenance, supported runtime, model identity, and recorded
  passing evaluation. These serving checks do not refit the dataset on each request.
- Generation refuses a modified, missing, unsupported, or unverified checkpoint.
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

`GET /v1/models` now includes `available` and `health` for every registered model.
The historical `status` remains visible even if the checkpoint later becomes
unavailable. The web console selects only currently available models, clears
stale selections after failed refreshes, and refreshes preserved run state after
a failed verification. Registry decoding failures return HTTP 503; invalid model
artifacts return HTTP 422 on generation; unknown model IDs return HTTP 404.

`FoundryService.export_checkpoint(model_id, destination)` validates the source
before an atomic replacement. Portable exports must target a location outside
the managed Foundry root and cannot replace any other registered artifact,
including symlink aliases. Exporting to the verified source checkpoint itself is
a no-op. The pipeline retains its own internal export under `exports/`.

Run the complete bounded engineering check, including separate fresh CLI/API
processes and an isolated corruption fixture:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python scripts/verify_foundry_integrity.py \
  --output artifacts/integrity-attempt-1
```

Choose a new output directory for each attempt. The script uses the reviewed
repository corpus, starts only loopback API processes, stops them before exit,
and retains command exits, source/data hashes, metrics, exports, failure receipts,
and process peak RSS. The dated execution, resource-accounting qualification, and
source-to-test map are in [the integrity handoff](foundry-integrity-20261019.md).

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

New SFT runs use `supervised_token_mean_v2`: loss is the mean over supervised
next-token targets, with prompt and padding positions excluded. Gradient
accumulation uses the actual target count across each optimizer window, including
the final partial window. Changing microbatch sizes therefore preserves the
window's objective when its examples and forward behavior are the same.

The normalization version is stored in training logs, checkpoint configuration,
training summaries, held-out reports, and quantization reports. V2 run paths end
in `-tokens-v2`, separating them from historical run paths. Checkpoints without
this metadata retain `legacy_batch_mean_v1`; evaluation and quantization use that
legacy definition. To resume such a checkpoint, pass
`--loss-normalization legacy_batch_mean_v1` explicitly. Resuming with a different
normalization is rejected.

V2 changes optimization trajectories and reported losses. Its measurements must
remain bound to that version and source; they are not interchangeable with the
retained v1 losses in `OLYMPUS_MODEL_FOUNDRY_LEDGER.md`. The dated implementation
record is `research/SFT_TOKEN_NORMALIZATION_20261010.md`.

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

Evaluation and quantization reports identify the exact checkpoint snapshot they
consumed. Promotion reads the actual quantized artifact, verifies its hash, size,
source, schema, tensor encoding, finite values, and runtime loadability, and checks
that complete category/workflow evidence and quantization summaries agree. These
are artifact and report-consistency checks; supplied measurements and serving
claims are not independently authenticated by this gate.

Evaluation, quantization, and promotion require fresh output files. Promotion
also requires a fresh sibling `release-manifest.json`, so a later attempt cannot
overwrite or ambiguously reuse a previous release receipt. Use a separate output
directory for each quantization bit width and each promotion attempt. Output paths
must not alias inputs or each other. These checks preserve earlier negative and
positive evidence without changing historical quality thresholds.

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

# Changelog

All notable changes to Olympus are recorded here.

## 0.2.0 — 2026-08-22

- Added a licensed, versioned instruction-data pipeline with immutable split
  hashes, category coverage, exact/normalized deduplication, cross-split
  contamination checks, quality controls, and PII/secret rejection.
- Added real bounded causal-transformer SFT, LoRA, and nibble-packed 4-bit QLoRA
  with deterministic seeds, sequence packing, category mixing, validation,
  gradient accumulation, atomic checkpoints, adapter exports, and resume state.
- Added an exclusive RAM/swap governor, held-out category and workflow
  evaluation, int4/int8 round-trip quantization measurement, and fail-closed
  hash-bound promotion gates.
- Diagnosed the local 8B Ollama load failure and verified the recovery path with
  a 0.6B Q4_K_M baseline through the real Olympus adapter.
- Added the Hermes base-candidate decision, post-training admission gates,
  Prometheus/Perseus/Atlas/Kronos roadmaps, and a measured Foundry ledger. No
  reserved model identity was promoted.
- Expanded the release suite to 72 tests with 87.90% branch coverage.

## 0.1.0 — 2026-07-23

- Added the typed cognitive runtime, Forge compiler/runtime, LabOS portfolio
  orchestration, local memory, ingestion, retrieval, evaluation, CLI, API, and
  React web console.
- Added deterministic demonstrations and Python/web automated tests.
- Hardened HTTP ingestion with explicit network policy, HTTPS and domain
  allowlisting, public-address validation, redirect denial, content-type
  checks, and bounded response sizes.
- Replaced the web console's synthetic fallback state with live API loading,
  explicit error handling, retry support, and runtime payload validation.
- Removed aspirational application directories and unimplemented model-family
  constants, including unsupported parameter-count claims.
- Added reproducible Python and web distribution build gates.
- Added deterministic training seeds, strict dataset provenance validation,
  explicit SQLite lifecycle management, resilient SDK error handling, and an
  enforced branch-coverage threshold.
- Added continuous dependency audits, CycloneDX SBOM generation, artifact
  attestations, GitHub Releases, and PyPI Trusted Publishing automation.
- Added the durable Olympus Model Foundry: immutable dataset registration,
  experiment provenance, atomic content-addressed checkpoints, held-out baseline
  evaluation, guarded promotion, portable export, append-only evidence, restart
  recovery, and checkpoint tamper detection.
- Added a real low-cost character-language-model golden path, a stable local
  model API, Ollama discovery/generation adapter, and a live Model Foundry web
  console. The verification model is explicitly separated from Hermes claims.

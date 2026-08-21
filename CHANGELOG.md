# Changelog

All notable changes to Olympus are recorded here.

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

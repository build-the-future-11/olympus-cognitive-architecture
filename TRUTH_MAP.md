# Truth Map

| Capability | State | Evidence boundary |
| --- | --- | --- |
| Cognitive behavior runtime | IMPLEMENTED + VERIFIED | Automated unit/integration coverage; engineering behavior only. |
| Forge compiler and runtime | IMPLEMENTED + VERIFIED | Typed compilation, execution, API, and web controls tested. |
| Foundry registry and evidence log | IMPLEMENTED + VERIFIED | Fresh-wheel lifecycle and cross-process SQLite integrity verified. |
| Character-bigram lifecycle verifier | IMPLEMENTED + VERIFIED | Real train/evaluate/checkpoint/export/generate path; not an assistant model. |
| Deep tiny-model training controls | EXPERIMENTAL | SFT/LoRA/QLoRA/resume/quantization execute; promotion gates fail closed. |
| Forge web deployment | IMPLEMENTED + UNVERIFIED | Production build passes; no remote deployment was authorized or executed. |
| LabOS portfolio discovery/reporting | IMPLEMENTED + VERIFIED | 36 real sibling roots reported using bounded top-level markers. |
| LabOS per-project scientific execution | PARTIAL | Only projects with truthful entry points can be run; 34 are discovery-only. |
| Local `qwen3:0.6b` adapter | IMPLEMENTED + VERIFIED | Real local response observed through Olympus adapter. |
| Local `qwen3:8b` adapter | BLOCKED | Swap exhaustion violates the local resource gate. |
| Hermes Alpha | PLANNED | No qualifying immutable checkpoint. |
| Prometheus / Perseus / Atlas / Kronos / Aion | PLANNED | Roadmaps only; names cannot identify models before gate-passing checkpoints exist. |
| ICLR/preprint-quality Olympus claim | PARTIAL | Infrastructure is reproducible; novel model claims lack full frozen multi-seed evidence. |
| Package publication | BLOCKED | Clean artifacts verified locally; publishing requires owner credentials and authorization. |

File existence is never treated as completion. `IMPLEMENTED + VERIFIED` means
the integrated behavior ran successfully in this environment; it does not imply
scientific novelty, external replication, production deployment, or acceptance
by a venue.

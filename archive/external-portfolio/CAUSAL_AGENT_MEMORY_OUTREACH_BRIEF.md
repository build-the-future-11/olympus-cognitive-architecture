# Causal Diagnostics for Agent Memory

Status: external-validation collaboration brief  
Projects: Causal-Memory-Use and NPMS  
Evidence date: 2026-09-02

## Problem

Retrieving a relevant memory is not the same as causally using it. Most memory evaluations score answer quality or retrieval overlap, which can miss ignored memories, overwrite failures and downstream invariance.

## Supported controlled evidence

Causal-Memory-Use retains 245,640 controlled evaluation rows. Its construction-validity result shows that a control agent can retrieve the correct memory 100% of the time while exhibiting zero causal-memory-use score and complete invariance to downstream masking. Agents designed to use memory respond to matched semantic substitution and downstream interventions.

NPMS retains 360,000 main-matrix controlled episode evaluations and 72,000 controlled intervention evaluations, with overwrite-policy, trace-dependence, influence-matrix, calibration and efficiency diagnostics. Its LoCoMo and LongMemEval external pipeline is implemented, resumable and fail-closed, but API-backed results are intentionally absent.

## Requested collaboration

We seek access to one real agent-memory stack for a prespecified matched intervention study:

1. freeze ordinary benchmark questions and system configuration;
2. independently re-ingest matched gold-evidence, annotated-non-evidence and corrupted memories;
3. separate retrieval changes from downstream-use changes;
4. retain raw traces and paired outcomes;
5. publish positive, null or negative results under the same gate.

## Explicit non-claims

- Controlled construction validity does not establish behavior in deployed LLM agents.
- The external LoCoMo/LongMemEval hypothesis is untested until API-backed raw artifacts exist.
- High retrieval accuracy alone is not treated as successful memory use.

Sender identity, public repository URLs, API/data permissions and release status must be confirmed before external distribution.

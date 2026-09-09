# Research Specification

## Research Question
Can a frozen research-package protocol with canonical artifact hashes, isolated execution, explicit provenance, canary contamination checks, and structured numerical comparison localize common reproducibility failures more reliably than numerical agreement or pairwise artifact comparison alone?

## Hypothesis
On controlled packages with known injected faults, canonical package auditing will correctly distinguish clean reproduction from data, environment, implementation, protocol, metric, and hidden-state failures, including correlated/shared drift that pairwise comparison cannot detect.

## Proposed Contribution
Pantheon is a claim-level replication protocol and deterministic auditing layer. The present repository implements package freezing, fresh-workspace execution, provenance capture, disagreement metrics, a rule adjudicator, a seeded-fault benchmark, and a public-data cross-implementation case study.

After this controlled specification was executed, the repository added a frozen external phase using two small local agents on a 17-task CORE-Bench v1.1 OOD subset. That phase is confirmatory with respect to its frozen task/model/runner configuration, but it produced a negative capability result rather than evidence for cross-agent false-consensus behavior.

## Experimental Prediction
The full canonical audit should outperform (1) numeric-only comparison and (2) pairwise artifact comparison on fault localization, especially for shared-bug and hidden-contamination cases.

## Primary Metric
Disagreement localization accuracy on injected-fault packages.

## Secondary Metrics
Exact reproduction rate, direction agreement, standardized disagreement `D_z`, relative effect error, CI overlap, shared-bug false-consensus behavior, and public-data effect agreement.

## Baselines
1. Numeric-only comparison.
2. Pairwise proposer-vs-replicator artifact comparison without a canonical manifest.
3. Full Pantheon canonical audit.

## Ablations
- Remove artifact hashes (numeric-only).
- Remove canonical manifest (pairwise artifact diff only).
- Full system.

## Threats to Validity
Injected faults are intentionally aligned to observable provenance signals; perfect controlled-benchmark localization does not imply real-world performance. The controlled phase uses deterministic scripts. A later external phase used two distinct small local LLM identities through one provider and host, but both returned 0/20 non-null answers and 0/34 agent-authored reports. The public-data case study is a method-level cross-implementation comparison, not a replication of a published paper claim. Twenty synthetic seeds are used; the public case uses ten overlapping train/test splits, which are not independent samples for formal significance testing.

## Success Criteria
The implementation must run from a clean checkout, preserve all run artifacts, pass tests, outperform both baselines on the seeded-fault benchmark, and explicitly detect the correlated shared-bug case.

## Failure Criteria
Failure includes hidden shared state, missing provenance, claims based only on terminal output, silent failed runs, inability to detect shared drift, or describing controlled synthetic validation as demonstrated cross-model agent replication.

For the external phase, artifact completeness must be separated from scientific capability. Harness-normalized report files do not count as agent-authored reports, joint null answers do not count as agreement, and a false-consensus rate is not interpreted when agents fail the minimum answer/report floor.

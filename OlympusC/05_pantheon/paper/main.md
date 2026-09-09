# Pantheon: Canonical Artifact Auditing for Research-Agent Replication

> **Historical controlled-study draft.** This Markdown snapshot predates the
> completed external-agent capability run. The canonical source is `main.tex`;
> its current compiled artifact is
> `../output/pdf/pantheon-canonical-auditing.pdf`. The adjacent `main.pdf` is a
> historical render and is not authoritative. The bounded post-study verdict is
> `poststudy/SCIENTIFIC_READINESS.md`: the artifact is complete, but both
> evaluated agents failed the capability floor. The controlled results below
> remain valid and must not be read as the final scientific conclusion.

## Abstract

Autonomous research agents can propose hypotheses, modify code, execute experiments, and write scientific claims, but agreement between two agent runs is not itself evidence of independent replication. We introduce **Pantheon**, a claim-level replication protocol that freezes a canonical research package, executes proposer and replicator runs in isolated workspaces, records artifact and environment provenance, and adjudicates disagreements using canonical hashes, hidden-state canaries, and numerical result comparisons. We evaluate the current rule-based prototype on a controlled seeded-fault benchmark spanning twenty independently generated packages and eight conditions. Across 160 frozen proposer/replicator comparisons, Pantheon localizes all injected fault classes (160/160), compared with 100/160 for pairwise artifact comparison and 20/160 for numerical comparison alone. The key controlled failure is **correlated shared drift**: when proposer and replicator execute the same corrupted implementation, both weaker baselines declare reproduction in all twenty cases, while canonical auditing detects noncanonical code. We additionally run a cross-implementation public-data case study on the Breast Cancer Wisconsin Diagnostic dataset. A scikit-learn logistic-regression proposer and an independent numpy gradient-descent replicator agree on the direction of improvement over a majority baseline on all 10 prespecified splits, with a mean absolute effect difference of 0.00699 accuracy points. These results validate infrastructure mechanisms, not autonomous-agent reliability: no external LLM agents or published research package were replicated in this study. Pantheon is therefore presented as an evaluation and provenance layer for future research-agent replication benchmarks rather than as a demonstrated solution to cross-agent scientific verification.

## 1. Introduction

Language-model research agents increasingly automate substantial parts of scientific workflows. Systems such as The AI Scientist and Agent Laboratory generate or support ideas, code, experiments, and reports, while AgentRxiv explores collaboration between autonomous laboratories. This creates a reliability problem downstream of generation: if one agent reports a result, what does it mean for a second agent to “replicate” it?

Existing agent benchmarks increasingly measure research reproduction and replication directly. CORE-Bench evaluates computational reproducibility on tasks derived from scientific papers; PaperBench evaluates from-scratch replication of recent machine-learning papers; AutoExperiment varies the amount of code withheld to interpolate from reproduction toward reimplementation. These benchmarks primarily evaluate whether agents can complete research tasks. Pantheon targets a complementary question: **how should two research runs be compared when their outputs agree or disagree?**

A central failure mode is false consensus. Two runs can agree because they share the same bug, stale data, hidden context, metric implementation, or contaminated artifact. Pairwise equality can therefore be misleading. Pantheon introduces a canonical package as a third reference point. A run is not merely compared to another run; both are compared to frozen research artifacts that define the claim boundary, data, protocol, metric, implementation provenance, and environment description.

This paper makes three bounded contributions:

1. a concrete claim-package and provenance protocol for isolated research replication;
2. a deterministic disagreement adjudicator that prioritizes canonical artifact drift and hidden-state evidence before numerical interpretation;
3. a controlled benchmark showing where canonical auditing adds information beyond numerical and pairwise comparisons, plus a public-data cross-implementation case study.

We deliberately do **not** claim that Pantheon has been validated across LLM agents. The current experiments validate mechanisms required for such a study.

## 2. Related Work

### Reproducibility in machine learning

Raff (2019) studied independent reproducibility by manually reimplementing 255 papers without consulting released author code, emphasizing that code release and independent replication are distinct. The NeurIPS reproducibility program formalized reporting and artifact expectations through code submission, a reproducibility challenge, and a checklist (Pineau et al., 2020). Pantheon operationalizes related principles at the claim-package level: enough information must be frozen to tell whether a later run executed the same scientific object.

### Autonomous research agents

The AI Scientist (Lu et al., 2024) automates idea generation, coding, experimentation, visualization, and paper writing. Agent Laboratory (Schmidgall et al., 2025) structures LLM agents across research stages, and AgentRxiv (Schmidgall & Moor, 2025) studies collaborative autonomous research. These systems motivate an explicit separation between collaboration and replication: shared context can improve research while simultaneously making two outputs less independent as verification evidence.

### Research-agent replication benchmarks

CORE-Bench (Siegel et al., 2024) evaluates computational reproducibility agents on 270 tasks from 90 papers. PaperBench (Starace et al., 2025) evaluates agents replicating 20 ICML 2024 papers from scratch. AutoExperiment (Kim et al., 2026) evaluates agents as progressively more implementation is masked. Pantheon is not a competing task benchmark. It is intended as an artifact/provenance and disagreement-resolution layer that could be applied to trajectories from such benchmarks.

## 3. Method

### 3.1 Claim package

Pantheon represents a claim as a tuple

\[
c=(H,P,D,M,E,K),
\]

where \(H\) is the hypothesis, \(P\) the protocol, \(D\) the dataset and hash, \(M\) the metric definition, \(E\) the expected effect or claim boundary, and \(K\) the code/environment provenance. The repository freezes these components into a package containing a claim file, protocol, data manifest, environment lock, analysis implementation, expected schema, stopping rule, and canonical hashes.

The canonical manifest is important because pairwise comparison cannot detect correlated drift. If proposer and replicator both modify the same file in the same way, their pairwise hashes remain equal while both differ from the frozen package.

### 3.2 Isolation and provenance

Each run receives a fresh temporary workspace containing a copy of the package. The analysis executes in a separate operating-system process. The runner records role, seed, status, runtime, canonical and observed hashes, data/code/protocol/environment hashes, metric mode, canary exposure, and an independence vector. In the current implementation, process/workspace isolation is real, but model independence is explicitly recorded as absent because no external LLM model is involved.

### 3.3 Numerical disagreement

For proposer effect \(\hat\delta_P\) with standard error \(s_P\) and replicator effect \(\hat\delta_R\) with \(s_R\), we report

\[
D_z = \frac{|\hat\delta_P-\hat\delta_R|}{\sqrt{s_P^2+s_R^2+\epsilon}},
\]

relative effect error, sign agreement, and confidence-interval overlap. These quantities summarize numerical disagreement but do not determine its cause.

### 3.4 Adjudication order

The prototype rule adjudicator checks:

1. proposer-only canary exposure;
2. replicator data drift from canonical;
3. environment-lock drift;
4. proposer or replicator implementation drift from canonical;
5. protocol drift;
6. runtime metric-mode drift;
7. numerical agreement/disagreement.

This ordering prevents a numerically matching result from masking a provenance violation.

## 4. Experimental Setup

### 4.1 Controlled seeded-fault benchmark

We generate twenty synthetic packages (seeds 0–19), each with 320 observations from

\[
y = 0.8x + 1.25t + \varepsilon, \quad \varepsilon\sim\mathcal N(0,1),
\]

and a canonical OLS analysis estimating the treatment coefficient while adjusting for \(x\). Each package is frozen before execution.

We evaluate eight conditions: clean, data mismatch, environment mismatch, implementation mismatch, protocol ambiguity, metric mismatch, hidden-state contamination, and shared correlated implementation bug. The shared-bug condition modifies proposer and replicator code identically, deliberately creating pairwise consensus around noncanonical code.

Three adjudication methods see the same frozen run pairs:

- **Numeric only:** uses only effect disagreement.
- **Pairwise artifacts:** compares proposer and replicator hashes, but not either run to the canonical package.
- **Pantheon:** canonical audit plus canary and metric-mode evidence.

The benchmark comprises 200 actual subprocess executions and 160 proposer/replicator comparisons. Ground-truth fault labels are determined by the injector.

### 4.2 Public-data cross-implementation case study

We use the Breast Cancer Wisconsin Diagnostic dataset bundled with scikit-learn: 569 examples and 30 numeric features. Raw data are not copied into the repository; the experiment loads the bundled dataset at runtime and records a semantic SHA-256 hash.

For ten prespecified stratified 75/25 splits, both paths fit train-only standardization. The proposer trains scikit-learn logistic regression with L-BFGS. The replicator trains a separately implemented numpy logistic regression using deterministic batch gradient descent. The primary effect is test accuracy minus test-set majority accuracy; ROC AUC is secondary.

This is a method-level cross-implementation case study, not a reproduction of the original WDBC papers.

## 5. Results

### 5.1 Fault localization

| Method | Correct | Localization accuracy |
|---|---:|---:|
| Numeric only | 20 / 160 | 0.125 |
| Pairwise artifacts | 100 / 160 | 0.625 |
| Pantheon | 160 / 160 | 1.000 |

Full Pantheon is correct on every seeded condition in this controlled suite. The result is expected to be easier than natural fault diagnosis because the faults are designed to produce observable provenance signals. The meaningful comparison is where the weaker views provably lack information.

The strongest example is correlated shared drift. In all twenty shared-bug cases, proposer and replicator execute identical corrupted code and produce agreeing outputs. Numeric-only and pairwise-artifact baselines classify all twenty as reproduced. Pantheon detects that both code hashes differ from the frozen canonical implementation and assigns implementation mismatch.

Hidden-state contamination provides a second qualitative distinction. A proposer-only canary is exposed in replicator output without a package-file delta. Numeric-only and pairwise-artifact checks again label all twenty cases as reproduced, while the canary-aware canonical audit localizes contamination.

### 5.2 Public-data cross-implementation agreement

| Metric (mean ± SD over 10 split seeds) | Proposer | Replicator |
|---|---:|---:|
| Accuracy | 0.9727 ± 0.0134 | 0.9783 ± 0.0101 |
| ROC AUC | 0.9872 ± 0.0101 | 0.9948 ± 0.0050 |
| Accuracy − majority | 0.3434 ± 0.0134 | 0.3490 ± 0.0101 |

The improvement direction agrees on all 10 splits. Mean absolute difference in the effect metric is 0.00699 accuracy points; the maximum observed difference is 0.02098. These split-wise summaries are descriptive because the train/test splits overlap and are not independent studies. We do not claim that one optimizer is significantly better than the other.

## 6. Ablations

The benchmark supports a direct evidence ablation because all three methods are evaluated on the same run artifacts.

| Evidence available | Accuracy |
|---|---:|
| No artifact hashes | 0.125 |
| Pairwise artifacts, no canonical manifest | 0.625 |
| Full canonical audit | 1.000 |

Removing the canonical reference is specifically damaging when errors are correlated across proposer and replicator. Removing artifact evidence altogether collapses causal localization into generic numerical agreement/disagreement.

## 7. Analysis

### What the controlled result establishes

The benchmark establishes that canonical reference artifacts, canaries, and runtime metric provenance contain information that is absent from pairwise numerical agreement. It also provides a concrete executable protocol for freezing and comparing research claims.

### What it does not establish

The fault taxonomy is engineered and small; the adjudicator rules are aware of the kinds of provenance channels that the benchmark manipulates. Perfect localization therefore does not estimate performance on naturally occurring research failures. No LLM agent generates, interprets, or adjudicates a claim in the present study. A system could pass this benchmark while failing on semantically subtle protocol ambiguities or adversarially equivalent artifacts.

### Toward a harder evaluation

The next high-value experiments are to run Pantheon on trajectories from real research-agent benchmarks, blind the replicator to proposer numerical outputs, and compare canonical auditing against human adjudicators on naturally occurring failures. Cross-model replication should treat model/provider identity, context isolation, filesystem visibility, network policy, and cache state as first-class independence variables.

## 8. Limitations

1. **No external research agents.** The current study validates infrastructure, not agent cognition or autonomy.
2. **No published-paper replication.** The public case study is not a replication of a paper claim.
3. **Synthetic benchmark structure.** Injected failures are easy to localize because they map to explicit provenance channels.
4. **Hash semantics.** Byte-level code drift is not equivalent to semantic method drift.
5. **Isolation scope.** Fresh workspaces/processes do not equal separate containers, hosts, or model providers.
6. **Small taxonomy.** Statistical disagreement, underpowered claims, interpretation disagreement, and external-validity failures require richer experiments.
7. **No human baseline.** We do not yet know whether the adjudicator improves on expert artifact review.

## 9. Threats to Validity

The main internal-validity risk is benchmark-adjudicator co-design: the same project specifies the injected fault taxonomy and the evidence channels used for classification. We mitigate overclaiming by treating the benchmark as mechanism validation only. For the public dataset, preprocessing is train-only and split seeds are prespecified, but overlapping splits mean repeated results should not be treated as independent samples for inferential testing. External validity to agent-generated science is untested.

## 10. Conclusion

Pantheon turns “another agent agreed” into a more structured question: did the second run execute the same frozen scientific object, under what independence conditions, and what artifact evidence explains any disagreement? In controlled experiments, canonical package auditing catches seeded failures that numerical and pairwise comparisons cannot, including correlated shared implementation drift. A small cross-implementation real-data case study shows the protocol can compare independently implemented methods without requiring exact numerical equality. The next step is not to enlarge the claim, but to apply this machinery to genuine independent research agents and published research packages.

## References

See `paper/references.bib` and `research/RELATED_WORK.md`.

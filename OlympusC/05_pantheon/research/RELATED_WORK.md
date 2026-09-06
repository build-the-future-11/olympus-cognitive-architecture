# Related Work and Novelty Check

## Automated research agents

**The AI Scientist** (Lu et al., 2024; arXiv:2408.06292) demonstrated an end-to-end framework in which language-model agents generate ideas, write code, run experiments, visualize results, and draft papers. Pantheon targets a different stage: independent verification and disagreement diagnosis after a claim package is frozen.

**Agent Laboratory** (Schmidgall et al., Findings of EMNLP 2025, DOI 10.18653/v1/2025.findings-emnlp.320) organizes LLM agents across literature review, experimentation, and writing. **AgentRxiv** (Schmidgall & Moor, 2025; arXiv:2503.18102) studies collaborative autonomous research. These systems motivate the need to distinguish collaboration from independent replication.

## Research-agent replication benchmarks

**CORE-Bench** (Siegel et al., 2024/2025; arXiv:2409.11363, TMLR) evaluates computational reproducibility agents on 270 tasks from 90 scientific papers. **PaperBench** (Starace et al., ICML 2025, PMLR 267) evaluates agents replicating 20 ICML 2024 papers from scratch. **AutoExperiment** / *From Reproduction to Replication* (Kim et al., ICLR 2026) evaluates research agents under progressive code masking from rerun-like reproduction toward from-scratch replication.

Pantheon is complementary rather than a replacement: the distinctive proposed contribution is a **claim-package provenance protocol and artifact-grounded disagreement taxonomy** for comparing proposer and replicator outputs, especially correlated agreement where both parties drift from a frozen canonical package.

## Reproducibility methodology

Raff (NeurIPS 2019; arXiv:1909.06674) empirically studied independent reproducibility by reimplementing 255 papers without consulting released author code. Pineau et al. (2020; arXiv:2003.12206) documented the NeurIPS reproducibility program and checklist. Pantheon adopts the same underlying principle that code availability and same-code reruns are not identical to independent replication.

## Novelty boundary

The current repository does **not** support a claim that Pantheon is the first benchmark for AI research replication; CORE-Bench, PaperBench, and AutoExperiment are direct prior art. A defensible novelty direction is narrower: canonical-package auditing plus explicit disagreement localization and hidden-state/false-consensus tests as an evaluation layer that can sit on top of agent replication benchmarks. This novelty still requires broader empirical comparison on real agent trajectories.

## Verified references

1. Edward Raff. “A Step Toward Quantifying Independently Reproducible Machine Learning Research.” NeurIPS 2019. arXiv:1909.06674.
2. Joelle Pineau et al. “Improving Reproducibility in Machine Learning Research (A Report from the NeurIPS 2019 Reproducibility Program).” arXiv:2003.12206, 2020.
3. Chris Lu, Cong Lu, Robert Tjarko Lange, Jakob Foerster, Jeff Clune, David Ha. “The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery.” arXiv:2408.06292, 2024.
4. Zachary S. Siegel, Sayash Kapoor, Nitya Nadgir, Benedikt Stroebl, Arvind Narayanan. “CORE-Bench: Fostering the Credibility of Published Research Through a Computational Reproducibility Agent Benchmark.” arXiv:2409.11363; TMLR 2025.
5. Samuel Schmidgall et al. “Agent Laboratory: Using LLM Agents as Research Assistants.” Findings of EMNLP 2025, pp. 5977–6043. DOI:10.18653/v1/2025.findings-emnlp.320.
6. Samuel Schmidgall and Michael Moor. “AgentRxiv: Towards Collaborative Autonomous Research.” arXiv:2503.18102, 2025.
7. Giulio Starace et al. “PaperBench: Evaluating AI’s Ability to Replicate AI Research.” ICML 2025, PMLR 267:56843–56873.
8. Gyeongwon James Kim, Alex Wilf, Louis-Philippe Morency, Daniel Fried. “From Reproduction to Replication: Evaluating Research Agents with Progressive Code Masking.” ICLR 2026.

5. Nitya Nadgir et al. “Life After Benchmark Saturation: A Case Study of CORE-Bench.” arXiv:2606.26158 (2026). The paper releases CORE-Bench v1.1 and an OOD suite and argues for evaluating reliability, efficiency, model/scaffold effects, and construct validity after raw benchmark accuracy saturates. Pantheon’s final external harness defaults to a curated subset of this OOD suite because it is a materially stronger test than the original public training split. Public availability still creates a contamination threat, which Pantheon records rather than treating OOD as perfectly hidden.

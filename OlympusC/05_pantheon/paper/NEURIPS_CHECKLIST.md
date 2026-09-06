# NeurIPS-Style Research Checklist (project self-audit)

This file is a working self-audit aligned with the current NeurIPS paper-checklist themes. It is not a substitute for the official LaTeX checklist required by a specific submission year.

1. **Claims reflect scope:** Yes. The paper explicitly limits claims to controlled mechanism validation and a public-data case study.
2. **Limitations discussed:** Yes. See Sections 8–9 and `RESEARCH_TRUTH.md`.
3. **Theory assumptions/proofs:** N/A. No theorem is claimed.
4. **Experimental reproducibility:** Yes for the experiments in this repository. Exact commands are in the README and `scripts/reproduce_all.sh`.
5. **Open access to code/data:** Code and generated synthetic data are contained here. Public WDBC data are loaded from scikit-learn at runtime rather than redistributed.
6. **Experimental details:** Yes. Seeds, split procedure, implementations, fault definitions, and metrics are recorded.
7. **Statistical uncertainty:** Partially. Mean and SD are reported for ten public split seeds; no significance claim is made because splits overlap. Controlled faults use exact counts.
8. **Compute resources:** Yes at prototype level. Run registry records CPU execution and runtime; hardware is not benchmarked across devices.
9. **Code of ethics / societal impact:** The main risk is false confidence in automated scientific verification. The paper emphasizes that provenance checks do not establish scientific truth.
10. **Human subjects / crowdsourcing:** N/A.

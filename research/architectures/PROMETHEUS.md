# Prometheus Architecture

## Purpose and falsifiable claim

Prometheus performs multi-source scientific synthesis and checkable reasoning.
Its testable contribution is a proposer/verifier separation that improves valid
conclusions per unit of inference compute—not longer hidden reasoning text.

## Target architecture

1. **Problem formalizer:** emits variables, assumptions, candidate hypotheses,
   evidence gaps, and falsification tests as a `Proposal` envelope.
2. **Branch proposer:** a Prometheus LoRA adapter samples a bounded hypothesis
   lattice. Branches store concise claims and planned checks, not private prose.
3. **Tool-backed checker:** calculators, code sandboxes, proof checkers, and
   retrieval operate through Perseus receipts where available.
4. **Process verifier:** a separately trained adapter or small encoder scores
   each checkable transition. Training and evaluation split proposer and
   verifier authorship/source to reduce correlated error.
5. **Evidence adjudicator:** reconciles support and refutation edges, preserving
   unresolved conflicts rather than forcing consensus.
6. **Selector:** chooses or abstains using verifier score, evidence coverage,
   branch diversity, and budget. The final answer routes through Hermes.

Proposer and verifier initially share frozen backbone weights but have disjoint
adapters and heads. If correlated-error tests fail, move the verifier to a
separate architecture before scaling the proposer.

## Data and losses

Train on licensed proofs and derivations, executable mathematics, systematic
reviews with source spans, retracted/contradictory claims, negative results, and
perturbed invalid steps. Add branch-diversity, process-validity, contradiction,
and final-outcome losses. Teacher-generated examples are quarantined by teacher
revision and never enter the sequestered test set.

## Evaluation and stop rule

Compare greedy decoding, equal-token self-consistency, tree search without a
learned verifier, and proposer+verifier. Measure exact final outcome, verified
step precision/recall, citation entailment, contradiction detection,
calibration, abstention, compute, and verifier/proposer error correlation.
Prometheus is rejected if equal-compute sampling matches it or if verifier score
selects persuasive but invalid branches more often than an outcome checker.

Target input: `AuthorizedWorkspaceView + ResearchQuestion`. Target output:
`Proposal | Answer | Stop`.
Current state: **reference implementation exists, no qualifying checkpoint**.
`olympus/models/prometheus.py` implements bounded typed branches, objective
comparisons, canonical transition receipts, trainable proposer/verifier losses,
and deterministic selection/abstention. The selector uses an authorized
workspace view and counts only host-allowlisted receipt hashes whose canonical
receipt binds the workspace ID, model identity, branch, claim, and compared
values. The host trust decision is not a receipt signature or issuer-validation
service, and the learned scorers are not composed with the selector. This code
does not establish scientific-synthesis capability.

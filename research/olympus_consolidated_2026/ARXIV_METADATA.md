# arXiv submission metadata

- Title: Simple Controls Remain Hard to Beat in Learned Multimodal Routing:
  Five Controlled Studies and a Supported Residual Dynamics Result
- Author: Ryan
- Affiliation: Independent Researcher
- Primary category: cs.LG
- Suggested secondary category: cs.AI
- Comments: 7 pages, 3 tables. Five controlled routing studies and one
  residual-dynamics study. Source and evidence map are available in the linked
  repository.
- Report number: none
- Journal reference: none
- DOI: none

## Abstract

Adaptive fusion is often motivated by the intuition that a learned router should
down-weight unreliable inputs better than a fixed mixture. We tested this intuition
in a sequence of prospectively specified studies built around Olympus, an auditable
reference architecture for heterogeneous prediction. Four synthetic studies
progressively added explicit reliability supervision, independently trained experts,
cross-fitted routing, broader corruption coverage, and development-only route
selection. A fifth study used the licensed EditCLEVR paired-image benchmark. Across
all five studies, the registered learned routing candidate did not beat its strongest
matched simple control on the primary endpoint. The successive primary effects were
-0.0469 selective-accuracy points versus uniform fusion, -0.0125 versus semantic
routing, -0.0682 accuracy points versus semantic routing, -0.00068 versus semantic
routing on novel shift, and -0.00005 versus a development-selected EditCLEVR control.
In contrast, a separately registered residual temporal predictor reduced clean
mean-squared error from 0.07133 for the prior predictor to 0.01457, beat persistence
by 0.01540 with a paired descriptive 95% interval of [-0.01583, -0.01492], and
retained corrupted-input recovery. The evidence supports mixed-corruption training,
sample-dependent semantic routing in one observed corruption regime, and residual
prediction for the synthetic dynamics task. It does not support learned routing as a
general improvement, natural-data robustness, deployment, safety efficacy, or a full
cognitive architecture. The contribution is a preserved sequence of falsifiable
iterations showing where increasingly elaborate routing repairs failed, where one
simpler architectural prior succeeded, and how negative outcomes can be reported
without retroactive endpoint selection.

## Submission boundary

The repository is prepared for arXiv upload but has not been submitted. The author
must use an authenticated arXiv account, confirm the displayed author name and
affiliation, select the submission license, and perform the external submit action.
No repository receipt may be interpreted as evidence that those owner-controlled
steps occurred.

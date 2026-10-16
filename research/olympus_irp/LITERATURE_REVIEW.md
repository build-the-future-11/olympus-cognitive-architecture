# Literature and novelty audit

## Field boundary

The project sits at the intersection of heterogeneous multimodal fusion, mixture-of-
experts routing, selective prediction and calibrated risk, constrained decision systems,
world models, and joint-embedding predictive learning. None of those ingredients is new
by itself.

I-JEPA established masked prediction in representation space, while V-JEPA 2 extended
the approach to video, action-conditioned world modeling, and planning. Expert Choice
and Switch Transformers study routing capacity, sparse activation, imbalance, and
training instability. Auxiliary-loss-free balancing directly challenges the use of a
balance penalty because such a penalty can inject interfering gradients. SelectiveNet
optimizes a reject option and reports risk-coverage behavior. Conformal Risk Control
adds finite-sample control for monotone losses when its calibration assumptions hold.
Deep ensembles remain a strong uncertainty baseline. Recent missing-modality studies
show that fusion strategies are dataset-dependent and may be fragile under absent or
low-quality modalities.

## Exact novelty boundary

The defensible candidate contribution is not "multimodal fusion," "mixture of experts,"
"uncertainty," "counterfactual ablation," "risk control," or "JEPA." It is the combined
contract in which:

1. heterogeneous representations are projected and normalized before soft gating;
2. every leave-one-component-out decision is emitted as a first-class audit output;
3. representation disagreement conditions a separately supervised intrinsic-risk head;
4. learned intrinsic risk combines with host risk by a monotone noisy-OR, so learned
   evidence cannot lower the host risk floor; and
5. host-owned legal-transition masks and a deterministic STOP threshold remain outside
   learned authority.

The current search did not identify an exact prior implementation of that full contract.
That is a provisional non-collision finding, not proof of novelty. An independent search
must repeat the review before submission.

## Strongest alternatives

- best individual family and a matched single-backbone model;
- concatenation plus MLP;
- uniform projected averaging;
- ordinary softmax gating;
- sparse token- or expert-choice routing;
- loss-free routing bias instead of an auxiliary balance loss;
- modality dropout and masked-modality learning;
- host risk alone, model risk alone, maximum fusion, and calibrated conformal control;
- entropy, maximum softmax probability, SelectiveNet, and deep ensembles for failure
  detection;
- bicubic, pixel-only, latent-only, and matched-compute reconstruction models for JEPA
  super-resolution.

## Sources reviewed

- Assran et al. (2023), I-JEPA: https://arxiv.org/abs/2301.08243
- Assran et al. (2025), V-JEPA 2: https://arxiv.org/abs/2506.09985
- Zhou et al. (2022), Expert Choice: https://arxiv.org/abs/2202.09368
- Wang et al. (2024), auxiliary-loss-free balancing:
  https://arxiv.org/abs/2408.15664
- Fedus et al. (2021), Switch Transformers: https://arxiv.org/abs/2101.03961
- Geifman and El-Yaniv (2019), SelectiveNet: https://arxiv.org/abs/1901.09192
- Angelopoulos et al. (2022), Conformal Risk Control:
  https://arxiv.org/abs/2208.02814
- Lakshminarayanan et al. (2016), deep ensembles:
  https://arxiv.org/abs/1612.01474
- Ma et al. (2022), missing-modality robustness:
  https://arxiv.org/abs/2204.05454
- Maheshwari et al. (2023), M3L: https://arxiv.org/abs/2304.10756
- Loshchilov and Hutter (2016), SGDR: https://arxiv.org/abs/1608.03983

## One-sentence research direction

Existing fusion approaches struggle to remain auditable and robust when heterogeneous
components are missing, corrupted, or safety-relevant; Olympus proposes host-monotone
counterfactual fusion because explicit component interventions and monotone risk
composition predict improved failure detection without allowing learned evidence to
weaken external constraints.

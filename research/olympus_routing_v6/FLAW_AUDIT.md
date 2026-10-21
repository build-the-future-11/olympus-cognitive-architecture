# Flaw, cause, severity, repair and acceptance evidence

This checklist distinguishes implemented repairs from scientific evidence.
No implementation is a proof of empirical improvement.

| Flaw / cause | Severity | Repair | Required evidence / present state |
|---|---|---|---|
| Routing novelty overlaps MoE/stacking | High | Narrow contribution to empirical decomposition | Prior-art comparison; no novelty or superiority asserted |
| Correctness probabilities normalized as mixture weights optimize a proxy | High | Direct mixture NLL plus reference correction | Direct-NLL controls; successor outcome not yet observed |
| Familiar/shift tradeoff | High | Matched mean/max-group objective factorial | All regimes and nonregression gates; not yet evaluated |
| Expert redundancy / weak headroom | High | Expert-only controls and raw per-expert probabilities | Rescue/harm and error-overlap diagnostics; no oracle called an upper bound |
| Reference reuses correction labels | High | Four distinct meta-data partitions | Identity rejection and complete split manifest |
| Router train/inference ensemble mismatch | High | Same two-bank averaged states/probabilities throughout | Replay through reconstructed banks and routers |
| Copied frozen alternative silently cannot train | Critical implementation | Explicit requires_grad_(True) on copied alternative | Regression starts from frozen production reference |
| Self-consistent rehashed prediction bundle could pass | Critical integrity | Regenerate data, reload checkpoint and compare all raw values | Source-ID and rehashed-prediction adversarial regressions |
| Checkpoint replay omits bank parameters | High | Recompute selection features using restored banks | Exact full bank/head/alpha replay |
| Missing/relabelled cells accepted by count | High | Exact tuple and file matrices | Reject changed manifest membership and denominator |
| More updates explain improvement | High | Direct controls continue for same 300 steps | Same data/order/optimizer; retained update budgets |
| Constant mixing could suffice | High | Constant ERM and robust controls | Primary selected-control comparison includes both |
| Uniform or single expert could suffice | High | Fixed/uniform and all five single experts | Same-bank control selection on separate development |
| Anchor parameter confound | Medium | Report 106 adaptive vs one constant vs zero direct parameters | Do not claim exact parameter matching |
| Anchor penalty confounded with mixing | Medium | Adaptive robust no-penalty ablation | Fixed secondary contrast; no post-test selection |
| Frozen alpha initialization could block alternative learning | Medium | Alpha starts .1; direct endpoint controls | All correction parameters get finite gradients |
| Empty groups or unequal averaging | High | Require every group and average group means explicitly | Hand-calculated loss test |
| NLL improvements misreported as accuracy gains | High | Accuracy primary; retain calibration and unchanged decisions | Rescue/harm identity and separate NLL/Brier/ECE |
| Old OOD types relabelled novel | High | Describe reserved configurations of known transform types | Explicit protocol boundary; no novelty-of-shift claim |
| Multiple candidates / favorable reruns | High | One fixed primary candidate, one campaign, zero revisions | Frozen protocol, terminal result regardless of sign |
| Five optimizer seeds mistaken for independent data | High | Five datasets, two seeds averaged per dataset | Cluster deltas; descriptive intervals only |
| Aggregate masks clean or familiar harm | High | Separate nonregression against reference and selected control | Six mandatory nonregression comparisons |
| Runtime/resource escape | High | Pre-import and periodic guards plus outer timeout | Local refusal retained; hosted admission pending |
| Placeholder empirical results | Critical integrity | Generate final result text only from verified summary | NOT_EVALUATED until source-bound run exists |
| Personal style missing | Resolved artifact issue | Recovered supplied style byte-for-byte | SHA256 08cbf7d94d1460e4002c29768c3d576f8c716e3ac173d3ead34e526d865fb59d |
| Kaggle editor unavailable | External blocker | Dedicated authenticated tab and bounded retry | Editor Loading persisted; no Kaggle result claimed |
| Source-only review confused with replication | High | Separate source review, hosted tests, replay, retraining | Independent agent review is read-only, not replication |
| Full-family capability inferred from small routing study | Critical claim issue | Preserve 0/6 promotion and all original outcomes | Scope excludes target-scale/full-system claims |

## Mathematical guarantees actually available

If reference r and alternative q are probability vectors and alpha is in [0,1],
p=(1-alpha)r+alpha*q is a probability vector, equals r at alpha=0, and equals q
at alpha=1. Its total variation from r is alpha*TV(q,r) <= alpha. This proves
bounded probability movement only. Arbitrarily small movement can flip a class
decision near a tie, so there is no derived accuracy or safety guarantee.

For each labeled example, partition transitions between reference/candidate
correctness into unchanged, rescue and harm. Summing indicators gives exact
accuracy difference (rescues-harms)/N. This is an audit identity, not a claim
that rescues will outnumber harms.

## Independent review disposition

The user-requested independent read-only review identified the frozen-gradient,
partial-checkpoint-replay and self-consistent-result-tampering defects above
before any scientific execution. Repairs are included in the successor. The
reviewer did not execute numerical tests or independently retrain the model.

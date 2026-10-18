# Evidence map

| Claim | Experiment | Raw artifact | Analysis | Paper location | Status |
|---|---|---|---|---|---|
| Risk composition cannot lower host risk | algebra plus invariant tests | `tests/test_olympus_system.py` | `research/OLYMPUS_PAPER_DRAFT.md` | Method, Eq. 7 | Supported for implemented formula |
| Forbidden transitions remain masked and high risk forces allowed STOP | model/runtime tests | `tests/test_olympus_system.py` | full test receipt | Method and verification | Software-tested at reference scale |
| Cosine improves smoke-test loss over constant rate | five paired exploratory seeds | corrected run evaluations and logs | `aggregate.json` | Engineering case study | Development-only |
| Validation-best restoration improves smoke-test loss | five paired exploratory seeds | corrected run evaluations and logs | `aggregate.json` | Engineering case study | Development-only |
| Larger tiny models lower loss in this matrix | three exploratory scales | corrected run evaluations | `aggregate.json` and efficiency plot | Scale ablation | Confounded exploratory evidence |
| Sequence packing helps | packing ablation | paired logs | exact hash equality | Failure analysis | Not evaluated because intervention was inactive |
| Tiny SFT solves held-out instructions | 24 runs | exploratory evaluations | exact match summaries | Results | Rejected: 0% exact match |
| Olympus fusion improves task quality or risk detection | none valid | no aligned dataset | no analysis | Limitations | Not evaluated |
| JEPA additions improve super-resolution | none valid | no rights-reviewed HR/LR corpus | no analysis | Limitations | Not evaluated |

The paper may cite only the first six rows as evidence, with their stated scope. The last
three rows are negative or unavailable evidence and must remain visible.

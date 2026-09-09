# Portfolio claim/source ledger

Audit date: 2026-09-07

This ledger separates repository evidence from external context. Local test
counts are observations from this audit environment. They are not scientific
validation unless the row says so.

| Claim | Source | What the source supports | Boundary |
| --- | --- | --- | --- |
| Olympus has executable code for six named roles | `docs/model-families.md`, `olympus/models/`, focused model tests | Typed contracts, deterministic gates, compact trainable components, and a fixed synthetic composition run | No pretrained family weights, scaling run, external benchmark, or promoted model |
| Olympus's built-in model lifecycle runs end to end | fresh `foundry verify-pipeline` run in this audit | Dataset registration, bigram training, held-out evaluation, checkpoint, export, registry, and status all executed | The character bigram is an infrastructure verifier, not an assistant |
| The Olympus wheel is operational | `scripts/validate_installed_wheel.py` against the newly built wheel | Fresh isolated install passed identity, Foundry, and composition checks | Distribution payload certification remains blocked because new sources are untracked |
| The six-role composition is real but narrow | `olympus/models/composition.py`, `evidence/model_composition_smoke_20260906.json` | One process-local, manifest-approved, declared non-material synthetic replay | No sandbox, durable state, crash recovery, or learned end-to-end graph |
| Causal Memory has real local-model evidence | `results/real/controlled_raw.csv`, `results/evidence_provenance_v1.json` | Complete 20,250-cell controlled factorial with three models/seeds/tasks and zero recorded provider/parse errors | Small local models and controlled tasks only |
| Causal Memory's naturalistic causal claim is falsified | `results/resource_bounded_v1/causal_inference.csv`, `paper/final_main.tex` | Irrelevant deletion changes 86.7--93.3% of exact outputs, invalidating exchangeability | Ten selected transformable questions; not a full benchmark estimate |
| LongMemEval is a 500-question long-term memory benchmark | [LongMemEval primary paper](https://arxiv.org/abs/2410.10813) | Benchmark purpose, scale, and decomposition of indexing/retrieval/reading | Does not validate this portfolio's selected subset or scoring |
| Broad novelty for causal memory intervention is unavailable | [CMI primary paper](https://arxiv.org/abs/2605.17641) | Prior causal intervention-based memory selection and Causal-LoCoMo evaluation | Novelty may remain in diagnostic decomposition or falsification protocol, subject to review |
| FNO and neural operators are established baselines | [FNO paper](https://arxiv.org/abs/2010.08895), [Neural Operator paper](https://arxiv.org/abs/2108.08481) | Established operator-learning formulation and PDE benchmarks | Does not validate APEN/FIM/GMF variants |
| Model and dataset documentation are established practices | [Model Cards](https://arxiv.org/abs/1810.03993), [Datasheets for Datasets](https://arxiv.org/abs/1803.09010) | Prior art for transparent model/dataset reporting | Olympus's evidence graph must claim stronger enforceable behavior, not novelty from cards alone |
| Data validation is an established production ML concern | [Data Validation for Machine Learning](https://proceedings.mlsys.org/paper_files/paper/2019/hash/928f1160e52192e3e0017fb63ab65391-Abstract.html) | Schema/anomaly validation in production ML pipelines | MIT-STAN-064 novelty must be framed around its specific evidence contracts and falsification gates |
| TMLR is a plausible venue for bounded negative/diagnostic work | [TMLR official OpenReview page](https://openreview.net/group?id=TMLR) | Rolling submission and emphasis on technical correctness over subjective significance | Venue fit is a recommendation, not acceptance evidence |
| ICLR 2027 is imminent and has strict requirements | [ICLR 2027 call](https://www.iclr.cc/Conferences/2027/CallForPapers), [author guide](https://iclr.cc/Conferences/2027/AuthorGuidelines) | Abstract deadline 2026-09-18, paper deadline 2026-09-25, nine-page main-text limit, double blindness | None of the model-family claims is ready for this deadline |
| ARR offers a later NLP route | [ARR official dates](https://aclrollingreview.org/dates) | October 2026 cycle submission date 2026-10-12 and later venue commitments | Causal Memory still needs independent scoring/control redesign before a strong empirical claim |


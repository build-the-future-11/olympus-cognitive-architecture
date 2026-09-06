# Portfolio Tier List, Venue Map, and Outreach Gate

Audit date: 2026-09-02  
Scope: 42 top-level directories under `GitHub-Every-Repo`  
Basis: retained experiments, truth/evidence ledgers, runnable code, tests, papers, external-data status, and deployment surface. A directory name or generated paper is not counted as evidence.

## Tier definitions

- **S — submission/outreach ready:** a narrow claim is backed by retained, reproducible evidence and a complete paper or artifact.
- **A — collaboration ready:** technically serious and worth showing to a specialist, but one named validation gate remains before archival submission.
- **B — credible prototype:** useful code or a promising controlled result; requires natural/external evaluation and stronger baselines.
- **C — scaffold:** runnable or well specified, but not yet a defensible research contribution.
- **D — support, duplicate, empty, or archive:** useful as infrastructure or provenance, not an independent paper.

Research and application tiers are separate. A strong product can be weak conference evidence, and a rigorous negative result can have little immediate product value.

## Complete top-level tier list

| Project | Research | Application | Defensible present value | Best venue path | Best external users |
|---|---:|---:|---|---|---|
| Assumption-Integrity | **S** | A | Mixed-result study of conditioning assumptions under shift; 167 registered runs, natural Adult benchmark, uncertainty baselines, formal identifiability witness, validated artifacts | **ICLR 2027 primary**; TMLR fallback; UAI/AISTATS later | MIT Madry Lab, Oxford OATML, Stanford CRFM, NIST AI measurement, safety/reliability teams |
| NGMT | **S** | B | Complete falsification study: 457 registered rows, a 165-cell natural-data matrix, 60 sensitivity cells and 36 mechanism-recovery cells support Student-t output likelihoods but reject a general benefit from the tested non-Gaussian latent-memory update | **TMLR primary**; AISTATS 2027 only if its eventual call welcomes the bounded negative/mechanism result | Amazon Science forecasting, Google DeepMind forecasting, Microsoft Research, Nixtla, energy/finance forecasting groups |
| Olympus / Pantheon | **A** | **A** | Canonical artifact auditing localized 160/160 seeded faults; public-data cross-implementation case study agrees on 10/10 splits | MLSys 2027 after real multi-agent validation; NeurIPS Evaluations & Datasets 2027; TMLR | NIST AI Consortium/CAISI, Stanford CRFM, Anthropic evaluation, METR, ML platform teams |
| Causal-Memory-Use | **A** | B | 245,640 controlled rows demonstrate that retrieval and downstream causal use can be separated; real-LLM claim remains open | TMLR or ICLR/NeurIPS 2027 after real-system study | Microsoft A4P, Letta/MemGPT, Anthropic agents, Stanford CRFM, Berkeley agent-evaluation groups |
| NPMS | **A** | B | 360,000 controlled evaluations plus a hardened LoCoMo/LongMemEval pipeline; external API-backed evidence is intentionally absent | TMLR/ACL/EMNLP/COLM after external run | Microsoft A4P, Letta, LangChain/LangMem, model-platform memory teams |
| MLInvention | B | C | Evidence-gated falsification foundry for 64 mechanism proxies; valuable meta-research artifact, not 64 validated inventions | NeurIPS Evaluations & Datasets 2027 or TMLR after independent reproduction | ML evaluation groups, research tooling teams, reproducibility educators |
| ML4Science | B | B | 64 executable workstreams and four pinned external proxies; no full proposed external method beats its selected strong baseline | NeurIPS Evaluations & Datasets 2027; ML4Science workshops; JOSS for tooling | Princeton AI for Accelerating Invention, national labs, scientific-ML institutes |
| MIT-Stanford-Princeton-Research | B | C | Transparent 64-proposal funnel with bounded survivors and natural-shift lane; explicitly independent and unaffiliated | TMLR/NeurIPS E&D only after promoted projects receive external replication | Research-methodology and reproducibility groups; education programs |
| ML4Industry | B | **A** | Reproducible 64-workstream industrial-ML foundry with evidence boundaries; strongest value is applied benchmarking | IAAI/MLSys/KDD Applied Data Science after real partner data | Siemens, Bosch, Schneider Electric, GE Aerospace, NVIDIA manufacturing teams |
| ML4SemanticIntelligence | B | B | Controlled hypothesis foundry for semantic reasoning; first tranche is synthetic diagnostics rather than general-language evidence | ACL/EMNLP/COLM after public benchmark and frontier-model runs | Allen Institute for AI, Cohere For AI, Microsoft Research, multilingual/NLP labs |
| LAM-JEPA | B | B | Substantial educational-reasoning pipeline with calibration, OOD, student-state and intervention components; evidence appears predominantly generated/synthetic | AIED, EDM, LAK; EAAI next cycle after authentic learner data | Duolingo Research, Khan Academy, CMU HCII, Stanford Accelerator for Learning |
| Eigen-JEPA | B | B | Spectral financial world-model package with benchmarks and paper assets; needs leakage-safe walk-forward external replication | ACM ICAIF, KDD ADS, TMLR after external replication | Oxford-Man Institute, Bloomberg, Two Sigma, Citadel academic programs |
| FI-JEPA | B | B | Runnable financial JEPA with ablations and benchmark harness; current short benchmark settings are not paper-grade evidence | ACM ICAIF/TMLR after full multi-seed public-data study | Bloomberg, Man Group/Oxford-Man, quantitative research groups |
| FIM | **S** | B | Complete bounded falsification study: a source-bound 24-cell, two-task, three-seed component matrix finds no stable benefit from memory, retrieval, or salience; the retained compact suite also preserves a DeepONet baseline win | **TMLR primary** as a mechanism-audit/negative-results paper; no architecture-superiority claim | Microsoft A4P, long-context research labs, scientific forecasting teams |
| Fabric-Induced-Memory | B | B | Alternate FIM tree; current ablations do not establish memory/retrieval value | **Merge with FIM**, then use the FIM venue path | Same as FIM |
| IRIS-Draft | B | C | Reproduced negative/inconclusive robust-memory results with good provenance; blocked on canonical raw trajectory provenance | TMLR negative-results path after provenance closure | Robust sequence-model and reproducibility researchers |
| Saphir-Whoof | B | B | Multilingual epistemic-invariance toolkit with activation capture and causal patching; toy evidence is not enough | ACL/EMNLP/COLM after real multilingual model matrix | Cohere For AI, AI2, Google multilingual research, University of Edinburgh NLP |
| APEN / Synthica | B | B | Broad experimental/statistical machinery, but its own truth file records incomplete multi-seed, OOD and baseline program | NeurIPS/ICML only after completing the frozen benchmark; otherwise TMLR | Scientific-ML, operator-learning and PDE-surrogate groups |
| Project-2424 | C | B | Excellent evidence-control architecture over 2,424 records; most child packages are bounded synthetic contract falsifiers, not scientific claims | MLSys demo/artifact or software paper after production use | Research operations teams, accelerators, institutional labs |
| PercyxLyla | C | **A** | Local-first execution/evidence control plane with 64 bounded workstreams; product engineering is ahead of real-world validation | MLSys after production workload study; OSS/product outreach now | Agent-platform teams, regulated R&D, local-first developer tooling groups |
| NeuroCAD | C | **A** | Public-alpha natural-language CAD and OpenSCAD toolkit with strong software test surface | SIGGRAPH/CHI/UIST or ASME IDETC after user study and geometric benchmark | Autodesk AI Lab, Siemens NX, Dassault Systèmes, PTC, NVIDIA Omniverse |
| ColorWorld / DaVinci-JEPA | C | **A** | Useful cinematic grading prototype with LUT export and video pipeline; no retained paper-grade study | ACM Multimedia/SIGGRAPH/CVPR workshop after perceptual user study | Adobe Research, Blackmagic Design, Dolby, Netflix, Foundry |
| BU1LDLanding | D | **A** | Evidence-aware member and institution platform; not a research paper | Product launch/case study, not a main research venue | Student research organizations, universities, incubators |
| FinanceMeta-Landing | D | **A** | Finance/economics member portal with deployment prerequisites | Product launch; education/fintech partnership | Universities, student finance societies, fintech education programs |
| Adaptive-Theory-Geometry | C | C | Manuscript plus reproducible synthetic figures; synthetic benchmark alone cannot carry the claim | AISTATS/COLT only after theorem strengthening and natural tasks | Theory/representation-learning labs |
| CDW | C | B | Runnable counterfactual PDE defect-world prototype | ML4Science workshop/NeurIPS workshop after strong PDE baselines | National labs, digital-twin and reliability groups |
| DRPT | C | B | Runnable phase-transition representation prototype | AISTATS/NeurIPS workshop after scale and baseline study | Adaptive-systems and representation-learning labs |
| EPU | C | B | Hopfield-inspired scaffold with Python, Verilog and blueprint artifacts; no hardware evidence | ASPLOS/MICRO/ISCA only after synthesis, FPGA results and comparisons | Neuromorphic hardware groups, Intel Labs, IBM Research |
| GaussianMemory | C | C | Explicitly an experimental scaffold with illustrative manuscript plots | TMLR/ICLR only after generated multi-seed evidence replaces illustrations | Memory and neural-field groups |
| QFIM | D | C | Appears to be another FIM variant without an independent evidence identity | Merge/archive into canonical FIM | None until merged |
| RIPII | C | C | Runnable coarse-graining and hierarchical motif prototype with smoke benchmarks | NeurIPS/AISTATS workshop after real datasets and strong baselines | Representation-learning and complex-systems labs |
| Residual-Event-Tokenization | C | B | Runnable sparse residual-event tokenization prototype | ICML/NeurIPS workshop after scaling and tokenizer baselines | Edge video, event-camera and efficient-model teams |
| SCDMIT | C | C | Coupled flow/geometry concept with code/tests but no retained research-result package | AISTATS/COLT/NeurIPS workshop after proofs and experiments | Dynamical-systems and geometric-ML labs |
| Research-Pilot / Research Muse | D | B | Generic student research-workflow web app; template metadata remains | Product/user-study route, not a research paper yet | Libraries, schools, research-skills programs |
| RIS | D | C | Minimal adaptive IRR-stabilization surface with insufficient evidence | No venue until a real repository and study exist | Finance education only after implementation |
| VertexED | D | B | Durable control snapshots for Percy; provenance/infrastructure rather than an independent project | Merge into PercyxLyla systems paper | Agent infrastructure users |
| Finance-Meta-Research-LGWM-Hedge-Fund-* | D | D | Mirror/archive with no independent visible evidence package | Archive or reconnect to canonical source | None as a standalone project |
| Finance-Meta-Research-org-infra-* | D | D | Organization-infrastructure mirror, not a paper | Archive/infra only | None as a standalone project |
| EigenFinance | D | D | No substantive top-level research artifact detected | Populate or archive | None yet |
| IY-ERN | D | D | No substantive top-level research artifact detected | Populate or archive | None yet |
| Speechly | D | D | No substantive top-level research artifact detected | Populate or archive | None yet |
| SourceZips | D | D | Source/provenance bundle, not a project | Retain as archive with checksums | Internal provenance only |

NGMT reached S only after the frozen matrix, source-bound provenance, statistical rebuild, 30-test suite, artifact validator, publication gate, package checksum/CRC validation, and page-by-page PDF inspection all passed. Its S tier denotes a complete and reproducible narrow negative result, not evidence for the originally proposed memory mechanism.

FIM reached S only after its frozen 24-cell matrix completed on one backend and protocol, 37 tests passed, manifest and byte-hash validation passed, exact paired statistics were rebuilt, the seven-page paper and checksum sidecar were inspected, and the 518-file submission package passed CRC validation. Its S tier is restricted to the component-necessity falsification claim on two generated tasks; it is not evidence that the architecture is generally competitive.

## Conference strategy as of 2026-09-02

| Venue | Live status | Portfolio use |
|---|---|---|
| **ICLR 2027** | Abstract **2026-09-18**, paper **2026-09-25**, both 11:59 PM AoE; double blind; 9 main-text pages; required AI-use statement | Submit **Assumption-Integrity only**. If no coauthor is an eligible reciprocal reviewer, ICLR caps each author at one paper. Do not spend that slot on an unfinished portfolio item. |
| **IAAI 2027** | Paper deadline **2026-09-08** | None currently meets the real-world deployment bar. Do not rush a synthetic prototype into an applications venue. |
| **EAAI 2027** | Abstract deadline **2026-09-01** has passed; paper **2026-09-08** | LAM-JEPA should target the next cycle after authentic learner data, not this cycle. |
| **MLSys 2027** | Opens **2026-10-10**; paper deadline **2026-10-30 20:00 UTC** | Pantheon or PercyxLyla only if a real multi-agent/production workload study is completed in time. |
| **TMLR** | Rolling, year-round; correctness-focused; generally aims near nine weeks | Best fallback for Assumption-Integrity and the primary home for the completed NGMT and FIM falsification studies; IRIS remains eligible only after its provenance gate closes. |
| **NeurIPS 2026** | Main/E&D deadline passed on 2026-05-06 | Do not target the closed main cycle. Prepare Pantheon/MLInvention/ML4Science for NeurIPS Evaluations & Datasets 2027. |
| **AAAI 2027 main** | Deadline passed on 2026-07-28 | Closed. Use ICLR/TMLR/MLSys paths instead. |
| **ICML/AISTATS/UAI/KDD/ACL/EMNLP/COLM 2027** | Use only after each official 2027 call is posted and checked | These are next-cycle targets, not claims of an open deadline. |

Official sources:

- ICLR 2027 author guide: https://iclr.cc/Conferences/2027/AuthorGuidelines
- MLSys 2027 CFP: https://mlsys.org/Conferences/2027/CallForResearchPapers
- TMLR: https://jmlr.org/tmlr/
- IAAI 2027: https://aaai.org/conference/aaai/aaai-27/iaai-27-call/
- EAAI 2027: https://aaai.org/conference/aaai/aaai-27/eaai-27-call/
- NeurIPS 2026 dates: https://neurips.cc/Conferences/2026/Dates

## Outreach decision

### Start now: evidence-bearing collaboration outreach

1. **Assumption-Integrity** — send a short feedback/collaboration note to MIT Madry Lab and Oxford OATML after sender identity is confirmed. Attach the anonymous preprint and one-page evidence brief, not the whole portfolio.
2. **Olympus/Pantheon** — approach NIST's AI Consortium through its formal interest process; separately pitch Stanford CRFM on a real-agent canonical-auditing replication.
3. **Causal-Memory-Use + NPMS** — approach Microsoft Research A4P and one open memory-system team for API/data collaboration. Lead with the controlled falsifier and explicitly state that real-system evidence is missing.
4. **NeuroCAD** — approach Autodesk AI Lab only after producing a 20–50 prompt geometric correctness benchmark and a two-minute demo. The software is outreach-worthy; the research claim is not yet.
5. **NGMT** — request technical feedback from one probabilistic-forecasting group using the completed anonymous preprint and reproducibility package. Lead with the five-dataset Student-t likelihood result and the falsification of the latent-memory claim; do not pitch the rejected mechanism as an innovation.
6. **FIM** — request methods feedback from one memory-systems group using the completed anonymous paper and checksum-verified package. Lead with the preregistered component-necessity audit and its null/unstable effects; do not imply a general architecture comparison.

Named first contacts should be selected for topic fit, not prestige: Oxford OATML exposes a group contact at `oatml@cs.ox.ac.uk`; NIST exposes the formal Consortium contact at `aiconsortium@nist.gov`; Stanford CRFM is directed by Percy Liang; Microsoft A4P publishes its current team roster; and Autodesk lists Daniele Grandi among researchers working at the AI/ML–mechanical-engineering interface. Individual cold email should use only a recipient's public institutional channel.

### Hold until gate closes

- **LAM-JEPA:** obtain authentic or checksum-pinned public learner data and a strong knowledge-tracing baseline before Duolingo/CMU outreach.
- **ColorWorld:** add a blinded perceptual evaluation and compare against commercial/open grading baselines before Adobe/Blackmagic outreach.
- **All 64-project foundries:** pitch one promoted survivor or the evaluation methodology, never “64 breakthroughs.”

### Do not outreach as standalone projects

Empty directories, mirrors, source archives, landing pages, VertexED state, and duplicate FIM variants should not generate cold outreach. Merge or archive them first.

## Current institution/company fit

| Target | Why the fit is real | Project package |
|---|---|---|
| MIT Madry Lab | Explicit focus on reliable real-world ML and distribution shift | Assumption-Integrity |
| Oxford OATML | Reliability, uncertainty, OOD generalization and calibration | Assumption-Integrity; NGMT falsification study |
| Stanford CRFM | Foundation-model evaluation, systems, transparency and new technical paradigms | Pantheon; Causal-Memory-Use |
| NIST AI Consortium / CAISI | Open participation around AI measurement, standards, test systems and evaluations | Pantheon; Assumption-Integrity methodology |
| Microsoft Research A4P | Public program explicitly includes procedural memory, context management and reliable long-horizon agents | Causal-Memory-Use; NPMS; PercyxLyla |
| Anthropic evaluation/alignment research | Behavioral evaluation, agent autonomy, trustworthy agents and auditing | Pantheon; Causal-Memory-Use |
| Autodesk AI Lab | CAD-geometry generative AI and evaluation | NeuroCAD |
| Adobe Research | Image/video editing research with industry transfer | ColorWorld |
| Duolingo Research | ML, learning science, assessment and large-scale learner data | LAM-JEPA after data gate |
| Amazon/Google/Microsoft forecasting groups | Probabilistic forecasting and operational time series | Completed NGMT evidence package |

Relevant public pages:

- MIT Madry Lab: https://madrylab.mit.edu/
- Oxford OATML publications: https://www.cs.ox.ac.uk/oatml/publications.html
- Stanford CRFM: https://crfm.stanford.edu/
- NIST AI Consortium: https://www.nist.gov/artificial-intelligence/nist-ai-consortium
- Microsoft Research A4P: https://www.microsoft.com/en-us/research/group/agents-for-productivity-a4p/
- Anthropic Research: https://www.anthropic.com/research
- Autodesk AI Lab: https://www.research.autodesk.com/research-areas/science/ai-lab/
- Adobe Research careers/collaboration surface: https://research.adobe.com/careers/
- Duolingo Research: https://research.duolingo.com/

## Outreach packet gate

No message is sent until all boxes for that project are true:

- one-sentence claim matches the truth ledger;
- retained raw results and validation command are available;
- anonymous paper or two-page technical brief exists;
- recipient-specific reason and requested action are explicit;
- sender name, affiliation/status, reply email, public links and coauthor approval are confirmed;
- ICLR anonymity and dual-submission constraints are not violated.

The first four items are prepared or nearly prepared for the priority projects. The final identity/contact item requires the portfolio owner's confirmation before any external message can truthfully be sent.

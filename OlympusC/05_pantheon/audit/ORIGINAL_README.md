# Pantheon
### Cross-Agent Replication and Disagreement Resolution for Research Claims

> **Research status boundary.** Pantheon is currently conditional. The tracker’s critical next step is one genuine end-to-end reproduction of a research package. “Two agents looked at the same repo” is not independent replication. Independence must be designed and audited.

---

## 1. Thesis

AI research agents can generate hypotheses, code, experiments, and interpretations rapidly. This creates a new reliability problem:

> If one agent reports a scientific result, how do we determine whether another independently operating agent can reproduce it, identify why it disagrees, and decide which claims survive?

Pantheon is a protocol and system for **cross-agent replication**, **artifact-grounded disagreement diagnosis**, and **claim-level resolution**.

---

## 2. Unit of analysis: a research claim

Represent claim \(c\) as:

\[
c
=
(H,P,D,M,E,K),
\]

where:

- \(H\): hypothesis / natural-language claim;
- \(P\): experimental protocol;
- \(D\): dataset specification + hash;
- \(M\): metric definition;
- \(E\): expected result/effect;
- \(K\): code/environment provenance.

A claim cannot be replicated from prose alone.

---

## 3. Research package

A Pantheon package should contain:

```text
claim.yaml
protocol.md
environment.lock
data_manifest.json
git_commit.txt
configs/
scripts/
expected_schema.json
stopping_rule.md
claim_boundary.md
```

Crucially, it does **not** need to reveal the original numerical result to the replicator during a blinded run.

---

## 4. Agent roles

### Proposer \(A_P\)

Creates or submits the original claim package.

### Replicator \(A_R\)

Receives only the frozen package and independently executes the protocol.

### Auditor \(A_A\)

Checks:
- environment;
- hashes;
- code changes;
- protocol deviations;
- raw output integrity.

### Adjudicator \(A_J\)

Runs only after independent results are frozen.

Its role is not to invent a compromise; it classifies disagreement and proposes discriminating tests.

The simplest initial system can use two agents plus deterministic auditing code.

---

## 5. Independence requirement

Replication is genuine only if hidden state is isolated.

At minimum:

- separate process/container;
- separate conversation/context;
- no access to proposer chain-of-thought;
- no shared scratchpad;
- fresh checkout;
- independent seed set;
- independent environment creation;
- frozen artifact package;
- all communication logged through explicit files.

Define an independence vector:

\[
I=(i_c,i_s,i_e,i_r,i_m),
\]

for context, seed, environment, runtime, and model independence.

A replication run should report this vector rather than using a vague “independent” label.

---

## 6. Replication modes

### R0 — exact rerun

Same code/config/data, fresh environment.

Tests reproducibility.

### R1 — independent environment reconstruction

Same code, environment rebuilt from specification.

Tests environment completeness.

### R2 — independent reimplementation

Replicator receives method/protocol but not implementation.

Tests method-level reproducibility.

### R3 — cross-model replication

Different agent/model executes the same protocol.

Tests agent-specific dependence.

### R4 — conceptual replication

Different implementation or dataset targeting the same scientific effect.

This is much stronger and should not be conflated with exact rerun.

---

## 7. Claim-level numerical comparison

Suppose proposer obtains:

\[
\hat\delta_P
\]

with uncertainty \(s_P\), and replicator obtains:

\[
\hat\delta_R
\]

with \(s_R\).

Define standardized disagreement:

\[
D_z
=
\frac{
|\hat\delta_P-\hat\delta_R|
}{
\sqrt{s_P^2+s_R^2+\epsilon}
}.
\]

Also define sign agreement:

\[
A_{\text{sign}}
=
\mathbb 1[
\operatorname{sign}(\hat\delta_P)
=
\operatorname{sign}(\hat\delta_R)
].
\]

Relative effect error:

\[
D_{\text{rel}}
=
\frac{
|\hat\delta_P-\hat\delta_R|
}{
|\hat\delta_P|+\epsilon
}.
\]

Confidence-interval overlap:

\[
O
=
\frac{
|CI_P\cap CI_R|
}{
|CI_P\cup CI_R|+\epsilon
}.
\]

No single metric determines replication. Pantheon should report a vector.

---

## 8. Replication status taxonomy

Recommended states:

- **REPRODUCED**
- **PARTIALLY_REPRODUCED**
- **DIRECTION_ONLY**
- **NOT_REPRODUCED**
- **PROTOCOL_AMBIGUOUS**
- **ENVIRONMENT_FAILURE**
- **DATA_MISMATCH**
- **IMPLEMENTATION_MISMATCH**
- **METRIC_MISMATCH**
- **UNDERPOWERED**
- **CLAIM_OVERSTATED**
- **UNRESOLVED**

This prevents the binary “worked / failed” framing.

---

## 9. Disagreement taxonomy

Pantheon should localize disagreement into categories.

### D1 — artifact mismatch
Different file/data/config hash.

### D2 — environment mismatch
Dependency/hardware/runtime difference.

### D3 — implementation mismatch
Code does not implement the same method.

### D4 — protocol ambiguity
Original instructions permit multiple reasonable implementations.

### D5 — metric mismatch
Same experiment, different metric computation.

### D6 — statistical disagreement
Same method/data but stochastic outcomes differ.

### D7 — interpretation disagreement
Numbers agree; scientific conclusion differs.

### D8 — external-validity disagreement
Exact replication succeeds but conceptual replication fails.

### D9 — hidden-state / agent contamination
Replicator was not independent.

---

## 10. Disagreement localization

Create artifact delta vector:

\[
\Delta
=
(
\Delta_D,
\Delta_E,
\Delta_C,
\Delta_P,
\Delta_M,
\Delta_S
)
\]

for data, environment, code, protocol, metric, and stochastic seed differences.

The adjudication agent receives:

- frozen proposer result;
- frozen replicator result;
- artifact hashes;
- execution logs;
- delta vector.

It must produce:

1. disagreement class;
2. supporting evidence;
3. minimal discriminating experiment;
4. revised claim boundary.

---

## 11. Minimal discriminating experiment

Suppose disagreement may be due to either code or seed.

Do not rerun everything.

Construct test:

\[
T^\star
=
\arg\max_T
\frac{
\operatorname{ExpectedInformationGain}(T)
}{
\operatorname{Cost}(T)
}.
\]

In practice, the first version uses rules:

- same code + swapped seeds;
- same seed + swapped environment;
- exact proposer environment;
- replicator reimplementation with proposer data;
- metric recomputation from same raw predictions.

This “minimal disagreement resolver” can be a strong systems contribution.

---

## 12. Replication graph

For claims \(c_1,\ldots,c_n\) and agents \(a_1,\ldots,a_m\), create a bipartite/typed graph.

Edge:

\[
e_{ij}
=
(a_i,c_j,r_{ij})
\]

where \(r_{ij}\) is a replication result vector.

Claim survival score can be descriptive:

\[
S_c
=
\sum_i
w_i
g(r_{ic}),
\]

where \(w_i\) reflects independence level.

Avoid presenting this as a universal scientific truth score.

---

## 13. False consensus

A key risk in multi-agent science is correlated agreement.

Two agents may agree because they:

- share model weights;
- share prompt priors;
- share code;
- share data bug;
- share hidden context.

Define empirical false-consensus rate on seeded flawed packages:

\[
FCR
=
\Pr(
\text{agents agree with same incorrect claim}
).
\]

Pantheon should explicitly benchmark false consensus.

---

## 14. Benchmark suite

### Benchmark A — exact reproducibility packages

Small research packages with deterministic or near-deterministic expected outputs.

Goal: test infrastructure.

### Benchmark B — seeded faults

Inject controlled faults:

- wrong dataset split;
- wrong metric sign;
- seed dependence;
- leakage;
- dependency version mismatch;
- unit conversion bug;
- stale cache;
- missing preprocessing;
- cherry-picked run.

Ground-truth fault class is known.

Measure disagreement localization accuracy.

### Benchmark C — stochastic claims

Experiments whose effect varies across seeds.

Tests whether Pantheon labels “statistical disagreement” rather than hallucinating implementation failure.

### Benchmark D — one real Research Atlas package

The tracker’s required milestone:

> reproduce one existing package end-to-end from frozen artifacts.

No claim of general validation until this is done.

---

## 15. Primary metrics

### Exact reproduction rate

\[
RR_{\text{exact}}
=
\frac{
\# \text{packages meeting exact tolerance}
}{
N
}.
\]

### Direction replication rate

\[
RR_{\text{sign}}
=
\frac1N\sum_iA_{\text{sign},i}.
\]

### Disagreement localization accuracy

\[
DLA
=
\Pr(
\hat d=d^\star
)
\]

on injected-fault packages.

Also top-k accuracy if multiple causes exist.

### Resolution efficiency

\[
RE
=
\frac{
\text{resolved disagreements}
}{
\text{additional compute cost}
}.
\]

### False consensus rate

\[
FCR.
\]

### Claim survival rate
Descriptive fraction of claims remaining within their original confidence boundary after independent replication.

---

## 16. Baselines

1. **Single-agent self-check**
2. **Same-agent second run**
3. **Two agents sharing full context**
4. **Two isolated agents**
5. **Artifact diff only**
6. **Human-authored deterministic checker**, where possible
7. **No adjudicator**
8. **Rule-based adjudicator**
9. **LLM adjudicator grounded in artifact diffs**

The critical comparison is:

> Does genuine isolation plus structured artifact exchange reduce false consensus and improve fault localization compared with ordinary “ask another agent to review it”?

---

## 17. Ablations

| Ablation | Question |
|---|---|
| shared context | how much “replication” is actually copying? |
| isolated context | independence effect |
| same model vs different model | model correlation |
| same code vs reimplementation | reproducibility depth |
| full artifacts vs prose only | artifact completeness |
| known original result vs blinded | anchoring bias |
| no artifact hashes | provenance value |
| no adjudicator | resolution value |
| rule adjudicator vs model adjudicator | where intelligence is needed |
| no seeded fault labels | real-world behavior |
| one vs multiple replicators | marginal value of extra agents |

---

## 18. Blinding protocol

For a stronger experiment, replicator should initially receive:

- method;
- data manifest;
- configs;
- stopping rule;
- metric definition;

but **not**:

- original scalar result;
- original interpretation;
- original plots.

After result freeze, unblind and compare.

This tests anchoring.

---

## 19. Artifact provenance

Every run:

```json
{
  "claim_id": "...",
  "agent_id": "...",
  "agent_model": "...",
  "container_digest": "...",
  "git_commit": "...",
  "data_hash": "...",
  "config_hash": "...",
  "seed": 0,
  "start_time": "...",
  "end_time": "...",
  "raw_outputs": "...",
  "metrics_file": "...",
  "independence_vector": {}
}
```

A result without provenance is not a replication result.

---

## 20. Security / contamination checks

### Shared-state probe
Plant a canary string in proposer-only context.

The replicator should never output it.

### File visibility audit
List exactly what files replicator can access.

### Environment reset
Fresh temp/home/cache directories.

### Network policy
Optionally disable network during exact reproduction.

### Secret/data boundary
Do not package credentials or private unrelated data.

---

## 21. Statistical protocol

For continuous effects, report proposer and replicator estimates with uncertainty.

For multiple seeded packages:

- paired effect difference;
- CI overlap;
- sign agreement;
- heterogeneity.

A random-effects summary can be used:

\[
\hat\delta_i
\sim
\mathcal N(\delta,\sigma_i^2+\tau^2)
\]

where \(\tau^2\) models between-replication heterogeneity.

This is optional for the first version.

---

## 22. Failure modes

Pantheon fails scientifically if:

- “independent” agents share hidden state;
- only exact reruns are tested but claims say conceptual replication;
- the adjudicator invents explanations without artifact evidence;
- false consensus is never measured;
- injected-fault taxonomy is trivial;
- reproducer is given original outputs and simply matches them;
- a successful rerun is mislabeled external validation.

---

## 23. Expected strongest contribution

A realistic strong claim would be:

> “A structured, isolated cross-agent replication protocol with artifact-level disagreement localization detects reproducibility failures that ordinary same-context multi-agent review misses.”

That requires the baseline comparison and seeded-fault benchmark.

---

## 24. Target repository structure

```text
pantheon/
├── README.md
├── schemas/
│   ├── claim.schema.json
│   ├── run.schema.json
│   └── disagreement.schema.json
├── configs/
│   ├── smoke.yaml
│   ├── seeded_faults.yaml
│   └── real_package.yaml
├── src/pantheon/
│   ├── package.py
│   ├── isolation.py
│   ├── runner.py
│   ├── provenance.py
│   ├── comparator.py
│   ├── disagreement.py
│   ├── adjudication.py
│   └── metrics.py
├── scripts/
│   ├── make_demo_package.py
│   ├── run_proposer.py
│   ├── run_replicator.py
│   ├── compare_results.py
│   ├── run_seeded_faults.py
│   ├── reproduce_real_package.py
│   └── build_paper_assets.py
├── tests/
├── packages/
├── runs/
└── paper/
```

---

## 25. Boot-up contract

```bash
cd pantheon

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt

pytest -q

# Create a tiny deterministic research package
python scripts/make_demo_package.py \
  --out packages/demo/

# Run proposer in isolated workspace
python scripts/run_proposer.py \
  --package packages/demo/ \
  --run-dir runs/demo/proposer/

# Run genuinely separate replicator
python scripts/run_replicator.py \
  --package packages/demo/ \
  --run-dir runs/demo/replicator/ \
  --fresh-workspace

# Compare frozen outputs
python scripts/compare_results.py \
  --proposer runs/demo/proposer/ \
  --replicator runs/demo/replicator/ \
  --out runs/demo/comparison.json

# Seeded disagreement benchmark
python scripts/run_seeded_faults.py \
  --config configs/seeded_faults.yaml

# Required real-package milestone
python scripts/reproduce_real_package.py \
  --config configs/real_package.yaml

# Generate paper tables/plots
python scripts/build_paper_assets.py \
  --runs runs/ \
  --out paper/generated/
```

---

## 26. Minimal first experiment

To make Pantheon real quickly:

1. Build one deterministic package.
2. Introduce five fault classes.
3. Run isolated proposer/replicator.
4. Verify canary-based independence.
5. Measure disagreement localization.
6. Compare with shared-context “second reviewer.”
7. Then reproduce one real Research Atlas package.

That is enough to determine whether the core idea has legs.

---

## 27. Definition of done

Pantheon is paper-ready when:

- independence is auditable;
- one genuine real package is reproduced end-to-end;
- seeded-fault benchmark exists;
- at least one false-consensus test exists;
- disagreement taxonomy maps to artifacts;
- exact rerun is distinguished from reimplementation;
- original results can be blinded;
- all agent communication is explicit;
- all results are frozen before adjudication;
- every claim status is evidence-backed.

The project should measure **replication quality**, not merely generate more agent conversations.

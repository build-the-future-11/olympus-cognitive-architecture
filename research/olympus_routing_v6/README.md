# Olympus routing v6: completed negative study

One frozen, bounded synthetic study is complete. The model and experiment system are fully implemented for this protocol. The broader Olympus project is not finished or promoted: 0/6 families remain ready. There is no empirical proof of a perfect model.

## Result and explanation

Adaptive robust routing scored 99.6191% versus 99.6973% for the development-selected control on composite shift. The paired difference is -0.078125 percentage points, with a descriptive 95% cluster-bootstrap interval [-0.41015625, +0.1171875]. The registered +2-point improvement claim failed. All six nonregression criteria passed.

The familiar-corruption gain over the frozen reference was +0.605469 points, but the stronger continued-training controls matched it. Under composite shift, the candidate rescued 9 reference errors and introduced 22 errors. This establishes the observed failure; it does not uniquely identify its cause. NLL improved versus reference while accuracy declined. The gate is not a reliable detector of beneficial corrections.

The realized control ceiling left only 0.302734 points of possible gain, below the registered two-point target. That limits this benchmark's discriminatory value. The original threshold and negative outcome are preserved. No protected subset was selected and no favorable rerun was performed.

## Completion checklist

| Deliverable | Current disposition |
|---|---|
| Question, prior-art boundary, falsifier, scope and fixed budget | Frozen in protocol v6.0.2 before scientific execution |
| Actual model, training, inference, controls, ablations and serialization | Implemented; no pseudocode in the executed path |
| Artificial qualification | 14 tests passed at the scientific source revision |
| Scientific execution | One campaign; 5 datasets x 2 initializations; 20 bank fits; 40 protected reports |
| Execution platform | Free GitHub CPU; Kaggle editor remained stuck loading, so no Kaggle result is claimed |
| Raw evidence | Full ZIP retained with externally verified digest; checkpoints and prediction tensors included |
| Replay and analysis | Exact checkpoint/prediction replay; independent recomputation of all 8 bootstrap comparisons |
| Independent review | Source, mathematics, provenance and arithmetic reviewed; not independent retraining |
| Full software release gate | Passed on d0cae48: 665 Python tests, 3 skips, 85.05% coverage, 6 web tests, lint, types, audits and builds |
| Paper | Complete manuscript, generated tables/figure, supplied arxiv.sty retained byte-for-byte |
| GitHub | PR31, stacked on the preserved predecessor research branch |
| arXiv | Source package prepared; not submitted or venue-approved |
| Broader claims | No real-world generalization, scaling, full-family readiness, or model promotion established |

## Evidence map

| Claim | Raw/recorded artifact | Analysis and paper |
|---|---|---|
| Primary failure and six nonregression passes | retained-campaign/routing-v6-campaign/protected/; SUMMARY.json | scripts/render_routing_v6_results.py; generated/contrasts.tex and contrast-plot.tex |
| Arm accuracy and ablations | Same 40 protected reports and raw .pt files in ZIP | generated/accuracy.tex |
| 9 rescues, 22 harms; familiar gains | Raw predictions plus report intervention diagnostics | generated/interventions.tex; results.tex |
| Timing and resource admission | campaign.log; retained campaign events | Results efficiency section; only initial RSS, not full campaign peak memory |
| Provenance and replay | CONTRACT.json, SCIENTIFIC_FREEZE.json, VERIFIED.json | EVIDENCE_RECEIPT.json; INDEPENDENT_REVIEW.json |
| Preserved earlier studies | PREDECESSOR_HASHES.json and original dated ledgers | Paper chronology; endpoints differ, so this is not a common learning curve |

Campaign: https://github.com/build-the-future-11/olympus-cognitive-architecture/actions/runs/39552083459

Validated code release: https://github.com/build-the-future-11/olympus-cognitive-architecture/actions/runs/39553045858

PR: https://github.com/build-the-future-11/olympus-cognitive-architecture/pull/31

## Reproduce analysis without retraining

From the repository root, restore raw files if needed with Python's standard-library ZIP command, then regenerate verified tables and the plot source:

```sh
python3 -m zipfile -e research/olympus_routing_v6/campaign-evidence.zip /tmp/olympus-v6-evidence
python3 -m scripts.render_routing_v6_results \
  --evidence /tmp/olympus-v6-evidence/routing-v6-campaign \
  --output /tmp/olympus-v6-tables
tectonic research/olympus_routing_v6/paper/manuscript/main.tex
```

The archive SHA256 is `77919e4f02d226ce804984f12c9e81359a27b33d441536a1610a91d305ebb6e5` (13,000,239 bytes). Raw .pt files are inside the ZIP because the repository ignores standalone tensor artifacts. Do not treat locally edited reports as original evidence.

Independent experimental reproduction must use scientific commit `18a70523d2039803b9aeb2b8dcad2c8e27026589` and its lockfile, adequate resources, fresh outputs, and separately recorded admission. Follow the paper's public qualify/run/verify commands. Later release commits patch Vite, certifi and Pillow; they are not the environment that produced the scientific result. Do not directly invoke internal workers to evade admission. The Kaggle handoff notebook defaults to artificial qualification, with scientific reproduction disabled.

## Next scientific direction, not an executed successor

1. Establish corrective opportunity using development-only expert errors: can another expert actually rescue the reference, and can available features predict rescue versus harm?
2. Calibrate development task difficulty so the strongest baseline leaves room for the practical effect being claimed. Freeze new data and a fresh protected set.
3. Test a materially different rescue-versus-harm intervention rule against continued training, constant mixing, and the frozen reference with matched budgets.
4. Admit a successor only with a new protocol, falsifiers and fixed budget. Require positive net rescues and the registered primary gain on untouched confirmation data. No empirical improvement is guaranteed.

See FLAW_AUDIT.md for flaw -> cause -> severity -> repair -> required evidence. Scaling, broad sensitivity, real-world evidence, full-run peak-memory/FLOP profiling, and independent retraining remain limitations, not completed deliverables. Existing proprietary licensing applies; public viewing does not confer unrestricted reuse. No external submission or license change was made.

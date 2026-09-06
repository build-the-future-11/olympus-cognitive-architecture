# Olympus Architecture Paper

This directory contains an architecture and executable-reference paper, not an
empirical model paper. The reference components under `olympus/models/` do not
constitute trained or evaluated family models. This paper must continue to say
that no Hermes, Prometheus, Perseus, Olympus-Atlas, Kronos, or Aion checkpoint
has been promoted until immutable evidence changes the repository truth map.
Target architecture prose must remain explicitly distinguishable from the
bounded, mostly process-local reference APIs that exist today.

## Contents

- `olympus_model_architecture.tex`: paper source.
- `references.bib`: primary arXiv references.
- `report-source.md`: canonical literature synthesis and frozen decisions.
- `CLAIM_SOURCE_LEDGER.md`: claim-to-evidence audit.
- `../../output/pdf/olympus_model_architecture.pdf`: compiled artifact.

## Rebuild

From this directory:

```bash
tectonic --keep-logs --outdir ../../output/pdf olympus_model_architecture.tex
```

The build is acceptable only if the log has no undefined citations or
references, all pages render through an independent PDF rasterizer such as
Poppler or PyMuPDF, and visual inspection finds no overlap, cutoff, or blank
page. Any future numerical result also requires an
immutable run manifest, source/data/checkpoint hashes, and per-example outputs.

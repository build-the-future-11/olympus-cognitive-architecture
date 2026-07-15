# Remaining External Actions

These are actions LabOS cannot honestly complete without external data, credentials, legal/security review, or permission changes.

## Project Genesis: Economics Edition

- Status: `Complete and smoke-tested`
- Why: Further external input is required.
- Path: `/Users/ryan/Documents/GenesisE`
- Data actions: synthetic economic simulation fixtures; future licensed market or customer data only after review
- Review actions: Financial-data licensing and permitted use require review before real-data use.; Outputs are research artifacts, not investment, tax, legal, credit, custody, payment, or suitability advice.; Qualified legal review is required before regulated production deployment.
- Go/no-go criteria: Go if all tests and synthetic benchmark artifacts reproduce.; Go if target users identify a painful workflow and agree to a pilot.; No-go for production if legal/data-rights review is unresolved.
- Exact external action: obtain qualified legal/security review before any regulated or real-customer financial use.
- Environment variables: none; do not add credentials or real customer data until review is complete.
- SQL: none required by the current local smoke workflow.
- Smoke verification: `cd /Users/ryan/Documents/Olympus && .venv/bin/python -m olympus.cli labos run-project genesis-econ --workspace /Users/ryan/Documents --artifacts artifacts --profile smoke`.
- Expected output: LabOS run status remains smoke-tested; production claims remain blocked until review is complete.

## Project Ledger Foundry

- Status: `Complete and smoke-tested`
- Why: Further external input is required.
- Path: `/Users/ryan/Documents/Ledger`
- Data actions: local synthetic or repository-bundled data
- Review actions: Financial-data licensing and permitted use must be reviewed.; Suitability, disclosures, payment/custody boundaries, and jurisdiction limits require qualified legal review.; Do not claim legal compliance without counsel-approved production controls.
- Go/no-go criteria: Go: deterministic smoke workflow passes locally.; Go: benchmark output is reproducible and stored with provenance.; No-go: core claim depends on unavailable private data or unstated manual steps.; No-go: legal/compliance review rejects the proposed operating boundary.
- Exact external action: obtain qualified legal/security review before any regulated or real-customer financial use.
- Environment variables: none; do not add credentials or real customer data until review is complete.
- SQL: none required by the current local smoke workflow.
- Smoke verification: `cd /Users/ryan/Documents/Olympus && .venv/bin/python -m olympus.cli labos run-project ledger --workspace /Users/ryan/Documents --artifacts artifacts --profile smoke`.
- Expected output: LabOS run status remains smoke-tested; production claims remain blocked until review is complete.

## Project Atlas — High-Impact Machine Learning Portfolio

- Status: `Complete except external dataset access`
- Why: One or more datasets require manual download or acceptance.
- Path: `/Users/ryan/Documents/ATLAS`
- Data actions: Place `train_FD00*.txt`, `test_FD00*.txt`, and `RUL_FD00*.txt` in `/Users/ryan/Documents/ATLAS/data/raw`, then run `atlas-preprocess --data-dir data/raw --output-dir data/processed` from `/Users/ryan/Documents/ATLAS`. Verify with `test -d /Users/ryan/Documents/ATLAS/data/processed`.
- Review actions: No regulated-production claim is made; project-specific legal review is required before deployment.
- Go/no-go criteria: Go: deterministic smoke workflow passes locally.; Go: benchmark output is reproducible and stored with provenance.; No-go: core claim depends on unavailable private data or unstated manual steps.
- Exact external action: download NASA C-MAPSS from the NASA Prognostics Data Repository after accepting the current terms.
- Required files: `train_FD001.txt`..`train_FD004.txt`, `test_FD001.txt`..`test_FD004.txt`, and `RUL_FD001.txt`..`RUL_FD004.txt`.
- Destination path: `/Users/ryan/Documents/ATLAS/data/raw/`.
- Environment variables: none required for the local preprocessing command.
- Verification command: `cd /Users/ryan/Documents/ATLAS && atlas-preprocess --data-dir data/raw --output-dir data/processed`.
- Expected output: processed C-MAPSS artifacts under `data/processed/`.
- LabOS verification: `cd /Users/ryan/Documents/Olympus && .venv/bin/python -m olympus.cli labos run-project project-atlas-portfolio --workspace /Users/ryan/Documents --artifacts artifacts --profile benchmark`.

## Project Genesis

- Status: `Complete except external dataset access`
- Why: One or more datasets require manual download or acceptance.
- Path: `/Users/ryan/Documents/Genesis`
- Data actions: From `/Users/ryan/Documents/Genesis`, run `python -m genesis.train --config configs/split_mnist.json --download-data` after approving the benchmark dataset download. Verify with `test -d /Users/ryan/Documents/Genesis/runs`.
- Review actions: No regulated-production claim is made; project-specific legal review is required before deployment.
- Go/no-go criteria: Go: deterministic smoke workflow passes locally.; Go: benchmark output is reproducible and stored with provenance.; No-go: core claim depends on unavailable private data or unstated manual steps.
- Exact external action: permit the first public benchmark dataset download for Split-MNIST/MNIST-family experiments.
- Destination path: `/Users/ryan/Documents/Genesis/data/`.
- Environment variables: none required for the documented local command.
- Download command: `cd /Users/ryan/Documents/Genesis && python -m genesis.train --config configs/split_mnist.json --download-data`.
- Expected output: MNIST-family files under the project data directory and a training run artifact under `runs/`.
- LabOS verification: `cd /Users/ryan/Documents/Olympus && .venv/bin/python -m olympus.cli labos run-project project-genesis --workspace /Users/ryan/Documents --artifacts artifacts --profile benchmark`.

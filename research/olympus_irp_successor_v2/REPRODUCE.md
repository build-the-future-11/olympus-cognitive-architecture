# Reproducing the controlled fusion study

Run from the repository root with the locked environment. Every command keeps the repository resource guard enabled and refuses to overwrite immutable scientific outputs.

```bash
PYTHONPATH=. .venv/bin/python scripts/run_controlled_fusion_study.py prepare-data
PYTHONPATH=. .venv/bin/python scripts/run_controlled_fusion_study.py develop
PYTHONPATH=. .venv/bin/python scripts/run_controlled_fusion_study.py freeze
PYTHONPATH=. .venv/bin/python scripts/run_controlled_fusion_study.py confirm
PYTHONPATH=. .venv/bin/python scripts/run_controlled_fusion_study.py analyze
```

The default dataset root is `artifacts/research/olympus-controlled-fusion-v2/dataset`. The default run root is `artifacts/research/olympus-controlled-fusion-v2/runs`.

`prepare-data` is one-shot. `develop` resumes missing reports and stops on a resource or workload-lock block. `freeze` requires all 25 development reports and binds the protocol, code, data, configurations, and checkpoints by SHA-256. `confirm` refuses to open protected outcomes before that receipt exists. `analyze` refuses to overwrite locked evidence.

The synthetic dataset's CC BY 4.0 designation applies to the generated data artifact, not to repository source code or other upstream datasets.

## Real-data staging

`datasets/external/editclevr` now contains the complete revision-pinned
EditCLEVR archive snapshot, extracted split registry, validation report, and
phase-one manifest. Its repository admission report and bounded real-payload
loader smoke are `PASS`; the revision-specific local payload is materialized.
This data was acquired after the frozen v2 study and was not used by it; the
negative v2 result and protocol remain unchanged. See `PROVENANCE.md` for
hashes, the retained materialization resource deviation, and scope.

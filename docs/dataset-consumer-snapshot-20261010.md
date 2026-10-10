# Bind dataset consumers to verified record snapshots

PR #23 verifies the complete manifest contract and parses each split from the
same bytes that it hashes. Three consumers then discarded those verified records
and reopened the split files: SFT training and validation, held-out evaluation,
and post-training quantization. A file replacement after verification could
therefore change the consumed records without changing the recorded manifest
identity. File removal could also fail an otherwise complete verified read.

`load_verified_dataset` now returns the manifest together with the parsed split
records after every hash, record, category, cross-split, and quality-report check
has passed. The three consumers use those records directly. The public
`verify_dataset_manifest` function remains a compatibility wrapper returning the
same manifest model. The obsolete SFT file-reopening helper is removed. SFT and
quantization release their split dictionaries after selecting or encoding the
needed records.

This is a data-consumption repair. It does not change manifest identities,
record order, model architecture, token weighting, losses, quantization math,
metrics, promotion rules, or recorded historical outcomes. The independently
proposed SFT loss correction in PR #26 has a small shared file surface but a
separate purpose. This PR is stacked on PR #23 and does not incorporate #26.

## Discriminating regression

The new tests prepare an artificial 36-record dataset with all declared
categories and splits. They replace or delete a split immediately after its
original bytes are returned to the real verifier. The exact records received
by evaluation and the SFT/quantization encoding boundary must still match the
verified snapshot. The unchanged-file cases check identity and ordering.

The parent revision fails 10 cases and passes 5 unchanged-data controls. The
repair passes all 15 new cases and all 19 existing manifest-contract cases:
34 focused tests. SFT tests stop before model construction and training;
quantization tests serialize artificial untrained tensors and stop before loss
evaluation or text generation. No capability result is produced.

## Local verification boundary

The available interpreter is Python 3.12.14 with PyTorch 2.14.1+cpu and Pydantic
2.13.5. The package declares Python 3.14 and its complete API dependencies are
not installed locally. The focused run imports the exact source modules while
bypassing only the package initializer that imports the unrelated web API:

```sh
python - <<'PY'
import sys
import types
from pathlib import Path

for name in ('olympus', 'olympus.core', 'olympus.foundry'):
    module = types.ModuleType(name)
    module.__path__ = [str(Path(name.replace('.', '/')).resolve())]
    sys.modules[name] = module

import pytest
raise SystemExit(pytest.main([
    'tests/test_dataset_consumer_snapshot.py',
    'tests/test_dataset_manifest_integrity.py', '-q',
]))
PY
```

No source transformation or production stub is used. Ruff and strict Mypy
checks pass for the four changed implementation modules and the new test file.
The normal Python 3.14 package, full integration suite, coverage, distribution,
and web checks remain required in the existing exact-source release workflow.
The local focused result alone does not establish that complete release gate.

`RESEARCH_STATE.json` preserves the prior negative capability finding and links
the dated predecessor ledgers. No Hermes model is promoted by this work.

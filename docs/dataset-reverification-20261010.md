# Reapply the instruction-dataset contract when reading manifests

The manifest reader checked file hashes and local counts, but did not enforce
the content rules used by dataset preparation. A self-consistent manifest could
omit a split, repeat record identities or text, introduce cross-split eight-token
contamination, remove category coverage, or retain a false quality report.

Preparation and verification now use one content validator. Verification still
checks the manifest hash, every split hash, record labels, counts and category
counts before applying the global constraints and comparing the computed quality
report. All three split descriptors are required. Valid prepared datasets retain
their exact manifest identity; this repair does not migrate or rewrite them.

Preparation also parses the same source byte snapshot that it hashes. A source
replacement between two file reads can no longer bind one version's hash to
another version's prepared records.

## Verification

The 19 new regression cases use the repository's tiny infrastructure fixture and
rehash modified manifests and split files to isolate content-admission defects.
The prior implementation fails 18 cases; the unchanged-valid-data control passes.
All 19 pass with the repair. The connected dataset, SFT, quantization and
promotion tests pass as well: 26 focused tests. The complete Python suite passes
179 tests with 87.38% branch-aware coverage under Python 3.12.14 and CPU PyTorch
2.14.1. The normal hosted release gate targets Python 3.14.

```sh
python -m pytest tests/test_dataset_manifest_integrity.py tests/test_deep_foundry.py -q
```

The change is an engineering correction to the documented dataset contract. It
does not establish model capability, alter held-out records, run a scientific
campaign, or promote a model. Previously retained artifacts remain untouched.

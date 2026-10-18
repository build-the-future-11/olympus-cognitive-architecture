# EditCLEVR admitted archive snapshot

Status: all five pinned archives downloaded and admitted; revision-specific payload materialized.  
Upstream dataset: `torux/EditCLEVR` on Hugging Face  
Upstream revision pinned: `734efa9b0164ba742cd2c39b9c2945c494497c23`, observed 2026-10-11  
Dataset license: CC BY 4.0  
Code license reported by upstream: MIT

The admitted `editclevr_splits.tar.gz` contains `splits.json`,
`validation_report.json`, and `phase1_manifest.json`. The four suite archives
contain the RGB images, masks, and scene JSON payloads. Acquisition resumed only
under the repository's unchanged 2 GiB available-memory floor and 90% swap
ceiling; the prior interrupted fragments were retained and resumed rather than
discarded.

`UPSTREAM_MANIFEST.json` records the immutable revision, observed upstream
timestamp, license, byte size, and LFS SHA-256 for every archive. The complete
snapshot is stored under `upstream-734efa9/`. Admission requires
`scripts/validate_editclevr_payload.py` to return `PASS` for all five archives,
the exact split-archive membership and staged split bytes, and every
split-referenced path.

## Integrity

- `UPSTREAM_README.md`: `d6e5deb64570e8c17eefe37e6794e0d04b9c17c2bda3f12abaa470685f86032c`
- `editclevr_splits.tar.gz`: `170a29a91bfe24515916c5da744c2dfb61b882637232090ff4e8d1113b02eb5f`
- `editclevr_atomic_id.tar.gz`: `b0f188e2f2e8b32b54416cfefa43740e976157345c1ea0186411e7082de48c4f`
- `editclevr_cogent_ood.tar.gz`: `0e2601daf5ec90b10f6f37b384050b4ea9e7d138f588dc0d3d68de9634026e0d`
- `editclevr_hard_distractor.tar.gz`: `458b4dcb4be6393c8dd943f0671dd89786d57af6cf88d1880506450ad1b0d73b`
- `editclevr_no_edit.tar.gz`: `b6c133ff5e39370e1e339a3ac77b34ff46218524ca161e06feac790663ac81be`
- `PAYLOAD_VALIDATION.json`: `a329f9df42468db22cb44a158673450548ebadb40455f7717c54835738f8b74a`

Every archive digest and byte count matches the upstream LFS metadata. The full
scan admitted 202,835 payload files with zero unsafe members, 20,000 unique
pairs, 120,000 valid referenced paths, zero missing paths, and zero invalid
records. The staged `splits.json` also matches the pinned split-archive member
byte-for-byte. See `PAYLOAD_VALIDATION.json`. The earlier interrupted state is
retained in `PAYLOAD_VALIDATION_PARTIAL_20261011.json`.

The archives were atomically materialized to the ignored local directory
`payload-734efa9/` with 202,838 extracted files plus a content-bound marker. The
first materialization finished with exact data integrity but its final snapshot
recorded 90.083% swap, narrowly above the unchanged 90% policy ceiling. That
deviation is retained in
`MATERIALIZATION_RECEIPT_RESOURCE_DEVIATION_20261011.json`; it is not rewritten
as a clean run. The materializer now rechecks the guard after each archive and
before atomic promotion. A subsequent bounded loader smoke verified all 2,000
CoGenT records and decoded a real 320x240 RGB pair, both scene JSON objects, and
both no-pickle NPZ mask bundles under the admitted ceiling. See
`LOADER_SMOKE_20261011.json`.

## Registered upstream split counts

- train: 10,000
- validation: 1,000
- in-distribution atomic test: 3,000
- no-edit control test: 2,000
- hard-distractor test: 2,000
- CoGenT OOD test: 2,000

## Olympus role

EditCLEVR is an admitted external visual intervention/OOD benchmark and a
possible source of paired images for future JEPA experiments. It does not
contain the complete Olympus aligned language, temporal, governance, risk, and
transition contract, so it cannot by itself close the predecessor study. No
performance claim is made from dataset admission alone.

## Upstream access limitations observed

The Hugging Face Dataset Viewer reported the dataset as non-previewable and failed to parse split names because the tar archives are not WebDataset-formatted. Reproduction therefore requires the upstream download helper or direct archive retrieval.

## Pinned acquisition and admission

```bash
hf download torux/EditCLEVR --type dataset \
  --revision 734efa9b0164ba742cd2c39b9c2945c494497c23 \
  --local-dir datasets/external/editclevr/upstream-734efa9 \
  --max-workers 1

python scripts/validate_editclevr_payload.py \
  --archive-dir datasets/external/editclevr/upstream-734efa9 \
  --manifest datasets/external/editclevr/UPSTREAM_MANIFEST.json \
  --splits datasets/external/editclevr/splits.json \
  --output datasets/external/editclevr/PAYLOAD_VALIDATION.json

python scripts/materialize_editclevr_payload.py \
  --archive-dir datasets/external/editclevr/upstream-734efa9 \
  --manifest datasets/external/editclevr/UPSTREAM_MANIFEST.json \
  --validation datasets/external/editclevr/PAYLOAD_VALIDATION.json \
  --output datasets/external/editclevr/payload-734efa9 \
  --receipt datasets/external/editclevr/MATERIALIZATION_RECEIPT.json
```

The download directory and archives are intentionally ignored by Git. The
manifest, split registry, provenance, and validation receipts are the
reviewable artifacts. The admission report is `PASS`, and the payload is
materialized. Any new development or confirmatory use must still follow a
separately declared, resource-admitted protocol and may not rewrite the frozen
v2/v3 studies.

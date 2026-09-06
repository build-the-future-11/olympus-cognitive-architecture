# Stage 05 — Training and Data Reliability

**State: SMOKE_TESTED.** The 36-record owned smoke corpus has 12 records per
immutable split and every declared category. Manifest SHA-256 is
`f64a41e9d3d79d5911e5abcf34e635bb76582abbdbb5d858b6405a1921c89285`.
Schema, licensing metadata, deterministic transforms, deduplication,
cross-split contamination, PII/secret checks, and test isolation are enforced.

Blocker: scale gates require 10,000 reviewed train and 1,200 held-out records.

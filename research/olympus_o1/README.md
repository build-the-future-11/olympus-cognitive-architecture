# Olympus O1 protocol migration

Status: **MIGRATED_PREREGISTRATION_NOT_RUN** / **UNFROZEN_DO_NOT_RUN**.

This directory makes the Olympus repository the canonical continuation owner
of the historical O1 protocol without rewriting its history or authorizing an
experiment. `upstream/` contains byte-exact copies of the nine files from the
closed, unmerged VertexED PR 653 at commit
`bc22ecc73201659cf099a20b6d038675b7307f28`. The prior cumulative scientific
head was `91061a8e9c19c12519841ac96f9f13395bc946de`.

`MIGRATION_PROVENANCE.json` binds every copied file to its SHA-256 and records
the only migration differences: repository path and explicit provenance. The
four arms, 100-task requirement, fixed promotion gates, 10,000-resample paired
bootstrap, seed, artifact schema, and negative/null/mixed-result acceptance are
unchanged.

## Execution boundary

No candidate task manifest, provider identity, runtime identity, prompt hash,
decoding configuration, tool allowlist, budget, or execution authorization is
present. The null-bearing templates are evidence of an open freeze gate, not
values to infer. Synthetic tests validate analysis and fail-closed behavior
only; they are not O1 results.

An outcome-bearing run remains prohibited until a separately reviewed change:

1. supplies exactly 100 pre-scored tasks satisfying the historical manifest
   contract;
2. replaces every execution-freeze null and binds the exact file hash;
3. freezes all four prompt/config definitions and a common provider, tool,
   retry, time, token, and runtime budget;
4. validates the artifact and budget-parity gates; and
5. records explicit execution authorization without modifying the protocol.

The later Olympus v1–v3 fusion studies are separate research lineages and do
not count as execution of this O1 protocol.

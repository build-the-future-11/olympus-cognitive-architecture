# Current-state audit collector

This optional, offline collector takes a fresh snapshot of a local Git workspace.
It is a discovery aid, not a model evaluator or release gate. It does not infer
present-day model status from the August 21 ledger or from a green engineering CI run.

## Run

From the actual repository root, with an existing Python 3.11+ on macOS or Linux:

```sh
python3 scripts/audit_current_state.py --repo "$PWD"
```

No dependency installation, model loading, network call, training, deployment,
promotion, or change to existing source/weights is performed. It writes a new,
private-permission directory under `reports/olympus-audit/`, containing
`CURRENT_STATUS.md` and `EVIDENCE.json`. Existing reports are never overwritten.
Review these reports before sharing: local paths and artifact identities can be
private even when model, training, prompt and credential contents are omitted.

## What it records

- Actual local HEAD, branch, dirty-worktree status, sanitized remote identities,
  and the ten most recent local commit hashes/timestamps.
- Candidate artifact file paths, sizes and filesystem timestamps under the six
  scoped roots: `artifacts`, `checkpoints`, `models`, `exports`, `runs`, `releases`.
- Git LFS pointer files, empty files, absent/unsafe checkpoint shards, symlinks
  deliberately not followed, JSON parse failures, and scan limits.
- SHA-256 of small metadata files actually read, separately from unverified
  checkpoint hashes and promotion-status claims contained in those files.

A `.pt`/`.bin` file is not unpickled; a `.safetensors` or `.gguf` file is not loaded.
Files whose names look like checkpoints are only *candidate artifacts*; they may
not contain weights at all. Complete shard paths do not establish compatible,
uncorrupted or runnable weights. The tool deliberately keeps training completion,
generation, capability evaluation, Ghost implementation and release readiness
unestablished until independently verified.

## Limits and boundaries

The defaults cap traversal at 20,000 entries, 2,000 reported candidate files and
eight directory levels below each artifact root. A JSON read is capped at 256 KiB;
the total JSON-read budget is 8 MiB; no more than 512 unique shards are inspected
per index. Weight reads are limited to a 128-byte pointer probe. Each Git read has
a five-second timeout and a 128 KiB stdout cap; stderr is not copied into reports.
Only audit-owned Git child processes may be terminated for those limits.

The inventory is not a transactional filesystem snapshot. Avoid running it while
another process rewrites model files. Symbolic links are not traversed, including
checkpoint shards and report parents. Parent-directory mutation races are outside
this discovery tool's security boundary; use a trusted, quiescent local workspace.
It is not a sandbox for hostile repositories or filesystem mounts.

Remote HEAD, other worktrees, external model caches, cloud training runs, deployed
services and target-hardware inference are not checked. No system-wide or GPU RAM
cap is claimed. The tool is small metadata work, not a heavy model workload.

## Finish the audit in Cursor

1. Read the snapshot and reconcile untracked/modified code with remote HEAD using
   authenticated, read-only access. Do not reset, switch, stash, or clean the owner’s
   worktree. Keep separate Olympus repositories separate.
2. Trace each candidate through dataset, training configuration/logs, weights,
   held-out evaluation, export and serving evidence. Bind each item to its exact
   artifact identity; preserve all existing quality gates.
3. Inspect the implementation of Full/Ghost mechanisms. A name, wrapper, sharded
   weight file or ordinary quantization is not evidence of approximate products or
   learned jumpers. Record what is implemented and what is merely proposed.
4. Admit only lightweight, explicitly bounded tests supported by existing local
   dependencies and available memory. Do not download/train/promote to make a
   status check look successful. Preserve test data and frozen research protocols.
5. Report current verified findings, historical evidence, blockers and unverified
   items separately, with exact commands, exit codes and artifact identities.

The standalone collector supports Python 3.11+ for inspection convenience. This
**does not change** the Olympus package's Python >=3.14 runtime requirement.

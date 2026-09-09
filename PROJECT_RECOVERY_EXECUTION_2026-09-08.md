# Live job recovery execution — 2026-09-08

Fixed an operational gap in `olympus/foundry/jobs.py`: recovery previously ran only
at manager construction. An existing observer could continue reporting a dead
worker as RUNNING, and a new start could bury that record without resolving it.

State/history inspection and explicit retry now attempt reconciliation under the
exclusive journal lease. A live lease owner prevents reconciliation. New starts
reconcile abandoned RUNNING/CANCEL_REQUESTED entries before creating a new record,
preserving their identity, failure explanation and completion timestamp. Failure
during this reconciliation releases the newly acquired lease.

Regression tests kill a real worker and use an already-created observer to detect
its failure. Additional tests cover preservation of abandoned running and
cancel-requested records when another job starts. `docs/foundry.md` describes the
behavior. These tests do not establish exactly-once external effects or every
possible crash transition.

Targeted worker/job suite: 19 passed before the additional cancellation-state
parameterization. Repository Ruff and strict mypy (105 files) passed. No deployment,
new trained model, new benchmark result or completed portfolio is claimed.

Final full-suite verification: **299 passed in 563.28 seconds, 85.39% combined
coverage**, meeting the unchanged 85% gate. The host was under elevated load.
`git diff --check` passed. Frontend, packaging and deployment were not rerun in
this recovery-focused pass.

Remaining scope includes the full fault-injection matrix, job-history UI,
evidence-consumer replay/revocation, qualified trained checkpoints, independent
evaluation and review of untracked release sources. Public model status remains
NOT READY. Existing unrelated changes and frozen research were preserved.

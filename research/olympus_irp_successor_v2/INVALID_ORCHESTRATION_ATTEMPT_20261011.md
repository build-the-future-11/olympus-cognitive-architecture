# Invalid orchestration attempt — 2026-10-11

The first development command continued after its caller's 30-second output window and held the exclusive Olympus training lock. A second command was started before that process's state had been resolved. It encountered the lock after the first command completed the five `full` runs and wrote 21 `failure.json` receipts for the still-missing variant/seed directories.

Classification: **INVALID orchestration attempt**, not model evidence.

Cause: the campaign runner treated lock contention as a per-run failure and continued through the matrix rather than treating the exclusive-lock conflict as a campaign-level stop.

Correction: lock contention now writes one campaign block receipt and returns immediately. Existing failure receipts are retained. A later successful immutable `report.json` and checkpoint in the same run directory supersede the lock-contention receipt for execution status but do not erase its history.

Scientific impact: none of the protected clean, corrupted, or OOD outcomes had been opened. The completed `full` development reports remain valid development artifacts. No protocol, model, hyperparameter, metric, or protected split was changed.

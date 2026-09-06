# Stage 08 — Resource-Safe Local Foundry

**State: SMOKE_TESTED.** Small and medium workload policies use live available
memory, swap ratio, exclusive concurrency, bounded queueing, atomic state, and
cooperative cancellation. On this 16 GiB host the small workload was admitted;
the medium workload was refused because available memory was below 8 GiB and
swap exceeded its 50% ceiling.

Claim boundary: refusal is a successful safety outcome, not a failed run.

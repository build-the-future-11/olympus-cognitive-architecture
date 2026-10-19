# Workflow memory-policy review

Read-only mathematical and documentation review, 2026-10-19. This review did not execute the workflow or change repository code. Preserve the failed original preflight and version any revised admission policy.

## Finding

Crediting reclaimable file cache is a reasonable diagnostic estimate when raw cgroup usage is dominated by cache, but the originally proposed formula does not establish that the credited pages are simultaneously inactive, clean, and unmapped:

`min(inactive_file, max(0, file - shmem - file_mapped - file_dirty - file_writeback))`

It upper-bounds the possible intersection of those sets. For example, if total file cache is 100 units, inactive file cache is 10, and all 10 inactive units are dirty, the formula still credits 10 clean inactive units because the remaining 90 clean units may all be active.

A more conservative estimate is:

```text
disqualifying = shmem + file_mapped + file_dirty + file_writeback
credit = min(used, max(0, min(inactive_file, file) - disqualifying))
raw_headroom = max(0, limit - used)
estimated_headroom = max(0, limit - used + credit)
effective_estimate = min(host_MemAvailable, each_finite_ancestor_estimated_headroom)
```

Subtracting overlapping disqualifying counters more than once can under-credit cache; it is conservative for this purpose. Use no credit if required fields are absent, invalid, or from an inconsistent scope. Keep all quantities in bytes. Do not sum credits across child and ancestor cgroups. Cap credit by observed usage and preserve signed `limit-used` internally: clamping a negative raw headroom before adding credit would incorrectly erase existing overage.

The kernel describes `file` as a type-based count including tmpfs/shared memory, while `inactive_file` is an internal reclaim-list count. It explicitly warns that type totals and active/inactive list totals need not match. Mapped/dirty/writeback statistics are not an atomic per-page intersection. Docker subtracts inactive file cache when displaying Linux container usage; that convention is not a guarantee that all credited memory can immediately satisfy an allocation. [K1, K2]

Therefore name the result a **conservative reclaimable-headroom estimate**, record raw usage and raw headroom beside it, and retain the declared 2 GiB threshold for the estimate. Reclaim delay, changing counters, refaults, sibling activity, and protected memory can still affect actual availability. Record the exact policy version and inputs. Retain child time limits, polling aborts, and the global watchdog; the estimate does not replace process supervision. If the study requires 2 GiB of strictly uncharged capacity, cache credit changes that contract and cannot be described as merely the same measurement.

For cgroup v1, only use matching hierarchical `total_*` statistics with hierarchical usage/limits; otherwise give no cache credit. For cgroup v2, inspect every visible finite ancestor limit. The same cache can appear in several ancestor counts, which is why the final comparison uses the minimum rather than the sum.

## Boundary cases to cover before the rerun

| Case | Required behavior |
|---|---|
| `file=100`, `inactive=10`, `dirty=10`, other exclusions zero | Credit zero; catches the set-intersection error. |
| Inactive cache exists, one required stat missing | Credit zero. |
| Exclusion counters exceed inactive cache or overlap | Clamp credit to zero; do not produce negative availability credit. |
| Current usage is 10 above limit, eligible credit is 20 | Estimated headroom is 10, not 20. |
| Parent allowance is smaller than child allowance | Parent binds. |
| Host available memory is smaller than all cgroup estimates | Host binds. |
| Exactly 2 GiB estimated headroom, then one byte less | First admits, second rejects. |
| Clean inactive cache exceeds observed usage due to inconsistent snapshots | Credit never exceeds usage. |

These are proposed mathematical boundary cases, not executed test results.

## Primary sources

- K1: [Linux kernel cgroup v2 documentation](https://docs.kernel.org/admin-guide/cgroup-v2.html), `memory.stat` definitions for file/shmem/mapped/dirty/writeback and internal LRU lists; `memory.max` and `memory.high` semantics. Retrieved refs: `turn16view0`, `turn16view1`, `turn16view2`.
- K2: [Docker container stats documentation](https://docs.docker.com/reference/cli/docker/container/stats/), Linux memory display note: v2 uses `inactive_file`; v1 uses `total_inactive_file`. Retrieved ref: `turn16view3`.

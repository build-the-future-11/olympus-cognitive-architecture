# Results regenerated from retained evidence

Source: `92cc121e6f1074de72eaee5a8cb670170107bbc7`. Both archives and their members were verified.

## Complete Python suite

| Measurement | Result |
|---|---:|
| Test cases | 432 |
| Failures | 0 |
| Errors | 0 |
| Skips | 0 |
| Combined statement/branch coverage | 88.53% |
| Statement coverage | 92.02% |
| Branch-only coverage | 73.46% |

## Held-out character-bigram lifecycle

| Metric | Uniform baseline | Candidate |
|---|---:|---:|
| negative_log_likelihood | 3.17805383035 | 1.9001575711 |
| perplexity | 24 | 6.68694802902 |
| top1_accuracy | 0.0416666666667 | 0.392592592593 |
| evaluated_transitions | 135 | 135 |

## Measured execution bounds

| Measurement | Result |
|---|---:|
| Total workflow seconds | 4.272115 |
| Maximum child RSS, bytes | 67338240 |
| Minimum estimated headroom, bytes | 2231209984 |
| Swap-use change, bytes | 0 |
| Checkpoint/export bytes | 1712 |

## Preserved lifecycle attempts

| Attempt | Status | Children launched | Seconds |
|---|---|---:|---:|
| v1 | FAILED | 0 | 0.018619 |
| v2 | FAILED | 3 | 1.848801 |
| v3 | FAILED | 0 | 0.017611 |
| v4 | PASSED_ENGINEERING_ONLY | 8 | 4.272115 |

## Residency diagnostic

Disposition: **BLOCKED**. One fixture worker; no arm or primary metric.
Minimum estimated headroom: 2123210752 bytes; shortfall: 24272896 bytes.

This regeneration checks retained bytes and arithmetic. It does not rerun a model, reproduce a scientific experiment, or reconstruct unrecorded memory.stat inputs.

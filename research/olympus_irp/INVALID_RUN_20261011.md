# Invalid matrix attempt retained

The first execution attempt is preserved at
`artifacts/research/olympus-irp-v1`.

- The resource governor rejected all 24 planned runs before training because swap
  utilization was 91.2%, above the frozen 90.0% ceiling.
- Those resource rejections are valid operational observations and were written to
  `failures.jsonl` and `progress.json`.
- After the rejections, the analysis process raised `StatisticsError` while attempting
  to plot an empty completed-run set. The failure prevented a completion receipt, so the
  attempt is classified `INVALID_ANALYSIS_AFTER_VALID_RESOURCE_REJECTIONS`.
- The defect was repaired by making the figure generator return an empty figure list
  when no run completed and by guarding legend construction.
- No failed artifact was deleted or overwritten. A corrected run must use a new output
  root and remains subject to the unchanged resource ceiling.

This bug affected only post-run aggregation. It did not allow training, expose new model
outcomes, or alter the frozen matrix.

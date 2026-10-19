# Development audit before protected outcomes

The four declared blocks completed. The selection rule chose configuration b:
geometric mean full-arm development NMSE20 was0.018629816693557427, versus
0.02250838671504566 for a. No other hyperparameter was changed after results.

## What the unprotected evidence says

For selected b, arithmetic means over the two development seeds were:

| Family | Full | Direct,500updates | Direct,matched CPU | Cubic ridge |
|---|---:|---:|---:|---:|
| Pendulum |0.023767753879974998|0.019875413736617744|0.011524071002882025|0.000039091717476852705|
| Duffing |0.016929768177640896|0.013028465197967501|0.008444084747065886|0.000000046674459693227575|

The candidate is not the best development model. More direct optimization
improves this network, while current-state cubic features match these simple
dynamics particularly well. This is evidence against the proposed practical
auxiliary benefit, not evidence for a new latent architecture.

## Observation, explanation, action and discriminating evidence

Observation: increasing the two auxiliary coefficients together improved the
full arm on the declared selection endpoint, but did not overcome direct or
polynomial controls.

Hypothesized causes: the fully observed two-dimensional problem has little
reason to discard nuisance information; latent/reconstruction constraints can
compete with observable future-state fitting; time spent on auxiliary terms
can be more useful as additional direct optimization. Polynomial structure
provides a favorable inductive bias for the chosen numerical dynamics.

These are competing explanations, not established causes. The prespecified
factorial, fixed-target and shuffled-target arms provide descriptive evidence
about them. The primary confirmation comparison remains full versus the same
architecture at matched training process CPU time.

Action: no further search, horizon changes, seed additions, baseline removal or
threshold change. Freeze b and perform the already bounded falsification study.
A valid negative finding will close this hypothesis; a new scientific mechanism
requires a separately versioned successor contract. The inexpensive baseline's
success cannot be reassigned to the auxiliary mechanism.

## Readiness evidence

The complete dry-run verifier passed both families and all ten arms. The
supervisor recorded a377053184byte largest-child RSS and minimum1352998912byte
sampled remaining headroom. The original2GiB preflight failure remains retained.
The final engineering gate passed520Python tests,11web tests,87.82% branch-enabled
coverage, Ruff, strict MyPy, both dependency audits, both builds and Twine.
The built wheel's runtime Python files match the tested source exactly.

Independent reviewers inspected novelty/scope, resource admission, checkpoint
failure preservation, target behavior and analysis completeness. One separately
authored fault-injection test module was integrated and executed by root.
Reviewers did not independently repeat scientific training.

No protected trajectories were generated or inspected while writing this audit.

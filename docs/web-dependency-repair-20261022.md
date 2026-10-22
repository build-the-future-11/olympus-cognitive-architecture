# Web dependency security repair — 2026-10-22

## Observed release failure

Combined PR 18 head `2a1050a4c854237616ac0f912569ce46a7dd78c6` passed the Python job and the web test step in [release run 39773984513](https://github.com/build-the-future-11/olympus-cognitive-architecture/actions/runs/39773984513). Web job 119808202579 then failed the existing `npm audit --audit-level=moderate` gate with two moderate and one high vulnerable dependency entries. Its production-build step was skipped. The full failed job log is retained, and this repair does not relabel that failed run as passed.

## Minimal affected dependency graph

| Package | Previous lock | Repaired lock | Reason |
| --- | --- | --- | --- |
| `vite` |7.3.6|7.3.7|Published fix for the five Vite advisory entries below.|
| `vitest` |5.0.0|5.0.3|Patch release requiring fixed mocker.|
| `@vitest/mocker` |5.0.0|5.0.3|Published fix for both reported mocker advisories.|
| `@vitest/spy` |5.0.0|5.0.3|Exact mocker dependency.|
| `why-is-node-running` |2.3.0|3.2.1|Exact dependency required by Vitest 5.0.3.|

The direct manifest minimums become `vite: ^7.3.7` and `vitest: ^5.0.3`. The lock root mirrors those values. The five entries use exact versions, tarball URLs, SHA512 integrity strings and dependency/peer/engine metadata read from the official npm registry. `siginfo` and `stackback` are removed because the replacement why-is-node-running package no longer depends on them and no remaining lock entry references them. Other locked package versions remain unchanged. There is no major Vite/Vitest upgrade, unrelated refresh, application source, test, workflow or audit-threshold change.

## Official patch evidence

GitHub's reviewed advisory records identify Vite 7.3.7 as fixed for:

- [GHSA-698v-4r3x-3vcv](https://github.com/advisories/GHSA-698v-4r3x-3vcv)
- [GHSA-r825-w8mm-5pxw](https://github.com/advisories/GHSA-r825-w8mm-5pxw)
- [GHSA-587h-7wmj-h5pm](https://github.com/advisories/GHSA-587h-7wmj-h5pm)
- [GHSA-9jrq-w75r-8gcw](https://github.com/advisories/GHSA-9jrq-w75r-8gcw)
- [GHSA-rq7h-c2jc-7f22](https://github.com/advisories/GHSA-rq7h-c2jc-7f22)

The affected mocker 5.x line is fixed at 5.0.3 for [GHSA-qxg6-89mg-rqv8](https://github.com/advisories/GHSA-qxg6-89mg-rqv8) and [GHSA-xp26-3454-mjgp](https://github.com/advisories/GHSA-xp26-3454-mjgp). Source Git blob identities, relevant official affected/fixed ranges and the five exact npm registry manifests are retained in `registry_and_advisory_receipt.json`. This review concerns dependency patch selection; it does not claim that the application's configuration was exploited.

## Validation and remaining hosted gate

The existing npm 11.9.0 Arborist loaded the repaired virtual lock tree successfully: **303 dependency edges**, **zero invalid edges**, and **199 lock entries including the root**. All five affected manifest identities, dependencies and peer declarations match the fetched registry metadata. Their declared engines admit the unchanged hosted Node 22.23.2. Local Node was 24.19.0. No repaired dependency code was installed or executed; these checks used existing npm tooling.

A separate registry-backed command used `npm audit --package-lock-only --audit-level=moderate --include=dev --json --ignore-scripts`, exited 0 and reported **zero total vulnerabilities** across the 198-package graph (including 195 dev entries; npm's categories overlap). It left the lockfile bytes unchanged. The graph-validation command took 0.078805639 seconds; the audit command took 0.3838561509983265 seconds. The prospective one-graph-check/one-audit budget is closed. Full command/options, complete audit output and stderr, hashes and environment are retained. The initial sandbox denied read-only npm metadata network requests; approved read-only registry access then succeeded. No version guesses or substituted integrity hashes were used.

No `node_modules` directory was installed. No local application tests or production build were run. The fresh exact-head hosted release gate remains required, including its unchanged Python job and web `npm ci`, `npm test`, `npm audit --audit-level=moderate`, and `npm run build`. A zero lock audit alone is not a claim of final release verification.

## Preserved scientific boundary

This is dependency maintenance only. The exact prior canonical state and failed hosted log are retained, and the new state appends this event. Model ledgers, frozen outcomes, prior budgets, failed capability promotion, provider boundaries and historical `NOT_PROMOTED` status remain unchanged. No protected evaluations, new datasets, training, model promotion, provider call, paid compute or release publication occurred.

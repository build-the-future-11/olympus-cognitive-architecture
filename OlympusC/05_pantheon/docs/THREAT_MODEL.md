# Threat Model

Pantheon assumes a frozen canonical package can be created before replication and that the auditor can read the relevant artifacts after execution.

## Threats addressed by the prototype
- one-sided data/config/code/protocol drift;
- runtime metric mode drift;
- shared correlated code drift relative to canonical artifacts;
- proposer-only canary exposure in replicator output;
- numerical disagreement without silently discarding raw metrics.

## Threats not addressed
- compromised host or malicious auditor;
- sophisticated exfiltration through side channels;
- undeclared network access;
- hardware nondeterminism;
- hidden model-provider state;
- prompt memorization or training-data contamination;
- semantically equivalent code whose hashes differ;
- semantically different code whose output matches on the tested sample.

A production research-agent evaluation should add containers/VMs, network policy, file capability controls, signed manifests, and external model/process identity attestation.

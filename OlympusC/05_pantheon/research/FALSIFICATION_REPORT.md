# Falsification Report

Pantheon’s broad original thesis is **not yet established**. The following potential falsifiers and weaknesses were actively checked.

1. **Could simple numerical comparison be enough?** No on the seeded suite: it localizes only the clean class correctly and misses artifact-cause information.
2. **Could pairwise artifact comparison be enough?** No on the controlled shared-bug, metric-runtime, and canary cases. In the shared-bug condition, proposer and replicator agree with each other while both diverge from the frozen canonical implementation.
3. **Does perfect controlled accuracy prove real-world robustness?** No. The benchmark is engineered and currently small. This is the strongest limitation.
4. **Is cross-agent independence demonstrated?** The frozen OOD study used different local model identities in separate ephemeral workspaces without sharing transcripts or canonical answers. Both models nevertheless used the same Ollama provider and host, and both failed the capability floor. Provider independence, separately administered infrastructure, and independence among capable agents remain untested.
5. **Is a genuine published research claim replicated?** No. The public-data experiment is a cross-implementation case study, and neither local agent answered a CORE-Bench question or authored a valid report. The originally requested Research Atlas package was not present in the supplied archive.
6. **Could the public-data agreement be an artifact of one implementation?** The two training implementations differ (scikit-learn L-BFGS vs. in-repo numpy batch gradient descent) and are evaluated on ten prespecified splits, reducing but not eliminating this concern.
7. **Did the completed OOD run validate the cross-agent thesis?** No. The artifact matrix is complete, but both agents returned 0/20 non-null answers and 0/34 agent-authored reports. The observed 0/20 false-consensus count is an all-abstention result and cannot estimate false-consensus prevalence among capable agents.

## Conclusion
The controlled evidence falsifies the idea that pairwise agreement alone is a sufficient replication criterion on the seeded failure modes. The external evidence shows that the current small local agents are below the capability floor required to test the broader thesis. It does **not** establish that Pantheon improves the reliability of autonomous LLM research agents in the wild.

# Paired Kaggle allocation, single-device execution

2026-10-19: After the CPU qualification was downloaded and verified, the user
explicitly answered: "Approve T4×2 runtime, use one GPU".

This admits Kaggle's free paired T4 runtime for the bounded CUDA qualification.
All qualification subprocesses must receive `CUDA_VISIBLE_DEVICES=0` before
Torch import; no test computation may use GPU 1. The paired allocation does not
authorize multi-GPU execution, paid compute, external data, scientific training,
protected benchmarking, or model promotion. Do not interrupt another task's job.

The unchanged frozen numerical protocol still limits each qualification to 120
seconds, one CPU thread, eight artificial optimizer steps, and its existing tiny
fixtures. The allocation exception is recorded here without rewriting the frozen
protocol or the CPU result. The imported CUDA r2 notebook has SHA-256
`ba209adce60d2f7f84e8c53f1f840ef2e36e3f90dc9a3f31bfde34653caf316a`
and embedded archive SHA-256
`827eaed50fc5aba49fa0e33a261c0fd1e71f74db96422371f725430e2f26b39a`.

Acceptance still requires saved-run outputs downloaded and verified against the
external archive digest, manifest/member hashes, runtime/test/protocol source
identities, regression outcomes, finite gradient coverage, and checkpoint reload.

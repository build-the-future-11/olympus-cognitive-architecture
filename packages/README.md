# Package Map

The Python implementation currently uses the `olympus` namespace package at the repository root. The package responsibilities align with the requested monorepo boundaries as follows:

- `olympus.core` maps to `olympus-core`, `olympus-routing`, `olympus-representations`, `olympus-prediction`, `olympus-verification`, `olympus-observability`, and `olympus-security`.
- `olympus.forge` maps to `olympus-forge` and behavior compilation/runtime concerns.
- `olympus.memory` maps to `olympus-memory`.
- `olympus.data` maps to `olympus-data`.
- `olympus.training` maps to `olympus-training`.
- `olympus.evaluation` maps to `olympus-evaluation`.
- `olympus.models` maps to `olympus-models` and inference surfaces for future model families.
- `olympus.sdk` maps to `olympus-sdk`.

This keeps runtime imports simple while the repository is still in active architectural growth.


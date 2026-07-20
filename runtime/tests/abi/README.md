# Model-runtime ABI baselines

This directory stores immutable declarations used to verify binary
compatibility. A baseline is a snapshot of a published ABI, not a second API
and not a version alias.

Rules:

- Preserve the published type names, field order, enum values, and data model.
- Compile baseline producers in isolated translation units that do not include
  the candidate model-runtime header.
- Include a baseline in a namespace only for compile-time layout comparison.
- Do not update a v1 baseline when additive tail fields are introduced; compare
  those fields through their own required-size probes.
- Do not install these headers or expose baseline-producer helpers from runtime DSOs.

`model_runtime_v1_abi_baseline.h` captures the v1 required prefix through
`release`, plus the v1 payload, descriptor, verbs, and enum layouts.

## Summary

<!-- Describe the behavior and ownership boundary changed by this PR. -->

## Design boundaries

<!-- State what remains model-, backend-, policy-, or mechanism-owned. -->

## Compatibility

<!-- Public API, payload, fingerprint, capsule, packaging, and migration impact. -->

## Validation

<!-- Use sanitized commands/results. Do not include private paths or environments. -->

- [ ] Focused tests cover success and rejection paths
- [ ] Affected CUDA-off/hardware configurations were checked or disclosed
- [ ] Numerical claims use a fixed, justified gate
- [ ] Low-precision claims include identity, artifact, and kernel-dispatch evidence
- [ ] Compared producer binaries were built from the same source revision
- [ ] Calibration artifacts and cache invalidation are identity-complete
- [ ] Single/multi-sample calibration and named-input rejection are covered
- [ ] STAGED ports have real matching verbs
- [ ] Identity uses observed runtime facts and changes with contract changes
- [ ] Hot-path allocation/capture/rebind claims are measured at the right scope
- [ ] Documentation and migration notes are updated
- [ ] Diff contains no private paths, hosts, containers, credentials, or logs
- [ ] Shared kernel/CMake ownership and packaging were reviewed
- [ ] Model semantics remain model-local; frozen runtime/exec stay generic
- [ ] External test migration preserves core contract/result coverage

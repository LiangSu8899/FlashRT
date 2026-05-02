"""Reference (oracle) implementations.

These modules host independent, framework-faithful reference
implementations used as ground-truth oracles for FlashRT's optimised
inference paths. Reference code is **not** part of the production
runtime — it is only loaded by tests and offline fixture generators.
A reference implementation must:

  * Mirror the paper's published math exactly (no fused tricks, no
    custom kernels).
  * Use the same model weights as the corresponding production path.
  * Be self-contained: the only acceptable upstream dependency is the
    upstream model package (e.g. ``openpi``).

See ``docs/precision_spec.md`` for the correctness contract this
module backstops.
"""

"""FlashVLA — framework-specific frontends.

Each frontend loads a checkpoint in its native format, builds the weight
pointer dict, constructs a model pipeline from ``flash_rt.models.*``,
captures CUDA Graphs, and exposes ``infer(obs) -> actions``.

Subdirectories: ``torch/`` and ``jax/``.
"""

"""FlashVLA — model-specific pipeline and calibration code.

Each subdirectory (pi05/, groot/, pi0/, pi0fast/) holds the pipeline
logic for one model family.  Hardware-specific attention backends live
in ``flash_vla.hardware.{rtx,thor}``; frontends (weight loading, graph
capture, framework glue) live in ``flash_vla.frontends.{torch,jax}``.
"""

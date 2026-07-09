"""FlashRT — model-specific pipeline and calibration code.

Each subdirectory holds the pipeline logic for one model family:
pi05/, pi0/, pi0fast/, groot/, groot_n17/, qwen3/, qwen3_vl/, qwen36/,
nexn2/, motus/, wan22/, cosmos3_video/, higgs_audio_v3/, omnivoice/,
lingbot/, melband_roformer/, minimax_remover/.

Hardware-specific attention backends live in
``flash_rt.hardware.{rtx,thor}``; frontends (weight loading, graph
capture, framework glue) live in ``flash_rt.frontends.{torch,jax}``.
"""

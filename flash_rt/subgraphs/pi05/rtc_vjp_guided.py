"""Pi0.5 full RTC guided-denoise capture contract.

The full Kinetix-style RTC algorithm needs a VJP of the denoiser with respect
to the current action chunk. Pi0.5's current FlashRT pipeline is an inference
graph made of custom kernels and GEMM launches, so it does not provide that
VJP by default. This module defines the producer-owned plug-in point without
pretending that prefix locking is full VJP guidance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from flash_rt.subgraphs.capture import (
    register_capture_hook,
    register_frontend_capture_hook,
)


@dataclass(frozen=True)
class VjpGuidanceConfig:
    """Capture-time constants for a VJP-guided action graph."""

    prefix_attention_schedule: str = "exp"


class DenoiserVjpProvider(Protocol):
    """Producer-supplied implementation of the guided action graph.

    The provider must capture or adopt a replayable graph named
    ``decode_rtc_vjp_guided`` and register it with
    ``flash_rt.subgraphs.capture.register_captured_graph`` or
    ``register_export_graph``. The graph must read these fixed-address ports:

    - ``rtc_prev_action_chunk``: raw model action chunk, shape (chunk, 32)
    - ``rtc_prefix_weights``: float32 prefix weights, shape (chunk,)
    - ``rtc_guidance_weight``: float32 scalar max guidance weight

    Nexus sees only the resulting stage and ports; the VJP math belongs here.
    """

    def capture_guided_action_graph(
        self,
        pipeline: object,
        stream_handle: object,
        stream_int: int,
        config: VjpGuidanceConfig,
    ) -> None:
        ...


def enable(target: object, *, provider: DenoiserVjpProvider | None = None,
           config: VjpGuidanceConfig | None = None) -> None:
    """Enable ``context -> decode_rtc_vjp_guided`` for a Pi0.5 producer.

    A provider is mandatory. The stock Pi0.5 FlashRT inference pipeline has no
    autograd/VJP surface, and exporting this plan without a real provider would
    be a false implementation.
    """
    from flash_rt.subgraphs.pi05.context_action import enable as _context_enable
    from flash_rt.subgraphs.pi05 import stage_plans as _stage_plans  # noqa: F401

    if provider is None:
        raise RuntimeError(
            "context_rtc_vjp_guided_action requires a DenoiserVjpProvider. "
            "The current Pi0.5 FlashRT inference pipeline has no built-in "
            "denoiser VJP/backward graph; use context_rtc_prefix_action for "
            "the prefix-lock RTC graph, or provide a producer implementation "
            "that captures/registers decode_rtc_vjp_guided.")

    cfg = config or VjpGuidanceConfig()
    _context_enable(target)
    frontend = getattr(target, "_pipe", target)
    if hasattr(frontend, "record_infer_graph"):
        _install_pipeline(frontend, provider, cfg)
        return
    if hasattr(frontend, "pipeline"):
        register_frontend_capture_hook(
            frontend, lambda pipeline: _install_pipeline(
                pipeline, provider, cfg))
        return
    raise TypeError("rtc_vjp_guided.enable expects a VLAModel, frontend, "
                    "or pipeline")


def _install_pipeline(pipeline: object, provider: DenoiserVjpProvider,
                      config: VjpGuidanceConfig) -> None:
    def _hook(pl: object, stream_handle: object, stream_int: int) -> None:
        provider.capture_guided_action_graph(
            pl, stream_handle, stream_int, config)

    register_capture_hook(pipeline, _hook)


__all__ = ["DenoiserVjpProvider", "VjpGuidanceConfig", "enable"]

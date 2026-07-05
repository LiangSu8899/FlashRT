"""Optional Pi0.5 RTC-prefix action graph.

This is a producer-side graph cut. It keeps the default Pi0.5 capture intact
and adds a second action graph that reads ``rtc_prev_action_chunk`` inside the
denoise loop. Nexus still sees only ordinary stages and ports.
"""

from __future__ import annotations

from flash_rt.subgraphs.capture import (
    capture_graph,
    register_capture_hook,
    register_captured_graph,
    register_frontend_capture_hook,
)


def enable(target: object, *, prefix_len: int) -> None:
    """Enable ``context -> decode_rtc_prefix`` for a Pi0.5 producer.

    ``prefix_len`` is fixed at capture time, because CUDA graph replay records
    fixed kernel arguments. Different prefix lengths should be represented as
    distinct producer plans/variants.
    """
    prefix = int(prefix_len)
    if prefix < 0:
        raise ValueError("prefix_len must be >= 0")
    from flash_rt.subgraphs.pi05.context_action import enable as _context_enable
    from flash_rt.subgraphs.pi05 import stage_plans as _stage_plans  # noqa: F401

    _context_enable(target)
    frontend = getattr(target, "_pipe", target)
    if hasattr(frontend, "record_infer_graph"):
        _install_pipeline(frontend, prefix)
        return
    if hasattr(frontend, "pipeline"):
        register_frontend_capture_hook(
            frontend, lambda pipeline: _install_pipeline(pipeline, prefix))
        return
    raise TypeError("rtc_prefix.enable expects a VLAModel, frontend, "
                    "or pipeline")


def _install_pipeline(pipeline: object, prefix_len: int) -> None:
    setattr(pipeline, "_rtc_prefix_len", int(prefix_len))

    def _hook(pl: object, stream_handle: object, stream_int: int) -> None:
        _capture_rtc_prefix_graph(pl, stream_handle, stream_int, prefix_len)

    register_capture_hook(pipeline, _hook)


def _capture_rtc_prefix_graph(pl: object, stream_handle: object,
                              stream_int: int, prefix_len: int) -> None:
    capture = getattr(pl, "capture_rtc_prefix_graph", None)
    if capture is not None:
        capture(stream_handle, stream_int, prefix_len)
        return
    if getattr(pl, "_decode_rtc_prefix_graph", None) is not None:
        return
    if int(prefix_len) > int(getattr(pl, "chunk_size", 0)):
        raise ValueError(
            f"prefix_len {prefix_len} exceeds chunk_size "
            f"{getattr(pl, 'chunk_size', None)}")

    graph = capture_graph(
        pl, stream_handle,
        lambda: pl.transformer_decoder(stream_int, rtc_prefix_len=prefix_len))

    pl._decode_rtc_prefix_graph = graph
    register_captured_graph(
        pl, "decode_rtc_prefix", graph,
        exec_name=f"pi05_decode_rtc_prefix_{int(prefix_len)}",
        stream=0,
        variants=(0,),
    )


__all__ = ["enable"]

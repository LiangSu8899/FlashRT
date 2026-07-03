"""Optional Pi0.5 context/action graph cut.

Default Pi0.5 capture remains the full graph plus decoder-only graph. Call
``enable(pipeline)`` before ``record_infer_graph`` to add a separate context
graph (prompt copy + vision encoder + transformer encoder) and export it under
the name ``context``. The matching stage plan is registered by
``flash_rt.subgraphs.pi05.stage_plans``.
"""

from __future__ import annotations

from flash_rt.subgraphs.capture import (
    capture_graph,
    register_frontend_capture_hook,
    register_captured_graph,
    register_capture_hook,
)


def enable(target: object) -> None:
    """Enable the Pi0.5 ``context -> decode_only`` split.

    ``target`` may be the public ``VLAModel``, the Pi0.5 frontend, or an
    already-built pipeline. For lazy frontends the hook is applied when the
    pipeline is created, before graph capture.
    """
    from flash_rt.subgraphs.pi05 import stage_plans as _stage_plans  # noqa: F401
    frontend = getattr(target, "_pipe", target)
    if hasattr(frontend, "record_infer_graph"):
        register_capture_hook(frontend, _capture_context_graph)
        return
    if hasattr(frontend, "pipeline"):
        register_frontend_capture_hook(frontend, _enable_pipeline)
        return
    raise TypeError("context_action.enable expects a VLAModel, frontend, "
                    "or pipeline")


def _enable_pipeline(pipeline: object) -> None:
    register_capture_hook(pipeline, _capture_context_graph)


def _run_context(pl: object, stream: int) -> None:
    # Pi-style VLA detail: the encoder mutates its residual stream, so refresh
    # prompt/text embeddings before the encoder stage if they are hot inputs.
    pl._copy_lang_embeds_to_encoder_x(stream=stream)
    pl.vision_encoder(stream)
    pl.transformer_encoder(stream)


def _capture_context_graph(pl: object, stream_handle: object,
                           stream_int: int) -> None:
    capture = getattr(pl, "capture_context_action_graphs", None)
    if capture is not None:
        capture(stream_handle, stream_int)
        return
    if getattr(pl, "_context_graph", None) is not None:
        return
    graph = capture_graph(
        pl, stream_handle, lambda: _run_context(pl, stream_int))

    pl._context_graph = graph
    register_captured_graph(pl, "context", graph, exec_name="pi05_context",
                            stream=0, variants=(0,))


__all__ = ["enable"]

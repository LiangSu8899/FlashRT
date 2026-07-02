"""Shared runtime-export producer for the Pi0.5 pipelines (FP8 and FP16).

Both pipeline classes expose the same capture surface (``_graph`` /
``_decoder_only_graph`` / ``_graph_stream`` / ``bufs``), so one producer
packages either of them as an ``frt_runtime_export_v1``
(see docs/runtime_contract.md). The pipelines delegate here from
``_exec_lazy_init`` (env opt-in replay routing) and ``export_runtime``.
"""

from __future__ import annotations


def exec_enable(pl) -> None:
    """Create the exec ctx/graphs for a captured pipeline and adopt any
    already-captured graph execs (idempotent)."""
    if getattr(pl, "_exec_enabled", False):
        return
    from flash_rt.runtime import exec as _frt
    pl._exec_ctx = _frt.Ctx()
    pl._exec_gs_id = pl._exec_ctx.wrap_stream(int(pl._graph_stream.value))
    pl._exec_full = pl._exec_ctx.graph("pi05_infer", 1)
    pl._exec_dec = pl._exec_ctx.graph("pi05_decode_only", 1)
    if getattr(pl, "_graph", None) is not None:
        pl._exec_full.adopt(0, pl._graph._graph_exec.value)
    if getattr(pl, "_decoder_only_graph", None) is not None:
        pl._exec_dec.adopt(0, pl._decoder_only_graph._graph_exec.value)
    pl._exec_enabled = True
    pl._use_exec = True
    pl._exec_inited = True


def export_runtime(pl, identity=None, extra_regions=None):
    """Package a captured Pi0.5 pipeline as an ``frt_runtime_export_v1``.

    Returns a :class:`flash_rt.runtime.export.RuntimeExport` whose ``ptr`` a
    native consumer (e.g. a capsule/state host) adopts. Requires
    ``record_infer_graph()`` first; enables exec routing if the env opt-in was
    not set (exporting implies exec replay).

    - ``identity``: extra canonical identity pairs (dict, emitted in order).
      Production deployments should pass a weights digest; the structural
      identity (precision flags, graph names, region layout) is included
      automatically.
    - ``extra_regions``: additional restorable-state regions as
      ``(name, CudaBuffer, offset, nbytes)`` tuples appended after the
      default rollout boundary (``diffusion_noise`` — the region set
      validated by serving/robot_recap/verify_capsule.py).
    """
    parts = _parts(pl, identity, extra_regions)
    from flash_rt.runtime import export as _rt

    return _rt.build_export(
        pl._exec_ctx,
        streams=parts["streams"],
        graphs=parts["graphs"],
        buffers=parts["buffers"],
        regions=parts["regions"],
        identity=parts["identity"],
        owner=parts["owner"],
    )


def export_model_runtime(pl, identity=None, extra_regions=None):
    """Package a captured Pi0.5 pipeline as an ``frt_model_runtime_v1``.

    The Python-producer face mirrors the native one: the frontend already
    delivers normalized device tensors, so every input is a SWAP window
    (raw bytes, no staged transform in the loop):

      images    IN  SWAP  the normalized observation tensor window
      noise     IN  SWAP  the diffusion seed (also the action output window)
      encoder_x IN  SWAP  the encoder residual-stream/prompt slot
      actions   OUT SWAP  raw bf16 action chunk (= diffusion_noise after step)

    Prompt staging (text -> embeds) stays with the frontend / the native
    tokenizer producer. ``step`` replays the infer graph; the decode_only
    graph remains addressable through the export for temporal-caching hosts.
    """
    parts = _parts(pl, identity, extra_regions)
    from flash_rt.runtime import export as _rt

    wrap = parts["wrap"]
    num_views = int(getattr(pl, "num_views", 0) or 0)
    chunk = wrap["diffusion_noise"].nbytes() // (32 * 2)  # (chunk, 32) bf16
    ports = [
        _rt.PortSpec("images", "image", "bf16", "nhwc", "in", "swap",
                     required=True, shape=(num_views, 224, 224, 3),
                     cadence_hz=30,
                     buffer=wrap["observation_images_normalized"]),
        _rt.PortSpec("noise", "tensor", "bf16", "flat", "in", "swap",
                     shape=(chunk, 32), buffer=wrap["diffusion_noise"]),
        _rt.PortSpec("encoder_x", "state", "bf16", "flat", "in", "swap",
                     buffer=wrap["encoder_x"]),
        _rt.PortSpec("actions", "action", "bf16", "flat", "out", "swap",
                     shape=(chunk, 32), buffer=wrap["diffusion_noise"]),
    ]
    stages = [_rt.StageSpec("infer")]

    def step():
        return pl._exec_full.replay(0, pl._exec_gs_id)

    return _rt.build_model_runtime(
        pl._exec_ctx,
        streams=parts["streams"],
        graphs=parts["graphs"],
        buffers=parts["buffers"],
        regions=parts["regions"],
        ports=ports,
        stages=stages,
        identity=parts["identity"],
        owner=parts["owner"],
        step=step,
    )


def _parts(pl, identity, extra_regions):
    """Shared assembly for the plain export and the model-runtime export."""
    if getattr(pl, "_graph", None) is None:
        raise RuntimeError("export_runtime requires record_infer_graph() first")
    exec_enable(pl)
    from flash_rt.runtime import export as _rt

    ctx = pl._exec_ctx
    # Wrap the pipeline-owned IO buffers as frt buffers (metadata only; the
    # pipeline keeps ownership and the export anchors the pipeline).
    wrap = {
        name: ctx.wrap(name, pl.bufs[name].ptr.value, pl.bufs[name].nbytes)
        for name in ("observation_images_normalized", "diffusion_noise",
                     "encoder_x")
    }

    streams = [_rt.StreamSpec(
        "main", pl._exec_gs_id,
        native_handle=int(pl._graph_stream.value or 0))]
    graphs = [
        _rt.GraphSpec("infer", pl._exec_full, 0, (0,)),
        _rt.GraphSpec("decode_only", pl._exec_dec, 0, (0,)),
    ]
    buffers = [
        _rt.BufferSpec("observation_images_normalized",
                       wrap["observation_images_normalized"], "input"),
        _rt.BufferSpec("diffusion_noise", wrap["diffusion_noise"],
                       ("input", "output")),
        _rt.BufferSpec("encoder_x", wrap["encoder_x"], ("input", "state")),
    ]
    regions = [_rt.RegionSpec("rollout_boundary", wrap["diffusion_noise"])]
    anchored = [wrap]
    for name, buf, offset, nbytes in (extra_regions or ()):
        fb = ctx.wrap(name, buf.ptr.value + offset, nbytes)
        anchored.append(fb)
        regions.append(_rt.RegionSpec(name, fb))

    ident = {
        "model": "pi05",
        "pipeline": type(pl).__name__,
        "use_fp8": str(bool(getattr(pl, "use_fp8", False))),
        "use_int8_decoder": str(bool(getattr(pl, "use_int8_decoder", False))),
        "num_views": str(getattr(pl, "num_views", "")),
        "max_prompt_len": str(getattr(pl, "max_prompt_len", "")),
    }
    ident.update({str(k): str(v) for k, v in (identity or {}).items()})

    return {
        "wrap": wrap,
        "streams": streams,
        "graphs": graphs,
        "buffers": buffers,
        "regions": regions,
        "identity": ident,
        "owner": (pl, anchored),
    }

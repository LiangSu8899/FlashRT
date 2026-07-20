"""Shared runtime-export producer for the Pi0.5 pipelines (FP8 and FP16).

Both pipeline classes expose the same capture surface (``_graph`` /
``_decoder_only_graph`` / ``_graph_stream`` / ``bufs``), so one producer
packages either of them as an ``frt_runtime_export_v1``
(see docs/runtime_contract.md). The pipelines delegate here from
``_exec_lazy_init`` (env opt-in replay routing) and ``export_runtime``.
"""

from __future__ import annotations

import hashlib


def exec_enable(pl) -> None:
    """Create the exec ctx/graphs for a captured pipeline and adopt any
    already-captured graph execs (idempotent)."""
    if getattr(pl, "_exec_enabled", False):
        return
    from flash_rt.runtime import exec as _frt
    pl._exec_ctx = _frt.Ctx()
    pl._exec_gs_id = pl._exec_ctx.wrap_stream(int(pl._graph_stream.value))
    pl._exec_full = pl._exec_ctx.graph("pi05_infer", 1)
    has_decode_only = getattr(pl, "_decoder_only_graph", None) is not None
    pl._exec_dec = (
        pl._exec_ctx.graph("pi05_decode_only", 1)
        if has_decode_only else None)
    if getattr(pl, "_graph", None) is not None:
        pl._exec_full.adopt(0, pl._graph._graph_exec.value)
    if has_decode_only:
        pl._exec_dec.adopt(0, pl._decoder_only_graph._graph_exec.value)
    from flash_rt.subgraphs.capture import materialize_captured_graphs
    materialize_captured_graphs(pl)
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
    parts = _parts(
        pl, identity, extra_regions,
        robot_action_dim=_resolve_robot_action_dim(
            pl, None, required=False))
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


def export_model_runtime(pl, identity=None, extra_regions=None,
                         stage_plan="full", io="python",
                         stage_plan_kwargs=None, robot_action_dim=None,
                         state_dim=None, native_overlay=None):
    """Package a captured Pi0.5 pipeline as an ``frt_model_runtime_v1``.

    ``io="python"`` mirrors the Python frontend: the frontend already delivers
    normalized device tensors, so every input is a SWAP window (raw bytes, no
    staged transform in the loop):

      images    IN  SWAP  the normalized observation tensor window
      noise     IN  SWAP  the diffusion seed (also the action output window)
      encoder_x IN  SWAP  the encoder residual-stream/prompt-embedding slot
                          (TENSOR — STATE is reserved for real proprioception)
      actions   OUT SWAP  raw bf16 action chunk (= diffusion_noise after step)

    ``io="native"`` publishes the native C++ runtime face over the same
    captured graphs: images/actions are STAGED, noise is SWAP, and
    ``native_overlay`` atomically installs the C++ verbs before the runtime is
    returned. The callback receives the temporary declaration pointer and must
    return the non-zero pointer produced by
    ``frt_pi05_model_runtime_create_over``.

    ``io="native_v2"`` extends that face with prompt/state STAGED ports for
    fixed state-prompt deployments. The declaration is intended to be consumed
    through the C++ verb overlay; port/window/region changes are part of the
    export identity and therefore intentionally change the fingerprint.

    Prompt staging (text -> embeds) stays with the frontend / the native
    tokenizer producer. ``stage_plan`` defaults to the full infer graph; an
    explicit StagePlan or registered plan name may select already-captured
    graphs from this export. ``stage_plan_kwargs`` are passed only to
    registered plan factories, for deployment-specific graph cuts. Native
    declarations require ``robot_action_dim`` from checkpoint metadata;
    ``native_v2`` also requires the corresponding ``state_dim``. These
    deployment dimensions are deliberately not owned by the model pipeline.
    """
    native_io = io in ("native", "native_v2")
    if native_io and native_overlay is None:
        raise ValueError(
            f"io={io!r} requires an immediate native_overlay; "
            "a STAGED declaration cannot be published without real verbs")
    robot_action_dim = _resolve_robot_action_dim(
        pl, robot_action_dim, required=io in ("native", "native_v2"))
    state_dim = _resolve_state_dim(
        pl, state_dim, required=io == "native_v2")
    identity_for_parts = identity
    if io == "native_v2":
        _require_native_v2_ready(pl, state_dim)
        identity_for_parts = {
            **{str(k): str(v) for k, v in (identity or {}).items()},
            "io": "native_v2",
        }
    parts = _parts(
        pl, identity_for_parts, extra_regions,
        robot_action_dim=robot_action_dim, state_dim=state_dim)
    from flash_rt.runtime import export as _rt
    from flash_rt.subgraphs.pi05 import stage_plans as _pi05_stage_plans  # noqa: F401
    from flash_rt.subgraphs.stage_plan import resolve_stage_plan

    wrap = parts["wrap"]
    plan = resolve_stage_plan(stage_plan, model="pi05",
                              **(stage_plan_kwargs or {}))
    uses_rtc_prefix = any(
        stage.graph_name() == "decode_rtc_prefix" for stage in plan.stages)
    uses_rtc_vjp = any(
        stage.graph_name() == "decode_rtc_vjp_guided"
        for stage in plan.stages)
    num_views = int(getattr(pl, "num_views", 0) or 0)
    chunk = wrap["diffusion_noise"].nbytes() // (32 * 2)  # 2-byte action dtype
    tensor_dtype = _tensor_dtype(pl)
    if io == "python":
        ports = [
            _rt.PortSpec("images", "image", tensor_dtype, "nhwc", "in", "swap",
                         required=True, shape=(num_views, 224, 224, 3),
                         cadence_hz=30,
                         buffer=wrap["observation_images_normalized"]),
            _rt.PortSpec("noise", "tensor", tensor_dtype, "flat", "in", "swap",
                         shape=(chunk, 32), buffer=wrap["diffusion_noise"]),
            _rt.PortSpec("encoder_x", "tensor", tensor_dtype, "flat", "in", "swap",
                         buffer=wrap["encoder_x"]),
            _rt.PortSpec("actions", "action", tensor_dtype, "flat", "out", "swap",
                         shape=(chunk, 32), buffer=wrap["diffusion_noise"]),
        ]
        if uses_rtc_prefix or uses_rtc_vjp:
            ports.extend([
                _rt.PortSpec("prev_action_chunk", "tensor", tensor_dtype,
                             "flat", "in", "swap", shape=(chunk, 32),
                             buffer=wrap["rtc_prev_action_chunk"]),
                _rt.PortSpec("actions_raw", "tensor", tensor_dtype, "flat",
                             "out", "swap", shape=(chunk, 32),
                             buffer=wrap["diffusion_noise"]),
            ])
        if uses_rtc_vjp:
            ports.extend([
                _rt.PortSpec("prefix_weights", "tensor", "f32", "flat",
                             "in", "swap", shape=(chunk,),
                             buffer=wrap["rtc_prefix_weights"]),
                _rt.PortSpec("guidance_weight", "tensor", "f32", "flat",
                             "in", "swap", shape=(1,),
                             buffer=wrap["rtc_guidance_weight"]),
            ])
    elif io in ("native", "native_v2"):
        ports = [
            _rt.PortSpec("images", "image", tensor_dtype, "nhwc", "in", "staged",
                         required=True, shape=(num_views, 224, 224, 3),
                         cadence_hz=30,
                         buffer=wrap["observation_images_normalized"]),
            _rt.PortSpec("noise", "tensor", tensor_dtype, "flat", "in", "swap",
                         shape=(chunk, 32), buffer=wrap["diffusion_noise"]),
            _rt.PortSpec("actions", "action", "f32", "flat", "out",
                         "staged", shape=(chunk, robot_action_dim),
                         nbytes=chunk * robot_action_dim * 4),
        ]
        if io == "native_v2":
            ports = [
                _rt.PortSpec("prompt", "text", "u8", "flat", "in", "staged",
                             required=True, shape=(-1,)),
                _rt.PortSpec("state", "state", "f32", "flat", "in", "staged",
                             required=True, shape=(state_dim,)),
            ] + ports
            if not (uses_rtc_prefix or uses_rtc_vjp):
                ports.append(
                    _rt.PortSpec("actions_raw", "tensor", tensor_dtype,
                                 "flat", "out", "swap", shape=(chunk, 32),
                                 buffer=wrap["diffusion_noise"]))
        if uses_rtc_prefix or uses_rtc_vjp:
            ports.extend([
                _rt.PortSpec("prev_action_chunk", "tensor", tensor_dtype,
                             "flat", "in", "swap", shape=(chunk, 32),
                             buffer=wrap["rtc_prev_action_chunk"]),
                _rt.PortSpec("actions_raw", "tensor", tensor_dtype, "flat",
                             "out", "swap", shape=(chunk, 32),
                             buffer=wrap["diffusion_noise"]),
            ])
        if uses_rtc_vjp:
            ports.extend([
                _rt.PortSpec("prefix_weights", "tensor", "f32", "flat",
                             "in", "swap", shape=(chunk,),
                             buffer=wrap["rtc_prefix_weights"]),
                _rt.PortSpec("guidance_weight", "tensor", "f32", "flat",
                             "in", "swap", shape=(1,),
                             buffer=wrap["rtc_guidance_weight"]),
            ])
    else:
        raise ValueError(f"unknown Pi05 model-runtime io face {io!r}")
    plan.validate(graph_names=[g.name for g in parts["graphs"]],
                  stream_names=[s.name for s in parts["streams"]])
    graph_stream = {g.name: g.stream for g in parts["graphs"]}
    for stage in plan.stages:
        if graph_stream[stage.graph_name()] != stage.stream:
            raise ValueError(
                f"stage {stage.name!r} declares stream {stage.stream!r}, "
                f"but graph {stage.graph_name()!r} was exported on "
                f"{graph_stream[stage.graph_name()]!r}")
    stages = plan.to_stage_specs(_rt)

    def step():
        rc = 0
        graph_by_name = {g.name: g.graph for g in parts["graphs"]}
        stream_by_name = {s.name: s.stream_id for s in parts["streams"]}
        for stage in plan.stages:
            graph = graph_by_name[stage.graph_name()]
            stream_id = stream_by_name[stage.stream]
            rc = graph.replay(0, stream_id)
            if rc != 0:
                return rc
        return rc

    manifest_extra = {"stage_plan": plan.manifest(), "io": io}
    if io == "native_v2":
        manifest_extra["prompt"] = {
            "state_prompt_mode": "fixed",
            "max_prompt_len": int(getattr(pl, "max_prompt_len", 0) or 0),
            "state_dim": state_dim,
        }
    if not native_io:
        return _rt.build_model_runtime(
            pl._exec_ctx,
            streams=parts["streams"],
            graphs=parts["graphs"],
            buffers=parts["buffers"],
            regions=parts["regions"],
            ports=ports,
            stages=stages,
            identity=parts["identity"],
            manifest_extra=manifest_extra,
            owner=parts["owner"],
            step=step,
        )

    # The declaration exists only inside this call. Placeholder callbacks make
    # the generic builder's STAGED invariant true while the model-private C++
    # overlay is installed; they can never escape to a consumer.
    declaration = _rt.build_model_runtime(
        pl._exec_ctx,
        streams=parts["streams"],
        graphs=parts["graphs"],
        buffers=parts["buffers"],
        regions=parts["regions"],
        ports=ports,
        stages=stages,
        identity=parts["identity"],
        manifest_extra=manifest_extra,
        owner=parts["owner"],
        set_input=lambda _port, _payload, _stream: -3,
        get_output=lambda _port, _stream: b"",
        step=step,
    )
    try:
        over_ptr = int(native_overlay(declaration.ptr) or 0)
        if not over_ptr:
            raise RuntimeError("Pi05 native model-runtime overlay failed")
        return _rt.ModelRuntime(
            ptr=over_ptr,
            export_ptr=declaration.export_ptr,
            fingerprint=declaration.fingerprint,
            identity=declaration.identity,
            manifest=declaration.manifest,
            _anchor=(declaration._anchor, native_overlay),
        )
    finally:
        declaration.release()


def _tensor_dtype(pl):
    """Device tensor dtype for Pi0.5 IO buffers."""
    dtype = getattr(pl, "tensor_dtype", None)
    if dtype:
        return str(dtype)
    if type(pl).__name__.endswith("FP16"):
        return "f16"
    return "bf16"


def _resolve_robot_action_dim(pl, value, *, required):
    """Logical robot action dimension exposed by the STAGED output face."""
    if value is not None:
        value = int(value)
        if value <= 0 or value > 32:
            raise ValueError("robot_action_dim must be in [1, 32]")
        return value
    try:
        q01 = pl.norm_stats["actions"]["q01"]
        if q01:
            return int(min(len(q01), 32))
    except Exception:
        pass
    if required:
        raise ValueError(
            "native model-runtime export requires robot_action_dim from "
            "the frontend checkpoint metadata")
    from flash_rt.core.utils.actions import LIBERO_ACTION_DIM
    return int(LIBERO_ACTION_DIM)


def _resolve_state_dim(pl, value, *, required):
    """Raw proprioception dimension exposed by native_v2 STATE/STAGED."""
    if value is not None:
        value = int(value)
        if value <= 0:
            raise ValueError("state_dim must be positive")
        return value
    try:
        return int(len(pl.norm_stats["state"]["q01"]))
    except Exception:
        if not required:
            return None
        raise ValueError(
            "native_v2 export requires state_dim from the frontend "
            "checkpoint metadata") from None


def _tokenizer_sha256() -> str:
    from flash_rt.utils.paligemma_tokenizer import (
        resolve_paligemma_tokenizer_path,
    )
    h = hashlib.sha256()
    with open(resolve_paligemma_tokenizer_path(), "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _require_native_v2_ready(pl, state_dim) -> None:
    mode = getattr(pl, "_state_prompt_mode", None)
    fixed = bool(getattr(pl, "_fixed_shape", False))
    if mode != "fixed" and not fixed:
        raise ValueError(
            "Pi05 native_v2 requires state_prompt_mode='fixed'")
    if int(getattr(pl, "max_prompt_len", 0) or 0) < 200:
        raise ValueError("Pi05 native_v2 requires max_prompt_len >= 200")
    if state_dim is None:
        raise ValueError("Pi05 native_v2 requires state_dim")


def _parts(pl, identity, extra_regions, *, robot_action_dim=None,
           state_dim=None):
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
                     "encoder_x", "rtc_prev_action_chunk",
                     "rtc_prefix_weights", "rtc_guidance_weight")
    }

    streams = [_rt.StreamSpec(
        "main", pl._exec_gs_id,
        native_handle=int(pl._graph_stream.value or 0))]
    graphs = [_rt.GraphSpec("infer", pl._exec_full, 0, (0,))]
    if getattr(pl, "_decoder_only_graph", None) is not None:
        graphs.append(_rt.GraphSpec("decode_only", pl._exec_dec, 0, (0,)))
    from flash_rt.subgraphs.capture import export_graph_records
    for rec in export_graph_records(pl):
        graphs.append(_rt.GraphSpec(
            name=rec.name,
            graph=rec.graph,
            default_key=rec.variants[0],
            keys=rec.variants,
            stream=rec.stream,
        ))
    buffers = [
        _rt.BufferSpec("observation_images_normalized",
                       wrap["observation_images_normalized"], "input"),
        _rt.BufferSpec("diffusion_noise", wrap["diffusion_noise"],
                       ("input", "output")),
        _rt.BufferSpec("encoder_x", wrap["encoder_x"], ("input", "state")),
        _rt.BufferSpec("rtc_prev_action_chunk",
                       wrap["rtc_prev_action_chunk"], "input"),
        _rt.BufferSpec("rtc_prefix_weights",
                       wrap["rtc_prefix_weights"], "input"),
        _rt.BufferSpec("rtc_guidance_weight",
                       wrap["rtc_guidance_weight"], "input"),
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
        "hardware": str(getattr(pl, "hardware", "")),
        "tensor_dtype": _tensor_dtype(pl),
        "use_fp8": str(bool(getattr(pl, "use_fp8", False))),
        "use_int8_decoder": str(bool(getattr(pl, "use_int8_decoder", False))),
        "num_views": str(getattr(pl, "num_views", "")),
        "max_prompt_len": str(getattr(pl, "max_prompt_len", "")),
        "chunk_size": str(getattr(pl, "chunk_size", "")),
        "model_action_dim": "32",
        "robot_action_dim": str(robot_action_dim),
    }
    ident.update({str(k): str(v) for k, v in (identity or {}).items()})
    if ident.get("io") == "native_v2":
        ident["state_prompt_mode"] = "fixed"
        ident["state_dim"] = str(state_dim)
        ident["tokenizer_sha256"] = _tokenizer_sha256()

    return {
        "wrap": wrap,
        "streams": streams,
        "graphs": graphs,
        "buffers": buffers,
        "regions": regions,
        "identity": ident,
        "owner": (pl, anchored),
    }

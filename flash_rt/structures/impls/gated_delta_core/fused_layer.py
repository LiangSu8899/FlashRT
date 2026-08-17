"""Fused decode form of a whole gated-delta layer.

The transformers fallback runs this layer as ~75 launches of Python
glue per token; serving its pieces individually is measurably negative
on a launch-bound host (a quantized projection swap *lost* throughput
here — the receipts are on the record). This impl owns the layer's
cached-decode step as one short chain of Hub kernels:

    packed in_proj GEMV -> causal_conv1d_update -> broadcast QKV split
    -> gating -> gated-delta recurrent core -> gated RMSNorm -> out_proj

A scheme may route the two projection GEMVs through the dynamic NVFP4
band (``gdn_projection_format="nvfp4_dynamic"``); the BF16 weights are
retained for prefill and detach either way.

Everything else — prefill, uncached calls, masked batches — dispatches
to the retained host layer and is counted.

Cache contract (the host's, followed not replaced): the layer reads and
writes ``cache_params.conv_states[idx]`` and ``recurrent_states[idx]``.
The host keeps the last K raw inputs in the conv state; the Hub update
kernel keeps the previous K-1, so the impl feeds ``state[..., 1:]`` and
rolls the host slot forward itself. The recurrent state slot is
normalised to a stable BF16 tensor on the first decode step and never
re-pointed after that: the core writes a scratch buffer (it cannot
write the slot it is reading within the same step) and the result is
copied back into the slot, which is what graph replay requires.
"""

from __future__ import annotations

from functools import lru_cache

import torch

from ...guard import CAST_OK, PROCEED, GuardedSeam

#: a range, not a pin: the resolver picks the newest release whose
#: build matrix covers the caller's torch/CUDA pair, and the entries
#: this structure calls have been stable since 3. Pinning the newest
#: release strands any host whose variant that release did not build.
GDA_DEP = {"provider": "hf", "repo": "flashrt/gated-delta-attention",
           "version": ">=3"}
CONV_DEP = {"provider": "hf", "repo": "flashrt/causal-conv1d-state",
            "version": ">=1"}
FUSED_DEP = {"provider": "hf", "repo": "flashrt/transformer-fused-ops",
             "version": ">=1"}


@lru_cache(maxsize=1)
def _native_ltri_inv():
    """The native batched 64x64 unit-lower-triangular inverse.

    Same forward-substitution recurrence class as the batched cuBLAS
    solve it replaces (fp32 error band matches), with the identity/
    tril preparation folded in — the eye/expand/tril materializations
    disappear with the solve. Registered as a torch custom op with a
    fake so the compiled prefill traces through it. Absence is not a
    refusal: the cuBLAS solve path keeps serving.
    """
    try:
        from flash_rt import flash_rt_kernels as _fk
    except ImportError:
        return None
    fn = getattr(_fk, "batched_unit_ltri_inv64_f32", None)
    if fn is None:
        return None

    from torch.library import custom_op

    @custom_op("flashrt_native::unit_ltri_inv64", mutates_args=())
    def _op(big_a: torch.Tensor) -> torch.Tensor:
        flat = big_a.reshape(-1, 64, 64).contiguous()
        x = torch.empty_like(flat)
        rc = fn(flat.data_ptr(), x.data_ptr(), flat.shape[0],
                torch.cuda.current_stream().cuda_stream)
        if rc != 0:
            raise RuntimeError(
                f"unit_ltri_inv64 refused rc={rc} for "
                f"B={flat.shape[0]}")
        return x.reshape(big_a.shape)

    @_op.register_fake
    def _(big_a):
        return torch.empty_like(big_a)

    return _op


import os as _os

_NCP_ON = _os.environ.get("FRT_WY_NCP_V2", "1") != "0"


@lru_cache(maxsize=1)
def _native_gated_norm_quant():
    """The gated norm that also emits its consumer's NVFP4 input.

    The output projection is the norm's only consumer and quantizes
    what it receives; the quantizer's blocks tile a head's lanes
    exactly, so the whole step fits in the block that produced the
    row. Measured bit-identical (normed, packed and scales alike)
    against the packaged norm followed by the production quantize,
    at 2.9x of the pair.
    """
    if _os.environ.get("FRT_GDN_NORMQUANT", "1") == "0":
        return None
    # hub artifact first: its entry is a torch op with a fake, so a
    # host that compiles this call traces it unaided and a process
    # that cannot load our native extension still gets the producer
    from flash_rt.structures.impls import hub_kernel
    try:
        hub = hub_kernel(FUSED_DEP["repo"], FUSED_DEP["version"])
    except Exception:  # noqa: BLE001 — absence is not a refusal
        hub = None
    hub_fn = getattr(hub, "rms_norm_gated_silu_quant_fp4_bf16", None)
    if hub_fn is not None:
        def _hub_entry(x, gate, weight, out, packed, sfa, eps):
            hub_fn(x, gate, weight, eps=eps, out=out, packed=packed,
                   sfa=sfa)
        return _hub_entry
    try:
        from flash_rt import flash_rt_kernels as _fk
    except ImportError:
        return None
    fn = getattr(_fk, "rms_norm_gated_silu_quant_fp4_bf16", None)
    if fn is None:
        return None

    from torch.library import custom_op

    @custom_op("flashrt_native::gated_norm_quant",
               mutates_args=("out", "packed", "sfa"))
    def _op(x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor,
            out: torch.Tensor, packed: torch.Tensor, sfa: torch.Tensor,
            eps: float) -> None:
        m, d = int(x.shape[0]), int(x.shape[1])
        rc = fn(x.data_ptr(), gate.data_ptr(), weight.data_ptr(),
                out.data_ptr(), packed.data_ptr(), sfa.data_ptr(),
                m, d, float(eps),
                torch.cuda.current_stream().cuda_stream)
        if rc != 0:
            raise RuntimeError(
                f"gated_norm_quant refused rc={rc} M={m} D={d}")

    @_op.register_fake
    def _(x, gate, weight, out, packed, sfa, eps):
        return None

    return _op


@lru_cache(maxsize=1)
def _native_recurrent_stream():
    """The streaming-column form of the gated-delta recurrent step.

    Bit-identical to the packaged kernel — same block reduction, same
    per-column arithmetic in the same order. The column simply streams
    in two passes instead of living in a 128-entry per-thread array
    that spills to local memory, which is where the step was paying
    DRAM for state it believed was resident. Measured 1.5x.
    """
    if _os.environ.get("FRT_GDN_STREAM", "1") == "0":
        return None
    from flash_rt.structures.impls import hub_kernel
    try:
        hub = hub_kernel(GDA_DEP["repo"], GDA_DEP["version"])
    except Exception:  # noqa: BLE001 — absence is not a refusal
        hub = None
    hub_fn = getattr(hub, "gdn_recurrent_inout_stream_bf16", None)
    if hub_fn is not None:
        def _hub_entry(q, k, v, g, beta, state_in, state_out, out):
            return hub_fn(q, k, v, g, beta, state_in,
                          use_qk_l2norm=True, state_out=state_out,
                          out=out)
        return _hub_entry
    try:
        from flash_rt import flash_rt_kernels as _fk
    except ImportError:
        return None
    fn = getattr(_fk, "gdn_recurrent_inout_stream_bf16", None)
    if fn is None:
        return None

    from torch.library import custom_op

    @custom_op("flashrt_native::gdn_recurrent_stream",
               mutates_args=("state_out", "out"))
    def _op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
            g: torch.Tensor, beta: torch.Tensor,
            state_in: torch.Tensor, state_out: torch.Tensor,
            out: torch.Tensor) -> None:
        h, d = int(q.shape[1]), int(q.shape[2])
        rc = fn(q.data_ptr(), k.data_ptr(), v.data_ptr(), g.data_ptr(),
                beta.data_ptr(), state_in.data_ptr(),
                state_out.data_ptr(), out.data_ptr(), 1, h, d, True,
                torch.cuda.current_stream().cuda_stream)
        if rc != 0:
            raise RuntimeError(
                f"gdn_recurrent_stream refused rc={rc} H={h} D={d}")

    @_op.register_fake
    def _(q, k, v, g, beta, state_in, state_out, out):
        return None

    def _entry(q, k, v, g, beta, state_in, state_out, out):
        _op(q.contiguous(), k.contiguous(), v.contiguous(),
            g.contiguous(), beta.contiguous(), state_in, state_out,
            out)
        return out, state_out

    return _entry


@lru_cache(maxsize=1)
def _native_conv_steps_gqa():
    """The step-batched conv1d update with GQA split outputs.

    The packaged chunk-parallel conv reads every input element K times
    from DRAM (one token per thread); this arm rolls the taps through
    registers over 8 consecutive tokens with the packaged tap order and
    fma chain — bit-exact — and writes the q/k/v splits directly.
    Fixed 2048/2048/6144, K=4 family only.
    """
    try:
        from flash_rt import flash_rt_kernels as _fk
    except ImportError:
        return None
    fn = getattr(_fk, "causal_conv1d_update_steps_gqa_bf16", None)
    if fn is None:
        return None

    from torch.library import custom_op

    @custom_op("flashrt_native::conv1d_steps_gqa", mutates_args=())
    def _op(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor,
            state: torch.Tensor) -> list[torch.Tensor]:
        # bias is required: the Optional-tensor marshalling was
        # measured to corrupt the following pointer operands; a
        # bias-free host passes an explicit zeros row (acc starts at
        # 0.0 either way - bit-identical to the null-bias path)
        s = x.shape[0]
        q16 = torch.empty((s, 16, 128), device=x.device,
                          dtype=torch.bfloat16)
        k16 = torch.empty((s, 16, 128), device=x.device,
                          dtype=torch.bfloat16)
        v48 = torch.empty((s, 48, 128), device=x.device,
                          dtype=torch.bfloat16)
        rc = fn(x.data_ptr(), w.data_ptr(), bias.data_ptr(),
                state.data_ptr(), q16.data_ptr(), k16.data_ptr(),
                v48.data_ptr(), s, True,
                torch.cuda.current_stream().cuda_stream)
        if rc != 0:
            raise RuntimeError(f"conv1d_steps_gqa refused rc={rc}")
        return [q16, k16, v48]

    @_op.register_fake
    def _(x, w, bias, state):
        s = x.shape[0]
        return [x.new_empty((s, 16, 128)), x.new_empty((s, 16, 128)),
                x.new_empty((s, 48, 128))]

    return _op


@lru_cache(maxsize=1)
def _native_norm_cumsum_pack():
    """The native v2 launch of the WY norm/pack + gate-cumsum pair.

    Math transcribed verbatim from the packaged fast arm; the gate
    cumsum is parallelized over the independent (chunk, head) pairs
    (the packaged kernel walks the whole prompt from one 64-thread
    block). Bit-exact against the packaged pair. Fixed 16/48/128/64
    family only — absence or another family keeps the packaged op.
    """
    try:
        from flash_rt import flash_rt_kernels as _fk
    except ImportError:
        return None
    fn = getattr(_fk, "gdn_wy_norm_cumsum_pack_qk_v2_bf16", None)
    if fn is None:
        return None

    from torch.library import custom_op

    @custom_op("flashrt_native::gdn_wy_ncp_v2", mutates_args=())
    def _op(q16: torch.Tensor, k16: torch.Tensor,
            g: torch.Tensor) -> list[torch.Tensor]:
        s = q16.shape[0]
        c = (s + 63) // 64
        q16_l2 = torch.empty_like(q16)
        k16_l2 = torch.empty_like(k16)
        q_pack_hv = torch.empty((c, 48, 64, 128), device=q16.device,
                                dtype=q16.dtype)
        k_pack_hk = torch.empty((c, 16, 64, 128), device=q16.device,
                                dtype=q16.dtype)
        g_cumsum = torch.empty_like(g)
        rc = fn(q16.data_ptr(), k16.data_ptr(), g.data_ptr(),
                q16_l2.data_ptr(), k16_l2.data_ptr(),
                q_pack_hv.data_ptr(), k_pack_hk.data_ptr(),
                g_cumsum.data_ptr(), s,
                torch.cuda.current_stream().cuda_stream)
        if rc != 0:
            raise RuntimeError(f"gdn_wy_ncp_v2 refused rc={rc} S={s}")
        return [q16_l2, k16_l2, q_pack_hv, k_pack_hk, g_cumsum]

    @_op.register_fake
    def _(q16, k16, g):
        s = q16.shape[0]
        c = (s + 63) // 64
        return [torch.empty_like(q16), torch.empty_like(k16),
                q16.new_empty((c, 48, 64, 128)),
                q16.new_empty((c, 16, 64, 128)),
                torch.empty_like(g)]

    def _entry(q16, k16, g):
        if (q16.shape[1:] != (16, 128)
                or g.shape[-1] != 48):
            return None
        return _op(q16.contiguous(), k16.contiguous(), g.contiguous())

    return _entry


def _wy_ai_inverse(big_a: torch.Tensor) -> torch.Tensor:
    """inv(I + strict_tril(A)) for the WY chain, batched 64x64 fp32."""
    import os
    if (big_a.dtype is torch.float32
            and big_a.shape[-2:] == (64, 64)
            and os.environ.get("FRT_TRSM_NATIVE", "1") != "0"):
        inv = _native_ltri_inv()
        if inv is not None:
            return inv(big_a)
    eye = torch.eye(64, device=big_a.device,
                    dtype=big_a.dtype).expand_as(big_a).contiguous()
    return torch.linalg.solve_triangular(
        eye + torch.tril(big_a, -1), eye, upper=False)


@lru_cache(maxsize=1)
def _native_stash_op():
    """The native per-row-stash arm of the from-conv chunk core.

    Registered lazily as a mutating torch custom op so the compiled
    and captured spec-verify passes trace through it. Absence of the
    native build is not a refusal — the plain hub chunk kernel keeps
    serving, and rejected rounds re-drive the state sublayers instead
    of selecting a stash row.
    """
    global _STASH_OP
    if _STASH_OP is not None:
        return _STASH_OP if _STASH_OP is not False else None
    try:
        from flash_rt import flash_rt_kernels as _fk
        fn = getattr(_fk, "gdn_chunk_from_conv_smem_h_stash_bf16", None)
    except ImportError:
        fn = None
    if fn is None:
        _STASH_OP = False
        return None

    @torch.library.custom_op(
        "flashrt_native::gdn_chunk_stash",
        mutates_args=("state", "out", "stash"))
    def _op(conv_out: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
            neg_exp_a: torch.Tensor, dt_bias: torch.Tensor,
            state: torch.Tensor, out: torch.Tensor,
            stash: torch.Tensor, num_v_heads: int, num_k_heads: int,
            head_dim: int) -> None:
        fn(conv_out.data_ptr(), a.data_ptr(), b.data_ptr(),
           neg_exp_a.data_ptr(), dt_bias.data_ptr(), state.data_ptr(),
           out.data_ptr(), stash.data_ptr(), conv_out.shape[0],
           num_v_heads, num_k_heads, head_dim, a.stride(0), b.stride(0),
           True, torch.cuda.current_stream().cuda_stream)

    @_op.register_fake
    def _(conv_out, a, b, neg_exp_a, dt_bias, state, out, stash,
          num_v_heads, num_k_heads, head_dim):
        return None

    _STASH_OP = _op
    return _op


_STASH_OP = None


def _packages():
    from flash_rt.structures.impls import hub_kernel

    gda = hub_kernel(GDA_DEP["repo"], GDA_DEP["version"])
    conv = hub_kernel(CONV_DEP["repo"], CONV_DEP["version"])
    fused = hub_kernel(FUSED_DEP["repo"], FUSED_DEP["version"])
    for pkg, name in ((gda, "lin_split_qkv_broadcast_bf16"),
                      (gda, "gdn_gating_bf16"),
                      (gda, "gated_delta_recurrent_inout_bf16"),
                      (conv, "causal_conv1d_update_bf16"),
                      (fused, "rms_norm_gated_silu_bf16")):
        if not hasattr(pkg, name):
            raise ValueError(
                f"refused: installed build lacks {name}; a release "
                "carrying the fused decode chain is required")
    return gda, conv, fused


class FusedGatedDeltaDecodeLayer(GuardedSeam, torch.nn.Module):
    """Drop-in replacement for one gated-delta layer module."""

    _frt_host_attr = "host_layer"
    _frt_can_fallback = True

    def __init__(self, host, layer_idx: int,
                 projection_format: str | None = None):
        super().__init__()
        gda, conv, fused = _packages()
        self._gda, self._conv, self._fused = gda, conv, fused
        self.host_layer = host
        self._idx = int(layer_idx)
        self._hv = int(host.num_v_heads)
        self._hk = int(host.num_k_heads)
        self._d = int(host.head_v_dim)
        if (self._d != 128 or int(host.head_k_dim) != 128
                or self._hv <= 0 or self._hk <= 0
                or self._hv % self._hk):
            raise ValueError(
                "fused decode chain serves D=128 profiles whose v-head "
                "count is a multiple of the k-head count; other "
                "profiles keep the host layer")
        # the original 48/16 profile keeps its dedicated entries,
        # byte-for-byte; every other profile routes the head-generic
        # entries, whose absence from an older build is a clean bind
        # refusal (the ladder falls back to the callable-slot rule)
        if (self._hv, self._hk) == (48, 16):
            self._split_fn = gda.lin_split_qkv_broadcast_bf16
            self._gate_fn = gda.gdn_gating_bf16
            self._chunk_name = "gdn_chunk_from_conv_smem_bf16"
        else:
            for name in ("lin_split_qkv_broadcast_h_bf16",
                         "gdn_gating_h_bf16"):
                if not hasattr(gda, name):
                    raise ValueError(
                        f"refused: installed build lacks {name}; the "
                        f"{self._hv}/{self._hk}-head profile needs the "
                        "head-generic chain entries")
            hv, hk, d = self._hv, self._hk, self._d

            def _split(conv_out):
                return gda.lin_split_qkv_broadcast_h_bf16(
                    conv_out, hv, hk, d)

            def _gate(a, b, neg_exp_a, dt_bias):
                return gda.gdn_gating_h_bf16(a, b, neg_exp_a, dt_bias,
                                             num_heads=hv)

            self._split_fn = _split
            self._gate_fn = _gate
            self._chunk_name = "gdn_chunk_from_conv_smem_h_bf16"
        dev = host.in_proj_qkv.weight.device
        # the four input projections read the same activation; packing
        # them row-wise turns four GEMV launches into one, bit-identical
        # per output row. The host projections are rebound onto views of
        # the packed rows so the layer still carries one copy of these
        # weights — prefill and detach see the exact same values.
        self._packed_w = torch.cat(
            [host.in_proj_qkv.weight.detach(),
             host.in_proj_z.weight.detach(),
             host.in_proj_b.weight.detach(),
             host.in_proj_a.weight.detach()], dim=0)
        self._splits = []
        off = 0
        for name in ("in_proj_qkv", "in_proj_z", "in_proj_b",
                     "in_proj_a"):
            lin = getattr(host, name)
            n = int(lin.weight.shape[0])
            lin.weight = torch.nn.Parameter(
                self._packed_w[off:off + n],
                requires_grad=lin.weight.requires_grad)
            self._splits.append((off, off + n))
            off += n
        self._conv_w = host.conv1d.weight.detach().squeeze(1).contiguous()
        self._conv_b = (host.conv1d.bias.detach().contiguous()
                        if host.conv1d.bias is not None else None)
        # dense bias operand for the step-batched conv arm (required
        # tensor; zeros reproduce the null-bias accumulator exactly)
        self._conv_b_dense = (self._conv_b if self._conv_b is not None
                              else torch.zeros_like(self._conv_w[:, 0])
                              .contiguous())
        self._neg_exp_a = (-host.A_log.detach().float().exp()).contiguous()
        self._dt_bias = host.dt_bias.detach().float().contiguous()
        self._eps = float(getattr(host.norm, "variance_epsilon",
                                  getattr(host.norm, "eps", 1e-6)))
        d_model = int(host.in_proj_qkv.weight.shape[1])
        # optional W4A4 decode band on the two projection GEMVs — a
        # scheme decision, never a default. The BF16 weights (and the
        # host views into them) are retained: prefill and detach stay
        # exact, only the decode band changes representation. A refusal
        # (missing package, unqualified shape) degrades to the BF16
        # band for this layer and is counted, not raised.
        self._proj_in = self._proj_out = None
        if projection_format == "nvfp4_dynamic":
            from ..linear_proj import nvfp4_dynamic
            try:
                self._proj_in, rel_in = nvfp4_dynamic.bind_proj_seam(
                    {"w": self._packed_w})
                self._proj_out, rel_out = nvfp4_dynamic.bind_proj_seam(
                    {"w": host.out_proj.weight.detach()})
                self._proj_rel = (rel_in, rel_out)
            except ValueError:
                self._proj_in = self._proj_out = None
        elif projection_format == "nvfp4_balance":
            # balanced fold: same FP4 band, with the per-input-channel
            # balance fitted on calibrated activation amax — the lever
            # that cut the single-seat tail error 2.5x on this host.
            # Calibration is a precondition, not a default: a host layer
            # without the attached amax keeps the BF16 band (counted as
            # a refusal, never raised), and the amax only ever comes
            # from calibrate_gdn_channel_amax's real-forward statistics.
            from ..linear_proj import nvfp4_balance
            amax = getattr(host, "_frt_gdn_channel_amax", None)
            if amax is not None:
                try:
                    self._proj_in = nvfp4_balance.bind_proj_seam(
                        {"w": self._packed_w}, channel_amax=amax["in"])
                    self._proj_out = nvfp4_balance.bind_proj_seam(
                        {"w": host.out_proj.weight.detach()},
                        channel_amax=amax["out"])
                except ValueError:
                    self._proj_in = self._proj_out = None
        elif projection_format is not None:
            raise ValueError(
                f"refused: unknown gdn projection format "
                f"{projection_format!r}")
        self._state_a = torch.empty(1, self._hv, self._d, self._d,
                                    device=dev, dtype=torch.bfloat16)
        self._core_out = torch.empty(1, self._hv, self._d, device=dev,
                                     dtype=torch.bfloat16)
        self._gn_packed = self._gn_sfa = self._gn_normed = None
        self._norm_w = None
        # prefill chain needs the chunk entries; their absence is not a
        # bind refusal — prompts simply keep the host form
        self._chunk_ok = (
            hasattr(conv, "causal_conv1d_update_chunk_parallel_bf16")
            and hasattr(gda, self._chunk_name))
        # the WY pipeline is the prompt-length core: one fused chain
        # over all chunks, state carried inside the kernels — the
        # serial per-chunk walk of the fallback core is the measured
        # long-prompt TTFT term. Its entries are 48/16-shaped; other
        # profiles keep the chunk walk until head-generic WY ships.
        self._wy_h = (self._hv, self._hk) != (48, 16)
        _wy_names = ((
            "gdn_wy_norm_cumsum_pack_qk_h_bf16",
            "gdn_wy_kkt_b64_h_bf16",
            "gdn_wy_cast_ai_h_f32_to_bf16",
            "gdn_wy_recompute_wu_b64_mma_fla_h_bf16",
            "gdn_wy_chunk_h_b64_mma_fla_h_bf16",
            "gdn_wy_output_o_b64_mma_fla_h_bf16",
        ) if self._wy_h else (
            "lin_split_qkv_gqa_bf16",
            "gdn_wy_norm_cumsum_pack_qk_bf16",
            "gdn_wy_kkt_b64_bf16",
            "gdn_wy_cast_ai_f32_to_bf16",
            "gdn_wy_recompute_wu_b64_mma_fla_bf16",
            "gdn_wy_chunk_h_b64_mma_fla_bf16",
            "gdn_wy_output_o_b64_mma_fla_bf16",
        ))
        self._wy_ok = (self._chunk_ok
                       and all(hasattr(gda, n) for n in _wy_names))
        guard = self._frt_arm(dtypes=CAST_OK, device=dev, k=d_model)
        guard.notes["host_form_calls"] = 0
        guard.notes["proj_band"] = ("nvfp4" if self._proj_in is not None
                                    else "bf16")

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "host_layer":
                raise
            return getattr(super().__getattr__("host_layer"), name)

    def _host_form(self, *args, **kwargs):
        if getattr(self, "_released", False):
            raise ValueError(
                "refused: host projection weights were released "
                "(one-way band); this call shape has no host fallback")
        guard = self._frt_guard
        if guard is not None and not torch.compiler.is_compiling():
            guard.notes["host_form_calls"] += 1
        return self.host_layer(*args, **kwargs)

    @torch.no_grad()
    def frt_enable_stash(self, rows, cache_params):
        """Arm the per-row state stash for spec-verify passes.

        Called eagerly by the speculative runner before any compiled
        pass traces: the buffers exist up front, so the compiled chunk
        branch is straight-line. Returns False (and arms nothing) when
        the native build does not carry the stash kernel.
        """
        hub = getattr(self._gda, "gdn_chunk_from_conv_smem_stash_bf16",
                      None)
        self._stash_hub = hub
        if hub is None and _native_stash_op() is None:
            return False
        if not self._chunk_ok or self._chunk_name not in (
                "gdn_chunk_from_conv_smem_bf16",
                "gdn_chunk_from_conv_smem_h_bf16"):
            return False
        dev = self._conv_w.device
        self._stash_rec = torch.empty(
            rows, self._hv, self._d, self._d, device=dev,
            dtype=torch.bfloat16)
        self._stash_mixed = torch.empty(
            rows, self._conv_w.shape[0], device=dev,
            dtype=torch.bfloat16)
        reg = getattr(cache_params, "frt_stash_layers", None)
        if reg is None:
            reg = {}
            cache_params.frt_stash_layers = reg
        reg[self._idx] = self
        return True

    def _prefill_chain(self, hidden_states, cache_params):
        """Whole-prompt form: conv chunk + fused gating/split/recurrent.

        Chunks of 64 carry the conv state and the recurrent state
        forward in place, so any prompt length runs through the same
        two kernels per chunk. Larger slabs are on the record as a
        negative: at S>64 the chunk kernel's internal combine is not
        run-to-run stable (the repeat gate caught it) and the latency
        win measured under three percent — the fixed-order 64 chunk is
        the contract. Both host cache slots are written with the
        host's own semantics (last-K raw inputs; final state).
        """
        host = self.host_layer
        S = hidden_states.shape[1]
        # hand the seam the caller's tensor, not a view of it: an
        # upstream producer that pre-quantized this activation keys the
        # handoff on tensor identity, and a .view here would break it
        allp = (self._proj_in(hidden_states).view(S, -1)
                if self._proj_in is not None
                else torch.nn.functional.linear(
                    hidden_states.view(S, -1), self._packed_w))
        (q0, q1), (z0, z1), (b0, b1), (a0, a1) = self._splits
        mixed = allp[:, q0:q1].contiguous()
        a_all = allp[:, a0:a1].contiguous()
        b_all = allp[:, b0:b1].contiguous()
        kk = self._conv_w.shape[-1]
        # continuation (a verify batch mid-stream) seeds from the live
        # slots; a fresh prompt starts from zero. The signal is an
        # explicit attribute only loop-owned caches carry — host caches
        # lack it and always get prompt semantics.
        # three continuation sources, one rule: a filled conv slot means
        # mid-stream unless the cache explicitly says fresh. Hosts that
        # chunk long prompts re-enter this branch per chunk with their
        # own cache carrying state (the 2K receipts caught the zero-
        # reset); loop-owned caches say False around a fresh prompt and
        # True around a verify batch.
        flag = getattr(cache_params, "frt_continue", None)
        old_slot = cache_params.conv_states[self._idx]
        cont = torch.is_tensor(old_slot) and flag is not False
        if cont:
            conv_state = old_slot[:, :, 1:].contiguous().clone()
            state = cache_params.recurrent_states[self._idx] \
                .view(self._hv, self._d, self._d) \
                .to(torch.bfloat16).contiguous().clone()
        else:
            conv_state = torch.zeros(1, mixed.shape[1], kk - 1,
                                     device=mixed.device,
                                     dtype=mixed.dtype)
            state = torch.zeros(self._hv, self._d, self._d,
                                device=mixed.device, dtype=torch.bfloat16)
        if self._wy_ok and S > 64:
            # the WY pipeline packs the whole span up front — gigabytes
            # of transients at deep prompts. Slabs bound the working
            # set: conv_state and state carry in place across slab
            # calls exactly as they do across the 64-chunks inside, so
            # the chunk sequence (and the arithmetic) is unchanged.
            slab = 8192
            if S > slab:
                core_out = torch.empty(S, self._hv, self._d,
                                       device=mixed.device,
                                       dtype=torch.bfloat16)
                for s0 in range(0, S, slab):
                    s1 = min(s0 + slab, S)
                    core_out[s0:s1] = self._wy_core(
                        mixed[s0:s1], a_all[s0:s1], b_all[s0:s1],
                        conv_state, state, s1 - s0)
            else:
                core_out = self._wy_core(mixed, a_all, b_all,
                                         conv_state, state, S)
            return self._prefill_epilogue(
                hidden_states, cache_params, allp, mixed, core_out,
                state, cont, old_slot, S)
        stash_rec = getattr(self, "_stash_rec", None)
        if stash_rec is not None and S <= stash_rec.shape[0]:
            # spec-verify arm: same conv update, same chunk recurrence,
            # plus the per-row state stash a rejected round selects
            # from instead of re-driving this layer
            self._stash_mixed[:S].copy_(mixed)
            conv_out = self._conv.causal_conv1d_update_chunk_parallel_bf16(
                mixed.view(1, S, -1), self._conv_w, conv_state,
                self._conv_b, apply_silu=True)
            core_out = torch.empty(S, self._hv, self._d,
                                   device=mixed.device,
                                   dtype=torch.bfloat16)
            hub = getattr(self, "_stash_hub", None)
            if hub is not None:
                hub(conv_out.view(S, -1), a_all, b_all,
                    self._neg_exp_a, self._dt_bias, state,
                    self._stash_rec, num_v_heads=self._hv,
                    num_k_heads=self._hk, head_dim=self._d,
                    out=core_out)
            else:
                _native_stash_op()(
                    conv_out.view(S, -1), a_all, b_all,
                    self._neg_exp_a, self._dt_bias, state,
                    core_out.view(S, -1), self._stash_rec, self._hv,
                    self._hk, self._d)
            return self._prefill_epilogue(
                hidden_states, cache_params, allp, mixed, core_out,
                state, cont, old_slot, S)
        core_out = torch.empty(S, self._hv, self._d,
                               device=mixed.device, dtype=torch.bfloat16)
        for s0 in range(0, S, 64):
            s1 = min(s0 + 64, S)
            conv_out = self._conv.causal_conv1d_update_chunk_parallel_bf16(
                mixed[s0:s1].view(1, s1 - s0, -1), self._conv_w,
                conv_state, self._conv_b, apply_silu=True)
            if self._chunk_name.endswith("_h_bf16"):
                self._gda.gdn_chunk_from_conv_smem_h_bf16(
                    conv_out.view(s1 - s0, -1), a_all[s0:s1],
                    b_all[s0:s1], self._neg_exp_a, self._dt_bias, state,
                    num_v_heads=self._hv, num_k_heads=self._hk,
                    head_dim=self._d, use_qk_l2norm=True,
                    out=core_out[s0:s1])
            else:
                self._gda.gdn_chunk_from_conv_smem_bf16(
                    conv_out.view(s1 - s0, -1), a_all[s0:s1],
                    b_all[s0:s1], self._neg_exp_a, self._dt_bias, state,
                    use_qk_l2norm=True, out=core_out[s0:s1])
        return self._prefill_epilogue(
            hidden_states, cache_params, allp, mixed, core_out, state,
            cont, old_slot, S)

    def _wy_core(self, mixed, a_all, b_all, conv_state, state, S):
        """Whole-prompt gated-delta core: the WY pipeline, one pass.

        The conv update runs the full prompt in one launch; the WY
        chain (norm/cumsum -> KKT -> triangular solve -> WU recompute
        -> chunk-state carry -> output) keeps its chunks inside the
        kernels, carrying ``state`` in place — no serial per-chunk walk
        on the host, which is the measured long-prompt TTFT term the
        fallback core pays.
        """
        gda = self._gda
        # the GQA conv variant writes the q/k/v splits directly (same
        # channel mapping and tap order as conv -> lin_split, bit-exact)
        # and saves the full-width conv_out round trip; the head-generic
        # arm keeps the plain conv + host-side slicing
        conv_gqa = (None if self._wy_h
                    or _os.environ.get("FRT_CONV_GQA", "1") == "0"
                    else getattr(
            self._conv, "causal_conv1d_update_chunk_parallel_gqa_bf16",
            None))
        if conv_gqa is None:
            conv_out = self._conv.causal_conv1d_update_chunk_parallel_bf16(
                mixed.view(1, S, -1), self._conv_w, conv_state,
                self._conv_b, apply_silu=True)
            co = conv_out.view(S, -1)
        g, beta = self._gate_fn(
            a_all.view(S, self._hv), b_all.view(S, self._hv),
            self._neg_exp_a, self._dt_bias)
        if self._wy_h:
            return self._wy_core_h(gda, co, g, beta, state, S)
        if conv_gqa is not None:
            steps = _native_conv_steps_gqa()
            if steps is not None and mixed.shape[-1] == 10240:
                q16, k16, v48 = steps(
                    mixed.view(S, -1), self._conv_w.view(-1, 4),
                    self._conv_b_dense, conv_state.view(-1, 3))
            else:
                q16, k16, v48 = conv_gqa(
                    mixed.view(1, S, -1), self._conv_w, conv_state,
                    self._conv_b, apply_silu=True)
                q16 = q16.view(S, 16, 128)
                k16 = k16.view(S, 16, 128)
                v48 = v48.view(S, 48, 128)
        else:
            q16, k16, v48 = gda.lin_split_qkv_gqa_bf16(co)
        ncp = _native_norm_cumsum_pack() if _NCP_ON else None
        packed_qk = ncp(q16, k16, g) if ncp is not None else None
        if packed_qk is not None:
            q16_l2, k16_l2, q_pack_hv, _k_pack_hk, g_cumsum = packed_qk
        else:
            q16_l2, k16_l2, q_pack_hv, _k_pack_hk, g_cumsum = \
                gda.gdn_wy_norm_cumsum_pack_qk_bf16(q16, k16, g)
        # the wmma Gram tier replaces the scalar walk where the
        # installed artifact carries it - same signature, same A
        # layout, 32.8x on the measured long-prompt term
        kkt = getattr(gda, "gdn_wy_kkt_b64_mma_bf16", None) \
            or gda.gdn_wy_kkt_b64_bf16
        big_a = kkt(k16_l2, beta, g_cumsum)
        # inv(I + strict_tril(A)): the native fused inverse where the
        # build carries it (same fp32 forward-substitution recurrence
        # as the batched cuBLAS solve, eye/tril prep folded in), the
        # cuBLAS solve otherwise
        ai = _wy_ai_inverse(big_a).contiguous()
        ai_pack = gda.gdn_wy_cast_ai_f32_to_bf16(ai, S)
        w_pack, u_pack = gda.gdn_wy_recompute_wu_b64_mma_fla_bf16(
            k16_l2, v48, beta, g_cumsum, ai_pack)
        h0, _v_new, v_new_pack, k_pack_hv = \
            gda.gdn_wy_chunk_h_b64_mma_fla_bf16(
                k16_l2, w_pack, u_pack, g_cumsum, state)
        return gda.gdn_wy_output_o_b64_mma_fla_bf16(
            q_pack_hv, k_pack_hv, v_new_pack, h0, g_cumsum)

    def _wy_core_h(self, gda, co, g, beta, state, S):
        """The head-generic arm of the WY pipeline (non-48/16 hosts).

        The GQA split is contiguous column slices of the conv output —
        pinned bit-equal to the dedicated split kernel on the record —
        so the head-generic arm slices instead of asking for a kernel.
        """
        kd = self._hk * self._d
        hp = {"num_v_heads": self._hv, "num_k_heads": self._hk,
              "head_dim": self._d}
        q = co[:, :kd].contiguous().view(S, self._hk, self._d)
        k = co[:, kd:2 * kd].contiguous().view(S, self._hk, self._d)
        v = co[:, 2 * kd:].contiguous().view(S, self._hv, self._d)
        q_l2, k_l2, q_pack_hv, _k_pack_hk, g_cumsum = \
            gda.gdn_wy_norm_cumsum_pack_qk_h_bf16(q, k, g, **hp)
        big_a = gda.gdn_wy_kkt_b64_h_bf16(k_l2, beta, g_cumsum, **hp)
        ai = _wy_ai_inverse(big_a).contiguous()
        ai_pack = gda.gdn_wy_cast_ai_h_f32_to_bf16(
            ai, S, num_v_heads=self._hv)
        w_pack, u_pack = gda.gdn_wy_recompute_wu_b64_mma_fla_h_bf16(
            k_l2, v, beta, g_cumsum, ai_pack, **hp)
        h0, _v_new, v_new_pack, k_pack_hv = \
            gda.gdn_wy_chunk_h_b64_mma_fla_h_bf16(
                k_l2, w_pack, u_pack, g_cumsum, state, **hp)
        return gda.gdn_wy_output_o_b64_mma_fla_h_bf16(
            q_pack_hv, k_pack_hv, v_new_pack, h0, g_cumsum, **hp)

    def _prefill_epilogue(self, hidden_states, cache_params, allp,
                          mixed, core_out, state, cont, old_slot, S):
        host = self.host_layer
        (_q0, _q1), (z0, z1), _b, _a = self._splits
        kk = self._conv_w.shape[-1]
        normed = self._fused.rms_norm_gated_silu_bf16(
            core_out.reshape(S * self._hv, self._d),
            allp[:, z0:z1].contiguous().view(S * self._hv, self._d),
            host.norm.weight, eps=self._eps)
        flat_norm = normed.view(S, -1)
        out = (self._proj_out(flat_norm) if self._proj_out is not None
               else torch.nn.functional.linear(flat_norm,
                                               host.out_proj.weight))
        # write INTO existing slots when they match — a repoint here
        # would strand a captured graph on the old tensors
        state4 = state.view(1, self._hv, self._d, self._d)
        rec = cache_params.recurrent_states[self._idx]
        if (torch.is_tensor(rec) and rec.shape == state4.shape
                and rec.dtype == state4.dtype):
            rec.copy_(state4)
        else:
            cache_params.recurrent_states[self._idx] = state4
        # the host slot keeps the last K *raw* projected inputs
        take = min(kk, S)
        cslot = cache_params.conv_states[self._idx]
        if not (torch.is_tensor(cslot)
                and cslot.shape == (1, mixed.shape[1], kk)
                and cslot.dtype == mixed.dtype):
            cslot = mixed.new_zeros(1, mixed.shape[1], kk)
            cache_params.conv_states[self._idx] = cslot
        if cont and S < kk:
            # short continuation: the slot keeps the last kk raw inputs
            # across the old tail and the new tokens
            head = old_slot[:, :, S:].clone()
            cslot[:, :, :kk - S].copy_(head)
        else:
            cslot.zero_()
        cslot[0, :, kk - take:] = mixed[S - take:].t()
        return out.view(1, S, -1).to(hidden_states.dtype)

    def forward(self, hidden_states, cache_params=None,
                attention_mask=None):
        admitted = self._frt_admit(hidden_states)
        if admitted is not PROCEED:
            return admitted
        decode = (cache_params is not None
                  and getattr(cache_params, "has_previous_state", False)
                  and hidden_states.shape[0] == 1
                  and hidden_states.shape[1] == 1
                  and (attention_mask is None
                       or bool(attention_mask.all())))
        if not decode:
            if (self._chunk_ok and cache_params is not None
                    and hidden_states.shape[0] == 1
                    and hidden_states.shape[1] > 1
                    and (attention_mask is None
                         or bool(attention_mask.all()))):
                return self._prefill_chain(hidden_states, cache_params)
            return self._host_form(hidden_states, cache_params,
                                   attention_mask)

        return self._decode_one(hidden_states, cache_params)

    def _decode_one(self, hidden_states, cache_params):
        host = self.host_layer
        # same identity-preserving handoff as the prefill chain
        allp = (self._proj_in(hidden_states).view(1, -1)
                if self._proj_in is not None
                else torch.nn.functional.linear(
                    hidden_states.view(1, -1), self._packed_w))
        # column slices of a single-row output stay contiguous
        (q0, q1), (z0, z1), (b0, b1), (a0, a1) = self._splits
        mixed = allp[:, q0:q1]
        z = allp[:, z0:z1]
        b = allp[:, b0:b1]
        a = allp[:, a0:a1]

        conv_host = cache_params.conv_states[self._idx]
        hub_state = conv_host[:, :, 1:].contiguous()
        conv_out = self._conv.causal_conv1d_update_bf16(
            mixed, self._conv_w, hub_state, self._conv_b,
            apply_silu=True)
        # the host slot keeps the last K raw inputs; roll it forward.
        # hub_state is a snapshot, so the two writes never overlap reads.
        conv_host[:, :, :-1].copy_(hub_state)
        conv_host[:, :, -1:].copy_(mixed.view(1, -1, 1))

        q, k, v = self._split_fn(conv_out)
        g, beta = self._gate_fn(
            a.view(1, self._hv), b.view(1, self._hv),
            self._neg_exp_a, self._dt_bias)
        state_in = cache_params.recurrent_states[self._idx]
        if state_in.dtype != torch.bfloat16 or not state_in.is_contiguous():
            # normalise the cache slot to a contiguous BF16 tensor once
            # (first decode after prefill); after this the slot pointer
            # never changes, which is what graph replay requires
            state_in = state_in.to(torch.bfloat16).contiguous()
            cache_params.recurrent_states[self._idx] = state_in
        stream_rec = (_native_recurrent_stream() if self._d == 128
                      else None)
        if stream_rec is not None:
            core_out, new_state = stream_rec(
                q.view(1, self._hv, self._d),
                k.view(1, self._hv, self._d),
                v.view(1, self._hv, self._d), g, beta, state_in,
                self._state_a, self._core_out)
        else:
            core_out, new_state = \
                self._gda.gated_delta_recurrent_inout_bf16(
                    q.view(1, self._hv, self._d),
                    k.view(1, self._hv, self._d),
                    v.view(1, self._hv, self._d), g, beta,
                    state_in, use_qk_l2norm=True,
                    state_out=self._state_a, out=self._core_out)
        # scratch -> slot copy keeps the slot pointer stable; the core
        # cannot write the slot it is reading within the same step
        state_in.copy_(new_state)

        gnq = (_native_gated_norm_quant()
               if self._proj_out is not None
               and self._d == 128 else None)
        if gnq is not None:
            # the norm hands the projection packed rows directly: its
            # only consumer would otherwise re-read the row to
            # quantize it, one launch per layer
            if self._gn_packed is None:
                self._arm_gated_norm_quant()
            gnq(core_out.view(self._hv, self._d),
                z.view(self._hv, self._d), self._norm_w,
                self._gn_normed, self._gn_packed, self._gn_sfa,
                self._eps)
            out = self._proj_out._mm_packed(self._gn_packed,
                                            self._gn_sfa)
            return out.view(1, 1, -1).to(hidden_states.dtype)
        normed = self._fused.rms_norm_gated_silu_bf16(
            core_out.view(self._hv, self._d), z.view(self._hv, self._d),
            host.norm.weight, eps=self._eps)
        flat_norm = normed.view(1, -1)
        out = (self._proj_out(flat_norm) if self._proj_out is not None
               else torch.nn.functional.linear(flat_norm,
                                               host.out_proj.weight))
        return out.view(1, 1, -1).to(hidden_states.dtype)

    @torch.no_grad()
    def _arm_gated_norm_quant(self):
        """Stable buffers for the fused gated-norm producer.

        Allocated once, before any capture: the graph records these
        addresses, and the norm weight is copied detached so the op
        never sits on an autograd edge."""
        host = self.host_layer
        dev = self._conv_w.device
        n = self._hv * self._d
        self._norm_w = host.norm.weight.detach().to(
            dev, torch.bfloat16).contiguous().clone()
        self._gn_normed = torch.empty(self._hv, self._d, device=dev,
                                      dtype=torch.bfloat16)
        self._gn_packed = torch.empty(1, n // 2, device=dev,
                                      dtype=torch.uint8)
        self._gn_sfa = torch.zeros(((n + 63) // 64) * 512, device=dev,
                                   dtype=torch.uint8)


@torch.no_grad()
def bind_fused_decode_layer(host, layer_idx: int,
                            projection_format: str | None = None,
                            release_host_weights: bool = False):
    """Bind one layer; a smoke step runs on zeros before handing out."""
    bound = FusedGatedDeltaDecodeLayer(host, layer_idx,
                                       projection_format)

    class _Cache:
        pass

    cache = _Cache()
    d_model = int(host.in_proj_qkv.weight.shape[1])
    conv_k = int(host.conv1d.weight.shape[-1])
    conv_dim = int(host.conv1d.weight.shape[0])
    dev = host.in_proj_qkv.weight.device
    cache.conv_states = {layer_idx: torch.zeros(
        1, conv_dim, conv_k, device=dev, dtype=torch.bfloat16)}
    cache.recurrent_states = {layer_idx: torch.zeros(
        1, bound._hv, bound._d, bound._d, device=dev,
        dtype=torch.bfloat16)}
    cache.has_previous_state = True
    probe = bound(torch.zeros(1, 1, d_model, device=dev,
                              dtype=torch.bfloat16), cache, None)
    if probe.shape != (1, 1, d_model) or \
            not torch.isfinite(probe.float()).all():
        raise ValueError("refused: fused decode chain smoke failed")
    guard = bound._frt_guard
    if guard is not None:
        guard.calls = 0
    if release_host_weights and bound._proj_in is not None \
            and bound._proj_out is not None:
        # one-way: the FP4 band passed its smoke, the BF16 projection
        # weights go. From here the host form refuses instead of
        # falling back, and detach restores structure, not bytes.
        empty = torch.nn.Parameter(
            host.in_proj_qkv.weight.new_empty(0), requires_grad=False)
        for name in ("in_proj_qkv", "in_proj_z", "in_proj_b",
                     "in_proj_a", "out_proj"):
            getattr(host, name).weight = empty
        bound._packed_w = None
        bound._released = True
        if guard is not None:
            guard.notes["host_weights"] = "released (one-way)"
    return bound


@torch.no_grad()
def calibrate_gdn_channel_amax(lm, run_once, *, samples: int = 1,
                               percentile: float = 99.9,
                               verbose: bool = False) -> int:
    """Attach calibrated per-input-channel amax to gated-delta hosts.

    The house calibration front door: the structures ``Collector``
    observes the two projection inputs (``in_proj_qkv``, ``out_proj``)
    of every gated-delta host layer while ``run_once`` drives a real
    forward — the statistics only ever come from the host's own data
    path, never from synthetic tensors. Each observed layer receives
    ``_frt_gdn_channel_amax = {"in": [K], "out": [K2]}``, which is the
    precondition the ``nvfp4_balance`` projection format checks at
    bind. Returns the number of layers calibrated.
    """
    from types import SimpleNamespace

    from ...points import Collector, Point

    mods = dict(lm.named_modules())
    hosts = [(name, mod) for name, mod in mods.items()
             if hasattr(mod, "conv1d") and hasattr(mod, "A_log")
             and hasattr(mod, "in_proj_qkv")]
    points, request = [], {}
    for name, _mod in hosts:
        for attr in ("in_proj_qkv", "out_proj"):
            path = f"{name}.{attr}" if name else attr
            p = Point("x", path, "input")
            points.append(p)
            request[f"{p.path}|{p.name}"] = SimpleNamespace(
                stat="amax", granularity="channel")
    collector = Collector(points=points)
    collector.request = request
    handles = collector.hooks(lambda path: mods[path])
    try:
        for _ in range(max(1, samples)):
            run_once()
            collector.end_sample()
    finally:
        for h in handles:
            h.remove()
    collector.reduce(percentile, verbose=verbose,
                     label="gdn_channel_amax")
    calibrated = 0
    for name, mod in hosts:
        pin = f"{name}.in_proj_qkv" if name else "in_proj_qkv"
        pout = f"{name}.out_proj" if name else "out_proj"
        a_in = collector.channel_amax(pin, "x")
        a_out = collector.channel_amax(pout, "x")
        if a_in is None or a_out is None:
            continue
        mod._frt_gdn_channel_amax = {
            "in": torch.as_tensor(a_in, dtype=torch.float32),
            "out": torch.as_tensor(a_out, dtype=torch.float32)}
        calibrated += 1
    return calibrated

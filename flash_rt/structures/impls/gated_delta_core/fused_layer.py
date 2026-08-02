"""Fused decode form of a whole gated-delta layer.

The transformers fallback runs this layer as ~75 launches of Python
glue per token; serving its pieces individually is measurably negative
on a launch-bound host (a quantized projection swap *lost* throughput
here — the receipts are on the record). This impl owns the layer's
cached-decode step as one short chain of Hub kernels:

    in_proj GEMVs -> causal_conv1d_update -> broadcast QKV split
    -> gating -> gated-delta recurrent core -> gated RMSNorm -> out_proj

Everything else — prefill, uncached calls, masked batches — dispatches
to the retained host layer and is counted.

Cache contract (the host's, followed not replaced): the layer reads and
writes ``cache_params.conv_states[idx]`` and ``recurrent_states[idx]``.
The host keeps the last K raw inputs in the conv state; the Hub update
kernel keeps the previous K-1, so the impl feeds ``state[..., 1:]`` and
rolls the host slot forward itself. The recurrent state is written
back in BF16 through ping-pong buffers — the core reads the previous
state while writing the next, and one buffer serving both sides of the
same step would race.
"""

from __future__ import annotations

from functools import lru_cache

import torch

from ...guard import CAST_OK, PROCEED, GuardedSeam

GDA_DEP = {"provider": "hf", "repo": "flashrt/gated-delta-attention",
           "version": ">=3"}
CONV_DEP = {"provider": "hf", "repo": "flashrt/causal-conv1d-state",
            "version": ">=1"}
FUSED_DEP = {"provider": "hf", "repo": "flashrt/transformer-fused-ops",
             "version": ">=1"}


@lru_cache(maxsize=1)
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

    def __init__(self, host, layer_idx: int):
        super().__init__()
        gda, conv, fused = _packages()
        self._gda, self._conv, self._fused = gda, conv, fused
        self.host_layer = host
        self._idx = int(layer_idx)
        self._hv = int(host.num_v_heads)
        self._d = int(host.head_v_dim)
        if (self._hv, int(host.num_k_heads), self._d,
                int(host.head_k_dim)) != (48, 16, 128, 128):
            raise ValueError(
                "fused decode chain serves the 48/16-head, D=128 "
                "profile; other profiles keep the host layer")
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
        self._neg_exp_a = (-host.A_log.detach().float().exp()).contiguous()
        self._dt_bias = host.dt_bias.detach().float().contiguous()
        self._eps = float(getattr(host.norm, "variance_epsilon",
                                  getattr(host.norm, "eps", 1e-6)))
        d_model = int(host.in_proj_qkv.weight.shape[1])
        self._state_a = torch.empty(1, self._hv, self._d, self._d,
                                    device=dev, dtype=torch.bfloat16)
        self._state_b = torch.empty_like(self._state_a)
        self._flip = False
        self._core_out = torch.empty(1, self._hv, self._d, device=dev,
                                     dtype=torch.bfloat16)
        guard = self._frt_arm(dtypes=CAST_OK, device=dev, k=d_model)
        guard.notes["host_form_calls"] = 0

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "host_layer":
                raise
            return getattr(super().__getattr__("host_layer"), name)

    def _host_form(self, *args, **kwargs):
        guard = self._frt_guard
        if guard is not None and not torch.compiler.is_compiling():
            guard.notes["host_form_calls"] += 1
        return self.host_layer(*args, **kwargs)

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
            return self._host_form(hidden_states, cache_params,
                                   attention_mask)

        host = self.host_layer
        x = hidden_states.view(1, -1)
        allp = torch.nn.functional.linear(x, self._packed_w)
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

        q, k, v = self._gda.lin_split_qkv_broadcast_bf16(conv_out)
        g, beta = self._gda.gdn_gating_bf16(
            a.view(1, self._hv), b.view(1, self._hv),
            self._neg_exp_a, self._dt_bias)
        state_in = cache_params.recurrent_states[self._idx]
        if state_in.dtype != torch.bfloat16:
            state_in = state_in.to(torch.bfloat16)
        state_out = self._state_b if self._flip else self._state_a
        self._flip = not self._flip
        core_out, new_state = self._gda.gated_delta_recurrent_inout_bf16(
            q.view(1, self._hv, self._d), k.view(1, self._hv, self._d),
            v.view(1, self._hv, self._d), g, beta,
            state_in.contiguous(), use_qk_l2norm=True,
            state_out=state_out, out=self._core_out)
        cache_params.recurrent_states[self._idx] = new_state

        normed = self._fused.rms_norm_gated_silu_bf16(
            core_out.view(self._hv, self._d), z.view(self._hv, self._d),
            host.norm.weight, eps=self._eps)
        out = torch.nn.functional.linear(normed.view(1, -1),
                                         host.out_proj.weight)
        return out.view(1, 1, -1).to(hidden_states.dtype)


@torch.no_grad()
def bind_fused_decode_layer(host, layer_idx: int):
    """Bind one layer; a smoke step runs on zeros before handing out."""
    bound = FusedGatedDeltaDecodeLayer(host, layer_idx)

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
    return bound

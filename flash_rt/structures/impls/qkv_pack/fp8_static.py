"""qkv_pack — pack sibling linears that share one input into one GEMM.

Sibling projections consumed in a fixed call order (q/k/v of an
attention block, gate/up of an MLP) each pay a small-M GEMM whose cost
is launch/latency floor, not bandwidth. Packing their weights into one
``[sum(N_i), K]`` matrix turns the group into a single GEMM; the later
siblings become buffer reads. Two bind forms cover the hosts seen so
far:

- **leaf**: the host calls the sibling modules separately and there is
  no enclosing attention-module boundary. The first sibling's slot gets
  a :class:`PackedLinear` (runs the packed GEMM, writes the other
  outputs into preallocated buffers); the later slots get
  :class:`StashReader` (return the buffer). The host's own call order
  is the data dependency — functionalization keeps copy/read ordered
  inside compiled and captured graphs.
- **module**: the host has an attention module with
  ``q_proj/k_proj/v_proj/out_proj`` attributes and a standard
  projections → attention → out_proj forward. :class:`AttnBlockPacked`
  replaces the whole module: packed GEMM, SDPA at a declared compute
  dtype, original ``out_proj``.

Both forms quantize the packed weight to FP8 with one joint per-tensor
scale (the joint-scale rounding difference is covered by the parity
gate). Inputs enter either as FP8 (a producer seam supplies the shared
``act_scale``) or as BF16 through the fused-quantize entry with a
calibrated ``act_scale``.
"""

from __future__ import annotations

from typing import Sequence

import torch

from .. import hub_kernel

_FP8 = torch.float8_e4m3fn


def _pack_weights(mods: Sequence[torch.nn.Module]):
    ws = [m.weight.detach() for m in mods]
    w = torch.cat(ws, 0)
    scale = (w.float().abs().max() / 448.0).clamp(min=1e-8).view(1)
    w8 = (w.float() / scale).clamp(-448, 448).to(_FP8)
    splits = [wi.shape[0] for wi in ws]
    biases = []
    for m in mods:
        b = getattr(m, "bias", None)
        biases.append(b.detach().to(torch.bfloat16) if b is not None
                      else torch.zeros(m.weight.shape[0],
                                       device=w.device,
                                       dtype=torch.bfloat16))
    return w8, scale, torch.cat(biases), splits


class PackedLinear(torch.nn.Module):
    """Leaf-form head: one packed GEMM, later siblings stashed."""

    def __init__(self, mods: Sequence[torch.nn.Module],
                 act_scale: torch.Tensor, rows: int,
                 in_dtype: str = "fp8_static"):
        super().__init__()
        self.host_linear = mods[0]
        kf = hub_kernel("flashrt/flashrt-fp8-ffn", ">=1")
        self.in_dtype = in_dtype
        w8, w_scale, bias, splits = _pack_weights(mods)
        self.splits = splits
        self.register_buffer("w8", w8)
        self.register_buffer("w_scale", w_scale)
        self.register_buffer("bias_cat", bias)
        self.register_buffer("act_scale", act_scale)
        dev = w8.device
        for i, n in enumerate(splits[1:], 1):
            self.register_buffer(f"stash{i}", torch.zeros(
                rows, n, device=dev, dtype=torch.bfloat16))
        if in_dtype == "fp8_static":
            self._fn = kf.fp8_linear_bias_bf16
        else:
            self._fn = kf.bf16_fp8_linear_bias_bf16
            k = mods[0].weight.shape[1]
            self.register_buffer("x8_buf", torch.empty(
                rows, k, device=dev, dtype=_FP8))
            self.register_buffer("y_buf", torch.empty(
                rows, sum(splits), device=dev, dtype=torch.bfloat16))

    def forward(self, x):
        flat = x.reshape(-1, x.shape[-1])
        if self.in_dtype == "fp8_static":
            y = self._fn(flat, self.w8, self.bias_cat, self.act_scale,
                         self.w_scale)
        else:
            y = self._fn(flat.to(torch.bfloat16).contiguous(),
                         self.w8, self.bias_cat, self.act_scale,
                         self.w_scale, input_fp8=self.x8_buf,
                         out=self.y_buf)
        off = self.splits[0]
        for i, n in enumerate(self.splits[1:], 1):
            getattr(self, f"stash{i}").copy_(y[:, off:off + n])
            off += n
        out = y[:, :self.splits[0]].contiguous()
        out = out.reshape(*x.shape[:-1], self.splits[0])
        # the kernel's output dtype is BF16 by contract; only cast back
        # when the host boundary itself is a compute dtype. On the
        # fp8_static entry the input is FP8 (a producer seam supplies
        # it) and casting to it would hand FP8 activations to the
        # host's next op.
        return out if x.dtype is _FP8 else out.to(x.dtype)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("host_linear"), name)


class StashReader(torch.nn.Module):
    """Leaf-form tail: return the packed head's stashed output."""

    def __init__(self, orig: torch.nn.Module, packed: PackedLinear,
                 index: int):
        super().__init__()
        self.host_linear = orig
        self._packed = (packed,)
        self.index = index

    def forward(self, x):
        buf = getattr(self._packed[0], f"stash{self.index}")
        out = buf.reshape(*x.shape[:-1], buf.shape[-1])
        return out if x.dtype is _FP8 else out.to(x.dtype)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("host_linear"), name)


def bind_qkv_pack(mods: Sequence[torch.nn.Module],
                  act_scale: torch.Tensor, rows: int,
                  in_dtype: str = "fp8_static"):
    """Bind a sibling group; returns replacements in sibling order."""
    if len(mods) < 2:
        raise ValueError("qkv_pack: need at least two siblings")
    kdims = {m.weight.shape[1] for m in mods}
    if len(kdims) != 1:
        raise ValueError(f"qkv_pack: sibling K dims differ {kdims}")
    packed = PackedLinear(mods, act_scale, rows, in_dtype=in_dtype)
    out = [packed]
    for i, m in enumerate(mods[1:], 1):
        out.append(StashReader(m, packed, i))
    return out


class AttnBlockPacked(torch.nn.Module):
    """Module-form: packed QKV + SDPA at a declared dtype + out_proj.

    Fits attention modules exposing ``q_proj/k_proj/v_proj/out_proj``,
    ``head_dim`` and ``scale`` with the standard block forward
    (SigLIP/CLIP-family vision towers and friends).
    """

    def __init__(self, orig: torch.nn.Module, act_scale: torch.Tensor,
                 rows: int, sdpa_dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.host_attn = orig
        kf = hub_kernel("flashrt/flashrt-fp8-ffn", ">=1")
        self._fn = kf.bf16_fp8_linear_bias_bf16
        w8, w_scale, bias, splits = _pack_weights(
            [orig.q_proj, orig.k_proj, orig.v_proj])
        if len(set(splits)) != 1:
            raise ValueError("attn_block: q/k/v widths differ")
        self.e = splits[0]
        self.register_buffer("w8", w8)
        self.register_buffer("w_scale", w_scale)
        self.register_buffer("bias_cat", bias)
        self.register_buffer("in_scale", act_scale)
        k = orig.q_proj.weight.shape[1]
        dev = w8.device
        self.register_buffer("x8_buf", torch.empty(
            rows, k, device=dev, dtype=_FP8))
        self.register_buffer("y_buf", torch.empty(
            rows, 3 * self.e, device=dev, dtype=torch.bfloat16))
        self.sdpa_dtype = sdpa_dtype

    def forward(self, hidden_states, attention_mask=None, **kw):
        a = self.host_attn
        bsz, seq, dim = hidden_states.shape
        flat = hidden_states.reshape(-1, dim).to(
            torch.bfloat16).contiguous()
        y = self._fn(flat, self.w8, self.bias_cat, self.in_scale,
                     self.w_scale, input_fp8=self.x8_buf,
                     out=self.y_buf)
        hd = a.head_dim

        def split(t):
            return t.contiguous().view(bsz, seq, -1, hd).transpose(
                1, 2).to(self.sdpa_dtype)

        e = self.e
        mask = (attention_mask.to(self.sdpa_dtype)
                if attention_mask is not None else None)
        o = torch.nn.functional.scaled_dot_product_attention(
            split(y[:, :e]), split(y[:, e:2 * e]), split(y[:, 2 * e:]),
            attn_mask=mask, scale=a.scale)
        o = o.to(hidden_states.dtype).transpose(1, 2).reshape(
            bsz, seq, dim).contiguous()
        return a.out_proj(o), None

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("host_attn"), name)


def bind_attn_block(orig: torch.nn.Module, act_scale: torch.Tensor,
                    rows: int,
                    sdpa_dtype: torch.dtype = torch.bfloat16
                    ) -> AttnBlockPacked:
    for attr in ("q_proj", "k_proj", "v_proj", "out_proj", "head_dim",
                 "scale"):
        if not hasattr(orig, attr):
            raise ValueError(f"attn_block: host lacks {attr!r}")
    return AttnBlockPacked(orig, act_scale, rows, sdpa_dtype=sdpa_dtype)

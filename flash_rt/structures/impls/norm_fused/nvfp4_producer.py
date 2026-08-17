"""A (1+w)-form RMSNorm that also emits its consumers' NVFP4 input.

The pipeline fact this serves: a pre-norm decoder norm's output has
exactly one consumer group — the bound FP4 projections that read it —
and each group's first act is to quantize that activation. The fused
producer computes the host norm exactly (fp32, one bf16 round) and
quantizes the same values in the same kernel, publishing the packed
FP4 + SFA through the identity-keyed share cell the consumer group
already honors. The normed BF16 tensor still flows through the host
graph unchanged, so nothing off the calibrated path ever sees a packed
tensor: at worst a consumer misses the cell and quantizes for itself.
"""

from __future__ import annotations

import os
from functools import lru_cache

import torch

from ..linear_proj.nvfp4_dynamic import (LinearProjNvfp4Dynamic,
                                         _ShareCell)


@lru_cache(maxsize=1)
def _native_norm_quant():
    """The local build's fused RMSNorm + NVFP4 quantize producer."""
    try:
        from flash_rt import flash_rt_kernels as _fk
    except ImportError:
        return None
    fn = getattr(_fk, "rms_norm_quantize_fp4_sfa_bf16", None)
    if fn is None:
        return None

    from torch.library import custom_op

    @custom_op("flashrt_native::rms_norm_quant_fp4", mutates_args=())
    def _op(x: torch.Tensor, w: torch.Tensor,
            eps: float) -> list[torch.Tensor]:
        flat = x.reshape(-1, x.shape[-1]).contiguous()
        m, d = flat.shape
        normed = torch.empty_like(flat)
        packed = torch.empty(m, d // 2, device=x.device,
                             dtype=torch.uint8)
        # zero-filled: the tail rows of the 128-row scale atom are
        # never written for a partial block, and run-to-run garbage
        # there is the one nondeterministic input a consumer could see
        sfa = torch.zeros(
            ((m + 127) // 128) * ((d + 63) // 64) * 512,
            device=x.device, dtype=torch.uint8)
        rc = fn(flat.data_ptr(), w.data_ptr(), float(eps),
                normed.data_ptr(), packed.data_ptr(), sfa.data_ptr(),
                m, d, torch.cuda.current_stream().cuda_stream)
        if rc != 0:
            raise RuntimeError(
                f"rms_norm_quant_fp4 refused rc={rc} for M={m} D={d}")
        return [normed, packed, sfa]

    @_op.register_fake
    def _(x, w, eps):
        flat_shape = (x.numel() // x.shape[-1], x.shape[-1])
        m, d = flat_shape
        return [x.new_empty(flat_shape),
                x.new_empty((m, d // 2), dtype=torch.uint8),
                x.new_empty(
                    (((m + 127) // 128) * ((d + 63) // 64) * 512,),
                    dtype=torch.uint8)]

    return _op


class RMSNormQuantFp4Producer(torch.nn.Module):
    """Drop-in for the host RMSNorm; feeds the consumer group's cell."""

    def __init__(self, host_norm: torch.nn.Module, cell: _ShareCell):
        super().__init__()
        self.host_norm = host_norm
        self._cell = cell
        self._op = _native_norm_quant()
        self._eps = float(getattr(host_norm, "variance_epsilon",
                                  getattr(host_norm, "eps", 1e-6)))
        self._min_m = int(os.environ.get("FRT_NORM_QUANT_MIN_M", "64"))
        # a detached copy keeps autograd out of the custom op: the
        # host weight is a Parameter, and a traced graph that sees it
        # demands a backward formula this producer does not carry
        self.register_buffer(
            "w", host_norm.weight.detach().to(torch.bfloat16)
            .contiguous().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._op is None or x.dtype is not torch.bfloat16:
            return self.host_norm(x)
        # measured M-dispatch: a host norm the compiler folded into
        # its neighbours can beat this producer's standalone launch at
        # narrow widths, so the band where the fused pass pays is
        # measured per host rather than assumed
        if x.numel() // x.shape[-1] < self._min_m:
            return self.host_norm(x)
        # inference producer: detach severs the autograd edge a
        # Parameter-derived input would otherwise demand a backward
        # formula for (the op carries none)
        normed, packed, sfa = self._op(x.detach(), self.w, self._eps)
        out = normed.reshape(x.shape)
        cell = self._cell
        cell.x, cell.a, cell.sfa = out, packed, sfa
        return out


def _consumer_cell(layer: torch.nn.Module,
                   which: str) -> _ShareCell | None:
    """Locate/attach the share cell of a norm's consumer group."""
    if which == "attn":
        attn = getattr(layer, "self_attn", None)
        if attn is not None:
            seams = [getattr(attn, n, None)
                     for n in ("q_proj", "k_proj", "v_proj")]
            if all(isinstance(s, LinearProjNvfp4Dynamic)
                   for s in seams):
                cell = getattr(seams[0], "_share", None)
                if cell is None:
                    cell = _ShareCell()
                    for s in seams:
                        s._share = cell
                return cell
        gdn = getattr(layer, "linear_attn", None)
        seam = getattr(gdn, "_proj_in", None) if gdn is not None else None
        if isinstance(seam, LinearProjNvfp4Dynamic):
            cell = getattr(seam, "_share", None)
            if cell is None:
                cell = _ShareCell()
                seam._share = cell
            return cell
        return None
    mlp = getattr(layer, "mlp", None)
    if mlp is not None and hasattr(mlp, "gate_up"):
        cell = getattr(mlp, "_share", None)
        if cell is None:
            cell = _ShareCell()
            mlp._share = cell
        return cell
    return None


@torch.no_grad()
def adopt_norm_quant(root: torch.nn.Module,
                     verbose: bool = False) -> int:
    """Swap each decoder norm whose consumer group is FP4-bound."""
    if _native_norm_quant() is None:
        return 0
    count = 0
    layers = getattr(root, "layers", None)
    if layers is None:
        return 0
    for li, layer in enumerate(layers):
        for norm_name, which in (("input_layernorm", "attn"),
                                 ("post_attention_layernorm", "mlp")):
            norm = getattr(layer, norm_name, None)
            if norm is None or not hasattr(norm, "weight"):
                continue
            cell = _consumer_cell(layer, which)
            if cell is None:
                continue
            setattr(layer, norm_name,
                    RMSNormQuantFp4Producer(norm, cell))
            count += 1
            if verbose:
                print(f"[norm_fused.nvfp4_producer] layer {li} "
                      f"{norm_name} -> producer")
    return count

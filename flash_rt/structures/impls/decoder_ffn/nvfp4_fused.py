"""NVFP4 fused-GLU implementation of the ``decoder_ffn`` structure.

Composes the whole SwiGLU block behind one seam over already-bound
``linear_proj_nvfp4`` projections: the gate and up weights concatenate
into a single merged seam (one weight stream, one launch per tier), the
activation between them collapses into the ``flashrt/fp4-fused-ops``
silu-mul producer that emits the down projection's packed FP4 input
directly — no BF16 round-trip, no standalone quantize launch, no
elementwise-mul kernel.

Adoption is a post-pass over a model whose projections the
``linear_proj_nvfp4`` scheme has already bound: the three bound seams
are fused in place, reusing their packed weights (the concat is layout
sound because the SFB blocks are N-major and both halves are multiples
of the 128-row block). The host MLP is retained whole for fallback and
introspection, same contract as the other ``decoder_ffn`` impls.
"""

from __future__ import annotations

from functools import lru_cache

import torch

from ...guard import CAST_OK, PROCEED, GuardedSeam
from ..linear_proj.nvfp4_dynamic import (LinearProjNvfp4Dynamic,
                                         _quantize_activation)

FUSED_DEP = {"provider": "hf", "repo": "flashrt/fp4-fused-ops",
             "version": ">=1"}


@lru_cache(maxsize=1)
def _native_silu_mul():
    """The fused SwiGLU + NVFP4 quantize producer.

    Bit-exact against the split (elementwise mul kernel -> production
    quantize kernel) chain by construction, so adopting it moves no
    numerics.

    Hub artifact first: its entry is already a torch op with a fake, so
    a host that compiles this call traces it without help, and a host
    process that cannot load our native extension (a serving engine on
    a different torch/CUDA pair) still gets the fused producer. The
    local build follows, wrapped as a custom op for the same reason.
    """
    from flash_rt.structures.impls import hub_kernel

    try:
        hub = hub_kernel(FUSED_DEP["repo"], FUSED_DEP["version"])
    except Exception:  # noqa: BLE001 — absence is not a refusal
        hub = None
    hub_fn = getattr(hub, "silu_mul_quantize_fp4_sfa_bf16", None)
    if hub_fn is not None:
        def _hub_entry(merged):
            # allocate the outputs here rather than letting the wrapper
            # size them: its size helper is a custom op without a fake,
            # so a host tracing this call on Meta tensors dies inside
            # it. The layout is the packaged quantizer's own and stated
            # in its contract, so computing it here is reading the
            # contract, not guessing at it.
            m, two_h = merged.shape
            h = two_h // 2
            packed = merged.new_empty((m, h // 2), dtype=torch.uint8)
            sfa = merged.new_zeros(
                (((m + 127) // 128) * ((h + 63) // 64) * 512,),
                dtype=torch.uint8)
            return hub_fn(merged, packed=packed, sfa=sfa)

        return _hub_entry
    try:
        from flash_rt import flash_rt_kernels as _fk
    except ImportError:
        return None
    fn = getattr(_fk, "silu_mul_quantize_fp4_sfa_bf16", None)
    if fn is None:
        return None

    from torch.library import custom_op

    @custom_op("flashrt_native::silu_mul_quant_fp4", mutates_args=())
    def _op(merged: torch.Tensor) -> list[torch.Tensor]:
        m, two_h = merged.shape
        h = two_h // 2
        packed = torch.empty(m, h // 2, device=merged.device,
                             dtype=torch.uint8)
        # zero-filled: a partial 128-row scale atom leaves tail rows
        # unwritten, and run-to-run garbage there is a nondeterminism
        # channel for any consumer that loads whole atoms
        sfa = torch.zeros(
            ((m + 127) // 128) * ((h + 63) // 64) * 512,
            device=merged.device, dtype=torch.uint8)
        rc = fn(merged.data_ptr(), packed.data_ptr(), sfa.data_ptr(),
                m, h, torch.cuda.current_stream().cuda_stream)
        if rc != 0:
            raise RuntimeError(
                f"silu_mul_quant_fp4 refused rc={rc} for "
                f"M={m} H={h}")
        return [packed, sfa]

    @_op.register_fake
    def _(merged):
        m, two_h = merged.shape
        h = two_h // 2
        return [merged.new_empty((m, h // 2), dtype=torch.uint8),
                merged.new_empty(
                    (((m + 127) // 128) * ((h + 63) // 64) * 512,),
                    dtype=torch.uint8)]

    return _op


class FusedGluMlpNvfp4(GuardedSeam, torch.nn.Module):
    """MLP seam: quantize once -> merged gate|up -> silu-mul-quant -> down.

    No host retention: like the bound projection seams it fuses, the
    packed weights are the only copy — retaining the pre-fusion seams
    would hold a second full FFN weight set on device.
    """

    def __init__(self, gate_up: LinearProjNvfp4Dynamic,
                 down: LinearProjNvfp4Dynamic, silu_mul):
        super().__init__()
        self.gate_up = gate_up
        self.down = down
        self._silu_mul = silu_mul
        self._d = int(down._n)
        self._frt_arm(dtypes=CAST_OK,
                      device=gate_up._w_packed.device,
                      k=int(gate_up._k))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        admitted = self._frt_admit(x)
        if admitted is not PROCEED:
            return admitted
        shape = x.shape
        cell = getattr(self, "_share", None)
        if cell is not None and cell.x is x:
            a_packed, a_sfa = cell.a, cell.sfa
        else:
            flat = x.reshape(-1, shape[-1])
            a_packed, a_sfa = _quantize_activation(self.gate_up._kern,
                                                   flat)
            if cell is not None:
                cell.x, cell.a, cell.sfa = x, a_packed, a_sfa
        merged = self.gate_up._mm_packed(a_packed, a_sfa)
        p2, s2 = self._silu_mul(merged.contiguous())
        y = self.down._mm_packed(p2, s2)
        return y.reshape(*shape[:-1], self._d).type_as(x)


@torch.no_grad()
def fuse_bound_mlp(mlp: torch.nn.Module) -> FusedGluMlpNvfp4 | None:
    """Fuse one MLP module whose projections are already bound seams.

    Returns None (leave the host untouched) when any projection is not
    a bound ``linear_proj_nvfp4`` seam, carries a bias, or the merged
    halves would break the SFB block layout (either N not a multiple of
    the 128-row scale block).
    """
    g = getattr(mlp, "gate_proj", None)
    u = getattr(mlp, "up_proj", None)
    d = getattr(mlp, "down_proj", None)
    for seam in (g, u, d):
        if not isinstance(seam, LinearProjNvfp4Dynamic):
            return None
        if seam._bias is not None:
            return None
    if g._n % 128 or u._n % 128 or g._n != u._n or g._k != u._k:
        return None
    silu_mul = _native_silu_mul()
    if silu_mul is None:
        return None
    wp = torch.cat([g._w_packed, u._w_packed], dim=0).contiguous()
    ws = torch.cat([g._w_sfb, u._w_sfb], dim=0).contiguous()
    merged = LinearProjNvfp4Dynamic(wp, ws, None, 2 * int(g._n),
                                    int(g._k))
    return FusedGluMlpNvfp4(merged, d, silu_mul)


@torch.no_grad()
def adopt_fused_glu(root: torch.nn.Module, verbose: bool = False) -> int:
    """Post-pass: fuse every bound SwiGLU MLP under ``root`` in place."""
    count = 0
    for name, mod in list(root.named_modules()):
        for child_name, child in list(mod.named_children()):
            if not (hasattr(child, "gate_proj")
                    and hasattr(child, "up_proj")
                    and hasattr(child, "down_proj")):
                continue
            fused = fuse_bound_mlp(child)
            if fused is None:
                continue
            setattr(mod, child_name, fused)
            count += 1
            if verbose:
                print(f"[decoder_ffn.nvfp4_fused] {name}.{child_name} "
                      f"fused (2x{fused.gate_up._n // 2} -> "
                      f"{fused._d})")
    return count

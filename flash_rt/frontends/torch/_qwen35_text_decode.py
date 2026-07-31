"""The decode step's arithmetic, issued as kernels against fixed addresses.

Every buffer a step needs is allocated once, by :class:`Workspace`, and the
step itself moves integers. Nothing here builds a tensor, takes a slice or asks
for a shape, because those are dispatches and a step that does forty of them
per layer spends more on asking than on arithmetic -- measured on a sibling
model as 4274 dispatched operators for one token, more of the step than either
the kernels or the storage it was blamed on.

That discipline is also what makes the step capturable: a graph replays the
addresses it was captured with, so a path that allocates per call has nothing
stable to capture.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from flash_rt.frontends.torch._qwen35_text_weights import TextWeights


@dataclass
class _Buffer:
    """A device allocation and the address the kernels are given."""

    tensor: torch.Tensor
    address: int

    @classmethod
    def make(cls, *shape: int, dtype=torch.bfloat16,
             device: str = "cuda:0") -> "_Buffer":
        tensor = torch.empty(*shape, dtype=dtype, device=device)
        return cls(tensor=tensor, address=int(tensor.data_ptr()))


class Workspace:
    """Every scratch buffer a decode step uses, allocated once.

    Sized from the geometry rather than from the first call, so the addresses
    exist before anything runs and do not move afterwards.
    """

    def __init__(self, weights: TextWeights, device: str = "cuda:0",
                 max_batch: int = 1):
        dims = weights.dims
        self.device = device
        self.max_batch = max_batch
        self.group_size = weights.group_size

        widest_fused = max(
            2 * dims.intermediate,               # gate and up together
            2 * dims.lin_key_width + dims.lin_value_width,
            dims.q_width + 2 * dims.kv_width,
        )
        self.hidden = _Buffer.make(max_batch, dims.hidden, device=device)
        self.normed = _Buffer.make(max_batch, dims.hidden, device=device)
        self.residual = _Buffer.make(max_batch, dims.hidden, device=device)
        self.fused = _Buffer.make(max_batch, widest_fused, device=device)
        self.gated = _Buffer.make(max_batch, dims.intermediate, device=device)
        self.attn_out = _Buffer.make(
            max_batch, max(dims.attn_width, dims.lin_value_width),
            device=device)
        self.logits = _Buffer.make(max_batch, dims.vocab_size, device=device)
        # The sampled token, kept on the device so a greedy step never has to
        # come back to the host between tokens.
        self.token = _Buffer.make(max_batch, dtype=torch.int64, device=device)

    def close(self) -> None:
        for name in ("hidden", "normed", "residual", "fused", "gated",
                     "attn_out", "logits", "token"):
            setattr(self, name, None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _element_size(width: int) -> int:
    """Bytes per row of a bf16 buffer of this width."""
    return width * 2


def mlp_block(layer: dict[str, int], work: Workspace, fvk, x: int, out: int,
              rows: int, stream: int) -> None:
    """out = W_down * (silu(gate) * up), with gate and up one weight.

    Three launches: the fused projection, the gated product, and the
    contraction. ``x`` and ``out`` are addresses, and may be the same buffer
    only if the caller means them to be.
    """
    intermediate = layer["gate_up_up_offset"]
    call = (fvk.w4a16_packed_matvec_bf16 if rows == 1
            else fvk.w4a16_packed_gemm_bf16)
    extra = () if rows == 1 else (rows,)

    rc = call(x, layer["gate_up_packed"], layer["gate_up_scale"],
              work.fused.address, *extra, layer["gate_up_n"],
              layer["gate_up_k"], work.group_size, stream)
    if rc:
        raise RuntimeError(f"gate/up projection failed with {rc}")

    # silu(gate) * up over the fused output: the two halves are contiguous and
    # a row apart, which is arithmetic on the address rather than a slice.
    rc = fvk.silu_mul_sm120_bf16(
        work.fused.address,
        work.fused.address + _element_size(intermediate),
        work.gated.address, rows * intermediate, stream)
    if rc:
        raise RuntimeError(f"gated product failed with {rc}")

    rc = call(work.gated.address, layer["down_packed"], layer["down_scale"],
              out, *extra, layer["down_n"], layer["down_k"], work.group_size,
              stream)
    if rc:
        raise RuntimeError(f"down projection failed with {rc}")

"""The packed 4-bit GEMV against the layout quantized checkpoints ship.

The layout is a contract set by the producer, and two details of it are the
kind that fail silently rather than loudly:

- the nibbles are **offset binary**, not two's complement. Reading them as
  signed gives a weight distribution with a hole at zero and products that stay
  finite and plausible.
- value ``c`` of a row lives in nibble ``c % 8`` of word ``c / 8``, so the
  eight values of a word are eight consecutive columns.

Both are asserted here against a reference decode written straight from the
producer's own unpacking rule, so a kernel that gets either wrong fails on a
tensor rather than on a benchmark.

No checkpoint and no network: the packed bytes are generated, which is enough,
because what is under test is the decode and not any particular weight.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():                        # pragma: no cover
    pytest.skip("needs a GPU", allow_module_level=True)

try:
    from flash_rt import flash_rt_kernels as fvk
except ImportError:                                      # pragma: no cover
    pytest.skip("flash_rt_kernels is not built", allow_module_level=True)

if not hasattr(fvk, "w4a16_packed_matvec_bf16"):       # pragma: no cover
    pytest.skip("built without the packed 4-bit kernels",
                allow_module_level=True)

GROUP = 32
# Shapes a 4B-class checkpoint issues: an MLP pair, its down projection, the
# gated-DeltaNet projections, and a full-attention key projection.
SHAPES = [(9216, 2560), (2560, 9216), (8192, 2560), (4096, 2560),
          (2560, 4096), (1024, 2560)]
# Producers publish all three; the kernel takes the size rather than assuming.
GROUPS = [32, 64, 128]


def pack(values: torch.Tensor) -> torch.Tensor:
    """Fold signed values in [-8, 7] into the producer's int32 layout."""
    rows, columns = values.shape
    nibbles = (values.to(torch.int32) + 8) & 0xF
    packed = torch.zeros(rows, columns // 8, dtype=torch.int32,
                         device=values.device)
    for i in range(8):
        packed |= nibbles[:, i::8] << (4 * i)
    return packed


def reference(packed: torch.Tensor, scale: torch.Tensor,
              columns: int, group: int = GROUP) -> torch.Tensor:
    """The producer's unpacking rule, written out."""
    rows = packed.shape[0]
    unpacked = torch.zeros(rows, columns, dtype=torch.int32,
                           device=packed.device)
    for i in range(8):
        unpacked[:, i::8] = (packed >> (4 * i)) & 0xF
    values = (unpacked - 8).float()
    return values * scale.float().repeat_interleave(group, dim=1)


@pytest.mark.parametrize("n,k", SHAPES)
@pytest.mark.parametrize("group", GROUPS)
def test_matvec_matches_the_decoded_weight(n, k, group):
    torch.manual_seed(0)
    device = "cuda:0"
    values = torch.randint(-8, 8, (n, k), dtype=torch.int32, device=device)
    scale = (torch.rand(n, k // group, device=device) * 0.02 + 1e-3).to(
        torch.bfloat16)
    packed = pack(values)
    x = torch.randn(1, k, dtype=torch.bfloat16, device=device)

    out = torch.empty(1, n, dtype=torch.bfloat16, device=device)
    rc = fvk.w4a16_packed_matvec_bf16(
        x.data_ptr(), packed.data_ptr(), scale.data_ptr(), out.data_ptr(),
        n, k, group, torch.cuda.current_stream(device).cuda_stream)
    assert rc == 0

    want = (x.float() @ reference(packed, scale, k, group).T)
    torch.cuda.synchronize(device)
    error = (out.float() - want).abs().max() / want.abs().max().clamp_min(1e-6)
    assert error < 5e-3, (
        f"relative error {error:.3g} at N={n}, K={k}, group={group}")


def test_offset_binary_is_not_twos_complement():
    # The two readings differ on exactly the nibbles above seven, which is half
    # of them, so a kernel that confuses them is wrong on half the weights --
    # and wrong by a bounded amount, which is why it survives an eyeball.
    device = "cuda:0"
    k, n = 2560, 64
    values = torch.full((n, k), -8, dtype=torch.int32, device=device)
    scale = torch.ones(n, k // GROUP, dtype=torch.bfloat16, device=device)
    packed = pack(values)
    x = torch.ones(1, k, dtype=torch.bfloat16, device=device)

    out = torch.empty(1, n, dtype=torch.bfloat16, device=device)
    fvk.w4a16_packed_matvec_bf16(
        x.data_ptr(), packed.data_ptr(), scale.data_ptr(), out.data_ptr(),
        n, k, GROUP, torch.cuda.current_stream(device).cuda_stream)
    torch.cuda.synchronize(device)

    # -8 stores as nibble 0. Read as two's complement that is 0, and the row
    # sums to nothing; read as offset binary it is -8 and the row sums to -8K.
    assert out.float().mean().item() == pytest.approx(-8.0 * k, rel=1e-3)


@pytest.mark.parametrize("m", [2, 20])
def test_gemm_agrees_with_the_matvec_row_by_row(m):
    torch.manual_seed(1)
    device = "cuda:0"
    n, k = 2560, 2560
    values = torch.randint(-8, 8, (n, k), dtype=torch.int32, device=device)
    scale = (torch.rand(n, k // GROUP, device=device) * 0.02 + 1e-3).to(
        torch.bfloat16)
    packed = pack(values)
    x = torch.randn(m, k, dtype=torch.bfloat16, device=device)
    stream = torch.cuda.current_stream(device).cuda_stream

    batched = torch.empty(m, n, dtype=torch.bfloat16, device=device)
    assert fvk.w4a16_packed_gemm_bf16(
        x.data_ptr(), packed.data_ptr(), scale.data_ptr(),
        batched.data_ptr(), m, n, k, GROUP, stream) == 0

    single = torch.empty(m, n, dtype=torch.bfloat16, device=device)
    for row in range(m):
        source = x[row:row + 1].contiguous()
        assert fvk.w4a16_packed_matvec_bf16(
            source.data_ptr(), packed.data_ptr(), scale.data_ptr(),
            single[row].data_ptr(), n, k, GROUP, stream) == 0
    torch.cuda.synchronize(device)

    # Same order of accumulation in both, so the bar is equality.
    assert torch.equal(batched, single)

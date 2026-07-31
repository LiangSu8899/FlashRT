"""Per-output-channel INT8, as a projection and as a table.

A tied output projection and an embedding table are one tensor read two ways,
so both readings are checked against the same decoded weight -- a scale that
applies correctly in one direction and not the other would otherwise show up
as a model that embeds well and predicts badly.
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

if not hasattr(fvk, "w8a16_rowwise_matvec_bf16"):        # pragma: no cover
    pytest.skip("built without the rowwise INT8 kernels",
                allow_module_level=True)

DEVICE = "cuda:0"
# A vocabulary-sized projection and the backbone shapes around it.
SHAPES = [(248320, 2560), (2560, 2560), (9216, 2560), (1024, 2560)]


def quantize_rowwise(weight: torch.Tensor):
    """Symmetric per-output-channel INT8, the scale a property of the row."""
    scale = weight.abs().amax(dim=1).clamp_min(1e-8) / 127.0
    values = (weight / scale[:, None]).round().clamp(-127, 127).to(torch.int8)
    return values, scale.to(torch.float16)


# Rows per reference chunk. The vocabulary-sized case is 248320 x 2560: a
# reference that materialises the decoded weight wants 2.5 GiB of fp32 beside
# the int8 it decoded from, which does not fit the part this targets. The
# kernel still runs over every row -- only the comparison is chunked.
CHUNK_ROWS = 8192


def _quantized_rows(rows: int, k: int, generator: torch.Generator):
    """One chunk of weight, and its INT8 form, without keeping the fp32."""
    weight = torch.randn(rows, k, device=DEVICE, generator=generator) * 0.02
    values, scale = quantize_rowwise(weight)
    del weight
    return values, scale


@pytest.mark.parametrize("n,k", SHAPES)
def test_matvec_matches_the_decoded_weight(n, k):
    generator = torch.Generator(device=DEVICE).manual_seed(0)
    values = torch.empty(n, k, dtype=torch.int8, device=DEVICE)
    scale = torch.empty(n, dtype=torch.float16, device=DEVICE)
    for start in range(0, n, CHUNK_ROWS):
        stop = min(start + CHUNK_ROWS, n)
        chunk_values, chunk_scale = _quantized_rows(stop - start, k, generator)
        values[start:stop] = chunk_values
        scale[start:stop] = chunk_scale
        del chunk_values, chunk_scale
    x = torch.randn(1, k, dtype=torch.bfloat16, device=DEVICE)

    out = torch.empty(1, n, dtype=torch.bfloat16, device=DEVICE)
    rc = fvk.w8a16_rowwise_matvec_bf16(
        x.data_ptr(), values.data_ptr(), scale.data_ptr(), out.data_ptr(),
        n, k, torch.cuda.current_stream(DEVICE).cuda_stream)
    assert rc == 0
    torch.cuda.synchronize(DEVICE)

    # Compare a chunk at a time against the same rows decoded, so the largest
    # fp32 alive is one chunk rather than the whole weight.
    xf = x.float()
    worst = 0.0
    scale_of = 0.0
    for start in range(0, n, CHUNK_ROWS):
        stop = min(start + CHUNK_ROWS, n)
        decoded = values[start:stop].float() * scale[start:stop].float()[:, None]
        want = xf @ decoded.T
        worst = max(worst, float((out[:, start:stop].float() - want).abs().max()))
        scale_of = max(scale_of, float(want.abs().max()))
        del decoded, want
    error = worst / max(scale_of, 1e-6)
    assert error < 5e-3, f"relative error {error:.3g} at N={n}, K={k}"


def test_the_gather_reads_the_same_table_as_the_projection():
    # One tensor, two readings. A scale applied along the wrong axis would
    # embed plausibly and predict badly, which is hard to attribute later.
    torch.manual_seed(1)
    table_rows, k = 4096, 2560
    weight = torch.randn(table_rows, k, device=DEVICE) * 0.02
    values, scale = quantize_rowwise(weight)
    ids = torch.tensor([0, 7, 4095, 1234], dtype=torch.int64, device=DEVICE)

    out = torch.empty(ids.numel(), k, dtype=torch.bfloat16, device=DEVICE)
    rc = fvk.int8_rowwise_gather_bf16(
        ids.data_ptr(), values.data_ptr(), scale.data_ptr(), out.data_ptr(),
        ids.numel(), table_rows, k,
        torch.cuda.current_stream(DEVICE).cuda_stream)
    assert rc == 0
    torch.cuda.synchronize(DEVICE)

    decoded = (values.float() * scale.float()[:, None])[ids]
    assert torch.equal(out, decoded.to(torch.bfloat16))


@pytest.mark.parametrize("index", [-1, 4096, 10_000])
def test_an_index_outside_the_table_writes_zeros(index):
    # Reading past the table would return whatever lies there, which looks
    # like an embedding and is not one.
    table_rows, k = 4096, 256
    weight = torch.randn(table_rows, k, device=DEVICE)
    values, scale = quantize_rowwise(weight)
    ids = torch.tensor([index], dtype=torch.int64, device=DEVICE)

    out = torch.full((1, k), 7.0, dtype=torch.bfloat16, device=DEVICE)
    fvk.int8_rowwise_gather_bf16(
        ids.data_ptr(), values.data_ptr(), scale.data_ptr(), out.data_ptr(),
        1, table_rows, k, torch.cuda.current_stream(DEVICE).cuda_stream)
    torch.cuda.synchronize(DEVICE)

    assert torch.count_nonzero(out) == 0


def test_halving_the_output_projection_is_what_it_costs_to_read():
    # The reason this kernel exists: at a 248k vocabulary the tied table is a
    # large share of what a token reads, and int8 is half of bfloat16.
    vocab, hidden = 248320, 2560
    assert vocab * hidden * 2 == pytest.approx(1.271e9, rel=0.01)
    assert vocab * hidden * 1 + vocab * 2 == pytest.approx(0.636e9, rel=0.01)

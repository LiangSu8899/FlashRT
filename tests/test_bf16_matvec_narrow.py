"""A matvec whose output is narrow and whose contraction is long.

The arrangement that suits a wide projection starves a narrow one: a warp per
output row is sixty-four warps at N=64, and the launch then costs many times
what its bytes are worth. This checks the arithmetic, and that the shapes a
narrow projection actually has are covered.
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

if not hasattr(fvk, "bf16_matvec_narrow_bf16"):          # pragma: no cover
    pytest.skip("built without the portable decode kernels",
                allow_module_level=True)

DEVICE = "cuda:0"
# The decay projection of a 4B model, then a wider one and a single row.
SHAPES = [(64, 2560), (64, 5120), (256, 2560), (1, 4096), (8, 9216)]


@pytest.mark.parametrize("n,k", SHAPES)
def test_it_matches_the_product_it_stands_for(n, k):
    generator = torch.Generator(device=DEVICE).manual_seed(n * 31 + k)
    weight = torch.randn(n, k, dtype=torch.bfloat16, device=DEVICE,
                         generator=generator)
    x = torch.randn(k, dtype=torch.bfloat16, device=DEVICE,
                    generator=generator)
    out = torch.empty(n, dtype=torch.bfloat16, device=DEVICE)

    rc = fvk.bf16_matvec_narrow_bf16(
        x.data_ptr(), weight.data_ptr(), out.data_ptr(), n, k,
        torch.cuda.current_stream(DEVICE).cuda_stream)
    assert rc == 0
    torch.cuda.synchronize(DEVICE)

    want = weight.float() @ x.float()
    error = (out.float() - want).abs().max() / want.abs().max()
    assert error < 1e-2, f"relative error {error:.3g} at N={n}, K={k}"


def test_it_agrees_with_the_arrangement_it_replaces():
    # Two kernels for one product: they have to give the same answer, or a
    # change of arrangement is a change of model.
    n, k = 64, 2560
    torch.manual_seed(0)
    weight = torch.randn(n, k, dtype=torch.bfloat16, device=DEVICE)
    x = torch.randn(k, dtype=torch.bfloat16, device=DEVICE)
    stream = torch.cuda.current_stream(DEVICE).cuda_stream

    narrow = torch.empty(n, dtype=torch.bfloat16, device=DEVICE)
    fvk.bf16_matvec_narrow_bf16(x.data_ptr(), weight.data_ptr(),
                                narrow.data_ptr(), n, k, stream)
    wide = torch.empty(n, dtype=torch.bfloat16, device=DEVICE)
    fvk.bf16_matvec_qwen36_bf16(x.data_ptr(), weight.data_ptr(),
                                wide.data_ptr(), n, k, stream)
    torch.cuda.synchronize(DEVICE)

    scale = max(wide.float().abs().max().item(), 1e-6)
    assert (narrow.float() - wide.float()).abs().max().item() / scale < 1e-2


def test_a_contraction_that_does_not_vectorize_is_refused():
    # The row is read eight values at a time; a length that is not a multiple
    # of eight would read past the row rather than fail.
    buffer = torch.zeros(1024, dtype=torch.bfloat16, device=DEVICE)
    rc = fvk.bf16_matvec_narrow_bf16(
        buffer.data_ptr(), buffer.data_ptr(), buffer.data_ptr(), 4, 100,
        torch.cuda.current_stream(DEVICE).cuda_stream)
    assert rc == -1

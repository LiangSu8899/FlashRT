"""Splitting a convolved QKV stream and producing the recurrence's gates.

The two halves of this kernel share a launch and nothing else, so they are
checked separately against the arithmetic each is standing in for. The
broadcast is the part worth stating: query and key are published with fewer
heads than value, and a value head must read the key head it belongs to --
off by one group and every head still gets plausible numbers.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():                        # pragma: no cover
    pytest.skip("needs a GPU", allow_module_level=True)

try:
    from flash_rt import flash_rt_kernels as fvk
except ImportError:                                      # pragma: no cover
    pytest.skip("flash_rt_kernels is not built", allow_module_level=True)

if not hasattr(fvk, "linear_attn_split_broadcast_gate_bf16"):  # pragma: no cover
    pytest.skip("built without the portable decode kernels",
                allow_module_level=True)

DEVICE = "cuda:0"
# The 4B geometry, then a different broadcast and an odd head count.
GEOMETRIES = [(16, 32, 128, 128), (8, 8, 128, 128), (4, 12, 64, 96)]


def _run(rows, k_heads, v_heads, head_k, head_v, seed=0):
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    conv_width = 2 * k_heads * head_k + v_heads * head_v
    conv = torch.randn(rows, conv_width, dtype=torch.bfloat16, device=DEVICE,
                       generator=generator)
    # a and b arrive as halves of one projection, so both are strided views
    # of a single row rather than tensors of their own.
    ab = torch.randn(rows, 2 * v_heads, dtype=torch.bfloat16, device=DEVICE,
                     generator=generator)
    neg_exp_a_log = -torch.rand(v_heads, device=DEVICE,
                                generator=generator).mul(4).exp()
    dt_bias = torch.randn(v_heads, device=DEVICE, generator=generator)

    q = torch.empty(rows, v_heads, head_k, dtype=torch.bfloat16, device=DEVICE)
    k = torch.empty_like(q)
    v = torch.empty(rows, v_heads, head_v, dtype=torch.bfloat16, device=DEVICE)
    g = torch.empty(rows, v_heads, dtype=torch.bfloat16, device=DEVICE)
    beta = torch.empty_like(g)

    rc = fvk.linear_attn_split_broadcast_gate_bf16(
        conv.data_ptr(), ab.data_ptr(), ab.data_ptr() + v_heads * 2,
        neg_exp_a_log.data_ptr(), dt_bias.data_ptr(),
        q.data_ptr(), k.data_ptr(), v.data_ptr(),
        g.data_ptr(), beta.data_ptr(),
        rows, k_heads, v_heads, head_k, head_v, 2 * v_heads, 2 * v_heads,
        torch.cuda.current_stream(DEVICE).cuda_stream)
    assert rc == 0
    torch.cuda.synchronize(DEVICE)
    return conv, ab, neg_exp_a_log, dt_bias, q, k, v, g, beta


@pytest.mark.parametrize("k_heads,v_heads,head_k,head_v", GEOMETRIES)
def test_the_split_repeats_key_heads_onto_value_heads(
        k_heads, v_heads, head_k, head_v):
    rows = 3
    conv, _, _, _, q, k, v, _, _ = _run(rows, k_heads, v_heads, head_k, head_v)

    key_width = k_heads * head_k
    repeat = v_heads // k_heads
    want_q = conv[:, :key_width].reshape(rows, k_heads, head_k)
    want_k = conv[:, key_width:2 * key_width].reshape(rows, k_heads, head_k)
    want_v = conv[:, 2 * key_width:].reshape(rows, v_heads, head_v)

    assert torch.equal(q, want_q.repeat_interleave(repeat, dim=1))
    assert torch.equal(k, want_k.repeat_interleave(repeat, dim=1))
    assert torch.equal(v, want_v)


@pytest.mark.parametrize("k_heads,v_heads,head_k,head_v", GEOMETRIES)
def test_the_gates_match_the_rule_that_defines_them(
        k_heads, v_heads, head_k, head_v):
    rows = 3
    _, ab, neg_exp_a_log, dt_bias, _, _, _, g, beta = _run(
        rows, k_heads, v_heads, head_k, head_v)

    a = ab[:, :v_heads].float()
    b = ab[:, v_heads:].float()
    want_g = neg_exp_a_log * torch.nn.functional.softplus(a + dt_bias)
    want_beta = torch.sigmoid(b)

    assert torch.allclose(g.float(), want_g, atol=8e-3, rtol=8e-3)
    assert torch.allclose(beta.float(), want_beta, atol=4e-3, rtol=4e-3)


def test_a_large_decay_does_not_overflow_the_softplus():
    # softplus goes through an exponential, which overflows long before the
    # result stops being its own argument. A run of infinities in the decay
    # would zero the state rather than fail.
    v_heads, head = 8, 64
    rows = 1
    conv = torch.zeros(rows, 2 * v_heads * head + v_heads * head,
                       dtype=torch.bfloat16, device=DEVICE)
    ab = torch.zeros(rows, 2 * v_heads, dtype=torch.bfloat16, device=DEVICE)
    ab[:, :v_heads] = 200.0
    neg_exp_a_log = -torch.ones(v_heads, device=DEVICE)
    dt_bias = torch.zeros(v_heads, device=DEVICE)

    q = torch.empty(rows, v_heads, head, dtype=torch.bfloat16, device=DEVICE)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    g = torch.empty(rows, v_heads, dtype=torch.bfloat16, device=DEVICE)
    beta = torch.empty_like(g)

    fvk.linear_attn_split_broadcast_gate_bf16(
        conv.data_ptr(), ab.data_ptr(), ab.data_ptr() + v_heads * 2,
        neg_exp_a_log.data_ptr(), dt_bias.data_ptr(),
        q.data_ptr(), k.data_ptr(), v.data_ptr(), g.data_ptr(), beta.data_ptr(),
        rows, v_heads, v_heads, head, head, 2 * v_heads, 2 * v_heads,
        torch.cuda.current_stream(DEVICE).cuda_stream)
    torch.cuda.synchronize(DEVICE)

    assert torch.isfinite(g.float()).all()
    assert torch.allclose(g.float(), torch.full_like(g.float(), -200.0),
                          atol=1.0)


def test_a_value_head_count_that_is_not_a_multiple_is_refused():
    # Nothing about the layout says which key head a value head belongs to
    # unless the counts divide, so this is refused rather than rounded.
    head = 32
    buffer = torch.zeros(1, 4096, dtype=torch.bfloat16, device=DEVICE)
    scalar = torch.zeros(8, device=DEVICE)
    rc = fvk.linear_attn_split_broadcast_gate_bf16(
        buffer.data_ptr(), buffer.data_ptr(), buffer.data_ptr(),
        scalar.data_ptr(), scalar.data_ptr(),
        buffer.data_ptr(), buffer.data_ptr(), buffer.data_ptr(),
        buffer.data_ptr(), buffer.data_ptr(),
        1, 3, 8, head, head, 16, 16,
        torch.cuda.current_stream(DEVICE).cuda_stream)
    assert rc == -1


def test_the_decay_is_head_local():
    # Each head has its own decay constant, and a kernel that indexed the
    # constants by anything else would still produce a full row of gates.
    v_heads, head = 8, 64
    rows = 1
    conv = torch.zeros(rows, 3 * v_heads * head, dtype=torch.bfloat16,
                       device=DEVICE)
    ab = torch.zeros(rows, 2 * v_heads, dtype=torch.bfloat16, device=DEVICE)
    neg_exp_a_log = -torch.arange(1, v_heads + 1, device=DEVICE,
                                  dtype=torch.float32)
    dt_bias = torch.zeros(v_heads, device=DEVICE)

    q = torch.empty(rows, v_heads, head, dtype=torch.bfloat16, device=DEVICE)
    k, v = torch.empty_like(q), torch.empty_like(q)
    g = torch.empty(rows, v_heads, dtype=torch.bfloat16, device=DEVICE)
    beta = torch.empty_like(g)

    fvk.linear_attn_split_broadcast_gate_bf16(
        conv.data_ptr(), ab.data_ptr(), ab.data_ptr() + v_heads * 2,
        neg_exp_a_log.data_ptr(), dt_bias.data_ptr(),
        q.data_ptr(), k.data_ptr(), v.data_ptr(), g.data_ptr(), beta.data_ptr(),
        rows, v_heads, v_heads, head, head, 2 * v_heads, 2 * v_heads,
        torch.cuda.current_stream(DEVICE).cuda_stream)
    torch.cuda.synchronize(DEVICE)

    # softplus(0) = log 2, so head h decays by -(h + 1) * log 2.
    want = -torch.arange(1, v_heads + 1, device=DEVICE,
                         dtype=torch.float32) * math.log(2.0)
    assert torch.allclose(g.float().flatten(), want, atol=6e-3, rtol=6e-3)

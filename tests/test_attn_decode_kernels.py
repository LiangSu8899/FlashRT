"""Staging a fused QKV row, and the attention that reads what it staged.

The two kernels are checked against the arithmetic they replace and then
against each other, because the thing that breaks between them is a layout
agreement rather than a formula: the staging kernel decides where a head's
query, gate and cache slot go, and the attention kernel decides where it
looks for them. Both can be self-consistently wrong.
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

if not hasattr(fvk, "gqa_decode_attention_bf16"):        # pragma: no cover
    pytest.skip("built without the portable decode kernels",
                allow_module_level=True)

DEVICE = "cuda:0"
EPS = 1e-6


def rms_norm(x, weight, eps=EPS):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * \
        weight.float()


def rotate(x, cos, sin, rope_dim):
    """The reference rotation: half the rotated span swaps with the other."""
    out = x.clone()
    half = rope_dim // 2
    low, high = x[..., :half], x[..., half:rope_dim]
    out[..., :half] = low * cos - high * sin
    out[..., half:rope_dim] = high * cos + low * sin
    return out


def tables(positions, rope_dim, theta=1e7):
    index = torch.arange(0, rope_dim, 2, dtype=torch.float32, device=DEVICE)
    frequency = 1.0 / (theta ** (index / rope_dim))
    angle = torch.arange(positions, dtype=torch.float32,
                         device=DEVICE)[:, None] * frequency[None, :]
    return angle.cos().to(torch.bfloat16), angle.sin().to(torch.bfloat16)


def stage(qkv, q_norm_w, k_norm_w, cos, sin, k_cache, v_cache, pos,
          q_heads, kv_heads, head_dim, rope_dim, has_gate=True, rows=1):
    q_out = torch.empty(rows, q_heads, head_dim, dtype=torch.bfloat16,
                        device=DEVICE)
    gate_out = torch.empty(rows, q_heads * head_dim, dtype=torch.bfloat16,
                           device=DEVICE) if has_gate else None
    rc = fvk.attn_qkv_norm_rope_write_bf16(
        qkv.data_ptr(), q_norm_w.data_ptr(), k_norm_w.data_ptr(),
        cos.data_ptr(), sin.data_ptr(), q_out.data_ptr(),
        gate_out.data_ptr() if has_gate else 0,
        k_cache.data_ptr(), v_cache.data_ptr(),
        rows, pos, 0, q_heads, kv_heads, head_dim, rope_dim, has_gate, EPS,
        torch.cuda.current_stream(DEVICE).cuda_stream)
    assert rc == 0
    torch.cuda.synchronize(DEVICE)
    return q_out, gate_out


# The 4B full-attention layer, then a rotation covering the whole head and a
# geometry with no grouping at all.
GEOMETRIES = [(16, 4, 256, 64), (8, 8, 128, 128), (4, 2, 64, 32)]


@pytest.mark.parametrize("q_heads,kv_heads,head_dim,rope_dim", GEOMETRIES)
def test_staging_matches_the_arithmetic_it_replaces(
        q_heads, kv_heads, head_dim, rope_dim):
    torch.manual_seed(0)
    capacity, pos = 32, 7
    width = q_heads * head_dim * 2 + 2 * kv_heads * head_dim
    qkv = torch.randn(1, width, dtype=torch.bfloat16, device=DEVICE) * 0.5
    q_norm_w = torch.randn(head_dim, dtype=torch.bfloat16, device=DEVICE)
    k_norm_w = torch.randn(head_dim, dtype=torch.bfloat16, device=DEVICE)
    cos, sin = tables(capacity, rope_dim)
    k_cache = torch.zeros(capacity, kv_heads, head_dim, dtype=torch.bfloat16,
                          device=DEVICE)
    v_cache = torch.zeros_like(k_cache)

    q_out, gate_out = stage(qkv, q_norm_w, k_norm_w, cos, sin, k_cache,
                            v_cache, pos, q_heads, kv_heads, head_dim,
                            rope_dim)

    query_width = q_heads * head_dim * 2
    key_width = kv_heads * head_dim
    query = qkv[0, :query_width].reshape(q_heads, 2 * head_dim)
    want_q = rotate(rms_norm(query[:, :head_dim], q_norm_w),
                    cos[pos].float(), sin[pos].float(), rope_dim)
    want_gate = query[:, head_dim:].reshape(-1)
    key = qkv[0, query_width:query_width + key_width].reshape(kv_heads,
                                                              head_dim)
    want_k = rotate(rms_norm(key, k_norm_w), cos[pos].float(),
                    sin[pos].float(), rope_dim)
    want_v = qkv[0, query_width + key_width:].reshape(kv_heads, head_dim)

    assert torch.allclose(q_out[0].float(), want_q, atol=2e-2, rtol=2e-2)
    assert torch.equal(gate_out[0], want_gate)
    assert torch.allclose(k_cache[pos].float(), want_k, atol=2e-2, rtol=2e-2)
    assert torch.equal(v_cache[pos], want_v)


def test_staging_writes_only_the_position_it_was_given():
    # The cache is one allocation for the whole sequence, so a write that
    # strays is a previous token quietly replaced.
    torch.manual_seed(1)
    q_heads, kv_heads, head_dim, rope_dim = 4, 2, 64, 32
    capacity, pos = 16, 5
    width = q_heads * head_dim * 2 + 2 * kv_heads * head_dim
    qkv = torch.randn(1, width, dtype=torch.bfloat16, device=DEVICE)
    weight = torch.ones(head_dim, dtype=torch.bfloat16, device=DEVICE)
    cos, sin = tables(capacity, rope_dim)
    k_cache = torch.full((capacity, kv_heads, head_dim), 3.0,
                         dtype=torch.bfloat16, device=DEVICE)
    v_cache = torch.full_like(k_cache, 3.0)

    stage(qkv, weight, weight, cos, sin, k_cache, v_cache, pos,
          q_heads, kv_heads, head_dim, rope_dim)

    untouched = torch.cat([k_cache[:pos], k_cache[pos + 1:]])
    assert torch.all(untouched == 3.0)
    assert not torch.all(k_cache[pos] == 3.0)


def test_a_position_can_come_from_the_device():
    # A captured graph replays the addresses it was captured with, so the
    # position has to be readable from one rather than passed by value.
    torch.manual_seed(2)
    q_heads, kv_heads, head_dim, rope_dim = 4, 2, 64, 32
    capacity = 16
    width = q_heads * head_dim * 2 + 2 * kv_heads * head_dim
    qkv = torch.randn(1, width, dtype=torch.bfloat16, device=DEVICE)
    weight = torch.ones(head_dim, dtype=torch.bfloat16, device=DEVICE)
    cos, sin = tables(capacity, rope_dim)
    position = torch.tensor([9], dtype=torch.int32, device=DEVICE)

    from_host = torch.zeros(capacity, kv_heads, head_dim,
                            dtype=torch.bfloat16, device=DEVICE)
    from_host_v = torch.zeros_like(from_host)
    stage(qkv, weight, weight, cos, sin, from_host, from_host_v, 9,
          q_heads, kv_heads, head_dim, rope_dim)

    from_device = torch.zeros_like(from_host)
    from_device_v = torch.zeros_like(from_host)
    q_out = torch.empty(1, q_heads, head_dim, dtype=torch.bfloat16,
                        device=DEVICE)
    gate_out = torch.empty(1, q_heads * head_dim, dtype=torch.bfloat16,
                           device=DEVICE)
    fvk.attn_qkv_norm_rope_write_bf16(
        qkv.data_ptr(), weight.data_ptr(), weight.data_ptr(),
        cos.data_ptr(), sin.data_ptr(), q_out.data_ptr(), gate_out.data_ptr(),
        from_device.data_ptr(), from_device_v.data_ptr(),
        1, 0, position.data_ptr(), q_heads, kv_heads, head_dim, rope_dim,
        True, EPS, torch.cuda.current_stream(DEVICE).cuda_stream)
    torch.cuda.synchronize(DEVICE)

    assert torch.equal(from_host, from_device)


@pytest.mark.parametrize("q_heads,kv_heads,head_dim", [(16, 4, 256),
                                                       (8, 8, 128),
                                                       (4, 1, 64)])
@pytest.mark.parametrize("seq_len", [1, 7, 64, 300])
def test_attention_matches_an_explicit_softmax(q_heads, kv_heads, head_dim,
                                               seq_len):
    torch.manual_seed(3)
    scale = 1.0 / math.sqrt(head_dim)
    q = torch.randn(q_heads, head_dim, dtype=torch.bfloat16, device=DEVICE)
    k_cache = torch.randn(seq_len, kv_heads, head_dim, dtype=torch.bfloat16,
                          device=DEVICE)
    v_cache = torch.randn(seq_len, kv_heads, head_dim, dtype=torch.bfloat16,
                          device=DEVICE)
    out = torch.empty(q_heads, head_dim, dtype=torch.bfloat16, device=DEVICE)

    rc = fvk.gqa_decode_attention_bf16(
        q.data_ptr(), k_cache.data_ptr(), v_cache.data_ptr(), 0,
        out.data_ptr(), seq_len, 0, q_heads, kv_heads, head_dim, scale,
        torch.cuda.current_stream(DEVICE).cuda_stream)
    assert rc == 0
    torch.cuda.synchronize(DEVICE)

    group = q_heads // kv_heads
    keys = k_cache.float().repeat_interleave(group, dim=1)
    values = v_cache.float().repeat_interleave(group, dim=1)
    scores = torch.einsum("hd,shd->hs", q.float(), keys) * scale
    want = torch.einsum("hs,shd->hd", scores.softmax(-1), values)

    error = (out.float() - want).abs().max() / want.abs().max()
    assert error < 2e-2, f"relative error {error:.3g}"


def test_the_gate_is_applied_where_the_result_is_produced():
    # Applying the gate in the epilogue saves a pass over the output; the
    # thing to check is that it is the same gate, per head and per dimension.
    torch.manual_seed(4)
    q_heads, kv_heads, head_dim, seq_len = 8, 2, 128, 33
    scale = 1.0 / math.sqrt(head_dim)
    q = torch.randn(q_heads, head_dim, dtype=torch.bfloat16, device=DEVICE)
    k_cache = torch.randn(seq_len, kv_heads, head_dim, dtype=torch.bfloat16,
                          device=DEVICE)
    v_cache = torch.randn(seq_len, kv_heads, head_dim, dtype=torch.bfloat16,
                          device=DEVICE)
    gate = torch.randn(q_heads, head_dim, dtype=torch.bfloat16, device=DEVICE)

    plain = torch.empty(q_heads, head_dim, dtype=torch.bfloat16, device=DEVICE)
    gated = torch.empty_like(plain)
    stream = torch.cuda.current_stream(DEVICE).cuda_stream
    fvk.gqa_decode_attention_bf16(
        q.data_ptr(), k_cache.data_ptr(), v_cache.data_ptr(), 0,
        plain.data_ptr(), seq_len, 0, q_heads, kv_heads, head_dim, scale,
        stream)
    fvk.gqa_decode_attention_bf16(
        q.data_ptr(), k_cache.data_ptr(), v_cache.data_ptr(), gate.data_ptr(),
        gated.data_ptr(), seq_len, 0, q_heads, kv_heads, head_dim, scale,
        stream)
    torch.cuda.synchronize(DEVICE)

    want = plain.float() * torch.sigmoid(gate.float())
    assert torch.allclose(gated.float(), want, atol=2e-2, rtol=2e-2)


def test_a_length_can_come_from_the_device():
    torch.manual_seed(5)
    q_heads, kv_heads, head_dim, capacity = 8, 2, 128, 64
    scale = 1.0 / math.sqrt(head_dim)
    q = torch.randn(q_heads, head_dim, dtype=torch.bfloat16, device=DEVICE)
    k_cache = torch.randn(capacity, kv_heads, head_dim, dtype=torch.bfloat16,
                          device=DEVICE)
    v_cache = torch.randn(capacity, kv_heads, head_dim, dtype=torch.bfloat16,
                          device=DEVICE)
    length = torch.tensor([21], dtype=torch.int32, device=DEVICE)

    from_host = torch.empty(q_heads, head_dim, dtype=torch.bfloat16,
                            device=DEVICE)
    from_device = torch.empty_like(from_host)
    stream = torch.cuda.current_stream(DEVICE).cuda_stream
    fvk.gqa_decode_attention_bf16(
        q.data_ptr(), k_cache.data_ptr(), v_cache.data_ptr(), 0,
        from_host.data_ptr(), 21, 0, q_heads, kv_heads, head_dim, scale,
        stream)
    fvk.gqa_decode_attention_bf16(
        q.data_ptr(), k_cache.data_ptr(), v_cache.data_ptr(), 0,
        from_device.data_ptr(), 0, length.data_ptr(), q_heads, kv_heads,
        head_dim, scale, stream)
    torch.cuda.synchronize(DEVICE)

    assert torch.equal(from_host, from_device)


def test_query_heads_that_do_not_divide_are_refused():
    # Without a whole number of query heads per key head there is no rule for
    # which cache a head reads, so this is refused rather than rounded.
    buffer = torch.zeros(4096, dtype=torch.bfloat16, device=DEVICE)
    rc = fvk.gqa_decode_attention_bf16(
        buffer.data_ptr(), buffer.data_ptr(), buffer.data_ptr(), 0,
        buffer.data_ptr(), 4, 0, 6, 4, 64, 1.0,
        torch.cuda.current_stream(DEVICE).cuda_stream)
    assert rc == -1


def test_staging_and_attention_agree_on_where_a_head_lives():
    # End to end over the pair: stage several positions, then attend. A head
    # offset that both kernels share would pass each of the tests above.
    torch.manual_seed(6)
    q_heads, kv_heads, head_dim, rope_dim = 16, 4, 256, 64
    capacity = 24
    width = q_heads * head_dim * 2 + 2 * kv_heads * head_dim
    q_norm_w = torch.randn(head_dim, dtype=torch.bfloat16, device=DEVICE)
    k_norm_w = torch.randn(head_dim, dtype=torch.bfloat16, device=DEVICE)
    cos, sin = tables(capacity, rope_dim)
    k_cache = torch.zeros(capacity, kv_heads, head_dim, dtype=torch.bfloat16,
                          device=DEVICE)
    v_cache = torch.zeros_like(k_cache)

    length = 12
    keys, values = [], []
    for position in range(length):
        qkv = torch.randn(1, width, dtype=torch.bfloat16, device=DEVICE) * 0.5
        q_out, gate_out = stage(qkv, q_norm_w, k_norm_w, cos, sin, k_cache,
                                v_cache, position, q_heads, kv_heads,
                                head_dim, rope_dim)
        keys.append(k_cache[position].clone())
        values.append(v_cache[position].clone())

    out = torch.empty(q_heads, head_dim, dtype=torch.bfloat16, device=DEVICE)
    scale = 1.0 / math.sqrt(head_dim)
    fvk.gqa_decode_attention_bf16(
        q_out.data_ptr(), k_cache.data_ptr(), v_cache.data_ptr(),
        gate_out.data_ptr(), out.data_ptr(), length, 0, q_heads, kv_heads,
        head_dim, scale, torch.cuda.current_stream(DEVICE).cuda_stream)
    torch.cuda.synchronize(DEVICE)

    group = q_heads // kv_heads
    stacked_k = torch.stack(keys).float().repeat_interleave(group, dim=1)
    stacked_v = torch.stack(values).float().repeat_interleave(group, dim=1)
    scores = torch.einsum("hd,shd->hs", q_out[0].float(), stacked_k) * scale
    want = torch.einsum("hs,shd->hd", scores.softmax(-1), stacked_v)
    want = want * torch.sigmoid(gate_out[0].reshape(q_heads, head_dim).float())

    error = (out.float() - want).abs().max() / want.abs().max()
    assert error < 3e-2, f"relative error {error:.3g}"

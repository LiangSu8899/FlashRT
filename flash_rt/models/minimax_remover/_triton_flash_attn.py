"""
Triton Flash-Attention (sm120), providing both fp16 and fp8 implementations.

Motivation: MiniMax-Remover's attention accounts for 47% of a single step
(FA2 ≈ SDPA ≈ 6.3ms/layer, already at the fp16 compute roofline of 44 TFLOP/s).
This file provides two equivalent kernels:

  flash_attn_fp16(q,k,v,scale)  -- fp16 baseline (verified cos=1.0 vs FA2, same speed)
  flash_attn_fp8 (q,k,v,scale)  -- FP8 QK^T + fp32 online softmax + FP8 P·V -> fp16

Why FP8 can still be effective inside flash-attention (even though a standalone fp8
dot is only 1.2~1.4x faster):
  For each Q-block, FA must iterate over all K/V blocks; the larger S is, the more the
  HBM->SRAM loading of K/V dominates (memory-bound tendency). Storing K/V as fp8 directly
  halves the required memory bandwidth for this part, and the tensor-core computation
  also runs in fp8 (higher throughput). Combined, end-to-end this can be 1.3~1.7x faster
  than fp16 FA.

Input layout: [B, S, H, Dd] (matches the q/k/v view of FlashRTFA2Processor, zero-copy).
Non-causal (full attention). head_dim is fixed at 128 (for this model).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fa_fwd_kernel(
    Q, K, V, O,
    sm_scale,
    stride_qb, stride_qs, stride_qh,
    stride_kb, stride_ks, stride_kh,
    stride_vb, stride_vs, stride_vh,
    stride_ob, stride_os, stride_oh,
    S,
    H: tl.constexpr, Dd: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr,
    USE_FP8: tl.constexpr,
):
    """A single program handles (batch, head, one Q-block). Online (streaming) softmax.

    Q,K,V layout [B,S,H,Dd] row-major (stride_d=1)."""
    pid_m = tl.program_id(0)   # Q-block index
    pid_h = tl.program_id(1)   # head
    pid_b = tl.program_id(2)   # batch

    rm = pid_m * BM + tl.arange(0, BM)        # [BM] Q rows
    rk = tl.arange(0, Dd)                      # [Dd] head dim
    rn = tl.arange(0, BN)                      # [BN] K/V rows (within block)

    q_base = Q + pid_b * stride_qb + pid_h * stride_qh
    k_base = K + pid_b * stride_kb + pid_h * stride_kh
    v_base = V + pid_b * stride_vb + pid_h * stride_vh

    # Q tile [BM, Dd] (Q loaded only once)
    q_mask = rm[:, None] < S
    q = tl.load(q_base + rm[:, None] * stride_qs + rk[None, :],
                mask=q_mask, other=0.0)
    if USE_FP8:
        q = q.to(tl.float8e4nv)

    # Online softmax accumulators
    m_i = tl.full([BM], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BM], dtype=tl.float32)
    acc = tl.zeros((BM, Dd), dtype=tl.float32)

    n_blocks = tl.cdiv(S, BN)
    for j in range(0, n_blocks):
        kj = j * BN + rn                       # [BN] K/V rows in this block
        kv_mask = kj < S
        # K tile [BN, Dd]
        k = tl.load(k_base + kj[:, None] * stride_ks + rk[None, :],
                    mask=kv_mask[:, None], other=0.0)
        # V tile [BN, Dd]
        v = tl.load(v_base + kj[:, None] * stride_vs + rk[None, :],
                    mask=kv_mask[:, None], other=0.0)

        if USE_FP8:
            k = k.to(tl.float8e4nv)
            v = v.to(tl.float8e4nv)

        # QK^T: [BM, Dd] @ [Dd, BN] -> [BM, BN]
        qk = tl.dot(q, tl.trans(k)).to(tl.float32) * sm_scale
        # Set positions outside the K/V block (padding) to -inf, excluding from softmax
        qk = tl.where(kv_mask[None, :], qk, -float('inf'))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))         # [BM] new running max
        alpha = tl.exp(m_i - m_ij)                          # scale old accumulator
        p = tl.exp(qk - m_ij[:, None])                      # [BM, BN] unnormalized probs
        l_ij = tl.sum(p, axis=1)                            # [BM] prob sum of this block

        # P · V : [BM, BN] @ [BN, Dd] -> [BM, Dd]
        if USE_FP8:
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.float8e4nv), v).to(tl.float32)
        else:
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_ij
        l_i = l_i * alpha + l_ij

    # Normalize and write out fp16
    out = acc / l_i[:, None]
    o_base = O + pid_b * stride_ob + pid_h * stride_oh
    tl.store(o_base + rm[:, None] * stride_os + rk[None, :],
             out.to(tl.float16), mask=q_mask)


def _launch(q, k, v, scale, use_fp8, bm=128, bn=128, num_stages=2, num_warps=8):
    """q,k,v: [B,S,H,Dd] fp16 contiguous. Returns out [B,S,H,Dd] fp16.

    Default BM=128/BN=128: optimal for S=6688 in practice (fewer K/V blocks, better
    loading amortization).
    num_stages=2: with BM=BN=128, stages>=3 exceeds the 99KB shared memory limit on sm120."""
    assert q.dtype == torch.float16
    B, S, H, Dd = q.shape
    assert Dd == 128, "this kernel targets head_dim=128"
    out = torch.empty_like(q)
    grid = (triton.cdiv(S, bm), H, B)
    _fa_fwd_kernel[grid](
        q, k, v, out, scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        S, H=H, Dd=Dd, BM=bm, BN=bn,
        USE_FP8=use_fp8,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


def flash_attn_fp16(q, k, v, scale):
    """fp16 flash-attention (baseline). q,k,v: [B,S,H,Dd] fp16."""
    return _launch(q, k, v, scale, use_fp8=False)


def flash_attn_fp8(q, k, v, scale):
    """FP8 flash-attention: QK^T and P·V both use fp8 tensor cores; softmax in fp32."""
    return _launch(q, k, v, scale, use_fp8=True)

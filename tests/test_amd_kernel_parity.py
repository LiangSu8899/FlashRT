"""AMD pi05 kernel surface — numerical parity vs torch references.

Standalone parity gates for the representative kernels the Pi0.5 CDNA4
pipeline is assembled from, each checked against an independent torch
reference on real-distribution inputs (randn-scaled — constant or integer
ramps would hide calibration- and rounding-path bugs):

    rms_norm / layer_norm     — norm math + weight semantics (plain ``w``,
                                NOT the Gemma ``1+w`` fold: the fold happens
                                at weight-conversion time, so a kernel that
                                secretly applied 1+w would double it)
    residual_add              — bit-exact fp32-add/bf16-round contract
    gelu_inplace              — tanh-approx GELU (not SiLU, not erf-GELU)
    qkv_split_rope            — interleaved-pair rotation + V passthrough
    quantize_fp8_static       — BYTE-exact vs torch float8_e4m3fn with the
                                SAME scale (never compared to raw fp32)
    GemmRunner bf16_nn        — hipBLASLt BF16 layout convention
    GemmRunner fp8_nn/nt_dev  — FP8 layouts + device-scale semantics
    attention_decoder_gqa     — split-KV flash decoder vs exact softmax
                                reference, exact and seqused (fixed-shape
                                graph) modes

Binding signatures were read from csrc/amd/bindings.cpp and
csrc/amd/gemm/bindings_gemm.inc before writing any call here; every tensor
is bound to a named variable before ``.data_ptr()`` (an inline temporary
would be GC'd mid-launch).

Skip conditions: ROCm torch + a visible device + the built extension. On
NVIDIA/no-ROCm CI the module collects and every test skips with a reason.
"""

from __future__ import annotations

import importlib
import math

import numpy as np
import pytest


@pytest.fixture(scope="module")
def env():
    """(torch, extension) on a ROCm box with the extension built."""
    try:
        ext = importlib.import_module("flash_rt.amd.flash_rt_amd_kernels")
    except ImportError as exc:
        pytest.skip(f"flash_rt_amd_kernels not importable: {exc}")
    torch = pytest.importorskip("torch")
    if not getattr(torch.version, "hip", None):
        pytest.skip("torch is not a ROCm build")
    if not torch.cuda.is_available():
        pytest.skip("no ROCm device visible to torch")
    return torch, ext


def _stream(torch) -> int:
    """Launch kernels on torch's current stream so torch-side tensor prep
    and the raw-pointer launches are ordered without extra events."""
    return torch.cuda.current_stream().cuda_stream


def _cos(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


# Numerical gates, justified once:
#  - COS_ELEMENTWISE 0.9999: elementwise/norm kernels share the reference's
#    fp32 math except reduction order → agreement is a few bf16 ULPs; any
#    real bug (wrong weight semantics, shifted row, wrong pair) lands far
#    below this.
#  - ATOL_BF16 2e-2: one bf16 ULP at unit scale is ~2^-8 ≈ 4e-3; 2e-2 gives
#    headroom for a couple of chained roundings while still failing hard on
#    O(1) math errors.
#  - COS_GEMM 0.9999: hipBLASLt accumulates fp32 like the torch reference;
#    only split-K order and the bf16 store differ.
#  - COS_ATTN 0.999: attention output is bf16 with fp32 softmax; matches
#    the gate used for this kernel's bring-up validation.
COS_ELEMENTWISE = 0.9999
ATOL_BF16 = 2e-2
COS_GEMM = 0.9999
COS_ATTN = 0.999


# ---------------------------------------------------------------------------
# Norm kernels
# ---------------------------------------------------------------------------


def test_rms_norm_matches_torch(env):
    torch, ext = env
    seq, dim = 64, 2048  # encoder width; dim must be even (packed pairs)
    torch.manual_seed(0)
    x = (0.5 * torch.randn(seq, dim, device="cuda")).to(torch.bfloat16)
    w = (1.0 + 0.1 * torch.randn(dim, device="cuda")).to(torch.bfloat16)
    out = torch.empty_like(x)

    ext.rms_norm(x.data_ptr(), w.data_ptr(), out.data_ptr(),
                 seq, dim, 1e-6, _stream(torch))
    torch.cuda.synchronize()

    xf = x.float()
    ref = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)
           * w.float()).to(torch.bfloat16)
    assert _cos(out.float().cpu(), ref.float().cpu()) > COS_ELEMENTWISE
    torch.testing.assert_close(out.float(), ref.float(),
                               atol=ATOL_BF16, rtol=0.0)


def test_layer_norm_matches_torch(env):
    torch, ext = env
    seq, dim = 64, 1152  # SigLIP width
    torch.manual_seed(1)
    x = (0.5 * torch.randn(seq, dim, device="cuda")).to(torch.bfloat16)
    w = (1.0 + 0.1 * torch.randn(dim, device="cuda")).to(torch.bfloat16)
    b = (0.1 * torch.randn(dim, device="cuda")).to(torch.bfloat16)
    out = torch.empty_like(x)

    ext.layer_norm(x.data_ptr(), w.data_ptr(), b.data_ptr(), out.data_ptr(),
                   seq, dim, 1e-6, _stream(torch))
    torch.cuda.synchronize()

    xf = x.float()
    mean = xf.mean(-1, keepdim=True)
    var = (xf - mean).pow(2).mean(-1, keepdim=True)
    ref = ((xf - mean) * torch.rsqrt(var + 1e-6) * w.float()
           + b.float()).to(torch.bfloat16)
    assert _cos(out.float().cpu(), ref.float().cpu()) > COS_ELEMENTWISE
    torch.testing.assert_close(out.float(), ref.float(),
                               atol=ATOL_BF16, rtol=0.0)


# ---------------------------------------------------------------------------
# Elementwise
# ---------------------------------------------------------------------------


def test_residual_add_is_bit_exact(env):
    """residual += x is one fp32 add + one RNE bf16 round per element in
    both kernel and reference — nothing order-dependent, so the gate is
    bitwise equality, not a tolerance."""
    torch, ext = env
    n = 64 * 2048
    torch.manual_seed(2)
    residual = torch.randn(n, device="cuda").to(torch.bfloat16)
    x = torch.randn(n, device="cuda").to(torch.bfloat16)
    ref = (residual.float() + x.float()).to(torch.bfloat16)

    ext.residual_add(residual.data_ptr(), x.data_ptr(), n, _stream(torch))
    torch.cuda.synchronize()

    assert torch.equal(residual.view(torch.uint16), ref.view(torch.uint16)), (
        "residual_add is not bit-exact vs fp32-add + bf16-RNE")


def test_gelu_inplace_is_tanh_approx(env):
    """Must be the tanh-approx GELU — erf-GELU or SiLU here would pass a
    loose cos gate on symmetric data, so the reference pins the exact
    approximate= variant."""
    torch, ext = env
    n = 64 * 1024
    torch.manual_seed(3)
    x = (0.7 * torch.randn(n, device="cuda")).to(torch.bfloat16)
    ref = torch.nn.functional.gelu(
        x.float(), approximate="tanh").to(torch.bfloat16)

    ext.gelu_inplace(x.data_ptr(), n, _stream(torch))
    torch.cuda.synchronize()

    assert _cos(x.float().cpu(), ref.float().cpu()) > COS_ELEMENTWISE
    torch.testing.assert_close(x.float(), ref.float(),
                               atol=ATOL_BF16, rtol=0.0)


# ---------------------------------------------------------------------------
# QKV split + RoPE
# ---------------------------------------------------------------------------


def test_qkv_split_rope_matches_reference(env):
    """Interleaved-pair rotation (verified in csrc/amd/kernels/rope.hip):
    within each head, elements (2p, 2p+1) rotate with rope_weights[row]
    laid out cos@even / sin@odd over head_dim. Q: all heads rotated; K:
    single head, same per-row table; V: pure copy (bit-exact gate)."""
    torch, ext = env
    seq, hd = 16, 256
    q_dim, k_dim, v_dim = 8 * hd, hd, hd
    torch.manual_seed(4)
    qkv = (0.5 * torch.randn(seq, q_dim + k_dim + v_dim,
                             device="cuda")).to(torch.bfloat16)
    # Real rotary table (positions x standard inv-freq), interleaved.
    pos = torch.arange(seq, dtype=torch.float32, device="cuda")
    inv_freq = 1.0 / (10000.0 ** (
        torch.arange(0, hd, 2, dtype=torch.float32, device="cuda") / hd))
    ang = pos[:, None] * inv_freq[None, :]           # (seq, hd/2)
    rope = torch.empty(seq, hd, device="cuda")
    rope[:, 0::2] = torch.cos(ang)
    rope[:, 1::2] = torch.sin(ang)
    rope = rope.to(torch.bfloat16)

    Q = torch.empty(seq, q_dim, dtype=torch.bfloat16, device="cuda")
    K = torch.empty(seq, k_dim, dtype=torch.bfloat16, device="cuda")
    V = torch.empty(seq, v_dim, dtype=torch.bfloat16, device="cuda")

    ext.qkv_split_rope(qkv.data_ptr(), rope.data_ptr(),
                       Q.data_ptr(), K.data_ptr(), V.data_ptr(),
                       seq, q_dim, k_dim, v_dim, hd, _stream(torch))
    torch.cuda.synchronize()

    c = rope.float()[:, 0::2]
    s = rope.float()[:, 1::2]

    def rot(part):
        xf = part.float().reshape(seq, -1, hd // 2, 2)
        x0, x1 = xf[..., 0], xf[..., 1]
        cc, ss = c[:, None, :], s[:, None, :]
        even = x0 * cc - x1 * ss
        odd = x1 * cc + x0 * ss
        return torch.stack([even, odd], dim=-1).reshape(seq, -1)

    q_ref = rot(qkv[:, :q_dim]).to(torch.bfloat16)
    k_ref = rot(qkv[:, q_dim:q_dim + k_dim]).to(torch.bfloat16)
    v_ref = qkv[:, q_dim + k_dim:]

    assert torch.equal(V.view(torch.uint16), v_ref.contiguous().view(torch.uint16)), (
        "V must be a bit-exact passthrough")
    for got, ref, name in [(Q, q_ref, "Q"), (K, k_ref, "K")]:
        assert _cos(got.float().cpu(), ref.float().cpu()) > COS_ELEMENTWISE, name
        torch.testing.assert_close(got.float(), ref.float(),
                                   atol=ATOL_BF16, rtol=0.0)


# ---------------------------------------------------------------------------
# FP8 quantize
# ---------------------------------------------------------------------------


def test_quantize_fp8_static_byte_exact_vs_torch(env):
    """BYTE comparison against torch's float8_e4m3fn cast with the SAME
    scale, replicating the kernel's exact arithmetic (multiply by the fp32
    reciprocal of the scale, clamp to ±448, RNE convert — verified in
    csrc/amd/kernels/quantize_fp8.hip). Comparing against unquantized fp32
    would be a category error: the contract is the encoding, not
    closeness."""
    torch, ext = env
    n = 32768
    torch.manual_seed(5)
    x = (2.0 * torch.randn(n, device="cuda")).to(torch.bfloat16)
    # Production-style scale: amax/448 (per-tensor symmetric).
    amax = x.float().abs().max().item()
    scale = torch.tensor([max(amax / 448.0, 1e-12)],
                         dtype=torch.float32, device="cuda")
    out = torch.empty(n, dtype=torch.float8_e4m3fn, device="cuda")

    ext.quantize_fp8_static(x.data_ptr(), out.data_ptr(), scale.data_ptr(),
                            n, _stream(torch))
    torch.cuda.synchronize()

    inv_s = torch.reciprocal(scale)          # fp32, same op as the kernel
    ref = (x.float() * inv_s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    mismatched = int((out.view(torch.uint8) != ref.view(torch.uint8))
                     .sum().item())
    assert mismatched == 0, (
        f"{mismatched}/{n} FP8 bytes differ from torch e4m3 with the same "
        "scale — encoding drift would silently poison every FP8 GEMM")


# ---------------------------------------------------------------------------
# GemmRunner (hipBLASLt)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gemm(env):
    _, ext = env
    return ext.GemmRunner()


def test_gemm_bf16_nn_matches_torch(env, gemm):
    """bf16_nn contract: D(M,N) = A(M,K) @ B(K,N), all row-major, no
    transpose (the col-major swap inside the runner must be invisible)."""
    torch, _ = env
    M, N, K = 64, 512, 1024
    torch.manual_seed(6)
    A = (0.3 * torch.randn(M, K, device="cuda")).to(torch.bfloat16)
    B = (0.3 * torch.randn(K, N, device="cuda")).to(torch.bfloat16)
    D = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

    gemm.bf16_nn(A.data_ptr(), B.data_ptr(), D.data_ptr(), M, N, K,
                 _stream(torch))
    torch.cuda.synchronize()

    ref = (A.float() @ B.float())
    assert _cos(D.float().cpu(), ref.cpu()) > COS_GEMM
    # A transposed-layout bug produces garbage, not near-misses; the loose
    # allclose is a shape/layout tripwire on top of the cos gate.
    torch.testing.assert_close(D.float(), ref, atol=0.5, rtol=0.05)


def test_gemm_fp8_nn_dev_matches_quantized_reference(env, gemm):
    """fp8_nn_dev: D_bf16 = (A_fp8(M,K) @ B_fp8(K,N)) * sa * sb, with the
    scales read from DEVICE pointers (A/B_SCALE_POINTER semantics). The
    reference multiplies the same fp8-decoded operands — never the
    pre-quantization fp32 originals."""
    torch, _ = env
    M, N, K = 64, 512, 1024
    torch.manual_seed(7)
    A = (0.3 * torch.randn(M, K, device="cuda")).to(torch.float8_e4m3fn)
    B = (0.3 * torch.randn(K, N, device="cuda")).to(torch.float8_e4m3fn)
    sa = torch.tensor([0.01], dtype=torch.float32, device="cuda")
    sb = torch.tensor([0.02], dtype=torch.float32, device="cuda")
    D = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

    gemm.fp8_nn_dev(A.data_ptr(), B.data_ptr(), D.data_ptr(), M, N, K,
                    sa.data_ptr(), sb.data_ptr(), _stream(torch))
    torch.cuda.synchronize()

    ref = (A.float() @ B.float()) * (sa * sb)
    assert _cos(D.float().cpu(), ref.cpu()) > COS_GEMM
    torch.testing.assert_close(D.float(), ref, atol=0.05, rtol=0.05)


def test_gemm_fp8_nt_dev_matches_quantized_reference(env, gemm):
    """fp8_nt_dev: B stored (N,K) row-major, D = A @ B^T * sa * sb. This is
    the production pi05 layout (fp8_layout='nk', -2.8 ms E2E vs kn), so its
    transpose convention gets its own gate."""
    torch, _ = env
    M, N, K = 64, 512, 1024
    torch.manual_seed(8)
    A = (0.3 * torch.randn(M, K, device="cuda")).to(torch.float8_e4m3fn)
    Bt = (0.3 * torch.randn(N, K, device="cuda")).to(torch.float8_e4m3fn)
    sa = torch.tensor([0.01], dtype=torch.float32, device="cuda")
    sb = torch.tensor([0.02], dtype=torch.float32, device="cuda")
    D = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

    gemm.fp8_nt_dev(A.data_ptr(), Bt.data_ptr(), D.data_ptr(), M, N, K,
                    sa.data_ptr(), sb.data_ptr(), _stream(torch))
    torch.cuda.synchronize()

    ref = (A.float() @ Bt.float().t()) * (sa * sb)
    assert _cos(D.float().cpu(), ref.cpu()) > COS_GEMM
    torch.testing.assert_close(D.float(), ref, atol=0.05, rtol=0.05)


# ---------------------------------------------------------------------------
# Decoder GQA attention
# ---------------------------------------------------------------------------

# The one production site (csrc/amd/attention/decoder_flash.hip): Sq ~10
# action tokens, Hq=8 query heads, single KV head, D=256, Skv ~590-876.
_SQ, _HQ, _D = 10, 8, 256
_SCALE = 1.0 / math.sqrt(_D)  # 0.0625, the binding default


def _attn_ref(torch, Q, K, V, skv, scale):
    """Exact softmax reference in fp64 (bidirectional GQA Hq:1)."""
    q = Q.double().permute(1, 0, 2)          # (Hq, Sq, D)
    k = K[:skv].double()                     # (Skv, D)
    v = V[:skv].double()
    scores = q @ k.t() * scale               # (Hq, Sq, Skv)
    probs = torch.softmax(scores, dim=-1)
    o = probs @ v                            # (Hq, Sq, D)
    return o.permute(1, 0, 2).contiguous()   # (Sq, Hq, D)


def test_attention_decoder_gqa_exact_mode(env):
    torch, ext = env
    skv = 590
    torch.manual_seed(9)
    Q = (0.5 * torch.randn(_SQ, _HQ, _D, device="cuda")).to(torch.bfloat16)
    K = (0.5 * torch.randn(skv, _D, device="cuda")).to(torch.bfloat16)
    V = (0.5 * torch.randn(skv, _D, device="cuda")).to(torch.bfloat16)
    O = torch.empty_like(Q)
    # Caller-owned fp32 scratch, >= 32*Hq*Sq*(D+2) floats (binding contract;
    # the fused v3 path ignores it but the v2 split path requires it).
    ws = torch.empty(32 * _HQ * _SQ * (_D + 2),
                     dtype=torch.float32, device="cuda")

    ext.attention_decoder_gqa(Q.data_ptr(), K.data_ptr(), V.data_ptr(),
                              O.data_ptr(), ws.data_ptr(),
                              _SQ, skv, _HQ, _D, 0, _SCALE, _stream(torch))
    torch.cuda.synchronize()

    ref = _attn_ref(torch, Q, K, V, skv, _SCALE)
    assert _cos(O.float().cpu(), ref.cpu()) > COS_ATTN
    torch.testing.assert_close(O.double(), ref, atol=0.05, rtol=0.05)


def test_attention_decoder_gqa_seqused_masks_padding(env):
    """Fixed-shape graph mode: seqused is a DEVICE int32 whose [0] is the
    runtime KV length; the host Skv argument is ignored. The padded tail is
    filled with large garbage so any masking failure moves the output far
    outside the gate instead of hiding inside it."""
    torch, ext = env
    skv_pad, skv_eff = 640, 590
    torch.manual_seed(10)
    Q = (0.5 * torch.randn(_SQ, _HQ, _D, device="cuda")).to(torch.bfloat16)
    K = (0.5 * torch.randn(skv_pad, _D, device="cuda")).to(torch.bfloat16)
    V = (0.5 * torch.randn(skv_pad, _D, device="cuda")).to(torch.bfloat16)
    K[skv_eff:] = 30.0   # poison rows: huge scores if the mask leaks
    V[skv_eff:] = -30.0
    O = torch.empty_like(Q)
    ws = torch.empty(32 * _HQ * _SQ * (_D + 2),
                     dtype=torch.float32, device="cuda")
    seqused = torch.tensor([skv_eff], dtype=torch.int32, device="cuda")

    ext.attention_decoder_gqa(Q.data_ptr(), K.data_ptr(), V.data_ptr(),
                              O.data_ptr(), ws.data_ptr(),
                              _SQ, skv_pad, _HQ, _D,
                              seqused.data_ptr(), _SCALE, _stream(torch))
    torch.cuda.synchronize()

    ref = _attn_ref(torch, Q, K, V, skv_eff, _SCALE)
    assert _cos(O.float().cpu(), ref.cpu()) > COS_ATTN
    torch.testing.assert_close(O.double(), ref, atol=0.05, rtol=0.05)

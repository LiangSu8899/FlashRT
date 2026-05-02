"""FP8 GEMM as a ``torch.autograd.Function`` over a frozen base weight.

The forward path routes through FlashVLA's existing FP8 kernel
(``flash_rt.flash_rt_kernels.GemmRunner.fp8_nn_dev``) and the runtime
activation quantizer (``quantize_fp8_static``). Both are the same
kernels used by the inference pipeline — the training stack is
expected to "eat" the FP8 speedup, not just reuse a fp8 dtype tag in
PyTorch.

Backward is intentionally BF16: the base weight is frozen (no grad on
the FP8 storage tensor), so we only need to propagate ``grad_x = grad_y
@ W^T``. We dequantize ``W_fp8 → BF16`` into a *transient* buffer for
that GEMM and free it immediately, capping the per-step memory
overhead at one layer's worth of BF16 weight (≤ ~33 MB for pi0.5 FFN
``gate_up``). FP8 E5M2 backward is a W9 polish item — its API hook
lives in ``Fp8MatmulFunction.backward`` (``ctx.bwd_strategy``).

Numerics conventions match the inference path
(``flash_rt/core/weights/transformer.py:quantize_fp8_e4m3``):

* FP8 storage: ``torch.float8_e4m3fn`` viewed as ``torch.uint8`` bytes.
* Per-tensor scale ``s = amax / 448`` lives on-device as a 1-element
  ``float32`` tensor. ``quantize_fp8_static`` multiplies by ``1/s``
  internally; cuBLASLt FP8 matmul applies ``s_a * s_b`` to descale.
"""

from __future__ import annotations

import torch

_FP8_E4M3_MAX = 448.0
_FP8_SCALE_FLOOR = 1e-12

_RUNNER = None
_KERNELS = None


def _get_runner():
    """Lazily build the process-wide ``GemmRunner`` + kernel module handle."""
    global _RUNNER, _KERNELS
    if _RUNNER is None:
        from flash_rt import flash_rt_kernels as kern
        _KERNELS = kern
        _RUNNER = kern.GemmRunner()
    return _RUNNER, _KERNELS


def quantize_weight_to_fp8(w_bf16: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Per-tensor FP8 E4M3 quantize for a frozen weight tensor.

    Args:
        w_bf16: Input weight in BF16/FP16/FP32, any shape.

    Returns:
        A tuple ``(w_uint8_bytes, scale)`` where ``w_uint8_bytes`` carries
        the FP8 bit pattern in ``torch.uint8`` (same shape as input)
        and ``scale = amax / 448`` matches the inference quantizer.
    """
    w_f32 = w_bf16.detach().to(torch.float32)
    amax = float(w_f32.abs().max().item())
    scale = max(amax / _FP8_E4M3_MAX, _FP8_SCALE_FLOOR)
    w_scaled = (w_f32 / scale).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)
    w_fp8 = w_scaled.to(torch.float8_e4m3fn)
    w_uint8 = w_fp8.view(torch.uint8).contiguous()
    return w_uint8, scale


def dequant_fp8_to_bf16(w_uint8: torch.Tensor, scale: float) -> torch.Tensor:
    """Dequantize FP8 (uint8 storage) → BF16 with a per-tensor scale.

    Used on the backward path to materialise ``W_bf16`` for a frozen
    layer; the caller is expected to ``del`` the result immediately
    after the BF16 GEMM.
    """
    return (w_uint8.view(torch.float8_e4m3fn).to(torch.float32) * scale).to(torch.bfloat16)


# ── torch.library custom_ops for the FP8 forward GEMM ──────────────
#
# The vendored kernels (``flash_rt.flash_rt_kernels``) drive cuBLASLt
# FP8 directly via raw device pointers (``x.data_ptr()``). Plain Python
# wrappers around these calls are opaque to Dynamo / Inductor — when
# called from inside a ``torch.compile``'d region or a captured
# ``torch.cuda.graph`` the tracer cannot infer their output shape /
# dtype, so the produced ``y_flat`` tensor is left uninitialised and
# downstream consumers read garbage. Symptom observed in the W-Lever-A
# bench: training loss starts ~4× lower than eager and diverges within
# ~10 steps despite gradients being finite.
#
# Two shapes of the op are exposed:
#
# * ``flashvla::fp8_lora_gemm_bf16``           — allocating, returns y.
#   Convenient for eager + ``torch.compile(mode="default")``: Inductor
#   sees the op as a black box that produces a fresh tensor each call.
#   Internal ``torch.empty`` for the FP8-quantised activation scratch
#   and the BF16 output is fine here because Inductor doesn't try to
#   share addresses across calls.
#
# * ``flashvla::fp8_lora_gemm_bf16_into``      — non-allocating, mutates
#   pre-allocated ``x_fp8_scratch`` and ``y_out`` in place. Required for
#   ``torch.compile(mode="reduce-overhead")`` and ``torch.cuda.graph``
#   capture: cudagraph_trees only tracks tensor lifetimes for buffers
#   that flow through the traced region, so an op that allocates new
#   tensors internally for every call ends up referencing stale
#   addresses on the second replay → NaN at step 0. Pre-allocating in
#   the caller (``Fp8MatmulFunction.forward``) lets cudagraph_trees
#   manage the buffers in its private pool with stable addresses.
#
# Both ops route to the SAME kernel pair
# (``quantize_fp8_static`` + ``fp8_nn_dev``); the only difference is
# who owns the scratch and output buffers.
@torch.library.custom_op("flashvla::fp8_lora_gemm_bf16", mutates_args=())
def fp8_lora_gemm_bf16(
    x_bf16: torch.Tensor,
    w_fp8: torch.Tensor,
    act_scale_dev: torch.Tensor,
    weight_scale_dev: torch.Tensor,
) -> torch.Tensor:
    """Allocating variant: ``y = quantize_fp8(x_bf16) @ w_fp8`` (eager / compile-default).

    Both kernel launches go onto ``torch.cuda.current_stream()``. Without
    this the bindings default to ``stream=0`` (the legacy default stream)
    while torch.compile / cudagraph_trees capture on a per-thread stream
    — the kernels then escape capture and produce NaN at replay time.
    """
    runner, kern = _get_runner()
    K = x_bf16.shape[-1]
    K_w, N = w_fp8.shape
    if K != K_w:
        raise ValueError(f"fp8_lora_gemm_bf16: x last-dim {K} != W first-dim {K_w}")
    x_flat = x_bf16.reshape(-1, K).contiguous()
    M = x_flat.shape[0]
    stream = torch.cuda.current_stream(x_flat.device).cuda_stream
    x_fp8 = torch.empty(M, K, dtype=torch.uint8, device=x_flat.device)
    kern.quantize_fp8_static(
        x_flat.data_ptr(),
        x_fp8.data_ptr(),
        act_scale_dev.data_ptr(),
        M * K,
        stream,
    )
    y_flat = torch.empty(M, N, dtype=torch.bfloat16, device=x_flat.device)
    runner.fp8_nn_dev(
        x_fp8.data_ptr(),
        w_fp8.data_ptr(),
        y_flat.data_ptr(),
        M,
        N,
        K,
        act_scale_dev.data_ptr(),
        weight_scale_dev.data_ptr(),
        stream,
    )
    return y_flat.reshape(*x_bf16.shape[:-1], N)


@fp8_lora_gemm_bf16.register_fake
def _fp8_lora_gemm_bf16_fake(
    x_bf16: torch.Tensor,
    w_fp8: torch.Tensor,
    act_scale_dev: torch.Tensor,
    weight_scale_dev: torch.Tensor,
) -> torch.Tensor:
    _, N = w_fp8.shape
    return x_bf16.new_empty((*x_bf16.shape[:-1], N), dtype=torch.bfloat16)


@torch.library.custom_op(
    "flashvla::fp8_lora_gemm_bf16_into",
    mutates_args=("x_fp8_scratch", "y_out"),
)
def fp8_lora_gemm_bf16_into(
    x_bf16: torch.Tensor,
    w_fp8: torch.Tensor,
    act_scale_dev: torch.Tensor,
    weight_scale_dev: torch.Tensor,
    x_fp8_scratch: torch.Tensor,
    y_out: torch.Tensor,
) -> None:
    """Non-allocating variant for CUDA-Graph capture.

    Caller pre-allocates ``x_fp8_scratch`` (shape ``(M, K)`` uint8) and
    ``y_out`` (shape ``(*x_bf16.shape[:-1], N)`` bfloat16). The op
    writes both in place via the cuBLASLt FP8 path; declaring the
    mutation via ``mutates_args`` lets cudagraph_trees track buffer
    lifetimes through its private mempool so addresses stay stable
    across replays.
    """
    runner, kern = _get_runner()
    K = x_bf16.shape[-1]
    K_w, N = w_fp8.shape
    if K != K_w:
        raise ValueError(f"fp8_lora_gemm_bf16_into: x last-dim {K} != W first-dim {K_w}")
    M = x_fp8_scratch.shape[0]
    if x_fp8_scratch.shape != (M, K):
        raise ValueError(
            f"x_fp8_scratch shape {tuple(x_fp8_scratch.shape)} != (M={M}, K={K})"
        )
    if y_out.numel() != M * N:
        raise ValueError(
            f"y_out numel {y_out.numel()} != M*N = {M}*{N} = {M * N}"
        )
    x_flat = x_bf16.reshape(M, K).contiguous()
    stream = torch.cuda.current_stream(x_flat.device).cuda_stream
    kern.quantize_fp8_static(
        x_flat.data_ptr(),
        x_fp8_scratch.data_ptr(),
        act_scale_dev.data_ptr(),
        M * K,
        stream,
    )
    y_view = y_out.view(M, N)
    runner.fp8_nn_dev(
        x_fp8_scratch.data_ptr(),
        w_fp8.data_ptr(),
        y_view.data_ptr(),
        M,
        N,
        K,
        act_scale_dev.data_ptr(),
        weight_scale_dev.data_ptr(),
        stream,
    )


@fp8_lora_gemm_bf16_into.register_fake
def _fp8_lora_gemm_bf16_into_fake(
    x_bf16: torch.Tensor,
    w_fp8: torch.Tensor,
    act_scale_dev: torch.Tensor,
    weight_scale_dev: torch.Tensor,
    x_fp8_scratch: torch.Tensor,
    y_out: torch.Tensor,
) -> None:
    return None


class Fp8MatmulFunction(torch.autograd.Function):
    """``y = x @ W`` with FP8 forward and BF16 backward.

    Layout convention follows ``GemmRunner.fp8_nn_dev``:

        D_bf16(M, N) = A_fp8(M, K) @ B_fp8(K, N)

    ``W`` is therefore stored in ``(K, N)`` order, the *transpose* of
    HuggingFace's ``nn.Linear.weight`` ``(out_features=N, in_features=K)``.
    Callers (``Fp8LinearWithLora``) handle the one-shot transpose at
    quantize time.

    Backward path:
        * If ``w_bf16_cached`` is provided (i.e. the caller has already
          dequantised the frozen base weight once after calibration),
          backward is a single BF16 GEMM with no per-call dequant cost
          — matching an all-BF16 reference's backward latency exactly.
        * If ``None``, backward dequants on the fly into a transient
          buffer and frees it immediately. This path is correct but
          ~30% slower on small-shape GEMMs because of the extra kernel
          launches + tensor alloc; ``Fp8LinearWithLora`` populates the
          cache during ``compile`` so production training never takes
          this slow path.

    Backward also early-outs when ``x`` does not require gradients
    (``ctx.needs_input_grad[0] is False``) — saves the entire dequant +
    GEMM step in single-layer benchmarks where ``x`` is a leaf with
    no upstream consumers.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        x_bf16: torch.Tensor,
        w_fp8: torch.Tensor,
        act_scale_dev: torch.Tensor,
        weight_scale_dev: torch.Tensor,
        weight_scale_f: float,
        w_bf16_cached: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x_bf16.dtype != torch.bfloat16:
            x_bf16 = x_bf16.to(torch.bfloat16)

        K_w, N = w_fp8.shape
        # Pre-allocate the FP8 activation scratch and the BF16 output
        # in the caller. With ``torch.compile(mode="reduce-overhead")``
        # / ``torch.cuda.graph`` capture, both flow through cudagraph_trees'
        # private mempool so the kernel-side raw-pointer writes hit
        # stable addresses on every replay. The non-allocating variant
        # of the op (``fp8_lora_gemm_bf16_into``) declares the mutation
        # so the tracer treats both buffers as outputs.
        M = 1
        for d in x_bf16.shape[:-1]:
            M *= d
        x_fp8_scratch = x_bf16.new_empty((M, K_w), dtype=torch.uint8)
        y_out = x_bf16.new_empty((*x_bf16.shape[:-1], N), dtype=torch.bfloat16)
        fp8_lora_gemm_bf16_into(
            x_bf16, w_fp8, act_scale_dev, weight_scale_dev,
            x_fp8_scratch, y_out,
        )

        # Save only what backward actually needs.
        if w_bf16_cached is not None:
            ctx.save_for_backward(w_bf16_cached)
            ctx.has_cache = True
        else:
            ctx.save_for_backward(w_fp8)
            ctx.has_cache = False
        ctx.weight_scale_f = weight_scale_f
        ctx.orig_shape = x_bf16.shape
        ctx.N = N
        ctx.K = K_w

        return y_out

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):  # type: ignore[override]
        # Early out: nothing upstream needs grad_x. saves ~30% on
        # small-shape single-layer benchmarks.
        if not ctx.needs_input_grad[0]:
            return None, None, None, None, None, None

        (saved,) = ctx.saved_tensors
        N = ctx.N

        if grad_y.dtype != torch.bfloat16:
            grad_y = grad_y.to(torch.bfloat16)

        # grad_x = grad_y @ W^T_NK  : (..., N) @ (N, K) → (..., K)
        # The cached path stashes the (N, K) layout already, so the GEMM
        # is a direct ``grad_y @ cached`` with no per-call transpose.
        if ctx.has_cache:
            w_nk_bf16 = saved
            grad_y_flat = grad_y.reshape(-1, N)
            grad_x_flat = grad_y_flat @ w_nk_bf16
        else:
            w_kn_bf16 = dequant_fp8_to_bf16(saved, ctx.weight_scale_f)
            grad_y_flat = grad_y.reshape(-1, N).contiguous()
            grad_x_flat = grad_y_flat @ w_kn_bf16.t().contiguous()
            del w_kn_bf16

        grad_x = grad_x_flat.reshape(*ctx.orig_shape)
        return grad_x, None, None, None, None, None

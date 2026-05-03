"""JAX-side FP8 GEMM op backed by FlashRT's cuBLASLt kernel.

This module mirrors the PyTorch path's ``training.lora.fp8_autograd``
on the JAX side:

* :func:`quantize_weight_to_fp8_bytes` — host-side per-tensor quantize
  for a frozen weight (one-shot at LoRA injection time, in numpy).
* :func:`quantize_fp8_static` — JAX op that quantizes a BF16
  activation tensor with a pre-computed device scale, via the
  ``flashvla::quantize_fp8_static`` FFI handler. CUDA-Graph
  compatible (no host-amax sync).
* :func:`fp8_gemm_bf16_out` — JAX :class:`jax.custom_vjp` exposing
  ``y_bf16 = scale_a*scale_b * x_fp8 @ w_fp8``. Forward routes
  through the ``flashvla::fp8_gemm_bf16_out`` FFI; backward is a
  single BF16 GEMM via :func:`jnp.einsum`. **No FP8 backward** —
  same trade-off the PyTorch path makes (see
  ``training/README.md`` "Why FP8 backward is not implemented").

Numerics convention matches the inference + PyTorch training paths:

* FP8 storage: ``float8_e4m3fn`` viewed as ``uint8`` bytes.
* Per-tensor scale ``s = amax / 448`` lives on-device as a
  1-element ``float32`` array. The kernel multiplies by ``1/s``
  internally and cuBLASLt FP8 matmul applies ``s_a * s_b`` to
  descale.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .ffi_loader import ensure_registered

FP8_E4M3_MAX = 448.0
FP8_SCALE_FLOOR = 1e-12


# ── Host-side weight quantize (one-shot at LoRA injection) ─────────


def quantize_weight_to_fp8_bytes(
    w_bf16: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-tensor FP8 E4M3 quantize for a frozen weight tensor (numpy).

    Args:
        w_bf16: numpy array of shape ``(K, N)``, dtype bfloat16/float16/float32.

    Returns:
        ``(w_uint8, scale_f32)`` where ``w_uint8`` carries the FP8
        bit pattern in ``np.uint8`` (same shape as input) and
        ``scale_f32`` is a 1-element ``np.float32`` array
        (``amax / 448``).
    """
    if w_bf16.ndim != 2:
        raise ValueError(
            f"weight must be 2-D (K, N); got shape {w_bf16.shape}"
        )
    w_f32 = np.asarray(w_bf16, dtype=np.float32)
    amax = float(np.abs(w_f32).max())
    scale = max(amax / FP8_E4M3_MAX, FP8_SCALE_FLOOR)
    w_scaled = np.clip(w_f32 / scale, -FP8_E4M3_MAX, FP8_E4M3_MAX)
    # Round-to-FP8 by casting through ml_dtypes.float8_e4m3fn, then
    # reinterpret the bytes as uint8 to keep the array dtype-agnostic
    # downstream (XLA refuses implicit promotion on float8 dtypes —
    # see flashvla-ft/docs/results_fp8_forward.md).
    from ml_dtypes import float8_e4m3fn
    w_fp8 = w_scaled.astype(float8_e4m3fn)
    w_uint8 = w_fp8.view(np.uint8).copy()    # bytes-equivalent; .copy() to detach from src
    if w_uint8.shape != w_f32.shape:
        # ``.view(np.uint8)`` keeps shape because itemsize == 1 for both.
        raise RuntimeError(
            f"unexpected reshape during FP8 byte view: "
            f"{w_f32.shape} → {w_uint8.shape}"
        )
    return w_uint8, np.array([scale], dtype=np.float32)


# ── JAX FFI ops ────────────────────────────────────────────────────


def quantize_fp8_static(x_bf16: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
    """BF16 → FP8 (uint8 bytes), per-tensor static scale.

    Args:
        x_bf16: any-rank BF16 tensor.
        scale:  1-element float32 device array (``s = amax / 448``).

    Returns:
        ``uint8`` tensor with the same shape as ``x_bf16``, carrying
        the FP8 E4M3 bit pattern.
    """
    if x_bf16.dtype != jnp.bfloat16:
        x_bf16 = x_bf16.astype(jnp.bfloat16)
    if scale.dtype != jnp.float32:
        scale = scale.astype(jnp.float32)
    if scale.shape != (1,):
        scale = scale.reshape((1,))
    ensure_registered()
    out_type = jax.ShapeDtypeStruct(x_bf16.shape, jnp.uint8)
    return jax.ffi.ffi_call(
        "flashvla::quantize_fp8_static",
        out_type,
    )(x_bf16, scale)


def _fp8_gemm_ffi(
    x_fp8: jnp.ndarray,
    w_fp8: jnp.ndarray,
    act_scale: jnp.ndarray,
    w_scale: jnp.ndarray,
) -> jnp.ndarray:
    """Invoke the cuBLASLt FP8 GEMM via XLA FFI.

    Layout: ``y_bf16(M, N) = scale_a*scale_b * x_fp8(M, K) @ w_fp8(K, N)``.
    ``x_fp8`` may carry leading axes — they are flattened by the
    handler (last dim == K, all leading axes collapse to M).
    """
    ensure_registered()
    if x_fp8.ndim < 2:
        raise ValueError(f"x_fp8 must have rank ≥ 2; got {x_fp8.shape}")
    if w_fp8.ndim != 2:
        raise ValueError(f"w_fp8 must have rank 2 (K, N); got {w_fp8.shape}")
    if x_fp8.shape[-1] != w_fp8.shape[0]:
        raise ValueError(
            f"K mismatch: x last-dim {x_fp8.shape[-1]} vs w first-dim {w_fp8.shape[0]}"
        )
    out_shape = (*x_fp8.shape[:-1], w_fp8.shape[1])
    out_type = jax.ShapeDtypeStruct(out_shape, jnp.bfloat16)
    if act_scale.dtype != jnp.float32:
        act_scale = act_scale.astype(jnp.float32)
    if w_scale.dtype != jnp.float32:
        w_scale = w_scale.astype(jnp.float32)
    if act_scale.shape != (1,):
        act_scale = act_scale.reshape((1,))
    if w_scale.shape != (1,):
        w_scale = w_scale.reshape((1,))
    return jax.ffi.ffi_call(
        "flashvla::fp8_gemm_bf16_out",
        out_type,
    )(x_fp8, w_fp8, act_scale, w_scale)


# ── Custom VJP: FP8 forward, BF16 backward ─────────────────────────


@jax.custom_vjp
def fp8_gemm_bf16_out(
    x_bf16: jnp.ndarray,
    w_bf16: jnp.ndarray,
    w_fp8: jnp.ndarray,
    act_scale: jnp.ndarray,
    w_scale: jnp.ndarray,
) -> jnp.ndarray:
    """``y_bf16 = x_bf16 @ w`` with FP8 forward, BF16 backward.

    Args:
        x_bf16:    activation, shape ``(..., K)`` BF16.
        w_bf16:    BF16 mirror of the weight, shape ``(K, N)``. Used
                   only by backward — the forward path consumes
                   ``w_fp8`` instead. Caller is responsible for
                   keeping ``w_bf16`` and ``w_fp8`` consistent (e.g.
                   via :func:`quantize_weight_to_fp8_bytes`).
        w_fp8:     FP8 weight bytes, shape ``(K, N)`` uint8.
        act_scale: 1-element float32, ``amax_x / 448``.
        w_scale:   1-element float32, ``amax_w / 448``.

    Returns:
        ``y_bf16`` of shape ``(..., N)``.

    Backward:
        * ``grad_x = grad_y @ w_bf16.T``  (BF16 einsum).
        * ``grad_w_bf16 = x_bf16.T @ grad_y``  (BF16 einsum) —
          dropped downstream by the openpi LoRA freeze filter, but
          computed for symmetry with the un-patched
          ``jnp.einsum`` baseline.
        * ``w_fp8`` / ``act_scale`` / ``w_scale`` carry no grad
          (None tangents).
    """
    # The primary entry-point body must run *only* during forward.
    # custom_vjp routes through `_fwd` once defvjp is wired below.
    x_fp8 = quantize_fp8_static(x_bf16, act_scale)
    return _fp8_gemm_ffi(x_fp8, w_fp8, act_scale, w_scale)


def _fp8_gemm_bf16_out_fwd(
    x_bf16: jnp.ndarray,
    w_bf16: jnp.ndarray,
    w_fp8: jnp.ndarray,
    act_scale: jnp.ndarray,
    w_scale: jnp.ndarray,
):
    x_fp8 = quantize_fp8_static(x_bf16, act_scale)
    y = _fp8_gemm_ffi(x_fp8, w_fp8, act_scale, w_scale)
    # Save for backward. We save (x_bf16, w_bf16) — the bf16 mirror
    # of the weight is the residual cost of "speed-priority" recipe;
    # the memory-priority recipe (Phase 1+ option) instead saves
    # x_bf16 only and runs an FP8→BF16 dequant in backward. Phase 1
    # ships the simpler cached-bf16 path; the dequant variant can be
    # added in Phase 2 alongside the LoRA patch's memory toggle.
    return y, (x_bf16, w_bf16)


def _fp8_gemm_bf16_out_bwd(res, grad_y: jnp.ndarray):
    x_bf16, w_bf16 = res
    if grad_y.dtype != jnp.bfloat16:
        grad_y = grad_y.astype(jnp.bfloat16)
    # grad_x = grad_y @ w^T : (..., N) @ (K, N)^T = (..., K)
    grad_x = jnp.einsum("...n,kn->...k", grad_y, w_bf16).astype(jnp.bfloat16)
    # grad_w = x^T @ grad_y : (K, ...M) @ (...M, N) = (K, N) — flatten
    # leading axes by treating them as the contracted dim of the einsum.
    grad_w = jnp.einsum("...k,...n->kn", x_bf16, grad_y).astype(jnp.bfloat16)
    # custom_vjp expects a tangent for every primal input. ``w_fp8``,
    # ``act_scale``, ``w_scale`` are not differentiable for our use
    # case — return ``None`` to drop their tangents (PyTorch's
    # autograd.Function uses the same convention via
    # ``return None``-typed slots).
    return (grad_x, grad_w, None, None, None)


fp8_gemm_bf16_out.defvjp(_fp8_gemm_bf16_out_fwd, _fp8_gemm_bf16_out_bwd)

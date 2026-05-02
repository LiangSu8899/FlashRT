"""JAX equivalents of the torch-tensor-using helpers in ``flash_rt.core.rl.reward``.

The PyTorch reference is ``flash_rt/core/rl/reward.py``:

* :func:`build_bin_centers` — :func:`torch.linspace` → :func:`jnp.linspace`.
* :func:`project_values_to_bins` — bilinear soft-target projection,
  scatter_add → :func:`jnp.ndarray.at[...].add(...)`.
* :func:`expected_value_from_logits` — softmax · bin_centers reduction.
* :func:`compute_soft_value_loss` — log_softmax · soft_target sum-mean.

These are byte-for-byte algorithm clones of the torch versions; any
divergence shows up in Phase 3 indicator-parity tests.
"""

from __future__ import annotations

import jax.numpy as jnp

DEFAULT_NUM_BINS = 201
DEFAULT_BIN_MIN = -1.0
DEFAULT_BIN_MAX = 0.0


def build_bin_centers(
    num_bins: int = DEFAULT_NUM_BINS,
    bin_min: float = DEFAULT_BIN_MIN,
    bin_max: float = DEFAULT_BIN_MAX,
) -> jnp.ndarray:
    """Evenly spaced bin centres in ``[bin_min, bin_max]`` (float32)."""
    return jnp.linspace(bin_min, bin_max, num_bins, dtype=jnp.float32)


def project_values_to_bins(
    values: jnp.ndarray,
    bin_centers: jnp.ndarray,
) -> jnp.ndarray:
    """Bilinear projection of scalar values onto a bin distribution.

    Args:
        values: Rank-1 array of shape (B,).
        bin_centers: Rank-1 array of shape (num_bins,).

    Returns:
        Soft target distributions, shape (B, num_bins).
    """
    if values.ndim != 1:
        raise ValueError(f"'values' must be rank-1, got shape={values.shape}")
    if bin_centers.ndim != 1:
        raise ValueError(f"'bin_centers' must be rank-1, got shape={bin_centers.shape}")
    if bin_centers.shape[0] < 2:
        raise ValueError("at least 2 bins are required")

    values = jnp.clip(values, bin_centers[0], bin_centers[-1])
    step = bin_centers[1] - bin_centers[0]
    scaled = (values - bin_centers[0]) / step
    low = jnp.floor(scaled).astype(jnp.int32)
    high = jnp.clip(low + 1, a_max=bin_centers.shape[0] - 1)
    high_weight = jnp.clip(scaled - low.astype(jnp.float32), 0.0, 1.0)
    low_weight = 1.0 - high_weight

    B, num_bins = values.shape[0], bin_centers.shape[0]
    target = jnp.zeros((B, num_bins), dtype=jnp.float32)
    rows = jnp.arange(B)
    target = target.at[rows, low].add(low_weight)
    target = target.at[rows, high].add(high_weight)
    return target


def expected_value_from_logits(
    logits: jnp.ndarray,
    bin_centers: jnp.ndarray,
) -> jnp.ndarray:
    """``V = Σ softmax(logits) * bin_centers`` along the last axis."""
    probs = jax_softmax(logits, axis=-1)
    return jnp.sum(probs * bin_centers, axis=-1)


def jax_softmax(x: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    """Numerically stable softmax (matches torch.softmax semantics)."""
    x_max = jnp.max(x, axis=axis, keepdims=True)
    exp = jnp.exp(x - x_max)
    return exp / jnp.sum(exp, axis=axis, keepdims=True)


def jax_log_softmax(x: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    """Numerically stable log-softmax (matches torch.log_softmax)."""
    x_max = jnp.max(x, axis=axis, keepdims=True)
    return (x - x_max) - jnp.log(jnp.sum(jnp.exp(x - x_max), axis=axis, keepdims=True))


def compute_soft_value_loss(
    predicted_logits: jnp.ndarray,
    target_values: jnp.ndarray,
    bin_centers: jnp.ndarray,
    sample_weights: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Distributional VF loss with soft targets (π\\*0.6 Eq. 1).

    ``L = - Σ soft_target * log_softmax(logits)`` averaged over the
    batch. Mirrors :func:`flash_rt.core.rl.reward.compute_soft_value_loss`.
    """
    soft_target = project_values_to_bins(target_values, bin_centers)
    log_probs = jax_log_softmax(predicted_logits, axis=-1)
    per_sample = -jnp.sum(soft_target * log_probs, axis=-1)
    if sample_weights is not None:
        per_sample = per_sample * sample_weights
    return jnp.mean(per_sample)

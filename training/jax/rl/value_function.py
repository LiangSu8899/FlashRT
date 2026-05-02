"""Distributional value-function modules for RECAP — JAX nnx port.

Mirror of :mod:`flash_rt.core.rl.value_function` (PyTorch). Same
architecture (state-encoder MLP + value-head MLP), same defaults
(``num_bins=201``, support ``[-1, 0]``), same forward semantics. The
weight-transfer helpers in :mod:`training.jax.tests` exist to assert
forward-parity at cosine ≥ 0.997 against the PyTorch reference.

Three classes mirror the PyTorch surface 1-for-1:

* :class:`StandaloneValueFunction` — state-only VF for synthetic-data
  testing (``training/rl/train_value.py``);
* :class:`ValueFunctionHead` — 3-layer MLP head over a pooled VLA
  prefix embedding (paper §IV-A);
* :class:`Pi05ValueFunction` — frozen pi0.5 backbone reference + a
  trainable :class:`ValueFunctionHead`.

Wiring matches ``flash_rt.core.rl.value_function`` and
``training.rl.pi05_vf`` byte-for-byte at the architecture level.
"""

from __future__ import annotations

import flax.nnx as nnx
import jax.numpy as jnp

from ._reward_jax import (
    DEFAULT_BIN_MAX,
    DEFAULT_BIN_MIN,
    DEFAULT_NUM_BINS,
    build_bin_centers,
    expected_value_from_logits,
)


class _LinearLayerNormGeluDropout(nnx.Module):
    """Linear → LayerNorm → GELU → Dropout block, matching the PyTorch head."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(in_dim, out_dim, rngs=rngs)
        self.layernorm = nnx.LayerNorm(out_dim, rngs=rngs)
        self.dropout = nnx.Dropout(rate=dropout, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.linear(x)
        x = self.layernorm(x)
        x = nnx.gelu(x)
        x = self.dropout(x)
        return x


class _StateEncoder(nnx.Module):
    """Linear → LayerNorm → GELU → Linear, no dropout. Matches PyTorch."""

    def __init__(self, state_dim: int, hidden_dim: int, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(state_dim, hidden_dim, rngs=rngs)
        self.layernorm = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.linear1(x)
        x = self.layernorm(x)
        x = nnx.gelu(x)
        x = self.linear2(x)
        return x


class StandaloneValueFunction(nnx.Module):
    """State-only distributional VF mirroring the PyTorch reference 1-for-1.

    Architecture::

        state_encoder = Linear → LayerNorm → GELU → Linear
        value_head    = (Linear → LayerNorm → GELU → Dropout) × (num_layers-1)
                        → Linear(num_bins)

    Args:
        state_dim: Width of the per-frame state vector.
        hidden_dim: Hidden width of both encoder + head.
        num_bins: Distributional bin count.
        num_layers: Total layers in ``value_head`` (≥ 1).
        dropout: Dropout in the head.
        bin_min, bin_max: Support of the bin grid.
        rngs: nnx Rngs for parameter init + dropout.
    """

    def __init__(
        self,
        *,
        state_dim: int = 32,
        hidden_dim: int = 256,
        num_bins: int = DEFAULT_NUM_BINS,
        num_layers: int = 3,
        dropout: float = 0.1,
        bin_min: float = DEFAULT_BIN_MIN,
        bin_max: float = DEFAULT_BIN_MAX,
        rngs: nnx.Rngs,
    ):
        if num_layers < 1:
            raise ValueError(f"num_layers must be ≥ 1, got {num_layers}")
        self.num_bins = num_bins
        self.bin_min = bin_min
        self.bin_max = bin_max
        self.num_layers = num_layers

        self.state_encoder = _StateEncoder(state_dim, hidden_dim, rngs=rngs)

        # value_head: matches the PyTorch loop construction. The block
        # before the final Linear is (num_layers - 1) repetitions of
        # the LinearLayerNormGeluDropout block.
        self.head_blocks = [
            _LinearLayerNormGeluDropout(
                hidden_dim, hidden_dim, dropout=dropout, rngs=rngs,
            )
            for _ in range(num_layers - 1)
        ]
        self.head_final = nnx.Linear(hidden_dim, num_bins, rngs=rngs)

        # Bin centres are buffer-equivalent in PyTorch
        # (``register_buffer(..., persistent=False)``) — i.e. not a
        # trainable param. Store as a plain attribute so nnx skips it
        # from the param-state filter automatically.
        # bin_centers is a deterministic constant of (num_bins, bin_min,
        # bin_max). Storing the materialised jnp.ndarray as an attribute
        # on an nnx.Module trips ``nnx.state(...)`` (raises on raw array
        # leaves), so expose it as a lazy property instead — JAX folds
        # the linspace into a compile-time constant under jit. Same
        # intent as PyTorch's register_buffer(persistent=False).

    @property
    def bin_centers(self) -> jnp.ndarray:
        return build_bin_centers(self.num_bins, self.bin_min, self.bin_max)

    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        """Distributional logits over bins, shape ``(B, num_bins)``."""
        feat = self.state_encoder(state)
        for block in self.head_blocks:
            feat = block(feat)
        return self.head_final(feat)

    def predict_value(self, state: jnp.ndarray) -> jnp.ndarray:
        """Continuous expected value, shape ``(B,)``."""
        logits = self(state)
        return expected_value_from_logits(logits, self.bin_centers)


# ── Pooled-prefix-embedding head (paper §IV-A) ─────────────────────


def pool_masked_mean(
    embeddings: jnp.ndarray,
    pad_mask: jnp.ndarray,
    *,
    eps: float = 1e-6,
) -> jnp.ndarray:
    """Mean-pool ``embeddings`` along the token axis using ``pad_mask``.

    JAX equivalent of :func:`training.rl.pi05_vf.pool_masked_mean`.

    Args:
        embeddings: ``(B, T, D)`` token embeddings.
        pad_mask: ``(B, T)`` boolean / 0-1 array; ``True``/``1`` marks
            valid (non-padded) positions, ``False``/``0`` is ignored.
        eps: Numerical floor for the per-row token count, in case a
            row has zero valid tokens. The fallback returns the
            unweighted mean of zeros (i.e. zero) for that row.

    Returns:
        Pooled embeddings of shape ``(B, D)`` in the same dtype as
        the input embeddings.
    """
    if embeddings.ndim != 3:
        raise ValueError(
            f"embeddings must be (B, T, D), got shape={embeddings.shape}"
        )
    if pad_mask.ndim != 2 or pad_mask.shape != embeddings.shape[:2]:
        raise ValueError(
            f"pad_mask shape {pad_mask.shape} must match "
            f"(B, T) of embeddings ({embeddings.shape[:2]})"
        )

    mask = pad_mask.astype(embeddings.dtype)[:, :, None]   # (B, T, 1)
    masked_sum = jnp.sum(embeddings * mask, axis=1)        # (B, D)
    counts = jnp.maximum(jnp.sum(mask, axis=1), eps)       # (B, 1)
    return masked_sum / counts


class ValueFunctionHead(nnx.Module):
    """Distributional VF head over a pooled VLA prefix embedding.

    Architecture mirrors :class:`flash_rt.core.rl.value_function.ValueFunctionHead`
    1-for-1::

        mlp = (Linear → LayerNorm → GELU → Dropout) × (num_layers - 1)
              → Linear(num_bins)

    Inputs are pre-pooled embeddings of shape ``(B, input_dim)``;
    pooling itself is the caller's responsibility (see
    :func:`pool_masked_mean`).

    Args:
        input_dim: Width of the pooled prefix embedding (``2048`` for
            pi0.5's Gemma-2B encoder).
        hidden_dim: Hidden width of the MLP.
        num_bins: Number of distributional value bins.
        num_layers: Total layers in the MLP (≥ 1).
        dropout: Dropout in the MLP body.
        bin_min, bin_max: Support of the bin grid.
        rngs: nnx Rngs for parameter init + dropout.
    """

    def __init__(
        self,
        *,
        input_dim: int = 2048,
        hidden_dim: int = 512,
        num_bins: int = DEFAULT_NUM_BINS,
        num_layers: int = 3,
        dropout: float = 0.1,
        bin_min: float = DEFAULT_BIN_MIN,
        bin_max: float = DEFAULT_BIN_MAX,
        rngs: nnx.Rngs,
    ):
        if num_layers < 1:
            raise ValueError(f"num_layers must be ≥ 1, got {num_layers}")
        self.input_dim = input_dim
        self.num_bins = num_bins
        self.bin_min = bin_min
        self.bin_max = bin_max
        self.num_layers = num_layers

        self.head_blocks = [
            _LinearLayerNormGeluDropout(
                input_dim if i == 0 else hidden_dim,
                hidden_dim,
                dropout=dropout,
                rngs=rngs,
            )
            for i in range(num_layers - 1)
        ]
        # When num_layers == 1, the head is a single Linear from input_dim
        # to num_bins (no hidden block at all).
        head_in = input_dim if num_layers == 1 else hidden_dim
        self.head_final = nnx.Linear(head_in, num_bins, rngs=rngs)

        # bin_centers is a deterministic constant of (num_bins, bin_min,
        # bin_max). Storing the materialised jnp.ndarray as an attribute
        # on an nnx.Module trips ``nnx.state(...)`` (raises on raw array
        # leaves), so expose it as a lazy property instead — JAX folds
        # the linspace into a compile-time constant under jit. Same
        # intent as PyTorch's register_buffer(persistent=False).

    @property
    def bin_centers(self) -> jnp.ndarray:
        return build_bin_centers(self.num_bins, self.bin_min, self.bin_max)

    def __call__(self, embeddings: jnp.ndarray) -> jnp.ndarray:
        """Distributional logits over bins, shape ``(B, num_bins)``."""
        x = embeddings
        for block in self.head_blocks:
            x = block(x)
        return self.head_final(x)

    def predict_value(self, embeddings: jnp.ndarray) -> jnp.ndarray:
        """Continuous expected value, shape ``(B,)``."""
        logits = self(embeddings)
        return expected_value_from_logits(logits, self.bin_centers)


class Pi05ValueFunction(nnx.Module):
    """Frozen pi0.5 backbone reference + trainable distributional VF head.

    JAX equivalent of :class:`training.rl.pi05_vf.Pi05ValueFunction`.

    The backbone is held by reference only — the VF wrapper does
    **not** register it as a child module (so the LoRA / pi0.5
    params stay out of the VF's nnx state-dict and out of the
    optimizer's update path). Caller is responsible for pre-running
    the backbone's ``embed_prefix(...)`` and feeding the resulting
    ``(B, T, D)`` tensor + ``(B, T)`` pad mask into ``__call__``.

    Args mirror :class:`ValueFunctionHead` for swap-ability with the
    PyTorch reference.

    Args:
        backbone: Any object exposing pi0.5-style prefix embeddings.
            Pure reference, not registered as a submodule. ``None`` is
            accepted for tests that drive the VF on raw prefix tensors.
        input_dim, hidden_dim, num_bins, num_layers, dropout, bin_min,
        bin_max: Same as :class:`ValueFunctionHead`.
        rngs: nnx Rngs for head init.
    """

    def __init__(
        self,
        backbone=None,
        *,
        input_dim: int = 2048,
        hidden_dim: int = 512,
        num_bins: int = DEFAULT_NUM_BINS,
        num_layers: int = 3,
        dropout: float = 0.1,
        bin_min: float = DEFAULT_BIN_MIN,
        bin_max: float = DEFAULT_BIN_MAX,
        rngs: nnx.Rngs,
    ):
        # Stash the backbone reference outside nnx's tracked attribute
        # graph — same intent as the PyTorch path's
        # ``object.__setattr__(self, "_backbone_ref", backbone)`` trick
        # at training/rl/pi05_vf.py:115. Using ``object.__setattr__``
        # bypasses nnx's ``Module.__setattr__`` graph-machinery so the
        # backbone is neither part of ``nnx.state(self)`` nor seen by
        # the optimizer's filter pass.
        object.__setattr__(self, "_backbone_ref", backbone)

        self.head = ValueFunctionHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_bins=num_bins,
            num_layers=num_layers,
            dropout=dropout,
            bin_min=bin_min,
            bin_max=bin_max,
            rngs=rngs,
        )

    @property
    def backbone(self):
        """The backbone reference (not a submodule, not in nnx state)."""
        return self._backbone_ref       # type: ignore[attr-defined]

    @property
    def bin_centers(self) -> jnp.ndarray:
        return self.head.bin_centers

    def __call__(
        self,
        prefix_embs: jnp.ndarray,
        prefix_pad_masks: jnp.ndarray,
    ) -> jnp.ndarray:
        """Distributional VF logits ``(B, num_bins)`` from prefix embeddings."""
        pooled = pool_masked_mean(prefix_embs, prefix_pad_masks)
        return self.head(pooled)

    def predict_value(
        self,
        prefix_embs: jnp.ndarray,
        prefix_pad_masks: jnp.ndarray,
    ) -> jnp.ndarray:
        """Continuous expected value ``(B,)`` from prefix embeddings."""
        logits = self(prefix_embs, prefix_pad_masks)
        return expected_value_from_logits(logits, self.bin_centers)

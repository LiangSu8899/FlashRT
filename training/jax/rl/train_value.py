"""Standalone distributional VF training driver — JAX nnx port.

Mirrors :func:`training.rl.train_value.train_value` step-for-step:

1. Build per-frame normalised value targets (already done by the
   caller-supplied :class:`SyntheticDataset`).
2. Train a :class:`StandaloneValueFunction` against the targets via
   the distributional soft-CE loss
   (:func:`training.jax.rl._reward_jax.compute_soft_value_loss`).
3. Run inference to predict ``V(o)``, derive N-step advantages, and
   compute per-task threshold ``ε_ℓ`` for the ~30 % positive ratio
   (:mod:`flash_rt.core.rl.advantage`).
4. Return the trained model + indicator array + loss history.

The synthetic dataset class is reused from
:mod:`training.rl.train_value` (numpy-only — framework-agnostic).
"""

from __future__ import annotations

from dataclasses import dataclass

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from flash_rt.core.rl.advantage import (
    binarize_advantages,
    compute_nstep_advantages,
    compute_per_task_thresholds,
)
from flash_rt.core.rl.reward import compute_dense_rewards_from_targets
from training.rl.train_value import SyntheticDataset

from ._reward_jax import compute_soft_value_loss
from .value_function import StandaloneValueFunction


@dataclass
class JaxTrainResult:
    model: StandaloneValueFunction
    thresholds: dict[int, float]
    indicators: np.ndarray
    loss_history: list[float]


def _set_dropout_deterministic(model: nnx.Module, deterministic: bool) -> None:
    """Toggle dropout layers between train/eval mode (post-init mutation).

    nnx doesn't ship a torch-style ``model.eval()`` / ``model.train()``
    pair — Dropout's ``deterministic`` is set at construction. For
    parity with the PyTorch path's ``model.train()`` / ``model.eval()``
    transitions, we walk the module tree and flip the flag.
    """
    for path, sub in model.iter_modules():
        if isinstance(sub, nnx.Dropout):
            sub.deterministic = deterministic


def train_value(
    dataset: SyntheticDataset,
    *,
    num_steps: int = 1_000,
    batch_size: int = 256,
    state_dim: int = 16,
    hidden_dim: int = 128,
    num_bins: int = 201,
    n_step: int = 50,
    positive_ratio: float = 0.3,
    lr: float = 3e-4,
    seed: int = 0,
) -> JaxTrainResult:
    """End-to-end RECAP VF pipeline on a flat dataset (JAX).

    Args mirror the PyTorch :func:`training.rl.train_value.train_value`
    1-for-1 (same defaults, same rng-seeded batch sampling, same loss).

    Returns:
        :class:`JaxTrainResult` carrying the trained model, indicator
        array, per-task thresholds, and loss history.
    """
    if num_steps <= 0:
        raise ValueError(f"num_steps must be > 0, got {num_steps}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")

    n_frames = dataset.states.shape[0]
    states = jnp.asarray(dataset.states, dtype=jnp.float32)
    targets = jnp.asarray(dataset.target_values, dtype=jnp.float32)

    rngs = nnx.Rngs(seed)
    model = StandaloneValueFunction(
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        num_bins=num_bins,
        rngs=rngs,
    )
    bin_centers = model.bin_centers
    optimizer = nnx.Optimizer(model, optax.adamw(lr))

    @nnx.jit
    def train_step(model, optimizer, state_batch, target_batch):
        def loss_fn(model):
            logits = model(state_batch)
            return compute_soft_value_loss(logits, target_batch, bin_centers)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(grads)
        return loss

    np_rng = np.random.default_rng(seed)
    history: list[float] = []
    _set_dropout_deterministic(model, deterministic=False)
    for _step in range(num_steps):
        idx = np_rng.integers(0, n_frames, size=batch_size)
        loss = train_step(model, optimizer, states[idx], targets[idx])
        history.append(float(loss))

    # Inference over the whole dataset → V(o).
    _set_dropout_deterministic(model, deterministic=True)

    @nnx.jit
    def predict_chunk(model, state_chunk):
        return model.predict_value(state_chunk)

    chunks: list[np.ndarray] = []
    for start in range(0, n_frames, 1024):
        end = min(start + 1024, n_frames)
        v_chunk = predict_chunk(model, states[start:end])
        chunks.append(np.asarray(v_chunk, dtype=np.float32))
    values = np.concatenate(chunks)

    rewards = compute_dense_rewards_from_targets(
        dataset.target_values,
        dataset.episode_indices,
        dataset.frame_indices,
    )
    advantages = compute_nstep_advantages(
        rewards,
        values,
        dataset.episode_indices,
        dataset.frame_indices,
        n_step=n_step,
    )
    thresholds = compute_per_task_thresholds(
        dataset.task_indices, advantages, positive_ratio=positive_ratio
    )
    indicators = binarize_advantages(dataset.task_indices, advantages, thresholds)

    return JaxTrainResult(
        model=model,
        thresholds=thresholds,
        indicators=indicators,
        loss_history=history,
    )

"""Annotate a dataset with ``acp_indicator`` using a trained JAX VF.

Mirrors :mod:`training.rl.value_infer` 1-for-1 on the JAX side. The
torch-touching parts (``predict_value`` callable, GPU eval batching)
are replaced with their JAX equivalents; the numpy advantage /
threshold / binarisation pipeline reuses the framework-agnostic
primitives in :mod:`flash_rt.core.rl.advantage` and
:mod:`flash_rt.core.rl.reward`.

Two surface variants matching PyTorch:

``annotate_with_value_function``
    Generic callable. Accepts any ``predict_value(eval_inputs) ->
    (N,) jnp.ndarray`` function plus the same flat-array layout
    :class:`training.rl.train_value.SyntheticDataset` uses.

``annotate_synthetic_dataset``
    Convenience wrapper for a ``SyntheticDataset`` and a JAX
    :class:`StandaloneValueFunction`. Pure plumbing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np

from flash_rt.core.rl.advantage import (
    binarize_advantages,
    compute_nstep_advantages,
    compute_per_task_thresholds,
)
from flash_rt.core.rl.reward import compute_dense_rewards_from_targets
from training.rl.train_value import SyntheticDataset

from .train_value import _set_dropout_deterministic


@dataclass
class JaxValueAnnotation:
    """Per-frame outputs of :func:`annotate_with_value_function`."""

    values: np.ndarray         # (N,) float32 predicted V(o_t)
    advantages: np.ndarray     # (N,) float32 N-step A(o_t, a_t)
    indicators: np.ndarray     # (N,) int64 binary I_t
    thresholds: dict[int, float]  # ε_ℓ per task index


def _predict_in_chunks(
    predict_fn: Callable[[jnp.ndarray], jnp.ndarray],
    eval_inputs: jnp.ndarray,
    *,
    chunk_size: int,
) -> np.ndarray:
    """Run ``predict_fn`` over a tensor in chunks; return host numpy."""
    chunks: list[np.ndarray] = []
    for start in range(0, eval_inputs.shape[0], chunk_size):
        end = min(start + chunk_size, eval_inputs.shape[0])
        v = predict_fn(eval_inputs[start:end])
        chunks.append(np.asarray(v, dtype=np.float32))
    return np.concatenate(chunks).astype(np.float32)


def annotate_with_value_function(
    *,
    predict_value: Callable[[jnp.ndarray], jnp.ndarray],
    eval_inputs: jnp.ndarray,
    target_values: np.ndarray,
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
    task_indices: np.ndarray,
    n_step: int = 50,
    positive_ratio: float = 0.30,
    chunk_size: int = 1024,
    interventions: np.ndarray | None = None,
    force_intervention_positive: bool = True,
) -> JaxValueAnnotation:
    """Run a trained VF over a dataset and produce the ACP annotations.

    Args mirror :func:`training.rl.value_infer.annotate_with_value_function`
    1-for-1. The only API difference: ``eval_inputs`` is a JAX array
    (``jnp.ndarray``), and ``predict_value`` returns a JAX array.

    Returns:
        :class:`JaxValueAnnotation` with values / advantages /
        indicators / thresholds populated.
    """
    if eval_inputs.shape[0] != target_values.shape[0]:
        raise ValueError(
            f"eval_inputs N={eval_inputs.shape[0]} != "
            f"target_values N={target_values.shape[0]}"
        )

    values = _predict_in_chunks(
        predict_value, eval_inputs, chunk_size=chunk_size
    )

    rewards = compute_dense_rewards_from_targets(
        target_values, episode_indices, frame_indices
    )
    advantages = compute_nstep_advantages(
        rewards,
        values,
        episode_indices,
        frame_indices,
        n_step=n_step,
    )
    thresholds = compute_per_task_thresholds(
        task_indices, advantages, positive_ratio=positive_ratio
    )
    indicators = binarize_advantages(
        task_indices,
        advantages,
        thresholds,
        interventions=interventions,
        force_intervention_positive=force_intervention_positive,
    )

    return JaxValueAnnotation(
        values=values,
        advantages=advantages.astype(np.float32),
        indicators=indicators,
        thresholds=thresholds,
    )


def annotate_synthetic_dataset(
    model: nnx.Module,
    dataset: SyntheticDataset,
    *,
    n_step: int = 50,
    positive_ratio: float = 0.30,
    chunk_size: int = 1024,
) -> JaxValueAnnotation:
    """Convenience: annotate a ``SyntheticDataset`` with a trained JAX VF."""
    if not hasattr(model, "predict_value"):
        raise TypeError(
            f"model must expose 'predict_value' (got {type(model).__name__})"
        )
    _set_dropout_deterministic(model, deterministic=True)
    states_j = jnp.asarray(dataset.states, dtype=jnp.float32)
    return annotate_with_value_function(
        predict_value=model.predict_value,
        eval_inputs=states_j,
        target_values=dataset.target_values,
        episode_indices=dataset.episode_indices,
        frame_indices=dataset.frame_indices,
        task_indices=dataset.task_indices,
        n_step=n_step,
        positive_ratio=positive_ratio,
        chunk_size=chunk_size,
    )

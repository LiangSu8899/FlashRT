"""Annotate a dataset with ``acp_indicator`` using a trained VF.

This is the second leg of a RECAP iteration: after ``train_value`` has
fit the distributional VF, ``annotate_with_value_function`` runs that
VF over every frame of a (typically larger / different) dataset and
produces

* ``values``       — predicted ``V(o_t)``;
* ``advantages``   — N-step advantage with episode-boundary aware
                     bootstrap (Appendix F);
* ``thresholds``   — per-task ``ε_ℓ`` chosen so ~``positive_ratio``
                     of frames are positive (paper default 0.30);
* ``indicators``   — binary ``I_t`` written back to the parquet by the
                     W8 driver.

Two surface variants:

``annotate_with_value_function``
    Generic callable. Accepts any ``predict_value(states_or_embs) →
    (N,)`` function plus the same flat-array layout
    ``train_value.SyntheticDataset`` uses. The W7 policy training
    test exercises this path with a ``StandaloneValueFunction``;
    W8 will reuse it for the VLA path by passing
    ``Pi05ValueFunction`` and a callable that streams pooled
    prefix embeddings batch-by-batch.

``annotate_synthetic_dataset``
    Convenience wrapper for a ``train_value.SyntheticDataset`` and a
    ``StandaloneValueFunction``. Pure plumbing on top of the generic
    helper.

Source: paper-aligned port of openpi-compiler/RL/recap/value_infer.py
(2026-04-25). Real LeRobot/parquet I/O is W8.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from flash_rt.core.rl.advantage import (
    binarize_advantages,
    compute_nstep_advantages,
    compute_per_task_thresholds,
)
from flash_rt.core.rl.reward import compute_dense_rewards_from_targets
from training.rl.train_value import SyntheticDataset


@dataclass
class ValueAnnotation:
    """Per-frame outputs of ``annotate_with_value_function``."""

    values: np.ndarray         # (N,) float32  predicted V(o_t)
    advantages: np.ndarray     # (N,) float32  N-step A(o_t, a_t)
    indicators: np.ndarray     # (N,) int64    binary I_t
    thresholds: dict[int, float]  # ε_ℓ per task index


def _predict_values_in_chunks(
    predict_fn: Callable[[torch.Tensor], torch.Tensor],
    inputs_t: torch.Tensor,
    *,
    chunk_size: int,
) -> np.ndarray:
    """Run ``predict_fn`` over a tensor in chunks, return numpy on host.

    Used to keep the eval pass memory-bounded — pi0.5 datasets can hit
    millions of frames and a single big eval would either OOM or
    require streaming inside the VF model.
    """
    chunks: list[np.ndarray] = []
    for start in range(0, inputs_t.shape[0], chunk_size):
        end = min(start + chunk_size, inputs_t.shape[0])
        chunks.append(predict_fn(inputs_t[start:end]).cpu().numpy())
    return np.concatenate(chunks).astype(np.float32)


def annotate_with_value_function(
    *,
    predict_value: Callable[[torch.Tensor], torch.Tensor],
    eval_inputs: torch.Tensor,
    target_values: np.ndarray,
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
    task_indices: np.ndarray,
    n_step: int = 50,
    positive_ratio: float = 0.30,
    chunk_size: int = 1024,
    interventions: np.ndarray | None = None,
    force_intervention_positive: bool = True,
) -> ValueAnnotation:
    """Run a trained VF over a dataset and produce the ACP annotations.

    Args:
        predict_value: Callable ``(B-shaped tensor) → (B,) tensor``.
            The wrapper around a trained
            :class:`StandaloneValueFunction` or
            :class:`Pi05ValueFunction` should expose this — both
            already do via ``predict_value``.
        eval_inputs: Inputs to ``predict_value``. For the synthetic
            path, a ``(N, state_dim)`` state tensor; for the VLA
            path, a ``(N, T, D)`` pooled-or-unpooled embedding
            tensor — caller pre-computes these or wraps them in a
            generator that yields one chunk at a time.
        target_values: Pre-computed ``g_norm`` per frame, shape ``(N,)``.
        episode_indices, frame_indices, task_indices: Standard flat
            metadata arrays, all shape ``(N,)``.
        n_step, positive_ratio: As in the paper / Appendix F.
        chunk_size: Eval batch size (memory bound).
        interventions: Optional ``(N,)`` flag tensor; ``> 0.5`` →
            intervention-step. When ``force_intervention_positive``
            is True (default) those frames override the threshold and
            get ``I_t = 1``.

    Returns:
        :class:`ValueAnnotation` with values / advantages / indicators
        / thresholds populated.
    """
    if eval_inputs.shape[0] != target_values.shape[0]:
        raise ValueError(
            f"eval_inputs N={eval_inputs.shape[0]} != "
            f"target_values N={target_values.shape[0]}"
        )

    with torch.no_grad():
        values = _predict_values_in_chunks(
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

    return ValueAnnotation(
        values=values,
        advantages=advantages.astype(np.float32),
        indicators=indicators,
        thresholds=thresholds,
    )


def annotate_synthetic_dataset(
    model: nn.Module,
    dataset: SyntheticDataset,
    *,
    n_step: int = 50,
    positive_ratio: float = 0.30,
    chunk_size: int = 1024,
    device: str | torch.device = "cuda",
) -> ValueAnnotation:
    """Convenience: annotate a ``SyntheticDataset`` with a trained VF."""
    if not hasattr(model, "predict_value"):
        raise TypeError(
            f"model must expose 'predict_value' (got {type(model).__name__})"
        )

    device = torch.device(device)
    model.eval()
    states_t = torch.from_numpy(dataset.states).to(device)

    return annotate_with_value_function(
        predict_value=model.predict_value,
        eval_inputs=states_t,
        target_values=dataset.target_values,
        episode_indices=dataset.episode_indices,
        frame_indices=dataset.frame_indices,
        task_indices=dataset.task_indices,
        n_step=n_step,
        positive_ratio=positive_ratio,
        chunk_size=chunk_size,
    )

r"""Advantage estimation primitives for RECAP.

Implements Appendix F of the π\*0.6 paper
([arXiv:2511.14759](https://arxiv.org/abs/2511.14759)):

    A(o_t, a_t) = Σ_{k=0}^{N-1} r_{t+k} + V(o_{t+N}) - V(o_t)

with episode-boundary aware bootstrap, per-task threshold
``ε_ℓ`` chosen so ~30 % of frames are positive (paper recommendation),
and optional intervention-step forcing.

Source: ported from ``openpi-compiler/RL/recap/advantage.py``
2026-04-25, with the legacy v1 API trimmed.
"""

from __future__ import annotations

import numpy as np


def compute_nstep_advantages(
    rewards: np.ndarray,
    values: np.ndarray,
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
    *,
    n_step: int = 50,
) -> np.ndarray:
    """N-step advantage with episode-boundary handling.

    ``A(o_t, a_t) = Σ r_{t+k} + V(o_{t+N}) - V(o_t)``, summed only
    while ``frame_indices`` stays contiguous within the same episode.
    Bootstrap term ``V(o_{t+N})`` is included only when the full
    N-step lookahead lies inside the episode; otherwise it is zero
    (the dense rewards already encode the terminal correction).

    Args:
        rewards: Dense per-frame rewards (typically from
            ``compute_dense_rewards_from_targets``), shape (N,).
        values: VF predictions, shape (N,).
        episode_indices: Episode index per frame, shape (N,).
        frame_indices: Frame index within episode, shape (N,).
        n_step: Lookahead window. RECAP uses 50 in post-training.

    Returns:
        Advantage values, shape (N,).
    """
    if n_step <= 0:
        raise ValueError("'n_step' must be > 0.")

    n = rewards.shape[0]
    advantages = np.zeros(n, dtype=np.float32)

    for i in range(n):
        ep_i = episode_indices[i]
        fi = frame_indices[i]

        discounted_sum = 0.0
        j = i
        steps = 0
        while steps < n_step and j < n:
            same_episode = episode_indices[j] == ep_i
            contiguous = frame_indices[j] == fi + steps
            if not same_episode or not contiguous:
                break
            discounted_sum += float(rewards[j])
            steps += 1
            j += 1

        # Bootstrap V(o_{t+N}) only when we cleared N steps and the
        # next frame is the contiguous in-episode (t+N) frame.
        if (
            steps == n_step
            and j < n
            and episode_indices[j] == ep_i
            and frame_indices[j] == fi + n_step
        ):
            bootstrap = float(values[j])
        else:
            bootstrap = 0.0

        advantages[i] = float(discounted_sum + bootstrap - values[i])

    return advantages


def compute_per_task_thresholds(
    task_indices: np.ndarray,
    advantages: np.ndarray,
    *,
    positive_ratio: float = 0.3,
) -> dict[int, float]:
    """Per-task advantage threshold ``ε_ℓ``.

    The threshold is set so ``positive_ratio`` of frames per task
    have ``A >= ε_ℓ``. Paper:

        "we set ε_ℓ so that approximately 30% of the data is labeled
         as improvement for pre-training data."

    Args:
        task_indices: Task index per frame, shape (N,).
        advantages: Advantage values, shape (N,).
        positive_ratio: Target fraction of positive indicators.

    Returns:
        ``{task_index: threshold}``.
    """
    if not 0.0 <= positive_ratio <= 1.0:
        raise ValueError("'positive_ratio' must be within [0, 1].")

    thresholds: dict[int, float] = {}
    quantile = 1.0 - positive_ratio  # top 30 % → 70th percentile

    for task_idx in np.unique(task_indices):
        task_adv = advantages[task_indices == task_idx]
        if task_adv.size == 0:
            thresholds[int(task_idx)] = float("inf")
        else:
            thresholds[int(task_idx)] = float(np.quantile(task_adv, quantile))

    return thresholds


def binarize_advantages(
    task_indices: np.ndarray,
    advantages: np.ndarray,
    thresholds: dict[int, float],
    *,
    interventions: np.ndarray | None = None,
    force_intervention_positive: bool = True,
) -> np.ndarray:
    """Binarise advantages into improvement indicators ``I_t``.

    ``I_t = 1`` iff ``A(o_t, a_t) >= ε_ℓ`` (per-task), else 0.
    Optionally forces intervention frames to ``I_t = 1`` (paper
    trick: the intervener's correction is by definition an
    improvement signal).

    Args:
        task_indices: Task index per frame, shape (N,).
        advantages: Advantage values, shape (N,).
        thresholds: Output of ``compute_per_task_thresholds``.
        interventions: Optional intervention flags, shape (N,).
            ``> 0.5`` is interpreted as "intervention happened".
        force_intervention_positive: If True (default), intervention
            frames override threshold and become positive.

    Returns:
        Binary indicators, shape (N,), dtype int64.
    """
    indicators = np.zeros_like(advantages, dtype=np.int64)

    for i in range(advantages.shape[0]):
        task_idx = int(task_indices[i])
        threshold = thresholds[task_idx]
        indicators[i] = 1 if float(advantages[i]) >= threshold else 0

    if force_intervention_positive and interventions is not None:
        intervention_mask = interventions.astype(np.float32) > 0.5
        indicators[intervention_mask] = 1

    return indicators

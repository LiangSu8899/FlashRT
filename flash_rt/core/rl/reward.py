r"""Reward and value-target primitives for RECAP.

Implements Equation (5) and Section IV-A of the π\*0.6 paper
([arXiv:2511.14759](https://arxiv.org/abs/2511.14759)):

    r_t = 0       if t = T and success
    r_t = -C_fail if t = T and failure
    r_t = -1      otherwise

with per-task normalisation
``g_norm = g_t / (task_max_length + c_fail)`` clipped to
``[bin_min, bin_max]`` and dense rewards derived from value-target
differencing — matching Evo-RL's ``compute_normalized_value_targets``
/ ``compute_dense_rewards_from_targets``.

The distributional value-function head trains against soft bin
targets (bilinear interpolation, Equation 1) rather than hard
one-hot, again following Evo-RL.

Source: ported from ``openpi-compiler/RL/recap/reward.py`` 2026-04-25.
The port drops the legacy v1 API (``compute_episode_rewards``,
``compute_returns``, ``normalize_returns``) — the v2 API is the
single source of truth.
"""

from __future__ import annotations

import numpy as np
import torch

# Defaults match Evo-RL / openpi-compiler.
DEFAULT_NUM_BINS = 201
DEFAULT_BIN_MIN = -1.0
DEFAULT_BIN_MAX = 0.0


def compute_normalized_value_targets(
    episode_length: int,
    frame_index: int,
    success: bool,
    task_max_length: int,
    *,
    c_fail_coef: float = 1.0,
    clip_min: float = DEFAULT_BIN_MIN,
    clip_max: float = DEFAULT_BIN_MAX,
) -> float:
    """Normalised value target for a single frame (Eq. 5 + per-task norm).

    Args:
        episode_length: Total length of the episode.
        frame_index: Current frame index within the episode.
        success: Whether the episode was successful.
        task_max_length: Maximum episode length for this task.
        c_fail_coef: Failure penalty coefficient
            (``c_fail = task_max_length * c_fail_coef``).
        clip_min: Minimum value (typically ``bin_min``).
        clip_max: Maximum value (typically ``bin_max``).

    Returns:
        Normalised target ``g_norm`` in ``[clip_min, clip_max]``.
    """
    c_fail = float(task_max_length) * c_fail_coef
    remaining_steps = episode_length - frame_index - 1
    g = -float(remaining_steps)
    if not success:
        g -= c_fail
    denom = float(task_max_length) + c_fail
    g_norm = g / denom
    return float(np.clip(g_norm, clip_min, clip_max))


def compute_episode_value_targets(
    episode_length: int,
    success: bool,
    task_max_length: int,
    *,
    c_fail_coef: float = 1.0,
    clip_min: float = DEFAULT_BIN_MIN,
    clip_max: float = DEFAULT_BIN_MAX,
) -> np.ndarray:
    """Vectorised ``compute_normalized_value_targets`` over one episode."""
    targets = np.zeros(episode_length, dtype=np.float32)
    for t in range(episode_length):
        targets[t] = compute_normalized_value_targets(
            episode_length=episode_length,
            frame_index=t,
            success=success,
            task_max_length=task_max_length,
            c_fail_coef=c_fail_coef,
            clip_min=clip_min,
            clip_max=clip_max,
        )
    return targets


def compute_dense_rewards_from_targets(
    targets: np.ndarray,
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
) -> np.ndarray:
    """Dense per-frame rewards via consecutive-target differencing.

    For frames inside an episode::

        reward[i] = target[i] - target[i+1]

    For the final frame of an episode (or any boundary)::

        reward[i] = target[i]

    The episode-boundary check requires both the same ``episode_indices``
    AND consecutive ``frame_indices`` to handle datasets where frames
    are reordered. Matches Evo-RL.
    """
    n = targets.shape[0]
    rewards = np.zeros(n, dtype=np.float32)
    for i in range(n):
        is_next_in_episode = (
            i + 1 < n
            and episode_indices[i + 1] == episode_indices[i]
            and frame_indices[i + 1] == frame_indices[i] + 1
        )
        if is_next_in_episode:
            rewards[i] = float(targets[i] - targets[i + 1])
        else:
            rewards[i] = float(targets[i])
    return rewards


# ── Soft target binning (Section IV-A) ─────────────────────────────


def build_bin_centers(
    num_bins: int = DEFAULT_NUM_BINS,
    bin_min: float = DEFAULT_BIN_MIN,
    bin_max: float = DEFAULT_BIN_MAX,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Evenly spaced bin centres in ``[bin_min, bin_max]``."""
    return torch.linspace(bin_min, bin_max, num_bins, dtype=torch.float32, device=device)


def project_values_to_bins(
    values: torch.Tensor,
    bin_centers: torch.Tensor,
) -> torch.Tensor:
    """Bilinear projection of scalar values onto a bin distribution.

    Each value is distributed across its two nearest bin centres
    weighted by distance — the soft-target form used by the
    distributional VF in Equation 1. Smoother gradients than a
    hard one-hot.

    Args:
        values: Rank-1 tensor of shape (B,).
        bin_centers: Rank-1 tensor of shape (num_bins,).

    Returns:
        Soft target distributions, shape (B, num_bins).
    """
    if values.ndim != 1:
        raise ValueError(f"'values' must be rank-1, got shape={tuple(values.shape)}.")
    if bin_centers.ndim != 1:
        raise ValueError(f"'bin_centers' must be rank-1, got shape={tuple(bin_centers.shape)}.")
    if bin_centers.shape[0] < 2:
        raise ValueError("At least 2 bins are required.")

    values = values.clamp(min=bin_centers[0], max=bin_centers[-1])
    step = bin_centers[1] - bin_centers[0]
    scaled = (values - bin_centers[0]) / step
    low = torch.floor(scaled).long()
    high = torch.clamp(low + 1, max=bin_centers.shape[0] - 1)
    high_weight = (scaled - low.float()).clamp(0.0, 1.0)
    low_weight = 1.0 - high_weight

    target = torch.zeros(
        values.shape[0],
        bin_centers.shape[0],
        device=values.device,
        dtype=torch.float32,
    )
    target.scatter_add_(1, low.unsqueeze(1), low_weight.unsqueeze(1))
    target.scatter_add_(1, high.unsqueeze(1), high_weight.unsqueeze(1))
    return target


def expected_value_from_logits(
    logits: torch.Tensor,
    bin_centers: torch.Tensor,
) -> torch.Tensor:
    """Continuous expected value from distributional logits.

    ``V = Σ softmax(logits) * bin_centers`` — the inference-time
    consumer of the distributional VF.

    Args:
        logits: ``(..., num_bins)``.
        bin_centers: ``(num_bins,)``.

    Returns:
        Expected values shape ``(...,)``.
    """
    probs = torch.softmax(logits, dim=-1)
    return (probs * bin_centers).sum(dim=-1)


def compute_soft_value_loss(
    predicted_logits: torch.Tensor,
    target_values: torch.Tensor,
    bin_centers: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Distributional VF loss with soft targets (π\*0.6 Eq. 1).

    ``L = - Σ soft_target * log_softmax(logits)`` averaged over the batch.

    Args:
        predicted_logits: ``(B, num_bins)``.
        target_values: ``(B,)`` continuous targets in ``[bin_min, bin_max]``.
        bin_centers: ``(num_bins,)``.
        sample_weights: Optional per-sample weights, shape ``(B,)``.

    Returns:
        Scalar loss.
    """
    soft_target = project_values_to_bins(target_values, bin_centers)
    log_probs = torch.log_softmax(predicted_logits, dim=-1)
    per_sample_loss = -(soft_target * log_probs).sum(dim=-1)

    if sample_weights is not None:
        per_sample_loss = per_sample_loss * sample_weights

    return per_sample_loss.mean()

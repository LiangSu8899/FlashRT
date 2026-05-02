"""Standalone distributional VF training driver.

End-to-end RECAP value-function pipeline on synthetic / mock data:

1. Build per-frame normalised value targets from episode metadata
   (``compute_episode_value_targets``);
2. Derive dense per-step rewards via consecutive-target differencing
   (``compute_dense_rewards_from_targets``) — the same dataset table
   ``train_policy`` will read in W7;
3. Train a ``StandaloneValueFunction`` against the targets with the
   distributional soft-CE loss (Eq. 1);
4. Run inference to predict ``V(o)`` on every frame, derive N-step
   advantages, and compute per-task threshold ``ε_ℓ`` for the
   ~30 % positive ratio (Appendix F);
5. Optionally write the resulting ``acp_indicator`` column back as a
   numpy array (no parquet writing here — that wiring is W8 once we
   have the LeRobot dataset adapter).

Real LeRobot/parquet loading will be added in W7/W8 alongside the
policy-training driver. For now ``run_synthetic_value_training``
provides a self-contained smoke test of the W6 contract.

Source: paper-aligned port of openpi-compiler/RL/recap/train_value.py
(2026-04-25). Standalone-only; the VLA-integrated head training lands
in :mod:`training.rl.train_value_pi05` once the prefix-embedding hook
is wired up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from flash_rt.core.rl.advantage import (
    binarize_advantages,
    compute_nstep_advantages,
    compute_per_task_thresholds,
)
from flash_rt.core.rl.reward import (
    compute_dense_rewards_from_targets,
    compute_episode_value_targets,
    compute_soft_value_loss,
)
from flash_rt.core.rl.value_function import StandaloneValueFunction


@dataclass
class SyntheticDataset:
    """Flat per-frame arrays mirroring the real LeRobot table layout."""

    states: np.ndarray         # (N, state_dim) float32
    target_values: np.ndarray  # (N,) float32
    episode_indices: np.ndarray  # (N,) int64
    frame_indices: np.ndarray    # (N,) int64
    task_indices: np.ndarray     # (N,) int64
    task_max_lengths: dict[int, int]


@dataclass
class TrainResult:
    """Output of ``train_value``."""

    model: StandaloneValueFunction
    thresholds: dict[int, float]
    indicators: np.ndarray
    loss_history: list[float]


# ────────────────────────────────────────────────────────────────────
# Synthetic data
# ────────────────────────────────────────────────────────────────────


def make_synthetic_dataset(
    n_episodes: int = 200,
    n_tasks: int = 4,
    state_dim: int = 16,
    min_length: int = 30,
    max_length: int = 100,
    success_rate: float = 0.5,
    seed: int = 0,
    *,
    state_signal_strength: float = 1.0,
) -> SyntheticDataset:
    """Mock LIBERO-style episodes for VF training smoke tests.

    State convention (so VF can actually learn something):
        state[0] = (frame_index / episode_length) * 2 - 1  ∈ [-1, 1)
                   — progress signal
        state[1] = +1 for success, -1 for failure
                   — outcome signal scaled by ``state_signal_strength``
        state[2:] = standard normal noise
    Setting ``state_signal_strength=0`` reverts to fully random state
    (useful for adversarial sanity checks).
    """
    rng = np.random.default_rng(seed)

    states_list: list[np.ndarray] = []
    targets_list: list[np.ndarray] = []
    ep_idx_list: list[np.ndarray] = []
    frame_idx_list: list[np.ndarray] = []
    task_idx_list: list[np.ndarray] = []
    task_max_lengths: dict[int, int] = {t: max_length for t in range(n_tasks)}

    for ep in range(n_episodes):
        length = int(rng.integers(min_length, max_length + 1))
        success = bool(rng.random() < success_rate)
        task = int(rng.integers(n_tasks))

        state = rng.standard_normal((length, state_dim)).astype(np.float32)
        if state_signal_strength > 0 and state_dim >= 2:
            progress = (np.arange(length, dtype=np.float32) / max(length, 1)) * 2.0 - 1.0
            outcome = (1.0 if success else -1.0) * state_signal_strength
            state[:, 0] = progress
            state[:, 1] = outcome
        states_list.append(state)
        targets_list.append(
            compute_episode_value_targets(
                episode_length=length,
                success=success,
                task_max_length=task_max_lengths[task],
            )
        )
        ep_idx_list.append(np.full(length, ep, dtype=np.int64))
        frame_idx_list.append(np.arange(length, dtype=np.int64))
        task_idx_list.append(np.full(length, task, dtype=np.int64))

    return SyntheticDataset(
        states=np.concatenate(states_list),
        target_values=np.concatenate(targets_list),
        episode_indices=np.concatenate(ep_idx_list),
        frame_indices=np.concatenate(frame_idx_list),
        task_indices=np.concatenate(task_idx_list),
        task_max_lengths=task_max_lengths,
    )


# ────────────────────────────────────────────────────────────────────
# Training
# ────────────────────────────────────────────────────────────────────


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
    device: str | torch.device = "cuda",
    seed: int = 0,
) -> TrainResult:
    """End-to-end RECAP VF pipeline on a flat dataset.

    Trains the distributional VF against pre-computed targets, then
    runs inference + N-step advantage + per-task threshold to derive
    the binary improvement indicator that downstream policy training
    will read.
    """
    if num_steps <= 0:
        raise ValueError(f"num_steps must be > 0, got {num_steps}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")

    device = torch.device(device)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    n_frames = dataset.states.shape[0]
    states_t = torch.from_numpy(dataset.states).to(device)
    targets_t = torch.from_numpy(dataset.target_values).to(device)

    model = StandaloneValueFunction(
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        num_bins=num_bins,
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    bin_centers = model.bin_centers.to(device)

    history: list[float] = []
    model.train()
    for step in range(num_steps):
        idx = rng.integers(0, n_frames, size=batch_size)
        idx_t = torch.from_numpy(idx).to(device)
        state_batch = states_t[idx_t]
        target_batch = targets_t[idx_t]

        logits = model(state_batch)
        loss = compute_soft_value_loss(logits, target_batch, bin_centers)
        optim.zero_grad()
        loss.backward()
        optim.step()
        history.append(float(loss.item()))

    # Inference over the whole dataset → V(o), then dense rewards →
    # N-step advantage → per-task threshold → binary indicator.
    model.eval()
    with torch.no_grad():
        chunks = []
        for start in range(0, n_frames, 1024):
            end = min(start + 1024, n_frames)
            chunks.append(model.predict_value(states_t[start:end]).cpu().numpy())
        values = np.concatenate(chunks).astype(np.float32)

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

    return TrainResult(
        model=model,
        thresholds=thresholds,
        indicators=indicators,
        loss_history=history,
    )


def run_synthetic_value_training(
    *,
    num_steps: int = 1_000,
    n_episodes: int = 200,
    n_tasks: int = 4,
    seed: int = 0,
    device: str | torch.device = "cuda",
    lr: float = 1e-3,
) -> TrainResult:
    """Convenience wrapper that builds the dataset + runs ``train_value``."""
    state_dim = 16
    dataset = make_synthetic_dataset(
        n_episodes=n_episodes, n_tasks=n_tasks, state_dim=state_dim, seed=seed
    )
    return train_value(
        dataset,
        num_steps=num_steps,
        state_dim=state_dim,
        device=device,
        seed=seed,
        lr=lr,
    )

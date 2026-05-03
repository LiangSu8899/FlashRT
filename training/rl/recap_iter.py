"""End-to-end RECAP iteration on a RecapMetadataDataset.

Dataset-agnostic by design — any object satisfying
:class:`training.rl.dataset_protocol.RecapMetadataDataset` plugs
in. LIBERO ships as the first concrete implementation
(:class:`training.rl.lerobot_libero.LeRobotLiberoDataset`); the
back-compat alias :func:`run_recap_iter_on_libero` is kept so any
existing caller keeps working unchanged.

One iteration = the three RECAP legs run once on the same dataset:

1. **Value-target build**     — derive Eq. 5 normalised targets from
   the episode metadata (length, success, task_max_length).
2. **VF training**             — fit a ``StandaloneValueFunction`` over
   the per-frame proprioceptive state. Cheap, no images. Acts as the
   placeholder VF until the W9 ``Pi05ValueFunction`` over real
   prefix embeddings replaces it.
3. **ACP indicator annotation** — run ``annotate_with_value_function``
   to derive ``acp_indicator`` per frame. Cached in
   :class:`RecapIterResult` (no parquet write here — the LeRobot v3
   parquet layout is read-only in our flow; W9 will write the
   annotation column to a side table if persistence is needed).

The driver is the contract surface for the W9 multi-iter loop and
the LIBERO eval — it returns the artefacts that ``train_policy``
consumes (states, actions, image-bytes loader, indicators, prompt
text per frame).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from flash_rt.core.rl.reward import compute_episode_value_targets
from flash_rt.core.rl.value_function import StandaloneValueFunction
from training.rl.lerobot_libero import LeRobotLiberoDataset
from training.rl.train_value import SyntheticDataset, train_value
from training.rl.value_infer import (
    ValueAnnotation,
    annotate_synthetic_dataset,
)


@dataclass
class RecapIterResult:
    """Outputs of one RECAP iteration."""

    annotation: ValueAnnotation
    vf_loss_history: list[float] = field(default_factory=list)
    positive_ratio_per_task: dict[int, float] = field(default_factory=dict)


def _build_dataset_view(
    dataset,
    *,
    c_fail_coef: float = 1.0,
) -> SyntheticDataset:
    """Cast any :class:`RecapMetadataDataset` to the flat-array form
    ``train_value`` / ``annotate_synthetic_dataset`` consume.

    Computes per-frame ``target_values`` from each episode's
    ``(length, success, task_max_length)`` triple via Eq. 5.
    """
    states, _ = dataset.ensure_state_action()
    n = dataset.num_frames

    target_values = np.zeros(n, dtype=np.float32)
    cursor = 0
    task_max_lengths = dataset.task_max_lengths
    for ep in dataset.episodes:
        targets = compute_episode_value_targets(
            episode_length=ep.length,
            success=ep.success,
            task_max_length=task_max_lengths[ep.task_index],
            c_fail_coef=c_fail_coef,
        )
        target_values[cursor : cursor + ep.length] = targets
        cursor += ep.length
    if cursor != n:
        raise RuntimeError(
            f"target build cursor mismatch: {cursor} vs {n} frames"
        )

    return SyntheticDataset(
        states=states,
        target_values=target_values,
        episode_indices=dataset.episode_indices,
        frame_indices=dataset.frame_indices,
        task_indices=dataset.task_indices,
        task_max_lengths=task_max_lengths,
    )


def run_recap_iter(
    dataset,
    *,
    num_steps: int = 1_000,
    batch_size: int = 256,
    hidden_dim: int = 128,
    num_bins: int = 201,
    n_step: int = 50,
    positive_ratio: float = 0.30,
    c_fail_coef: float = 1.0,
    lr: float = 1e-3,
    seed: int = 0,
    device: str | torch.device = "cuda",
) -> RecapIterResult:
    """Run one RECAP iteration on any :class:`RecapMetadataDataset`.

    The flow is **train-VF → annotate** end-to-end on the dataset's
    state + metadata arrays without touching the VLA backbone — that
    keeps this iteration cheap (~seconds at the default
    ``num_steps=1000``).

    Args:
        dataset: Any object satisfying
            :class:`training.rl.dataset_protocol.RecapMetadataDataset`.
            State + metadata only — image bytes are NOT read here.
        num_steps, batch_size, hidden_dim, num_bins, n_step,
        positive_ratio, c_fail_coef, lr, seed, device: as in
        :func:`train_value` / :func:`annotate_synthetic_dataset`.

    Returns:
        :class:`RecapIterResult` with the value annotation and the
        per-task positive-ratio summary.
    """
    view = _build_dataset_view(dataset, c_fail_coef=c_fail_coef)

    train_result = train_value(
        view,
        num_steps=num_steps,
        batch_size=batch_size,
        state_dim=dataset.state_dim,
        hidden_dim=hidden_dim,
        num_bins=num_bins,
        n_step=n_step,
        positive_ratio=positive_ratio,
        lr=lr,
        device=device,
        seed=seed,
    )

    annotation = annotate_synthetic_dataset(
        train_result.model,
        view,
        n_step=n_step,
        positive_ratio=positive_ratio,
        device=device,
    )

    pos_ratio_per_task: dict[int, float] = {}
    for task_idx in np.unique(view.task_indices):
        mask = view.task_indices == task_idx
        pos_ratio_per_task[int(task_idx)] = float(annotation.indicators[mask].mean())

    return RecapIterResult(
        annotation=annotation,
        vf_loss_history=train_result.loss_history,
        positive_ratio_per_task=pos_ratio_per_task,
    )


# Back-compat alias — old name kept so any existing caller (tests,
# training/rl/train_libero_recap.py's _resolve_acp_indicators
# fallback path before the refactor, etc.) keeps working. New code
# should use ``run_recap_iter``.
run_recap_iter_on_libero = run_recap_iter


def sample_acp_batch(
    dataset,
    annotation: ValueAnnotation,
    *,
    bsize: int,
    rng: np.random.Generator,
) -> tuple[list, np.ndarray, list[str], np.ndarray]:
    """Sample one (frames, actions, tasks, indicators) batch.

    Works on any :class:`RecapPolicyDataset`. The output drops
    straight into the policy loop::

        frames, actions, tasks, indicators = sample_acp_batch(...)
        loss = policy_train_step(
            ...,
            indicators=torch.from_numpy(indicators).long().to(device),
            tasks=tasks,
            hook=ACPPromptHook(...),
            tokenize_fn=PaligemmaTokenizer(...),
        )
    """
    n = dataset.num_frames
    if bsize > n:
        raise ValueError(f"bsize {bsize} > dataset frames {n}")
    idx = rng.integers(0, n, size=bsize)

    frames = [dataset.get_frame(int(i)) for i in idx]
    actions = np.stack([f.action for f in frames], axis=0)
    tasks = [f.task_name for f in frames]
    indicators = annotation.indicators[idx].astype(np.int64)
    return frames, actions, tasks, indicators

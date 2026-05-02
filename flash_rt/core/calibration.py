"""Helpers for multi-sample calibration.

Two responsibilities:

1. **amax reduction** — combine a list of per-sample per-tensor amax
   arrays into a single percentile-reduced amax vector.
2. **stratified sampling** — pick N representative calibration frames
   from a LIBERO-rollout-shaped dataset, matching the openpi-jax-mlir
   toolchain pattern (episode × frame-position stratification).
3. **scale-ceiling warning** — scan produced scales against a sanity
   threshold and emit a logger.warning for any layer whose amax
   exceeds it (FP8 dynamic-range exhaustion risk).

Kept deliberately framework-agnostic: no torch, no CUDA — so the logic
is trivially unit-testable on CPU.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Multiplicative outlier threshold for the scale-ceiling diagnostic.
# Different FlashVLA backends store the "FP8 act scale" buffer with
# slightly different semantics (some hold amax, some hold amax/448,
# depending on the kernel). Rather than pick an absolute threshold
# that would mis-fire on one backend or another, we flag scales that
# are more than ``FP8_SCALE_RATIO_WARN`` times the **median** across
# all scales in this calibration. 20× is far outside any normal
# per-layer variance yet still catches a single extreme outlier.
FP8_SCALE_RATIO_WARN = 20.0


def accumulate_amax(per_sample_amax: List[np.ndarray],
                    percentile: float = 99.9) -> np.ndarray:
    """Reduce a list of per-sample amax arrays to a single amax vector.

    Args:
        per_sample_amax: non-empty list of arrays, all with the same shape
            ``[num_points]``. Each entry is the per-tensor maximum observed
            on one calibration sample.
        percentile: percentile to compute along the sample axis.
            ``100.0`` == traditional max. ``99.9`` (default) excludes the
            top 0.1 % of samples, which guards against a single outlier
            frame inflating every scale.

    Returns:
        Array of shape ``[num_points]`` with the reduced amax.
    """
    if not per_sample_amax:
        raise ValueError("per_sample_amax must contain at least one entry")
    if not 0.0 <= percentile <= 100.0:
        raise ValueError(f"percentile must be in [0, 100], got {percentile}")
    stacked = np.stack([np.asarray(a, dtype=np.float64) for a in per_sample_amax],
                       axis=0)
    if stacked.ndim != 2:
        raise ValueError(
            f"each per-sample amax must be 1-D, got shapes "
            f"{[a.shape for a in per_sample_amax]}")
    return np.percentile(stacked, percentile, axis=0).astype(np.float32)


def summarize_amax_dispersion(per_sample_amax: List[np.ndarray],
                              final_amax: np.ndarray) -> dict:
    """Compute dispersion stats for logging / diagnostics.

    Useful output: for how many points does the percentile-reduced value
    differ materially from the true max? A large gap means outlier
    suppression is actually kicking in.
    """
    stacked = np.stack([np.asarray(a, dtype=np.float64) for a in per_sample_amax],
                       axis=0)
    per_point_max = stacked.max(axis=0)
    per_point_median = np.median(stacked, axis=0)

    # How much has the percentile cut back from the true max?
    cutback = (per_point_max - final_amax) / np.maximum(per_point_max, 1e-12)
    return {
        "num_samples": int(stacked.shape[0]),
        "num_points": int(stacked.shape[1]),
        "amax_max_over_points": float(per_point_max.max()),
        "amax_median_over_points": float(np.median(per_point_median)),
        "cutback_from_max_p50": float(np.percentile(cutback, 50)),
        "cutback_from_max_p99": float(np.percentile(cutback, 99)),
        "num_points_cut_gt_10pct": int((cutback > 0.10).sum()),
    }


def format_summary(summary: dict) -> str:
    """One-line human-readable rendering of :func:`summarize_amax_dispersion`."""
    return (
        f"[calibration] N={summary['num_samples']} samples × "
        f"{summary['num_points']} quant points; "
        f"median amax={summary['amax_median_over_points']:.3g}, "
        f"max amax={summary['amax_max_over_points']:.3g}; "
        f"outlier cutback p50={100 * summary['cutback_from_max_p50']:.2f}%, "
        f"p99={100 * summary['cutback_from_max_p99']:.2f}%; "
        f"{summary['num_points_cut_gt_10pct']} points clipped >10%"
    )


# ---------------------------------------------------------------------------
# Scale-ceiling diagnostic warning
# ---------------------------------------------------------------------------

def check_scale_ceiling(
    scales: Union[dict, np.ndarray, Iterable[float]],
    ratio: float = FP8_SCALE_RATIO_WARN,
    label: str = "calibration",
) -> List[Tuple[str, float]]:
    """Warn on extreme per-tensor scale outliers within this calibration.

    Backends differ in whether the ``fp8_act_scales`` buffer stores
    ``amax`` or ``amax/448``, so an **absolute** threshold would either
    over- or under-fire. Instead we flag entries whose scale exceeds
    ``ratio`` × the median of the same calibration. At ``ratio=20`` this
    passes clean on healthy Pi0 / Pi0.5 RTX calibrations (where normal
    inter-layer variance is 3-5×) and fires when a single layer's amax
    is pulled 20× higher than peers — the signature of a true outlier
    sample in the calibration set.

    This is a **diagnostic warning only** — FlashVLA captures a single
    CUDA Graph per calibration, so there is no runtime fallback to
    FP16 for an offending layer. If this fires, re-run calibration
    with a lower ``percentile`` or a different sample set.

    Args:
        scales: dict ``{tensor_name: scale_value}`` or a 1-D array-like.
        ratio: warn if ``scale > ratio * median(scales)``.
        label: short string embedded in the warning for log grep-ability.

    Returns:
        List of (name, scale) tuples that exceeded the ratio threshold.
    """
    if isinstance(scales, dict):
        items = [(str(k), float(np.asarray(v).reshape(-1)[0]))
                 for k, v in scales.items()]
    else:
        arr = np.asarray(
            list(scales) if not isinstance(scales, np.ndarray) else scales,
            dtype=np.float64).reshape(-1)
        items = [(f"[{i}]", float(v)) for i, v in enumerate(arr)]

    if not items:
        return []

    values = np.asarray([v for _, v in items], dtype=np.float64)
    median = float(np.median(values))
    threshold = ratio * median if median > 0 else float("inf")
    offenders = [(n, v) for n, v in items if v > threshold]

    if offenders:
        top = sorted(offenders, key=lambda t: -t[1])[:5]
        top_str = ", ".join(f"{n}={v:.3f}" for n, v in top)
        logger.warning(
            "[%s] %d scale(s) exceed %.1f x median (%.3f) — "
            "calibration set may contain outliers. Top offenders: %s. "
            "FP8 will still run but dynamic-range headroom on these "
            "layers is compressed. Consider lowering percentile, "
            "sampling more diversely, or keeping these layers in FP16.",
            label, len(offenders), ratio, median, top_str,
        )
    return offenders


# ---------------------------------------------------------------------------
# Stratified sampling helper
# ---------------------------------------------------------------------------

def stratified_sample_indices(
    metadata: "pd.DataFrame",  # type: ignore[name-defined]
    n: int = 8,
    *,
    task_filter: Optional[int] = None,
    task_col: str = "task_index",
    episode_col: str = "episode_index",
    frame_col: str = "frame_index",
    index_col: str = "index",
    exclude: Optional[Iterable[int]] = None,
) -> List[int]:
    """Pick N global frame indices, stratified by episode × frame-position.

    Matches the openpi-jax-mlir toolchain's default calibration pattern
    ("8 is sufficient"): choose
    ``min(num_episodes, max(n//2, 3))`` distinct episodes, then within
    each episode take equally-spaced frames.

    Args:
        metadata: a ``pandas.DataFrame`` with at minimum columns
            ``task_col``, ``episode_col``, ``frame_col``, ``index_col``.
            LIBERO rollouts have this layout by default.
        n: target sample count (default 8; the toolchain-validated
            sweet spot for fine-tuned models).
        task_filter: if set, only sample frames whose ``task_col`` equals
            this value.
        exclude: frame indices to exclude from the pool (e.g. the
            deployment target frame to avoid data leakage).

    Returns:
        List of exactly ``n`` global frame indices (or fewer if the
        filtered pool is smaller). Caller applies their own loader to
        turn them into observation dicts.

    Example::

        import pandas as pd
        from flash_vla.core.calibration import stratified_sample_indices
        df = pd.read_parquet("rollouts/meta.parquet")
        picks = stratified_sample_indices(df, n=8, task_filter=8)
        obs_list = [my_load_one_frame(i) for i in picks]
        model.calibrate(obs_list)
    """
    import pandas as pd  # local import to keep core CPU-light

    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if not isinstance(metadata, pd.DataFrame):
        raise TypeError(
            f"metadata must be a pandas.DataFrame, got {type(metadata).__name__}")
    required = {task_col, episode_col, frame_col, index_col}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"metadata missing columns: {sorted(missing)}")

    df = metadata
    if task_filter is not None:
        df = df[df[task_col] == task_filter]
        if len(df) == 0:
            raise ValueError(
                f"no rows match task_filter={task_filter} on column {task_col!r}")

    exclude_set = set(exclude) if exclude is not None else set()

    all_eps = sorted(df[episode_col].unique())
    # Toolchain pattern: min(num_episodes, max(n//2, 3)) distinct episodes.
    n_eps = min(len(all_eps), max(n // 2, 3))
    ep_stride = max(1, len(all_eps) // n_eps)
    chosen_eps = all_eps[::ep_stride][:n_eps]
    # Round UP so per_ep * num_eps >= n, ensuring full N is reachable.
    frames_per_ep = max(1, -(-n // max(len(chosen_eps), 1)))

    picks: List[int] = []
    for ep in chosen_eps:
        ep_df = df[df[episode_col] == ep].sort_values(frame_col)
        if len(ep_df) == 0:
            continue
        step = max(1, len(ep_df) // frames_per_ep)
        for i in range(0, len(ep_df), step):
            idx = int(ep_df.iloc[i][index_col])
            if idx in exclude_set:
                continue
            picks.append(idx)
            if len(picks) >= n:
                break
        if len(picks) >= n:
            break

    # Fallback top-up: if we still haven't hit n (e.g. excludes collided
    # with every stratified slot, or some episodes were tiny), walk the
    # filtered pool linearly to fill remaining slots without duplicates.
    if len(picks) < n:
        seen = set(picks) | exclude_set
        for idx in df[index_col].tolist():
            idx = int(idx)
            if idx in seen:
                continue
            picks.append(idx)
            seen.add(idx)
            if len(picks) >= n:
                break

    return picks[:n]


def stratified_sample(
    metadata: "pd.DataFrame",  # type: ignore[name-defined]
    load_fn: Callable[[int], Any],
    n: int = 8,
    **kwargs: Any,
) -> List[Any]:
    """Convenience wrapper: stratified indices → user-loaded observations.

    Calls :func:`stratified_sample_indices` to pick N indices, then
    applies ``load_fn`` to each, returning a list ready to pass into
    ``model.calibrate(obs_list)``.

    Args:
        metadata: see :func:`stratified_sample_indices`.
        load_fn: user function mapping a global frame index to an
            observation dict (e.g. ``{"image": np.ndarray, ...}``
            matching the frontend's ``predict`` contract).
        n: sample count (default 8).
        **kwargs: forwarded to :func:`stratified_sample_indices`
            (``task_filter``, column overrides, ``exclude``).
    """
    indices = stratified_sample_indices(metadata, n=n, **kwargs)
    return [load_fn(i) for i in indices]

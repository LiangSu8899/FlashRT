"""Unit tests for flash_rt.core.calibration.

CPU-only.
"""

import numpy as np
import pytest

from flash_rt.core.calibration import (
    accumulate_amax,
    format_summary,
    summarize_amax_dispersion,
)


def test_single_sample_equals_input():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = accumulate_amax([a], percentile=99.9)
    np.testing.assert_allclose(out, a, atol=1e-6)


def test_percentile_100_equals_max():
    samples = [np.array([1.0, 2.0]), np.array([3.0, 1.5]), np.array([2.0, 4.0])]
    out = accumulate_amax(samples, percentile=100.0)
    np.testing.assert_allclose(out, np.array([3.0, 4.0]), atol=1e-6)


def test_percentile_99p9_excludes_outlier():
    # 100 normal samples at ~1.0, then one outlier at 50.0.
    rng = np.random.default_rng(0)
    normals = [np.array([rng.normal(1.0, 0.05)], dtype=np.float32)
               for _ in range(100)]
    outlier = np.array([50.0], dtype=np.float32)
    samples = normals + [outlier]

    max_reduced = accumulate_amax(samples, percentile=100.0)
    p99_reduced = accumulate_amax(samples, percentile=99.0)

    assert max_reduced[0] >= 50.0  # true max includes the outlier
    assert p99_reduced[0] < 2.0    # 99th percentile skips the outlier


def test_empty_list_raises():
    with pytest.raises(ValueError):
        accumulate_amax([], percentile=99.9)


def test_non_1d_raises():
    with pytest.raises(ValueError):
        accumulate_amax([np.zeros((2, 2)), np.zeros((2, 2))], percentile=99.9)


def test_percentile_out_of_range():
    with pytest.raises(ValueError):
        accumulate_amax([np.array([1.0])], percentile=-1.0)
    with pytest.raises(ValueError):
        accumulate_amax([np.array([1.0])], percentile=101.0)


def test_summarize_shapes_and_keys():
    samples = [np.array([1.0, 2.0, 3.0]) * (1.0 + 0.01 * i) for i in range(10)]
    final = accumulate_amax(samples, percentile=99.9)
    summ = summarize_amax_dispersion(samples, final)
    assert summ["num_samples"] == 10
    assert summ["num_points"] == 3
    assert 0.0 <= summ["cutback_from_max_p50"] <= 1.0
    # format_summary must not crash and must mention N
    s = format_summary(summ)
    assert "N=10" in s


def test_outlier_cutback_detected_in_summary():
    """100 normals + 1 big outlier on point 0 only. With 101 samples and
    percentile=99.0, the interpolated rank lands just below the outlier,
    so the cutback for point 0 is ~100%."""
    normals = [np.array([1.0, 1.0]) for _ in range(100)]
    normals.append(np.array([1000.0, 1.0]))
    final = accumulate_amax(normals, percentile=99.0)
    summ = summarize_amax_dispersion(normals, final)
    assert summ["num_points_cut_gt_10pct"] >= 1


# ---------------------------------------------------------------------------
# check_scale_ceiling
# ---------------------------------------------------------------------------

from flash_rt.core.calibration import check_scale_ceiling, FP8_SCALE_RATIO_WARN


def test_check_scale_ceiling_no_offenders_quiet(caplog):
    """Healthy calibration: inter-layer variance 3-5× median, no warning."""
    caplog.clear()
    scales = {"l0": 0.05, "l1": 0.12, "l2": 0.3, "l3": 0.15, "l4": 0.08}
    offenders = check_scale_ceiling(scales, ratio=20.0)
    assert offenders == []
    assert not caplog.records


def test_check_scale_ceiling_warns_on_outlier(caplog):
    """One layer at 50× median → must warn."""
    import logging as _log
    caplog.set_level(_log.WARNING, logger="flash_rt.core.calibration")
    # median = 0.1, ratio 20 → threshold 2.0
    scales = {"l0": 0.1, "l1": 0.1, "l2": 0.1, "outlier": 5.0}
    offenders = check_scale_ceiling(scales, ratio=20.0, label="pytest_case")
    names = {n for n, _ in offenders}
    assert names == {"outlier"}
    assert any("pytest_case" in r.message for r in caplog.records)
    assert any("outlier" in r.message for r in caplog.records)


def test_check_scale_ceiling_array_input():
    # median([0.1, 0.1, 0.2, 0.15, 5.0]) = 0.2, ratio 20 → threshold 4.0
    offenders = check_scale_ceiling(
        [0.1, 0.1, 0.2, 0.15, 5.0], ratio=20.0)
    assert len(offenders) == 1
    assert offenders[0][0] == "[4]"


def test_check_scale_ceiling_matches_pi0_real_scales(caplog):
    """Regression guard: a real Pi0 calibration distribution (inter-layer
    variance 30× but no true outlier) must NOT fire at ratio=20."""
    caplog.clear()
    # Typical Pi0 fp8_act_scales distribution seen in production (0.05 median, 2 max).
    scales = {f"enc_{i}": v for i, v in enumerate(
        [0.03, 0.05, 0.08, 0.1, 0.12, 0.15, 0.3, 0.5, 0.6, 1.5, 2.0])}
    offenders = check_scale_ceiling(scales, ratio=20.0)
    # median=0.15, ratio 20 → threshold 3.0. Max was 2.0, no warning.
    assert offenders == []


# ---------------------------------------------------------------------------
# stratified_sample_indices
# ---------------------------------------------------------------------------

import pytest


def _make_dummy_meta():
    import pandas as pd
    # 3 episodes of task 8 (100 frames each), 3 episodes of task 9 (50 frames each).
    rows = []
    g = 0
    for ep_offset, task in enumerate([8, 8, 8, 9, 9, 9]):
        size = 100 if task == 8 else 50
        for f in range(size):
            rows.append({
                "index": g,
                "task_index": task,
                "episode_index": ep_offset,
                "frame_index": f,
            })
            g += 1
    return pd.DataFrame(rows)


def test_stratified_sample_indices_n8_task8():
    from flash_rt.core.calibration import stratified_sample_indices
    df = _make_dummy_meta()
    picks = stratified_sample_indices(df, n=8, task_filter=8)
    assert len(picks) == 8
    # All picks must belong to task 8
    chosen_tasks = set(df[df["index"].isin(picks)]["task_index"])
    assert chosen_tasks == {8}
    # All picks distinct (no dup) and spread across >= 2 distinct episodes
    assert len(set(picks)) == 8
    chosen_eps = set(df[df["index"].isin(picks)]["episode_index"])
    assert len(chosen_eps) >= 2


def test_stratified_sample_indices_excludes():
    from flash_rt.core.calibration import stratified_sample_indices
    df = _make_dummy_meta()
    exclude = {0, 100, 200}  # first frame of each task-8 episode
    picks = stratified_sample_indices(df, n=8, task_filter=8, exclude=exclude)
    assert not (set(picks) & exclude)


def test_stratified_sample_indices_missing_column():
    from flash_rt.core.calibration import stratified_sample_indices
    import pandas as pd
    df = pd.DataFrame({"wrong": [1, 2, 3]})
    with pytest.raises(ValueError, match="missing columns"):
        stratified_sample_indices(df, n=4)


def test_stratified_sample_indices_empty_after_filter():
    from flash_rt.core.calibration import stratified_sample_indices
    df = _make_dummy_meta()
    with pytest.raises(ValueError, match="no rows"):
        stratified_sample_indices(df, n=4, task_filter=99)


def test_stratified_sample_indices_bad_n():
    from flash_rt.core.calibration import stratified_sample_indices
    df = _make_dummy_meta()
    with pytest.raises(ValueError, match="n must be"):
        stratified_sample_indices(df, n=0, task_filter=8)


def test_stratified_sample_applies_load_fn():
    from flash_rt.core.calibration import stratified_sample
    df = _make_dummy_meta()
    load_fn = lambda idx: {"fake_frame_id": idx}
    obs = stratified_sample(df, load_fn, n=4, task_filter=8)
    assert len(obs) == 4
    assert all("fake_frame_id" in o for o in obs)
    # All frame ids should be from task 8 rows (global indices 0..299)
    ids = [o["fake_frame_id"] for o in obs]
    assert all(0 <= i < 300 for i in ids)

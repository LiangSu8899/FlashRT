"""Unit tests for the shared norm-stats loader.

Covers both the openpi (``norm_stats.json``) and lerobot
(``meta/stats.json``) on-disk schemas, plus the strict / non-strict
fallback behaviour. CPU-only — no torch / no GPU — so this runs in
every environment that has the package installed.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from flash_rt.core.utils.norm_stats import (
    load_norm_stats,
    lerobot_candidates,
    _is_lerobot_stats,
    _lerobot_to_openpi,
)


def _write(p: pathlib.Path, data: dict) -> pathlib.Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data))
    return p


# ─── openpi schema ────────────────────────────────────────────────────


def test_openpi_flat(tmp_path):
    payload = {
        "actions": {"q01": [-1.0, -1.0], "q99": [1.0, 1.0]},
        "state":   {"q01": [0.0, 0.0],  "q99": [1.0, 1.0]},
    }
    p = _write(tmp_path / "norm_stats.json", payload)
    out = load_norm_stats([p])
    assert out == payload


def test_openpi_wrapped(tmp_path):
    payload = {
        "norm_stats": {
            "actions": {"q01": [-1.0], "q99": [1.0]},
            "state":   {"q01": [0.0],  "q99": [1.0]},
        }
    }
    p = _write(tmp_path / "norm_stats.json", payload)
    out = load_norm_stats([p])
    assert out == payload["norm_stats"]


def test_openpi_first_hit_wins(tmp_path):
    p1 = _write(tmp_path / "a.json",
                {"actions": {"q01": [1.0], "q99": [2.0]}})
    p2 = _write(tmp_path / "b.json",
                {"actions": {"q01": [99.0], "q99": [100.0]}})
    out = load_norm_stats([p1, p2])
    assert out["actions"]["q01"] == [1.0]


# ─── lerobot schema ───────────────────────────────────────────────────


def test_lerobot_with_q01_q99(tmp_path):
    payload = {
        "observation.state": {
            "min": [0.0, 0.0], "max": [1.0, 1.0],
            "mean": [0.5, 0.5], "std": [0.1, 0.1],
            "q01": [0.05, 0.05], "q99": [0.95, 0.95],
        },
        "action": {
            "min": [-1.0], "max": [1.0],
            "mean": [0.0], "std": [0.5],
            "q01": [-0.9], "q99": [0.9],
        },
    }
    p = _write(tmp_path / "meta" / "stats.json", payload)
    out = load_norm_stats([p])
    assert out["actions"]["q01"] == [-0.9]
    assert out["actions"]["q99"] == [0.9]
    assert out["state"]["q01"] == [0.05, 0.05]
    # Min/max preserved
    assert out["actions"]["min"] == [-1.0]
    assert out["state"]["mean"] == [0.5, 0.5]


def test_lerobot_min_max_only_warns_and_falls_back(tmp_path, caplog):
    payload = {
        "action": {"min": [-2.0], "max": [2.0]},
        "observation.state": {"min": [0.0], "max": [10.0]},
    }
    p = _write(tmp_path / "meta" / "stats.json", payload)
    with caplog.at_level("WARNING"):
        out = load_norm_stats([p])
    # min/max copied into q01/q99
    assert out["actions"]["q01"] == [-2.0]
    assert out["actions"]["q99"] == [2.0]
    assert out["state"]["q01"] == [0.0]
    # Warning emitted at least once.
    assert any("min/max" in r.message.lower() or "min/max" in r.message
               for r in caplog.records)


def test_openpi_plural_actions_not_misread_as_lerobot(tmp_path):
    # openpi uses plural ``actions``; lerobot uses singular ``action``.
    # A payload with only the plural key should be recognised as openpi,
    # not translated through the lerobot adapter.
    payload = {"actions": {"q01": [-1.0], "q99": [1.0]}}
    p = _write(tmp_path / "meta" / "stats.json", payload)
    out = load_norm_stats([p])
    assert out == payload


# ─── selection / fallback ─────────────────────────────────────────────


def test_strict_raises_when_nothing_found(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        load_norm_stats([tmp_path / "nope.json", tmp_path / "no/way.json"])
    assert "openpi" in str(exc.value).lower()
    assert "lerobot" in str(exc.value).lower()


def test_non_strict_returns_none(tmp_path):
    out = load_norm_stats([tmp_path / "nope.json"], strict=False)
    assert out is None


def test_lerobot_then_openpi_skips_invalid_first_then_succeeds(tmp_path):
    # First candidate is unreadable JSON; second is a valid openpi file.
    bad = tmp_path / "bad.json"
    bad.write_text("not-json")
    good = _write(tmp_path / "good.json",
                  {"actions": {"q01": [0.0], "q99": [1.0]}})
    out = load_norm_stats([bad, good])
    assert out["actions"]["q99"] == [1.0]


def test_lerobot_candidates_paths(tmp_path):
    cands = lerobot_candidates(tmp_path)
    names = [c.name for c in cands]
    assert "stats.json" in names
    assert any(c.parent.name == "meta" for c in cands)


# ─── helper coverage ──────────────────────────────────────────────────


def test_is_lerobot_recognises_canonical_keys():
    assert _is_lerobot_stats(
        {"observation.state": {"min": [0], "max": [1]},
         "action": {"min": [-1], "max": [1]}})
    assert not _is_lerobot_stats(
        {"actions": {"q01": [0], "q99": [1]}})  # openpi flat
    assert not _is_lerobot_stats({})
    assert not _is_lerobot_stats({"foo": "bar"})


def test_lerobot_to_openpi_handles_missing_state_block():
    # Some lerobot dumps only include action stats.
    out = _lerobot_to_openpi(
        {"action": {"min": [-1.0], "max": [1.0]}})
    assert "actions" in out
    assert "state" not in out


# ─── Lerobot policy safetensors path (HF model release format) ────────


def _write_safetensors(path: pathlib.Path, tensors: dict) -> None:
    """Write a dict of {key: list[float]} as a safetensors file."""
    import numpy as np
    from safetensors.numpy import save_file
    arrs = {k: np.asarray(v, dtype=np.float32).reshape(-1) for k, v in tensors.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(arrs, str(path))


def test_lerobot_policy_safetensors_pair(tmp_path):
    """The HF lerobot model layout: policy_preprocessor + policy_postprocessor
    safetensors with flat ``<feature>.<stat>`` keys."""
    ckpt = tmp_path / "ckpt"
    pre = ckpt / "policy_preprocessor_step_2_normalizer_processor.safetensors"
    post = ckpt / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    _write_safetensors(pre, {
        "observation.state.q01": [0.0] * 8,
        "observation.state.q99": [1.0] * 8,
        "observation.state.mean": [0.5] * 8,
        "observation.images.image.q01": [0.0],
    })
    _write_safetensors(post, {
        "action.q01": [-0.5, -0.6, -0.7, -0.07, -0.1, -0.1, -1.0],
        "action.q99": [0.7, 0.7, 0.8, 0.08, 0.1, 0.1, 0.9],
        "action.mean": [0.0] * 7,
    })
    out = load_norm_stats([], checkpoint_dir=ckpt)
    assert "actions" in out and "state" in out
    assert len(out["actions"]["q01"]) == 7
    assert len(out["state"]["q01"]) == 8
    assert out["actions"]["q01"][6] == pytest.approx(-1.0)
    assert out["actions"]["q99"][0] == pytest.approx(0.7)
    assert "mean" in out["actions"]


def test_lerobot_policy_safetensors_step_index_varies(tmp_path):
    """Real releases use step_0, step_2, etc. — the glob must match any."""
    ckpt = tmp_path / "ckpt"
    pre = ckpt / "policy_preprocessor_step_5_normalizer_processor.safetensors"
    post = ckpt / "policy_postprocessor_step_3_unnormalizer_processor.safetensors"
    _write_safetensors(pre, {"observation.state.q01": [0.0],
                             "observation.state.q99": [1.0]})
    _write_safetensors(post, {"action.q01": [-1.0], "action.q99": [1.0]})
    out = load_norm_stats([], checkpoint_dir=ckpt)
    assert out["actions"]["q01"] == [-1.0]


def test_lerobot_policy_safetensors_only_unnormalizer_present(tmp_path):
    """If only one of the pair is present, the loader must fall through —
    a partial pair is not enough to produce coherent stats."""
    ckpt = tmp_path / "ckpt"
    post = ckpt / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    _write_safetensors(post, {"action.q01": [-1.0], "action.q99": [1.0]})
    out = load_norm_stats([], checkpoint_dir=ckpt, strict=False)
    assert out is None  # missing preprocessor; pair incomplete


def test_lerobot_policy_takes_precedence_when_no_json(tmp_path):
    """When candidates list is empty and only the safetensors pair
    exists, the loader picks them up via checkpoint_dir."""
    ckpt = tmp_path / "ckpt"
    _write_safetensors(
        ckpt / "policy_preprocessor_step_0_normalizer_processor.safetensors",
        {"observation.state.q01": [0.0], "observation.state.q99": [1.0]})
    _write_safetensors(
        ckpt / "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        {"action.q01": [-2.0], "action.q99": [2.0]})
    out = load_norm_stats([ckpt / "norm_stats.json"], checkpoint_dir=ckpt)
    assert out["actions"]["q99"] == [2.0]


def test_strict_error_lists_lerobot_policy_format(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        load_norm_stats([tmp_path / "nope.json"], checkpoint_dir=tmp_path)
    msg = str(exc.value)
    assert "policy_" in msg and "safetensors" in msg


def test_json_candidate_wins_over_policy_safetensors(tmp_path):
    """If both an openpi norm_stats.json and a lerobot policy
    safetensors pair are present, JSON takes precedence (it's
    explicitly listed in candidates and walked first)."""
    ckpt = tmp_path / "ckpt"
    j = _write(
        ckpt / "norm_stats.json",
        {"actions": {"q01": [99.0], "q99": [100.0]}})
    _write_safetensors(
        ckpt / "policy_preprocessor_step_0_normalizer_processor.safetensors",
        {"observation.state.q01": [0.0], "observation.state.q99": [1.0]})
    _write_safetensors(
        ckpt / "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        {"action.q01": [-1.0], "action.q99": [1.0]})
    out = load_norm_stats([j], checkpoint_dir=ckpt)
    assert out["actions"]["q01"] == [99.0]

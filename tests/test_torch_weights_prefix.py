"""Unit tests for the lerobot ``model.`` prefix auto-strip in
``SafetensorsSource`` / ``MultiSafetensorsSource``.

The lerobot HF policy releases (e.g. ``lerobot/pi05_libero_finetuned
_v044``) wrap every weight key under an extra ``model.`` namespace
relative to the openpi-converted layout the spec was written for.
The source classes auto-detect this and strip the prefix transparently
so the rest of the loader stays written in openpi keys. These tests
cover both the auto-detect heuristic and the explicit
``strip_prefix=`` constructor override.
"""
from __future__ import annotations

import pathlib

import pytest
import torch
from safetensors.torch import save_file

from flash_rt.executors.torch_weights import (
    MultiSafetensorsSource,
    SafetensorsSource,
    _autodetect_strip_prefix,
)


def _write_safetensors(path: pathlib.Path, keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {k: torch.zeros(2, dtype=torch.float16) for k in keys}
    save_file(tensors, str(path))


# ─── Auto-detect heuristic ────────────────────────────────────────────


def test_autodetect_no_strip_for_openpi_layout():
    keys = {
        "paligemma_with_expert.gemma_expert.lm_head.weight",
        "paligemma_with_expert.paligemma.model.embed_tokens.weight",
    }
    assert _autodetect_strip_prefix(keys) == ""


def test_autodetect_strips_model_prefix_for_lerobot_layout():
    keys = {
        "model.paligemma_with_expert.gemma_expert.lm_head.weight",
        "model.paligemma_with_expert.paligemma.model.embed_tokens.weight",
        "model.action_in_proj.weight",
        "model.time_mlp_in.weight",
    }
    assert _autodetect_strip_prefix(keys) == "model."


def test_autodetect_pi0fast_sentinel():
    """Pi0-FAST has no action expert, so its top-level namespace is
    ``paligemma.model.*`` rather than ``paligemma_with_expert.*``.
    The autodetect must still flag the lerobot wrap on this family."""
    keys = {
        "model.paligemma.model.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight",
        "model.paligemma.model.language_model.layers.0.self_attn.q_proj.weight",
    }
    assert _autodetect_strip_prefix(keys) == "model."

    # Bare pi0fast layout -> no strip.
    bare = {
        "paligemma.model.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight",
        "paligemma.model.language_model.layers.0.self_attn.q_proj.weight",
    }
    assert _autodetect_strip_prefix(bare) == ""


def test_autodetect_no_strip_when_bare_paligemma_keys_also_present():
    """Defensive: if the file mixes both layouts (corrupt / merged
    checkpoint), don't strip — better to fail loudly downstream than
    silently choose the wrong namespace."""
    keys = {
        "paligemma_with_expert.foo.weight",
        "model.paligemma_with_expert.bar.weight",
    }
    assert _autodetect_strip_prefix(keys) == ""


def test_autodetect_no_strip_for_empty_or_unknown():
    assert _autodetect_strip_prefix(set()) == ""
    # Unrelated keys (e.g. some other model family without paligemma)
    assert _autodetect_strip_prefix({"foo.bar", "baz.qux"}) == ""
    # Has prefix but no recognisable paligemma sentinel — don't strip.
    assert _autodetect_strip_prefix(
        {"model.foo.weight", "model.bar.weight"}) == ""


# ─── SafetensorsSource ────────────────────────────────────────────────


def test_safetensors_source_strips_lerobot_prefix(tmp_path):
    keys = [
        "model.paligemma_with_expert.gemma_expert.lm_head.weight",
        "model.paligemma_with_expert.paligemma.lm_head.weight",
        "model.action_in_proj.weight",
    ]
    p = tmp_path / "model.safetensors"
    _write_safetensors(p, keys)

    src = SafetensorsSource(str(p), device="cpu")
    # Spec-side keys (without ``model.``) are present and resolvable.
    spec_key = "paligemma_with_expert.gemma_expert.lm_head.weight"
    assert src.has(spec_key)
    t = src.get(spec_key)
    assert tuple(t.shape) == (2,)
    # Bare key for action_in_proj also works.
    assert src.has("action_in_proj.weight")


def test_safetensors_source_no_strip_for_openpi_layout(tmp_path):
    keys = [
        "paligemma_with_expert.gemma_expert.lm_head.weight",
        "paligemma_with_expert.paligemma.lm_head.weight",
    ]
    p = tmp_path / "model.safetensors"
    _write_safetensors(p, keys)

    src = SafetensorsSource(str(p), device="cpu")
    assert src.has("paligemma_with_expert.gemma_expert.lm_head.weight")
    # ``model.`` prefixed lookup must NOT incorrectly resolve.
    assert not src.has("model.paligemma_with_expert.gemma_expert.lm_head.weight")


def test_safetensors_source_explicit_strip_prefix_override(tmp_path):
    keys = ["wrapper.foo.weight", "wrapper.bar.weight"]
    p = tmp_path / "model.safetensors"
    _write_safetensors(p, keys)

    src = SafetensorsSource(str(p), device="cpu", strip_prefix="wrapper.")
    assert src.has("foo.weight")
    assert src.has("bar.weight")
    assert not src.has("wrapper.foo.weight")
    t = src.get("foo.weight")
    assert tuple(t.shape) == (2,)


def test_safetensors_source_explicit_empty_disables_autodetect(tmp_path):
    """``strip_prefix=""`` must disable auto-detect — useful when the
    user knows the file is openpi-style and wants strict matching."""
    keys = [
        "model.paligemma_with_expert.gemma_expert.lm_head.weight",
    ]
    p = tmp_path / "model.safetensors"
    _write_safetensors(p, keys)

    src = SafetensorsSource(str(p), device="cpu", strip_prefix="")
    # Auto-detect would have stripped ``model.``; with explicit empty
    # override it should not.
    assert src.has("model.paligemma_with_expert.gemma_expert.lm_head.weight")
    assert not src.has("paligemma_with_expert.gemma_expert.lm_head.weight")


# ─── MultiSafetensorsSource ────────────────────────────────────────────


def test_multi_safetensors_source_autodetect(tmp_path):
    p1 = tmp_path / "shard-001.safetensors"
    p2 = tmp_path / "shard-002.safetensors"
    _write_safetensors(p1, [
        "model.paligemma_with_expert.gemma_expert.lm_head.weight",
        "model.action_in_proj.weight",
    ])
    _write_safetensors(p2, [
        "model.paligemma_with_expert.paligemma.lm_head.weight",
        "model.time_mlp_in.weight",
    ])
    src = MultiSafetensorsSource([p1, p2], device="cpu")
    assert src.has("paligemma_with_expert.gemma_expert.lm_head.weight")
    assert src.has("paligemma_with_expert.paligemma.lm_head.weight")
    assert src.has("action_in_proj.weight")
    assert src.has("time_mlp_in.weight")


def test_multi_safetensors_source_dup_key_across_shards_still_raises(tmp_path):
    """Auto-strip must not mask duplicate-key corruption across shards."""
    p1 = tmp_path / "a.safetensors"
    p2 = tmp_path / "b.safetensors"
    _write_safetensors(p1, ["model.paligemma_with_expert.foo.weight"])
    _write_safetensors(p2, ["model.paligemma_with_expert.foo.weight"])
    with pytest.raises(ValueError, match="multiple shards"):
        MultiSafetensorsSource([p1, p2], device="cpu")

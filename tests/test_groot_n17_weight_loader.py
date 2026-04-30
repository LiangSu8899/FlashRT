"""Smoke + key-coverage tests for the N1.7 declarative WEIGHT_SPEC.

Verifies that:
  * ``build_spec()`` produces a non-empty ``ModelWeightSpec``;
  * every Item key (post layer-fmt expansion + composite-key resolution)
    is present in the actual N1.7 safetensors index;
  * total covered keys = block_items + singleton_items, and the diff vs
    the ckpt's full key set is the documented intentional skip
    (``backbone.model.lm_head.weight`` only — unused at inference time).

Skips automatically when the ckpt is not present (so the test is safe to
run on any machine, including CI without the model).
"""

from __future__ import annotations

import glob
import os

import pytest

from flash_vla.executors.weight_loader import Item, LayerBlock
from flash_vla.executors.torch_weights import Cat
from flash_vla.frontends.torch._groot_n17_thor_spec import build_spec


N17_CACHE_GLOB = (
    "/root/.cache/huggingface/hub/models--nvidia--GR00T-N1.7-3B/"
    "snapshots/*/model-*.safetensors"
)


def _ckpt_paths() -> list[str]:
    paths = sorted(glob.glob(N17_CACHE_GLOB))
    return paths


def _ckpt_keys() -> set[str]:
    """Read the union of safetensors keys across all N1.7 shards."""
    from safetensors import safe_open
    keys: set[str] = set()
    for p in _ckpt_paths():
        with safe_open(p, framework="pt", device="cpu") as h:
            keys.update(h.keys())
    return keys


def _expand_keys(spec) -> list[str]:
    """Expand all spec items into the concrete key strings they will read."""
    out: list[str] = []
    for blk in spec.blocks:
        for i in range(blk.num_layers):
            for it in blk.items:
                k = it.key
                if isinstance(k, str):
                    out.append(k.replace("{i}", str(i)))
                elif isinstance(k, Cat):
                    for sk in k.keys:
                        out.append(sk.replace("{i}", str(i)))
                else:
                    pytest.fail(f"unsupported composite key {type(k).__name__} in {it.name}")
    for it in spec.singletons:
        if isinstance(it.key, str):
            out.append(it.key)
        elif isinstance(it.key, Cat):
            out.extend(it.key.keys)
        else:
            pytest.fail(f"unsupported composite singleton key {type(it.key).__name__}")
    return out


def test_spec_builds():
    spec = build_spec()
    assert spec.framework == "torch"
    assert len(spec.blocks) == 4
    names = [b.name for b in spec.blocks]
    assert names == ["qwen3vl_vit", "qwen3vl_llm", "vl_self_attn", "dit"]
    counts = [(b.name, b.num_layers, len(b.items)) for b in spec.blocks]
    assert counts == [
        ("qwen3vl_vit", 24, 12),
        ("qwen3vl_llm", 16, 9),
        ("vl_self_attn", 4, 16),
        ("dit", 32, 14),
    ]
    assert len(spec.singletons) > 40


def test_spec_keys_present_in_ckpt():
    """Every key the spec asks for must exist in the actual N1.7 safetensors."""
    paths = _ckpt_paths()
    if not paths:
        pytest.skip(f"N1.7 ckpt not at {N17_CACHE_GLOB}")
    ckpt_keys = _ckpt_keys()
    spec = build_spec()
    requested = _expand_keys(spec)

    missing = [k for k in requested if k not in ckpt_keys]
    assert not missing, f"{len(missing)} spec keys missing from ckpt:\n  " + "\n  ".join(missing[:30])


def test_spec_covers_ckpt_minus_intentional_skips():
    """The set of keys the spec covers should equal ckpt_keys minus a small,
    documented set of intentional skips."""
    paths = _ckpt_paths()
    if not paths:
        pytest.skip(f"N1.7 ckpt not at {N17_CACHE_GLOB}")
    ckpt_keys = _ckpt_keys()
    spec = build_spec()
    requested = set(_expand_keys(spec))

    intentional_skips = {
        "backbone.model.lm_head.weight",  # not used at inference (we take hidden_states[-1])
    }

    extras_in_spec = requested - ckpt_keys
    assert not extras_in_spec, f"spec asks for keys not in ckpt: {extras_in_spec}"

    uncovered = ckpt_keys - requested - intentional_skips
    assert not uncovered, (
        f"{len(uncovered)} ckpt keys are uncovered by spec (and not in intentional_skips):\n  "
        + "\n  ".join(sorted(uncovered)[:30])
    )


def test_spec_no_duplicate_keys():
    spec = build_spec()
    requested = _expand_keys(spec)
    counts = {}
    for k in requested:
        counts[k] = counts.get(k, 0) + 1
    dups = {k: c for k, c in counts.items() if c > 1}
    assert not dups, f"duplicates: {dups}"

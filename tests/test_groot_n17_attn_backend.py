"""ThorGrootN17AttnBackend protocol conformance.

* Spec has 5 sites with the right shapes.
* ``get_slot_ptrs`` returns the same dict-of-ints across calls and is
  stable for the backend's lifetime (the contract from
  ``docs/extension/attention_backend.md`` §9).
* Out-of-range layer indices are rejected.

The actual ``run`` dispatch is exercised end-to-end by the precision
test in Phase 3d (it requires real device tensors and kernel calls).
"""

from __future__ import annotations

import pytest

from flash_vla.hardware.thor.attn_backend_groot_n17 import (
    ThorGrootN17AttnBackend,
    make_groot_n17_attention_spec,
)


class _FakeCtx:
    cpp = object()


def _make_backend():
    spec = make_groot_n17_attention_spec(
        num_views=2,
        llm_seq_max=1024,
        vl_self_attn_seq_max=1024,
        sa=41,           # 1 state + action_horizon 40
        s_kv_text=256,   # text tokens
        s_kv_image=512,  # 2 views x 256 patches
    )
    nL_cross = spec.site("dit_cross").num_layers
    return ThorGrootN17AttnBackend(
        spec,
        vit_slots={"qkv": 0xA000, "O": 0xB000, "D": 16 * 64},
        llm_slots={
            "ctx": _FakeCtx(),
            "Q": 0x10000, "K": 0x11000, "V": 0x12000, "O": 0x13000,
            "logits": 0x14000, "scale": 1.0 / (128 ** 0.5),
        },
        vl_self_attn_slots={
            "ctx": _FakeCtx(),
            "Q": 0x20000, "K": 0x21000, "V": 0x22000, "O": 0x23000,
            "logits": 0x24000, "scale": 1.0 / (64 ** 0.5),
        },
        dit_self_slots={
            "ctx": _FakeCtx(),
            "Q": 0x30000, "K": 0x31000, "V": 0x32000, "O": 0x33000,
            "logits": 0x34000, "scale": 1.0 / (48 ** 0.5),
        },
        dit_cross_slots={
            "ctx": _FakeCtx(),
            "Q": 0x40000,
            "K_layers": [0x40100 + i * 0x10 for i in range(nL_cross)],
            "V_layers": [0x40200 + i * 0x10 for i in range(nL_cross)],
            "O": 0x41000, "logits": 0x42000, "scale": 1.0 / (48 ** 0.5),
        },
    )


def test_spec_shapes():
    spec = make_groot_n17_attention_spec(
        num_views=2, llm_seq_max=968, vl_self_attn_seq_max=968,
        sa=41, s_kv_text=128, s_kv_image=512,
    )
    sites = {n: spec.site(n) for n in ("vit", "llm", "vl_self_attn", "dit_self", "dit_cross")}
    assert sites["vit"].num_layers == 24
    assert sites["vit"].num_q_heads == 16 and sites["vit"].head_dim == 64
    assert sites["vit"].batch_axis == 2

    assert sites["llm"].num_layers == 16
    assert sites["llm"].head_dim == 128
    assert sites["llm"].num_q_heads == 16  # GQA pre-expanded to MHA at kernel boundary

    assert sites["vl_self_attn"].num_layers == 4
    assert sites["vl_self_attn"].num_q_heads == 32 and sites["vl_self_attn"].head_dim == 64

    assert sites["dit_self"].num_layers == 16
    assert sites["dit_cross"].num_layers == 16
    assert sites["dit_cross"].max_kv_seq == 512  # max(text=128, image=512)


def test_pointer_stability_per_site_layer():
    backend = _make_backend()

    # ViT: layer_idx is irrelevant — same slots for any value.
    p1 = backend.get_slot_ptrs("vit", 0)
    p2 = backend.get_slot_ptrs("vit", 17)
    assert p1 == p2
    # K/V offsets reflect interleaved QKV layout.
    D = 16 * 64
    assert p1["Q"] == 0xA000
    assert p1["K"] == 0xA000 + D * 2
    assert p1["V"] == 0xA000 + 2 * D * 2
    assert p1["O"] == 0xB000

    # LLM, vl_self_attn, dit_self — same K/V across all layers (single buffer).
    for site, expected in [
        ("llm",          {"Q": 0x10000, "K": 0x11000, "V": 0x12000, "O": 0x13000}),
        ("vl_self_attn", {"Q": 0x20000, "K": 0x21000, "V": 0x22000, "O": 0x23000}),
        ("dit_self",     {"Q": 0x30000, "K": 0x31000, "V": 0x32000, "O": 0x33000}),
    ]:
        nL = backend._spec.site(site).num_layers
        for li in range(nL):
            assert backend.get_slot_ptrs(site, li) == expected

    # dit_cross — Q/O constant; K/V vary per layer.
    for li in range(16):
        p = backend.get_slot_ptrs("dit_cross", li)
        assert p["Q"] == 0x40000
        assert p["O"] == 0x41000
        assert p["K"] == 0x40100 + li * 0x10
        assert p["V"] == 0x40200 + li * 0x10


def test_out_of_range_layer_rejected():
    backend = _make_backend()
    with pytest.raises(IndexError):
        backend.get_slot_ptrs("llm", 16)
    with pytest.raises(IndexError):
        backend.get_slot_ptrs("dit_cross", 16)
    with pytest.raises(IndexError):
        backend.get_slot_ptrs("dit_self", -1)


def test_unknown_site_rejected():
    backend = _make_backend()
    with pytest.raises(KeyError):
        backend.get_slot_ptrs("siglip", 0)  # N1.6 site name; should be 'vit' for N1.7
    with pytest.raises(KeyError):
        backend.get_slot_ptrs("qwen3", 0)


def test_construction_validates_dit_cross_layer_lists():
    spec = make_groot_n17_attention_spec(
        num_views=1, llm_seq_max=512, vl_self_attn_seq_max=512,
        sa=41, s_kv_text=64, s_kv_image=256,
    )
    common = dict(
        vit_slots={"qkv": 1, "O": 2, "D": 16 * 64},
        llm_slots={"ctx": _FakeCtx(), "Q": 1, "K": 2, "V": 3, "O": 4, "logits": 5, "scale": 1.0},
        vl_self_attn_slots={"ctx": _FakeCtx(), "Q": 1, "K": 2, "V": 3, "O": 4, "logits": 5, "scale": 1.0},
        dit_self_slots={"ctx": _FakeCtx(), "Q": 1, "K": 2, "V": 3, "O": 4, "logits": 5, "scale": 1.0},
    )
    # K_layers length wrong
    with pytest.raises(ValueError, match="K_layers/V_layers length"):
        ThorGrootN17AttnBackend(
            spec,
            **common,
            dit_cross_slots={
                "ctx": _FakeCtx(), "Q": 1,
                "K_layers": [1] * 8,  # wrong: should be 16
                "V_layers": [2] * 16,
                "O": 3, "logits": 4, "scale": 1.0,
            },
        )
    # K_layers contains a null
    with pytest.raises(ValueError, match="is null"):
        ThorGrootN17AttnBackend(
            spec,
            **common,
            dit_cross_slots={
                "ctx": _FakeCtx(), "Q": 1,
                "K_layers": [1, 0] + [2] * 14,
                "V_layers": [3] * 16,
                "O": 4, "logits": 5, "scale": 1.0,
            },
        )

#!/usr/bin/env python3
"""Unit tests for ThorGrootAttnBackend — construction + get_slot_ptrs.

CPU-only; uses fake integer pointers, no GPU, no fvk dispatch beyond
the validation-error paths (which raise before the kernel call).
"""

from __future__ import annotations

import sys
import traceback


def _fake_slots():
    """Build a self-consistent slot dict set for GROOT."""
    D_sig = 16 * 72  # 1152

    # Each non-siglip site carries its own ctx (CKernelQwen3.ctx /
    # CKernelDiTHead.ctx are distinct in production).
    class _Ctx:
        def __init__(self, tag): self.cpp = tag

    siglip = {"qkv": 0x1000, "O": 0x2000, "D": D_sig}
    qwen3 = {
        "ctx":    _Ctx(0xAA1),
        "Q":      0x3000,
        "K":      0x3100,
        "V":      0x3200,
        "O":      0x3300,
        "logits": 0x3400,
        "scale":  1.0 / (128 ** 0.5),
    }
    dit_self = {
        "ctx":    _Ctx(0xAA2),
        "Q":      0x4000,
        "K":      0x4100,
        "V":      0x4200,
        "O":      0x4300,
        "logits": 0x4400,
        "scale":  1.0 / (48 ** 0.5),
    }
    dit_cross = {
        "ctx":    _Ctx(0xAA3),
        "Q":      0x5000,
        "K_layers": [0x6000 + i * 0x100 for i in range(16)],
        "V_layers": [0x7000 + i * 0x100 for i in range(16)],
        "O":      0x5100,
        "logits": 0x5200,
        "scale":  1.0 / (48 ** 0.5),
    }
    return siglip, qwen3, dit_self, dit_cross


def _build(spec_override=None, **slot_overrides):
    from flash_rt.hardware.thor.attn_backend_groot import (
        ThorGrootAttnBackend, make_groot_attention_spec,
    )
    spec = spec_override or make_groot_attention_spec(
        num_views=2, qwen3_seq_max=1024, sa=17, s_kv=600)
    siglip, qwen3, dit_self, dit_cross = _fake_slots()
    slots = {"siglip_slots": siglip, "qwen3_slots": qwen3,
             "dit_self_slots": dit_self, "dit_cross_slots": dit_cross}
    slots.update(slot_overrides)
    return ThorGrootAttnBackend(spec, **slots)


def test_construct_ok():
    b = _build()
    assert set(b.sites()) == {"siglip", "qwen3", "dit_self", "dit_cross"}, b.sites()
    assert b.head_dim("qwen3") == 128
    assert b.num_q_heads("dit_self") == 32
    assert b.num_kv_heads("dit_cross") == 32
    print("  PASS  test_construct_ok")


def test_groot_spec_mha_kernels():
    """qwen3, dit_self, dit_cross all carry extra={'kernel': 'mha'}."""
    from flash_rt.hardware.thor.attn_backend_groot import make_groot_attention_spec
    spec = make_groot_attention_spec(num_views=2, qwen3_seq_max=1024, sa=17, s_kv=600)
    for name in ("qwen3", "dit_self", "dit_cross"):
        assert spec.site(name).extra.get("kernel") == "mha", name
    # siglip does NOT carry a kernel tag (uses fmha_strided_full path).
    assert "kernel" not in spec.site("siglip").extra
    print("  PASS  test_groot_spec_mha_kernels")


def test_siglip_slot_ptrs():
    b = _build()
    D = 16 * 72
    for li in (0, 5, 26):
        p = b.get_slot_ptrs("siglip", li)  # layer_idx ignored for siglip
        assert p["Q"] == 0x1000
        assert p["K"] == 0x1000 + D * 2
        assert p["V"] == 0x1000 + 2 * D * 2
        assert p["O"] == 0x2000
    print("  PASS  test_siglip_slot_ptrs")


def test_qwen3_slot_ptrs():
    b = _build()
    for li in (0, 7, 15):  # 16 layers share the same buffers
        p = b.get_slot_ptrs("qwen3", li)
        assert p["Q"] == 0x3000
        assert p["K"] == 0x3100
        assert p["V"] == 0x3200
        assert p["O"] == 0x3300
    print("  PASS  test_qwen3_slot_ptrs")


def test_dit_self_slot_ptrs():
    b = _build()
    p = b.get_slot_ptrs("dit_self", 0)
    assert p["Q"] == 0x4000 and p["K"] == 0x4100 and p["O"] == 0x4300
    print("  PASS  test_dit_self_slot_ptrs")


def test_dit_cross_per_layer_kv():
    """dit_cross K/V come from per-layer precomputed lists."""
    b = _build()
    for li in (0, 7, 15):
        p = b.get_slot_ptrs("dit_cross", li)
        assert p["Q"] == 0x5000
        assert p["K"] == 0x6000 + li * 0x100, (li, hex(p["K"]))
        assert p["V"] == 0x7000 + li * 0x100, (li, hex(p["V"]))
        assert p["O"] == 0x5100
    print("  PASS  test_dit_cross_per_layer_kv")


def test_layer_idx_oob():
    b = _build()
    for site, nL in [("qwen3", 16), ("dit_self", 16), ("dit_cross", 16)]:
        for bad in (-1, nL, nL + 5):
            try:
                b.get_slot_ptrs(site, bad)
            except IndexError:
                continue
            raise AssertionError(f"{site} layer_idx={bad} should have raised")
    print("  PASS  test_layer_idx_oob")


def test_reject_null_ptr():
    siglip, qwen3, dit_self, dit_cross = _fake_slots()
    qwen3["Q"] = 0
    try:
        _build(qwen3_slots=qwen3)
    except ValueError as e:
        assert "null" in str(e).lower(), str(e)
        print("  PASS  test_reject_null_ptr")
        return
    raise AssertionError("null ptr should have been rejected")


def test_reject_bad_D():
    siglip, qwen3, dit_self, dit_cross = _fake_slots()
    siglip["D"] = 99  # wrong
    try:
        _build(siglip_slots=siglip)
    except ValueError as e:
        assert "num_q_heads*head_dim" in str(e), str(e)
        print("  PASS  test_reject_bad_D")
        return
    raise AssertionError("bad D should have been rejected")


def test_reject_wrong_kv_layers_len():
    siglip, qwen3, dit_self, dit_cross = _fake_slots()
    dit_cross["K_layers"] = dit_cross["K_layers"][:5]  # too short
    try:
        _build(dit_cross_slots=dit_cross)
    except ValueError as e:
        assert "K_layers" in str(e) or "V_layers" in str(e), str(e)
        print("  PASS  test_reject_wrong_kv_layers_len")
        return
    raise AssertionError("wrong K_layers length should have been rejected")


def test_reject_extra_site():
    from flash_rt.hardware.thor.attn_backend_groot import ThorGrootAttnBackend
    from flash_rt.hardware.backend import AttentionSpec
    spec = AttentionSpec()
    for name in ("siglip", "qwen3", "dit_self", "dit_cross", "rogue"):
        spec.add_site(name, num_layers=1, num_q_heads=1, num_kv_heads=1,
                       head_dim=1, max_q_seq=1)
    siglip, qwen3, dit_self, dit_cross = _fake_slots()
    try:
        ThorGrootAttnBackend(spec, siglip_slots=siglip,
                              qwen3_slots=qwen3, dit_self_slots=dit_self,
                              dit_cross_slots=dit_cross)
    except ValueError as e:
        assert "sites" in str(e), str(e)
        print("  PASS  test_reject_extra_site")
        return
    raise AssertionError("extra site should have been rejected")


def main() -> int:
    tests = [
        test_construct_ok,
        test_groot_spec_mha_kernels,
        test_siglip_slot_ptrs,
        test_qwen3_slot_ptrs,
        test_dit_self_slot_ptrs,
        test_dit_cross_per_layer_kv,
        test_layer_idx_oob,
        test_reject_null_ptr,
        test_reject_bad_D,
        test_reject_wrong_kv_layers_len,
        test_reject_extra_site,
    ]
    print("=== ThorGrootAttnBackend unit tests ===")
    failures = 0
    for t in tests:
        try:
            t()
        except Exception:
            failures += 1
            print(f"  FAIL  {t.__name__}")
            traceback.print_exc()
    print(f"--- {len(tests) - failures}/{len(tests)} passed ---")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

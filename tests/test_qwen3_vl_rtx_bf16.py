"""Smoke tests for the Qwen3-VL BF16 (Orin) multimodal frontend.

CI-friendly: no checkpoint, no GPU. Covers import wiring, the fail-fast
kernel-module check, kernel-list completeness, and the weight-loader
invariant assertions. Mirrors the coverage level of
``test_qwen3_vl_smoke.py`` and ``test_qwen3_vl_fp8_sm89.py``.

Run:
    PYTHONPATH=. python -m pytest tests/test_qwen3_vl_rtx_bf16.py -v
"""
from __future__ import annotations

import importlib

import pytest


# ── wiring / fail-fast ──

def test_qwen3_vl_rtx_bf16_frontend_imports():
    m = importlib.import_module('flash_rt.frontends.torch.qwen3_vl_rtx_bf16')
    assert hasattr(m, 'Qwen3VlTorchFrontendRtxBF16')
    assert hasattr(m, '_require_qwen3_vl_rtx_bf16_kernels')


def test_qwen3_vl_rtx_bf16_kernel_lists_are_bf16_only():
    m = importlib.import_module('flash_rt.frontends.torch.qwen3_vl_rtx_bf16')
    names = set(m._QWEN3_VL_RTX_BF16_CORE_FNS) | set(m._QWEN3_VL_RTX_BF16_VISION_FNS)
    assert 'bf16_matmul_bf16' in names
    assert 'qwen3_vl_bf16_gemv_m1' in names
    assert 'qwen3_q_norm_rope_qstage_bf16' in names
    assert not any('fp8' in name or 'nvfp4' in name for name in names)


# ── kernel function lists: non-empty, no duplicates ──

def test_core_kernel_list_non_empty_and_unique():
    m = importlib.import_module('flash_rt.frontends.torch.qwen3_vl_rtx_bf16')
    assert len(m._QWEN3_VL_RTX_BF16_CORE_FNS) > 0
    assert len(set(m._QWEN3_VL_RTX_BF16_CORE_FNS)) == len(m._QWEN3_VL_RTX_BF16_CORE_FNS)


def test_vision_kernel_list_non_empty_and_unique():
    m = importlib.import_module('flash_rt.frontends.torch.qwen3_vl_rtx_bf16')
    assert len(m._QWEN3_VL_RTX_BF16_VISION_FNS) > 0
    assert len(set(m._QWEN3_VL_RTX_BF16_VISION_FNS)) == len(m._QWEN3_VL_RTX_BF16_VISION_FNS)


# ── fail-fast: missing kernel symbol detection ──

def test_require_kernels_raises_on_missing_core_symbol():
    """Dropping a single core kernel function must raise at the check."""
    m = importlib.import_module('flash_rt.frontends.torch.qwen3_vl_rtx_bf16')
    target = m._QWEN3_VL_RTX_BF16_CORE_FNS[0]

    class _FakeVlk:
        pass
    for fn in m._QWEN3_VL_RTX_BF16_VISION_FNS:
        setattr(_FakeVlk, fn, lambda *a, **k: None)

    class _FakeFvk:
        pass
    for fn in m._QWEN3_VL_RTX_BF16_CORE_FNS:
        if fn != target:
            setattr(_FakeFvk, fn, lambda *a, **k: None)

    import unittest.mock as mock
    with mock.patch.dict('sys.modules', {
        'flash_rt.flash_rt_kernels': _FakeFvk,
        'flash_rt.flash_rt_qwen3_vl_kernels': _FakeVlk,
    }):
        with pytest.raises(RuntimeError) as ei:
            m._require_qwen3_vl_rtx_bf16_kernels()
        assert target in str(ei.value)


def test_require_kernels_raises_on_missing_vision_symbol():
    """Dropping a single vision kernel function must raise at the check."""
    m = importlib.import_module('flash_rt.frontends.torch.qwen3_vl_rtx_bf16')
    target = m._QWEN3_VL_RTX_BF16_VISION_FNS[0]

    class _FakeFvk:
        pass
    for fn in m._QWEN3_VL_RTX_BF16_CORE_FNS:
        setattr(_FakeFvk, fn, lambda *a, **k: None)

    class _FakeVlk:
        pass
    for fn in m._QWEN3_VL_RTX_BF16_VISION_FNS:
        if fn != target:
            setattr(_FakeVlk, fn, lambda *a, **k: None)

    import unittest.mock as mock
    with mock.patch.dict('sys.modules', {
        'flash_rt.flash_rt_kernels': _FakeFvk,
        'flash_rt.flash_rt_qwen3_vl_kernels': _FakeVlk,
    }):
        with pytest.raises(RuntimeError) as ei:
            m._require_qwen3_vl_rtx_bf16_kernels()
        assert target in str(ei.value)


# ── kernel completeness (gated: only runs when .so is built) ──

def test_kernel_modules_complete_when_built():
    """If both kernel modules are built for BF16 (SM87/Orin), they must
    expose every symbol the BF16 frontend calls. Skips when the modules
    aren't built or when the build targets a non-BF16 arch (SM120/SM89
    builds include the VL kernel module but omit the BF16-only GEMV)."""
    try:
        from flash_rt import flash_rt_kernels as fvk
        from flash_rt import flash_rt_qwen3_vl_kernels as vlk
    except ImportError:
        pytest.skip('flash_rt kernel modules not built')
    m = importlib.import_module('flash_rt.frontends.torch.qwen3_vl_rtx_bf16')
    if not hasattr(vlk, 'qwen3_vl_bf16_gemv_m1'):
        pytest.skip('BF16 GEMV kernel not present (non-SM87 build)')
    fvk_missing = [fn for fn in m._QWEN3_VL_RTX_BF16_CORE_FNS
                   if not hasattr(fvk, fn)]
    vlk_missing = [fn for fn in m._QWEN3_VL_RTX_BF16_VISION_FNS
                   if not hasattr(vlk, fn)]
    assert not fvk_missing, f'flash_rt_kernels missing: {fvk_missing}'
    assert not vlk_missing, f'flash_rt_qwen3_vl_kernels missing: {vlk_missing}'


# ── weight-loader invariants (gated on importability) ──

def test_bf16_weight_loader_exports_invariant_fn():
    """The BF16 weight loader module must export the assertion function."""
    m = importlib.import_module('flash_rt.frontends.torch._qwen3_vl_bf16_weights')
    assert hasattr(m, 'assert_extraction_invariants_qwen3_vl_bf16')
    assert callable(m.assert_extraction_invariants_qwen3_vl_bf16)


def test_bf16_weight_loader_exports_extract_fn():
    """The BF16 weight loader module must export the extraction function."""
    m = importlib.import_module('flash_rt.frontends.torch._qwen3_vl_bf16_weights')
    assert hasattr(m, 'extract_weights_qwen3_vl_bf16')
    assert callable(m.extract_weights_qwen3_vl_bf16)


# ── constructor signature ──

def test_constructor_signature_has_required_kwargs():
    """Verify the public constructor signature hasn't drifted."""
    import inspect
    m = importlib.import_module('flash_rt.frontends.torch.qwen3_vl_rtx_bf16')
    sig = inspect.signature(m.Qwen3VlTorchFrontendRtxBF16.__init__)
    params = set(sig.parameters.keys()) - {'self'}
    for required in ('checkpoint_path', 'device', 'max_seq'):
        assert required in params, f'missing parameter: {required}'

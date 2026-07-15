"""Smoke tests for MiniMax-Remover FlashRT integration.

These tests run in **any** build configuration:
  - default build (SM120 NVFP4 kernels absent): import succeeds,
    ``load_nvfp4_kernels`` / ``load_fp8_kernels`` raise ``RuntimeError``
    naming the missing symbols, and pipeline construction fails fast.
  - gated build (SM120 NVFP4 kernels present): every required symbol is
    present and callable.

Both the NVFP4 (``MiniMaxRemoverPipeline``) and FP8
(``MiniMaxRemoverPipelineFP8``) paths are covered.

No GPU, no model checkpoint, no MiniMax-Remover source tree is required.
"""
import sys
import types

import pytest


# ── helpers ──

def _stub_kernels(symbols=()):
    """Register an empty flash_rt_kernels stub exposing only ``symbols``."""
    fake_mod = types.ModuleType("flash_rt.flash_rt_kernels")
    for s in symbols:
        setattr(fake_mod, s, lambda *a, **k: None)
    sys.modules["flash_rt.flash_rt_kernels"] = fake_mod
    return fake_mod


def _restore_kernels():
    sys.modules.pop("flash_rt.flash_rt_kernels", None)


# ── 1. Package import always succeeds (no optional deps, no kernels) ──

def test_package_import():
    """Importing the model package must not require flash_rt_kernels."""
    from flash_rt.models.minimax_remover import (MiniMaxRemoverPipeline,
                                                 MiniMaxRemoverPipelineFP8)
    assert MiniMaxRemoverPipeline is not None
    assert MiniMaxRemoverPipelineFP8 is not None


def test_utils_module_import():
    """The _utils module owns the single kernel-surface source of truth."""
    from flash_rt.models.minimax_remover import _utils
    assert hasattr(_utils, "load_nvfp4_kernels")
    assert hasattr(_utils, "load_fp8_kernels")
    assert hasattr(_utils, "_load_kernels")
    # NVFP4 surface
    assert "nvfp4_sf_swizzled_bytes" in _utils._REQUIRED_NVFP4_SYMBOLS
    # FP8 surface must list every symbol the FP8 Linear actually calls
    # (quantize + gemm + bias-add), so a missing build fails fast.
    assert "quantize_fp8_static_fp16" in _utils._REQUIRED_FP8_SYMBOLS
    assert "fp8_gemm_descale_fp16" in _utils._REQUIRED_FP8_SYMBOLS
    assert "add_bias_fp16" in _utils._REQUIRED_FP8_SYMBOLS
    # Shared block-fusion surface: gelu_inplace(_fp16) is on the default hot
    # path of both pipelines (gelu_mode="inplace"), so it must be validated
    # alongside the precision surface to fail fast.
    assert hasattr(_utils, "_REQUIRED_BLOCK_SYMBOLS")
    assert "gelu_inplace" in _utils._REQUIRED_BLOCK_SYMBOLS
    assert "gelu_inplace_fp16" in _utils._REQUIRED_BLOCK_SYMBOLS


def test_pipeline_reexports_kernel_surface():
    """pipeline.py re-exports _load_kernels/_REQUIRED_NVFP4_SYMBOLS for back-compat."""
    from flash_rt.models.minimax_remover import pipeline
    from flash_rt.models.minimax_remover import _utils
    assert pipeline._REQUIRED_NVFP4_SYMBOLS is _utils._REQUIRED_NVFP4_SYMBOLS
    assert pipeline._load_kernels is _utils._load_kernels


def test_attention_forward_fa2_does_not_import_sageattention(monkeypatch):
    """The documented fa2 fallback must not require sageattention."""
    import sys
    import types

    import torch

    import flash_rt
    from flash_rt.models.minimax_remover import _attention

    calls = []
    fake_fa2 = types.SimpleNamespace(
        fwd_fp16=lambda *args: calls.append(args))
    monkeypatch.setattr(flash_rt, "flash_rt_fa2", fake_fa2, raising=False)
    monkeypatch.setitem(sys.modules, "flash_rt.flash_rt_fa2", fake_fa2)
    monkeypatch.setattr(
        _attention, "_get_sage",
        lambda: pytest.fail("fa2 mode must not import sageattention"))

    class _FakeStream:
        cuda_stream = 0

    monkeypatch.setattr(torch.cuda, "current_stream",
                        lambda: _FakeStream(), raising=False)

    q = torch.empty(1, 2, 1, 4, dtype=torch.float16)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    out = _attention.attention_forward(q, k, v, 0.5, "fa2")

    assert out.shape == q.shape
    assert calls


def test_manual_fused_block_uses_shared_attention_forward():
    """The manual fused block must respect FLASHRT_ATTN_MODE."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    src = (root / "flash_rt/models/minimax_remover/_manual_denoise.py").read_text()

    assert "from ._attention import attention_forward" in src
    assert "_sage_attn" not in src
    assert "attention_forward(q, k, v, scale, _attention_mode())" in src


def test_runtime_optional_dependencies_are_lazy_imported():
    """Package import must not require diffusers/einops."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for rel in (
        "flash_rt/models/minimax_remover/_fp8_pipeline.py",
        "flash_rt/models/minimax_remover/_fp8_manual_denoise.py",
        "flash_rt/models/minimax_remover/_manual_denoise.py",
    ):
        src = (root / rel).read_text()
        assert "from diffusers" not in "\n".join(
            line for line in src.splitlines()[:80])
        assert "from einops" not in "\n".join(
            line for line in src.splitlines()[:80])
    fp8_src = (root / "flash_rt/models/minimax_remover/_fp8_pipeline.py").read_text()
    top_level = fp8_src.split("class MiniMaxRemoverPipelineFP8:", 1)[0]
    assert "_fp8_manual_denoise import FP8ManualDenoise" not in top_level


def test_minimax_remover_cmake_requires_blackwell_nvfp4():
    """The standalone MiniMax module contains SM120 FP8/NVFP4 kernels."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    cmake = (root / "CMakeLists.txt").read_text()
    start = cmake.index("if(FLASHRT_ENABLE_MINIMAX_REMOVER AND NOT ENABLE_NVFP4)")
    end = cmake.index("endif()", start)
    block = cmake[start:end]
    assert "FLASHRT_ENABLE_MINIMAX_REMOVER requires Blackwell NVFP4" in block


# ── 2. load_*_kernels validate the kernel surface ──

def test_load_nvfp4_kernels_raises_when_symbols_absent():
    """Without the NVFP4 kernels, load_nvfp4_kernels raises a clear RuntimeError."""
    from flash_rt.models.minimax_remover import _utils
    _stub_kernels(symbols=())  # none of the required symbols
    try:
        with pytest.raises(RuntimeError) as excinfo:
            _utils.load_nvfp4_kernels()
        msg = str(excinfo.value)
        assert "NVFP4" in msg
        assert "nvfp4_sf_swizzled_bytes" in msg
        assert "bf16_weight_to_nvfp4_swizzled" in msg
    finally:
        _restore_kernels()


def test_load_nvfp4_kernels_succeeds_when_symbols_present():
    """With all required NVFP4 symbols, load_nvfp4_kernels returns the module."""
    from flash_rt.models.minimax_remover import _utils
    fake_mod = _stub_kernels(symbols=_utils._REQUIRED_NVFP4_SYMBOLS + _utils._REQUIRED_BLOCK_SYMBOLS)
    try:
        assert _utils.load_nvfp4_kernels() is fake_mod
    finally:
        _restore_kernels()


def test_load_fp8_kernels_raises_when_symbols_absent():
    """Without the FP8 kernels, load_fp8_kernels raises a clear RuntimeError."""
    from flash_rt.models.minimax_remover import _utils
    _stub_kernels(symbols=())  # none of the required symbols
    try:
        with pytest.raises(RuntimeError) as excinfo:
            _utils.load_fp8_kernels()
        msg = str(excinfo.value)
        # Every required FP8 symbol is named in the error.
        for s in _utils._REQUIRED_FP8_SYMBOLS:
            assert s in msg
    finally:
        _restore_kernels()


def test_load_fp8_kernels_raises_when_bias_symbol_missing():
    """A build that lacks add_bias_fp16 must fail fast (regression guard).

    Every other required symbol (FP8 precision + shared block) is present,
    so the only missing symbol is add_bias_fp16.
    """
    from flash_rt.models.minimax_remover import _utils
    full = _utils._REQUIRED_FP8_SYMBOLS + _utils._REQUIRED_BLOCK_SYMBOLS
    partial = tuple(s for s in full if s != "add_bias_fp16")
    _stub_kernels(symbols=partial)
    try:
        with pytest.raises(RuntimeError) as excinfo:
            _utils.load_fp8_kernels()
        assert "add_bias_fp16" in str(excinfo.value)
    finally:
        _restore_kernels()


def test_load_fp8_kernels_succeeds_when_symbols_present():
    """With all required FP8 symbols, load_fp8_kernels returns the module."""
    from flash_rt.models.minimax_remover import _utils
    fake_mod = _stub_kernels(symbols=_utils._REQUIRED_FP8_SYMBOLS + _utils._REQUIRED_BLOCK_SYMBOLS)
    try:
        assert _utils.load_fp8_kernels() is fake_mod
    finally:
        _restore_kernels()


# ── 3. Pipeline construction validates kernel availability (fail fast) ──

class _FakePipe:
    """Minimal stub matching the diffusers pipeline contract.

    Construction must fail at kernel validation before any pipe attribute is
    touched, so the stub is never actually read.
    """


def test_nvfp4_pipeline_constructor_validates_kernels(monkeypatch):
    """NVFP4 pipeline construction must fail before touching model internals."""
    from flash_rt.models.minimax_remover import pipeline

    def _raise_missing():
        raise RuntimeError(
            "MiniMax-Remover requires the SM120 NVFP4 kernels which are not "
            "compiled into flash_rt_kernels. Rebuild with the Blackwell NVFP4 "
            "build option enabled.")

    # The constructor calls load_nvfp4_kernels (imported from _utils).
    monkeypatch.setattr(pipeline, "load_nvfp4_kernels", _raise_missing)
    with pytest.raises(RuntimeError, match="NVFP4"):
        pipeline.MiniMaxRemoverPipeline(_FakePipe())


def test_nvfp4_pipeline_constructor_calls_load_kernels(monkeypatch):
    """load_nvfp4_kernels is invoked exactly once during NVFP4 construction."""
    from flash_rt.models.minimax_remover import pipeline

    calls = []

    def _fake_load():
        calls.append(1)
        raise RuntimeError("stop construction here")

    monkeypatch.setattr(pipeline, "load_nvfp4_kernels", _fake_load)
    with pytest.raises(RuntimeError, match="stop construction"):
        pipeline.MiniMaxRemoverPipeline(_FakePipe())
    assert len(calls) == 1


def test_fp8_pipeline_constructor_validates_kernels(monkeypatch):
    """FP8 pipeline construction must fail before touching model internals."""
    from flash_rt.models.minimax_remover import _fp8_pipeline

    def _raise_missing():
        raise RuntimeError(
            "MiniMax-Remover FP8 requires flash_rt_kernels with the FP8 "
            "symbols (quantize_fp8_static_fp16 / fp8_gemm_descale_fp16 / "
            "add_bias_fp16). Rebuild flash_rt_kernels.")

    # The constructor calls load_fp8_kernels (imported from _utils).
    monkeypatch.setattr(_fp8_pipeline, "load_fp8_kernels", _raise_missing)
    with pytest.raises(RuntimeError, match="FP8"):
        _fp8_pipeline.MiniMaxRemoverPipelineFP8(_FakePipe())


def test_fp8_pipeline_constructor_calls_load_kernels(monkeypatch):
    """load_fp8_kernels is invoked exactly once during FP8 construction."""
    from flash_rt.models.minimax_remover import _fp8_pipeline

    calls = []

    def _fake_load():
        calls.append(1)
        raise RuntimeError("stop fp8 construction here")

    monkeypatch.setattr(_fp8_pipeline, "load_fp8_kernels", _fake_load)
    with pytest.raises(RuntimeError, match="stop fp8 construction"):
        _fp8_pipeline.MiniMaxRemoverPipelineFP8(_FakePipe())
    assert len(calls) == 1


def test_fp8_pipeline_call_does_not_patch_pipe_class(monkeypatch):
    """Wrapping one FP8 pipe must not alter all instances of that pipe class."""
    from flash_rt.models.minimax_remover import _fp8_pipeline

    # Exercise the delegation path (orig pipe __call__) rather than the
    # eager-manual denoise default, so the stub does not need a real
    # transformer/scheduler. The class-isolation guarantee under test is
    # independent of the steady-state dispatch mode.
    monkeypatch.setenv("FLASHRT_FP8_EAGER_MANUAL", "0")

    class _Param:
        dtype = "fp16"

    class _Transformer:
        def __init__(self):
            self.config = types.SimpleNamespace(eps=1e-6)
            self._hooks = []

        def to(self, _dtype):
            return self

        def parameters(self):
            return iter([_Param()])

        def register_forward_hook(self, fn):
            self._hooks.append(fn)

            class _Handle:
                def __init__(self, hooks, f):
                    self._hooks = hooks
                    self._f = f

                def remove(self):
                    if self._f in self._hooks:
                        self._hooks.remove(self._f)

            return _Handle(self._hooks, fn)

        def _fire_hooks(self):
            for fn in list(self._hooks):
                fn(self, None, None)

    class _Vae:
        def parameters(self):
            return iter([_Param()])

    class _CallablePipe:
        def __init__(self, name):
            self.name = name
            self.transformer = _Transformer()
            self.vae = _Vae()
            self.calls = []

        def __call__(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            # Simulate the transformer forward so the one-shot calibration
            # freeze hook fires during the wrapped pipe's first call.
            self.transformer._fire_hooks()
            return self.name, args, kwargs

    set_calibration_calls = []
    freeze_calls = []

    def _fake_runtime():
        def install_flashrt_fp8(_transformer, verbose=True, target="all"):
            return 0

        def set_calibration(_transformer, on):
            set_calibration_calls.append(on)

        def freeze_calibration(_transformer, margin=1.1):
            freeze_calls.append(margin)
            return 3

        def install_fused_blocks(_transformer):
            return 0

        def install_fa2_attention(_transformer):
            return 0

        return (install_flashrt_fp8, set_calibration, freeze_calibration,
                install_fused_blocks, install_fa2_attention)

    monkeypatch.setattr(_fp8_pipeline, "load_fp8_kernels", lambda: object())
    monkeypatch.setattr(_fp8_pipeline, "_import_runtime_fp8", _fake_runtime)

    pipe1 = _CallablePipe("pipe1")
    pipe2 = _CallablePipe("pipe2")
    original_call = _CallablePipe.__call__

    wrapped = _fp8_pipeline.MiniMaxRemoverPipelineFP8(pipe1)

    assert _CallablePipe.__call__ is original_call
    assert pipe2("unwrapped") == ("pipe2", ("unwrapped",), {})
    assert not set_calibration_calls
    assert not freeze_calls

    assert wrapped("wrapped", flag=True) == (
        "pipe1", ("wrapped",), {"flag": True})
    assert set_calibration_calls == [True]
    assert freeze_calls == [1.1]
    assert wrapped._calibrated

    assert wrapped("again") == ("pipe1", ("again",), {})
    assert set_calibration_calls == [True]
    assert freeze_calls == [1.1]

    assert wrapped._calibrated

    assert wrapped("again") == ("pipe1", ("again",), {})
    assert set_calibration_calls == [True]
    assert freeze_calls == [1.1]


# ── 4. Gated build: required symbols present and callable ──

def _get_kernels_or_skip():
    try:
        from flash_rt import flash_rt_kernels as fvk
    except ImportError:
        try:
            import flash_rt_kernels as fvk  # type: ignore
        except ImportError:
            pytest.skip("flash_rt_kernels not built")
    return fvk


def test_nvfp4_symbols_present_when_gated():
    """In a gated build, every required NVFP4 symbol is present & callable."""
    fvk = _get_kernels_or_skip()
    from flash_rt.models.minimax_remover._utils import _REQUIRED_NVFP4_SYMBOLS
    missing = [s for s in _REQUIRED_NVFP4_SYMBOLS if not hasattr(fvk, s)]
    if missing:
        pytest.skip(f"SM120 NVFP4 kernels not compiled (missing: {', '.join(missing)})")
    for sym in _REQUIRED_NVFP4_SYMBOLS:
        assert callable(getattr(fvk, sym)), f"{sym} is not callable"


def test_fp8_symbols_present_when_gated():
    """In a build with FP8 kernels, every required FP8 symbol is callable."""
    fvk = _get_kernels_or_skip()
    from flash_rt.models.minimax_remover._utils import _REQUIRED_FP8_SYMBOLS
    missing = [s for s in _REQUIRED_FP8_SYMBOLS if not hasattr(fvk, s)]
    if missing:
        pytest.skip(f"FP8 kernels not compiled (missing: {', '.join(missing)})")
    for sym in _REQUIRED_FP8_SYMBOLS:
        assert callable(getattr(fvk, sym)), f"{sym} is not callable"


def test_block_symbols_present_in_default_build():
    """The shared gelu block-fusion symbols ship in the default build and are callable.

    Both pipelines call gelu_inplace(_fp16) on the default hot path, so a
    default flash_rt_kernels build must expose them regardless of NVFP4 gating.
    """
    fvk = _get_kernels_or_skip()
    from flash_rt.models.minimax_remover._utils import _REQUIRED_BLOCK_SYMBOLS
    for sym in _REQUIRED_BLOCK_SYMBOLS:
        assert callable(getattr(fvk, sym)), f"{sym} is not callable"


def test_nvfp4_symbols_absent_in_default_build():
    """In a default (non-NVFP4) build, load_nvfp4_kernels documents the gap.

    This documents the 'compile option OFF' case end-to-end: if any required
    NVFP4 symbol is missing the pipeline refuses to construct.
    """
    fvk = _get_kernels_or_skip()
    from flash_rt.models.minimax_remover import _utils

    missing = [s for s in _utils._REQUIRED_NVFP4_SYMBOLS if not hasattr(fvk, s)]
    if not missing:
        pytest.skip("this build has the NVFP4 kernels (gated build) — covered elsewhere")

    # Stub the kernels module exposing only what this build actually has, then
    # verify load_nvfp4_kernels raises and names every missing symbol.
    _stub_kernels(symbols=[s for s in _utils._REQUIRED_NVFP4_SYMBOLS if hasattr(fvk, s)])
    try:
        with pytest.raises(RuntimeError) as excinfo:
            _utils.load_nvfp4_kernels()
        for s in missing:
            assert s in str(excinfo.value)
    finally:
        _restore_kernels()

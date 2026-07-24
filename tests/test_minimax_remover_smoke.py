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

    # use_universal_scale=False keeps this class-isolation test off the
    # cross-resolution scale-cache path, which needs a realistic dict-like
    # transformer.config and would write to ~/.flash_rt/calibration/.
    wrapped = _fp8_pipeline.MiniMaxRemoverPipelineFP8(
        pipe1, use_universal_scale=False)

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


# ── 5. TeaCache step-caching plumbing (skip_steps) ──

def test_fp8_pipeline_call_forwards_skip_steps_to_pipe(monkeypatch):
    """skip_steps reaches the wrapped pipe's __call__ on the calibration call.

    The single-call quickstart runs the first (calibration) call through the
    diffusers reference ``__call__``, so the TeaCache schedule must be
    forwarded there for it to take effect without a warm-up pass.
    """
    import torch

    from flash_rt.models.minimax_remover import _fp8_pipeline

    # Exercise the delegation path (orig pipe __call__) rather than the
    # eager-manual denoise default, so the stub does not need a real
    # transformer/scheduler.
    monkeypatch.setenv("FLASHRT_FP8_EAGER_MANUAL", "0")
    monkeypatch.setattr(_fp8_pipeline, "load_fp8_kernels", lambda: object())

    def _fake_runtime():
        def install_flashrt_fp8(_t, verbose=True, target="all"):
            return 0

        def set_calibration(_t, on):
            return None

        def freeze_calibration(_t, margin=1.1):
            return 3

        def install_fused_blocks(_t):
            return 0

        def install_fa2_attention(_t):
            return 0

        return (install_flashrt_fp8, set_calibration, freeze_calibration,
                install_fused_blocks, install_fa2_attention)

    monkeypatch.setattr(_fp8_pipeline, "_import_runtime_fp8", _fake_runtime)

    class _Param:
        dtype = torch.float16

    class _Transformer:
        def __init__(self):
            self.config = types.SimpleNamespace(eps=1e-6)
            self._hooks = []

        def to(self, _d):
            return self

        def parameters(self):
            return iter([_Param()])

        def register_forward_hook(self, fn):
            self._hooks.append(fn)

            class _Handle:
                def remove(_self):
                    pass

            return _Handle()

        def _fire_hooks(self):
            for fn in list(self._hooks):
                fn(self, None, None)

    class _Vae:
        def parameters(self):
            return iter([_Param()])

    received = []

    class _Pipe:
        def __init__(self):
            self.transformer = _Transformer()
            self.vae = _Vae()

        def __call__(self, *args, **kwargs):
            received.append(dict(kwargs))
            # Fire the one-shot calibration freeze hook on the first call.
            self.transformer._fire_hooks()
            return ("ok", args, kwargs)

    # use_universal_scale=False avoids the disk-backed scale cache so this
    # skip_steps-forwarding test stays hermetic (no ~/.flash_rt writes).
    wrapped = _fp8_pipeline.MiniMaxRemoverPipelineFP8(
        _Pipe(), use_universal_scale=False)
    out = wrapped(num_frames=5, skip_steps=[3, 5, 7, 9])

    assert out[0] == "ok"
    assert received, "wrapped pipe __call__ was never invoked"
    assert received[0].get("skip_steps") == [3, 5, 7, 9], (
        "skip_steps must be forwarded to the reference __call__ on the "
        "calibration (first) call")


def test_fp8_pipeline_does_not_forward_unsupported_skip_steps(caplog):
    """A standard pipeline without ``skip_steps`` still calibrates safely."""
    import logging

    from flash_rt.models.minimax_remover import _fp8_pipeline

    class _Handle:
        def remove(self):
            pass

    class _Transformer:
        def register_forward_hook(self, hook):
            self.hook = hook
            return _Handle()

    class _Pipe:
        def __call__(self, *, num_frames):
            return num_frames

    pipe = _Pipe()
    wrapped = object.__new__(_fp8_pipeline.MiniMaxRemoverPipelineFP8)
    wrapped._calibrated = False
    wrapped._scale_dirty = False
    wrapped._pipe_accepts_skip_steps = False
    wrapped._warned_skip_steps_unsupported = False
    wrapped._set_calibration = lambda _on: None
    wrapped._freeze_calibration = lambda: 0
    wrapped.transformer = _Transformer()
    wrapped._orig_pipe_call = pipe.__call__
    wrapped.calib_margin = 1.1

    with caplog.at_level(logging.WARNING):
        assert wrapped(num_frames=5, skip_steps=[3, 5]) == 5

    assert "does not accept skip_steps" in caplog.text


def test_fp8_manual_denoise_supports_skip_steps():
    """The FP8 manual denoise carries skip_steps through every entry point."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    src = (root / "flash_rt/models/minimax_remover/_fp8_manual_denoise.py").read_text()

    # Zeroth-order TeaCache reuse logic (skip step reuses cached noise_pred).
    assert "cached_noise_pred" in src
    assert "if step in skip_set and cached_noise_pred is not None:" in src
    # The public denoise() / _denoise_loop_body() / _capture_graph() all
    # carry the parameter.
    assert src.count("skip_steps=None") >= 3
    # The captured-graph cache key is per skip-schedule, because the skip
    # set is baked into the graph at capture time (like Motus).
    assert "skip_key" in src
    assert "skip_steps=skip_steps" in src


def test_fp8_pipeline_threads_skip_steps_to_manual_call():
    """_manual_call carries skip_steps and forwards it to FP8ManualDenoise."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    src = (root / "flash_rt/models/minimax_remover/_fp8_pipeline.py").read_text()

    # __call__ pops skip_steps out of the public kwargs...
    assert 'skip_steps = kwargs.pop("skip_steps", None)' in src
    # ...forwards it when the wrapped pipeline explicitly supports it...
    assert "self._pipe_accepts_skip_steps" in src
    assert 'fwd_kwargs["skip_steps"] = skip_steps' in src
    # ...and threads it into _manual_call on the steady-state branches.
    assert "skip_steps=skip_steps, **kwargs" in src
    # _manual_call signature carries the parameter...
    assert "skip_steps=None):" in src
    # ...and forwards it to the FP8ManualDenoise.denoise().
    assert "use_graph=use_graph, skip_steps=skip_steps" in src


def test_quickstart_approximate_paths_are_opt_in():
    """The quickstart preserves the original denoise and scale behavior."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    src = (root / "examples/minimax_remover_quickstart.py").read_text()

    # Both approximate optimizations require explicit CLI flags.
    assert 'TEACACHE_SKIP_DEFAULT = ""' in src
    assert "default=TEACACHE_SKIP_DEFAULT" in src
    universal_flag = src.index('p.add_argument("--universal-scale"')
    assert "default=False" in src[universal_flag:universal_flag + 500]
    # The reference __call__ carries the parameter...
    assert "skip_steps: Optional[List[int]] = None" in src
    # ...with zeroth-order reuse in the denoise loop.
    assert "cached_noise_pred" in src
    assert "if i in skip_set and cached_noise_pred is not None:" in src
    # --no-flashrt is the master "pure reference" switch: it disables ALL
    # FlashRT optimisations (no need to also pass --no-vae-opt).
    assert "vae_opt = args.vae_opt and not args.no_flashrt" in src
    # TeaCache is never applied on the reference path (ground truth = full
    # N-step denoise for PSNR/timing A/B) nor on the NVFP4 transformer
    # path (its wrapper does not accept skip_steps).
    assert ("if args.teacache_skip.strip() and not args.no_flashrt "
            "and not args.nvfp4_transformer:") in src


def test_quickstart_checks_cv2_write_result():
    """A failed OpenCV write must not be reported as a saved frame."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    src = (root / "examples/minimax_remover_quickstart.py").read_text()

    assert "saved_ok = cv2.imwrite(" in src
    assert "if not saved_ok:" in src
    assert 'raise OSError(f"failed to write output frame: {dst}")' in src


def test_sm89_attention_fallback_matches_only_known_error():
    """Unrelated runtime errors must not be swallowed by the fallback."""
    from flash_rt.models.minimax_remover._attention import (
        _is_sm89_query_scale_shape_error,
    )

    assert _is_sm89_query_scale_shape_error(
        RuntimeError("query_scale must have shape (1, 32)")
    )
    assert not _is_sm89_query_scale_shape_error(
        RuntimeError("output must have shape (1, 32)")
    )
    assert not _is_sm89_query_scale_shape_error(
        RuntimeError("query_scale is on the wrong device")
    )


def _make_scale_cache_wrapper(tmp_path, checkpoint_digest="a" * 64):
    import torch

    from flash_rt.models.minimax_remover import _fp8_pipeline

    class _Layer:
        in_features = 8
        out_features = 16

        def __init__(self):
            self.weight_fp8 = torch.zeros(1)
            self.act_amax_max = torch.tensor([224.0])
            self.act_scale = torch.tensor([99.0])
            self.calibrating = True

    layer = _Layer()
    wrapped = object.__new__(_fp8_pipeline.MiniMaxRemoverPipelineFP8)
    wrapped.transformer = types.SimpleNamespace(config={"layers": 1})
    wrapped.fp8_target = "all"
    wrapped._compute_dtype_name = "float16"
    wrapped._checkpoint_digest = checkpoint_digest
    wrapped._scale_cache_dir = tmp_path
    wrapped.universal_margin = 1.3
    wrapped._scale_layers = lambda: [("block.proj", layer)]
    return wrapped, layer


def test_universal_scale_cache_is_checkpoint_bound(tmp_path):
    """Changing checkpoint bytes changes both the digest and cache key."""
    from flash_rt.models.minimax_remover import _fp8_pipeline

    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"first weights")
    first = _fp8_pipeline._checkpoint_weights_digest(checkpoint)
    checkpoint.write_bytes(b"second weights")
    second = _fp8_pipeline._checkpoint_weights_digest(checkpoint)

    assert first != second
    first_wrapper, _ = _make_scale_cache_wrapper(tmp_path / "cache", first)
    second_wrapper, _ = _make_scale_cache_wrapper(tmp_path / "cache", second)
    assert first_wrapper._model_fingerprint() != second_wrapper._model_fingerprint()
    assert first_wrapper._scale_cache_path() != second_wrapper._scale_cache_path()


def test_universal_scale_cache_round_trip_and_validation(tmp_path):
    """Only a complete, finite, schema-matching cache may mutate scales."""
    import json

    wrapped, layer = _make_scale_cache_wrapper(tmp_path)
    assert wrapped._dump_universal_scales()
    cache = wrapped._scale_cache_path()
    payload = json.loads(cache.read_text())
    assert payload["schema_version"] == 2
    assert payload["checkpoint_digest"] == "a" * 64
    assert payload["fp8_target"] == "all"
    assert payload["compute_dtype"] == "float16"

    layer.act_scale.fill_(99.0)
    assert wrapped._load_universal_scales()
    assert layer.act_scale.item() == pytest.approx(224.0 * 1.3 / 448.0)
    assert not layer.calibrating

    for invalid in (
        [],
        {**payload, "schema_version": 1},
        {
            **payload,
            "layers": [{**payload["layers"][0], "amax_max": float("nan")}],
        },
    ):
        cache.write_text(json.dumps(invalid))
        layer.act_scale.fill_(77.0)
        assert not wrapped._load_universal_scales()
        assert layer.act_scale.item() == 77.0


def test_universal_scale_cache_write_failure_is_nonfatal(
        tmp_path, monkeypatch, caplog):
    """Read-only or failed cache storage must not interrupt inference."""
    import logging

    from flash_rt.models.minimax_remover import _fp8_pipeline

    wrapped, _ = _make_scale_cache_wrapper(tmp_path)

    def _fail_replace(_source, _destination):
        raise OSError("read-only cache")

    monkeypatch.setattr(_fp8_pipeline.os, "replace", _fail_replace)
    with caplog.at_level(logging.WARNING):
        assert not wrapped._dump_universal_scales()
    assert "inference will continue" in caplog.text
    assert not list(tmp_path.glob("*.tmp"))


def test_universal_scale_cache_read_failure_is_nonfatal(caplog):
    """An inaccessible cache falls back without touching existing scales."""
    import logging

    wrapped, layer = _make_scale_cache_wrapper(None)

    class _UnreadableCache:
        @staticmethod
        def is_file():
            return True

        @staticmethod
        def read_text(**_kwargs):
            raise OSError("cache is not readable")

    wrapped._scale_cache_path = lambda: _UnreadableCache()
    with caplog.at_level(logging.WARNING):
        assert not wrapped._load_universal_scales()
    assert layer.act_scale.item() == 99.0
    assert "cache ignored" in caplog.text


def test_fp8_pipeline_universal_scale_default_is_off():
    """Constructing the wrapper must retain per-resolution calibration."""
    import inspect

    from flash_rt.models.minimax_remover import _fp8_pipeline

    parameter = inspect.signature(
        _fp8_pipeline.MiniMaxRemoverPipelineFP8
    ).parameters["use_universal_scale"]
    assert parameter.default is False


# ── 6. --nvfp4-transformer naming (replaces the confusing --use-fp4) ──

def test_quickstart_nvfp4_transformer_flag_naming():
    """The transformer-NVFP4 flag is named --nvfp4-transformer, not --use-fp4.

    The old ``--use-fp4`` name was misleading because the default path
    *already* uses NVFP4 for the VAE. The renamed flag makes explicit that
    it switches the **transformer** (the 12-step iterative denoise, where
    FP4 error accumulates) — distinct from the always-on NVFP4 VAE.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    src = (root / "examples/minimax_remover_quickstart.py").read_text()

    # New flag is present...
    assert '"--nvfp4-transformer"' in src
    assert "args.nvfp4_transformer" in src
    # ...and the old confusing name is gone from the quickstart.
    assert '"--use-fp4"' not in src
    assert "args.use_fp4" not in src
    # The NVFP4 VAE install is gated on the new attribute.
    assert "not args.nvfp4_transformer and not args.no_nvfp4_vae" in src


def test_quickstart_nvfp4_transformer_excludes_nvfp4_vae():
    """--nvfp4-transformer disables the NVFP4 VAE (the two NVFP4 paths are
    mutually exclusive: VAE NVFP4 is validated only on the FP8 transformer
    path, and the NVFP4 transformer path is a standalone small-region
    experiment)."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    src = (root / "examples/minimax_remover_quickstart.py").read_text()

    # nvfp4_vae is False when nvfp4_transformer is set.
    assert "nvfp4_vae = (vae_opt and not args.nvfp4_transformer" in src


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

"""LTX-2.5 runtime contracts: import, fallback, alignment, and residency.

These cover the parts that are easy to get wrong without a checkpoint on
disk: that the model package imports with none of its optional pieces
present, that a broken extension is not mistaken for an absent one, that
the FFN swap declines the shapes CUTLASS declines, and that the residency
lease can be ended twice and taken again.

The heavy paths (a real pipeline, real kernels, a device) are not
reachable here and are covered by the model's own benchmark runs.
"""

import sys
import types

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402


# --------------------------------------------------------------------
# import smoke: the package must import with no LTX install, no kernels
# --------------------------------------------------------------------

def test_model_package_import_is_clean_or_loudly_broken():
    """The upstream LTX packages are never required to import this one.

    The extension is a different case and the distinction is the point: it
    being *absent* must not stop the import (the swaps fall back), while it
    being present and unloadable must say so. So this asserts the
    distinction rather than an outcome, and steps aside on a host whose
    own build is broken -- there the loud failure is the correct result.
    """
    try:
        from flash_rt.models.ltx25 import _attn_swap, _nvfp4_ffn_swap
    except ImportError as exc:
        assert getattr(exc, "name", None) != "flash_rt.flash_rt_kernels", (
            "an absent extension must be a fallback, not an import failure")
        pytest.skip(f"extension present but not loadable here: {exc}")
    assert isinstance(_attn_swap.fvk_sage2_available(), bool)
    assert isinstance(_nvfp4_ffn_swap.fvk_ffn_available(), bool)


def test_frontend_module_imports_and_declares_its_surface():
    from flash_rt.frontends.torch.ltx25_rtx import Ltx25TorchFrontendRtx

    for name in ("set_prompt", "infer", "release_resident", "close",
                 "get_latency_stats"):
        assert callable(getattr(Ltx25TorchFrontendRtx, name)), name


def test_config_is_registered_on_the_public_api():
    import inspect

    from flash_rt import api

    source = inspect.getsource(api)
    assert '"ltx25"' in source


# --------------------------------------------------------------------
# optional import: absent is a fallback, broken is a bug
# --------------------------------------------------------------------

@pytest.mark.parametrize("module_name", [
    "flash_rt.models.ltx25._attn_swap",
    "flash_rt.models.ltx25._nvfp4_ffn_swap",
])
def test_broken_extension_is_not_swallowed_as_absent(module_name, monkeypatch):
    """An extension that fails to load must not read as 'not built'.

    The swaps treat the extension's own absence as a fallback. Every other
    load failure -- an undefined symbol, an ABI mismatch, a transitive
    import error -- has to propagate, or a broken build silently runs the
    slow path and reports nothing.
    """
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "flash_rt" and args and "flash_rt_kernels" in (args[2] or ()):
            raise ImportError("undefined symbol: _Z9brokenABIv")
        return real_import(name, *args, **kwargs)

    for name in [module_name, "flash_rt.models.ltx25"]:
        sys.modules.pop(name, None)
    monkeypatch.setattr("builtins.__import__", fake_import)
    with pytest.raises(ImportError, match="undefined symbol"):
        real_import(module_name, {}, {}, ["*"])


@pytest.mark.parametrize("module_name", [
    "flash_rt.models.ltx25._attn_swap",
    "flash_rt.models.ltx25._nvfp4_ffn_swap",
])
def test_absent_extension_is_a_fallback(module_name, monkeypatch):
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "flash_rt" and args and "flash_rt_kernels" in (args[2] or ()):
            raise ModuleNotFoundError(
                "No module named 'flash_rt.flash_rt_kernels'",
                name="flash_rt.flash_rt_kernels")
        return real_import(name, *args, **kwargs)

    for name in [module_name, "flash_rt.models.ltx25"]:
        sys.modules.pop(name, None)
    monkeypatch.setattr("builtins.__import__", fake_import)
    module = real_import(module_name, {}, {}, ["*"])
    assert module.fvk is None


# --------------------------------------------------------------------
# attention selection
# --------------------------------------------------------------------

def test_attention_selection_refuses_unknown_kinds():
    from flash_rt.models.ltx25._attn_swap import make_ltx25_attention

    with pytest.raises(ValueError):
        make_ltx25_attention("no_such_backend")


def test_sdpa_selection_never_depends_on_the_extension():
    """The baseline backend must be reachable on a host with no kernels."""
    from flash_rt.models.ltx25._attn_swap import make_ltx25_attention

    attn = make_ltx25_attention("sdpa")
    assert attn is None or getattr(attn, "label", "") == "sdpa"


def test_explicit_backend_fails_fast_when_unavailable(monkeypatch):
    """``sage2`` asked for by name must not silently become SDPA."""
    from flash_rt.models.ltx25 import _attn_swap

    monkeypatch.setattr(_attn_swap, "fvk_sage2_available", lambda: False)
    with pytest.raises(RuntimeError, match="sage2"):
        _attn_swap.make_ltx25_attention("sage2-fvk")


# --------------------------------------------------------------------
# FFN alignment: the swap declines exactly what the kernel declines
# --------------------------------------------------------------------

@pytest.mark.parametrize("rows,swapped", [
    (128, True), (256, True), (2688, True), (24576, True),
    (1, False), (126, False), (127, False), (129, False),
])
def test_ffn_swap_routes_by_row_alignment(rows, swapped):
    """M % 128 decides the arm, because that is what can_implement decides.

    The CUTLASS chain reports a validation failure for unaligned M without
    writing an output, so an unaligned call that reached it would produce
    silent garbage. The predicate is checked directly here: it holds with
    or without a device.
    """
    from flash_rt.models.ltx25._nvfp4_ffn_swap import rows_are_swappable

    assert rows_are_swappable(rows) is swapped


# --------------------------------------------------------------------
# residency lease
# --------------------------------------------------------------------

class _FakeModel:
    def __init__(self):
        self.disposed = 0

    def dispose(self):
        self.disposed += 1


class _FakeBuilder:
    """Inner builder standing in for the upstream stage builder."""

    def __init__(self):
        self.builds = 0

    def build(self, **kwargs):
        self.builds += 1
        return _FakeModel()


def _resident_builder(monkeypatch):
    from flash_rt.models.ltx25 import _resident_graph

    # the X0Model patch reaches into the upstream package; the lease
    # semantics under test do not need it
    monkeypatch.setattr(_resident_graph, "_patch_x0_dispose", lambda: None)
    return _resident_graph.ResidentSwapBuilder(_FakeBuilder(), [])


def test_residency_is_taken_once_and_reused(monkeypatch):
    builder = _resident_builder(monkeypatch)
    first = builder.build()
    assert builder.build() is first
    assert builder._inner.builds == 1
    assert builder.is_resident
    assert builder.keeps_gpu_resident_weights


def test_release_is_idempotent_and_disposes_once(monkeypatch):
    builder = _resident_builder(monkeypatch)
    model = builder.build()
    builder.release()
    assert not builder.is_resident
    assert model.disposed == 1
    builder.release()
    builder.release()
    assert model.disposed == 1, "a second release must not re-dispose"


def test_a_released_lease_can_be_taken_again(monkeypatch):
    builder = _resident_builder(monkeypatch)
    first = builder.build()
    builder.release()
    second = builder.build()
    assert second is not first
    assert builder.is_resident
    assert builder._inner.builds == 2


def test_rewrapped_builders_share_one_lease(monkeypatch):
    """The stage rewraps its builder per call; the lease must not fork."""
    builder = _resident_builder(monkeypatch)
    model = builder.build()
    rewrapped = builder._rewrap(builder._inner)
    assert rewrapped.build() is model
    rewrapped.release()
    assert not builder.is_resident


# --------------------------------------------------------------------
# prompt cache: a hit keeps the lease, a miss ends it
# --------------------------------------------------------------------

def test_cached_prompt_does_not_disturb_residency():
    from flash_rt.models.ltx25._resident_graph import CachingPromptEncoder

    calls, misses = [], []
    encoder = CachingPromptEncoder(
        lambda prompts, **kw: calls.append(prompts) or "embeds",
        on_miss=lambda: misses.append(1))

    assert encoder(["a fisherman"]) == "embeds"
    assert encoder(["a fisherman"]) == "embeds"
    assert len(calls) == 1, "a repeat prompt must not re-run the encoder"
    assert len(misses) == 1, "only the first encode ends the lease"


def test_new_prompt_ends_the_lease_before_encoding():
    """The release must happen *before* the encoder loads, not after.

    Ordering is the whole point: the encoder's weights and the resident
    transformer do not fit together, so a release that ran afterwards would
    free memory the encode had already failed to get.
    """
    from flash_rt.models.ltx25._resident_graph import CachingPromptEncoder

    order = []
    encoder = CachingPromptEncoder(
        lambda prompts, **kw: order.append("encode") or "embeds",
        on_miss=lambda: order.append("release"))

    encoder(["first"])
    encoder(["second"])
    assert order == ["release", "encode", "release", "encode"]


def test_prompt_cache_keys_on_encoder_arguments():
    from flash_rt.models.ltx25._resident_graph import CachingPromptEncoder

    calls = []
    encoder = CachingPromptEncoder(
        lambda prompts, **kw: calls.append(kw) or "embeds")

    encoder(["p"], enhance=False)
    encoder(["p"], enhance=True)
    assert len(calls) == 2, "different encoder arguments are different work"


def test_prompt_cache_clear_is_idempotent():
    from flash_rt.models.ltx25._resident_graph import CachingPromptEncoder

    calls = []
    encoder = CachingPromptEncoder(lambda prompts, **kw: calls.append(1))
    encoder(["p"])
    encoder.clear()
    encoder.clear()
    encoder(["p"])
    assert len(calls) == 2


def test_release_entries_are_safe_before_a_pipeline_exists():
    """``release_resident``/``close`` must not require a loaded pipeline."""
    from flash_rt.frontends.torch.ltx25_rtx import Ltx25TorchFrontendRtx

    frontend = Ltx25TorchFrontendRtx.__new__(Ltx25TorchFrontendRtx)
    frontend._pipe = None
    assert frontend.release_resident() == 0
    assert frontend.close() == 0
    assert frontend.close() == 0


def test_release_resident_is_a_no_op_outside_capture_mode():
    """Non-capture modes hold no lease, so there is nothing to release."""
    from flash_rt.frontends.torch.ltx25_rtx import Ltx25TorchFrontendRtx

    frontend = Ltx25TorchFrontendRtx.__new__(Ltx25TorchFrontendRtx)
    stage = types.SimpleNamespace(_transformer_builder=object())
    frontend._pipe = types.SimpleNamespace(stage=stage)
    assert frontend.release_resident() == 0

"""HyVLA Orin hardware-gate and isolation tests (no GPU needed)."""

import importlib

import pytest

torch = pytest.importorskip("torch")

try:
    import flash_rt.frontends.torch.hyvla_orin as orin_mod
except ModuleNotFoundError as exc:  # pragma: no cover
    if exc.name != "flash_rt.flash_rt_kernels":
        raise
    pytest.skip("flash_rt_kernels was not built", allow_module_level=True)


class _Probe:
    """Run _require_arch against mocked CUDA state."""

    _cls = orin_mod.HyVLATorchFrontendOrin

    def run(self):
        obj = object.__new__(self._cls)
        return self._cls._require_arch(obj)


def test_rejects_when_cuda_unavailable(monkeypatch):
    monkeypatch.delenv("FLASHRT_HYVLA_FORCE_ARCH", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        _Probe().run()


def test_rejects_wrong_capability(monkeypatch):
    monkeypatch.delenv("FLASHRT_HYVLA_FORCE_ARCH", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (11, 0))
    with pytest.raises(RuntimeError, match="requires Jetson Orin SM87"):
        _Probe().run()


def test_accepts_sm87(monkeypatch):
    monkeypatch.delenv("FLASHRT_HYVLA_FORCE_ARCH", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 7))
    _Probe().run()  # must not raise


def test_documented_env_override_skips_probe(monkeypatch):
    monkeypatch.setenv("FLASHRT_HYVLA_FORCE_ARCH", "1")
    # No CUDA mocking: the override must return before touching torch.cuda.
    _Probe().run()


def test_fp4_rejected_before_any_cuda_work():
    # SM87 has no FP4 tensor cores; the constructor must fail fast,
    # before checkpoint loading or CUDA allocation.
    with pytest.raises(RuntimeError, match="does not support FP4"):
        orin_mod.HyVLATorchFrontendOrin("/nonexistent/fake-ckpt", use_fp4=True)


def test_missing_prompt_raises_runtime_error():
    fe = orin_mod.HyVLATorchFrontendOrin.__new__(orin_mod.HyVLATorchFrontendOrin)
    # Minimal state to reach the prompt contract check only.
    fe._prompt = None
    fe._lang_tokens = None
    with pytest.raises(RuntimeError, match="set_prompt"):
        orin_mod.HyVLATorchFrontendOrin.predict_actions(fe, images=None)


def _bare_frontend():
    fe = orin_mod.HyVLATorchFrontendOrin.__new__(
        orin_mod.HyVLATorchFrontendOrin)
    fe.max_state_dim = 8
    fe.chunk = 4
    fe.max_action_dim = 3
    fe._prompt = None
    fe._lang_tokens = object()
    fe._vit_merge = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("invalid inputs reached ViT"))
    return fe


def test_invalid_state_rejected_before_gpu_or_cache_work():
    with pytest.raises(ValueError, match="state has 9 dims"):
        _bare_frontend().predict_actions(None, state=[0] * 9)


def test_invalid_noise_rejected_before_gpu_or_cache_work():
    with pytest.raises(ValueError, match="noise must have 12 elements"):
        _bare_frontend().predict_actions(None, noise=[0] * 11)


def test_old_torch_rms_norm_fallback_is_module_local(monkeypatch):
    import torch.nn.functional as functional
    import flash_rt.models.hyvla.pipeline_orin as pipeline_orin

    monkeypatch.delattr(functional, "rms_norm", raising=False)
    pipeline_orin = importlib.reload(pipeline_orin)

    assert not hasattr(functional, "rms_norm")
    x = torch.tensor([[3.0, 4.0]])
    expected = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-5)
    assert torch.allclose(pipeline_orin._rms_norm(x, (2,)), expected)


def test_attention_does_not_hide_nonfinite_output(monkeypatch):
    from flash_rt.models.hyvla.pipeline_orin import HyVLAOrinBF16Pipeline
    from flash_rt.models.hyvla.pipeline_thor import HyVLAThorBF16Pipeline

    expected = torch.tensor([float("nan")])
    monkeypatch.setattr(
        HyVLAThorBF16Pipeline, "_attn",
        lambda _self, _q, _k, _v, _mask: expected,
    )
    pipe = object.__new__(HyVLAOrinBF16Pipeline)
    result = pipe._attn(None, None, None, None)
    assert torch.isnan(result).all()

"""HyVLA hardware-gate (fail-fast) tests — torch.cuda is mocked, no GPU needed."""

import os

import pytest

torch = pytest.importorskip("torch")

try:
    import flash_rt.frontends.torch.hyvla_thor as hy_mod
except ModuleNotFoundError as exc:  # pragma: no cover
    if exc.name != "flash_rt.flash_rt_kernels":
        raise
    pytest.skip("flash_rt_kernels was not built", allow_module_level=True)


class _Probe:
    """Run _require_arch against mocked CUDA state."""

    def __init__(self, available, capability):
        self._cls = hy_mod.HyVLATorchFrontendThor
        self._available = available
        self._capability = capability

    def run(self):
        obj = object.__new__(self._cls)
        return self._cls._require_arch(obj)


def test_rejects_when_cuda_unavailable(monkeypatch):
    monkeypatch.delenv("FLASHRT_HYVLA_FORCE_ARCH", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        _Probe(False, None).run()


def test_rejects_wrong_capability(monkeypatch):
    monkeypatch.delenv("FLASHRT_HYVLA_FORCE_ARCH", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 9))
    with pytest.raises(RuntimeError, match="requires Jetson Thor SM110"):
        _Probe(True, (8, 9)).run()


def test_accepts_sm110(monkeypatch):
    monkeypatch.delenv("FLASHRT_HYVLA_FORCE_ARCH", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (11, 0))
    _Probe(True, (11, 0)).run()  # must not raise


def test_documented_env_override_skips_probe(monkeypatch):
    monkeypatch.setenv("FLASHRT_HYVLA_FORCE_ARCH", "1")
    # No CUDA mocking: the override must return before touching torch.cuda.
    _Probe(False, None).run()


@pytest.mark.parametrize("value", ["0", "false", "False", "yes"])
def test_other_env_values_do_not_skip_probe(monkeypatch, value):
    monkeypatch.setenv("FLASHRT_HYVLA_FORCE_ARCH", value)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        _Probe(False, None).run()


def _bare_frontend():
    obj = object.__new__(hy_mod.HyVLATorchFrontendThor)
    obj.max_state_dim = 8
    obj.chunk = 4
    obj.max_action_dim = 3
    obj._prompt = None
    obj._lang_tokens = object()
    obj._graph_cache = {}
    obj._vit_merge = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("invalid inputs reached ViT"))
    return obj


def test_invalid_state_rejected_before_gpu_or_graph_work():
    with pytest.raises(ValueError, match="state has 9 dims"):
        _bare_frontend().predict_actions(None, state=[0] * 9)


def test_invalid_noise_rejected_before_gpu_or_graph_work():
    with pytest.raises(ValueError, match="noise must have 12 elements"):
        _bare_frontend().predict_actions(None, noise=[0] * 11)

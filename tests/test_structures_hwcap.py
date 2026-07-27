"""The hardware line: the package's own arch declaration is the truth.

A Hub kernel package ships ``metadata.json`` naming the CUDA archs it
was built for; the shared loader reads that declaration and refuses a
device outside it with a message a person can act on — before the
kernel produces an unrelated-looking runtime error. The structures
layer keeps no arch table of its own: hardware support is maintained on
the kernels side, and absence of a declaration loads as before.
"""

import json
import types

import pytest

from flash_rt.structures import impls


def _fake_module(tmp_path, meta=None):
    pkg = tmp_path / "build"
    pkg.mkdir(exist_ok=True)
    if meta is not None:
        (pkg / "metadata.json").write_text(json.dumps(meta))
    mod = types.SimpleNamespace(__file__=str(pkg / "__init__.py"))
    return mod


def test_declared_archs_read_from_package_metadata(tmp_path):
    mod = _fake_module(tmp_path, {"backend": {"archs": ["12.0a", "12.1"]}})
    assert impls._declared_archs(mod) == ["12.0a", "12.1"]


def test_missing_metadata_means_no_declaration(tmp_path):
    assert impls._declared_archs(_fake_module(tmp_path)) is None
    mod = _fake_module(tmp_path, {"backend": {}})
    assert impls._declared_archs(mod) is None


def test_matching_device_passes_including_arch_specific_builds(
        tmp_path, monkeypatch):
    mod = _fake_module(tmp_path, {"backend": {"archs": ["12.0a", "12.1"]}})
    monkeypatch.setattr(impls, "_device_cc", lambda: (12, 0))
    impls._check_arch("flashrt/x", mod)          # "12.0a" serves sm 12.0
    monkeypatch.setattr(impls, "_device_cc", lambda: (12, 1))
    impls._check_arch("flashrt/x", mod)


def test_wrong_device_gets_a_clean_refusal(tmp_path, monkeypatch):
    mod = _fake_module(tmp_path, {"backend": {"archs": ["12.0a"]}})
    monkeypatch.setattr(impls, "_device_cc", lambda: (8, 9))
    with pytest.raises(ValueError, match="refused.*8.9"):
        impls._check_arch("flashrt/x", mod)


def test_no_cuda_device_defers_to_the_bind_path(tmp_path, monkeypatch):
    # without a device the check has nothing truthful to say; binding
    # refuses later at weight transfer, with its own reason
    mod = _fake_module(tmp_path, {"backend": {"archs": ["12.0a"]}})
    monkeypatch.setattr(impls, "_device_cc", lambda: None)
    impls._check_arch("flashrt/x", mod)


def test_undeclared_package_is_not_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(impls, "_device_cc", lambda: (8, 0))
    impls._check_arch("flashrt/x", _fake_module(tmp_path))


def test_refusal_does_not_reload_the_package(monkeypatch, tmp_path):
    # get_kernel must run at most once per repo even when the arch check
    # refuses: a second load re-registers fake ops and torch.library
    # raises an error that has nothing to do with the real problem
    calls = {"n": 0}
    mod = _fake_module(tmp_path, {"backend": {"archs": ["12.0a"]}})

    def fake_get_kernel(repo, version=None):
        calls["n"] += 1
        return mod

    import kernels

    monkeypatch.setattr(kernels, "get_kernel", fake_get_kernel)
    monkeypatch.setattr(impls, "_device_cc", lambda: (8, 9))
    impls._LOADED.pop(("test/arch-refused", ">=1"), None)
    impls.hub_kernel.cache_clear()
    with pytest.raises(ValueError, match="refused"):
        impls.hub_kernel("test/arch-refused", ">=1")
    with pytest.raises(ValueError, match="refused"):
        impls.hub_kernel("test/arch-refused", ">=1")
    assert calls["n"] == 1
    impls._LOADED.pop(("test/arch-refused", ">=1"), None)
    impls.hub_kernel.cache_clear()

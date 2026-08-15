"""Chameleon frontend contract tests — no checkpoint required.

Covers registry/lazy-import behavior, hardware fail-fast gates, the
Thor prompt pad-to-16 capacity boundary, the Orin generation-parameter
boundary, and the load_model(config="chameleon") redirect. GPU-heavy
eager-vs-graph consistency lives in the precision scripts
(scripts/check_chameleon_thor_precision.py) which need a checkpoint.
"""

import importlib
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from flash_rt.hardware import _PIPELINE_MAP, resolve_pipeline_class

REPO_ROOT = Path(__file__).resolve().parents[1]
_OPTIONAL_BUILD_MODULES = {
    "flash_rt.flash_rt_kernels",
    "flash_rt.flash_rt_fa2",
}


def _chameleon_build_enabled() -> bool:
    build_dir = Path(os.environ.get("FLASHRT_BUILD_DIR", REPO_ROOT / "build"))
    cache = build_dir / "CMakeCache.txt"
    if not cache.is_file():
        return False
    return any(
        line.startswith("FLASHRT_ENABLE_CHAMELEON:BOOL=")
        and line.rsplit("=", 1)[-1].upper() in {"ON", "TRUE", "1"}
        for line in cache.read_text(errors="replace").splitlines()
    )


def _load_frontend(module_name: str, class_name: str):
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if _chameleon_build_enabled() or exc.name not in _OPTIONAL_BUILD_MODULES:
            raise
        return None
    return getattr(module, class_name)


ChameleonTorchFrontendThor = _load_frontend(
    "flash_rt.frontends.torch.chameleon_thor", "ChameleonTorchFrontendThor")
ChameleonTorchFrontendRtxSm87 = _load_frontend(
    "flash_rt.frontends.torch.chameleon_rtx_sm87",
    "ChameleonTorchFrontendRtxSm87",
)
_THOR_IMPORT = ChameleonTorchFrontendThor is not None
_ORIN_IMPORT = ChameleonTorchFrontendRtxSm87 is not None
PAD_ID = 1

needs_thor = pytest.mark.skipif(not _THOR_IMPORT,
                                reason="chameleon_thor frontend not importable")
needs_orin = pytest.mark.skipif(not _ORIN_IMPORT,
                                reason="chameleon_rtx_sm87 frontend not importable")


# ---------------------------------------------------------------- registry

def test_registry_maps_to_expected_frontends():
    assert _PIPELINE_MAP[("chameleon", "torch", "thor")] == (
        "flash_rt.frontends.torch.chameleon_thor",
        "ChameleonTorchFrontendThor")
    assert _PIPELINE_MAP[("chameleon", "torch", "rtx_sm87")] == (
        "flash_rt.frontends.torch.chameleon_rtx_sm87",
        "ChameleonTorchFrontendRtxSm87")


def test_registry_entries_are_lazy_module_strings():
    for key in (("chameleon", "torch", "thor"),
                ("chameleon", "torch", "rtx_sm87")):
        mod, cls_name = _PIPELINE_MAP[key]
        assert isinstance(mod, str) and isinstance(cls_name, str)


def test_frontend_import_link_errors_are_not_swallowed(monkeypatch):
    def fail_import(_):
        raise ImportError("undefined symbol: chameleon_test_symbol")

    monkeypatch.setattr(importlib, "import_module", fail_import)
    with pytest.raises(ImportError, match="undefined symbol"):
        _load_frontend("flash_rt.frontends.torch.chameleon_thor", "Frontend")


@needs_thor
def test_resolve_thor_pipeline_class():
    cls = resolve_pipeline_class("chameleon", "torch", "thor")
    assert cls.__module__ == "flash_rt.frontends.torch.chameleon_thor"
    assert cls.__name__ == "ChameleonTorchFrontendThor"


@needs_orin
def test_resolve_orin_pipeline_class():
    cls = resolve_pipeline_class("chameleon", "torch", "rtx_sm87")
    assert cls.__module__ == "flash_rt.frontends.torch.chameleon_rtx_sm87"
    assert cls.__name__ == "ChameleonTorchFrontendRtxSm87"


def test_sm87_allowlist_rejects_unsupported_config():
    with pytest.raises(RuntimeError, match="SM87"):
        resolve_pipeline_class("groot_n17", "torch", "rtx_sm87")


def test_load_model_chameleon_redirects_with_clear_error():
    pytest.importorskip("numpy")
    import flash_rt
    with pytest.raises(Exception, match="not served through"):
        flash_rt.load_model("/nonexistent/fake-ckpt", config="chameleon",
                            framework="torch", hardware="thor")


# ------------------------------------------------------- Thor hardware gate

def _thor_probe():
    obj = object.__new__(ChameleonTorchFrontendThor)
    return ChameleonTorchFrontendThor._require_arch(obj)


@needs_thor
def test_thor_rejects_when_cuda_unavailable(monkeypatch):
    monkeypatch.delenv("FLASHRT_CHAMELEON_THOR_FORCE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        _thor_probe()


@needs_thor
def test_thor_rejects_wrong_capability(monkeypatch):
    monkeypatch.delenv("FLASHRT_CHAMELEON_THOR_FORCE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (8, 7))
    with pytest.raises(RuntimeError, match="targets SM110"):
        _thor_probe()


@needs_thor
def test_thor_accepts_sm110(monkeypatch):
    monkeypatch.delenv("FLASHRT_CHAMELEON_THOR_FORCE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (11, 0))
    _thor_probe()  # must not raise


@needs_thor
def test_thor_documented_env_override_skips_probe(monkeypatch):
    monkeypatch.setenv("FLASHRT_CHAMELEON_THOR_FORCE", "1")
    # No CUDA mocking: the override must return before touching torch.cuda.
    _thor_probe()


@needs_thor
@pytest.mark.parametrize("max_seq", [-16, -1, 0, 15])
def test_thor_rejects_too_small_max_seq(monkeypatch, tmp_path, max_seq):
    monkeypatch.setenv("FLASHRT_CHAMELEON_THOR_FORCE", "1")
    with pytest.raises(ValueError, match="max_seq must be at least 16"):
        ChameleonTorchFrontendThor(tmp_path, max_seq=max_seq)


# ----------------------------------------------- Thor prompt padding bounds

def _bare_thor(se_max):
    fe = object.__new__(ChameleonTorchFrontendThor)
    fe._Se_max = se_max
    fe._use_autotune = False
    fe._use_cuda_graph = False
    fe._embed_ids = lambda ids: None
    return fe


@needs_thor
def test_prompt_padding_rejects_overshoot_on_nonaligned_capacity():
    # 30-token prompt pads to 32, which exceeds a 31-token capacity.
    fe = _bare_thor(31)
    fe.encode_prompt = lambda text, images: list(range(30))
    with pytest.raises(ValueError, match="padded sequence length"):
        fe.set_prompt("x")


@needs_thor
def test_prompt_padding_accepts_prompt_within_capacity():
    fe = _bare_thor(32)
    fe.encode_prompt = lambda text, images: list(range(30))
    ids = fe.set_prompt("x")
    assert fe._real_len == 30
    assert fe.Se == 32 and len(ids) == 32
    assert ids[30:] == [PAD_ID, PAD_ID]


@needs_thor
def test_prompt_padding_accepts_exact_multiple_of_16():
    fe = _bare_thor(32)
    fe.encode_prompt = lambda text, images: list(range(32))
    ids = fe.set_prompt("x")
    assert fe.Se == 32 and len(ids) == 32 and fe._real_len == 32


# ------------------------------------------------------ Orin hardware gate

@needs_orin
def test_orin_rejects_wrong_capability_before_checkpoint_work(monkeypatch):
    monkeypatch.delenv("FLASHRT_CHAMELEON_SM87_FORCE", raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (11, 0))
    with pytest.raises(RuntimeError, match="targets SM87"):
        ChameleonTorchFrontendRtxSm87("/nonexistent/fake-ckpt")


@needs_orin
def test_orin_env_override_bypasses_arch_gate(monkeypatch):
    monkeypatch.setenv("FLASHRT_CHAMELEON_SM87_FORCE", "1")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (11, 0))
    # The gate must be skipped; construction then fails later for an
    # unrelated reason (missing checkpoint), never the SM87 gate error.
    with pytest.raises(Exception) as excinfo:
        ChameleonTorchFrontendRtxSm87("/nonexistent/fake-ckpt")
    assert "targets SM87" not in str(excinfo.value)


# --------------------------------------------- Orin generation-param bounds

@needs_orin
def _bare_orin():
    fe = object.__new__(ChameleonTorchFrontendRtxSm87)
    fe._prompt_ready = True
    fe.S = 16
    fe.max_seq = 4096
    return fe


@needs_orin
def test_generate_negative_max_new_tokens_raises():
    with pytest.raises(ValueError, match="max_new_tokens"):
        _bare_orin().generate(max_new_tokens=-1)


@needs_orin
def test_generate_zero_max_new_tokens_returns_empty():
    assert _bare_orin().generate(max_new_tokens=0) == ""
    assert _bare_orin().generate(max_new_tokens=0, return_ids=True) == []

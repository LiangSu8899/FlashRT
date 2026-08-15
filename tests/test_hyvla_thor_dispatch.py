"""Dispatch and registration smoke tests for HyVLA (no GPU required)."""

import pytest


def test_hyvla_thor_dispatch_resolves():
    from flash_rt.hardware import resolve_pipeline_class

    try:
        cls = resolve_pipeline_class("hyvla", "torch", "thor")
    except ModuleNotFoundError as exc:
        if exc.name != "flash_rt.flash_rt_kernels":
            raise
        pytest.skip("flash_rt_kernels was not built")
    assert cls.__name__ == "HyVLATorchFrontendThor"
    assert cls.__module__ == "flash_rt.frontends.torch.hyvla_thor"


def test_hyvla_pipeline_map_is_one_to_one():
    from flash_rt.hardware import _PIPELINE_MAP

    entries = {k: v for k, v in _PIPELINE_MAP.items() if k[0] == "hyvla"}
    assert ("hyvla", "torch", "thor") in entries
    classes = [v[1] for v in entries.values()]
    assert len(classes) == len(set(classes)), "multiple tuples share a class"


def test_hyvla_is_a_supported_load_model_config():
    import inspect

    import flash_rt

    doc = inspect.getsource(flash_rt.load_model)
    assert '"hyvla"' in doc

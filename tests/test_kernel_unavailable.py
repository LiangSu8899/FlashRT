"""An absent kernel package is an outcome, not a crash.

The distribution layer can come up empty in several unrelated ways —
unpublished repo, offline cache without it staged, no build variant for
the host, arch declaration that excludes it. Callers do not need to
tell those apart; they need one catchable event, so that a lever or a
variant that cannot be served is written down and the rest of the run
continues.
"""

from __future__ import annotations

import pytest

from flash_rt.structures import impls


def test_every_unavailability_raises_one_catchable_type(monkeypatch):
    for failure in (OSError("no build variant for this host"),
                    RuntimeError("kernel 'x/y' is not staged"),
                    ValueError("Version >=1 not found")):
        impls.hub_kernel.cache_clear()
        impls._LOADED.clear()

        def boom(*args, _f=failure, **kwargs):
            raise _f

        monkeypatch.setattr("kernels.get_kernel", boom)
        with pytest.raises(impls.KernelUnavailable) as caught:
            impls.hub_kernel("some/repo", ">=1")
        # the operator still needs to know which fix applies
        assert str(failure) in str(caught.value)
        assert "some/repo" in str(caught.value)


def test_unavailability_is_a_value_error_so_refusal_paths_catch_it():
    # the recipe engine records a lever refused on ValueError and the
    # variant families step to the next member on ValueError; an absent
    # package must land in both, not sail past them
    assert issubclass(impls.KernelUnavailable, ValueError)


def test_arch_refusal_uses_the_same_type(monkeypatch):
    impls.hub_kernel.cache_clear()
    impls._LOADED.clear()

    class Module:
        __file__ = "/nonexistent/kernel/__init__.py"

    monkeypatch.setattr("kernels.get_kernel", lambda *a, **k: Module())
    monkeypatch.setattr(impls, "_declared_archs", lambda module: ["9.0"])
    monkeypatch.setattr(impls, "_device_cc", lambda: (11, 0))

    with pytest.raises(impls.KernelUnavailable) as caught:
        impls.hub_kernel("some/repo", ">=1")
    assert "sm 11.0" in str(caught.value)

    impls.hub_kernel.cache_clear()
    impls._LOADED.clear()

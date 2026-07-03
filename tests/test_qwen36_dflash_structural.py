"""Structural tests for the Qwen3.6 DFlash spec-decode path.

These run without model checkpoints or a GPU: they validate the
contracts that hardware benchmarks cannot guard cheaply —

  * the per-token window commit reads tap rows 0..N BEFORE the
    end-of-cycle taps[:, 0] shuffle overwrites row 0, and stores a
    copy (later tap mutation must not alias into the window input);
  * the spec-decode loop keeps that ordering (source-order guard);
  * generate fails fast with a clear error when no drafter is loaded;
  * the public ``init_dflash_drafter`` wrapper delegates to the
    loader;
  * Thor's per-token window env routing (default on, opt-out, window
    length override).

GPU/end-to-end evidence for this path lives in the hardware-gated
benchmarks; see docs/qwen36_dflash.md.
"""

from __future__ import annotations

import inspect

import pytest

torch = pytest.importorskip("torch")

from flash_rt.frontends.torch import _qwen36_rtx_dflash_forward as dff  # noqa: E402
from flash_rt.frontends.torch.qwen36_rtx import (  # noqa: E402
    Qwen36TorchFrontendRtx,
)
from flash_rt.frontends.torch.qwen36_thor import (  # noqa: E402
    Qwen36TorchFrontendThor,
)


HIDDEN = 8
KV = 16


def _stub_rtx():
    fe = Qwen36TorchFrontendRtx.__new__(Qwen36TorchFrontendRtx)
    taps = torch.zeros(5, KV, HIDDEN)
    for row in range(KV):
        taps[:, row] = row + 1
    fe._dflash_taps_buf = taps
    fe._dflash_buf = {
        "pt_taps_rows": torch.zeros(KV, 5, HIDDEN),
    }
    return fe


def test_window_commit_reads_rows_before_shuffle(monkeypatch):
    fe = _stub_rtx()
    seen = []
    monkeypatch.setattr(
        dff, "pertoken_window_append",
        lambda frontend, rows: seen.append(rows))

    N = 3
    fe._dflash_window_commit(N)

    assert len(seen) == 1
    rows = seen[0]
    assert rows.shape == (N + 1, 5, HIDDEN)
    # Row order: oldest committed row first, values 1..N+1 per the
    # stub filling — row 0 must be the ORIGINAL row 0, not row N.
    expect = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert torch.equal(rows[:, 0, 0], expect)

    # The end-of-cycle shuffle overwrites tap row 0 with row N; the
    # committed rows must be a copy, not a view into the tap buffer.
    fe._dflash_taps_buf[:, 0].copy_(fe._dflash_taps_buf[:, N])
    assert torch.equal(rows[:, 0, 0], expect)


def test_window_commit_full_accept_covers_all_rows(monkeypatch):
    fe = _stub_rtx()
    seen = []
    monkeypatch.setattr(
        dff, "pertoken_window_append",
        lambda frontend, rows: seen.append(rows))

    fe._dflash_window_commit(KV - 1)
    assert seen[0].shape[0] == KV
    assert torch.equal(
        seen[0][:, 0, 0], torch.arange(1.0, KV + 1))


def test_generate_loop_commits_window_before_tap_shuffle():
    src = inspect.getsource(
        Qwen36TorchFrontendRtx.generate_own_speculative_DFlash_nvfp4)
    commit = src.index("_dflash_window_commit")
    shuffle = src.index(
        "_dflash_taps_buf[:, 0].copy_", commit)
    assert commit < shuffle, (
        "the per-token window must be committed before the taps[:, 0] "
        "shuffle overwrites row 0")


def test_generate_fails_fast_without_drafter():
    fe = Qwen36TorchFrontendRtx.__new__(Qwen36TorchFrontendRtx)

    class _Weights:
        ptrs = {}

    fe._weights = _Weights()
    with pytest.raises(RuntimeError, match="DFlash drafter not loaded"):
        fe.generate_own_speculative_DFlash_nvfp4(
            torch.zeros(1, 4, dtype=torch.long), max_new_tokens=4)


def test_public_drafter_init_delegates(monkeypatch):
    fe = Qwen36TorchFrontendRtx.__new__(Qwen36TorchFrontendRtx)
    calls = []
    monkeypatch.setattr(
        Qwen36TorchFrontendRtx, "_load_dflash_drafter",
        lambda self, ckpt_dir=None: calls.append(ckpt_dir))
    fe.init_dflash_drafter("/tmp/ckpt")
    assert calls == ["/tmp/ckpt"]


def _thor_drafter_load(monkeypatch):
    """Run Thor's _load_dflash_drafter with the base loader stubbed."""
    monkeypatch.setattr(
        Qwen36TorchFrontendRtx, "_load_dflash_drafter",
        lambda self, ckpt_dir=None: None)
    fe = Qwen36TorchFrontendThor.__new__(Qwen36TorchFrontendThor)
    fe._fp8_K_cache = torch.zeros(1)     # skip FP8 cache allocation
    fe._K_save_max = 16                  # skip checkpoint-buffer grow
    fe._MAX_PUBLIC_SPEC_K = 15
    fe._load_dflash_drafter()
    return fe


def test_thor_pertoken_default_on(monkeypatch):
    monkeypatch.delenv("FLASHRT_QWEN36_DFLASH_PERTOKEN", raising=False)
    monkeypatch.delenv("FLASHRT_QWEN36_DFLASH_WINDOW", raising=False)
    fe = _thor_drafter_load(monkeypatch)
    assert fe._dflash_pertoken_window is True
    assert fe._dflash_pertoken_win == 128


def test_thor_pertoken_env_opt_out(monkeypatch):
    monkeypatch.setenv("FLASHRT_QWEN36_DFLASH_PERTOKEN", "0")
    fe = _thor_drafter_load(monkeypatch)
    assert fe._dflash_pertoken_window is False


def test_thor_pertoken_window_env_override(monkeypatch):
    monkeypatch.delenv("FLASHRT_QWEN36_DFLASH_PERTOKEN", raising=False)
    monkeypatch.setenv("FLASHRT_QWEN36_DFLASH_WINDOW", "64")
    fe = _thor_drafter_load(monkeypatch)
    assert fe._dflash_pertoken_win == 64

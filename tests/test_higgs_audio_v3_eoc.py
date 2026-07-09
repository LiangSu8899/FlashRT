import torch

from flash_rt.frontends.torch.higgs_audio_v3_rtx import (
    _delayed_eoc_countdown,
    _repeat_code_run,
)


def test_delayed_eoc_countdown_flushes_full_tail():
    nc = 8
    eoc = 1025
    codes = torch.tensor([eoc, 1, 2, 3, 4, 5, 6, 7])
    assert _delayed_eoc_countdown(codes, nc, eoc) == nc - 1


def test_delayed_eoc_countdown_accepts_later_codebook_eoc():
    nc = 8
    eoc = 1025
    codes = torch.tensor([10, 11, 12, eoc, 14, 15, 16, 17])
    assert _delayed_eoc_countdown(codes, nc, eoc) == 4


def test_delayed_eoc_countdown_absent():
    assert _delayed_eoc_countdown(torch.arange(8), 8, 1025) is None


def test_repeat_code_run_counts_identical_rows_and_resets():
    prev, count = _repeat_code_run(torch.tensor([1, 2, 3]), None, 0)
    assert prev == (1, 2, 3)
    assert count == 1

    prev, count = _repeat_code_run(torch.tensor([1, 2, 3]), prev, count)
    assert prev == (1, 2, 3)
    assert count == 2

    prev, count = _repeat_code_run(torch.tensor([1, 2, 4]), prev, count)
    assert prev == (1, 2, 4)
    assert count == 1

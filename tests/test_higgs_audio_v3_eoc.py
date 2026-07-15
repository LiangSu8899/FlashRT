import inspect
from types import SimpleNamespace

import pytest
import torch

from flash_rt.frontends.torch.higgs_audio_v3_rtx import (
    HiggsAudioV3TorchFrontendRtx,
    _delayed_eoc_countdown,
    _normalize_sampling_options,
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


@pytest.mark.parametrize("method", ["decode_stream", "predict", "generate",
                                     "generate_stream"])
def test_higgs_public_api_keeps_greedy_as_default(method):
    signature = inspect.signature(getattr(HiggsAudioV3TorchFrontendRtx, method))
    assert signature.parameters["temperature"].default == 0.0


def test_sampling_options_preserve_greedy_without_random_seed(monkeypatch):
    def unexpected_random(_):
        raise AssertionError("greedy path must not request entropy")

    monkeypatch.setattr("os.urandom", unexpected_random)
    assert _normalize_sampling_options(0.0, None) == (0.0, 0)
    assert _normalize_sampling_options(1e-5, None) == (1e-5, 0)


def test_sampling_options_use_one_explicit_request_seed():
    assert _normalize_sampling_options(1.0, 1234) == (1.0, 1234)


def test_sampling_options_materialize_one_random_seed_per_request(monkeypatch):
    calls = []

    def request_seed(size):
        calls.append(size)
        return bytes(range(size))

    monkeypatch.setattr("os.urandom", request_seed)
    assert _normalize_sampling_options(1.0, None) == (
        1.0,
        int.from_bytes(bytes(range(8)), "little"),
    )
    assert calls == [8]


@pytest.mark.parametrize("temperature", [-0.1, float("nan"), float("inf")])
def test_sampling_options_reject_invalid_temperature(temperature):
    with pytest.raises(ValueError, match="temperature"):
        _normalize_sampling_options(temperature, 0)


@pytest.mark.parametrize("seed", [-1, 1 << 64])
def test_sampling_options_reject_out_of_range_seed(seed):
    with pytest.raises(ValueError, match="seed"):
        _normalize_sampling_options(1.0, seed)


@pytest.mark.parametrize("seed", [True, 1.5, "1234"])
def test_sampling_options_reject_non_integer_seed(seed):
    with pytest.raises(TypeError, match="seed"):
        _normalize_sampling_options(1.0, seed)


def _decode_only_frontend():
    frontend = object.__new__(HiggsAudioV3TorchFrontendRtx)
    frontend._cfg = {
        "num_codebooks": 8,
        "codebook_vocab": 1026,
        "hidden": 1,
    }
    frontend._gen_pos = 3
    frontend._gen_logits = torch.zeros(8, 1026, dtype=torch.bfloat16)
    frontend._weights = {
        "codebook": torch.zeros(8 * 1026, 1, dtype=torch.bfloat16)
    }
    frontend._codes_dev = torch.zeros(8, dtype=torch.long)
    frontend._embed_buf = torch.zeros(1, 1, dtype=torch.bfloat16)
    frontend.max_new_frames = 1
    frontend._decode_logits = lambda *_: frontend._gen_logits
    return frontend


def test_default_decode_does_not_depend_on_sampling_binding(monkeypatch):
    calls = []
    kernels = SimpleNamespace(
        delayed_codebook_argmax_embed_bf16=lambda *args: calls.append(args)
    )
    monkeypatch.setattr("flash_rt.flash_rt_kernels", kernels, raising=False)
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda: SimpleNamespace(cuda_stream=0))

    assert list(_decode_only_frontend().decode_stream()) == []
    assert len(calls) == 1


def test_sampling_fallback_fails_clearly_with_an_old_kernel_build(monkeypatch):
    kernels = SimpleNamespace(delayed_codebook_argmax_embed_bf16=lambda *args: None)
    monkeypatch.setattr("flash_rt.flash_rt_kernels", kernels, raising=False)

    with pytest.raises(RuntimeError, match="sampling kernel"):
        list(_decode_only_frontend().decode_stream(
            temperature=1.0, seed=1234))

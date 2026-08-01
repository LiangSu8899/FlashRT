"""Driving the runtime: where the cursor ends up, and what a graph replays.

The checkpoint here is synthesized, so nothing it emits means anything. What
is under test is the driving: that a prompt read in one pass and the same
prompt read in several leave the same state behind, that the cursor lands
where the next token goes, and that a captured step replays as the sequence
grows rather than repeating the step it was captured from. Each of those is
wrong in a way that still produces fluent-looking tokens.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

if not torch.cuda.is_available():                        # pragma: no cover
    pytest.skip("needs a GPU", allow_module_level=True)

try:
    from flash_rt import flash_rt_kernels as fvk
except ImportError:                                      # pragma: no cover
    pytest.skip("flash_rt_kernels is not built", allow_module_level=True)

if not hasattr(fvk, "gqa_decode_attention_bf16"):        # pragma: no cover
    pytest.skip("built without the portable decode kernels",
                allow_module_level=True)

from flash_rt.frontends.torch.qwen35_text import (  # noqa: E402
    StepTimings,
    TextRuntime,
)
from tests.test_qwen35_text_weights import _write_checkpoint  # noqa: E402

PROMPT = [3, 17, 200, 41, 9, 128, 77, 5, 300, 12]


def _close(got, want, tolerance=2e-2):
    """Agreement relative to the size of what is being compared.

    A prompt read in one pass and the same prompt read in several do not
    produce identical numbers and are not meant to: the batched projection
    reduces in a different order than the single-row one. What has to hold is
    that the difference stays at the size of that rounding.
    """
    got, want = got.float(), want.float()
    scale = max(want.abs().max().item(), 1e-6)
    return (got - want).abs().max().item() / scale < tolerance


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    directory = tmp_path_factory.mktemp("qwen35_text")
    _write_checkpoint(directory)
    return str(directory)


def _runtime(checkpoint, chunk):
    return TextRuntime.from_checkpoint(checkpoint, max_seq=128,
                                       max_chunk=chunk)


def test_the_cursor_lands_where_the_next_token_goes(checkpoint):
    # The staging kernel writes the key and value at the cursor and the
    # attention reads up to the length beside it. One off in either and a
    # token attends to a row nothing wrote.
    runtime = _runtime(checkpoint, 8)
    runtime.reset()
    runtime.read_prompt(PROMPT)
    assert runtime.work.cursor.tensor.tolist() == [len(PROMPT), len(PROMPT) + 1]

    runtime.step()
    torch.cuda.synchronize()
    assert runtime.work.cursor.tensor.tolist() == [len(PROMPT) + 1,
                                                   len(PROMPT) + 2]
    runtime.close()


def test_a_prompt_read_in_chunks_leaves_the_state_it_would_have(checkpoint):
    # A prompt longer than the workspace is read in several passes, and the
    # recurrence carries across them. If it did not, only the last chunk would
    # reach the model -- which shortens the context silently.
    whole = _runtime(checkpoint, len(PROMPT))
    whole.reset()
    whole.read_prompt(PROMPT)
    torch.cuda.synchronize()
    expected_token = whole.token()
    expected_state = [b.tensor.clone() for b in whole.work.recurrent]
    expected_conv = [b.tensor.clone() for b in whole.work.conv]
    expected_keys = [b.tensor[:len(PROMPT)].clone() for b in whole.work.keys]
    whole.close()

    for chunk in (1, 3, 4):
        runtime = _runtime(checkpoint, chunk)
        runtime.reset()
        runtime.read_prompt(PROMPT)
        torch.cuda.synchronize()

        assert runtime.token() == expected_token, f"chunk {chunk}"
        for got, want in zip(runtime.work.recurrent, expected_state):
            assert _close(got.tensor, want), f"recurrent, chunk {chunk}"
        for got, want in zip(runtime.work.conv, expected_conv):
            assert _close(got.tensor, want), f"convolution, chunk {chunk}"
        for got, want in zip(runtime.work.keys, expected_keys):
            assert _close(got.tensor[:len(PROMPT)], want), f"keys, chunk {chunk}"
        runtime.close()


def test_a_captured_step_advances_the_sequence(checkpoint):
    # A graph replays the addresses it was captured with. If the position were
    # passed by value instead of read from the device, every replayed token
    # would be the captured token again -- which reads as a model that
    # repeats itself rather than as a bug.
    runtime = _runtime(checkpoint, 8)
    runtime.reset()
    eager = runtime.generate(PROMPT, max_new_tokens=8)

    runtime.reset()
    runtime.read_prompt(PROMPT)
    runtime.capture()
    replayed = runtime.generate(PROMPT, max_new_tokens=8)

    assert replayed == eager
    # The cursor is the thing that has to move: a position passed by value
    # would be baked into the capture, and every replay would rewrite the
    # same cache row.
    assert runtime.work.cursor.tensor[0].item() == len(PROMPT) + len(eager) - 1
    runtime.close()


def test_capture_leaves_the_sequence_where_it_found_it(checkpoint):
    # Capturing runs the step, several times, for warm-up. A capture that did
    # not put the state back would consume tokens of context that the caller
    # never asked for.
    runtime = _runtime(checkpoint, 8)
    runtime.reset()
    runtime.read_prompt(PROMPT)
    torch.cuda.synchronize()
    before = (runtime.work.cursor.tensor.clone(), runtime.token(),
              [b.tensor.clone() for b in runtime.work.recurrent])

    runtime.capture()
    torch.cuda.synchronize()

    assert torch.equal(runtime.work.cursor.tensor, before[0])
    assert runtime.token() == before[1]
    for got, want in zip(runtime.work.recurrent, before[2]):
        assert torch.equal(got.tensor, want)
    runtime.close()


def test_an_empty_prompt_is_refused(checkpoint):
    runtime = _runtime(checkpoint, 8)
    with pytest.raises(ValueError):
        runtime.read_prompt([])
    runtime.close()


def test_a_prompt_longer_than_the_workspace_is_refused(checkpoint):
    runtime = TextRuntime.from_checkpoint(checkpoint, max_seq=16, max_chunk=8)
    with pytest.raises(ValueError):
        runtime.read_prompt(list(range(20)))
    runtime.close()


def test_the_timings_report_the_distribution_not_the_mean(checkpoint):
    # A control loop misses on the slow steps, and a mean hides exactly those.
    runtime = _runtime(checkpoint, 8)
    timings = StepTimings()
    runtime.generate(PROMPT, max_new_tokens=6, timings=timings)
    summary = timings.summary()
    assert summary["prompt_tokens"] == len(PROMPT)
    assert summary["p99_ms"] >= summary["p50_ms"]
    assert summary["max_ms"] >= summary["p99_ms"]
    assert summary["tokens_per_second"] > 0
    runtime.close()


def test_a_recurrence_width_the_kernel_does_not_implement_is_refused(tmp_path):
    # The recurrence kernel is written for one head width and returns without
    # doing anything for any other, leaving its output buffer holding whatever
    # was there before. On a fresh allocation that is zeros and the model
    # merely degrades; on a reused one it is the previous sequence. Neither
    # reports anything, so the width is refused at load.
    _write_checkpoint(tmp_path, overrides={"linear_key_head_dim": 64,
                                           "linear_value_head_dim": 64})

    with pytest.raises(RuntimeError, match="head width"):
        TextRuntime.from_checkpoint(str(tmp_path))

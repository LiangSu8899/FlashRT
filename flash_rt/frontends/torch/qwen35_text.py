"""A ``qwen3_5_text`` checkpoint, run as kernels against fixed addresses.

The runtime is two passes over the same blocks. A prompt goes through in
chunks, wide enough that the weights are read once for many positions; a token
goes through one row at a time, where reading the weights is the whole cost
and the point is to issue the reads and nothing else.

The decode step is captured as a graph. That is worth more here than the
launch count suggests: the step is around three hundred small launches, and on
a part where a launch costs microseconds that is a fixed tax on every token,
paid whether or not the part is busy. Capturing it needs the step to touch
nothing that moves -- which is why the position and the length are read from
device memory rather than passed in, and why the token the step chooses is
written where the next step's embedding lookup will look for it.

    runtime = TextRuntime.from_checkpoint(path)
    runtime.generate([1, 2, 3], max_new_tokens=32)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from flash_rt.frontends.torch import _qwen35_text_decode as decode
from flash_rt.frontends.torch._qwen35_text_spec import validate_checkpoint
from flash_rt.frontends.torch._qwen35_text_weights import (
    TextWeights,
    load_text_weights,
)

# The head width the gated-delta recurrence is written for. It is a property
# of the kernel, not of any checkpoint, and is stated here because the kernel
# does not report being handed another one.
RECURRENCE_HEAD_DIM = 128


@dataclass
class StepTimings:
    """What a run cost, kept as the numbers rather than as an average.

    A control loop cares about the distribution: a step that is usually fast
    and occasionally not is a loop that occasionally misses, and a mean hides
    exactly that.
    """

    prefill_ms: float = 0.0
    prompt_tokens: int = 0
    steps: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, float]:
        ordered = sorted(self.steps)
        if not ordered:
            return {"prefill_ms": self.prefill_ms,
                    "prompt_tokens": float(self.prompt_tokens)}

        def at(fraction: float) -> float:
            index = min(len(ordered) - 1, int(fraction * len(ordered)))
            return ordered[index]

        return {
            "prefill_ms": self.prefill_ms,
            "prompt_tokens": float(self.prompt_tokens),
            "tokens": float(len(ordered)),
            "p50_ms": at(0.50),
            "p99_ms": at(0.99),
            "max_ms": ordered[-1],
            "tokens_per_second": 1000.0 / at(0.50),
        }


class TextRuntime:
    """One loaded checkpoint and the buffers a sequence runs in."""

    def __init__(self, weights: TextWeights, work: decode.Workspace, fvk):
        self.weights = weights
        self.work = work
        self.fvk = fvk
        self.dims = weights.dims
        self.device = work.device
        self._graph: torch.cuda.CUDAGraph | None = None
        self._graph_pool = None

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cuda:0",
                        max_seq: int = 4096, max_chunk: int = 64
                        ) -> "TextRuntime":
        from flash_rt import flash_rt_kernels as fvk

        for name in ("w4a16_packed_matvec_bf16",
                     "linear_attn_split_broadcast_gate_bf16",
                     "gqa_decode_attention_bf16"):
            if not hasattr(fvk, name):
                raise RuntimeError(
                    f"flash_rt_kernels was built without {name}; this runtime "
                    "needs the packed weight-only and portable decode kernels")

        contract = validate_checkpoint(path)
        dims = contract["dims"]
        # The recurrence kernel is written for one head width and returns
        # without doing anything for any other, which leaves its output buffer
        # holding whatever was there before. That reads as a model that works
        # and then drifts, so the width is refused here instead.
        if (dims.lin_key_head_dim != RECURRENCE_HEAD_DIM
                or dims.lin_value_head_dim != RECURRENCE_HEAD_DIM):
            raise RuntimeError(
                f"the recurrence kernel implements head width "
                f"{RECURRENCE_HEAD_DIM}, and this checkpoint uses "
                f"{dims.lin_key_head_dim} for keys and "
                f"{dims.lin_value_head_dim} for values")
        weights = load_text_weights(path, contract, device=device)
        work = decode.Workspace(weights, device=device, max_chunk=max_chunk,
                                max_seq=max_seq)
        return cls(weights, work, fvk)

    # ── the two passes ──

    def _stream(self) -> int:
        # Read at the point of use rather than stored: during capture the
        # current stream is the capture stream, and a kernel issued on any
        # other one is not in the graph.
        return torch.cuda.current_stream(self.device).cuda_stream

    def reset(self) -> None:
        """Start a new sequence, keeping every address where it was."""
        self.work.reset()

    def read_prompt(self, token_ids: list[int]) -> None:
        """Take the prompt into the state, in chunks.

        Nothing is returned. What the prompt leaves behind is the recurrent
        state, the convolution's tail and the key/value cache, and the row
        after the last one is where generation starts.
        """
        if not token_ids:
            raise ValueError("the prompt is empty")
        if len(token_ids) >= self.work.max_seq:
            raise ValueError(
                f"prompt of {len(token_ids)} does not fit a workspace built "
                f"for {self.work.max_seq} positions")

        work = self.work
        stream = self._stream()
        position = 0
        for start in range(0, len(token_ids), work.max_chunk):
            chunk = token_ids[start:start + work.max_chunk]
            work.token.tensor[:len(chunk)] = torch.tensor(
                chunk, dtype=torch.int64, device=self.device)
            work.seek(position, len(chunk))
            decode.forward(self.weights, work, self.fvk, len(chunk), stream)
            position += len(chunk)

        # Only the last position has a distribution anyone reads, and it is
        # the last row of the last chunk that carries it.
        last_row = (len(token_ids) - 1) % work.max_chunk
        decode.project_to_vocabulary(self.weights, work, self.fvk, stream,
                                     row=last_row)
        self.fvk.qwen36_argmax_bf16(
            work.logits.address, work.token.address, 1,
            self.weights.top["vocab_size"], stream)
        work.seek(position, 1)

    def step(self) -> None:
        """One token, through the graph if there is one."""
        if self._graph is not None:
            self._graph.replay()
            return
        decode.decode_step(self.weights, self.work, self.fvk, self._stream())

    def token(self) -> int:
        """The token the last pass chose."""
        return int(self.work.token.tensor[0].item())

    # ── capture ──

    def capture(self, warmup: int = 3) -> None:
        """Capture one decode step, so a token is one launch.

        Captured from a state the caller has already put in place: the graph
        records addresses, not values, so it does not matter which sequence
        was loaded when it was taken. It does matter that the warm-up runs on
        a side stream, which is what the capture wants, and that the state the
        warm-up disturbs is put back -- a warm-up step advances the cursor and
        writes a cache row like any other.
        """
        work = self.work
        cursor = work.cursor.tensor.clone()
        state = [buffer.tensor.clone() for buffer in work.recurrent]
        convolution = [buffer.tensor.clone() for buffer in work.conv]
        token = work.token.tensor.clone()

        side = torch.cuda.Stream(device=self.device)
        side.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(side):
            for _ in range(warmup):
                decode.decode_step(self.weights, work, self.fvk,
                                   side.cuda_stream)
        torch.cuda.current_stream(self.device).wait_stream(side)
        torch.cuda.synchronize(self.device)

        self._restore(cursor, state, convolution, token)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            decode.decode_step(self.weights, work, self.fvk, self._stream())
        torch.cuda.synchronize(self.device)

        self._restore(cursor, state, convolution, token)
        self._graph = graph

    def _restore(self, cursor, state, convolution, token) -> None:
        work = self.work
        work.cursor.tensor.copy_(cursor)
        work.token.tensor.copy_(token)
        for buffer, saved in zip(work.recurrent, state):
            buffer.tensor.copy_(saved)
        for buffer, saved in zip(work.conv, convolution):
            buffer.tensor.copy_(saved)

    # ── the loop ──

    def generate(self, token_ids: list[int], max_new_tokens: int = 32,
                 stop_ids: tuple[int, ...] = (),
                 timings: StepTimings | None = None) -> list[int]:
        """Read the prompt, then emit tokens until told to stop.

        Each step is synchronized because the caller is asked whether to stop,
        and that question needs the token on the host. A caller that knows how
        many tokens it wants can replay the graph that many times and read
        them at the end instead.
        """
        self.reset()
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        self.read_prompt(token_ids)
        torch.cuda.synchronize(self.device)
        if timings is not None:
            timings.prefill_ms = (time.perf_counter() - started) * 1e3
            timings.prompt_tokens = len(token_ids)

        emitted = [self.token()]
        if emitted[0] in stop_ids:
            return emitted
        for _ in range(max_new_tokens - 1):
            started = time.perf_counter()
            self.step()
            torch.cuda.synchronize(self.device)
            if timings is not None:
                timings.steps.append((time.perf_counter() - started) * 1e3)
            chosen = self.token()
            emitted.append(chosen)
            if chosen in stop_ids:
                break
        return emitted

    # ── what it cost to load ──

    def footprint(self) -> dict[str, float]:
        """Where the memory went, in the terms that decide whether it fits."""
        return {
            "weights_gib": self.weights.resident_bytes / 2 ** 30,
            "state_gib": self.work.state_bytes / 2 ** 30,
            "reserved_gib": torch.cuda.memory_reserved(self.device) / 2 ** 30,
        }

    def close(self) -> None:
        self._graph = None
        self.work.close()
        self.weights.close()

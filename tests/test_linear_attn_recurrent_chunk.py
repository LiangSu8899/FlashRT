"""The recurrence over a chunk, against the recurrence one position at a time.

A prompt and the tokens after it go through different kernels and must leave
the same state, because the second reads what the first wrote. So the chunked
form is checked against the per-position form it replaces -- both the outputs
and, more importantly, the state, which is what carries into the next token
and is the part no output would reveal.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():                        # pragma: no cover
    pytest.skip("needs a GPU", allow_module_level=True)

try:
    from flash_rt import flash_rt_kernels as fvk
except ImportError:                                      # pragma: no cover
    pytest.skip("flash_rt_kernels is not built", allow_module_level=True)

if not hasattr(fvk, "linear_attn_recurrent_chunk_f32state_bf16"):
    pytest.skip("built without the portable decode kernels",  # pragma: no cover
                allow_module_level=True)

DEVICE = "cuda:0"
HEAD_DIM = 128


def _inputs(steps, heads, head_dim, seed):
    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    def make(*shape):
        return torch.randn(*shape, dtype=torch.bfloat16, device=DEVICE,
                           generator=generator)

    return (make(steps, heads, head_dim), make(steps, heads, head_dim),
            make(steps, heads, head_dim),
            # a decay in the range the gate produces: at most zero, so the
            # state contracts rather than growing without bound
            -make(steps, heads).abs() * 0.5,
            torch.sigmoid(make(steps, heads).float()).to(torch.bfloat16))


def _stepwise(q, k, v, g, beta, heads, head_dim):
    steps = q.shape[0]
    state = torch.zeros(heads, head_dim, head_dim, dtype=torch.float32,
                        device=DEVICE)
    out = torch.empty(steps, heads, head_dim, dtype=torch.bfloat16,
                      device=DEVICE)
    stream = torch.cuda.current_stream(DEVICE).cuda_stream
    row = heads * head_dim * 2
    gate = heads * 2
    for step in range(steps):
        fvk.gated_deltanet_recurrent_qwen36_f32state_bf16io(
            q.data_ptr() + step * row, k.data_ptr() + step * row,
            v.data_ptr() + step * row, g.data_ptr() + step * gate,
            beta.data_ptr() + step * gate, state.data_ptr(),
            out.data_ptr() + step * row, 1, heads, head_dim, head_dim,
            True, stream)
    torch.cuda.synchronize(DEVICE)
    return out, state


def _chunked(q, k, v, g, beta, heads, head_dim, state=None):
    steps = q.shape[0]
    if state is None:
        state = torch.zeros(heads, head_dim, head_dim, dtype=torch.float32,
                            device=DEVICE)
    out = torch.empty(steps, heads, head_dim, dtype=torch.bfloat16,
                      device=DEVICE)
    rc = fvk.linear_attn_recurrent_chunk_f32state_bf16(
        q.data_ptr(), k.data_ptr(), v.data_ptr(), g.data_ptr(),
        beta.data_ptr(), state.data_ptr(), out.data_ptr(), steps, heads,
        head_dim, head_dim, True,
        torch.cuda.current_stream(DEVICE).cuda_stream)
    assert rc == 0
    torch.cuda.synchronize(DEVICE)
    return out, state


@pytest.mark.parametrize("steps", [1, 2, 7, 64, 129])
@pytest.mark.parametrize("heads", [4, 32])
def test_a_chunk_is_the_positions_it_contains(steps, heads):
    q, k, v, g, beta = _inputs(steps, heads, HEAD_DIM, seed=steps + heads)
    want_out, want_state = _stepwise(q, k, v, g, beta, heads, HEAD_DIM)
    got_out, got_state = _chunked(q, k, v, g, beta, heads, HEAD_DIM)

    scale = max(want_out.float().abs().max().item(), 1e-6)
    error = (got_out.float() - want_out.float()).abs().max().item() / scale
    assert error < 2e-2, f"outputs differ by {error:.3g}"

    scale = max(want_state.abs().max().item(), 1e-6)
    error = (got_state - want_state).abs().max().item() / scale
    assert error < 2e-2, f"state differs by {error:.3g}"


def test_a_chunk_continues_from_the_state_it_is_given():
    # A prompt longer than one chunk is several chunks, and the second has to
    # start where the first stopped. A kernel that ignored the incoming state
    # would pass every single-chunk test.
    heads, split = 8, 5
    q, k, v, g, beta = _inputs(12, heads, HEAD_DIM, seed=3)
    want_out, want_state = _stepwise(q, k, v, g, beta, heads, HEAD_DIM)

    first_out, state = _chunked(q[:split], k[:split], v[:split], g[:split],
                                beta[:split], heads, HEAD_DIM)
    second_out, state = _chunked(q[split:].contiguous(),
                                 k[split:].contiguous(),
                                 v[split:].contiguous(),
                                 g[split:].contiguous(),
                                 beta[split:].contiguous(),
                                 heads, HEAD_DIM, state=state)

    got = torch.cat([first_out, second_out])
    scale = max(want_out.float().abs().max().item(), 1e-6)
    assert (got.float() - want_out.float()).abs().max().item() / scale < 2e-2
    scale = max(want_state.abs().max().item(), 1e-6)
    assert (state - want_state).abs().max().item() / scale < 2e-2


def test_the_state_it_leaves_decides_the_next_token():
    # The output of a prompt is discarded except for its last row; what is
    # kept is the state. A kernel with the right outputs and the wrong state
    # answers the prompt correctly and everything after it wrongly.
    heads = 16
    q, k, v, g, beta = _inputs(20, heads, HEAD_DIM, seed=11)
    _, stepwise_state = _stepwise(q, k, v, g, beta, heads, HEAD_DIM)
    _, chunked_state = _chunked(q, k, v, g, beta, heads, HEAD_DIM)

    # One further position from each state must agree.
    nq, nk, nv, ng, nbeta = _inputs(1, heads, HEAD_DIM, seed=12)
    after_stepwise, _ = _chunked(nq, nk, nv, ng, nbeta, heads, HEAD_DIM,
                                 state=stepwise_state.clone())
    after_chunked, _ = _chunked(nq, nk, nv, ng, nbeta, heads, HEAD_DIM,
                                state=chunked_state.clone())
    scale = max(after_stepwise.float().abs().max().item(), 1e-6)
    error = (after_chunked.float()
             - after_stepwise.float()).abs().max().item() / scale
    assert error < 2e-2, f"the next position differs by {error:.3g}"


def test_a_head_width_that_is_not_implemented_is_refused():
    buffer = torch.zeros(4096, dtype=torch.bfloat16, device=DEVICE)
    state = torch.zeros(64, dtype=torch.float32, device=DEVICE)
    rc = fvk.linear_attn_recurrent_chunk_f32state_bf16(
        buffer.data_ptr(), buffer.data_ptr(), buffer.data_ptr(),
        buffer.data_ptr(), buffer.data_ptr(), state.data_ptr(),
        buffer.data_ptr(), 1, 2, 96, 96, True,
        torch.cuda.current_stream(DEVICE).cuda_stream)
    assert rc == -1

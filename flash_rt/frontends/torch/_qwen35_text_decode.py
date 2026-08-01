"""The decode step's arithmetic, issued as kernels against fixed addresses.

Every buffer a step needs is allocated once, by :class:`Workspace`, and the
step itself moves integers. Nothing here builds a tensor, takes a slice or asks
for a shape, because those are dispatches and a step that does forty of them
per layer spends more on asking than on arithmetic -- measured on a sibling
model as 4274 dispatched operators for one token, more of the step than either
the kernels or the storage it was blamed on.

That discipline is also what makes the step capturable: a graph replays the
addresses it was captured with, so a path that allocates per call has nothing
stable to capture. The position and the sequence length live on the device for
the same reason -- passed by value they would be baked into the capture, and
every replayed token would attend to the cache the capture saw.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from flash_rt.frontends.torch._qwen35_text_weights import TextWeights


@dataclass
class _Buffer:
    """A device allocation and the address the kernels are given."""

    tensor: torch.Tensor
    address: int

    @classmethod
    def make(cls, *shape: int, dtype=torch.bfloat16,
             device: str = "cuda:0") -> "_Buffer":
        tensor = torch.empty(*shape, dtype=dtype, device=device)
        return cls(tensor=tensor, address=int(tensor.data_ptr()))


def _rotary_tables(positions: int, rope_dim: int, theta: float,
                   device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Cosine and sine for every position the workspace can reach.

    One entry per pair rather than per dimension: the second half of a
    rotation turns through the same angle as the first, so storing it twice
    would double a table that is read once per head per step.
    """
    index = torch.arange(0, rope_dim, 2, dtype=torch.float32, device=device)
    frequency = 1.0 / (theta ** (index / rope_dim))
    angle = torch.arange(positions, dtype=torch.float32,
                         device=device)[:, None] * frequency[None, :]
    return angle.cos().to(torch.bfloat16), angle.sin().to(torch.bfloat16)


class Workspace:
    """Every scratch buffer a decode step uses, allocated once.

    Sized from the geometry rather than from the first call, so the addresses
    exist before anything runs and do not move afterwards.
    """

    def __init__(self, weights: TextWeights, device: str = "cuda:0",
                 max_chunk: int = 64, max_seq: int = 4096):
        dims = weights.dims
        self.device = device
        self.dims = dims
        # Every buffer between the embedding and the output projection is
        # sized for this many positions, and a prompt longer than it is read
        # in several passes. The alternative is to size them for the longest
        # prompt, which for the widest of them is tens of megabytes standing
        # idle through every decode step that follows.
        self.max_chunk = max_chunk
        self.group_size = weights.group_size

        self.residual = _Buffer.make(max_chunk, dims.hidden, device=device)
        self.normed = _Buffer.make(max_chunk, dims.hidden, device=device)
        self.mixed = _Buffer.make(max_chunk, dims.hidden, device=device)

        # A projection's output is read by a kernel that knows the width of
        # one row, so each fused projection gets a buffer of exactly its own
        # width. Sharing one wide buffer would work at a single row and read
        # every row after the first from the wrong offset -- which is the
        # kind of mistake that produces a fluent model with no memory.
        self.attn_fused = _Buffer.make(
            max_chunk, dims.q_width + 2 * dims.kv_width, device=device)
        # The recurrence's projection is fused at one row, where its consumer
        # cannot tell, and split at many, where it can.
        self.lin_fused = _Buffer.make(
            1, dims.lin_qkv_width + dims.lin_value_width, device=device)
        self.lin_qkv = _Buffer.make(max_chunk, dims.lin_qkv_width,
                                    device=device)
        self.lin_z = _Buffer.make(max_chunk, dims.lin_value_width,
                                  device=device)
        self.mlp_gate = _Buffer.make(max_chunk, dims.intermediate,
                                     device=device)
        self.mlp_up = _Buffer.make(max_chunk, dims.intermediate, device=device)
        self.gated = _Buffer.make(max_chunk, dims.intermediate, device=device)
        self.mixer_out = _Buffer.make(
            max_chunk, max(dims.attn_width, dims.lin_value_width),
            device=device)
        # One row: a prompt is read for its effect on the state, and only the
        # position after it has a distribution anyone reads.
        self.logits = _Buffer.make(1, dims.vocab_size, device=device)
        # The tokens of the current pass, kept on the device so a greedy step
        # never has to come back to the host between tokens.
        self.token = _Buffer.make(max_chunk, dtype=torch.int64, device=device)

        # The recurrence's staging: the convolved stream, the two narrow
        # projections that set the decay, and the per-head views the
        # recurrence reads.
        self.conv_out = _Buffer.make(max_chunk, dims.lin_qkv_width,
                                     device=device)
        self.decay = _Buffer.make(max_chunk, 2 * dims.lin_value_heads,
                                  device=device)
        self.lin_q = _Buffer.make(max_chunk, dims.lin_value_heads,
                                  dims.lin_key_head_dim, device=device)
        self.lin_k = _Buffer.make(max_chunk, dims.lin_value_heads,
                                  dims.lin_key_head_dim, device=device)
        self.lin_v = _Buffer.make(max_chunk, dims.lin_value_heads,
                                  dims.lin_value_head_dim, device=device)
        self.lin_g = _Buffer.make(max_chunk, dims.lin_value_heads,
                                  device=device)
        self.lin_beta = _Buffer.make(max_chunk, dims.lin_value_heads,
                                     device=device)

        # Full attention's staging.
        self.query = _Buffer.make(max_chunk, dims.q_heads, dims.head_dim,
                                  device=device)
        self.attn_gate = _Buffer.make(max_chunk, dims.attn_width,
                                      device=device)
        self.cos, self.sin = _rotary_tables(
            max_seq, dims.rotary_dim, dims.rope_theta, device)
        self.cos_address = int(self.cos.data_ptr())
        self.sin_address = int(self.sin.data_ptr())

        # Where the sequence is, as two numbers the kernels read rather than
        # arguments they are given: the position the next row is written to,
        # and how much of the cache is valid once it has been. They advance
        # together, so they share one allocation and one increment.
        self.cursor = _Buffer.make(2, dtype=torch.int32, device=device)
        self.position_address = self.cursor.address
        self.length_address = self.cursor.address + 4

        # Per-layer state, allocated with the workspace because it is the
        # same kind of thing: an address the step writes into that has to
        # outlive the call and never move.
        #
        # The two regimes carry state of different shapes, and the difference
        # is the reason this model suits a device like this. A recurrence
        # holds a fixed amount however long the context gets; only the full
        # attention layers grow with it, and there are a quarter as many of
        # those. A prompt twice as long costs twice the KV of eight layers,
        # not of thirty-two.
        self.recurrent = [
            _Buffer.make(dims.lin_value_heads, dims.lin_key_head_dim,
                         dims.lin_value_head_dim, dtype=torch.float32,
                         device=device)
            for _ in dims.linear_attention_layers
        ]
        self.conv = [
            _Buffer.make(1, dims.lin_qkv_width, dims.lin_conv_kernel - 1,
                         device=device)
            for _ in dims.linear_attention_layers
        ]
        self.keys = [
            _Buffer.make(max_seq, dims.kv_heads, dims.head_dim, device=device)
            for _ in dims.full_attention_layers
        ]
        self.values = [
            _Buffer.make(max_seq, dims.kv_heads, dims.head_dim, device=device)
            for _ in dims.full_attention_layers
        ]
        # Which slot of the two lists above a layer index belongs to, resolved
        # once so the step does not count layer types while it runs.
        self.state_slot = {
            layer: rank
            for rank, layer in enumerate(dims.linear_attention_layers)
        }
        self.state_slot.update({
            layer: rank
            for rank, layer in enumerate(dims.full_attention_layers)
        })
        self.max_seq = max_seq
        self.state_bytes = sum(
            buffer.tensor.numel() * buffer.tensor.element_size()
            for group in (self.recurrent, self.conv, self.keys, self.values)
            for buffer in group)
        self.reset()

    def reset(self) -> None:
        """Forget the sequence, keep the addresses.

        A control loop starts a new sequence often, and re-allocating would
        move every address a captured graph was built around. Zeroing is what
        starting over means here. The key and value caches are left alone
        because nothing reads past the cursor.
        """
        for group in (self.recurrent, self.conv):
            for buffer in group:
                buffer.tensor.zero_()
        self.seek(0)

    def seek(self, position: int, rows: int = 1) -> None:
        """Place the cursor for a pass of ``rows`` positions.

        The pair is where the next row is written and how much of the cache
        is valid once every row of the pass has been, which for a prompt is
        not one apart.
        """
        self.cursor.tensor[0] = position
        self.cursor.tensor[1] = position + rows

    def close(self) -> None:
        for name in ("residual", "normed", "mixed", "attn_fused", "lin_fused",
                     "lin_qkv", "lin_z", "mlp_gate", "mlp_up", "gated",
                     "mixer_out", "logits", "token", "conv_out", "decay",
                     "lin_q", "lin_k", "lin_v", "lin_g", "lin_beta",
                     "query", "attn_gate", "cursor"):
            setattr(self, name, None)
        self.cos = self.sin = None
        self.recurrent = []
        self.conv = []
        self.keys = []
        self.values = []
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _element_size(width: int) -> int:
    """Bytes per row of a bf16 buffer of this width."""
    return width * 2


def _check(rc, what: str) -> None:
    """Refuse to continue past a kernel that reported a bad geometry."""
    if rc:
        raise RuntimeError(f"{what} failed with {rc}")


def mlp_block(layer: dict[str, int], work: Workspace, fvk, x: int, out: int,
              rows: int, stream: int) -> None:
    """out = W_down * (silu(gate) * up), with gate and up one weight.

    At one row this is two launches: the projection applies its own gate, and
    the contraction follows. The prompt pass keeps the three-launch form,
    which needs the elementwise kernel -- at many rows the intermediate is
    large enough that a separate pass is not obviously worse, and the batched
    projection has no gated variant yet.

    ``x`` and ``out`` are addresses, and may be the same buffer only if the
    caller means them to be.
    """
    intermediate = layer["gate_up_up_offset"]

    if rows == 1:
        # The gate is applied inside the projection: a warp owns a row and its
        # partner half a weight away and combines them in registers. One
        # launch rather than two, and no dependence on an elementwise kernel
        # that lives in a model-specific build tier.
        _check(fvk.w4a16_packed_matvec_gated_bf16(
            x, layer["gate_up_packed"], layer["gate_up_scale"],
            work.gated.address, layer["gate_up_n"], layer["gate_up_k"],
            work.group_size, stream), "gated gate/up projection")
        _check(fvk.w4a16_packed_matvec_bf16(
            work.gated.address, layer["down_packed"], layer["down_scale"],
            out, layer["down_n"], layer["down_k"], work.group_size, stream),
            "down projection")
        return

    # At many rows the two halves go to buffers of their own, because the
    # elementwise kernel that follows walks them as flat arrays and the
    # halves of a fused row are not adjacent across rows.
    _project_part(fvk, work, x, layer, "gate_up", "gate", work.mlp_gate.address,
                  rows, stream, "gate projection")
    _project_part(fvk, work, x, layer, "gate_up", "up", work.mlp_up.address,
                  rows, stream, "up projection")
    _check(fvk.silu_mul_qwen36_bf16(
        work.mlp_gate.address, work.mlp_up.address, work.gated.address,
        rows * intermediate, stream), "gated product")
    _check(fvk.w4a16_packed_gemm_bf16(
        work.gated.address, layer["down_packed"], layer["down_scale"],
        out, rows, layer["down_n"], layer["down_k"], work.group_size, stream),
        "down projection")


def _project(fvk, work: Workspace, x: int, layer: dict[str, int], prefix: str,
             out: int, rows: int, stream: int, what: str) -> None:
    """One packed projection, at whichever width the caller is running."""
    if rows == 1:
        _check(fvk.w4a16_packed_matvec_bf16(
            x, layer[prefix + "_packed"], layer[prefix + "_scale"], out,
            layer[prefix + "_n"], layer[prefix + "_k"], work.group_size,
            stream), what)
    else:
        _check(fvk.w4a16_packed_gemm_bf16(
            x, layer[prefix + "_packed"], layer[prefix + "_scale"], out, rows,
            layer[prefix + "_n"], layer[prefix + "_k"], work.group_size,
            stream), what)


def _project_part(fvk, work: Workspace, x: int, layer: dict[str, int],
                  prefix: str, short: str, out: int, rows: int, stream: int,
                  what: str) -> None:
    """One member of a fused projection, without unfusing the weight.

    Rows are independent in this layout, so a member is a row range: the
    weight and its scales are read from where that member starts. Fusing was
    for the single-row case, where it is one launch instead of several; here
    it costs nothing to take back.
    """
    columns = layer[prefix + "_k"]
    offset = layer[f"{prefix}_{short}_offset"]
    _check(fvk.w4a16_packed_gemm_bf16(
        x,
        layer[prefix + "_packed"] + offset * (columns // 8) * 4,
        layer[prefix + "_scale"] + offset * (columns // work.group_size) * 2,
        out, rows, layer[f"{prefix}_{short}_rows"], columns, work.group_size,
        stream), what)


def linear_attention_block(layer: dict[str, int], work: Workspace, fvk,
                           slot: int, x: int, out: int, rows: int,
                           stream: int) -> None:
    """The gated-delta recurrence, from its projection to its output.

    Seven launches, of which three read weights. The rest move a few tens of
    kilobytes each and exist because the recurrence wants its input in a
    different shape than the projection produces it -- which is why the
    split, the head broadcast and both gates are one kernel and not four.
    """
    dims = work.dims
    if rows == 1:
        # One launch for a projection whose two halves go to different
        # places: at a single row nothing downstream can tell they share a
        # buffer, and the weight is read once instead of twice.
        _project(fvk, work, x, layer, "in_proj", work.lin_fused.address, 1,
                 stream, "recurrence input projection")
        convolved_in = work.lin_fused.address
        output_gate = work.lin_fused.address + _element_size(
            dims.lin_qkv_width)
    else:
        _project_part(fvk, work, x, layer, "in_proj", "qkv",
                      work.lin_qkv.address, rows, stream,
                      "recurrence input projection")
        _project_part(fvk, work, x, layer, "in_proj", "z",
                      work.lin_z.address, rows, stream,
                      "recurrence gate projection")
        convolved_in = work.lin_qkv.address
        output_gate = work.lin_z.address

    # The two decay projections are left uncompressed by the producer and are
    # narrow enough that reading them is the whole cost either way.
    if rows == 1:
        _check(fvk.bf16_matvec_qwen36_bf16(
            x, layer["in_ab"], work.decay.address, layer["in_ab_n"],
            layer["in_ab_k"], stream), "decay projection")
    else:
        _check(fvk.bf16_matmul_bf16(
            x, layer["in_ab"], work.decay.address, rows, layer["in_ab_n"],
            layer["in_ab_k"], stream), "decay projection")

    conv_state = work.conv[slot].address
    if rows == 1:
        fvk.causal_conv1d_qwen36_update_bf16(
            convolved_in, layer["conv"], 0, work.conv_out.address,
            conv_state, 1, layer["conv_dim"], layer["conv_k"], True, stream)
    else:
        fvk.causal_conv1d_qwen36_update_chunk_bf16(
            convolved_in, layer["conv"], 0, work.conv_out.address,
            conv_state, 1, rows, layer["conv_dim"], layer["conv_k"], True,
            stream)

    _check(fvk.linear_attn_split_broadcast_gate_bf16(
        work.conv_out.address, work.decay.address,
        work.decay.address + _element_size(dims.lin_value_heads),
        layer["neg_exp_a_log"], layer["dt_bias"],
        work.lin_q.address, work.lin_k.address, work.lin_v.address,
        work.lin_g.address, work.lin_beta.address,
        rows, dims.lin_key_heads, dims.lin_value_heads,
        dims.lin_key_head_dim, dims.lin_value_head_dim,
        2 * dims.lin_value_heads, 2 * dims.lin_value_heads, stream),
        "recurrence staging")

    # The state is kept in float32. It is multiplied by a decay every step and
    # never re-derived, so rounding it each time is a drift that runs for as
    # long as the sequence does -- and a prompt and the tokens after it have
    # to see the same state, which is why a prompt is stepped through this
    # kernel rather than handed to the chunked one, whose state is bfloat16.
    state = work.recurrent[slot].address
    if rows == 1:
        fvk.gated_deltanet_recurrent_qwen36_f32state_bf16io(
            work.lin_q.address, work.lin_k.address, work.lin_v.address,
            work.lin_g.address, work.lin_beta.address, state,
            work.mixer_out.address, 1, dims.lin_value_heads,
            dims.lin_key_head_dim, dims.lin_value_head_dim, True, stream)
    else:
        # The recurrence is sequential and cannot be widened, but it does not
        # have to be re-entered: a thread keeps its column of the state in
        # registers across the whole chunk. A prompt otherwise costs a launch
        # per position per layer, which for a few hundred positions outweighs
        # the arithmetic and is most of the time to the first token.
        _check(fvk.linear_attn_recurrent_chunk_f32state_bf16(
            work.lin_q.address, work.lin_k.address, work.lin_v.address,
            work.lin_g.address, work.lin_beta.address, state,
            work.mixer_out.address, rows, dims.lin_value_heads,
            dims.lin_key_head_dim, dims.lin_value_head_dim, True, stream),
            "gated-delta recurrence")

    # The output gate is the other half of the input projection, so it is read
    # where it already is rather than copied out first.
    fvk.rms_norm_gated_silu_qwen36_bf16(
        work.mixer_out.address, output_gate,
        layer["gdn_norm"], work.mixer_out.address,
        rows * dims.lin_value_heads, dims.lin_value_head_dim,
        dims.rms_norm_eps, stream)

    _project(fvk, work, work.mixer_out.address, layer, "out", out, rows,
             stream, "recurrence output projection")


def full_attention_block(layer: dict[str, int], work: Workspace, fvk,
                         slot: int, x: int, out: int, rows: int,
                         stream: int) -> None:
    """Full attention, in four launches.

    Two read weights and two do everything else: the staging kernel turns the
    fused projection into a query, a gate and a cache row, and the attention
    reads the cache and applies the gate where the result is produced.
    """
    dims = work.dims
    _project(fvk, work, x, layer, "qkv", work.attn_fused.address, rows,
             stream, "attention input projection")

    _check(fvk.attn_qkv_norm_rope_write_bf16(
        work.attn_fused.address, layer["q_norm"], layer["k_norm"],
        work.cos_address, work.sin_address,
        work.query.address, work.attn_gate.address,
        work.keys[slot].address, work.values[slot].address,
        rows, 0, work.position_address,
        dims.q_heads, dims.kv_heads, dims.head_dim, dims.rotary_dim,
        dims.attn_output_gate, dims.rms_norm_eps, stream),
        "attention staging")

    _check(fvk.gqa_decode_attention_bf16(
        work.query.address, work.keys[slot].address, work.values[slot].address,
        work.attn_gate.address if dims.attn_output_gate else 0,
        work.mixer_out.address, 0, work.length_address,
        dims.q_heads, dims.kv_heads, dims.head_dim,
        1.0 / math.sqrt(dims.head_dim), rows, stream), "attention")

    _project(fvk, work, work.mixer_out.address, layer, "o", out, rows, stream,
             "attention output projection")


def forward(weights: TextWeights, work: Workspace, fvk, rows: int,
            stream: int) -> None:
    """Embedding to the last norm, for ``rows`` positions at the cursor.

    The residual and the norm after it are one kernel: the add writes the
    running hidden state back where it was and normalizes it into the buffer
    the next projection reads. A layer therefore costs two of those rather
    than four launches, and the norm a layer opens with is the one the layer
    before it closed with -- including the last, which closes with the final
    norm instead.
    """
    dims = work.dims
    eps = dims.rms_norm_eps
    hidden = dims.hidden

    fvk.embedding_lookup_bf16(
        work.token.address, weights.top["embed"], work.residual.address,
        rows, hidden, stream)
    fvk.rms_norm(work.residual.address, weights.layers[0]["input_norm"],
                 work.normed.address, rows, hidden, eps, stream)

    last = dims.num_layers - 1
    for index, layer in enumerate(weights.layers):
        slot = work.state_slot[index]
        block = (linear_attention_block if layer["linear_attention"]
                 else full_attention_block)
        block(layer, work, fvk, slot, work.normed.address,
              work.mixed.address, rows, stream)
        fvk.residual_add_rms_norm(
            work.residual.address, work.mixed.address, layer["post_norm"],
            work.normed.address, rows, hidden, eps, stream)

        mlp_block(layer, work, fvk, work.normed.address, work.mixed.address,
                  rows, stream)
        following = (weights.top["final_norm"] if index == last
                     else weights.layers[index + 1]["input_norm"])
        fvk.residual_add_rms_norm(
            work.residual.address, work.mixed.address, following,
            work.normed.address, rows, hidden, eps, stream)


def project_to_vocabulary(weights: TextWeights, work: Workspace, fvk,
                          stream: int, row: int = 0) -> None:
    """The output projection, for one row, in whichever form the table has.

    One row because that is all anyone reads: a prompt is taken for its effect
    on the state, and the distribution that matters is the one at the position
    after it. Projecting the rest would be the largest read in the step done
    once per prompt token for nothing.
    """
    top = weights.top
    source = work.normed.address + row * _element_size(top["hidden"])
    if top.get("lm_head_scale"):
        _check(fvk.w8a16_rowwise_matvec_bf16(
            source, top["lm_head"], top["lm_head_scale"],
            work.logits.address, top["vocab_size"], top["hidden"], stream),
            "output projection")
    else:
        fvk.bf16_matvec_qwen36_bf16(
            source, top["lm_head"], work.logits.address,
            top["vocab_size"], top["hidden"], stream)


def decode_step(weights: TextWeights, work: Workspace, fvk,
                stream: int) -> None:
    """One token: forward, project to the vocabulary, take the argmax.

    Nothing here returns to the host. The token the step chooses is written
    where the next step's embedding lookup reads it, and the cursor advances
    on the device, so a captured graph is a whole step rather than a prefix
    of one.
    """
    forward(weights, work, fvk, 1, stream)
    project_to_vocabulary(weights, work, fvk, stream)
    fvk.qwen36_argmax_bf16(work.logits.address, work.token.address, 1,
                           weights.top["vocab_size"], stream)
    # The position and the length advance together, on the device, which is
    # what makes a replayed graph attend to the row it has just written
    # rather than to the row the capture saw.
    work.cursor.tensor.add_(1)

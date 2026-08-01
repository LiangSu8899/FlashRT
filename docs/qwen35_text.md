# Running a `qwen3_5_text` checkpoint

A hybrid model: three quarters of its layers are a gated-delta recurrence and
one quarter is full attention. The recurrence carries a fixed amount of state
however long the context gets, so only the attention layers grow with it and
there are a quarter as many of those — a prompt twice as long costs twice the
key/value cache of eight layers, not of thirty-two. That is what makes the
family suit a part with little memory.

The weights are read in the layout the checkpoint publishes them in
(`pack-quantized`, four bits, one scale per group along K). Nothing is
dequantized or repacked at load, so what is resident is the checkpoint's own
size.

## Building

The kernels this needs are on by default and use SM80-family facilities only:

```
cmake -B build -DGPU_ARCH=<arch>
cmake --build build -j<jobs>
```

Two options cover them, both `ON` by default:

* `FLASHRT_ENABLE_W4A16_PACKED` — the packed 4-bit and per-channel INT8
  weight-only kernels.
* `FLASHRT_ENABLE_DECODE_ATTENTION` — the staging, attention and recurrence
  kernels a decode step needs.

Confirm they are present before running; a missing kernel is refused at load
rather than silently skipped:

```python
from flash_rt import flash_rt_kernels as fvk
assert hasattr(fvk, "w4a16_packed_matvec_bf16")
assert hasattr(fvk, "gqa_decode_attention_bf16")
```

## Running

```python
from flash_rt.frontends.torch.qwen35_text import TextRuntime, StepTimings

runtime = TextRuntime.from_checkpoint(path, max_seq=2048, max_chunk=64)
runtime.reset()
runtime.read_prompt(prompt_token_ids)
runtime.capture()                      # optional: capture the decode step

timings = StepTimings()
tokens = runtime.generate(prompt_token_ids, max_new_tokens=64,
                          stop_ids=(eos,), timings=timings)
print(timings.summary())
```

`max_chunk` is how many positions the prompt is read in at once, and sizes
every buffer between the embedding and the output projection. `max_seq` sizes
the key/value cache and the rotary tables.

Capturing the decode step is worth more than the launch count suggests: a step
is around three hundred small launches, and where a launch costs microseconds
that is a fixed tax on every token whether or not the part is busy. Capture is
possible because the step touches nothing that moves — the position and the
sequence length are read from device memory rather than passed in, and the
token a step chooses is written where the next step's embedding lookup reads
it.

## Measuring

```
python benchmarks/qwen35_text_latency.py --checkpoint <path>
python benchmarks/qwen35_text_latency.py --checkpoint <path> --no-graph
```

Time to the first token and the tokens after it are reported separately
because they are limited by different things, and the per-token figure is a
distribution rather than a mean: a loop that usually meets its deadline and
occasionally does not is a loop that misses.

## Things that are silent when wrong

Four properties of this family produce a model that still emits fluent text
when they are got wrong, which is a long way to trace back from. Three are
handled at load and the fourth is refused:

* **The plain RMSNorm scales by `1 + weight`,** and the parameter is stored
  centred on zero — several entries are negative, which a scale never is. The
  one is added at load. Without it the hidden state is scaled towards nothing
  at every layer. The *gated* norm inside the recurrence is an ordinary scale
  and must not be offset.
* **The packed nibbles are offset binary, not two's complement.** The
  subtraction of eight is part of the format rather than a normalization.
* **The decay is used as `-exp(A_log)` and never as `A_log`,** in float32: it
  multiplies a state that is never re-derived, and in bfloat16 a long memory
  and a permanent one round to the same number.
* **The recurrence kernel implements one head width** and returns without
  doing anything for any other, leaving its output holding whatever was there
  before. That width is checked at load and a checkpoint with another one is
  refused.

## The tied table

At a 248k vocabulary the embedding table is 1.2 GiB of bfloat16 and, being
tied, it is read twice per token -- once as a lookup and once as the output
projection. It is 39% of everything a token reads, more than the whole
gated-delta stack, and unlike the backbone it arrives unquantized.

`quantize_tied_table=True` (or `--int8-head`) stores it as per-row INT8 with
one scale per row. Both readings quantize together, because one tensor read
two ways with a scale applied to only one direction embeds plausibly and
predicts badly.

Measured on a 4B checkpoint: 3.061 -> 2.469 GiB resident, and a decode step
from 4.49 to 3.74 ms on an RTX 5090 -- a 20% saving against a 19% reduction in
bytes, which is what a step limited by its weight read should give. It is
opt-in because it is not free: over a passage of ordinary text the greedy
token agreed with the bfloat16 table at 43 of 44 positions, and the mean rank
of the true continuation moved from 334 to 337.

## Where the time goes

At batch one the weight read is the whole cost and the fraction of bandwidth
reached is the whole result. Two tools report it:

```
python benchmarks/w4a16_packed_roofline.py          # per weight shape
python benchmarks/qwen35_text_profile.py --checkpoint <path> --verify
```

The first times each projection shape on its own against a bandwidth measured
on the same part. The second attributes a whole step: the bytes each piece
accounts for beside the time it takes, so a piece whose share of the time is
larger than its share of the bytes is where the work is, and the ratio says
whether the answer is a better kernel or fewer bytes.

`--verify` is the check to run after changing a kernel: it reports how far
down the distribution the true continuation of a passage sits. A model that is
subtly wrong still emits fluent tokens and every timing still looks right, so
this is the measurement that does not.

### Before measuring anything on an embedded part

Check what power mode it is in. A Jetson defaults to a mode well below its
capability, and every number taken in it is a number about the mode:

```
sudo nvpmodel -q            # which mode
sudo nvpmodel -m <max>      # MAXN, or MAXN_SUPER where it exists
sudo jetson_clocks          # hold the clocks there
```

Measured on an eight-multiprocessor Orin: a device-to-device copy went from
51.2 to 85.9 GB/s and the projections a token issues from 39.9 to 35.1 ms,
with the part 3 degrees warmer. It also changes what the limit *is* -- in the
lower mode the projections ran at 105% of the copy bandwidth and were bound by
memory, in the higher one at 72% and were bound by everything else. A kernel
tuned in the wrong mode is tuned for the wrong problem.

### What the tile size is really keyed on

Every block stages the whole activation before it reads any weight, so that
work is proportional to the number of blocks -- which is inversely
proportional to the rows a warp takes. Which of those dominates depends on the
part and not on the shape:

* On eight multiprocessors, the widest contraction went from 47 to 72 GB/s
  between one row per warp and eight, its activation being staged by forty
  blocks instead of three hundred and twenty.
* On a hundred and seventy, one row measured fastest for most shapes: eight
  leaves too few blocks to fill the part.

`benchmarks/w4a16_packed_sweep.py` measures it; `FLASHRT_W4A16_ROWS` overrides
it without a rebuild.

### An integer dot product does not pay here

The bfloat16 path spends about seven arithmetic instructions per byte of
weight. Quantizing the activation to int8 lets a word of weight become two
`dp4a` instructions -- five per four bytes rather than twenty-eight -- and on
a part whose arithmetic pipeline is the limit that should be most of the gap.

Measured on the eight-multiprocessor part in its full power mode, it is
slower: 38.2 ms against 31.0 for the projections, at every tile size tried,
and worst on the longest contraction. The saving in the inner loop is smaller
than the cost of the staging it needs -- a maximum over each group, a rounding
and a repacking, where the bfloat16 path only copies. The arithmetic was the
limit in the *lower* power mode; in the higher one it is not, and the change
that follows from the first measurement is the wrong change for the second.

Relative error against the bfloat16 path was 0.005 across the shapes, so this
is a statement about speed and not about accuracy.

### The ceiling

A token reads 2528 MiB with the tied table in INT8, and cannot read much less:
the backbone is already four bits and the table is already one byte. So the
rate is decided by the part:

    tokens per second = bandwidth / 2.651 GB

That puts a hard ceiling on single-token decoding at whatever the part reads
at, and the only way past it is to make one read of the weights produce more
than one token. The checkpoint carries the means to do that: a 76 MiB
one-layer next-token head under `mtp.`, which drafts and lets the main model
verify several positions per pass. Its cost is dominated not by its own layer
but by the vocabulary projection each draft position needs, so a draft head
restricted to the frequent part of the vocabulary is most of what makes it
worth doing.

### A prompt that is mostly the same every turn

An agent's prompt is a system prompt, a set of tool definitions and some
documents, followed by a short tail that changes. Reading the unchanged part
again costs what it cost the first time, and for a long prefix that is most of
the time to the first token.

What a prefix leaves behind can be kept. For three quarters of the layers it
is a fixed size however long the prefix is -- a recurrence carries the same
state for ten tokens as for ten thousand -- and only the eight full-attention
layers keep something proportional:

```python
runtime.read_prompt(system_prompt_and_tools)
prefix = runtime.snapshot()
...
runtime.restore(prefix)          # addresses do not move; a graph stays valid
runtime.read_suffix(this_turn)
```

Measured with a 1024-token prefix and a 48-token turn: 1014 ms to read the
whole prompt each time against 47.6 ms to restore and read the tail, for the
same first token. The snapshot was 81 MiB.

The restore itself is 0.2 ms, so what a turn costs is its own tokens.

### The prompt pass

It does not yet get the benefit batching should give it: the batched
projection reads the weight once per activation row, so a prompt read sixteen
positions at a time costs nearly what sixteen separate positions would. Time
to the first token is close to linear in the prompt length for that reason,
and tiling the activation rows inside that kernel is the change that would fix
it. On a part with a large last-level cache this is partly hidden; on a small
one it is not.

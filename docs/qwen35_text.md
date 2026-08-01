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

The prompt pass does not yet get the benefit batching should give it: the
batched projection reads the weight once per activation row, so a prompt read
sixteen positions at a time costs nearly what sixteen separate positions
would. Time to the first token is close to linear in the prompt length for
that reason, and tiling the activation rows inside that kernel is the change
that would fix it.

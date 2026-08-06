# The explicit structure pipeline — GR00T N1.7

This folder is the **explicit** way into the structures layer: the
author declares every seat, runs their own calibration hooks, and calls
each structure family's binder directly. It is the hand-written
counterpart of the three-line automatic path, and both are measured
here on two different GR00T N1.7 hosts — the official Isaac-GR00T
repository and the LeRobot port — same checkpoint, same prepared input
tensors, same timing protocol.

```
groot_n17.py    the explicit assembly: seat tables, calibration hooks,
                per-family binder calls, attach, parity, eager+compiled
                timing with both baselines
full_graph.py   the same assembly at full speed: fixed-shape host
                lowering + whole-graph CUDA capture (the deployed form)
```

## Explicit versus the three-line automatic path

The automatic path is:

```python
plan = structures.auto_swaps(model, run_once)
handle = swap.attach(model, plan.swaps, observe=plan.observed,
                     revert=plan.revert)
```

Both paths produce the same kind of object — a set of bound structure
implementations attached onto the host's module tree, hot-pluggable,
guarded per seam. They differ in **who makes each decision**, not in
what can be reached:

| decision | automatic | explicit (this folder) |
|---|---|---|
| which modules are seats | discovery walks the model | the author's seat tables, path by path |
| calibration | one instrumented pass, library-owned points | the author's own hooks |
| qualification | work bands, shape envelopes, library judgment | the author's judgment — plus any runtime check they write |
| producer negotiation (e.g. FP8 norm → packed QKV) | resolved internally | written out as code: the norm binder and the pack binder share one scale tensor |
| a seat the library would refuse | stays refused | the author may claim it anyway and answer to the parity gate |
| safety net | per-seam guards; refusals recorded; production mode falls back to the host per call | identical — the same guards arm at bind time |

The practical meaning of the last rows showed up in this very
measurement. On the LeRobot host the automatic path binds a
`qkv_rope` seam whose cos/sin contract drifts at run time; the guard
catches it on every call, falls back to the host, and the ledger
reports 2568 fallbacks — while parity holds at 0.9999. The explicit
assembly never declared that seat, so it pays neither the drift nor
the fallback overhead. One path is armor, the other is aim.

Neither path is faster by construction: they bind the same
implementation layer, and at the captured form they land on the same
number. Explicit control matters where automatic qualification is
conservative — a specialist seat the discovery cannot prove safe, a
cadence the author knows (observation-rate cross-attention K/V versus
denoise-rate compute), a scheme choice per seat.

## Measured — RTX 5090, one checkpoint, one input set, one protocol

GR00T N1.7 has two hosts — the official Isaac-GR00T repository and the
LeRobot port. Both are measured on the same prepared model-level
inputs (exported once, loaded by both), pinned noise, median of
interleaved rounds, and all three execution forms on each host.
Parity is the treated output against that host's own untouched eager
run; each arm re-measures the stock graph in its own process (24 ±3%
across runs — read speedups, not cross-run milliseconds). The **same `build()` — every seat path, every binder call — ran
on both hosts without a single edit**: the LeRobot port vendors the
official module layout, so the seat tables transfer verbatim.

**Official Isaac-GR00T host:**

| form | host | automatic (3 lines) | explicit (this folder) |
|---|---|---|---|
| eager | 45.7 ms | 50.2 ms | 56.7 ms |
| compiled | 33.9 ms | — | 30.1 ms |
| **captured** | 23.4 ms | **14.79 ms** (1.58×, 285 seams) | **15.87 ms** (1.48×, 216 seats) |
| parity | — | 0.99990 | 0.99988 |

**LeRobot host:**

| form | host | automatic (3 lines) | explicit (this folder) |
|---|---|---|---|
| eager | 42.0 ms | 53.6 ms | 51.9 ms |
| compiled | 33.7 ms | 31.0 ms | 29.7 ms |
| **captured** | ~24 ms | 18.24 ms (1.37×) | **17.02 ms** (1.42×) |
| parity | — | 0.99989 | 0.99987 |
| runtime fallbacks | — | 72 (`qkv_rope`, guarded, refused before any work) | 0 |

What the tables say:

1. **The explicit assembly lands in one band on both hosts**
   (15.87 / 17.02 ms) from one unchanged file, at 0.9999 parity on
   each — and the captured form is one mechanism call:
   `structures.capture(torch.compile(hot), model=model)`. The
   registered host-family adapter pins the shape glue and lists its
   pins in the stage certification.
2. **Speedups may only be read within a row.** Eager pays a per-call
   Python admission check at every guarded seam and is *slower* than
   the host — printed, not hidden. Captured is the deployed form,
   where guards and glue are paid once at capture time.
3. **Where every seam holds, the automatic path wins** — its 69
   extra seams buy 7% on the official host (14.79 vs 15.87). On the LeRobot host the automatic path binds a
   `qkv_rope` family whose rope tables can never satisfy the FP32
   contract on this host (see the loading note below); the guards
   refuse every call before any work is done and run the host path,
   parity holds, and the ledger says so. The explicit assembly never
   declared that seat: armor versus aim, measured.
4. **A host loading note that costs real precision**: the LeRobot
   recipe `from_pretrained(...).to(dtype=bf16)` casts *every* floating
   buffer — including the rotary `inv_freq` that transformers'
   dtype-aware loading deliberately keeps in FP32. On this host the
   vision rope tables are therefore born BF16; loading with
   `from_pretrained(..., dtype=bf16)` instead keeps them FP32. This is
   why the `qkv_rope` family refuses here, and it is a precision loss
   the host pays with or without structures.
5. Host coupling lives in the capture lowering, and the lowering
   lives in the library: the registered Qwen3-VL family adapter
   carries the probed branches for both transformers vision-contract
   generations and the LeRobot wrapper glue. Porting the measurement
   between the two hosts changed none of the assembly and none of the
   capture code — the same door served both.

## Running

```bash
# explicit assembly, eager + compiled ladder
python groot_n17.py \
  --host /path/to/Isaac-GR00T \
  --checkpoint /path/to/GR00T-N1.7-3B \
  --backbone-assets /path/to/backbone-config-assets \
  --fixture /path/to/observation_fixture.pt \
  --compile --report report.json

# the same assembly, captured (the deployed form)
python full_graph.py --host ... --checkpoint ... \
  --backbone-assets ... --fixture ...
```

`--fixture` is a saved observation dict (`{"inputs": {...}}`) from the
host's own preprocessing. Kernel packages resolve from the Hugging Face
Hub per host; offline, stage them under any directory laid out as
`<org>/<name>/build/<variant>/` and point the kernels resolver at it.

## Reading the report

- `seats_bound` / `refused` — every declared seat lands in exactly one;
  a seat served by the host is an outcome, a silently skipped seat is
  a bug.
- `attention_variants` — which family member serves this host and the
  recorded reasons the preferred members stepped aside (on an RTX 5090
  FA2 binds; on a host without the FA2 build the family resolves to
  FA4 from the same line of code).
- `kernel_unavailable` — packages this host asked for and could not
  get, original error preserved: "never shipped here" and "broken
  here" must stay distinguishable.
- `ledger.fallbacks` — nonzero means a calibration assumption did not
  hold at run time and the guards routed those calls back to the host.

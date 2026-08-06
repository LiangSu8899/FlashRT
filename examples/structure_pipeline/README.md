# The explicit structure pipeline

`structures.auto_swaps` discovers seams, calibrates them and picks each
seat's implementation. This folder is the other way in: **the author
declares every seat, runs their own calibration hooks, and calls each
structure family's binder directly.** Nothing is discovered and nothing
is matched — what runs is exactly what is written.

Both paths share the same two layers underneath: the structure
implementations (one module per *execution form*, never per device) and
`hub_kernel`, the single boundary where device differences enter. The
automatic path supplies policy — discovery, qualification, variant
arbitration. It does not supply capability: everything it can bind, this
file binds by hand.

## Why the same file runs on different hardware

An explicit pipeline stays portable exactly as long as it follows four
rules, and `groot_n17.py` is written as a demonstration of them:

1. **Bind attention through the family entry** — never a variant module.
   The family ranks its members precision-first; whichever package this
   host can serve binds, and each loser's reason rides along on the
   bound core (`_frt_variant_trail`). On an RTX 5090 that resolves to
   FA2; on Thor (where FA2 ships no build) it resolves to FA4 — same
   line of code.
2. **Every seat wraps its binder in
   `except (KernelUnavailable, ValueError)`** — a package this host
   cannot supply, or a shape the form does not serve, is an outcome to
   record, never a reason to abort the build.
3. **Seat ownership is declared, not inferred.** The qkv seats also
   carry a runtime qualification (all three siblings must consume one
   tensor this tick) that calibration verifies; a cross-attention or
   masked site simply never qualifies and stays with the host.
4. **Refusals and absent packages are printed.** A seat served by the
   host is a legitimate result; a seat silently skipped is a bug. The
   report carries `refusal_details` and `kernel_unavailable` verbatim.

Pinning a variant module directly (`from ...attention_core import
fa4_cute`) is the one pattern that breaks portability: it turns "this
host lacks the package" into a crash instead of a recorded refusal.

## What `groot_n17.py` assembles

| Seats | Family binder | Count |
|---|---|---|
| Vision tower MLPs | `vision_ffn.bind_mlp_seam` (FP8) | 24 |
| Language-model MLPs | `decoder_ffn.bind_mlp_seam` (FP8 SwiGLU) | 16 |
| DiT feed-forwards | `vision_ffn.bind_mlp_seam` (FP8 GELU) | 32 |
| Language + DiT QKV triples | `qkv_pack.bind_qkv_pack` | up to 36 |
| DiT adaptive norms | `adaln_producer.bind_adaln_producer` | 32 |
| Dense attention cores | `bind_dense_attention_best` (family) | ~20 |
| Cross-attention K/V | `cadence_static.bind_cross_attention_kv` | 16 |

Calibration is one forward pass with author-owned hooks: per-tensor
amax where a seat quantises, `(cond, style)` pairs where a producer
replays a conditioning table, and captured Q/K/V where the attention
family qualifies a site.

## Running

```bash
python groot_n17.py \
  --host            /path/to/Isaac-GR00T \
  --checkpoint      /path/to/GR00T-N1.7-3B \
  --backbone-assets /path/to/backbone-config-assets \
  --fixture         /path/to/observation_fixture.pt \
  --compile --report report.json
```

`--fixture` is a saved observation dict (`{"inputs": {...}}`) from the
host's own preprocessing. `--compile` adds a `torch.compile` timing of
the treated form; the full-graph CUDA-capture numbers quoted in the
qualification receipts additionally use a fixed-shape host lowering —
see the qualification probe for that harness.

Kernel packages resolve from the Hugging Face Hub per host; on an
offline machine, stage them under any directory laid out as
`<org>/<name>/build/<variant>/` and point the kernels resolver at it.

## Reading the numbers

The report prints **two baseline columns, always**: eager against eager
(the community convention) and compiled against compiled (the
production form). In eager the treated graph can come out *slower* than
the host — every guarded seam pays a per-call Python admission check,
and the FP8 producers trade fused-kernel launches for work the eager
host never did. That cost is real and the report shows it. The form a
deployment actually ships is the compiled one, and that is the pair a
speedup may be quoted from. Quoting treated-compiled against
baseline-eager is not a measurement.

Measured by this file on an RTX 5090 (216 seats bound, 0 runtime
fallbacks, parity cosine 0.9999 on fixed noise):

| form | host | explicit pipeline | pair |
|---|---|---|---|
| eager | 45.5 ms | 56.7 ms | 0.80× — the guard cost, shown |
| compiled | 33.9 ms | **30.1 ms** | **1.13×** |

The qualification receipts quote larger wins for the same model because
they run the full harness on top of this assembly — the per-head
QK-norm/RoPE family, a fixed-shape host lowering, and whole-graph CUDA
capture. This file is the assembly those numbers stand on, kept small
enough to read.

## Reading the report

- `seats_bound` / `refused` — every declared seat lands in exactly one.
- `attention_variants` — which family member serves this host, plus the
  recorded reasons the preferred members stepped aside.
- `kernel_unavailable` — packages this host asked for and could not
  get, with the original error preserved: "never shipped here" and
  "broken here" must stay distinguishable.
- `parity_cosine` — treated vs host output on the same fixed noise.
- The ledger's `fallbacks` must be 0: a guarded seam that fell back at
  run time means a calibration assumption did not hold.

# Structures

> **Target audience**: engineers adding a structure, adding a backend for
> an existing structure, or working out why an attachment refused.
>
> **TL;DR**
> - A structure is a **spec** (boundary, weight slots, calibration points,
>   gates) plus a plain-torch **reference** that is the gate's ground
>   truth. Implementations are separate and plural.
> - Three layers, split by what varies: the **spec** names positions, a
>   **binding** says where they sit on one host, an **impl** decides what
>   to do there. A backend change touches only the third.
> - Calibration is **not this layer's**. Points come from the spec, the
>   reduction is `flash_rt.core.calibration`, the receipt is
>   `ModelPrecisionSpec`, the argument names are
>   `flash_rt.api.FlashRT.calibrate`'s. See [`calibration.md`](calibration.md).
> - Every swapped-in structure carries a **runtime contract and a ledger**.
>   Falling back to the host is numerically exact, so a seam that quietly
>   reverted is invisible to parity — the ledger is how you see it.

---

## 1. What a structure is

One model region, versioned, with four parts declared in
`flash_rt/structures/catalog/<name>/structure.yaml`:

| Part | What it fixes |
|---|---|
| `boundary` | the tensors in and out, in symbolic dims. `dtype: "@binding"` defers dtype to the host binding on purpose |
| `weights` | framework-neutral slots and their dims, not checkpoint key names |
| `calibration.points` | what has to be observed to calibrate it, **named by position in this structure's own dataflow** |
| `gates` | parity metrics and the latency rule that qualify an implementation |

plus `reference:` — a plain-torch implementation in the catalog, which is
what a gate compares against. A structure with no reference cannot gate
anything, so it is not a structure.

## 2. The three layers, and why the split is where it is

```
catalog/<name>/structure.yaml   the definition. Positions, slots, gates.
                                Changes ~never; a change is a version bump
                                and moves spec_digest.

bindings/<host>.yaml            how this host realises it: which module
                                path holds which weight slot, which
                                submodule a calibration point sits on,
                                what the boundary dtype actually is.
                                Changes once per host family.

impls/<name>/<backend>.py       the executable form: kernels, quantisation
                                format, what statistic to take at a point
                                and how to reduce it. Changes with every
                                new format.
```

The rule is **what varies at what rate**, not who needs it. A worked
example, because this is the part that goes wrong:

`decoder_ffn` declares `calibration.points: [x_after_norm, act_after_mul]`.
`act_after_mul` is a position — the gated activation — and it means the
same thing in a GGML graph as in a torch module tree. *Where* it sits is
per host (`...layers.{i}.mlp.down_proj`'s input, on a transformers-shaped
host). *What to measure there* is per backend: a per-tensor amax for FP8,
a per-column second moment for an importance-matrix flow, a per-block
statistic for a k-quant, nothing at all for a backend that quantises
activations dynamically.

Put the statistic in the spec and every new format needs a schema change.
Put the position in the impl and the same dataflow knowledge gets written
once per backend and drifts. Neither is recoverable later, so the split is
load-bearing.

### 2.1 Pipeline coverage bindings

A native pipeline is bound at two levels. Region bindings map weight slots
and calibration points as above. A pipeline binding maps the larger stage
seams and classifies every declared hot-path segment as one of:

| Classification | Owner |
|---|---|
| `structure` | one or more catalog regions own the composition |
| `state_region` | an explicit buffer/cache/window owns state and cadence |
| `host_stage` | host preprocessing, embedding, or other retained glue |
| `control` | loop, branch, or scheduling logic rather than a kernel region |

`structures.load_binding(name, require_pipeline_coverage=True)` validates
the binding against the catalog: stage names must exist, every hot-path name
must resolve to one segment, and every referenced region structure must
exist. Unknown classifications and unclassified hot-path names fail at load
time. The normalized `BindingSpec.manifest()` is JSON-serialisable for
runtime exporters and native consumers.

This is a composition contract, not a compiler IR. It does not encode a
kernel, target architecture, launch policy, or tensor lowering. Those stay
in each referenced structure's implementation and Hub package, so adding a
hardware target does not fork the pipeline declaration.

The catalog currently uses two schedule families:

- `autoregressive_decode_pipeline`: optional modality encoding, causal
  prefill/KV materialization, token decode, and token selection. Qwen3 and
  Qwen3-VL share this family; the latter binds the optional modality stage.
- `vla_tick_pipeline`: observation-cadence condition preparation, a
  fixed-step iterative update, and an optional output readout. Pi0.5 and
  Motus bind this family with different state and readout regions.

A `host_stage` entry is an explicit coverage result, not a claim that the
region is finished. For example, a fused Q/K norm plus RoPE path remains a
host stage until a catalog structure owns that boundary; landing the
structure changes the classification without changing the pipeline family.

## 3. Calibration: reuse, do not redefine

**The standard is `docs/calibration.md`. This layer adds no second
vocabulary for it.** Concretely:

| Concern | Where it comes from |
|---|---|
| what to observe | the spec's `calibration.points` |
| where it is on this host | `flash_rt/structures/points.py` + discovery |
| reduction across samples | `flash_rt.core.calibration.accumulate_amax` |
| dispersion diagnostics | `summarize_amax_dispersion` / `format_summary` |
| outlier-scale warning | `check_scale_ceiling` |
| picking calibration frames | `stratified_sample_indices`, `flash_rt.datasets.libero` |
| the receipt | `flash_rt.core.precision_spec.ModelPrecisionSpec` |
| argument names and defaults | `flash_rt.api.FlashRT.calibrate` |

```python
structures.auto_swaps(model, forward)                          # one sample
structures.auto_swaps(model, feed, observations=frames)         # N samples
structures.auto_swaps(model, feed, observations=frames,
                      percentile=95.0, max_samples=64)
plan.precision_spec        # ModelPrecisionSpec, same as rt.precision_spec
plan.notes["calibration"]  # method, samples, percentile, dispersion
```

### 3.1 The reduction is two-level, and both levels are the house's

**Within one sample: max over every call.** Required, not chosen —
`calibration.md` §4.2 records that per-step scales on a flow-matching host
gave the compiler inconsistent shapes and crashed it. One forward already
covers every step.

**Across samples: `accumulate_amax(per_sample, percentile)`.** Each sample
contributes one `[num_points]` vector, kept host-side. This ordering is
not a detail: a running max *across* samples destroys the per-sample
values as it produces them, which makes a percentile impossible rather
than merely unused. If you find yourself accumulating in place across
samples, that is the bug.

### 3.2 A point is measured where it is, never recomputed

Every activation scale an implementation needs is the amax at some host
GEMM's input, so it is one hook and one float. `decoder_ffn`'s hidden
scale is the amax at `down_proj`'s input; `vision_ffn`'s is at `fc2`'s.
Keeping the seam's input alive to re-run gate/up over it arrives at the
same number the host already produced, and costs GiB to do it.

Weight scales are derived from weights at bind time and are not
calibration — same division as `calibration.md` §2.1/§2.2.

### 3.3 Three kinds of capture, and only one is calibration

| Kind | Examples | Reduced by a percentile? |
|---|---|---|
| **statistic** | `x_after_norm`, `act_after_mul`, the shared q/k/v input | yes — this is calibration |
| **content** | an adaptive norm's step table, an attention prefix KV, a cadence buffer | no. The artefact *is* the output; a percentile over it is meaningless |
| **observation** | row counts, observed dtypes, a return convention | no. One scalar |

Content and observation are plan-time captures, bounded by construction,
and they get **no public calibration surface of their own**. Giving them
one would be exactly the extra entry point this layer must not add.

### 3.4 Reporting parity, and what the bands mean

Report **held-out** parity, and say when a figure is not. A parity measured
on the frame its scales were fitted to is a fit residual: on Pi0.5 that gap
is about 0.0029, measured three times. Report cosine **and** max-abs
against the reference — multi-sample calibration's benefit shows up mostly
in the worst case, so cosine alone can miss it entirely
(`calibration.md` §10).

Bands, from `gates.py`, are per output kind. Value outputs (a policy's
action tensor): `pass` at cosine 0.999 and above, `warn` from 0.995, and
`low` below that. Distribution outputs (a language model's logits) are
judged on **token agreement**, where cosine-grade edges do not transfer:
a clean static W8A8 with every per-seam gate passing sits at 0.95–0.98
agreement on real text — that is the grade of the quantisation, not
damage, which instead looks like agreement collapsing while seam-level
parity stays fine. So `pass` from 0.95, `warn` from 0.85, `low` below.
**None of the three refuses.** Low-precision execution is
increasingly the intent rather than a defect — a four-bit host belongs in
the bottom band by design — so a `low` band warns with the number and the
calibration method attached, and whether it is acceptable is the
deployment's call. Pass `floors={...}` to turn a number into a hard
requirement; that is the caller stating a requirement, which is the only
place such a number can honestly come from.

## 4. Doors

```python
plan = structures.attach(model, forward)      # gated: discover→gate→activate
plan = structures.auto_swaps(model, forward)  # build only, you own the gate
mod  = structures.get("decoder_ffn").bind(module, calibration=[x])
stage = structures.capture(hot, windows={...})
```

`attach` gates unit by unit — a unit is a structure, except that a
negotiated FP8 chain is one unit, because the producer emits under a scale
the consumer was bound for. It judges accuracy with the metric the host's
output type deserves, checks the ledger, and settles latency by timing
both arms in every round.

`auto_swaps` builds and does not judge. If you use it, you own the gate —
and read the ledger, or you have not checked that what you measured was on.

### 4.1 Selecting a precision profile

`scheme=` is the one precision entry, on both doors. A profile is a
registered quantisation scheme (§6) selected by name:

| name | what it does |
|---|---|
| `"auto"` (default) | resolves to the fastest profile this device can execute: `fp8_static` on FP8-capable hardware (bit-identical to the pre-profile default), `"none"` elsewhere. The resolution table is one function, so a profile that measures faster is promoted by editing one line |
| `"fp8_static"` | static per-tensor FP8, the shipped behaviour; `"fp8_static_keep_outliers"` keeps outlier seams at host precision by the house scale-ceiling criterion |
| `"w8a16_decode"` / `"w4a16_decode"` | weight-only INT8 / NVFP4 on `decoder_ffn`, decode band only, everything else at host precision |
| `"none"` | quantisation off. An explicit choice, not a degraded mode: fusion structures never consult a scheme decision and attach as usual, so a BF16/FP16 host under `"none"` still gets every fusion structure |

Quantisation happens **at attach time, from the host's own weights** —
the same discipline as this repo's native pipelines. The checkpoint is
loaded at host precision; weight scales and packed formats are derived
from the floating weights at bind, activation statistics come from
running the host's own forward. Nothing is destroyed: the original
module is retained, and detach restores it bit-exactly.

Hardware support is not declared here at all. A Hub kernel package
ships the archs it was built for in its own metadata; the shared loader
reads that declaration and refuses a device outside it with the package
name and both arch strings in the message. The structures layer keeps
no second table — hardware support is maintained where the kernels are.

## 5. Runtime contract and the ledger

Every swapped-in structure declares the form it was calibrated for
(device, input dtype, width, and a row count where buffers were
preallocated). Called outside it, a seam runs the retained host module and
records that it did.

```python
handle.report()             # per seam: calls, fallbacks, last_reason, form
handle.summary()["clean"]
handle.raise_on_fallback()
structures.swap.attach(..., on_guard_fail="raise")   # refuse instead
```

The first fallback per seam warns; 32 consecutive fallbacks restore the
host module for good. Counts are eager-only — inside a compiled or
captured region the kernel runs without re-entering Python, which is also
why the check costs nothing there.

Refused rather than approximated: training mode, device/dtype migration
while attached, `load_state_dict` while attached, and a second thread in
one seam. `state_dict()` delegates to the retained host module, so saving
while attached yields the unattached schema and bytes.

## 6. Adding things

**A backend for an existing structure**: add `impls/<name>/<backend>.py`.
Read the spec's points, take whatever statistic that format needs, declare
your own qualification band per executable form. Do not touch the spec —
if you need a position it does not name, that is a signal the boundary is
drawn wrong, and it should be raised rather than absorbed.

**A host family**: add `bindings/<host>.yaml` with the weight map and the
point addressing. Discovery covers hosts whose slots are findable
structurally; the binding is the receipt, and the only source for hosts
where they are not.

**A structure**: spec + reference + gates first, in the catalog, with an
implementation second. A spec whose points nothing can locate fails loudly
at plan time, which is the intent.

**A quantisation scheme**: register an instance in `schemes.py` — two
methods and nothing else. `statistics` declares what each calibration
point needs (statistic and granularity: per-tensor, per-channel,
per-16-block; `None` is legal and means the format quantises that point
dynamically at runtime). `decide` turns the reduced statistics into
per-seam outcomes: bind with these values, or keep the host at host
precision — a first-class decision recorded in the receipt with its
reason, not a refusal. Bytes are not the scheme's: scale-factor layouts,
sub-normal handling, packing and kernel choice live in the impl variant
that executes the decision, which is what lets one decision serve
different kernels. A statistic the collector cannot measure yet fails
loudly at plan time; extending the collector is the supported path, and
silently substituting per-tensor is not. Schemes add no calibration
entry point — the calibration axis is fixed, a scheme only declares what
to measure along it.

## 7. Norms that came from being wrong

- **Calibrate and judge on the host's real inference distribution.**
  Training-domain data through the host's own preprocessing chain, or
  deployment inputs — never a neighbouring dataset with a hand-assembled
  mapping, and never synthetic text repeated and padded to length. Two
  hosts here measured 0.02 of cosine and 13 points of token agreement
  worse on such inputs than on their real data, with the mechanism
  unchanged; the dirty figures nearly became a quantisation-scheme
  decision. Before attaching, prove the measurement itself: rerun the
  host unmodified and require bit-identical output.
- **Grep the repo before writing a mechanism.** Percentile reduction,
  stratified sampling, dispersion diagnostics, scale-ceiling warnings and
  the precision receipt all existed before this layer reimplemented worse
  versions of the first two and skipped the rest.
- **Grep the repo before saying something is not implemented.** Asserting
  a missing capability is worse than missing it, because the assertion
  becomes a documented design boundary.
- **A declaration nobody checks is a comment.** The spec's point names are
  safe upstream only because they resolve against something — the
  reference and the loud failure when they do not.
- **Report the increment, not the peak.** A resource number that includes
  the model weights and the bound plan is not the cost of the thing being
  measured.
- **Refusals record the form and the shape.** "refused" must never read as
  "this cannot be bound", only as "not in that form, at that size".

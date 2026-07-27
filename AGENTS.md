# AGENTS.md — producing structures for the FlashRT structures layer

You are working on the structures layer: verified, host-independent
acceleration structures assembled from Hub kernels. This document is the
complete operating procedure — how to extract a structure abstraction
from FlashRT's native implementations, how to keep it general, how to
wire Hub kernels, what the red lines are, and what acceptance looks
like. Follow it literally. When unsure, stop and ask; do not guess.

---

## 0. Required reading, in order

| Document | What you take from it |
|---|---|
| `docs/structures.md` | The norm: what a structure is, the three-layer split, calibration reuse, accuracy bands, runtime contract and ledger, how to add structures/backends/hosts/schemes, and §7 — norms that came from being wrong |
| `docs/calibration.md` | The house calibration standard (statistics, two-level reduction, diagnostics). You must not invent a second one |
| `catalog/*/structure.yaml` + `reference.py` | Worked precedents; copy the format of `decoder_ffn` |

Code landmarks under `flash_rt/structures/`: `catalog/` (spec +
reference), `bindings/` (per-host addressing receipts), `impls/`
(executable forms), `discover.py` (structural discovery),
`autobuild.py` (assembly), `points.py` (calibration collection),
`schemes.py` (quantisation schemes), `gates.py` (accuracy judgment),
`guard.py` + `swap.py` (runtime contract, ledger, attach/detach),
`frontdoor.py` (the gated one-call door).

---

## 1. Finding a structure abstraction in the native implementations

**Where the ore is** (same repository): the native pipelines under
`flash_rt/models/*/pipeline_*.py` and `flash_rt/frontends/torch/*.py`
are a record of fusion decisions that were already measured to pay;
`docs/kernel_catalog.md` and `docs/optimization-details.md` list the
kernels and their yields.

**What qualifies as a structure**: a region of dataflow that recurs
**across host families**, defined by four things — boundary tensors,
weight slots, calibration points, and gates. It is never one host's
module name.

**Procedure**:
1. In a native pipeline, list the fusion decisions: which ops were
   merged into one kernel, who shares a scale with whom, what is cached
   at observation cadence. Each such decision is a candidate structure.
2. Take the candidate to **at least two unrelated host families** and
   find the same stretch of dataflow. Found in both → it is a
   structure. Found in one → it is a binding specialisation and does
   not enter the catalog.
3. Name calibration points by **position in the structure's own
   dataflow** — `act_after_mul` (after the gated activation) means the
   same thing in a GGML graph as in a torch module tree. Never a name
   that only exists on one host.
4. Write `catalog/<name>/structure.yaml`: boundary (symbolic dims,
   dtype may be `"@binding"`), weights (framework-neutral slot names),
   `calibration.points`, gates. **Statistics do not go in the spec** —
   they belong to the quantisation scheme (`docs/structures.md` §2).
5. Write `catalog/<name>/reference.py`, a plain-torch reference.
   A structure without a reference cannot gate anything and is not a
   structure.

**Discovery is semantic, not name-blind.** Discovery rules may use
semantic slot names (`gate_proj`/`up_proj`/`down_proj` are a gated-MLP
signature), tensor shapes, dataflow relations, and forward signatures;
a non-standard host gets a family adapter. What is forbidden: matching
on model IDs, on concrete class-name whitelists, or on one incidental
module name as the only evidence.

**Generality is proven, not claimed**: run discovery on two families
and record the results as evidence — and verify the negative case too.
A structure must *not* be discovered on hosts it does not describe;
"correctly not found" is an acceptance item with test precedent.

---

## 2. Finding and wiring Hub kernels

1. **Shop before building**: check the `flashrt/*` and
   `kernels-community/*` Hub organisations for an existing package.
2. **Loading convention**: always go through the shared
   `impls.hub_kernel(repo, version)` helper. Loading the same package
   twice re-registers its fake ops; the loader must be shared.
3. **Resolve ops at bind time; the swapped forward must be a single
   custom-op call.** Calling a hub loader inside `forward` drags the
   version resolution into the compiler's trace and fragments the
   graph (a measured 26-graph-break incident).
4. **Standalone-bench the kernel before wiring it** at the host's real
   shapes, and separate kernel time from launch time — an eager
   preflight win can evaporate inside a captured graph.
5. Shapes the kernel does not cover get **declared dispatch** (the
   weight-only FFN precedent: decode band to the kernel, prefill back
   to the host, both counted separately in the ledger) — never a
   silent stretch of the kernel's envelope.
6. If the kernel you need does not exist, stop and report "a kernel is
   missing", with the shapes and the roofline. Do not emulate it with
   a chain of eager ops; a merge that is not one real kernel loses to
   the compiler's own fusion.
7. **Hardware support comes from the package, not from you.** The Hub
   package's own metadata declares the archs it was built for, and the
   shared loader enforces it with a clean refusal. Do not write arch
   tables into an impl — a second table drifts, and hardware support is
   maintained on the kernels side.
8. **A binder that loads a kernel runs a bind-time smoke**: one
   real launch through the entry point at the seam's own width before
   the seam is handed out (`w4a16_static.bind_mlp_seam` is the
   precedent). A stale build or missing symbol must surface as a bind
   refusal — in a fallback-capable system it can never be caught by
   comparing outputs, because falling back is numerically exact.

---

## 3. Landing one structure, step by step

1. **Read** the native implementation and the target host source. List:
   boundary tensors, weight slots, fusion decisions, and any state that
   changes with the observation — that state must be handled explicitly
   (see red lines).
2. **Spec + reference** (§1, steps 4–5).
3. **Addressing**: hosts whose slots are structurally findable go into
   `discover.py` rules; others get `bindings/<host>.yaml`. Binding
   YAMLs are the *receipt* of an addressing decision; structural
   discovery is the runtime mechanism.
4. **Impl** in `impls/<name>/<backend>.py`. It must:
   - subclass `GuardedSeam` and declare its executable form via
     `_frt_arm(dtypes/device/k/rows)`. A row-locked structure on a
     variable-length host falls back by contract — that is correct
     behaviour, and it must be documented on the impl;
   - state its weight-layout convention in the binder's docstring.
     Two binders with different layout conventions have already
     collided once, with the dimension check passing under swapped
     names — the convention must be written where the next caller
     will read it;
   - retain the original host module (fallback and `state_dict`
     delegation depend on it).
5. **Calibration**: through the `points.py` collector and the scheme
   interface only. Do not hook activations yourself; if you need a
   granularity the collector cannot measure, extend the collector —
   the loud failure you hit is intentional.
6. **Tests** — every new structure ships **public CPU contract tests**
   in `tests/` alongside the PR:
   - reference correctness against the spec;
   - positive discovery on a synthetic host of the right shape;
   - negative discovery: it must not fire on hosts it does not
     describe;
   - guard dispatch and fallback behaviour;
   - attach/detach reversibility (bit-exact restore, module count
     unchanged);
   - clean refusal when a capability is missing.
   GPU validation on real hosts follows the same ladder — per-seam
   gate at the declared boundary (residual included), end-to-end
   same-input/same-output against the unmodified host, held-out
   evaluation with a null check (rerunning the unmodified host must be
   bit-identical) and a negative control (deliberately break the
   refresh or window; the metric must visibly degrade, proving the
   test can detect the failure), a clean ledger, and **evidence that
   the kernel path actually ran** — in a fallback-capable system,
   identical output alone proves nothing, because falling back to the
   host is numerically exact.
7. **Data**: calibration and evaluation inputs must come from the
   host's real inference distribution, built through the host's own
   preprocessing chain (`docs/structures.md` §7). Out-of-distribution
   or synthetic inputs have mismeasured hosts here before.
8. **Reporting**: `plan.report()` / the plan notes are the receipt —
   discovered, activated or refused and why, band, measured speedup,
   calibration method, ledger. Latency claims come from paired
   alternating timing; single-arm wall-clock drifts several percent
   and is not accepted.

---

## 4. Red lines (any violation returns the work)

1. **Additive only.** Existing kernels, bindings, loaders and the
   catalog schema are not modified. Changing a shared helper's return
   type requires grepping every reader first.
2. **Grep before building.** Reuse the repo's existing mechanisms
   (calibration, diagnostics, sampling, receipts). Claiming something
   is "not implemented" requires grep evidence.
3. **No new calibration entry points, and no new precision entry
   points.** The calibration axis is `forward`/`samples`, once. A
   scheme declares and consumes statistics; it does not open a second
   door. Precision profiles are registered schemes selected by the
   existing `scheme=` parameter (`docs/structures.md` §4.1) — a new
   precision mode is a scheme registration, never a new parameter.
4. **Fail loudly; never degrade silently.** Unmeasurable granularity,
   unknown format variants, unlocatable points — raise, with the
   reason. No silent approximations.
5. **Identical output is not evidence on its own** — pair it with the
   ledger's fallback count and the target path's call count.
6. **The forward you swap in must be compiler-friendly**: no Python
   side effects in the hot path, no loader calls, one custom-op call.
7. Every API mentioned in docs or a PR description is checked against
   the source signature before it is written down.

---

## 5. Deliverables and acceptance

**You deliver**:
1. the repository diff — additions under `catalog/`, `bindings/`,
   `impls/`, `tests/`, `docs/` only;
2. the public CPU contract tests, green;
3. a one-page report: the results table (host / baseline stating eager
   or compiled / result / speedup / output match with worst case),
   the data-source statement, and the null-check / negative-control /
   ledger / detach results.

**The reviewer will**: run all tests; grep every API signature you
cite and every "not implemented" claim; check the data is the host's
real distribution; check the ledger is clean and the kernel path was
exercised; probe generality on a third host (correct discovery or
correct absence); and scan the diff for anything that does not belong
in a public repository.

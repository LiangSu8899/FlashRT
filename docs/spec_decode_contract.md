# Speculative-Decode Session — Roles and Contract

Authoritative module:
[`flash_rt/frontends/torch/spec_session.py`](../flash_rt/frontends/torch/spec_session.py).
One speculation loop serves every (model family x drafter type x
hardware) combination; everything model- or hardware-shaped is a role
implementation. The DFlash path
([`qwen36_dflash.md`](qwen36_dflash.md)) is the first instantiation.

## Roles

| Role | Owns | Knows about hardware? |
|---|---|---|
| `AcceptancePolicy` | which draft rows are accepted and which token each accepted row commits (`StrictArgmax`, `RelaxedThinking`) | no — pure tensor math, CPU unit-tested |
| `Drafter` | proposing K tokens per cycle and maintaining the drafter-side context (`DFlashBlockDrafter`: feature window; an MTP chain drafter fits the same surface) | only through kernel availability |
| Verifier seam | prompt prefill and the S=K verify forward (frontend hooks `_dflash_prefill_nvfp4`, verify graphs) | yes — per (model family, hardware) |
| StateCommitter seam | per-step rollback checkpoints written during the verify; constant-time `commit(N)` (frontend hooks `_dflash_snap_state`, `_dflash_partial_rollback`) | yes |
| `SpecSession` | the cycle: snap ∥ draft → verify → accept → commit tokens → rollback → drafter commit; telemetry; `step()` / `boundary()` | no |

## Invariants

Violations are correctness bugs, not tuning choices. Each is guarded by
a structural test in `tests/test_qwen36_dflash_structural.py`.

- **I1 — one kernel family.** The verify forward and the rollback
  state source must come from the same kernel family. Two families
  agree at the LSB almost always, then disagree once; committing rows
  from one while recovering with the other turns that into greedy
  divergence.
- **I2 — parity references share KV format.** Token-exact comparisons
  are only defined against a reference whose verify reads the same KV
  representation.
- **I3 — drafter commit precedes the taps shuffle.** The drafter
  consumes feedback rows `0..N`; the end-of-cycle `taps[:, 0]` shuffle
  overwrites row 0.
- **I4 — checkpoint slot semantics.** Slot `s` holds the committed
  state after verify row `s`; slot `K-1` equals the post-verify state.

## Session surface

```python
session = fe.make_dflash_session(max_new_tokens=256, K=15)
out = session.generate(ids)          # one-shot

session.begin(ids)                   # or drive it yourself:
while not session.done():
    n = session.step()               # ONE cycle; the interruption grid
session.request_interrupt()          # stop generate() at a boundary
session.boundary()                   # named buffers of the committed
                                     # boundary (KV, recurrent/conv
                                     # state, drafter window, cursor)
```

`step()` boundaries are quiescent points: state reflects exactly the
committed tokens, which makes them the legal grid for host-level
snapshot/restore/fork and for preemption by a co-scheduled real-time
workload. `boundary()` enumerates what such a snapshot must carry.

`generate_own_speculative_DFlash_nvfp4` remains the stable public
wrapper; existing callers are unaffected.

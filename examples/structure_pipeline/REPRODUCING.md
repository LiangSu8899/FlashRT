# Reproducing the two-host GR00T N1.7 matrix

Every number in the README comes from the scripts in this folder run
against **unmodified host checkouts**. This page is the complete
recipe: what to fetch, what each script touches, and — first, because
it is the question that matters — exactly how large the footprint is.

## The footprint

| where | what changes | size |
|---|---|---|
| official Isaac-GR00T source | **nothing** | 0 lines |
| LeRobot source | **nothing** | 0 lines |
| host process at run time | structure swaps attached onto the module tree | revertible; `detach()` restores the host bit-for-bit |
| host process at run time (captured form only) | fixed-shape lowering: a set of function pins applied before capture and undone after | in-process only, never written to disk; every pin is one function in `full_graph.py::lower_backbone_to_fixed_shapes` (176 lines), each pinning a shape-derived constant of one fixed request |

The author-owned code, by role:

| file | role | lines |
|---|---|---|
| `groot_n17.py::build` | **the explicit assembly itself** — seat tables, calibration hooks, binder calls | 215 |
| `groot_n17.py` (rest) | host loading (34), input capture (~30), timing/report harness (132) | 287 |
| `full_graph.py` | fixed-shape lowering (176) + capture/replay/noise-pin helpers (64) + harness (111) | 398 |
| `lerobot_host.py`, `lerobot_full_graph.py` | the same measurement on the LeRobot host; `build()` is imported, not rewritten | ~300 |
| `make_model_inputs.py` | exports one set of prepared model-level inputs both hosts consume | ~60 |

The automatic path, for comparison, is two calls:

```python
plan = structures.auto_swaps(model, run_once)
handle = swap.attach(model, plan.swaps, observe=plan.observed,
                     revert=plan.revert)
```

## What you need

1. **Hosts** — either or both, unmodified:
   - official: `github.com/NVIDIA/Isaac-GR00T`
   - LeRobot: a checkout with the GR00T N1.7 policy
     (`lerobot/policies/groot/groot_n1_7.py`)
2. **Checkpoint** — the public GR00T N1.7 3B release (one copy serves
   both hosts).
3. **Backbone config assets** — the backbone's config/processor files
   (`nvidia/Cosmos-Reason2-2B`); the official constructor's redundant
   base-weight download is redirected to these, construction-I/O only.
4. **This repository** on `PYTHONPATH`, plus this folder for the
   cross-host runners (`build()` is imported from `groot_n17.py`).
5. **Kernel packages** resolve from the Hugging Face Hub at first
   bind; offline, stage them as `<org>/<name>/build/<variant>/` and
   point the kernels resolver at the directory.

Environment notes, learned the hard way and probed in code rather than
pinned: the capture lowering supports both transformers vision-contract
generations (tuple-return and output-class); the LeRobot loading recipe
`from_pretrained(...).to(dtype=bf16)` casts the rotary `inv_freq`
buffer to BF16 — a host-side precision loss the README documents, and
the reason the `qkv_rope` family refuses on that host.

## Step by step

```bash
# 0) one observation fixture from the host's own preprocessing:
#    run the host policy once on any observation and save it —
#    torch.save({"inputs": observation_dict}, "obs_fixture.pt")

# 1) official host — explicit assembly, eager + compiled ladder
python groot_n17.py --host <Isaac-GR00T> --checkpoint <ckpt> \
  --backbone-assets <cosmos-reason2-assets> --fixture obs_fixture.pt \
  --compile --report official_explicit.json

# 2) official host — the same assembly, captured (deployed form)
python full_graph.py --host <Isaac-GR00T> --checkpoint <ckpt> \
  --backbone-assets <cosmos-reason2-assets> --fixture obs_fixture.pt

# 3) export the prepared model-level inputs both hosts consume
python make_model_inputs.py --host <Isaac-GR00T> --checkpoint <ckpt> \
  --backbone-assets <assets> --fixture obs_fixture.pt \
  --out model_inputs.pt

# 4) LeRobot host — baseline / auto, eager + compiled
python lerobot_host.py --lerobot-src <lerobot>/src \
  --checkpoint <ckpt> --inputs model_inputs.pt \
  --arm auto --compile --report lerobot_auto.json

# 5) LeRobot host — explicit and auto at the captured form
python lerobot_full_graph.py --lerobot-src <lerobot>/src \
  --checkpoint <ckpt> --inputs model_inputs.pt --arm explicit
python lerobot_full_graph.py --lerobot-src <lerobot>/src \
  --checkpoint <ckpt> --inputs model_inputs.pt --arm auto
```

Step 3 is what makes the two hosts comparable: both consume the same
prepared tensors, so host code is the only variable.

## What to expect

The README carries the measured matrix (RTX 5090). Judge a
reproduction by, in order:

1. `ledger.fallbacks` and `refused` match in *kind* (a refusal is an
   outcome; a silently missing seat is a bug);
2. parity ≥ 0.999 on every arm, against that host's own eager run;
3. speedups within a form row land in the same band — clocks move a
   few percent between runs, so read ratios, not milliseconds across
   processes.

# Weight Loading: the `WEIGHT_SPEC` API

> **Target audience**: anyone adding a new model, swapping in a different checkpoint format, or fusing weights at load time. This doc treats `WEIGHT_SPEC` as a **first-class API** — not as an internal of the frontend.
>
> **What this doc is for**: the contract — protocols, data classes, semantics of one `run()`.
> **What this doc is not**: the per-model walkthrough. For that see [`../adding_new_model.md`](../adding_new_model.md).
>
> **Source**: [`flash_rt/executors/weight_loader.py`](../../flash_rt/executors/weight_loader.py) — protocols + runner. Concrete sources / sinks live in `executors/torch_weights.py` and `executors/jax_weights.py`.

---

## 1. What `WEIGHT_SPEC` solves

Weight loading in a non-trivial inference engine has three painful properties:

1. **It is the most checkpoint-format-coupled code** in the whole stack — every change in HuggingFace's `safetensors` keys, every Orbax restructure, every `gate_up_proj` vs split-`gate_proj/up_proj` choice ripples through.
2. **It is the place transformations live** — transposes, FP8 quantization, fused-norm absorption, gate+up concatenation. These are not load-time bugs; they are the load-time *job*.
3. **It is duplicated per (model, framework) pair** if written imperatively. Pi0.5 torch, Pi0.5 JAX, GROOT torch, GROOT JAX would each have hundreds of lines of nearly-identical code.

`WEIGHT_SPEC` collapses all three into **one declarative description plus one runner**. The description is data. The runner is framework-agnostic. Concrete sources / sinks are interchangeable.

---

## 2. The three-layer shape

```
WeightSource  →  TransformPipeline  →  WeightSink
   (read)         (.T / fuse / quant)     (store)
```

- **WeightSource**: knows how to fetch a tensor by string key. `SafetensorsSource` wraps `safe_open(...)`. `OrbaxDictSource` wraps the JAX `engine_w` dict. You write one source per checkpoint format; the rest of the spec is reusable across formats.
- **Transform**: a pure function `(tensor, ctx) → tensor`. Examples: `Transpose()`, `QuantFP8()`, `FuseNorm("...norm.weight")` (multiplies an absorbed norm into the projection), `Cat(...)`, `FusedQKV(...)`.
- **WeightSink**: where the final tensor goes. `Attr("name")` does `setattr(target, "name", t)`. `TensorList("name")` appends to a per-layer list (built up across iterations of a `LayerBlock`). `FlatCat("name")` concatenates at finalize time into a single contiguous device tensor.

The runner walks the spec, fetches sources, applies transforms in order, and stores sinks. It is 50 lines of Python.

---

## 3. The three data classes

### 3.1 `Item` — one logical weight

```python
@dataclass
class Item:
    name:        str               # tag for diagnostics
    key:         str | CompositeKey # checkpoint key with {prefix}/{i} placeholders
    transforms:  list[Transform] = []
    sink:        WeightSink = None
    scale_into:  str | None = None # quant scale target list name
```

A literal `key` (string) is fetched directly from the source. A composite key (`Cat(...)`, `FusedQKV(...)`, `FusedGateUp(...)`) implements `resolve(ctx) → tensor` and reads multiple keys + concatenates.

`scale_into` matters when a transform produces a per-tensor scale (`QuantFP8` does). The scale value is appended to `ctx.scales[scale_into]` in spec-iteration order. At the end, the runner publishes scale lists onto the target as attributes — the frontend wraps them into device tensors.

### 3.2 `LayerBlock` — a `num_layers × items` rectangular loop

```python
@dataclass
class LayerBlock:
    prefix_fmt: str          # e.g. "encoder.layers.{i}"
    num_layers: int          # e.g. 18
    items:      list[Item]
    name:       str = ""     # diagnostics tag
```

Equivalent to:

```python
for i in range(num_layers):
    prefix = prefix_fmt.format(i=i)
    for item in items:
        run item with "{prefix}" → prefix in item.key
```

Most VLA / LLM weight tables have 80–95% of their items inside `LayerBlock`s — that is the point.

### 3.3 `ModelWeightSpec` — the top-level spec

```python
@dataclass
class ModelWeightSpec:
    framework:  str                        # "torch" | "jax"
    blocks:     list[LayerBlock] = []
    singletons: list[Item]       = []      # embed, lm_head, etc.
    buffers:    list[BufferSpec] = []      # pre-allocated device buffers
    dims:       dict[str, Any]   = {}      # symbolic shapes for buffers
```

`singletons` runs first; then each `LayerBlock`; then `buffers` (allocated, not loaded). One spec per `(model, framework)` pair lives at module scope of the corresponding frontend file:

```python
# flash_rt/frontends/torch/_pi05_thor_spec.py
WEIGHT_SPEC = ModelWeightSpec(framework="torch", blocks=[...], singletons=[...])
```

---

## 4. Running the spec

```python
from flash_rt.executors.weight_loader import WeightLoader

WeightLoader(
    source=SafetensorsSource("/path/to/model.safetensors"),
    target=self,                        # the frontend instance
    spec=self.WEIGHT_SPEC,
).run()
```

`run()` returns the final `LoaderContext`, but the body of `_load_weights` is typically just the three lines above. After `run()` returns, every attribute referenced by a `Sink` is populated on the frontend, and every accumulator named via `scale_into` is published as a list attribute.

The runner is:

- **Framework-agnostic** — the same runner walks the same spec against `SafetensorsSource` (torch) or `OrbaxDictSource` (JAX). Adding a new checkpoint format is one new `WeightSource` class, not a fork of the runner.
- **Order-preserving** — items run in declaration order. Scale lists end up indexed by spec position. This determinism matters because FP8 alphas (one per FP8 GEMM, ~370 per layer) are addressed positionally at graph capture.
- **Pure-Python** — no torch / no JAX imported by `weight_loader.py` itself. Concretes import their framework; the runner does not.

---

## 5. Composite keys (built-in)

Composite keys are `Item.key` values that read more than one source key and combine them. Three are shipped:

### `Cat(*keys, axis=0)`

Concatenates several source tensors along an axis. Used when a checkpoint stores e.g. `gate_proj` and `up_proj` separately but the kernel wants a fused `gate_up` tensor:

```python
Item(name="gate_up.weight",
     key=Cat("{prefix}.gate_proj.weight",
             "{prefix}.up_proj.weight",
             axis=0),
     transforms=[Transpose(), QuantFP8()],
     sink=TensorList("gate_up_w"),
     scale_into="gate_up_w_scales")
```

### `FusedQKV(q_key, k_key, v_key, ...)`

Reads three projection tensors and concatenates them in QKV order. Optional reshape parameters handle GQA layouts where Q has more heads than KV.

### `FusedGateUp(gate_key, up_key)`

Specialized version of `Cat` that knows which half is gate (consumed by GeGLU/SwiGLU activation) and which is up. Used when downstream kernels (`gate_geglu_merged_*`) read from one merged buffer.

Adding a new composite key is implementing `resolve(ctx) → tensor`:

```python
class MyComposite:
    def __init__(self, ...): ...
    def resolve(self, ctx):
        a = ctx.source.get(ctx.subkey("..."))
        b = ctx.source.get(ctx.subkey("..."))
        return my_combine(a, b)
```

---

## 6. Built-in sinks

### `Attr(name)` — single value

```python
Item(name="embed", key="model.embed.weight", sink=Attr("embed_weight"))
# After run: self.embed_weight is the tensor
```

### `TensorList(name)` — per-layer list

```python
LayerBlock(prefix_fmt="encoder.layers.{i}", num_layers=18, items=[
    Item(name="q.weight", key="{prefix}.q.weight", sink=TensorList("q_weights"))
])
# After run: self.q_weights is a list of 18 tensors, indexed by layer
```

### `FlatCat(name)` — concatenate at finalize

```python
TensorList collects into a list; FlatCat collects into a list AND
calls torch.cat (or jnp.concatenate) at finalize() time. Used when
the kernel wants one big contiguous buffer with all layers stacked.
```

### Custom sink

Implement `store(tensor, scale=None)` and `finalize()`:

```python
class MySink:
    def store(self, tensor, *, scale=None): ...
    def finalize(self): ...
```

Sinks are duck-typed (`@runtime_checkable Protocol`), so a custom sink class works as long as the methods exist.

---

## 7. Built-in transforms

The transforms shipped under `flash_rt/executors/torch_weights.py`:

| Transform | What it does | When you need it |
|---|---|---|
| `Transpose(dims=(1,0))` | `tensor.transpose(...)` | HF stores `(out, in)`; cuBLASLt FP8 wants `(in, out)` |
| `QuantFP8()` | `cast_fp16_fp8 + per-tensor scale` | Every FP8 weight |
| `FuseNorm(norm_key)` | `t = (1 + ctx.source.get(norm_key)) * t` | Paligemma absorbs RMSNorm `(1+w)` into Q/K/V projections |
| `Reshape(shape)` | reshape | Composite-key outputs that need a pre-quant rearrange |
| `ToDtype(dtype)` | cast | Down-cast bf16 → fp16 at load time |

Custom transforms — implement `apply(tensor, ctx) → tensor`:

```python
class MyTransform:
    def apply(self, t, ctx):
        # ctx.source — for reading extra keys
        # ctx.scratch["_pending_scale"] — set this if you produce a scale
        return modified_t
```

---

## 8. Stable contract guarantees

The following are guaranteed across all v0.x releases:

1. `WeightSource`, `Transform`, `WeightSink` protocols (their method signatures) do not break.
2. `Item`, `LayerBlock`, `ModelWeightSpec`, `BufferSpec`, `LoaderContext` field names and types do not break.
3. `_run_item` ordering (`subkey → source.get → transforms in order → sink.store → publish scale`) is the contract every transform / sink / source can rely on.
4. New optional fields may be added to data classes (keyword-only, with defaults).
5. Built-in composite keys (`Cat`, `FusedQKV`, `FusedGateUp`) and built-in sinks (`Attr`, `TensorList`, `FlatCat`) keep their constructor signatures.

Not guaranteed (subject to refactor without notice):

- The exact internal helpers `_run_item`, `_resolve_source`, `_finalize_sinks` (use the documented protocols).
- Specific transform names beyond the six listed above (always check the import path you depend on).
- `BufferSpec` semantics — currently a placeholder, will firm up in a future release.

---

## 9. Worked example: a 4-item LayerBlock

Real fragment from `flash_rt/frontends/torch/_pi05_thor_spec.py`, simplified:

```python
from flash_rt.executors.weight_loader import (
    Item, LayerBlock, ModelWeightSpec,
)
from flash_rt.executors.torch_weights import (
    Transpose, QuantFP8, FuseNorm,
    Cat, FusedQKV,
    Attr, TensorList,
)

ENCODER_BLOCK = LayerBlock(
    name="encoder",
    prefix_fmt="model.layers.{i}",
    num_layers=18,
    items=[
        # 1. Pre-attn norm: absorbed into QKV via FuseNorm in the next item.
        #    No standalone item — just register the key for FuseNorm to find.
        # (no Item here; FuseNorm reads {prefix}.input_layernorm.weight directly)

        # 2. Fused QKV (GQA: 8Q, 1KV).
        Item(name="qkv.weight",
             key=FusedQKV(q_key="{prefix}.self_attn.q_proj.weight",
                          k_key="{prefix}.self_attn.k_proj.weight",
                          v_key="{prefix}.self_attn.v_proj.weight"),
             transforms=[
                 FuseNorm("{prefix}.input_layernorm.weight"),
                 Transpose(),
                 QuantFP8(),
             ],
             sink=TensorList("qkv_weights"),
             scale_into="qkv_w_scales"),

        # 3. O projection.
        Item(name="o.weight",
             key="{prefix}.self_attn.o_proj.weight",
             transforms=[Transpose(), QuantFP8()],
             sink=TensorList("o_weights"),
             scale_into="o_w_scales"),

        # 4. Gate+Up fused for GeGLU.
        Item(name="gate_up.weight",
             key=Cat("{prefix}.mlp.gate_proj.weight",
                     "{prefix}.mlp.up_proj.weight",
                     axis=0),
             transforms=[
                 FuseNorm("{prefix}.post_attention_layernorm.weight"),
                 Transpose(),
                 QuantFP8(),
             ],
             sink=TensorList("gate_up_weights"),
             scale_into="gate_up_w_scales"),

        # … one more Item for down_proj …
    ],
)

WEIGHT_SPEC = ModelWeightSpec(
    framework="torch",
    blocks=[ENCODER_BLOCK, DECODER_BLOCK, ...],
    singletons=[Item(name="embed", key="model.embed.weight", sink=Attr("embed"))],
)
```

After `WeightLoader(...).run()` returns:
- `self.qkv_weights` is a list of 18 FP8 tensors, each shape `(D, num_heads * hd)` post-transpose.
- `self.qkv_w_scales` is a list of 18 floats — one weight scale per layer.
- `self.embed` is the embedding tensor.
- All `FuseNorm` calls have absorbed `(1 + norm.weight)` into the projection at load time, so the runtime forward never multiplies by `(1+w)` — it just feeds normalized residual through the FP8 GEMM.

This is the entire weight-loading code for one encoder layer family. Compare it to writing 600 lines of imperative `model.layers[i].q_proj.weight = ...` glue per `(framework, hardware)` combination.

---

## 10. Why this is more useful than `state_dict`

A flat `state_dict` does the *naming* job but none of the *transform* job. `WEIGHT_SPEC` does both:

| Concern | `state_dict` | `WEIGHT_SPEC` |
|---|---|---|
| Map checkpoint keys to runtime attributes | yes | yes |
| Transpose / dtype cast at load time | per-call boilerplate | declarative (one transform) |
| FP8 quant + per-tensor scale | not addressed | declarative |
| Fuse norm `(1 + w)` into projection | hand-written every time | one transform |
| Fuse Q/K/V or gate/up | hand-written every time | one composite key |
| Same loader for safetensors and Orbax | no | yes |
| Cross-format diff is just the source class | no | yes |

The win is not "less code per loader" — it is **the same loader runs against any checkpoint format**. Add a model: write `WEIGHT_SPEC` once. Add a checkpoint format: write a `WeightSource` once. Combinations multiply.

---

## 11. Where to look in the source

| What | File |
|---|---|
| Protocols + runner | [`executors/weight_loader.py`](../../flash_rt/executors/weight_loader.py) |
| Torch sources / transforms / sinks | [`executors/torch_weights.py`](../../flash_rt/executors/torch_weights.py) |
| JAX sources / transforms / sinks | [`executors/jax_weights.py`](../../flash_rt/executors/jax_weights.py) |
| Pi0.5 spec example | `flash_rt/frontends/torch/_pi05_thor_spec.py` |
| Pi0 spec example | `flash_rt/frontends/torch/_pi0_thor_spec.py` |
| GROOT spec example | `flash_rt/frontends/torch/_groot_thor_spec.py` |
| Pi0-FAST spec example | `flash_rt/frontends/torch/_pi0fast_thor_spec.py` |

Read one of the four spec files when adding a new model — they are the canonical reference and are kept short on purpose.

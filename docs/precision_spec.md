# PrecisionSpec

`PrecisionSpec` is the declarative description of *how* a tensor is
quantized. `ModelPrecisionSpec` is the aggregate for a whole model —
one spec per quantized weight and one per quantized activation point.

The goal is a single representation that spans:

- Scales produced by calibration (today)
- Scales baked into a QAT checkpoint (future)
- Scales that a future kernel dispatcher will consume to pick the right
  kernel variant

Today's shipped kernels only implement one point in this space
(`fp8_e4m3` + `per_tensor` + `symmetric`). The spec exists anyway so
that code written against it doesn't have to change when we add more.

---

## 1. Structure

```python
from flash_rt.core.precision_spec import PrecisionSpec, ModelPrecisionSpec

spec = PrecisionSpec(
    dtype="fp8_e4m3",            # fp32 | fp16 | bf16 | fp8_e4m3 | fp8_e5m2 | nvfp4 | int8 | int4
    granularity="per_tensor",    # per_tensor | per_channel | per_group
    scheme="symmetric",          # symmetric | asymmetric
    group_size=None,             # required when granularity="per_group"

    scale_source="calibration",  # calibration | qat_checkpoint | runtime_dynamic
    scale=np.array([...]),       # shape determined by granularity
    zero_point=None,             # required for asymmetric

    calibration_method="percentile",   # "single_frame" | "percentile" | "qat_fake_quant" | ...
    calibration_samples=64,
    calibration_percentile=99.9,
)
spec.validate()                  # hard-fails on anything outside the v1 capability set
```

### Currently supported

| dtype | granularity | scheme |
|---|---|---|
| `fp8_e4m3` | `per_tensor` | `symmetric` |

Any other combination raises `NotImplementedError`. This is deliberate:
we'd rather the kernel dispatcher be extended explicitly than silently
fall back to a kernel that doesn't match the user's QAT scheme.

### Future combinations (not shipped)

| dtype | granularity | scheme | Unlock requires |
|---|---|---|---|
| `fp8_e4m3` | `per_channel` | `symmetric` | Per-channel FP8 GEMM kernel variant |
| `fp8_e5m2` | `per_tensor` | `symmetric` | E5M2 descale variants |
| `nvfp4` | `per_group` (size 16) | `symmetric` | Already partially shipped on Thor as separate module (`flash_rt_fp4`) |
| `int8` | `per_tensor` | `symmetric` | INT8 GEMM path |
| anything | anything | `asymmetric` | Kernels that consume a zero point |

When adding support for a new combination, remove it from the "not
supported" set in `precision_spec.py`'s `validate()` and add the
corresponding kernel variant to the dispatcher.

---

## 2. Where `PrecisionSpec` comes from

### Calibration (today)

After `rt.calibrate(observations, ...)` completes, the frontend builds
a `ModelPrecisionSpec` with one `PrecisionSpec` per quantized tensor:

```python
rt.calibrate(obs_list, percentile=99.9)
spec = rt.precision_spec                      # ModelPrecisionSpec
print(spec.encoder_layer_specs["layer0.qkv"].scale)
print(spec.source)                            # "calibration"
```

Every calibration-origin spec carries `scale_source="calibration"` plus
the method / N / percentile used.

### QAT checkpoint (future)

A future `load_qat_checkpoint()` path will produce a
`ModelPrecisionSpec` with `source="qat_checkpoint"` and
`scale_source="qat_checkpoint"` on each spec. The scales come from the
checkpoint directly, no calibration forward needed.

### Manual (power users)

You can hand-build a `ModelPrecisionSpec` and attach it to a frontend
for experiments, as long as every constituent spec passes
`validate()` for the shipped kernel capability set.

---

## 3. Persistence

```python
rt.precision_spec.to_json("/path/to/spec.json")
```

Fully round-trips — numpy arrays are emitted as lists. Reload with
`ModelPrecisionSpec.from_json(path)` on another machine to pin the
exact same quantization. Useful for:

- Regression gates ("did the scales drift when I re-calibrated?")
- Shipping the spec alongside the model to a deployment env
- Archival audit ("what was the calibration strategy for this release?")

---

## 4. Design principles

- **Data, not behaviour.** The spec does not carry references to
  kernels, tensors, or CUDA resources. It is a description.
- **Hard-fail on unsupported combinations.** Ambiguous fall-back is a
  trap — if the user meant per-channel FP8 and we silently gave them
  per-tensor, they'd debug the resulting accuracy gap for hours.
- **Stable serialization format.** Fields are kept simple so `to_json`
  / `from_json` survive minor version bumps. When a new field is added,
  give it a default so old JSON still loads.

# Calibration: the framework view

> **Target audience**: anyone who wants to understand calibration as a **framework component** — its cache contract, its lifecycle, what a frontend must implement to use it. For the FP8 numerical mechanics (E4M3 dynamic range, why per-tensor scale, alpha = act_scale × weight_scale, debugging precision regressions) read [`../calibration.md`](../calibration.md) — this doc does not duplicate it.
>
> **Source**: [`flash_rt/core/quant/calibrator.py`](../../flash_rt/core/quant/calibrator.py) — cache implementation. Per-frontend `_calibrate(...)` methods follow the same protocol.

---

## 1. What the calibration framework owns

Calibration is the most cross-cutting piece of FlashRT's quantization story. It is the only component that:

- runs per-checkpoint AND per-sequence-length
- has a disk side-effect (`~/.flash_rt/calibration/`)
- has a 2.5–4 second cost on first run that must not repeat

The framework owns the cross-cutting parts so each frontend doesn't reinvent them:

| Responsibility | Owned by framework | Owned by frontend |
|---|---|---|
| Cache key derivation (checkpoint hash, Se key) | yes | — |
| Cache file location and JSON schema | yes | — |
| Save / load JSON (with `extra` field for model-specific scales) | yes | — |
| The actual measurement pass (1 forward, record amax per FP8 GEMM) | — | yes |
| Compounding `alpha = act_scale × weight_scale` | — | yes |
| Baking alphas as host scalars into captured graph | — | yes |
| Multi-sample refit / AWQ re-tune | — | yes (utilities provided) |

Reading frontend `_calibrate` code makes more sense once you know the framework is doing the boring half — file paths, hashing, JSON I/O.

---

## 2. The cache contract

### 2.1 Cache key

The framework caches per `(checkpoint, sequence_length)`. Concretely:

- **Checkpoint identity** — SHA256 of the first 64 KB of `model.safetensors` (or the equivalent first-shard / Orbax manifest), plus the file size as a discriminator. Truncated to 16 hex chars (64-bit collision-safe). Same model re-saved → same hash. Different fine-tune → different first-layer weights → different hash.
- **Sequence length** — `Se` (encoder seq length) or analogous for the model. Different prompt lengths produce different intermediate buffer shapes and therefore different scales.

Cache file:

```
~/.flash_rt/calibration/{ckpt_hash}_Se{N}.json
```

`save_calibration` and `load_calibration` in [`calibrator.py`](../../flash_rt/core/quant/calibrator.py) are the only public read / write entry points.

### 2.2 What is cached

A model-agnostic JSON shape:

```json
{
  "ckpt_hash":     "abcd1234deadbeef",
  "Se":            968,
  "enc_scales":    [...],   // activation scales, len = num_enc_layers * num_FP8_inputs_per_layer
  "enc_alpha":     [...],   // = enc_scales * enc_w_scales (precompounded for kernel use)
  "ae_scales":     [...],   // action-decoder activation scales (Pi0/Pi0.5)
  "enc_w_scales":  [...],   // weight scales (also stored at load time on the frontend, duplicated here for self-contained cache)
  "extra":         {...}    // free-form per-model fields (e.g. SigLIP scales, GROOT's DiT-cross alphas)
}
```

Models that need extra scale lists put them in `extra`. The framework does not interpret `extra`; the frontend writes and reads it symmetrically.

### 2.3 Cache invalidation

The cache invalidates automatically when:

- the checkpoint changes (different first-64 KB content or different file size → different hash)
- the sequence length changes (different `Se` → different filename)

Cache **does not** invalidate on:

- code changes in the kernel library (you have to delete the cache file by hand)
- changes in calibration sample data (deliberate — the cache is a snapshot of a measurement at one point in time)

To force recalibration: `rm ~/.flash_rt/calibration/{ckpt_hash}_Se*.json`. There is no environment flag to disable the cache; if you want fresh calibration every run, delete the file in your boot script.

---

## 3. The two-phase flow

The framework expects `_calibrate(...)` to follow this shape:

```
phase 1 — measurement
   for each FP8 GEMM site:
       run one forward with calibration sample
       record amax of the activation input tensor
       act_scale = amax / 448         # E4M3 max-magnitude

phase 2 — refit
   for each FP8 GEMM site:
       alpha = act_scale * weight_scale     # f32, never f64 (f64 multiply
                                            # produces a rounding mismatch
                                            # vs CUTLASS's f32-internal alpha
                                            # apply, ~1e-4 cosine drop)
       store alpha as a host scalar in the kernel arg list
```

Phase 1 is a one-shot warmup forward. Phase 2 happens once after measurement and bakes alphas into the structures the captured graph reads. After phase 2, every subsequent inference is `graph.replay()` — calibration is not re-run.

Pi0.5, Pi0, GROOT, and Pi0-FAST all follow this shape. The frontend's `_calibrate` body differs because each model has a different number of FP8 GEMM sites and a different sample shape, but the framework path (cache lookup → run-or-load → cache save → bake alphas) is shared.

---

## 4. The three checkpoint-format paths

The cache key derivation handles three checkpoint layouts the framework auto-detects under `_checkpoint_hash`:

| Layout | What gets hashed |
|---|---|
| Single-file safetensors (`model.safetensors`) | first 64 KB + file size |
| Sharded safetensors (HF, e.g. GROOT, Pi0-FAST) | shard-0 of first 64 KB + total file size; falls back to `model.safetensors.index.json` if the shard naming is unusual |
| Orbax (JAX, e.g. Physical Intelligence ckpts) | `params/manifest.ocdbt` first 64 KB + file size |

Adding a new layout is one elif in `_checkpoint_hash`. The candidate file list is the only place layout knowledge lives.

---

## 5. Frontend integration: the four lines

A frontend's `_calibrate` typically wraps the framework like this:

```python
from flash_rt.core.quant.calibrator import (
    _checkpoint_hash, _cache_path,
    save_calibration, load_calibration,
)

def _calibrate(self, sample_obs):
    ckpt_hash = _checkpoint_hash(self.checkpoint_path)
    cache = _cache_path(ckpt_hash, self.Se)

    if cache.exists():
        d = load_calibration(self.checkpoint_path, self.Se)
        self._bake_alphas_from_cache(d)            # frontend-specific
        return

    # Measurement pass: one forward with hooks recording amax.
    enc_scales, ae_scales, extra = self._measure_amax(sample_obs)

    # Refit: compound alpha = act_scale * weight_scale.
    enc_alpha = self._compound_alpha(enc_scales, self.enc_w_scales)

    save_calibration(
        self.checkpoint_path, self.Se,
        enc_scales=enc_scales, enc_alpha=enc_alpha,
        ae_scales=ae_scales, enc_w_scales=self.enc_w_scales,
        extra=extra,
    )
    self._bake_alphas_from_cache(
        dict(enc_scales=enc_scales, enc_alpha=enc_alpha,
             ae_scales=ae_scales, enc_w_scales=self.enc_w_scales,
             extra=extra)
    )
```

`_measure_amax`, `_compound_alpha`, `_bake_alphas_from_cache` are model-specific. The four lines that touch the cache are framework-level.

---

## 6. Multi-sample calibration (production preset)

A single calibration sample produces scales that are correct for *that* sample. For deployment, the production preset runs 8 stratified LIBERO samples and refits per-site:

- For each site, take the **max amax across all samples** as the activation scale (covers the worst case).
- For some sites that benefit from per-channel awareness, AWQ-style activation-aware refit recomputes weight scales with channel-wise activation statistics.

The framework provides utility helpers under `tests/test_thor_multi_sample_calibrate.py` and `tests/test_thor_calibrate_matrix.py` for running the multi-sample loop and producing a `.json` that has the same shape as the single-sample cache. Frontends that opt into multi-sample (Pi0.5 default, Pi0.5-FP4) write the multi-sample scales into the same cache file — the cache contract does not distinguish single- vs multi-sample data.

This is intentional: the cache stores measured scales; whoever measured them is the frontend's business.

---

## 7. Why the cache key is not just the checkpoint path

Two checkpoints at the same path on different machines will not collide because the cache is `~/`-scoped. But a **fine-tune** of the same base checkpoint at the same path *would* collide if the cache key were just the path. Hashing the actual file content avoids this. Specifically:

- Continuing fine-tune on top of an earlier ckpt → first-layer weights differ → first-64 KB hash differs → new cache file. No stale calibration.
- Renaming the same file → file size unchanged, content unchanged → same hash → cache hit. Correct.
- Truncated / corrupted file → file size differs → cache miss → recalibrates. Correct.

This is why the framework hashes content + size, not the path.

---

## 8. Where this differs from a "compile-time" calibration

In TensorRT, calibration is a build-time concern — the calibrator runs during `trtexec`, the result is baked into a `.engine` file, and you cannot recalibrate without rebuilding. FlashRT's calibration:

- runs at **first inference**, not at build
- writes a portable JSON (not a binary engine)
- can be force-recalibrated by deleting one file
- does not require a build / link / restart cycle to apply

This is the same property that makes "no compile, no export" possible: calibration is data, the kernels read it as a host scalar, and there is no compiler stage that needs to re-bake anything.

---

## 9. Where to look in the source

| What | File |
|---|---|
| Cache key + save / load + path constants | [`flash_rt/core/quant/calibrator.py`](../../flash_rt/core/quant/calibrator.py) |
| Mechanics doc (FP8 dynamic range, scales, alpha math) | [`../calibration.md`](../calibration.md) |
| Pi0.5 frontend `_calibrate` | `flash_rt/frontends/torch/pi05_thor.py` |
| GROOT frontend `_calibrate` | `flash_rt/frontends/torch/groot_thor.py` |
| Multi-sample helper | `tests/test_thor_multi_sample_calibrate.py` |
| AWQ refit utility | `flash_rt/core/quant/awq.py` (when present) |

---

## 10. Stable contract

The following are guaranteed across v0.x releases:

1. Cache directory `~/.flash_rt/calibration/`.
2. Cache filename pattern `{ckpt_hash}_Se{N}.json`.
3. `_checkpoint_hash(path)` is stable: the same file always produces the same hash.
4. JSON top-level keys: `ckpt_hash`, `Se`, `enc_scales`, `enc_alpha`, `ae_scales`, `enc_w_scales`, `extra`. New keys may appear; existing keys do not change name or semantic.
5. `extra` is owned by the frontend; the framework treats it as opaque.

Not guaranteed:

- Internal helpers `_checkpoint_hash`, `_cache_path` — use the public `save_calibration` / `load_calibration` entry points.
- The JSON values themselves (those depend on the frontend that wrote them).

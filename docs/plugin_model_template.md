# FlashRT — Plugin Model Template

This document shows how an **external package** integrates a new VLA model
into FlashRT without modifying the open-source repo. This is the intended
path for closed-source or partner-specific models (e.g. Pi0.6, Pi0.7).

No actual plugin code lives in this repo — this is purely documentation.

---

## Overview

FlashRT's dispatch is driven by `_PIPELINE_MAP` in
`flash_rt.hardware.__init__`. A plugin registers itself by:

1. Implementing `AttentionBackend` for the new model's attention shape.
2. Writing a pipeline class (model logic, framework-agnostic).
3. Writing a frontend class (weight loading, graph capture, `infer()`).
4. Monkey-patching `_PIPELINE_MAP` to route `(config, framework, arch)`
   to the plugin's frontend.

After registration, `flash_rt.load_model(config="mymodel")` works
exactly like the built-in models.

---

## Step 1: Implement `AttentionBackend`

```python
# mymodel_plugin/attention.py

from flash_rt.hardware.backend import AttentionBackendBase, AttentionSpec

class MyModelAttentionBackend(AttentionBackendBase):
    """Attention backend for MyModel on RTX (flash_attn)."""

    def __init__(self, spec: AttentionSpec):
        super().__init__(spec)
        # Allocate Q/K/V torch tensors for each (site, layer) ...
        # See flash_rt.hardware.rtx.attn_backend.TorchFlashAttnBackend
        # for a working example.

    def get_slot_ptrs(self, site: str, layer_idx: int) -> dict[str, int]:
        # Return {"Q": int_ptr, "K": int_ptr, "V": int_ptr}
        ...

    def run(self, site, layer_idx, q_seq, *, kv_seq=None, stream=0) -> int:
        # Dispatch flash_attn_func / fvk.attention_* and return output ptr
        ...
```

If the model's attention shape is similar to an existing model, consider
reusing `TorchFlashAttnBackend` (Pi0.5 pattern) or
`TorchFlashAttnBackendGroot` (multi-site pattern) directly. These are
importable from `flash_rt.hardware.rtx.attn_backend[_groot]`.

---

## Step 2: Write the pipeline

```python
# mymodel_plugin/pipeline.py

from flash_rt.core.cuda_buffer import CudaBuffer

class MyModelPipeline:
    """Framework-agnostic pipeline. Operates on raw device pointers."""

    def __init__(self, gemm, fvk, attn_backend, weights, **kwargs):
        self.attn = attn_backend
        # Allocate working buffers as CudaBuffer ...

    def run_pipeline(self, stream=0):
        # Vision encoder → backbone → decoder/DiT → actions
        # Use self.attn.run("site", layer, q_seq=N) for attention
        ...

    def forward(self):
        # Replay captured CUDA graph or fall back to run_pipeline
        ...
```

The pipeline should:
- Accept an `AttentionBackend` instance (injected by the frontend).
- Use `fvk.*` and `gemm.*` for all compute (pointer-based kernels).
- Not import `torch` or `jax` — those belong in the frontend.

---

## Step 3: Write the frontend

```python
# mymodel_plugin/frontend_torch.py

import torch
from mymodel_plugin.pipeline import MyModelPipeline
from mymodel_plugin.attention import MyModelAttentionBackend
from flash_rt.hardware.backend import AttentionSpec

class MyModelTorchFrontend:
    """Load safetensors checkpoint → build pipeline → infer."""

    def __init__(self, checkpoint_dir, num_views=2, autotune=3, **kwargs):
        # 1. Load weights from safetensors
        # 2. FP8 quantize large GEMM weights
        # 3. Build AttentionSpec and create backend
        spec = AttentionSpec()
        spec.add_site("encoder", num_layers=24, num_q_heads=16,
                       num_kv_heads=8, head_dim=128, max_q_seq=512)
        # ... add more sites ...
        self.attn_backend = MyModelAttentionBackend(spec)
        # 4. Build pipeline
        self.pipeline = MyModelPipeline(gemm, fvk, self.attn_backend, weights)
        # 5. Capture CUDA graph
        self.pipeline.record_infer_graph(...)

    def set_prompt(self, prompt_text):
        # Tokenize, embed, upload to pipeline
        ...

    def infer(self, observation) -> dict:
        # Upload images + noise, replay graph, download actions
        ...
        return {"actions": actions_numpy}
```

The frontend must expose `set_prompt()` and `infer()` — these are the
methods that `VLAModel.predict()` calls.

---

## Step 4: Register in `_PIPELINE_MAP`

```python
# mymodel_plugin/__init__.py

from flash_rt.hardware import _PIPELINE_MAP

# Register for RTX 5090
_PIPELINE_MAP[("mymodel", "torch", "rtx_sm120")] = (
    "mymodel_plugin.frontend_torch", "MyModelTorchFrontend"
)

# Register for Thor (if you have a Thor backend)
_PIPELINE_MAP[("mymodel", "torch", "thor")] = (
    "mymodel_plugin.frontend_torch_thor", "MyModelThorFrontend"
)
```

After this import, `flash_rt.load_model(config="mymodel")` will find
and instantiate `MyModelTorchFrontend`.

**Import order matters**: the plugin's `__init__.py` must run before
`load_model()` is called. Typical patterns:

```python
# Option A: explicit import in user code
import mymodel_plugin  # registers _PIPELINE_MAP entries
import flash_rt
model = flash_rt.load_model(config="mymodel", checkpoint="...")

# Option B: entry_points (setuptools)
# In mymodel_plugin's setup.cfg:
#   [options.entry_points]
#   flash_rt.plugins =
#       mymodel = mymodel_plugin
# FlashRT does NOT auto-discover entry_points today, but a future
# version may. For now, use Option A.
```

---

## What your plugin depends on (stable API)

Your plugin should import **only** from the stable API surface documented
in [stable_api.md](stable_api.md):

| Import | Purpose |
|---|---|
| `flash_rt.hardware.backend.AttentionBackend` | Protocol to implement |
| `flash_rt.hardware.backend.AttentionSpec` | Build attention site specs |
| `flash_rt.hardware.backend.SiteSpec` | Attention site descriptor |
| `flash_rt.hardware._PIPELINE_MAP` | Registration point |
| `flash_rt.core.cuda_buffer.CudaBuffer` | GPU buffer allocation |
| `flash_rt.core.cuda_graph.CUDAGraph` | CUDA graph capture/replay |
| `flash_rt.core.quant.calibrator.*` | FP8 calibration cache |

Everything else (`flash_rt.models.*`, `flash_rt.frontends.*`,
`flash_rt.hardware.rtx.*` internals) is internal and may change.

---

## Testing your plugin

```python
# Smoke test
import mymodel_plugin
import flash_rt

model = flash_rt.load_model(
    config="mymodel",
    checkpoint="/path/to/ckpt",
    framework="torch",
)
model.predict(images=[img1, img2], prompt="test prompt")

# Verify dispatch
from flash_rt.hardware import resolve_pipeline_class, detect_arch
cls = resolve_pipeline_class("mymodel", "torch", detect_arch())
assert cls.__name__ == "MyModelTorchFrontend"
```

---

## Development constraints (hard rules)

Reiterated from [`docs/adding_new_model.md` §0](adding_new_model.md)
so you have them in one place while writing your plugin. Violating
these turns into 3–5 day debug sessions later.

1. **One file per `(model, hardware)` compute path.**
   `models/<m>/pipeline_<hw>.py` where `<hw>` ∈ `{thor, rtx}`. No
   default `pipeline.py`. No runtime hardware forks like
   `if self._has_sm100`. If Thor and RTX overlap 90% — still two
   files; share via plain function imports.
2. **One file per `(model, framework, hardware)` IO path.** Class
   name `<Model><Fw>Frontend<Hw>`. Same no-fork rule: Thor torch
   and RTX torch are separate files.
3. **`hardware/<hw>/shared_primitives.py` is a closed set.** Only
   model-agnostic helpers (`_gpu_*`, `_measure_scale_gpu`,
   `siglip_forward`, `encoder_forward`, `encoder_forward_calibrate`).
   Anything model-specific belongs in `models/<m>/pipeline_<hw>.py`.
4. **`_PIPELINE_MAP` is strictly 1:1.** Each `(model, framework, hw)`
   tuple routes to exactly one file and one class. No tuple shares
   a class via runtime branching — that was the `pi0fast`
   anti-pattern and is explicitly forbidden for new models.
5. **Forward functions take pointers and plain Python ints.** No
   `torch.empty()`, `.cpu()`, `.numpy()`, `sync()`, or `F.*` inside
   a forward. Allocate everything in `_load_weights`.
6. **Attention goes through `attn.run(site, layer_idx, ...)`**. Do
   not call `fvk.attention_qkv_fp16` directly. The protocol is
   documented in [`docs/stable_api.md`](stable_api.md).
7. **Weight-spec op order must byte-match the reference.** See the
   precision-debug Q&A below — wrong op order is the #1 source of
   `cos ≈ 0.5` at first-light.
8. **Test with real (in-distribution) inputs, not `torch.randn`.**
   Random inputs produce unrepresentative activation distributions
   that corrupt FP8 calibration scales and make diagnoses lie to
   you. Always use a real benchmark frame for cosine and latency.

---

## Precision-debug Q&A (accumulated experience)

> This section is **high-value**. Each pattern below corresponds to
> a real debug session I went through while building Pi0 / Pi0.5 /
> GROOT — the specific cos ranges and their typical root causes
> are reproducible signals, not superstition. Use this as a routing
> table before you spend a day profiling.

### First-light cosine routing table

| Observed cos (vs FP32 reference) | Most likely root cause | First move |
|---|---|---|
| **≈ 1.000** (≥ 0.999) | Nothing — you're done | Ship it, run latency bench |
| **0.997–0.999** | FP8 calibration noise floor. Normal for a correctly-wired pipeline. | Run 3× A/B to confirm it's steady, not drifting. |
| **0.95–0.997** | Compound fp16/bf16 ULP drift amplified by FP8 (~225 attention ops in Pi0 will do this). Typically a minor kernel-path mismatch — e.g. wrong FA `num_splits` heuristic, or a kernel template instantiation that differs from the reference. | Compare kernel dispatch args against reference (site-level bisect below). |
| **0.85–0.95** | Single-point divergence somewhere mid-pipeline. Usually FP8 scale mis-calibration on one layer, or a fused kernel picking up the wrong weight buffer. | Layer-by-layer bisect. See "Divergence-point bisect" below. |
| **0.75–0.85** | Structural, but not catastrophic. Wrong norm weight fused with QKV, wrong RoPE table, or an `.unsqueeze` that silently flipped batch/seq axes. | Dump intermediate activations and compare shapes first. |
| **0.5–0.75** | Systematic axis error. QKV interleave count wrong for GQA (used `num_heads` where `num_kv_heads` belonged), or `.T.contiguous()` vs `.t().contiguous()` applied inconsistently. | Audit weight-spec op order byte-for-byte against the reference. |
| **0.1–0.5** | Catastrophic. Wrong checkpoint, wrong tokenizer, wrong image normalization, or a `.contiguous()` forgotten on a view that got fed to the kernel. | Verify the FP32 reference itself runs and its checkpoint hash matches. |
| **≈ 0 (random)** | Dead kernel — output is junk memory. Likely never got written to (missing `.copy_()` / wrong output ptr), or wrong dtype being read. | Dump the output tensor directly; should not be all zeros/NaN. |

Golden rule: **"0/3 is always a bug, never noise"** — if you cannot
reproduce cos ≥ 0.998 three times in a row, you have a real defect.

### Divergence-point bisect (the single most valuable technique)

If first-light cos is in the 0.75–0.95 range, the fastest way to
localize is to hook each site's forward and compare against a
reference. I used this exact pattern three times on the Pi0 FA2
integration:

```python
# Pseudocode — install hooks on attn_backend before calibrate_with_real_data
# so the pipeline's CUDA Graph capture sees the hooked calls.

real_vision_attn = rt.attn_backend.vision_attn
def hooked(stream=0):
    # SYNC BEFORE SNAPSHOT — otherwise you read stale junk from another stream
    rt._cudart.cudaStreamSynchronize(ctypes.c_void_p(stream))
    q = rt.attn_backend.vis_Q.clone()
    k = rt.attn_backend.vis_K.clone()
    v = rt.attn_backend.vis_V.clone()

    ptr = real_vision_attn(stream=stream)  # run the real kernel (fa2/pip_fa/...)
    my_out = torch_view_at(ptr)

    # Ground-truth reference on the SAME Q/K/V snapshot
    ref_out = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1,2).float(), k.transpose(1,2).float(), v.transpose(1,2).float(),
        dropout_p=0.0).transpose(1,2).to(torch.float16)

    cos = cosine(my_out, ref_out)
    print(f"{LAYER_IDX}: cos={cos:.6f}  max|delta|={(my_out.float()-ref_out.float()).abs().max():.5f}")
    return ptr

rt.attn_backend.vision_attn = hooked
rt.pipeline._graph = None   # disable CUDA Graph so hooks run eagerly
rt.infer(obs)
```

**Critical gotcha**: the `cudaStreamSynchronize` is not optional.
I once spent two hours debugging "layer 1 diverges to cos=0.035"
— turned out the Q/K/V snapshot was reading pre-write garbage
because the pipeline's write was still in-flight on a different
stream. After adding the sync, all layers matched at cos=1.000000.

### Site-level bisect via env var

Narrow the blame radius with a per-site switch before going to
layer-level hooks. `RtxFlashAttnBackend` exposes:

```bash
FVK_RTX_FA2=1 FVK_RTX_FA2_SITES=siglip          python rtx_pi0_cosine_vs_official.py
FVK_RTX_FA2=1 FVK_RTX_FA2_SITES=encoder         python rtx_pi0_cosine_vs_official.py
FVK_RTX_FA2=1 FVK_RTX_FA2_SITES=decoder         python rtx_pi0_cosine_vs_official.py
FVK_RTX_FA2=1 FVK_RTX_FA2_SITES=siglip,encoder  python ...
```

If encoder and decoder pass alone but siglip fails, the bug is in
the siglip path. Same pattern works for any plugin — expose a
per-site env var on your backend and you get O(log N)
localization.

### Isolating FP8 quantization noise

If the pipeline passes in pure fp16 but fails with FP8 enabled,
the issue is a calibration mismatch — typically your new kernel
produces output 1–2 ULPs off from the reference, and FP8
quantization amplifies that into bucket flips.

```bash
# Pi0 / Pi0.5 has an env var that short-circuits FP8:
PI0_DEBUG_NO_FP8=1 python rtx_pi0_cosine_vs_official.py     # should get cos ~= 1.000
# If this passes and the FP8 run fails at cos 0.98, it's calibration-sensitivity.
```

For your own plugin, add a similar `use_fp8=False` path that skips
the quantize fusions and runs `gmm_fp16` GEMMs. This lets you
isolate "kernel is wrong" from "kernel is right but FP8-sensitive".

### Determinism & noise control

Before declaring a cos value "final", check it's reproducible:

1. Run the test 3× with the same seed. Variance > 0.0005 means your
   noise sampler or FP8 calibration is not deterministic.
2. Compare against both `cos vs FP32` and `cos vs FP16` references.
   If vs-FP16 is tight (> 0.9995) but vs-FP32 is loose (0.997), you
   are hitting the natural FP16 precision ceiling — this is normal
   for any model, not a regression.
3. Run baseline (pip flash_attn, `FVK_RTX_FA2=0`) and new path back
   to back. Variance between them > 0.002 at cos level is a real
   signal; < 0.002 is likely noise.

### Before you report "it's just noise"

I've seen this exact excuse used three times to paper over real
bugs, each time costing 1+ week when they resurfaced. Checklist
before calling anything "noise":

- [ ] The bug is **reproducible** — same inputs always give the
      same bad cos.
- [ ] The bug **doesn't correlate with any code path you control**
      — toggling your change does not change the outcome.
- [ ] The variance is < 0.0005 cos, not 0.002.
- [ ] The failure rate is < 1/20, not 1/3.

If any of the above is false, it's a bug. Find the root cause.

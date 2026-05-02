# FlashVLA New-Model Template

Skeleton package for adding a new VLA model to FlashVLA. Targets the
**most common shape**: vision encoder (SigLIP-style) + LLM backbone
(Gemma/LLaMA-style transformer with KV cache) + diffusion-style
action decoder. If your model is autoregressive (Pi0-FAST-like) or
multi-stage (GROOT-like), copy this template anyway and read
`pi0fast.py` / `groot_thor.py` for the structural delta.

**Audience**: ML-infra engineers who already have the source model
running in PyTorch and want to deploy it on FlashVLA-supported
hardware (Thor / RTX). You should already understand FP8
quantization, KV cache layouts, and CUDA Graph capture. We do not
re-explain those here.

---

## Files in this template

```
_template/
├── README.md          (this file — read first, ~5 min)
├── frontend.py        (ENTRY POINT — load weights, capture graph, infer)
├── pipeline.py        (model-specific compute — forward functions)
├── weights_spec.py    (declarative weight loader + FP8 quant decisions)
└── attention.py       (AttentionSpec + backend wrapper)
```

For your new model `mymodel`, the final layout will be:

```
flash_vla/
├── frontends/torch/mymodel_thor.py     ← copy of frontend.py
├── frontends/torch/_mymodel_thor_spec.py ← copy of weights_spec.py
├── models/mymodel/pipeline_thor.py     ← copy of pipeline.py
└── hardware/thor/attn_backend.py       ← add make_mymodel_attention_spec from attention.py
```

(Same shape for RTX — append `_rtx` instead of `_thor`. Hard rule: one
file per `(model, hardware)`. See `docs/adding_new_model.md` §0.)

---

## Reading order

1. **README.md** (this file) — understand the file split and what
   each file is responsible for
2. `weights_spec.py` — start here. Mapping your checkpoint's tensor
   names to the FlashVLA weight slots is the most concrete first step.
3. `attention.py` — only ~60 lines, declares your model's attention
   sites (encoder / decoder / vision)
4. `pipeline.py` — translate your model's forward pass into a sequence
   of `fvk.*` kernel calls. **This is where the ML-infra work happens.**
5. `frontend.py` — wire it up, capture CUDA Graph, expose `set_prompt`
   / `infer`

Each file has a `# WHAT YOU TRANSLATE` block at the top showing the
mapping pattern: your model's PyTorch code on the left, our kernel
calls on the right. Use that as the source-of-truth for "what does
this step correspond to in my model".

---

## Model-code → FlashVLA mapping (the high-level mental model)

| Your model code | FlashVLA equivalent | Where it lives in template |
|---|---|---|
| `MyModel.from_pretrained(ckpt)` (loads safetensors) | `_load_weights()` (loads safetensors + does FP8 quant) | `frontend.py` STEP 1 + `weights_spec.py` |
| `state_dict["encoder.layers.5.self_attn.q_proj.weight"]` | `WEIGHT_SPEC` entry: `("encoder", 5, "qkv")` → `Quant("fp8")` | `weights_spec.py` |
| `model.eval(); with torch.no_grad():` | (implicit — we never train) | n/a |
| `out = model.encoder(images, text)` | `encoder_forward(ctx, fvk, bufs, weights, dims)` | `pipeline.py` |
| `out = model.decoder(noise, encoder_kv, steps=10)` | `decoder_forward(ctx, fvk, bufs, weights, dims)` | `pipeline.py` |
| `F.rms_norm(x, weight)` | `fvk.rms_norm_fp16(x_ptr, w_ptr, out_ptr, S, D, eps, stream)` | inside `pipeline.py` |
| `F.scaled_dot_product_attention(q, k, v)` | `attn.run("encoder", layer_idx, q_seq=S, stream=stream)` | inside `pipeline.py`, dispatched via `attention.py` |
| `silu(gate) * up` | `fvk.silu_mul_split_fp8_fp16(...)` (true SiLU) **or** `fvk.gate_geglu_fp16(...)` (GEGLU/Pi0.5-style) | inside `pipeline.py` |
| `q @ k.T` (any GEMM) | `gemm.bf16_nn(...)` (bf16) or `fvk.gemm_fp8_*(...)` (fp8); decided by `WEIGHT_SPEC` | `pipeline.py` |
| `tokenizer(prompt)` | `set_prompt(prompt)` (your frontend method, calls into shared embed util) | `frontend.py` STEP 4 |
| First `model(...)` call (warm-up) | `_recalibrate_with_real_data()` + `_capture_enc_ae_graph()` | `frontend.py` STEP 5 |
| Subsequent `model(...)` calls | `infer(obs)` → `cudaGraphLaunch(self._enc_ae_graph)` | `frontend.py` STEP 6 |

**Key insight**: FlashVLA replaces your model's `forward()` with a
fully captured CUDA Graph. The "code" you write in `pipeline.py` is
NOT executed at inference time — it runs once during graph capture,
and from then on the graph is replayed. This is why all kernel
calls take raw pointers (no torch tensor allocation inside forward).

---

## Done checklist (before you delete this README)

- [ ] You have a single safetensors checkpoint of your source model
      and can run it in plain PyTorch to get a reference output
- [ ] You can extract the `state_dict` and enumerate every weight
      tensor's name + shape + dtype
- [ ] You know your model's: # encoder layers, # decoder layers,
      hidden dim, # heads, # KV heads (GQA), head dim, RoPE
      base/period, action dim, # diffusion steps
- [ ] You've identified each attention site (vision-self,
      encoder-self, decoder-self, decoder-cross — depends on model)
- [ ] You decided which GEMMs to FP8-quantize (default: all FFN +
      QKV proj + output proj that are >4M FLOPS; below that, skip
      QDQ overhead)

If you cannot answer one of these, stop and finish your reference
PyTorch implementation first. Trying to integrate a model whose
forward you don't fully understand always ends in a 3-day cosine
debug session.

---

## After your model works

1. Delete every `# TODO`, `# WHAT YOU TRANSLATE`, and `# STEP N`
   marker from your copies of these files
2. Move the files to their final paths (see "Files in this template"
   above)
3. Add 4 lines to `flash_vla/hardware/__init__.py::_PIPELINE_MAP` to
   register your frontend
4. Write a config YAML at `flash_vla/configs/mymodel.yaml` (see any
   existing config for the shape)
5. Add one segment to `tests/test_all_models_precision.py`
6. Run the full validation protocol in `docs/adding_new_model.md` §3

The hard part is `pipeline.py` — translating your model's forward
into kernel calls. Budget 60-70% of your total time on it.

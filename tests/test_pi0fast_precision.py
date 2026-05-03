#!/usr/bin/env python3
"""FlashRT Pi0-FAST precision test (autoregressive variant).

Counterpart to test_all_models_precision.py, specialized for the
autoregressive Pi0-FAST architecture.

Differences from the diffusion-model test:
  - No matched noise injection (decoding is greedy/deterministic by argmax).
  - Cannot compare action arrays directly: pi0_fast_base is a multi-task base
    model, both backends produce zero/garbage actions on out-of-distribution
    dummy input. The validation metric is **per-segment LLM cosine vs JAX
    bf16 reference**, which directly answers whether the Torch/JAX backends
    are numerically equivalent to the JAX reference at the LLM level.
  - Latency is measured separately for prefill and per-token decode.

Tested backends:
  pi0fast_torch  : ThorPipelineTorchPi0Fast (safetensors checkpoint)
  pi0fast_jax    : ThorPipelineJaxPi0Fast (Orbax checkpoint)

Validation criteria (each on identical post-SigLIP+PostLN prefix):
  S2  prefill last hidden state      cosine vs JAX bf16  >= 0.98
  S2b first logit (before argmax)    cosine vs JAX bf16  >= 0.98
  S3  decode step 0 dec_xn           cosine vs JAX bf16  >= 0.98
  S4  decode step 0 logit            cosine vs JAX bf16  >= 0.98

The 0.98 threshold reflects the calibration noise floor for FP8 autoregressive
inference: Pi0/Pi0.5 (diffusion) routinely run at cos > 0.99, but Pi0-FAST's
greedy autoregressive decode is more sensitive to per-token logit perturbations.
Per empirical experience: cos < 0.8 indicates a computation alignment bug;
cos >= 0.8 is calibration/quantization noise territory.

Each backend run is isolated in a subprocess for clean state and to prevent
JAX/PyTorch GPU contention.

Usage:
    python3 tests/test_pi0fast_precision.py
    python3 tests/test_pi0fast_precision.py --backend torch
    python3 tests/test_pi0fast_precision.py --backend jax
"""
import argparse, subprocess, sys, os, json
import numpy as np

FLASH_VLA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Common dummy input — deterministic so subsequent runs are reproducible.
# Random uint8 images + zero state. Pi0-FAST is multi-task, so the model can
# accept any input shape; the goal is FVK ≡ JAX equivalence, not action quality.
import os as _os
_CKPT_BASE = _os.environ.get("PI0FAST_CKPT_BASE", "/workspace/checkpoints")
_PT_BASE   = _os.environ.get("PI0FAST_PT_BASE", "/workspace/pytorch_checkpoints")
TORCH_CHECKPOINT = f"{_PT_BASE}/pi0_fast_base_converted"
JAX_CHECKPOINT = f"{_CKPT_BASE}/pi0_fast_base"

# ══════════════════════════════════════════════════════════════
# Stage 1: capture FVK prefix + run prefill + decode step 0
# Used for both Torch and JAX backends. Saves intermediates to
# /tmp/pi0fast_prec_<backend>.npz for the JAX reference comparison.
# ══════════════════════════════════════════════════════════════
FVK_TEMPLATE = '''
import sys, os, time, json, numpy as np, torch
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, "ROOTDIR")
import logging; logging.basicConfig(level=logging.WARNING)

import flash_rt.flash_rt_kernels as fvk
from flash_rt.frontends.PIPELINE_MODULE import PIPELINE_CLASS
from flash_rt.models.pi0fast.pipeline import (
    prefill_forward_pi0fast, decode_step_pi0fast_bf16,
)

pipe = PIPELINE_CLASS("CHECKPOINT_PATH", num_views=NUM_VIEWS,
                      autotune=0, use_cuda_graph=False)
pipe.set_prompt("pick up the red block",
                state=np.zeros(32, dtype=np.float32))

np.random.seed(42)
imgs = []
for _ in range(NUM_VIEWS):
    imgs.append(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
images = np.stack([(im.astype(np.float32) / 127.5 - 1.0).astype(np.float16) for im in imgs])
pipe._img_buf.upload(images)
pipe._siglip_graph.replay()
torch.cuda.synchronize()

Se = pipe.Se; De = pipe.De; He = pipe.He; Le = pipe.Le; NHe = pipe.NHe; HDe = pipe.HDe

# Save the prefix bytes (BF16 → FP32 for JAX comparison)
prefix_fp32 = pipe._enc_x[:Se].float().cpu().numpy()

# Run prefill
pipe._Kc.zero_(); pipe._Vc.zero_()
prefill_bufs = {
    "x": pipe._enc_x.data_ptr(), "x_fp8": pipe._enc_x_fp8.data_ptr(),
    "qkv": pipe._enc_qkv_buf.data_ptr(), "logits": pipe._enc_logits.data_ptr(),
    "attn_out": pipe._enc_attn.data_ptr(), "o_fp8": pipe._enc_o_fp8.data_ptr(),
    "gate": pipe._enc_gate.data_ptr(), "hid_fp8": pipe._enc_hid_fp8.data_ptr(),
    "fg": pipe._enc_fg.data_ptr(), "xn": pipe._enc_xn.data_ptr(), "ctx": pipe._ctx,
}
prefill_weights = {
    # SM100 path
    "qkv_w": [w.data_ptr() for w in pipe._enc_qkv_w],
    "o_w":   [w.data_ptr() for w in pipe._enc_o_w],
    "gate_w":[w.data_ptr() for w in pipe._enc_gu_w],
    "down_w":[w.data_ptr() for w in pipe._enc_d_w],
    "alpha_host": pipe._enc_alpha_host,
    # SM120 path (flat cuBLASLt buffers + device w_scales)
    "qkv_w_flat":  pipe._enc_qkv_flat.data_ptr(),
    "o_w_flat":    pipe._enc_o_flat.data_ptr(),
    "gate_w_flat": pipe._enc_gu_flat.data_ptr(),
    "down_w_flat": pipe._enc_d_flat.data_ptr(),
    "w_scales":    pipe._enc_w_dev.data_ptr(),
    # Shared
    "rope": pipe._enc_rope.data_ptr(),
    "Kc": pipe._Kc.reshape(-1).data_ptr(),
    "Vc": pipe._Vc.reshape(-1).data_ptr(),
    "final_norm_w": pipe._final_norm_w.data_ptr(),
    "act_scales": pipe._enc_calib_scales.data_ptr(),
}
prefill_dims = {"Se": Se, "D": De, "H": He, "NH": NHe, "HD": HDe,
                "L": Le, "total_keys_max": pipe.max_total_keys}
prefill_forward_pi0fast(pipe._gemm, fvk, prefill_bufs, prefill_weights, prefill_dims)
torch.cuda.synchronize()

# Capture S2: prefill last hidden
prefill_xn_last = pipe._enc_xn[Se - 1].float().cpu().numpy()

# S2b: first logit (correct [D, V] layout via .T view)
last_hidden = pipe._enc_xn[Se - 1:Se].to(torch.float16)
torch.matmul(last_hidden, pipe.embedding_weight.T, out=pipe._logit_buf)
torch.cuda.synchronize()
first_logit = pipe._logit_buf[0].float().cpu().numpy()
first_token = int(first_logit.argmax())

# Decode step 0 with BF16 path
token_embed = pipe.embedding_weight[first_token] * float(De ** 0.5)
pipe._dec_x_bf16.copy_(token_embed.unsqueeze(0))

decode_bufs = {
    "x": pipe._dec_x_bf16.data_ptr(), "x_fp8": pipe._dec_x_fp8.data_ptr(),
    "qkv": pipe._dec_qkv.data_ptr(), "logits": pipe._dec_logits.data_ptr(),
    "attn_out": pipe._dec_attn.data_ptr(), "o_fp8": pipe._dec_o_fp8.data_ptr(),
    "gate": pipe._dec_gate.data_ptr(), "hid_fp8": pipe._dec_hid_fp8.data_ptr(),
    "fg": pipe._dec_fg_bf16.data_ptr(), "xn": pipe._dec_xn_bf16.data_ptr(),
    "fg_scratch": pipe._dec_fg_scratch.data_ptr(),
}
decode_weights = {
    "qkv_w_flat": pipe._enc_qkv_flat.data_ptr(),
    "o_w_flat": pipe._enc_o_flat.data_ptr(),
    "gate_w_flat": pipe._enc_gu_flat.data_ptr(),
    "alpha_host": pipe._enc_alpha_host,
    "down_w_flat": pipe._enc_d_flat.data_ptr(),
    "rope_base": pipe._full_rope.data_ptr(),
    "Kc": pipe._Kc.reshape(-1).data_ptr(),
    "Vc": pipe._Vc.reshape(-1).data_ptr(),
    "final_norm_w": pipe._final_norm_w.data_ptr(),
    "act_scales": pipe._enc_calib_scales.data_ptr(),
    "w_scales": pipe._enc_w_dev.data_ptr(),
}
# Optionally opt into the SM100-only cutlass_fp8_wide gate+up fast-path
# (1.41x speedup on Thor, not available on SM120 cuBLASLt).
if hasattr(fvk, "cutlass_fp8_wide"):
    decode_weights["gate_w_list"] = [w.data_ptr() for w in pipe._enc_gu_w]
decode_dims = {"D": De, "H": He, "NH": NHe, "HD": HDe, "L": Le,
               "prefill_len": Se, "total_keys_max": pipe.max_total_keys}
decode_step_pi0fast_bf16(pipe._ctx, fvk, decode_bufs, decode_weights, decode_dims, step=0)
torch.cuda.synchronize()

dec_xn_step0 = pipe._dec_xn_bf16[0].float().cpu().numpy()
dec_xn_fp16 = pipe._dec_xn_bf16.to(torch.float16)
torch.matmul(dec_xn_fp16, pipe.embedding_weight.T, out=pipe._logit_buf)
torch.cuda.synchronize()
dec_logit_step0 = pipe._logit_buf[0].float().cpu().numpy()

# Save artifacts for the JAX comparison subprocess
np.savez("/tmp/pi0fast_prec_BACKEND.npz",
         prefix=prefix_fp32, Se=np.array(Se),
         prefill_xn_last=prefill_xn_last,
         first_logit=first_logit,
         first_token=np.array(first_token),
         dec_xn_step0=dec_xn_step0,
         dec_logit_step0=dec_logit_step0)

# Latency: full pipe.infer() with 50 decode steps
obs = {"image": imgs[0], "wrist_image": imgs[1] if NUM_VIEWS > 1 else imgs[0]}
if NUM_VIEWS >= 3:
    obs["wrist_image_right"] = imgs[2]
obs["state"] = np.zeros(32, dtype=np.float32)

# Warmup (also triggers any lazy real-data recalibration)
for _ in range(2):
    pipe.infer(obs, max_steps=50)

# Benchmark
prefill_lat = []
per_tok_lat = []
total_lat = []
for _ in range(5):
    r = pipe.infer(obs, max_steps=50)
    prefill_lat.append(r["prefill_ms"])
    per_tok_lat.append(r["per_token_ms"])
    total_lat.append(r["latency_ms"])
prefill_lat.sort(); per_tok_lat.sort(); total_lat.sort()
mid = len(prefill_lat) // 2

print(json.dumps({
    "Se": int(Se),
    "first_token": int(first_token),
    "prefill_p50_ms": round(prefill_lat[mid], 1),
    "per_tok_p50_ms": round(per_tok_lat[mid], 2),
    "total_p50_ms_50tok": round(total_lat[mid], 1),
}))
'''


# ══════════════════════════════════════════════════════════════
# Stage 2: JAX reference (gemma_fast.Module.apply with same prefix)
# Loaded once, reused for both Torch and JAX backend comparisons.
# ══════════════════════════════════════════════════════════════
JAX_REF_TEMPLATE = '''
import sys, os, json, numpy as np
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false --xla_gpu_autotune_level=0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
# Force JAX to use the native CUDA backend instead of mlir_tensorrt.
# On RTX 5090 (SM120) the mlir_tensorrt JAX plugin miscompiles
# gemma_fast's dynamic_slice ops: TensorRT 10.13 reports
# "ISliceLayer has out of bounds access on axis 2" and silently falls
# back to a kernel that produces half-wrong output (cos ~0.54 vs FVK,
# first_token flips from 4022 to 13722). Forcing the cuda backend
# gives cos ~0.97 and correct first_token. Harmless on Thor where
# mlir_tensorrt isn't active.
os.environ.setdefault("JAX_PLATFORMS", "cuda")
_openpi_paths = [
    "<openpi_src>",
    "<openpi_src>",
]
for _p in _openpi_paths:
    if os.path.isdir(_p):
        sys.path.insert(0, _p)
        break
import jax, jax.numpy as jnp
from openpi.models.model import restore_params
import openpi.models.gemma_fast as _gemma

_JAX_CKPT = "JAX_CKPT_PATH"
raw = restore_params(f"{_JAX_CKPT}/params", restore_type=np.ndarray)
data = np.load("/tmp/pi0fast_prec_BACKEND.npz")
prefix = data["prefix"]; Se = int(data["Se"])
fvk_prefill_xn = data["prefill_xn_last"]
fvk_first_logit = data["first_logit"]
fvk_first_token = int(data["first_token"])
fvk_dec_xn = data["dec_xn_step0"]
fvk_dec_logit = data["dec_logit_step0"]

config = _gemma.get_config("gemma_2b")
llm = _gemma.Module(**config, embed_dtype="bfloat16", cache_dtype="bfloat16")
variables = {"params": raw["PaliGemma"]["llm"]}

prefix_bf16 = jnp.array(prefix).astype(jnp.bfloat16)[None]
positions = jnp.arange(Se, dtype=jnp.int32)[None]
cache_size = Se + 16
prefill_mask = jnp.zeros((1, 1, Se, cache_size), dtype=bool).at[:, :, :, :Se].set(True)

prefix_logits, kv_cache, _ = llm.apply(
    variables, embedded_prefix=prefix_bf16,
    positions=positions, mask=prefill_mask, decode=True,
    return_prelogits=True)
jax_prefill_xn = np.array(prefix_logits[0, -1], dtype=np.float32)

emb = np.array(raw["PaliGemma"]["llm"]["embedder"]["input_embedding"], dtype=np.float32)
jax_first_logit = jax_prefill_xn @ emb.T
jax_first_token = int(jax_first_logit.argmax())

# Decode step 0 — feed FVK's first token (so we test decode in isolation)
token_to_decode = fvk_first_token
token_arr = jnp.array([[token_to_decode]], dtype=jnp.int32)
decode_pos = jnp.array([[Se]], dtype=jnp.int32)
decode_mask = jnp.zeros((1, 1, 1, cache_size), dtype=bool).at[:, :, :, :Se+1].set(True)
pre_logits, _, _ = llm.apply(
    variables, tokens=token_arr,
    positions=decode_pos, mask=decode_mask,
    decode=True, kv_cache=kv_cache, return_prelogits=True)
jax_dec_xn = np.array(pre_logits[0, -1], dtype=np.float32)
jax_dec_logit = jax_dec_xn @ emb.T

def cos(a, b):
    a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

# Sink-aware: Pi0-FAST has a "massive activation" channel (typically dim 50 or 674)
# that can dominate energy at the post-final-RMSNorm output. Report cos with and
# without that channel for transparency.
def cos_no_sink(a, b):
    sink = int(np.argmax(np.abs(a)))
    am = a.copy(); am[sink] = 0
    bm = b.copy(); bm[sink] = 0
    if np.linalg.norm(am) < 1e-9 or np.linalg.norm(bm) < 1e-9:
        return float("nan")
    return cos(am, bm)

print(json.dumps({
    "first_token_match": jax_first_token == fvk_first_token,
    "fvk_first_token": fvk_first_token,
    "jax_first_token": jax_first_token,
    "S2_prefill_xn_cos": round(cos(fvk_prefill_xn, jax_prefill_xn), 6),
    "S2_prefill_xn_cos_no_sink": round(cos_no_sink(fvk_prefill_xn, jax_prefill_xn), 6),
    "S2b_first_logit_cos": round(cos(fvk_first_logit, jax_first_logit), 6),
    "S3_dec_xn_cos": round(cos(fvk_dec_xn, jax_dec_xn), 6),
    "S3_dec_xn_cos_no_sink": round(cos_no_sink(fvk_dec_xn, jax_dec_xn), 6),
    "S4_dec_logit_cos": round(cos(fvk_dec_logit, jax_dec_logit), 6),
}))
'''


BACKENDS = {
    "pi0fast_torch": {
        "name": "Pi0-FAST Torch",
        "module": "torch.pi0fast",
        "class": "Pi0FastTorchFrontend",
        "checkpoint": TORCH_CHECKPOINT,
        "num_views": 2,
    },
    "pi0fast_jax": {
        "name": "Pi0-FAST JAX",
        "module": "jax.pi0fast",
        "class": "Pi0FastJaxFrontend",
        "checkpoint": JAX_CHECKPOINT,
        "num_views": 2,
    },
}


def run_subprocess(name, script, timeout=900):
    r = subprocess.run(["python3", "-u", "-c", script],
                       capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        err_lines = r.stderr.strip().split("\n")[-10:]
        return {"error": "\n".join(err_lines)}
    for line in reversed(r.stdout.strip().split("\n")):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return {"error": "no JSON output\n" + r.stdout[-300:]}


def run_backend(key):
    info = BACKENDS[key]
    fvk_script = (FVK_TEMPLATE
                   .replace("ROOTDIR", FLASH_VLA_ROOT)
                   .replace("PIPELINE_MODULE", info["module"])
                   .replace("PIPELINE_CLASS", info["class"])
                   .replace("CHECKPOINT_PATH", info["checkpoint"])
                   .replace("NUM_VIEWS", str(info["num_views"]))
                   .replace("BACKEND", key))
    fvk_result = run_subprocess(f"{info['name']} (FVK side)", fvk_script)
    if "error" in fvk_result:
        return fvk_result

    jax_script = (JAX_REF_TEMPLATE
                  .replace("BACKEND", key)
                  .replace("JAX_CKPT_PATH", JAX_CHECKPOINT))
    jax_result = run_subprocess(f"{info['name']} (JAX reference)", jax_script)
    if "error" in jax_result:
        return jax_result

    return {**fvk_result, **jax_result}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=list(BACKENDS.keys()), default=None)
    args = parser.parse_args()
    targets = [args.backend] if args.backend else list(BACKENDS.keys())

    print("=" * 72)
    print("FlashRT — Pi0-FAST Precision Test (autoregressive)")
    print("=" * 72)
    print("Each backend runs prefill + decode step 0 with identical dummy input.")
    print("Per-segment cosine vs JAX bf16 reference (gemma_fast.Module.apply).")
    print("=" * 72)

    results = {}
    for key in targets:
        info = BACKENDS[key]
        print(f"\n── {info['name']} ──")
        r = run_backend(key)
        results[key] = r
        if "error" in r:
            print(f"  ERROR: {r['error']}")
            continue

        def status(c, t=0.98):
            return "PASS" if c >= t else ("WARN" if c >= 0.95 else "FAIL")

        print(f"  Se: {r['Se']}    first_token: FVK={r['fvk_first_token']} "
              f"JAX={r['jax_first_token']}    "
              f"[{'MATCH' if r['first_token_match'] else 'MISMATCH'}]")
        print(f"  S2  prefill xn cosine    : {r['S2_prefill_xn_cos']:.6f}  "
              f"[{status(r['S2_prefill_xn_cos'])}]")
        print(f"  S2b first logit cosine   : {r['S2b_first_logit_cos']:.6f}  "
              f"[{status(r['S2b_first_logit_cos'])}]")
        print(f"  S3  decode step 0 dec_xn : {r['S3_dec_xn_cos']:.6f}  "
              f"[{status(r['S3_dec_xn_cos'])}]")
        print(f"  S4  decode step 0 logit  : {r['S4_dec_logit_cos']:.6f}  "
              f"[{status(r['S4_dec_logit_cos'])}]")
        print(f"  Latency  prefill={r['prefill_p50_ms']:.1f}ms  "
              f"per_tok={r['per_tok_p50_ms']:.2f}ms  "
              f"50tok_total={r['total_p50_ms_50tok']:.1f}ms")

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    all_pass = True
    for key in targets:
        info = BACKENDS[key]
        r = results[key]
        if "error" in r:
            print(f"  {info['name']:18s}  [ERROR]")
            all_pass = False
            continue
        seg_cosines = [r['S2_prefill_xn_cos'], r['S2b_first_logit_cos'],
                       r['S3_dec_xn_cos'], r['S4_dec_logit_cos']]
        worst = min(seg_cosines)
        ok = all(c >= 0.98 for c in seg_cosines) and r['first_token_match']
        status = "PASS" if ok else "FAIL"
        print(f"  {info['name']:18s}  worst_segment_cos={worst:.4f}  "
              f"first_tok={'MATCH' if r['first_token_match'] else 'MISMATCH'}  "
              f"50tok={r['total_p50_ms_50tok']:.0f}ms  [{status}]")
        if not ok:
            all_pass = False
    print("=" * 72)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()

"""FlashVLA new-model template — weight specification.

Copy this file to `flash_rt/frontends/torch/_<mymodel>_<hw>_spec.py`
and fill in the `WEIGHT_SPEC` table. This is the **declarative** part
of model integration; getting this right is mostly mechanical
(translate state_dict keys to slots) and accounts for ~30% of the
total work.

# WHAT YOU TRANSLATE
=====================

Your source model:                       FlashVLA WEIGHT_SPEC entry:
---------------------                    -------------------------------
state_dict["encoder.layers.0.            ("encoder", 0, "qkv_proj_w"):
   self_attn.q_proj.weight"]                  (Stack(qw, kw, vw), Quant("fp8"))
state_dict["encoder.layers.0.            ("encoder", 0, "out_proj_w"):
   self_attn.o_proj.weight"]                  Quant("fp8")
state_dict["encoder.layers.0.            ("encoder", 0, "input_norm_w"):
   input_layernorm.weight"]                   Plain()    # too small for FP8
state_dict["encoder.layers.0.            ("encoder", 0, "ffn_gate_up_w"):
   mlp.gate_proj.weight",                     (CatRow(gate_w, up_w), Quant("fp8"))
   "encoder.layers.0.mlp.up_proj.weight"]
state_dict["encoder.layers.0.            ("encoder", 0, "ffn_down_w"):
   mlp.down_proj.weight"]                     Quant("fp8")
state_dict["lm_head.weight"]             ("output", None, "lm_head_w"):
                                              Quant("fp8")  # if your model has one

# RULES
========

1. **One row per logical weight slot**, not per state_dict tensor.
   Concatenations (gate+up, q+k+v) collapse multiple state_dict keys
   into one row using `Stack(...)` or `CatRow(...)`.

2. **Quant(...) decides FP8 vs plain BF16/FP16**. Default rule:
   - GEMM weight with M*N*K >= 4_000_000 -> Quant("fp8")
   - GEMM weight smaller than that -> Plain() (FP8 QDQ overhead exceeds the GEMM savings)
   - Norm / RoPE / embed / bias -> Plain() always (no FP8 path)
   - See docs/calibration.md §2.1 for the precise cutoff math.

3. **Op order within a Stack/CatRow MUST byte-match your reference**.
   If your reference does `torch.cat([gate, up], dim=0)` then your
   spec must use `CatRow(gate, up)` in that exact order. Reversing
   produces cos ≈ 0.5 with no other symptoms — see plugin_model_template.md
   "First-light cosine routing table".

4. **Don't quantize bias vectors.** They're tiny and FP8 quantizing
   biases provides zero benefit while adding a calibration headache.
"""

import numpy as np
from safetensors import safe_open

from flash_rt.executors.fp8_utils import quant_weight_fp8


# ──────────────────────────────────────────────────────────────────
# STEP 1: Declare your weight slots
# ──────────────────────────────────────────────────────────────────
#
# WEIGHT_SPEC is a dict mapping (site, layer_idx, slot_name) -> spec.
# The frontend iterates over this dict, applies the loader from each
# entry, and stores the result in self.weights["fp8"] or
# self.weights["plain"] for the pipeline to read.
#
# For a typical Pi0.5-shape VLA you will have ~6-8 slots per encoder
# layer, ~6-8 per decoder layer. Total table size: ~150-300 rows.

# TODO: replace these with your model's actual dimensions
NUM_ENCODER_LAYERS = 18    # e.g. Gemma-2B has 18 layers
NUM_DECODER_LAYERS = 18    # often = encoder layers in tied-backbone designs
HIDDEN_DIM = 2048          # model dim D
NUM_HEADS = 8              # number of attention heads NH
NUM_KV_HEADS = 1           # GQA KV heads (== NH if not GQA)
HEAD_DIM = 256             # HD; usually D / NH
FFN_HIDDEN = 16384         # FFN intermediate size H


def _enc_qkv_loader(state_dict, layer_idx):
    """STEP 2: write one loader per slot.

    The loader takes the raw state_dict (loaded from safetensors) and
    returns a numpy array in the layout the kernel expects.

    For QKV: most kernels expect [3, NH, HD, D] stacked on dim=0.
    Some expect [NH+2*NUM_KV_HEADS, HD, D] for GQA. Check your kernel's
    expected layout before you stack.

    TODO: replace key strings with your state_dict's actual key names.
    """
    qw = state_dict[f"encoder.layers.{layer_idx}.self_attn.q_proj.weight"]
    kw = state_dict[f"encoder.layers.{layer_idx}.self_attn.k_proj.weight"]
    vw = state_dict[f"encoder.layers.{layer_idx}.self_attn.v_proj.weight"]
    # Each is [out_dim, D]. Stack on dim=0 to get [3*out_dim, D].
    # IMPORTANT: GQA changes this — q_proj is [NH*HD, D] but k/v are
    # [NUM_KV_HEADS*HD, D]. You cannot naively stack. Use CatRow with
    # explicit shapes if your reference uses GQA.
    return np.concatenate([qw, kw, vw], axis=0)


def _enc_ffn_gate_up_loader(state_dict, layer_idx):
    """STEP 2 (cont.): FFN gate+up fused loader.

    Pi0.5 / many modern VLA models fuse `gate` and `up` projections
    into a single GEMM call to save kernel launch overhead. The kernel
    expects them stacked: [2*FFN_HIDDEN, D].

    TODO: rename keys + verify the order. If your reference does
    `up_proj(silu(gate_proj(x)))`, the GEMM output expected layout is
    typically `[gate; up]` (gate first). Reversing flips the SiLU
    gate and produces cos ≈ 0.7.
    """
    gate = state_dict[f"encoder.layers.{layer_idx}.mlp.gate_proj.weight"]
    up = state_dict[f"encoder.layers.{layer_idx}.mlp.up_proj.weight"]
    return np.concatenate([gate, up], axis=0)


# ──────────────────────────────────────────────────────────────────
# STEP 3: Build the WEIGHT_SPEC table
# ──────────────────────────────────────────────────────────────────
#
# Format: {(site, layer_idx, slot_name): (loader_callable, quant_decision)}
#
# `quant_decision` is one of:
#   ("fp8",)               — FP8 E4M3 with per-tensor symmetric scale
#   ("plain",)             — keep as fp16/bf16, no quant (default for small/noisy weights)
#   ("fp4_awq",)           — NVFP4 with AWQ pre-scale (Pi0.5 FP4 path; SM120+ only)

WEIGHT_SPEC = {}

for li in range(NUM_ENCODER_LAYERS):
    # GEMMs that get FP8 quantized
    WEIGHT_SPEC[("encoder", li, "qkv_w")] = (_enc_qkv_loader, ("fp8",))
    WEIGHT_SPEC[("encoder", li, "out_proj_w")] = (
        lambda sd, l=li: sd[f"encoder.layers.{l}.self_attn.o_proj.weight"],
        ("fp8",),
    )
    WEIGHT_SPEC[("encoder", li, "ffn_gate_up_w")] = (_enc_ffn_gate_up_loader, ("fp8",))
    WEIGHT_SPEC[("encoder", li, "ffn_down_w")] = (
        lambda sd, l=li: sd[f"encoder.layers.{l}.mlp.down_proj.weight"],
        ("fp8",),
    )
    # Norms and biases — keep as fp16
    WEIGHT_SPEC[("encoder", li, "input_norm_w")] = (
        lambda sd, l=li: sd[f"encoder.layers.{l}.input_layernorm.weight"],
        ("plain",),
    )
    WEIGHT_SPEC[("encoder", li, "post_attn_norm_w")] = (
        lambda sd, l=li: sd[f"encoder.layers.{l}.post_attention_layernorm.weight"],
        ("plain",),
    )

# TODO: repeat the loop for decoder layers, plus any model-specific
# slots (vision encoder, action input projection, action output projection,
# diffusion step embeddings, etc.). See _pi05_thor_spec.py for a
# full example covering all of these.

# Output projections (decoder action head)
WEIGHT_SPEC[("output", None, "action_out_w")] = (
    lambda sd: sd["action_head.weight"],
    ("fp8",),     # if action head is large enough; check your dim
)
WEIGHT_SPEC[("output", None, "action_out_b")] = (
    lambda sd: sd["action_head.bias"],
    ("plain",),
)


# ──────────────────────────────────────────────────────────────────
# STEP 4: Apply the spec
# ──────────────────────────────────────────────────────────────────

def load_weights(checkpoint_path: str) -> dict:
    """Load all state_dict tensors and apply WEIGHT_SPEC.

    Called once from the frontend's `__init__`. Returns a dict with
    two top-level keys:
        weights["fp8"][slot]   — fp8 tensor + saved fp32 scale
        weights["plain"][slot] — fp16/bf16 tensor

    The pipeline reads from this dict via slot keys.
    """
    # TODO: if your checkpoint is not safetensors, swap loader.
    state_dict = {}
    with safe_open(checkpoint_path, framework="pt") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key).numpy()

    out = {"fp8": {}, "plain": {}, "scales": {}}
    for slot_key, (loader, quant) in WEIGHT_SPEC.items():
        weight_np = loader(state_dict)
        if quant[0] == "fp8":
            fp8_w, scale = quant_weight_fp8(weight_np)
            out["fp8"][slot_key] = fp8_w
            out["scales"][slot_key] = scale
        elif quant[0] == "plain":
            out["plain"][slot_key] = weight_np
        else:
            raise NotImplementedError(f"quant scheme {quant[0]} not handled in template")
    return out


# ──────────────────────────────────────────────────────────────────
# DONE-CHECKLIST (verify before moving to attention.py)
# ──────────────────────────────────────────────────────────────────
# - [ ] Every state_dict key from your reference is referenced exactly
#       once in WEIGHT_SPEC (no orphans, no double-loads).
#       Quick check: `set(state_dict.keys()) - set(used_keys)` should be empty.
# - [ ] Every WEIGHT_SPEC slot is read by at least one kernel call in
#       pipeline.py (no dead weights).
# - [ ] FP8 quantized weights get assigned to GEMMs whose M*N*K > 4M
#       (else QDQ overhead > GEMM savings; see docs/calibration.md §2.1).
# - [ ] Concatenation order in your loaders byte-matches the reference's
#       `torch.cat(..., dim=0)` calls. This is the #1 source of silent
#       cos ≈ 0.5 bugs.

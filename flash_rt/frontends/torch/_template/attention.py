"""FlashVLA new-model template — attention specification.

Copy the body of `make_template_attention_spec()` into
`flash_rt/hardware/<hw>/attn_backend.py` (Thor or RTX). The
attention backend itself (the class that holds Q/K/V buffers and
dispatches to FA2 / cuBLAS / FMHA) does NOT need to be subclassed —
the existing `ThorFlashAttnBackend` / `TorchFlashAttnBackend` work
for any model whose attention sites you can describe with
`AttentionSpec`.

# WHAT YOU TRANSLATE
=====================

Your model has N attention "sites" — distinct attention operations
with different shapes / KV layouts. For a Pi0.5-shape model:

Your model code:                              FlashVLA AttentionSpec call:
-----------------                             ---------------------------
SiglipVisionTransformer (16 layers, 16        spec.add_site("vision",
   heads, full self-attn, fixed Sq=256/view)      num_layers=16, num_q_heads=16,
                                                  num_kv_heads=16, head_dim=72,
                                                  max_q_seq=256)
GemmaEncoder (18 layers, 8 GQA heads          spec.add_site("encoder",
   over 1 KV head, prompt+vision tokens,           num_layers=18, num_q_heads=8,
   variable Sq up to 512)                          num_kv_heads=1, head_dim=256,
                                                   max_q_seq=512)
ActionDecoder (18 layers, 8 heads,            spec.add_site("decoder",
   self-attn over action tokens, Sq=10 +           num_layers=18, num_q_heads=8,
   cross-attn into encoder KV)                     num_kv_heads=8, head_dim=256,
                                                   max_q_seq=10,
                                                   has_cross_kv=True)

Each site you declare here gets its own pre-allocated Q/K/V tensors
in the backend, sized for `max_q_seq`. The backend exposes
`attn.run("site_name", layer_idx, q_seq=actual_S, stream=stream)`
which the pipeline calls to dispatch attention for that layer.

# WHY SITES?
=============

Different sites have different shapes (vision is fixed Sq, encoder is
variable, decoder cross-attn pulls KV from a different buffer than
self-attn). Pre-allocating per-site avoids reshape gymnastics inside
the captured CUDA Graph. Also, profiling and debugging per-site
(via FVK_RTX_FA2_SITES env var) becomes trivial — see
plugin_model_template.md §"Site-level bisect via env var".
"""

from flash_rt.hardware.backend import AttentionSpec


def make_template_attention_spec() -> AttentionSpec:
    """STEP 1: declare every attention site your model uses.

    TODO: rename to `make_<mymodel>_attention_spec` and move into
    `flash_rt/hardware/<hw>/attn_backend.py` next to the existing
    `make_pi05_attention_spec` for reference.
    """
    spec = AttentionSpec()

    # STEP 2: Vision tower self-attention.
    # Most modern VLAs use a SigLIP / DINOv2 / CLIP vision tower with
    # full self-attention over a fixed grid of image patches.
    # Number of "views" (camera images) is folded into batch outside
    # the spec — one site declaration covers all views.
    spec.add_site(
        "vision",
        num_layers=16,        # vision tower depth
        num_q_heads=16,       # = num_kv_heads (vision is usually MHA, not GQA)
        num_kv_heads=16,
        head_dim=72,          # SigLIP-base = 72; SigLIP-large = 80
        max_q_seq=256,        # 14×14+CLS+1 patches typical for 224px input
    )

    # STEP 3: Encoder backbone (LLM that consumes prompt + vision tokens).
    # If your backbone uses GQA, num_kv_heads < num_q_heads. Pi0.5/Pi0
    # use Gemma's 8 query heads over 1 KV head (radical GQA = 8x
    # KV cache savings).
    spec.add_site(
        "encoder",
        num_layers=18,
        num_q_heads=8,
        num_kv_heads=1,       # GQA: change to num_q_heads if MHA
        head_dim=256,
        max_q_seq=512,        # max prompt + vision tokens you'll see
    )

    # STEP 4: Action decoder self-attention.
    # Diffusion-style decoders (Pi0.5, Pi0) self-attend across action
    # tokens (S=action_horizon=10 typical) per diffusion step.
    spec.add_site(
        "decoder",
        num_layers=18,
        num_q_heads=8,
        num_kv_heads=8,       # decoders usually drop GQA — small Sq, no win
        head_dim=256,
        max_q_seq=10,         # action horizon
        # has_cross_kv=True if your decoder cross-attends to encoder KV;
        # see Pi0.5 decoder cross-attention pattern.
    )

    # STEP 5 (optional): more sites.
    # GROOT-style multi-stage models add more sites (qwen3, dit_self,
    # dit_cross). Pi0-FAST adds an autoregressive "decode" site with
    # max_q_seq=1.
    # If your model has only encoder+decoder (no vision tower as a
    # separate attention path), drop the "vision" site.

    return spec


# ──────────────────────────────────────────────────────────────────
# DONE-CHECKLIST (verify before moving to pipeline.py)
# ──────────────────────────────────────────────────────────────────
# - [ ] Every distinct attention pattern in your model is one site
#       (don't merge sites with different shapes — graph capture
#       breaks if max_q_seq is wrong).
# - [ ] num_kv_heads correctly reflects GQA grouping (using num_q_heads
#       for both is the most common bug; produces cos ≈ 0.5–0.7 with
#       no other symptoms).
# - [ ] max_q_seq is a hard upper bound on what you'll feed in. The
#       buffer is allocated at this size; runtime q_seq must be ≤ it.
# - [ ] head_dim × num_q_heads = your model's hidden_dim D (this is
#       almost always the case; verify it isn't projected differently).

"""Shared LayerBlock builders for Thor torch frontends (stage 7.4).

Extracted from ``_pi05_thor_spec.py`` because Pi0.5 / Pi0 / Pi0-FAST /
GROOT (SigLIP2) share the SigLIP 27-layer block verbatim, and Pi0.5 /
Pi0 also share the Paligemma 18-layer encoder block. Keeping these in
one place removes copy-paste risk when the block schema changes.

Each factory returns a fresh ``LayerBlock`` — callers get a new instance
so spec modules don't alias mutable state across frontends.
"""

from __future__ import annotations

from flash_rt.executors.weight_loader import Item, LayerBlock
from flash_rt.executors.torch_weights import (
    Cat,
    FusedGateUp,
    FusedQKV,
    Quant,
    T,
    TensorList,
    ToFp16,
)


def paligemma_siglip_block(
    *,
    model_root: str = "paligemma_with_expert.paligemma.model",
    num_layers: int = 27,
) -> LayerBlock:
    """SigLIP encoder block used by Pi0.5 / Pi0 / Pi0-FAST torch frontends.

    27 layers × (LN1, LN2, fused QKV, O, FC1, FC2). All quantized GEMMs
    go through ``.T.contiguous()`` + per-tensor FP8 quant; scales land
    in ``target._sig_alpha`` in (q, o, up, down) order per layer.
    """
    vp = f"{model_root}.vision_tower.vision_model.encoder.layers.{{i}}"
    items = [
        Item("ln_attn_w", f"{vp}.layer_norm1.weight", [ToFp16()], TensorList("_sig_ln_attn_w")),
        Item("ln_attn_b", f"{vp}.layer_norm1.bias",   [ToFp16()], TensorList("_sig_ln_attn_b")),
        Item("ln_ffn_w",  f"{vp}.layer_norm2.weight", [ToFp16()], TensorList("_sig_ln_ffn_w")),
        Item("ln_ffn_b",  f"{vp}.layer_norm2.bias",   [ToFp16()], TensorList("_sig_ln_ffn_b")),

        Item("qkv_w",
             Cat([f"{vp}.self_attn.q_proj.weight",
                  f"{vp}.self_attn.k_proj.weight",
                  f"{vp}.self_attn.v_proj.weight"], dim=0),
             [T(), Quant()],
             TensorList("_sig_qkv_w"), scale_into="_sig_alpha"),
        Item("qkv_b",
             Cat([f"{vp}.self_attn.q_proj.bias",
                  f"{vp}.self_attn.k_proj.bias",
                  f"{vp}.self_attn.v_proj.bias"], dim=0),
             [],
             TensorList("_sig_qkv_b")),

        Item("o_w", f"{vp}.self_attn.out_proj.weight",
             [ToFp16(), T(), Quant()],
             TensorList("_sig_o_w"), scale_into="_sig_alpha"),
        Item("o_b", f"{vp}.self_attn.out_proj.bias",
             [ToFp16()], TensorList("_sig_o_b")),

        Item("up_w", f"{vp}.mlp.fc1.weight",
             [ToFp16(), T(), Quant()],
             TensorList("_sig_up_w"), scale_into="_sig_alpha"),
        Item("up_b", f"{vp}.mlp.fc1.bias",
             [ToFp16()], TensorList("_sig_up_b")),

        Item("down_w", f"{vp}.mlp.fc2.weight",
             [ToFp16(), T(), Quant()],
             TensorList("_sig_down_w"), scale_into="_sig_alpha"),
        Item("down_b", f"{vp}.mlp.fc2.bias",
             [ToFp16()], TensorList("_sig_down_b")),
    ]
    return LayerBlock(prefix_fmt="", num_layers=num_layers, items=items, name="siglip")


def paligemma_encoder_block(
    *,
    model_root: str = "paligemma_with_expert.paligemma.model",
    num_layers: int = 18,
) -> LayerBlock:
    """Paligemma encoder (18 layers, GQA, AdaRMSNorm fused into QKV/gate-up).

    Matches the encoder loader used by Pi0.5 and Pi0 torch frontends:
      qkv  : FusedQKV(interleave 8/1, norm_fuse=input_layernorm)  → [Quant()]
      o    : ToFp16 → Quant
      gu   : FusedGateUp(norm_fuse=post_attention_layernorm)      → [Quant()]
      d    : ToFp16 → Quant
    Scales append to ``target._enc_w_scales`` in (q, o, gu, d) order.
    """
    ep = f"{model_root}.language_model.layers.{{i}}"
    items = [
        Item("qkv_w",
             FusedQKV(q=f"{ep}.self_attn.q_proj.weight",
                      k=f"{ep}.self_attn.k_proj.weight",
                      v=f"{ep}.self_attn.v_proj.weight",
                      norm_fuse=f"{ep}.input_layernorm.weight",
                      interleave_q_heads=8,
                      interleave_k_heads=1),
             [Quant()],
             TensorList("_enc_qkv_w"), scale_into="_enc_w_scales"),
        Item("o_w", f"{ep}.self_attn.o_proj.weight",
             [ToFp16(), Quant()],
             TensorList("_enc_o_w"), scale_into="_enc_w_scales"),
        Item("gu_w",
             FusedGateUp(gate=f"{ep}.mlp.gate_proj.weight",
                         up=f"{ep}.mlp.up_proj.weight",
                         norm_fuse=f"{ep}.post_attention_layernorm.weight"),
             [Quant()],
             TensorList("_enc_gu_w"), scale_into="_enc_w_scales"),
        Item("d_w", f"{ep}.mlp.down_proj.weight",
             [ToFp16(), Quant()],
             TensorList("_enc_d_w"), scale_into="_enc_w_scales"),
    ]
    return LayerBlock(prefix_fmt="", num_layers=num_layers, items=items, name="encoder")


__all__ = ["paligemma_siglip_block", "paligemma_encoder_block"]

"""Shared LayerBlock builders for Thor JAX frontends (stage 7.7).

Pi0.5 / Pi0 JAX frontends share the SigLIP + Paligemma encoder +
Gemma-expert decoder blocks verbatim (openpi's ``engine_w`` export uses
the same key schema for both). Decoder modulation and other model-
specific loops live in each model's own spec file.
"""

from __future__ import annotations

import numpy as np

from flash_vla.executors.weight_loader import Item, LayerBlock
from flash_vla.executors.jax_weights import (
    Astype,
    CudaBufferFlat,
    JaxQuant,
    Transpose,
)


_FP16 = np.float16


def vision_siglip_block(*, num_layers: int = 27) -> LayerBlock:
    """SigLIP encoder (27 layers) — matches the openpi ``vision.layer.{i}.*``
    schema. Emits 12 per-slot CudaBuffers that the frontend composes into
    ``self.sig_wt_fp8``; cache keys preserved as ``sig_wt_fp8.{0..11}``."""
    vp = "vision.layer.{i}"
    items = [
        Item("ln1w", f"{vp}.attn_norm.weight", [Astype(_FP16)],
             CudaBufferFlat("_sig_ln1w_cb", cache="sig_wt_fp8.0")),
        Item("ln1b", f"{vp}.attn_norm.bias", [Astype(_FP16)],
             CudaBufferFlat("_sig_ln1b_cb", cache="sig_wt_fp8.1")),

        Item("qw", f"{vp}.qkv.weight", [Transpose(), JaxQuant()],
             CudaBufferFlat("_sig_qw_cb", cache="sig_wt_fp8.2"),
             scale_into="_sig_scales"),
        Item("qb", f"{vp}.qkv.bias", [Astype(_FP16)],
             CudaBufferFlat("_sig_qb_cb", cache="sig_wt_fp8.3")),

        Item("ow", f"{vp}.o.weight", [Transpose(), JaxQuant()],
             CudaBufferFlat("_sig_ow_cb", cache="sig_wt_fp8.4"),
             scale_into="_sig_scales"),
        Item("ob", f"{vp}.o.bias", [Astype(_FP16)],
             CudaBufferFlat("_sig_ob_cb", cache="sig_wt_fp8.5")),

        Item("ln2w", f"{vp}.ffn_norm.weight", [Astype(_FP16)],
             CudaBufferFlat("_sig_ln2w_cb", cache="sig_wt_fp8.6")),
        Item("ln2b", f"{vp}.ffn_norm.bias", [Astype(_FP16)],
             CudaBufferFlat("_sig_ln2b_cb", cache="sig_wt_fp8.7")),

        Item("uw", f"{vp}.ffn_up.weight", [Transpose(), JaxQuant()],
             CudaBufferFlat("_sig_uw_cb", cache="sig_wt_fp8.8"),
             scale_into="_sig_scales"),
        Item("ub", f"{vp}.ffn_up.bias", [Astype(_FP16)],
             CudaBufferFlat("_sig_ub_cb", cache="sig_wt_fp8.9")),

        Item("dw", f"{vp}.ffn_down.weight", [Transpose(), JaxQuant()],
             CudaBufferFlat("_sig_dw_cb", cache="sig_wt_fp8.10"),
             scale_into="_sig_scales"),
        Item("db", f"{vp}.ffn_down.bias", [Astype(_FP16)],
             CudaBufferFlat("_sig_db_cb", cache="sig_wt_fp8.11")),
    ]
    return LayerBlock(prefix_fmt="", num_layers=num_layers, items=items, name="siglip")


def paligemma_encoder_block(*, num_layers: int = 18) -> LayerBlock:
    """Paligemma encoder (18 layers) — keys ``encoder.layer.{i}.{qkv,o,gate_up,down}.weight``.

    engine_w stores these pre-transposed and pre-fused; only FP8 quant
    is applied. Scales land in ``target._enc_ws`` in (q, o, gu, d) order.
    """
    ep = "encoder.layer.{i}"
    items = [
        Item("qkv", f"{ep}.qkv.weight", [JaxQuant()],
             CudaBufferFlat("_enc_qkv_cb", cache="ew.0"), scale_into="_enc_ws"),
        Item("o",   f"{ep}.o.weight",   [JaxQuant()],
             CudaBufferFlat("_enc_o_cb",   cache="ew.1"), scale_into="_enc_ws"),
        Item("gu",  f"{ep}.gate_up.weight", [JaxQuant()],
             CudaBufferFlat("_enc_gu_cb",  cache="ew.2"), scale_into="_enc_ws"),
        Item("d",   f"{ep}.down.weight",    [JaxQuant()],
             CudaBufferFlat("_enc_d_cb",   cache="ew.4"), scale_into="_enc_ws"),
    ]
    return LayerBlock(prefix_fmt="", num_layers=num_layers, items=items, name="encoder")


def gemma_decoder_block(*, num_layers: int = 18) -> LayerBlock:
    """Gemma-expert decoder (18 layers). Output attrs are
    ``self.dec_{qkv,o,gu,d}_flat`` (legacy names preserved). Scales land
    in ``target._ae_ws``."""
    dp = "decoder.layer.{i}"
    items = [
        Item("qkv", f"{dp}.qkv.weight", [JaxQuant()],
             CudaBufferFlat("dec_qkv_flat", cache="dec_qkv_flat"),
             scale_into="_ae_ws"),
        Item("o",   f"{dp}.o.weight",   [JaxQuant()],
             CudaBufferFlat("dec_o_flat",   cache="dec_o_flat"),
             scale_into="_ae_ws"),
        Item("gu",  f"{dp}.gate_up.weight", [JaxQuant()],
             CudaBufferFlat("dec_gu_flat",  cache="dec_gu_flat"),
             scale_into="_ae_ws"),
        Item("d",   f"{dp}.down.weight",    [JaxQuant()],
             CudaBufferFlat("dec_d_flat",   cache="dec_d_flat"),
             scale_into="_ae_ws"),
    ]
    return LayerBlock(prefix_fmt="", num_layers=num_layers, items=items, name="decoder")


__all__ = [
    "vision_siglip_block",
    "paligemma_encoder_block",
    "gemma_decoder_block",
]

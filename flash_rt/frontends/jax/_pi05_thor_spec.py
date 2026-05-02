"""Declarative weight spec for Pi05JaxFrontendThor.

SigLIP + Paligemma encoder + Gemma-expert decoder blocks come from
``_thor_spec_common``. Pi0.5 adds a per-layer AdaRMS modulation Dense
block that Pi0 does not have.
"""

from __future__ import annotations

import numpy as np

from flash_vla.executors.weight_loader import Item, LayerBlock, ModelWeightSpec
from flash_vla.executors.jax_weights import Astype, NumpyList
from flash_vla.frontends.jax._thor_spec_common import (
    gemma_decoder_block,
    paligemma_encoder_block,
    vision_siglip_block,
)


_FP16 = np.float16


def _decoder_mods_block() -> LayerBlock:
    """Per-layer AdaRMS modulation Dense (numpy, CPU-resident)."""
    dp = "decoder.layer.{i}"
    items = [
        Item("attn_mod_w", f"{dp}.attn_mod.weight", [Astype(_FP16)],
             NumpyList("_attn_mod_w")),
        Item("attn_mod_b", f"{dp}.attn_mod.bias",   [Astype(_FP16)],
             NumpyList("_attn_mod_b")),
        Item("ffn_mod_w",  f"{dp}.ffn_mod.weight",  [Astype(_FP16)],
             NumpyList("_ffn_mod_w")),
        Item("ffn_mod_b",  f"{dp}.ffn_mod.bias",    [Astype(_FP16)],
             NumpyList("_ffn_mod_b")),
    ]
    return LayerBlock(prefix_fmt="", num_layers=18, items=items, name="decoder_mods")


def build_spec() -> ModelWeightSpec:
    return ModelWeightSpec(
        framework="jax",
        blocks=[
            vision_siglip_block(),
            paligemma_encoder_block(),
            gemma_decoder_block(),
            _decoder_mods_block(),
        ],
    )


__all__ = ["build_spec"]

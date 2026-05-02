"""Declarative weight spec for Pi0JaxFrontendThor (stage 7.7).

Pi0 shares the SigLIP + Paligemma encoder + Gemma-expert decoder blocks
with Pi0.5 (imported from ``_thor_spec_common``). Unlike Pi0.5 there is
no decoder modulation Dense block — Pi0 uses plain RMSNorm with a
scalar ``final_norm_w`` instead of AdaRMS Dense layers. Pi0-specific
singletons (action_time_mlp, state_proj, final_norm_w) stay inline in
the frontend for now.
"""

from __future__ import annotations

from flash_rt.executors.weight_loader import ModelWeightSpec
from flash_rt.frontends.jax._thor_spec_common import (
    gemma_decoder_block,
    paligemma_encoder_block,
    vision_siglip_block,
)


def build_spec() -> ModelWeightSpec:
    return ModelWeightSpec(
        framework="jax",
        blocks=[
            vision_siglip_block(),
            paligemma_encoder_block(),
            gemma_decoder_block(),
        ],
    )


__all__ = ["build_spec"]

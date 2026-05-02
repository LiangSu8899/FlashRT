"""FlashVLA -- GROOT N1.6 model pipelines and embodiment mapping.

Per the unified pipeline_<hw>.py contract:
    pipeline_thor.py  - Thor SM110 GrootSigLIP2 / GrootQwen3 / GrootDiT
    pipeline_rtx.py   - RTX SM120 same classes (re-exported below)

The RTX pipeline classes and embodiment data are re-exported here so
that frontends and tests can ``from flash_vla.models.groot import ...``
without knowing the internal file layout.
"""

from flash_vla.models.groot.pipeline_rtx import (
    GrootSigLIP2,
    GrootQwen3,
    GrootDiT,
)
from flash_vla.models.groot.embodiments import (
    EMBODIMENT_TAG_TO_INDEX,
    TRAINED_EMBODIMENT_IDS,
    PUBLIC_TRAINED_TAGS,
    is_embodiment_trained,
)

__all__ = [
    "GrootSigLIP2",
    "GrootQwen3",
    "GrootDiT",
    "EMBODIMENT_TAG_TO_INDEX",
    "TRAINED_EMBODIMENT_IDS",
    "PUBLIC_TRAINED_TAGS",
    "is_embodiment_trained",
]

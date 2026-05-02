"""Back-compat shim — moved to :mod:`training.jax.rl.train_recap`.

The driver is dataset-agnostic (it's the upstream
``train_jax_lora_recap.py`` from openpi-compiler invoked via the
FlashRT FP8 wrapper); the LIBERO-only filename was a misnomer.
This shim re-exports everything for any caller still importing
the old path. New code should import from
:mod:`training.jax.rl.train_recap` directly.
"""

from training.jax.rl.train_recap import *  # noqa: F401,F403
from training.jax.rl.train_recap import (
    DEFAULT_ACP_DROPOUT,
    DEFAULT_BATCH_SIZE,
    DEFAULT_END_LR_FACTOR,
    DEFAULT_GRAD_CLIP_NORM,
    DEFAULT_LORA_RANK,
    DEFAULT_LR,
    DEFAULT_WARMUP_FRACTION,
    DEFAULT_WARMUP_MAX,
    DEFAULT_WEIGHT_DECAY,
    fp8_wrapper_script_path,
    upstream_baseline_script_path,
)

__all__ = [
    "DEFAULT_ACP_DROPOUT",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_END_LR_FACTOR",
    "DEFAULT_GRAD_CLIP_NORM",
    "DEFAULT_LORA_RANK",
    "DEFAULT_LR",
    "DEFAULT_WARMUP_FRACTION",
    "DEFAULT_WARMUP_MAX",
    "DEFAULT_WEIGHT_DECAY",
    "fp8_wrapper_script_path",
    "upstream_baseline_script_path",
]

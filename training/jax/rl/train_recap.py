"""Long-form RECAP / ACP policy training driver — JAX path.

The driver itself lives in the upstream openpi-compiler script
``RL/scripts/train_jax_lora_recap.py``. It is **dataset-agnostic
infrastructure** despite its filename — RECAP / ACP works on any
LeRobot v3 parquet that ships an ``acp_indicator`` column (or
admits one being derived). LIBERO is the first concrete dataset
shipped; nothing in the driver is structurally LIBERO-specific.

The driver implements the recipe Phase 3d calls for:

* AdamW with ``weight_decay=1e-10``;
* warmup-cosine LR schedule
  (``init=0, peak=lr, warmup=min(100, steps//30),
  decay=steps, end=peak*0.1``) — see
  ``train_jax_lora_recap.py:317-321``;
* global L2 grad clip 1.0 — ``:322``;
* ``pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora",
  action_expert_variant="gemma_300m_lora")`` — ``:248-251``;
* ACP injection (30 % dropout) inline in the dataset's
  ``_get_raw_sample`` — ``:184-188``;
* freeze policy: ``cfg.get_freeze_filter()`` + vision-tower regex —
  ``:301-306``.

Re-implementing this entire driver in ``training/jax/rl/`` would
just duplicate openpi-compiler. Instead, FlashRT contributes:

* The FP8 LoRA patch (`training.jax.fp8.lora_patch`) — installed
  via ``training.jax.scripts.run_baseline_with_fp8_patch`` so the
  upstream driver routes its base GEMMs through cuBLASLt FP8.
* The cross-language primitives (``flash_rt.core.rl.acp_tags``,
  ``flash_rt.core.rl.advantage``,
  ``flash_rt.core.rl.reward``) — imported by both PyTorch and the
  upstream JAX driver.
* Forward-parity / annotation helpers for users who want to
  annotate a brand-new dataset (`training.jax.rl.train_value`,
  `training.jax.rl.value_infer`).

To run a long-form RECAP fine-tune with FP8 enabled::

    python -m training.jax.scripts.run_baseline_with_fp8_patch \
        --baseline-script $FLASHVLA_JAX_BASELINE_SCRIPT \
        --checkpoint_path $FLASHVLA_PI05_CKPT_JAX \
        --dataset_root    $FLASHVLA_RECAP_DATASET \
        --output_dir      <your-run-dir> \
        --steps 30000 --batch_size 4 --lr 2.5e-5 \
        --lora_rank 16 --acp_dropout 0.30 --log_freq 50

To run without FP8 (BF16 baseline)::

    python $FLASHVLA_JAX_BASELINE_SCRIPT --steps 30000 ...

The Phase 5 acceptance bench runs both side-by-side at 30k steps
and reports throughput + peak-mem + final-loss ratio.
"""

from __future__ import annotations

import os
from pathlib import Path

# Re-exports for namespace symmetry with training.rl.* (PyTorch).
# These are utility paths; the driver is the upstream script. Users
# who want a Python-level handle import from here so refactors of
# the upstream script's location don't ripple through caller code.

DEFAULT_LR = 2.5e-5
DEFAULT_WARMUP_FRACTION = 1 / 30        # min(100, steps // 30)
DEFAULT_WARMUP_MAX = 100
DEFAULT_END_LR_FACTOR = 0.10
DEFAULT_GRAD_CLIP_NORM = 1.0
DEFAULT_WEIGHT_DECAY = 1e-10
DEFAULT_BATCH_SIZE = 4
DEFAULT_ACP_DROPOUT = 0.30
DEFAULT_LORA_RANK = 16


def upstream_baseline_script_path() -> Path:
    """Resolve the upstream ``train_jax_lora_recap.py`` from env."""
    p = os.environ.get("FLASHVLA_JAX_BASELINE_SCRIPT")
    if not p:
        raise FileNotFoundError(
            "Set FLASHVLA_JAX_BASELINE_SCRIPT to the openpi "
            "RL/scripts/train_jax_lora_recap.py path."
        )
    pp = Path(p)
    if not pp.is_file():
        raise FileNotFoundError(f"baseline script not found: {pp}")
    return pp


def fp8_wrapper_script_path() -> Path:
    """Path to the FP8-enabling subprocess wrapper (Phase 2)."""
    here = Path(__file__).resolve()
    repo_root = here.parents[3]
    p = repo_root / "training" / "jax" / "scripts" / "run_baseline_with_fp8_patch.py"
    if not p.is_file():
        raise FileNotFoundError(f"wrapper not found: {p}")
    return p

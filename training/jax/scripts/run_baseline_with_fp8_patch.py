"""Run the openpi JAX baseline with FlashRT's FP8 LoRA patch enabled.

This is a thin wrapper around an existing
``RL/scripts/train_jax_lora_recap.py``-style baseline. It:

1. Imports + enables the LoRA patch BEFORE openpi.models.lora is
   transitively imported by the baseline script (the patch must be
   installed before any LoRA modules are constructed for it to take
   effect).
2. Re-execs the baseline script in this process so its CLI args,
   logging, and side effects behave exactly like the un-patched
   invocation — making subprocess-based loss-curve comparisons
   apples-to-apples.

Usage::

    python -m training.jax.scripts.run_baseline_with_fp8_patch \
        --baseline-script /path/to/train_jax_lora_recap.py \
        --checkpoint_path ... --dataset_root ... --steps 50 ...

The ``--baseline-script`` arg (or ``FLASHVLA_JAX_BASELINE_SCRIPT``
env var) names the upstream script to forward to. All remaining
args are passed through to that script's ``main()``.

Routing is gated on ``FLASHVLA_JAX_FP8`` (default ``"1"`` once the
patch is enabled). Set ``FLASHVLA_JAX_FP8=0`` in the env to keep
the patch installed but route every call back through the original
``jnp.einsum`` / ``jnp.dot`` — useful for "instrument-only" runs
that count call sites without actually using FP8.
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path


def _resolve_baseline(args_path: str | None) -> Path:
    if args_path:
        p = Path(args_path)
    else:
        env_path = os.environ.get("FLASHVLA_JAX_BASELINE_SCRIPT")
        if not env_path:
            raise SystemExit(
                "must pass --baseline-script or set FLASHVLA_JAX_BASELINE_SCRIPT"
            )
        p = Path(env_path)
    if not p.is_file():
        raise SystemExit(f"baseline script not found: {p}")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_baseline_with_fp8_patch",
        description=__doc__.split("\n", 1)[0],
        # Don't consume the baseline script's own flags.
        allow_abbrev=False,
    )
    parser.add_argument(
        "--baseline-script",
        default=None,
        help="Path to the openpi JAX baseline script (default: env "
             "FLASHVLA_JAX_BASELINE_SCRIPT).",
    )
    own_args, forwarded = parser.parse_known_args(argv)

    script_path = _resolve_baseline(own_args.baseline_script)

    # Default the patch to ON if the user has not explicitly set the env
    # var. The wrapper exists to install + activate the patch; running
    # it with FLASHVLA_JAX_FP8=0 makes it a no-op-but-installed,
    # primarily useful for routing-counter A/B comparisons.
    os.environ.setdefault("FLASHVLA_JAX_FP8", "1")

    # The patch installs onto openpi.models.lora — it must run BEFORE
    # the baseline's own imports pull lora in. ``enable()`` itself
    # imports openpi.models.lora, which is the earliest legitimate
    # touch-point.
    from training.jax.fp8 import enable_fp8_patch
    enable_fp8_patch()

    # Forward to the baseline script as if it were invoked directly.
    # runpy emulates ``python <script>`` semantics — its argv handling
    # is what the baseline's argparse expects.
    sys.argv = [str(script_path), *forwarded]
    runpy.run_path(str(script_path), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

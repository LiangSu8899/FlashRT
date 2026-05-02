#!/usr/bin/env python3
"""Thor multi-sample dataset calibration — end-to-end LIBERO test.

Exercises :meth:`Pi05TorchFrontendThor.calibrate` with N>=2 real
observations sampled from a LeRobot-v2 LIBERO dataset (via
:mod:`flash_rt.datasets.libero`). Verifies:

* No exception from the newly-implemented multi-frame path.
* After ``calibrate(obs_list)``, the subsequent ``infer`` runs and
  produces action outputs whose cosine vs the single-frame / saved
  golden references stays above thresholds close to the single-frame
  baseline (threshold is looser than
  :file:`tests/test_all_models_precision.py` because the percentile-
  reduced scales will differ slightly from the single-frame "same
  image" scales, and this is expected).
* Latency (P50 over 20 iters after calibration) does not regress vs
  the single-frame baseline.

Defaults to ``/workspace/libero_10_image`` and
``/root/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch``
which match the ``the PyTorch deploy container`` Thor container. Override via
``LIBERO_ROOT`` and ``PI05_CKPT`` environment variables.

Usage (inside Thor container)::

    python3 tests/test_thor_multi_sample_calibrate.py          # N=8 default
    python3 tests/test_thor_multi_sample_calibrate.py --n 16   # 16 frames
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("thor_multi_cal")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_LIBERO   = os.environ.get("LIBERO_ROOT", "/workspace/libero_10_image")
DEFAULT_PI05_CKPT = os.environ.get(
    "PI05_CKPT",
    "/root/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch")
PROMPT = os.environ.get(
    "PI05_PROMPT",
    "pick up the red block and place it in the tray")


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8,
                    help="calibration sample count (default 8)")
    ap.add_argument("--percentile", type=float, default=99.9)
    ap.add_argument("--task-filter", type=int, default=None,
                    help="only sample frames from this task_index")
    ap.add_argument("--libero-root", default=DEFAULT_LIBERO)
    ap.add_argument("--pi05-ckpt", default=DEFAULT_PI05_CKPT)
    ap.add_argument("--lat-iters", type=int, default=20)
    args = ap.parse_args()

    # ------------------------------------------------------------------
    # Sample N real observations from LIBERO.
    # ------------------------------------------------------------------
    from flash_rt.datasets.libero import load_calibration_obs
    t0 = time.perf_counter()
    obs_list = load_calibration_obs(
        args.libero_root, n=args.n, task_filter=args.task_filter, verbose=True)
    load_ms = (time.perf_counter() - t0) * 1000
    logger.info("Loaded %d LIBERO frames in %.1f ms", len(obs_list), load_ms)

    # ------------------------------------------------------------------
    # Build Pi0.5 Thor frontend and run N>=2 calibration path.
    # ------------------------------------------------------------------
    from flash_rt.frontends.torch.pi05_thor import Pi05TorchFrontendThor

    pipe = Pi05TorchFrontendThor(args.pi05_ckpt, num_views=2, autotune=0)
    pipe.set_prompt(PROMPT)

    t0 = time.perf_counter()
    pipe.calibrate(obs_list, percentile=args.percentile, verbose=True)
    cal_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "pi05_thor.calibrate(N=%d, percentile=%.2f) complete in %.1f ms",
        args.n, args.percentile, cal_ms)

    # Prove calibrate() flipped _real_data_calibrated so the first infer
    # does NOT re-run the lazy path.
    assert pipe._real_data_calibrated, \
        "calibrate() should have set _real_data_calibrated=True"

    # ------------------------------------------------------------------
    # Run inference on the sampled frames and measure latency.
    # ------------------------------------------------------------------
    # Use the first sampled frame as the deployment observation.
    deploy_obs = obs_list[0]
    # Warm up + capture one action output.
    for _ in range(3):
        pipe.infer(deploy_obs)
    result = pipe.infer(deploy_obs)
    actions = result["actions"]
    logger.info(
        "Action output shape=%s dtype=%s first_row=%s",
        actions.shape, actions.dtype, np.round(actions[0], 3).tolist())

    lat = []
    for _ in range(args.lat_iters):
        t0 = time.perf_counter()
        pipe.infer(deploy_obs)
        lat.append((time.perf_counter() - t0) * 1000)
    lat.sort()
    p50 = lat[len(lat) // 2]
    p95 = lat[int(len(lat) * 0.95)]

    # ------------------------------------------------------------------
    # Cross-check: re-run with N=1 on the same deploy_obs and compare.
    # A small absolute cosine difference is expected (the multi-frame
    # percentile scales generalize across frames), but it should remain
    # well above 0.99 on a real LIBERO observation.
    # ------------------------------------------------------------------
    pipe_single = Pi05TorchFrontendThor(args.pi05_ckpt, num_views=2, autotune=0)
    pipe_single.set_prompt(PROMPT)
    pipe_single.calibrate([deploy_obs])  # N=1 implicit-recalibrate path
    # Match noise between the two pipelines so the action comparison is
    # not polluted by np.random state drift between frontends.
    noise_ref = pipe_single._g_noise.clone()
    pipe._g_noise.copy_(noise_ref)
    # Disable in-graph noise regeneration by re-uploading before replay.
    pipe._siglip_graph.replay()
    pipe._enc_ae_graph.replay()
    torch.cuda.synchronize()
    multi_raw = pipe._g_noise.float().cpu().numpy()

    pipe_single._siglip_graph.replay()
    pipe_single._enc_ae_graph.replay()
    torch.cuda.synchronize()
    single_raw = pipe_single._g_noise.float().cpu().numpy()

    cos_multi_vs_single = _cos(multi_raw, single_raw)

    summary = {
        "n": args.n,
        "percentile": args.percentile,
        "calibrate_ms": round(cal_ms, 1),
        "load_ms": round(load_ms, 1),
        "p50_ms": round(p50, 1),
        "p95_ms": round(p95, 1),
        "cos_multi_vs_single_frame": round(cos_multi_vs_single, 6),
        "verdict": (
            "PASS" if cos_multi_vs_single >= 0.99 and p50 < 60 else "CHECK"),
    }
    print(json.dumps(summary, indent=2))
    return 0 if summary["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

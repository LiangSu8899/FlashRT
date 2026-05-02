#!/usr/bin/env python3
"""Thor multi-sample calibration benchmark vs PyTorch FP32 reference.

Reproduces the RTX 5090 measurement protocol (docs/calibration.md §10)
on the Jetson AGX Thor.

Reference: ``/tmp/pytorch_reference.npz`` — the canonical Pi0.5 PyTorch
FP32 reference shipped inside the Thor deploy container. Contains the
3-view inputs (base + left_wrist + right_wrist), state, 200-token
prompt, fixed noise, and ``pytorch_raw_output`` (1, 10, 32) — the
authoritative ground truth.

Calibration observations: stratified LIBERO-10 samples via
``flash_rt.datasets.libero``. LIBERO has only two camera streams;
for a 3-view frontend we duplicate the wrist image into the third
channel (activation statistics remain realistic; the goal is
distribution coverage, not exact geometry).

Measurement per N ∈ {1, 8, 16, 64, 256} (override with ``--ns``):

    1. Build a fresh Pi05TorchFrontendThor with num_views=3.
    2. Time ``pipe.calibrate(obs_list, percentile=99.9)``.
    3. Inject the ref noise and feed the ref observation through the
       captured graph (monkey-patched ``.normal_`` — same trick as
       ``tests/test_all_models_precision.py``).
    4. Compute cos(_g_noise_after_replay, pytorch_raw_output) and
       max |diff| — plus replay P50 latency.

N=1 uses the ref frame itself (matches the "self" row in the RTX
table). N >= 2 uses stratified LIBERO samples so the scales reflect
dataset statistics, not a single frame.

Usage (inside the PyTorch deploy container container)::

    python3 tests/bench_thor_calibration_vs_ref.py
    python3 tests/bench_thor_calibration_vs_ref.py --ns 1,8,16,64
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import sys
import time
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("bench_thor_cal")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_LIBERO = os.environ.get("LIBERO_ROOT", "/workspace/libero_10_image")
DEFAULT_CKPT   = os.environ.get(
    "PI05_CKPT",
    "<your_pi05_torch_ckpt>")
DEFAULT_REF    = os.environ.get("PI05_FP32_REF", "/tmp/pytorch_reference.npz")


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _load_ref(path: str):
    d = np.load(path, allow_pickle=True)
    tok_mask = d["arg8_tokenized_prompt_mask"][0]
    prompt_len = int(tok_mask.sum())
    return {
        "img":          d["arg0_base_rgb"][0],          # fp16 [-1,1]
        "wrist":        d["arg1_left_wrist_rgb"][0],
        "wrist_right":  d["arg2_right_wrist_rgb"][0],
        "state":        d["arg6_state"][0].astype(np.float32),
        "prompt_toks":  d["arg7_tokenized_prompt"][0][:prompt_len].astype(np.int64),
        "noise":        d["arg9_noise"][0].astype(np.float16),     # (10, 32)
        "pytorch_raw":  d["pytorch_raw_output"][0].astype(np.float32),  # (10, 32)
    }


def _build_pipe(ckpt: str, prompt_toks: np.ndarray):
    from flash_rt.frontends.torch.pi05_thor import Pi05TorchFrontendThor
    pipe = Pi05TorchFrontendThor(ckpt, num_views=3, autotune=0)
    pipe.set_prompt(prompt_toks.tolist())
    return pipe


def _ref_obs(ref) -> dict:
    return {
        "image":             ref["img"],
        "wrist_image":       ref["wrist"],
        "wrist_image_right": ref["wrist_right"],
    }


def _promote_obs_3view(obs: dict) -> dict:
    """Turn a 2-view LIBERO obs into a 3-view obs by duplicating wrist."""
    if "wrist_image_right" in obs:
        return obs
    return {
        "image":             obs["image"],
        "wrist_image":       obs["wrist_image"],
        "wrist_image_right": obs["wrist_image"],
    }


def _load_obs_lib(libero_root: str, n: int) -> list:
    from flash_rt.datasets.libero import load_calibration_obs
    base = load_calibration_obs(libero_root, n=n, verbose=False)
    return [_promote_obs_3view(o) for o in base]


def _replay_with_ref_noise(pipe, noise_fp16: np.ndarray, obs: dict) -> np.ndarray:
    """Run a full ``pipe.infer(obs)`` with ``_g_noise.normal_`` patched to
    copy the golden reference noise. Matches the pattern used by
    ``tests/test_all_models_precision.py``. Returns the (Sa, 32) fp32
    raw decoder output (``_g_noise`` post-replay)."""
    matched = torch.from_numpy(noise_fp16).to(
        dtype=torch.float16, device="cuda")
    _orig = torch.Tensor.normal_

    def _patched(self, *a, **kw):
        if self.data_ptr() == pipe._g_noise.data_ptr():
            self.copy_(matched)
            return self
        return _orig(self, *a, **kw)

    torch.Tensor.normal_ = _patched
    try:
        pipe.infer(obs)
    finally:
        torch.Tensor.normal_ = _orig
    return pipe._g_noise.float().cpu().numpy()


def _clear_cache() -> None:
    cdir = pathlib.Path.home() / ".flash_rt" / "calibration"
    if cdir.exists():
        for f in cdir.glob("*.json"):
            f.unlink()


def _run_one(*, n: int, ckpt: str, ref, libero_root: str):
    _clear_cache()
    pipe = _build_pipe(ckpt, ref["prompt_toks"])

    if n == 1:
        obs_list = [_ref_obs(ref)]
        label = "ref-frame"
    else:
        obs_list = _load_obs_lib(libero_root, n)
        label = "stratified"

    t0 = time.perf_counter()
    pipe.calibrate(obs_list, percentile=99.9)
    cal_ms = (time.perf_counter() - t0) * 1000

    deploy_obs = _ref_obs(ref)
    for _ in range(3):
        pipe.infer(deploy_obs)
    out = _replay_with_ref_noise(pipe, ref["noise"], deploy_obs)

    cos_ref = _cos(out, ref["pytorch_raw"])
    max_ref = float(np.max(np.abs(out - ref["pytorch_raw"])))

    lat = []
    for _ in range(20):
        t0 = time.perf_counter()
        pipe.infer(deploy_obs)
        lat.append((time.perf_counter() - t0) * 1000)
    lat.sort()
    p50 = lat[len(lat) // 2]

    return {
        "n": n,
        "label": label,
        "calibrate_ms": round(cal_ms, 1),
        "cos_vs_fp32_ref": round(cos_ref, 6),
        "maxdiff_vs_ref": round(max_ref, 4),
        "p50_ms": round(p50, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="1,8,16,64,256")
    ap.add_argument("--libero-root", default=DEFAULT_LIBERO)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--ref", default=DEFAULT_REF)
    args = ap.parse_args()

    ref = _load_ref(args.ref)
    logger.info(
        "Reference loaded  prompt_len=%d  pytorch_raw[0][:4]=%s",
        len(ref["prompt_toks"]),
        np.round(ref["pytorch_raw"][0][:4], 4).tolist())

    ns = [int(x) for x in args.ns.split(",") if x.strip()]
    results = []
    for n in ns:
        logger.info("\n── N = %d ──", n)
        r = _run_one(n=n, ckpt=args.ckpt, ref=ref, libero_root=args.libero_root)
        logger.info(
            "  N=%d (%s):  calibrate=%.1f ms  cos=%.6f  maxdiff=%.4f  p50=%.2f ms",
            n, r["label"], r["calibrate_ms"], r["cos_vs_fp32_ref"],
            r["maxdiff_vs_ref"], r["p50_ms"])
        results.append(r)

    print("\n" + "=" * 70)
    print(f"{'N':>4}  {'strategy':>12}  {'calibrate':>11}  "
          f"{'cos vs ref':>11}  {'maxdiff':>9}  {'p50':>7}")
    print("-" * 70)
    for r in results:
        print(f"{r['n']:>4}  {r['label']:>12}  {r['calibrate_ms']:>9.1f} ms  "
              f"{r['cos_vs_fp32_ref']:>11.6f}  {r['maxdiff_vs_ref']:>9.4f}  "
              f"{r['p50_ms']:>5.2f} ms")
    print("=" * 70)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

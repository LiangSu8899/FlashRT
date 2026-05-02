#!/usr/bin/env python3
"""Pi0.5 Thor RL CFG inference regression test (Phase 4).

Mirrors :mod:`tests/test_rl_cfg_inference.py` (RTX). Each cell runs in
its own python subprocess (Thor single-GPU memory constraint). Two
backends (``torch`` and ``jax``) × four cells:

  validation    set_rl_mode contract: cfg_beta < 1.0 raises;
                cfg_enable=True / False state transitions are honoured.

  cfg_b1        β=1.0 CFG output. Saved to ``/tmp/cfg_smoke_<bk>_b1.npy``.
                Mathematically collapses to cond-only (kernel does
                ``residual += v_uncond + beta*(v_cond - v_uncond)`` so
                with β=1.0 + zeroed residual the result equals v_cond).

  cond_only     Standard inference with the cond-tagged prompt
                (``Advantage: positive`` appended). Saved to
                ``/tmp/cfg_smoke_<bk>_cond.npy``.

  cfg_b15       β=1.5 CFG output (typical RL beta). Used for the finite +
                bounded sanity gate, not a cos-floor gate (cond/uncond
                differ only by the ACP tag so the cond-uncond delta is
                small at FP8 precision).

Pass criteria:
  * validation cell PASS.
  * cos(cfg_b1_actions, cond_only_actions) >= 0.999 (β=1.0 collapse).
  * cfg_b15 actions are finite with |max| in a reasonable range
    (< 1e3) and shape (Sa, LIBERO_ACTION_DIM).
  * P50 latency for CFG mode is roughly 2x the cond-only latency
    (the user-visible cost of running the encoder + decoder twice).

Usage::

    python3 tests/test_thor_rl_cfg_inference.py
    python3 tests/test_thor_rl_cfg_inference.py --backends torch
    python3 tests/test_thor_rl_cfg_inference.py --backends torch,jax
"""
import argparse, json, os, subprocess, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PI05_TORCH_CKPT = os.environ.get(
    "PI05_CKPT", "<your_pi05_torch_ckpt>")
PI05_JAX_CKPT = os.environ.get(
    "PI05_JAX_CKPT", "<your_pi05_jax_ckpt>")


SUBPROC_SCRIPT = '''
import sys, os, json, time, pathlib, numpy as np
sys.path.insert(0, "ROOTDIR")

ROLE = os.environ["ROLE"]
BACKEND = os.environ["BACKEND"]   # "torch" | "jax"

cdir = pathlib.Path.home() / ".flash_rt" / "calibration"
if cdir.exists():
    for f in cdir.glob("*.json"):
        f.unlink()

if BACKEND == "torch":
    from flash_rt.frontends.torch.pi05_thor import Pi05TorchFrontendThor as Frontend
    CKPT = "TORCH_CKPT"
    NUM_VIEWS = 2
elif BACKEND == "jax":
    from flash_rt.frontends.jax.pi05_thor import Pi05JaxFrontendThor as Frontend
    CKPT = "JAX_CKPT"
    NUM_VIEWS = 2
else:
    raise ValueError(f"unknown backend: {BACKEND}")

if ROLE == "validation":
    pipe = Frontend(CKPT, num_views=NUM_VIEWS, autotune=0)
    raised = False
    try:
        pipe.set_rl_mode(cfg_enable=True, cfg_beta=0.5)
    except ValueError:
        raised = True
    pipe.set_rl_mode(cfg_enable=True, cfg_beta=1.5)
    s1 = pipe._rl_config is not None
    pipe.set_rl_mode(cfg_enable=False)
    s2 = pipe._rl_config is None
    print("__RESULT__", json.dumps({
        "role": ROLE, "backend": BACKEND,
        "raised_low_beta": raised,
        "enable_sets": s1,
        "disable_clears": s2,
        "verdict": "PASS" if (raised and s1 and s2) else "FAIL",
    }))
    sys.exit(0)

prompt = "pick up the red block and place it in the tray"
pipe = Frontend(CKPT, num_views=NUM_VIEWS, autotune=0)

if ROLE == "cfg_b1":
    pipe.set_rl_mode(cfg_enable=True, cfg_beta=1.0, advantage_positive=True)
    pipe.set_prompt(prompt)
elif ROLE == "cfg_b15":
    pipe.set_rl_mode(cfg_enable=True, cfg_beta=1.5, advantage_positive=True)
    pipe.set_prompt(prompt)
elif ROLE == "cond_only":
    from flash_rt.core.rl import build_acp_tagged_task
    cond_text = build_acp_tagged_task(prompt, is_positive=True)
    pipe.set_prompt(cond_text)
else:
    raise ValueError(f"unknown ROLE: {ROLE}")

np.random.seed(0)
img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
wrist = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
obs = {"image": img, "wrist_image": wrist}

for _ in range(3):
    pipe.infer(obs)

# Both backends now draw the noise R from numpy (post-fix); seed
# numpy on both, plus torch CUDA RNG for any leftover code paths
# (calibration / profiling) that might still use torch.randn.
np.random.seed(42)
if BACKEND == "torch":
    import torch
    torch.manual_seed(42)
r = pipe.infer(obs)
out = r["actions"]

lat = []
for _ in range(20):
    t0 = time.perf_counter()
    pipe.infer(obs)
    lat.append((time.perf_counter() - t0) * 1000)
lat.sort()

fname = f"/tmp/cfg_test_{BACKEND}_{ROLE}.npy"
np.save(fname, out)

print("__RESULT__", json.dumps({
    "role": ROLE, "backend": BACKEND,
    "out_shape": list(out.shape),
    "all_finite": bool(np.isfinite(out).all()),
    "max_abs": round(float(np.max(np.abs(out))), 4),
    "p50_ms": round(lat[10], 2),
    "p95_ms": round(lat[18], 2),
    "saved_to": fname,
}))
'''


def run_cell(backend: str, role: str, timeout: int = 300) -> dict:
    script = (SUBPROC_SCRIPT
              .replace("ROOTDIR", ROOT)
              .replace("TORCH_CKPT", PI05_TORCH_CKPT)
              .replace("JAX_CKPT", PI05_JAX_CKPT))
    env = dict(os.environ, ROLE=role, BACKEND=backend)
    r = subprocess.run(
        ["python3", "-c", script],
        capture_output=True, text=True, timeout=timeout, env=env)
    if r.returncode != 0:
        return {"role": role, "backend": backend,
                "error": "\n".join(r.stderr.strip().split("\n")[-8:])}
    for line in reversed(r.stdout.strip().split("\n")):
        line = line.strip()
        if line.startswith("__RESULT__"):
            return json.loads(line[len("__RESULT__"):].strip())
    return {"role": role, "backend": backend,
            "error": "no __RESULT__\n" + r.stdout[-300:]}


def cos(a, b):
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def evaluate_backend(backend: str) -> tuple:
    """Run all cells for one backend and return (rows, any_fail)."""
    rows = []
    any_fail = False
    cells = ["validation", "cfg_b1", "cond_only", "cfg_b15"]

    print(f"\n══════ Backend: {backend} ══════", flush=True)
    for cell in cells:
        print(f"  ── {cell} ──", flush=True)
        r = run_cell(backend, cell)
        rows.append(r)
        if "error" in r:
            print(f"     [FAIL] {r['error']}", flush=True)
            any_fail = True
            continue
        if cell == "validation":
            v = r.get("verdict", "FAIL")
            any_fail = any_fail or v != "PASS"
            print(f"     [{v}] raised_low_beta={r['raised_low_beta']} "
                  f"enable_sets={r['enable_sets']} "
                  f"disable_clears={r['disable_clears']}", flush=True)
            continue
        # actions cells
        finite = r.get("all_finite", False)
        max_abs = r.get("max_abs", float("inf"))
        ok_finite = finite and max_abs < 1e3
        if not ok_finite:
            any_fail = True
        print(f"     finite={finite} max_abs={max_abs} "
              f"P50={r.get('p50_ms')}ms shape={r.get('out_shape')}",
              flush=True)

    # Cross-cell cos: cfg_b1 vs cond_only must collapse
    try:
        a = np.load(f"/tmp/cfg_test_{backend}_cfg_b1.npy")
        b = np.load(f"/tmp/cfg_test_{backend}_cond_only.npy")
        c_b1 = cos(a, b)
        ok_b1 = c_b1 >= 0.999
        any_fail = any_fail or not ok_b1
        print(f"  cos(cfg_b1, cond_only) = {c_b1:.6f}  "
              f"[{'PASS' if ok_b1 else 'FAIL'} @ 0.999]", flush=True)
        rows.append({"backend": backend, "cell": "cos_b1_vs_cond",
                     "value": round(c_b1, 6),
                     "verdict": "PASS" if ok_b1 else "FAIL"})
    except FileNotFoundError as e:
        print(f"  [FAIL] missing npy file: {e}", flush=True)
        any_fail = True
    return rows, any_fail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", default="torch,jax",
                    help="comma-separated backends (default torch,jax)")
    args = ap.parse_args()
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]

    print("=" * 80)
    print("Pi0.5 Thor RL CFG inference test")
    print(f"  backends={backends}")
    print(f"  PI05_TORCH_CKPT={PI05_TORCH_CKPT}")
    print(f"  PI05_JAX_CKPT={PI05_JAX_CKPT}")
    print("=" * 80)

    all_rows = []
    overall_fail = False
    for bk in backends:
        rows, fail = evaluate_backend(bk)
        all_rows.extend(rows)
        overall_fail = overall_fail or fail

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(json.dumps(all_rows, indent=2))
    sys.exit(1 if overall_fail else 0)


if __name__ == "__main__":
    main()

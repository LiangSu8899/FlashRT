#!/usr/bin/env python3
"""GROOT Thor N>=2 calibration regression test.

Each (N) cell runs in its own python subprocess (Thor single-GPU memory
constraint). Pre-existing LIBERO obs bundles must live at
``/tmp/libero_obs_2v_n{N}.npz`` (extracted via the openpi-deploy-clean
pre-extraction step described in tests/INTERNAL_TESTING.md §1.2).

Pass criteria (Phase 1):
  N=1 (single-frame implicit / cache path) — finite output, doesn't raise
  N=8 (multi-frame percentile reduce)      — cos vs PyTorch FP32 ref >= 0.997,
                                             precision_spec written,
                                             all scales finite + positive,
                                             actions finite.

Usage::

    python3 tests/test_thor_groot_calibrate.py
    python3 tests/test_thor_groot_calibrate.py --ns 1
    python3 tests/test_thor_groot_calibrate.py --ns 1,8
"""
import argparse, json, os, subprocess, sys

FLASH_VLA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GROOT_CKPT = os.environ.get("GROOT_THOR_CHECKPOINT", "/workspace/openpi-main/openpi-main/groot_ckpt")
GROOT_REF = os.environ.get(
    "GROOT_THOR_REFERENCE",
    "/workspace/openpi-main/openpi-main/groot_ref/groot_ref_e2e_full.pt")

# Single subprocess script — parametrised by N. Calls calibrate(obs_list)
# explicitly (not relying on infer-side implicit calibration), then runs
# inference with the same fixed seed used to build groot_ref_e2e_full.pt.
SUBPROC_SCRIPT = '''
import sys, os, time, json, pathlib, math
sys.path.insert(0, "ROOTDIR")
import numpy as np
import torch

# Force a fresh calibration pass — wipe any cached scales for this ckpt.
cdir = pathlib.Path.home() / ".flash_rt" / "calibration"
if cdir.exists():
    for f in cdir.glob("*.json"):
        f.unlink()

n = int(os.environ["N"])
ref = torch.load("REF", map_location="cpu", weights_only=False)
ref_actions = ref["actions"][0].float().numpy()
img_np = ref["img_np"]
prompt = ref["prompt"]
T_ref = ref_actions.shape[0]

# LIBERO obs bundle — 2-view, n samples
data = np.load(f"/tmp/libero_obs_2v_n{n}.npz") if n > 1 else None
obs_list = []
if n == 1:
    # Single-frame path: feed the ref obs directly so calibrate(obs) hits
    # the legacy single-sample path (bit-equal to the cached scales path).
    obs_list = [{"image": img_np, "wrist_image": img_np}]
else:
    for i in range(n):
        obs_list.append({
            "image": data[f"img_{i}"],
            "wrist_image": data[f"wrist_{i}"],
        })

from flash_rt.frontends.torch.groot_thor import GrootTorchFrontendThor
pipe = GrootTorchFrontendThor("CKPT", num_views=2, autotune=3)
pipe.set_prompt(prompt)

t0 = time.perf_counter()
pipe.calibrate(obs_list, percentile=99.9, verbose=False)
calibrate_ms = (time.perf_counter() - t0) * 1000.0

# precision_spec invariants (rule #10 — GROOT writes it)
spec = pipe.precision_spec
spec_ok = (spec is not None and spec.source == "calibration")
scales_ok = True
if spec_ok:
    for entry in list(spec.encoder_layer_specs.values()) + list(
            spec.decoder_layer_specs.values()):
        s = entry.scale[0] if hasattr(entry, "scale") else None
        if s is None or not np.isfinite(s) or s <= 0:
            scales_ok = False
            break

# Match reference seed for the e2e action comparison
infer_obs = {"image": img_np, "wrist_image": img_np}
for _ in range(5):
    pipe.infer(infer_obs)
torch.manual_seed(123)
result = pipe.infer(infer_obs)
fvk_actions = result["actions"][:T_ref]
all_finite = bool(np.isfinite(fvk_actions).all())

def cos(a, b):
    a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

c = cos(fvk_actions, ref_actions)
maxdiff = float(np.max(np.abs(fvk_actions - ref_actions)))

# Latency (warm)
lat = []
for _ in range(20):
    t0 = time.perf_counter()
    pipe.infer(infer_obs)
    lat.append((time.perf_counter() - t0) * 1000.0)
lat.sort()

print(json.dumps({
    "n": n,
    "calibrate_ms": round(calibrate_ms, 1),
    "cos_vs_pytorch_ref": round(c, 6),
    "maxdiff": round(maxdiff, 4),
    "p50_ms": round(lat[10], 1),
    "actions_shape": list(fvk_actions.shape),
    "all_finite": all_finite,
    "precision_spec_written": spec_ok,
    "all_scales_positive_finite": scales_ok,
}))
'''


def run_n(n: int) -> dict:
    script = (SUBPROC_SCRIPT
              .replace("ROOTDIR", FLASH_VLA_ROOT)
              .replace("CKPT", GROOT_CKPT)
              .replace("REF", GROOT_REF))
    env = dict(os.environ, N=str(n))
    r = subprocess.run(
        ["python3", "-c", script],
        capture_output=True, text=True, timeout=600, env=env)
    if r.returncode != 0:
        return {"error": "\n".join(r.stderr.strip().split("\n")[-8:])}
    for line in reversed(r.stdout.strip().split("\n")):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    return {"error": "no JSON\n" + r.stdout[-300:]}


def verdict_for(n: int, r: dict) -> str:
    if "error" in r:
        return "FAIL"
    if not r.get("all_finite"):
        return "FAIL"
    if not r.get("all_scales_positive_finite"):
        return "FAIL"
    if not r.get("precision_spec_written"):
        return "FAIL"
    if n == 1:
        # Single-frame path: only sanity gates (no cos floor, since the
        # legacy path is its own bit-equal baseline).
        return "PASS"
    # N>=2 multi-frame: cos floor 0.997 vs PyTorch FP32 ref
    if r.get("cos_vs_pytorch_ref", 0.0) < 0.997:
        return "FAIL"
    return "PASS"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ns", default="1,8",
                   help="comma-separated sample counts (default 1,8)")
    args = p.parse_args()
    ns = [int(x) for x in args.ns.split(",")]

    print("=" * 88)
    print(f"GROOT Thor multi-frame calibration test — N={ns}")
    print(f"  ckpt={GROOT_CKPT}")
    print(f"  ref ={GROOT_REF}")
    print("=" * 88)

    results = []
    any_fail = False
    for n in ns:
        print(f"\n── N={n} ──", flush=True)
        r = run_n(n)
        v = verdict_for(n, r)
        any_fail = any_fail or (v != "PASS")
        if "error" in r:
            print(f"  [{v}] ERROR: {r['error']}")
        else:
            print(f"  [{v}] cos_vs_ref={r.get('cos_vs_pytorch_ref', '-')} "
                  f"maxdiff={r.get('maxdiff', '-')} "
                  f"calibrate_ms={r.get('calibrate_ms', '-')} "
                  f"P50={r.get('p50_ms', '-')}ms "
                  f"finite={r.get('all_finite')} "
                  f"spec={r.get('precision_spec_written')} "
                  f"scales_ok={r.get('all_scales_positive_finite')}")
        results.append({"n": n, "verdict": v, **r})

    print("\n" + "=" * 88)
    print("SUMMARY")
    print("=" * 88)
    for row in results:
        print(f"  N={row['n']:<3}  [{row['verdict']}]  "
              f"cos={row.get('cos_vs_pytorch_ref', '-')}  "
              f"calibrate_ms={row.get('calibrate_ms', '-')}")
    print(json.dumps(results, indent=2))
    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()

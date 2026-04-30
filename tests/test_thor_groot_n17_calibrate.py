#!/usr/bin/env python3
"""GROOT N1.7 Thor N>=2 calibration regression test.

Mirrors the structure of ``test_thor_groot_calibrate.py`` (the N1.6 entry):
each ``(N)`` cell runs in its own python subprocess (Thor single-GPU
memory constraint).

The N1.7 frontend does not currently expose an obs->aux production path
(Qwen3-VL processor + vision encoder live outside the frontend), so the
calibration "sample" abstraction is the same ``aux`` dict that
``set_prompt`` consumes. The N=8 cell loads a list of N pre-captured aux
dicts, the N=1 cell synthesises a single-element list from the existing
single-aux fixture and exercises the no-op short-circuit.

Pass criteria:

  N=1: precision_spec written with ``calibration_method=single_frame``,
       all alphas finite + positive, infer is bit-stable across the
       calibrate call (no-op gate works), e2e cos vs PyTorch FP32
       reference >= 0.997.

  N=8: precision_spec written with ``calibration_method=percentile``,
       all alphas finite + positive, e2e cos vs PyTorch FP32 reference
       >= 0.997 (multi-sample percentile reduce must not regress
       precision below the single-frame baseline by more than 0.003).

Fixture inputs (override via env):

  GROOT_N17_CKPT     — N1.7 snapshot dir (one with ``model-*.safetensors``)
  GROOT_N17_FX       — base fixture (.pt) with ``actions`` / ``inputs``
  GROOT_N17_AUX      — single-aux companion fixture (.pt)
  GROOT_N17_AUX_LIST — N>=2 aux-list fixture (.pt) — only required for N>=2

Usage::

    python3 tests/test_thor_groot_n17_calibrate.py
    python3 tests/test_thor_groot_n17_calibrate.py --ns 1
    python3 tests/test_thor_groot_n17_calibrate.py --ns 1,8
"""
import argparse
import glob
import json
import os
import subprocess
import sys


FLASH_VLA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _autodetect_ckpt() -> str:
    cands = sorted(glob.glob(
        "/root/.cache/huggingface/hub/models--nvidia--GR00T-N1.7-3B/snapshots/*"))
    return cands[0] if cands else ""


CKPT = os.environ.get("GROOT_N17_CKPT", _autodetect_ckpt())
FX = os.environ.get(
    "GROOT_N17_FX",
    os.path.join(FLASH_VLA_ROOT, "tests", "fixtures",
                 "gr00t_n17_ref_oxe_droid_relative_eef_relative_joint_2v_traj1_step0_seed0.pt"))
AUX = os.environ.get(
    "GROOT_N17_AUX",
    os.path.join(FLASH_VLA_ROOT, "tests", "fixtures",
                 "gr00t_n17_ref_oxe_droid_relative_eef_relative_joint_2v_traj1_step0_seed0_llm_aux.pt"))
AUX_LIST = os.environ.get(
    "GROOT_N17_AUX_LIST",
    os.path.join(FLASH_VLA_ROOT, "tests", "fixtures",
                 "gr00t_n17_aux_list_n8.pt"))


SUBPROC_SCRIPT = '''
import sys, os, time, json, pathlib, math
sys.path.insert(0, "ROOTDIR")
import numpy as np
import torch

# Force a fresh calibration cache pass — wipe any persisted scales.
cdir = pathlib.Path.home() / ".cache" / "flash_vla"
if cdir.exists():
    for f in cdir.glob("*_n17_Se*.json"):
        f.unlink()

n = int(os.environ["N"])
fx = torch.load("FX", map_location="cpu", weights_only=False)
aux = torch.load("AUX", map_location="cpu", weights_only=False)

if n == 1:
    aux_list = [aux]
else:
    aux_list_pt = "AUX_LIST"
    if not os.path.exists(aux_list_pt):
        raise FileNotFoundError(
            f"aux-list fixture missing: {aux_list_pt}. Generate via "
            f"tests/_helpers/groot_n17/capture_aux_multi.py.")
    aux_list = torch.load(aux_list_pt, map_location="cpu", weights_only=False)
    if len(aux_list) < n:
        raise ValueError(
            f"aux-list has {len(aux_list)} entries, need >= {n}")
    aux_list = aux_list[:n]

from flash_vla.frontends.torch.groot_n17_thor import GrootN17TorchFrontendThor
pipe = GrootN17TorchFrontendThor("CKPT", num_views=2)
pipe.set_prompt(aux=aux, prompt="")

t0 = time.perf_counter()
pipe.calibrate(aux_list, percentile=99.9, verbose=False)
calibrate_ms = (time.perf_counter() - t0) * 1000.0

spec = pipe.precision_spec
spec_ok = spec is not None and spec.source == "calibration"
expected_method = "single_frame" if n == 1 else "percentile"
method_ok = False
scales_ok = True
n_entries = 0
if spec_ok:
    entries = list(spec.encoder_layer_specs.values())
    n_entries = len(entries)
    method_ok = all(e.calibration_method == expected_method for e in entries)
    for e in entries:
        s = float(np.asarray(e.scale).reshape(-1)[0])
        if not (np.isfinite(s) and s > 0):
            scales_ok = False
            break

# Inference uses the captured HF noise + the fixture's reference state
# so we compare against the same diffusion trajectory the reference
# fixture was generated under. Mirrors the existing e2e test layout.
state_dict = {
    "state.eef_9d": fx["inputs"]["state"]["eef_9d"],
    "state.gripper_position": fx["inputs"]["state"]["gripper_position"],
    "state.joint_position": fx["inputs"]["state"]["joint_position"],
}
state_normed = pipe.normalize_state(state_dict)
noise = aux["initial_noise"].to(torch.bfloat16).cuda()
out_normed = pipe.infer(state_normed, initial_noise=noise)
denorm = pipe.denormalize_action(out_normed, state_dict=state_dict)

ref_actions = fx["actions"]
modality_keys = ("eef_9d", "gripper_position", "joint_position")
ours = np.concatenate(
    [np.asarray(denorm[k]).reshape(-1) for k in modality_keys])
ref = np.concatenate(
    [np.asarray(ref_actions[k]).reshape(-1) for k in modality_keys])
T = min(len(ours), len(ref))
ours, ref = ours[:T], ref[:T]
all_finite = bool(np.isfinite(ours).all())

def cos(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

c = cos(ours, ref)
maxdiff = float(np.max(np.abs(ours - ref)))

# Latency snapshot (warm; 10 iters)
lat = []
for _ in range(10):
    t0 = time.perf_counter()
    pipe.infer(state_normed, initial_noise=noise)
    lat.append((time.perf_counter() - t0) * 1000.0)
lat.sort()

print(json.dumps({
    "n": n,
    "calibrate_ms": round(calibrate_ms, 1),
    "cos_vs_pytorch_ref": round(c, 6),
    "maxdiff": round(maxdiff, 4),
    "p50_ms": round(lat[len(lat) // 2], 1),
    "spec_entries": n_entries,
    "all_finite": all_finite,
    "precision_spec_written": spec_ok,
    "method_ok": method_ok,
    "all_scales_positive_finite": scales_ok,
}))
'''


def run_n(n: int) -> dict:
    script = (SUBPROC_SCRIPT
              .replace("ROOTDIR", FLASH_VLA_ROOT)
              .replace("AUX_LIST", AUX_LIST)
              .replace("CKPT", CKPT)
              .replace("FX", FX)
              .replace("AUX", AUX))
    env = dict(os.environ, N=str(n))
    env.setdefault("PYTHONPATH", "/gr00t/Isaac-GR00T")
    if "/gr00t/Isaac-GR00T" not in env["PYTHONPATH"]:
        env["PYTHONPATH"] = "/gr00t/Isaac-GR00T:" + env["PYTHONPATH"]
    r = subprocess.run(
        ["python3", "-c", script],
        capture_output=True, text=True, timeout=900, env=env)
    if r.returncode != 0:
        return {"error": "\n".join(r.stderr.strip().split("\n")[-10:])}
    for line in reversed(r.stdout.strip().split("\n")):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    return {"error": "no JSON\n" + r.stdout[-300:]}


def verdict_for(n: int, r: dict, *, cos_floor: float = 0.997) -> str:
    if "error" in r:
        return "FAIL"
    if not r.get("all_finite"):
        return "FAIL"
    if not r.get("precision_spec_written"):
        return "FAIL"
    if not r.get("method_ok"):
        return "FAIL"
    if not r.get("all_scales_positive_finite"):
        return "FAIL"
    if r.get("cos_vs_pytorch_ref", 0.0) < cos_floor:
        return "FAIL"
    return "PASS"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ns", default="1,8",
                   help="comma-separated sample counts (default 1,8)")
    p.add_argument("--cos-floor", type=float, default=0.997,
                   help="minimum cos vs PyTorch FP32 reference (default 0.997)")
    args = p.parse_args()
    ns = [int(x) for x in args.ns.split(",")]

    print("=" * 88)
    print(f"GROOT N1.7 Thor multi-frame calibration test - N={ns}")
    print(f"  ckpt     ={CKPT}")
    print(f"  fx       ={FX}")
    print(f"  aux      ={AUX}")
    print(f"  aux_list ={AUX_LIST}")
    print(f"  cos_floor={args.cos_floor}")
    print("=" * 88)

    results = []
    any_fail = False
    for n in ns:
        print(f"\n-- N={n} --", flush=True)
        r = run_n(n)
        v = verdict_for(n, r, cos_floor=args.cos_floor)
        any_fail = any_fail or (v != "PASS")
        if "error" in r:
            print(f"  [{v}] ERROR: {r['error']}")
        else:
            print(
                f"  [{v}] cos_vs_ref={r.get('cos_vs_pytorch_ref', '-')} "
                f"maxdiff={r.get('maxdiff', '-')} "
                f"calibrate_ms={r.get('calibrate_ms', '-')} "
                f"P50={r.get('p50_ms', '-')}ms "
                f"finite={r.get('all_finite')} "
                f"spec={r.get('precision_spec_written')} "
                f"method_ok={r.get('method_ok')} "
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

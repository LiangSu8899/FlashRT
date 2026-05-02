"""Generate the CFG oracle fixture file.

Produces ``tests/fixtures/cfg_reference_outputs.npz`` containing, for
each ``β ∈ {1.0, 1.5, 2.0, 2.5}``:

  * ``ref_actions_b{β}``        — FP32 reference final actions (10, 7)
  * ``ref_v_cond_b{β}``         — FP32 reference per-step v_cond
                                   (num_steps, action_horizon, action_dim)
  * ``ref_v_uncond_b{β}``       — FP32 reference per-step v_uncond
  * ``ref_noise_b{β}``          — FP32 reference per-step noise
                                   (num_steps + 1, action_horizon, action_dim)

  * ``serial_actions_b{β}``     — FlashVLA serial CFG final actions (10, 7)
  * ``serial_v_cond_b{β}``      — FlashVLA serial per-step v_cond traces
                                   (num_steps, action_horizon, action_dim)
  * ``serial_v_uncond_b{β}``
  * ``serial_noise_b{β}``

  * ``batched_actions_b{β}``    — FlashVLA batched CFG final actions
  * ``batched_v_cond_b{β}``
  * ``batched_v_uncond_b{β}``
  * ``batched_noise_b{β}``

These fixtures are the ground-truth backstops for the C2/C3/C5 oracle
tests in ``tests/test_cfg_correctness_oracle.py``. Regenerate after
any change that legitimately affects the per-tensor numerics (FP8
calibration, kernel changes, weights migration).

Each subprocess uses a fixed seed; pipelines / refs each sample
internally via CUDA BF16 ``.normal_()`` so subprocess isolation does
not change the noise stream.

Run from the FlashVLA repo root inside the pi0-stablehlo container::

    python tools/generate_cfg_oracle_fixtures.py
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPO_ROOT / "tests/fixtures/cfg_reference_outputs.npz"
CKPT = os.environ.get(
    "PI05_LIBERO_PYTORCH_CHECKPOINT",
    "<ckpts>/pi05_libero_pytorch")
SEED = 424242
BETAS = [1.0, 1.5, 2.0, 2.5]


CHILD_REF = r"""
import sys, numpy as np, torch
from flash_rt.refs.pi05_cfg_reference import Pi05CFGReference
beta, out_path, ckpt = float(sys.argv[1]), sys.argv[2], sys.argv[3]
obs = {"observation/image": np.zeros((224,224,3), dtype=np.uint8),
       "observation/wrist_image": np.zeros((224,224,3), dtype=np.uint8),
       "observation/state": np.zeros(8, dtype=np.float32),
       "prompt": "pick up the cup"}
ref = Pi05CFGReference(config_name="pi05_libero", checkpoint_dir=ckpt, device="cuda")
torch.manual_seed(424242)
out = ref.infer(obs, beta=beta)
np.savez(out_path,
         actions=out["actions"], v_cond=out["v_cond_per_step"],
         v_uncond=out["v_uncond_per_step"], noise=out["noise_per_step"])
"""

CHILD_FVLA = r"""
import sys, ctypes, numpy as np, torch
from flash_rt.frontends.torch.pi05_rtx import Pi05TorchFrontendRtx
mode, beta, out_path, ckpt = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4]
obs = {"image": np.zeros((224,224,3), dtype=np.uint8),
       "wrist_image": np.zeros((224,224,3), dtype=np.uint8),
       "state": np.zeros(8, dtype=np.float32)}
rt = Pi05TorchFrontendRtx(ckpt, num_views=2)
if mode == "batched":
    rt.set_batched_mode(enable=True)
rt.set_rl_mode(cfg_enable=True, cfg_beta=beta)
rt.set_prompt("pick up the cup")
rt.calibrate([obs])

# Production inference (graph replay) for the final actions.
torch.manual_seed(424242)
out = rt.infer(obs)
actions = out["actions"]

# Now trace mode: eager re-run that captures per-step velocities.
# CRITICAL: after the production infer() above the input_noise_buf
# holds the *final* action chunk (the in-place denoise integration
# overwrites it). We must reset it to a fresh noise sample under the
# same seed so the eager run starts from the same byte-equivalent
# initial noise as production. We do this by reusing the frontend's
# noise-staging helpers, then bypass the captured graph via
# run_pipeline_eager_with_trace().
p = rt.pipeline
p.enable_velocity_trace()
torch.manual_seed(424242)
rt._noise_buf.normal_()
if mode == "batched":
    for b in range(2):
        rt._noise_buf_b2[b].copy_(rt._noise_buf)
    rt._copy_tensor_to_pipeline_buf_stream(
        rt._noise_buf_b2, p.input_noise_buf_b2, 0)
else:
    rt._copy_tensor_to_pipeline_buf_stream(
        rt._noise_buf, p.input_noise_buf, 0)
rt._cudart.cudaDeviceSynchronize()

# Stage images too (mirrors what infer() does internally).
stacked = rt._stack_images(obs)
if mode == "batched":
    for b in range(2):
        rt._img_buf_b2[b].copy_(stacked)
    rt._copy_tensor_to_pipeline_buf_stream(
        rt._img_buf_b2, p.input_images_buf_b2, 0)
else:
    rt._copy_tensor_to_pipeline_buf_stream(
        stacked, p.input_images_buf, 0)
rt._cudart.cudaDeviceSynchronize()

p.run_pipeline_eager_with_trace(stream=0)
tr = p.read_velocity_trace()
np.savez(out_path,
         actions=actions, v_cond=tr["v_cond_per_step"],
         v_uncond=tr["v_uncond_per_step"], noise=tr["noise_per_step"])
"""


def _run_child(env_args, code) -> None:
    cmd = [sys.executable, "-c", code, *env_args]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900,
                       cwd=str(REPO_ROOT))
    if r.returncode != 0:
        raise RuntimeError(
            f"child failed rc={r.returncode}\n"
            f"cmd: {' '.join(cmd[:6])}...\n"
            f"stderr tail:\n{r.stderr[-2500:]}")


def main() -> None:
    if not Path(CKPT).exists():
        print(f"checkpoint not found at {CKPT}; aborting")
        sys.exit(1)
    with tempfile.TemporaryDirectory() as td:
        bundle = {}
        for beta in BETAS:
            beta_tag = f"b{beta:.1f}"
            print(f"\n=== β = {beta} ===")

            # Reference
            ref_npz = os.path.join(td, f"ref_{beta_tag}.npz")
            print("  reference (FP32) ...", end=" ", flush=True)
            _run_child([str(beta), ref_npz, CKPT], CHILD_REF)
            d = np.load(ref_npz)
            bundle[f"ref_actions_{beta_tag}"] = d["actions"]
            bundle[f"ref_v_cond_{beta_tag}"] = d["v_cond"]
            bundle[f"ref_v_uncond_{beta_tag}"] = d["v_uncond"]
            bundle[f"ref_noise_{beta_tag}"] = d["noise"]
            print(f"actions shape={d['actions'].shape}")

            # FlashVLA serial
            ser_npz = os.path.join(td, f"ser_{beta_tag}.npz")
            print("  FlashVLA serial ... ", end=" ", flush=True)
            _run_child(["serial", str(beta), ser_npz, CKPT], CHILD_FVLA)
            d = np.load(ser_npz)
            bundle[f"serial_actions_{beta_tag}"] = d["actions"]
            bundle[f"serial_v_cond_{beta_tag}"] = d["v_cond"]
            bundle[f"serial_v_uncond_{beta_tag}"] = d["v_uncond"]
            bundle[f"serial_noise_{beta_tag}"] = d["noise"]
            print(f"actions shape={d['actions'].shape}")

            # FlashVLA batched
            bat_npz = os.path.join(td, f"bat_{beta_tag}.npz")
            print("  FlashVLA batched ...", end=" ", flush=True)
            _run_child(["batched", str(beta), bat_npz, CKPT], CHILD_FVLA)
            d = np.load(bat_npz)
            bundle[f"batched_actions_{beta_tag}"] = d["actions"]
            bundle[f"batched_v_cond_{beta_tag}"] = d["v_cond"]
            bundle[f"batched_v_uncond_{beta_tag}"] = d["v_uncond"]
            bundle[f"batched_noise_{beta_tag}"] = d["noise"]
            print(f"actions shape={d['actions'].shape}")

        FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez(FIXTURE_PATH, **bundle)
        n_keys = len(bundle)
        n_bytes = FIXTURE_PATH.stat().st_size
        print(f"\nWrote {n_keys} arrays to {FIXTURE_PATH}  ({n_bytes/1024:.1f} KB)")


if __name__ == "__main__":
    main()

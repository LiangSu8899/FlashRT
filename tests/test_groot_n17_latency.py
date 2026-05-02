"""Phase 3d latency snapshot — eager infer (no CUDA Graph yet)."""
import glob
from pathlib import Path
import pytest
import torch


_CKPT_GLOB = "/root/.cache/huggingface/hub/models--nvidia--GR00T-N1.7-3B/snapshots/*"
_FIXTURE = Path(
    "/work/tests/fixtures/gr00t_n17_ref_oxe_droid_relative_eef_relative_joint_2v_traj1_step0_seed0.pt")
_AUX = _FIXTURE.with_name(_FIXTURE.stem + "_llm_aux.pt")


def test_eager_infer_latency():
    if not torch.cuda.is_available():
        pytest.skip("CUDA")
    matches = sorted(glob.glob(_CKPT_GLOB))
    if not matches or not _FIXTURE.exists() or not _AUX.exists():
        pytest.skip("ckpt/fixtures missing")
    from flash_rt.frontends.torch.groot_n17_thor import GrootN17TorchFrontendThor

    fe = GrootN17TorchFrontendThor(
        matches[0], num_views=2,
        embodiment_tag="oxe_droid_relative_eef_relative_joint")
    aux = torch.load(_AUX, weights_only=False, map_location="cpu")
    fx = torch.load(_FIXTURE, weights_only=False, map_location="cpu")
    fe.set_prompt(aux=aux, prompt="x")
    state_normed = fe.normalize_state({
        "state.eef_9d": fx["inputs"]["state"]["eef_9d"],
        "state.gripper_position": fx["inputs"]["state"]["gripper_position"],
        "state.joint_position": fx["inputs"]["state"]["joint_position"],
    })
    noise = aux["initial_noise"].to("cuda").half()

    # Warmup (set_prompt already does one), then 30 timed runs
    for _ in range(3):
        _ = fe.infer(state_normed, initial_noise=noise)
    torch.cuda.synchronize()

    times = []
    for _ in range(30):
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        _ = fe.infer(state_normed, initial_noise=noise)
        e1.record()
        torch.cuda.synchronize()
        times.append(e0.elapsed_time(e1))
    times.sort()
    n = len(times)
    p50 = times[n // 2]
    p10 = times[n // 10]
    p90 = times[(n * 9) // 10]
    mean = sum(times) / n
    print(f"\n[eager infer latency, 30 runs] "
          f"p10={p10:.2f}ms  p50={p50:.2f}ms  mean={mean:.2f}ms  p90={p90:.2f}ms",
          flush=True)

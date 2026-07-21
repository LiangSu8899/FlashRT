from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from flash_rt.models.cosmos3_edge.dump_replay import (  # noqa: E402
    EDGE_ACTION_MODEL_SHAPE,
    EDGE_FLAT_DIM,
    EDGE_VISION_SHAPE,
    EdgeDenoiseDump,
    EdgeDenoiseReplay,
)
from flash_rt.models.cosmos3_edge.static_unipc import EdgeStaticUniPCScheduler  # noqa: E402


def test_cosmos3_edge_split_flat_geometry():
    class _SyntheticDump:
        final_vision = torch.empty(EDGE_VISION_SHAPE)
        vision_dim = final_vision.numel()

        def split_flat(self, flat):
            flat = flat.reshape(-1)
            return (
                flat[: self.vision_dim].reshape(self.final_vision.shape),
                flat[self.vision_dim :].reshape(EDGE_ACTION_MODEL_SHAPE),
            )

    dump = _SyntheticDump()
    flat = torch.arange(EDGE_FLAT_DIM, dtype=torch.float32)
    vision, action = dump.split_flat(flat)

    assert vision.shape == EDGE_VISION_SHAPE
    assert action.shape == EDGE_ACTION_MODEL_SHAPE
    assert action[0, 0].item() == dump.vision_dim


def test_cosmos3_edge_local_dump_replays_scheduler_when_present():
    candidates = [
        Path(
            "/home/heima-thor/suliang/nvidia_fp8_80ms/official/flashrt-public/"
            "dev_scratch_cosmos3_thor/edge_av_inverse_0/tensors.safetensors"
        ),
        Path(
            "/work/official/flashrt-public/dev_scratch_cosmos3_thor/"
            "edge_av_inverse_0/tensors.safetensors"
        ),
    ]
    dump_path = next((path for path in candidates if path.exists()), None)
    if dump_path is None:
        pytest.skip("local Cosmos3-Edge P0 dump is not available")

    dump = EdgeDenoiseDump(dump_path)
    dump.validate_geometry()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = EdgeDenoiseReplay(dump, device=device).replay()
    final_vision_diff = (result.final_parts.vision.float() - dump.final_vision.float()).abs().max().item()

    assert result.timesteps[0] == 999
    assert result.timesteps[-1] == 256
    assert result.final_parts.vision.shape == EDGE_VISION_SHAPE
    assert result.final_parts.action_model.shape == EDGE_ACTION_MODEL_SHAPE
    assert final_vision_diff < 1e-6
    assert result.max_input_abs_diff < (1e-6 if device == "cuda" else 6e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="native static UniPC replay requires CUDA")
def test_cosmos3_edge_static_unipc_replays_scheduler_when_present():
    candidates = [
        Path(
            "/home/heima-thor/suliang/nvidia_fp8_80ms/official/flashrt-public/"
            "dev_scratch_cosmos3_thor/edge_av_inverse_0/tensors.safetensors"
        ),
        Path(
            "/work/official/flashrt-public/dev_scratch_cosmos3_thor/"
            "edge_av_inverse_0/tensors.safetensors"
        ),
    ]
    dump_path = next((path for path in candidates if path.exists()), None)
    if dump_path is None:
        pytest.skip("local Cosmos3-Edge P0 dump is not available")

    dump = EdgeDenoiseDump(dump_path)
    reference = EdgeDenoiseReplay(dump, device="cuda").replay(check_inputs=False).final_flat.cuda()
    static_scheduler = EdgeStaticUniPCScheduler(dump.num_steps, device=torch.device("cuda"))
    if not static_scheduler.native_available:
        pytest.skip("native UniPC step binding is not built")

    latent = dump.step_noise(0).to(device="cuda")
    static_scheduler.reset(latent)
    for step in range(dump.num_steps):
        latent = static_scheduler.step(latent, dump.step_velocity(step).to(device="cuda"), step)
    torch.cuda.synchronize()

    assert torch.allclose(latent, reference, rtol=0, atol=5e-5)


def test_cosmos3_edge_flashrt_engine_is_public():
    from flash_rt.models.cosmos3_edge import EdgeDenoiseFlashRT

    assert EdgeDenoiseFlashRT.__name__ == "EdgeDenoiseFlashRT"


def test_cosmos3_edge_prepare_slim_can_derive_condition_reference():
    from flash_rt.models.cosmos3_edge.action_only_official import (
        _derive_condition_reference_from_prepare_payload,
        _derive_initial_noise_from_prepare_payload,
        _prepare_payload_with_derived_initial_noise,
        _prepare_payload_with_derived_condition_reference,
        _slim_prepare_payload,
    )

    import numpy as np

    sequence_plans = [SimpleNamespace(has_vision=True, has_action=True, has_sound=False)]
    x0_vision = torch.arange(2 * 3, dtype=torch.float32).reshape(1, 2, 1, 1, 3)
    x0_action = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.bfloat16)
    gen_data_clean = SimpleNamespace(
        raw_state_vision=[torch.empty(1, 3, 2, 4, 4)],
        x0_tokens_vision=[x0_vision],
        x0_tokens_action=[x0_action],
        x0_tokens_sound=None,
        raw_action_dim=[torch.tensor(2)],
        num_vision_items_per_sample=None,
    )
    seed = [7]
    mask_vision = torch.tensor([1, 0, 1, 0, 1, 0], dtype=torch.float32).reshape(x0_vision.shape)
    mask_action = torch.zeros_like(x0_action, dtype=torch.float32)
    condition_mask = [torch.cat([mask_vision.reshape(-1), mask_action.reshape(-1)])]
    pure_vision = torch.from_numpy(
        np.random.RandomState(seed[0]).standard_normal(tuple(x0_vision.shape)).astype("float32")
    )
    pure_action = torch.from_numpy(
        np.random.RandomState(seed[0]).standard_normal(tuple(x0_action.shape)).astype("float32")
    ).to(torch.bfloat16)
    expected_noise_action = pure_action.clone()
    expected_noise_action[:, 2:] = 0
    initial_noise = [
        torch.cat(
            [
                (mask_vision * x0_vision + (1.0 - mask_vision) * pure_vision).reshape(-1),
                expected_noise_action.reshape(-1).to(torch.float32),
            ]
        )
    ]
    expected_action = x0_action.to(torch.float32).clone()
    expected_action[:, 2:] = 0
    condition_reference = [torch.cat([x0_vision.reshape(-1), expected_action.reshape(-1)])]
    payload = (sequence_plans, gen_data_clean, [], [], initial_noise, condition_reference, condition_mask)

    derived = _derive_condition_reference_from_prepare_payload(payload)
    assert torch.equal(derived[0], condition_reference[0])
    derived_noise = _derive_initial_noise_from_prepare_payload(payload, seed)
    assert torch.equal(derived_noise[0], initial_noise[0])

    slim = _slim_prepare_payload(
        payload,
        no_raw_state_vision=True,
        derive_condition_reference=True,
        derive_initial_noise=True,
    )
    assert slim[1].raw_state_vision is None
    assert slim[4] is None
    assert slim[5] is None

    restored = _prepare_payload_with_derived_initial_noise(slim, seed)
    restored = _prepare_payload_with_derived_condition_reference(restored)
    assert torch.equal(restored[4][0], initial_noise[0])
    assert torch.equal(restored[5][0], condition_reference[0])

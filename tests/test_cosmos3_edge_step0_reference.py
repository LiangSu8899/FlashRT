import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from flash_rt.models.cosmos3_edge.boundary_dump import EdgeBoundaryDump  # noqa: E402
from flash_rt.models.cosmos3_edge.denoise_ref import EdgeDenoiseTorchReference  # noqa: E402
from flash_rt.models.cosmos3_edge.dump_replay import EdgeDenoiseDump  # noqa: E402
from flash_rt.models.cosmos3_edge.layer_ref import EdgeTransformerTorchReference  # noqa: E402
from flash_rt.models.cosmos3_edge.static_engine import EdgeStaticBufferEngine  # noqa: E402
from flash_rt.models.cosmos3_edge.weights import EdgeTransformerWeights  # noqa: E402


def _local_paths() -> tuple[Path, Path] | None:
    checkpoint_raw = os.environ.get("COSMOS3_EDGE_CHECKPOINT")
    boundary_raw = os.environ.get("COSMOS3_EDGE_BOUNDARY_DUMP")
    if not checkpoint_raw or not boundary_raw:
        return None
    paths = (Path(checkpoint_raw).expanduser(), Path(boundary_raw).expanduser())
    return paths if paths[0].exists() and paths[1].exists() else None


def _local_denoise_dump_path() -> Path | None:
    raw = os.environ.get("COSMOS3_EDGE_REFERENCE_DUMP")
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path if path.exists() else None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="full step0 reference requires CUDA")
def test_cosmos3_edge_step0_full_sequence_reconstruction_matches_boundary():
    paths = _local_paths()
    if paths is None:
        pytest.skip("local Cosmos3-Edge checkpoint/boundary dump is not available")
    checkpoint, dump_path = paths

    dump = EdgeBoundaryDump(dump_path)
    ref = EdgeTransformerTorchReference(EdgeTransformerWeights(checkpoint), device="cuda")

    with torch.no_grad():
        full = ref.full_sequence_for_step(dump, dump.tensors["steps/00/noise_x"], dump.tensors["steps/00/timestep"])

    target = dump.layer0_input_full.to(device="cuda", dtype=torch.bfloat16)
    cos = torch.nn.functional.cosine_similarity(full.float().flatten(), target.float().flatten(), dim=0)

    assert cos.item() > 0.9999
    assert (full.float() - target.float()).abs().max().item() < 1e-2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="full step0 reference requires CUDA")
def test_cosmos3_edge_step0_torch_reference_matches_action_velocity():
    paths = _local_paths()
    if paths is None:
        pytest.skip("local Cosmos3-Edge checkpoint/boundary dump is not available")
    checkpoint, dump_path = paths

    dump = EdgeBoundaryDump(dump_path)
    ref = EdgeTransformerTorchReference(EdgeTransformerWeights(checkpoint), device="cuda")

    with torch.no_grad():
        action_velocity = ref.action_velocity_for_step(
            dump,
            dump.tensors["steps/00/noise_x"],
            dump.tensors["steps/00/timestep"],
        )
        und_cache = ref.precompute_und_kv_cache(dump)
        cached_action_velocity = ref.action_velocity_for_step_with_und_cache(
            dump,
            dump.tensors["steps/00/noise_x"],
            dump.tensors["steps/00/timestep"],
            und_cache,
        )

    official = dump.tensors["steps/00/velocity"]
    official_action = official[-60 * 64 :].reshape(60, 64).to(device="cuda", dtype=torch.bfloat16)
    cos = torch.nn.functional.cosine_similarity(action_velocity.float().flatten(), official_action.float().flatten(), dim=0)

    assert cos.item() > 0.9999
    assert (action_velocity.float() - official_action.float()).abs().max().item() < 2e-2
    assert (cached_action_velocity.float() - action_velocity.float()).abs().max().item() == 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="reference denoise requires CUDA")
def test_cosmos3_edge_reference_denoise_one_step_matches_dump():
    paths = _local_paths()
    denoise_path = _local_denoise_dump_path()
    if paths is None or denoise_path is None:
        pytest.skip("local Cosmos3-Edge checkpoint/dumps are not available")
    checkpoint, boundary_path = paths

    ref = EdgeDenoiseTorchReference(
        EdgeDenoiseDump(denoise_path),
        EdgeBoundaryDump(boundary_path),
        EdgeTransformerWeights(checkpoint),
        device="cuda",
    )

    with torch.no_grad():
        result = ref.run(max_steps=1)

    assert result.steps_run == 1
    assert result.timesteps == (999,)
    assert result.max_input_abs_diff == 0.0
    assert result.max_velocity_abs_diff < 2e-2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="static engine requires CUDA")
def test_cosmos3_edge_static_engine_step0_contract():
    paths = _local_paths()
    if paths is None:
        pytest.skip("local Cosmos3-Edge checkpoint/boundary dump is not available")
    checkpoint, boundary_path = paths

    dump = EdgeBoundaryDump(boundary_path)
    engine = EdgeStaticBufferEngine(EdgeTransformerWeights(checkpoint), dump, device="cuda")
    if not getattr(engine.reference, "native_attention_available", False):
        pytest.skip("native attention extension is not available")

    with torch.no_grad():
        full = engine.full_sequence_for_step(dump.tensors["steps/00/noise_x"], dump.tensors["steps/00/timestep"])
        velocity = engine.flat_velocity_for_step(dump.tensors["steps/00/noise_x"], dump.tensors["steps/00/timestep"])

    target_full = dump.layer0_input_full.to(device="cuda", dtype=torch.bfloat16)
    official = dump.tensors["steps/00/velocity"].to(device="cuda", dtype=torch.bfloat16)
    vision_dim = official.numel() - 60 * 64

    assert torch.nn.functional.cosine_similarity(full.float().flatten(), target_full.float().flatten(), dim=0) > 0.9999
    assert (full.float() - target_full.float()).abs().max().item() < 1e-2
    assert velocity[:vision_dim].float().abs().max().item() == 0.0
    assert (velocity[vision_dim:].float() - official[vision_dim:].float()).abs().max().item() < 2e-2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="static engine CUDA graph requires CUDA")
def test_cosmos3_edge_static_engine_velocity_graph_matches_eager():
    paths = _local_paths()
    if paths is None:
        pytest.skip("local Cosmos3-Edge checkpoint/boundary dump is not available")
    checkpoint, boundary_path = paths

    dump = EdgeBoundaryDump(boundary_path)
    engine = EdgeStaticBufferEngine(EdgeTransformerWeights(checkpoint), dump, device="cuda")
    if not getattr(engine.reference, "graph_attention_available", False):
        pytest.skip("graph-safe native attention extension is not available")

    with torch.no_grad():
        eager = engine.flat_velocity_for_step(
            dump.tensors["steps/00/noise_x"],
            dump.tensors["steps/00/timestep"],
        ).clone()
        graph = engine.replay_velocity_graph(
            dump.tensors["steps/00/noise_x"],
            dump.tensors["steps/00/timestep"],
        ).clone()

    assert (graph.float() - eager.float()).abs().max().item() == 0.0

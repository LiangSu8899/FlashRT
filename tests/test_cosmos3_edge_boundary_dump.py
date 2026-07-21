from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from flash_rt.models.cosmos3_edge.boundary_dump import (  # noqa: E402
    EDGE_BOUNDARY_ACTION_TOKENS,
    EDGE_BOUNDARY_GEN_TOKENS,
    EDGE_BOUNDARY_UND_TOKENS,
    EDGE_BOUNDARY_VISION_TOKENS,
    EDGE_HEAD_DIM,
    EDGE_HIDDEN_SIZE,
    EdgeBoundaryDump,
)


def _local_boundary_dump() -> Path | None:
    candidates = [
        Path(
            "/home/heima-thor/suliang/nvidia_fp8_80ms/official/flashrt-public/"
            "dev_scratch_cosmos3_thor/edge_av_inverse_0_boundary_step0/tensors.safetensors"
        ),
        Path(
            "/work/official/flashrt-public/dev_scratch_cosmos3_thor/"
            "edge_av_inverse_0_boundary_step0/tensors.safetensors"
        ),
    ]
    return next((path for path in candidates if path.exists()), None)


def _local_prelayer_boundary_dump() -> Path | None:
    candidates = [
        Path(
            "/home/heima-thor/suliang/nvidia_fp8_80ms/official/flashrt-public/"
            "dev_scratch_cosmos3_thor/edge_av_inverse_0/official_action_only_live_prelayer_boundary.safetensors"
        ),
        Path(
            "/work/official/flashrt-public/dev_scratch_cosmos3_thor/"
            "edge_av_inverse_0/official_action_only_live_prelayer_boundary.safetensors"
        ),
    ]
    return next((path for path in candidates if path.exists()), None)


def _local_slim_prepare_dump() -> Path | None:
    candidates = [
        Path(
            "/home/heima-thor/suliang/nvidia_fp8_80ms/official/flashrt-public/"
            "dev_scratch_cosmos3_thor/edge_av_inverse_0/"
            "official_action_only_live_prelayer_prepare_slim_derive_noise_dump.pt"
        ),
        Path(
            "/work/official/flashrt-public/dev_scratch_cosmos3_thor/"
            "edge_av_inverse_0/official_action_only_live_prelayer_prepare_slim_derive_noise_dump.pt"
        ),
    ]
    return next((path for path in candidates if path.exists()), None)


def _local_text_embedding_shard() -> Path | None:
    candidates = [
        Path(
            "/home/heima-thor/suliang/nvidia_fp8_80ms/models/Cosmos3-Edge/"
            "transformer/diffusion_pytorch_model-00001-of-00002.safetensors"
        ),
        Path("/work/models/Cosmos3-Edge/transformer/diffusion_pytorch_model-00001-of-00002.safetensors"),
    ]
    return next((path for path in candidates if path.exists()), None)


def _local_cosmos3_edge_model_root() -> Path | None:
    candidates = [
        Path("/home/heima-thor/suliang/nvidia_fp8_80ms/models/Cosmos3-Edge"),
        Path("/work/models/Cosmos3-Edge"),
    ]
    return next((path for path in candidates if path.exists()), None)


def test_cosmos3_edge_boundary_dump_geometry_when_present():
    dump_path = _local_boundary_dump()
    if dump_path is None:
        pytest.skip("local Cosmos3-Edge boundary dump is not available")

    dump = EdgeBoundaryDump(dump_path)
    dump.validate_geometry()
    shapes = dump.shapes

    assert shapes.und_tokens == EDGE_BOUNDARY_UND_TOKENS
    assert shapes.gen_tokens == EDGE_BOUNDARY_GEN_TOKENS
    assert shapes.vision_tokens == EDGE_BOUNDARY_VISION_TOKENS
    assert shapes.action_tokens == EDGE_BOUNDARY_ACTION_TOKENS
    assert shapes.hidden_size == EDGE_HIDDEN_SIZE
    assert shapes.head_dim == EDGE_HEAD_DIM
    assert dump.rope_cos_full.shape == (EDGE_BOUNDARY_GEN_TOKENS, EDGE_HEAD_DIM)


def test_cosmos3_edge_boundary_dump_from_tensors_when_present():
    dump_path = _local_boundary_dump()
    if dump_path is None:
        pytest.skip("local Cosmos3-Edge boundary dump is not available")

    file_dump = EdgeBoundaryDump(dump_path)
    memory_dump = EdgeBoundaryDump.from_tensors(dict(file_dump.tensors), path="<unit-test>")
    memory_dump.validate_geometry()

    assert memory_dump.path == Path("<unit-test>")
    assert memory_dump.shapes == file_dump.shapes
    assert torch.equal(memory_dump.rope_cos_full, file_dump.rope_cos_full)


def test_cosmos3_edge_prepare_payload_derives_step0_vfm_boundary_when_present():
    boundary_path = _local_prelayer_boundary_dump()
    prepare_path = _local_slim_prepare_dump()
    if boundary_path is None or prepare_path is None:
        pytest.skip("local Cosmos3-Edge pre-layer boundary/prepare dumps are not available")

    from safetensors.torch import load_file

    from flash_rt.models.cosmos3_edge.action_only_official import (  # noqa: E402
        _derive_step0_vfm_boundary_from_prepare_payload,
    )

    prepare = torch.load(prepare_path, map_location="cpu", weights_only=False)
    derived = _derive_step0_vfm_boundary_from_prepare_payload(prepare["payload"], seed=0)
    boundary = load_file(str(boundary_path), device="cpu")

    assert len(derived) == 27
    for key, tensor in derived.items():
        assert key in boundary
        expected = boundary[key]
        assert tuple(tensor.shape) == tuple(expected.shape), key
        assert torch.equal(tensor.cpu(), expected.cpu()), key


def test_cosmos3_edge_prepare_payload_derives_step0_causal_seq_when_weights_present():
    boundary_path = _local_prelayer_boundary_dump()
    prepare_path = _local_slim_prepare_dump()
    embedding_path = _local_text_embedding_shard()
    if boundary_path is None or prepare_path is None or embedding_path is None:
        pytest.skip("local Cosmos3-Edge pre-layer boundary/prepare/model dumps are not available")

    from safetensors import safe_open
    from safetensors.torch import load_file

    from flash_rt.models.cosmos3_edge.action_only_official import (  # noqa: E402
        _derive_step0_vfm_boundary_from_prepare_payload,
    )

    prepare = torch.load(prepare_path, map_location="cpu", weights_only=False)
    with safe_open(str(embedding_path), framework="pt", device="cpu") as handle:
        embedding = handle.get_tensor("embed_tokens.weight")
    derived = _derive_step0_vfm_boundary_from_prepare_payload(
        prepare["payload"],
        seed=0,
        text_embedding_weight=embedding,
    )
    boundary = load_file(str(boundary_path), device="cpu")

    assert len(derived) == 28
    assert torch.equal(derived["s00/lm_in/causal_seq"].cpu(), boundary["s00/lm_in/causal_seq"].cpu())


def test_cosmos3_edge_full_only_seq_derivation_probe_when_weights_present():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the full_only_seq derivation probe")
    boundary_path = _local_prelayer_boundary_dump()
    model_root = _local_cosmos3_edge_model_root()
    if boundary_path is None or model_root is None:
        pytest.skip("local Cosmos3-Edge pre-layer boundary/model root is not available")

    from safetensors.torch import load_file

    from flash_rt.models.cosmos3_edge import EdgeTransformerWeights
    from flash_rt.models.cosmos3_edge.layer_ref import EdgeTransformerTorchReference

    dump = EdgeBoundaryDump(boundary_path)
    boundary = load_file(str(boundary_path), device="cpu")
    weights = EdgeTransformerWeights(model_root)
    ref = EdgeTransformerTorchReference(weights, device="cuda", dtype=torch.bfloat16)
    full_probe = ref.full_sequence_for_step(
        dump,
        dump.tensors["steps/00/noise_x"].to("cuda"),
        dump.tensors["steps/00/timestep"].to("cuda"),
    ).cpu()

    expected = boundary["s00/lm_in/full_only_seq"]
    assert torch.equal(full_probe, expected)


def test_cosmos3_edge_slim_prepare_derives_executable_boundary_when_present():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the executable boundary derivation probe")
    boundary_path = _local_prelayer_boundary_dump()
    prepare_path = _local_slim_prepare_dump()
    model_root = _local_cosmos3_edge_model_root()
    if boundary_path is None or prepare_path is None or model_root is None:
        pytest.skip("local Cosmos3-Edge pre-layer boundary/prepare/model root is not available")

    from safetensors.torch import load_file

    from flash_rt.models.cosmos3_edge.action_only_official import (  # noqa: E402
        _derive_step0_executable_boundary_from_prepare_artifact,
    )

    derived = _derive_step0_executable_boundary_from_prepare_artifact(
        prepare_path,
        model_root,
        seed=0,
        device="cuda",
    )
    boundary = load_file(str(boundary_path), device="cpu")
    dump = EdgeBoundaryDump.from_tensors(derived, path="<derived-prepare>")
    dump.validate_geometry()

    assert len(derived) == 31
    assert torch.equal(derived["s00/layers/00/input/causal_seq"], derived["s00/lm_in/causal_seq"])
    assert torch.equal(derived["s00/layers/00/input/full_only_seq"], derived["s00/lm_in/full_only_seq"])
    assert torch.equal(derived["s00/lm_in/full_only_seq"], boundary["s00/lm_in/full_only_seq"])

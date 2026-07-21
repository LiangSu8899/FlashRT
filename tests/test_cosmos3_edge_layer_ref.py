import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from flash_rt.models.cosmos3_edge.boundary_dump import EdgeBoundaryDump  # noqa: E402
from flash_rt.frontends.torch._cosmos3_edge_thor_spec import SPEC  # noqa: E402
from flash_rt.models.cosmos3_edge.layer_ref import (  # noqa: E402
    EdgeLayer0TorchReference,
    EdgeTransformerFvkLinearReference,
    _attention,
)
from flash_rt.models.cosmos3_edge.weights import EdgeTransformerWeights  # noqa: E402


def _local_paths() -> tuple[Path, Path] | None:
    checkpoint_raw = os.environ.get("COSMOS3_EDGE_CHECKPOINT")
    boundary_raw = os.environ.get("COSMOS3_EDGE_BOUNDARY_DUMP")
    if not checkpoint_raw or not boundary_raw:
        return None
    paths = (Path(checkpoint_raw).expanduser(), Path(boundary_raw).expanduser())
    return paths if paths[0].exists() and paths[1].exists() else None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="layer0 reference requires CUDA for this shape")
def test_cosmos3_edge_layer0_torch_reference_matches_boundary_dump():
    paths = _local_paths()
    if paths is None:
        pytest.skip("local Cosmos3-Edge checkpoint/boundary dump is not available")
    checkpoint, dump_path = paths

    dump = EdgeBoundaryDump(dump_path)
    dump.validate_geometry()
    ref = EdgeLayer0TorchReference(EdgeTransformerWeights(checkpoint), device="cuda")

    with torch.no_grad():
        causal, full = ref.forward(dump)

    target_causal = dump.layer0_output_causal.to(device="cuda", dtype=torch.bfloat16)
    target_full = dump.layer0_output_full.to(device="cuda", dtype=torch.bfloat16)

    causal_cos = torch.nn.functional.cosine_similarity(causal.float().flatten(), target_causal.float().flatten(), dim=0)
    full_cos = torch.nn.functional.cosine_similarity(full.float().flatten(), target_full.float().flatten(), dim=0)

    assert causal_cos.item() > 0.9999
    assert full_cos.item() > 0.9999


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Thor native attention reference check requires CUDA")
def test_cosmos3_edge_native_attention_matches_reference_at_supported_shape():
    ref = EdgeTransformerFvkLinearReference(object(), device="cuda")
    if ref.ctx is None:
        pytest.skip("Thor attention_mha_bf16 is not available")

    generator = torch.Generator(device="cuda").manual_seed(1234)
    q = torch.randn(128, SPEC.num_heads, SPEC.head_dim, device="cuda", dtype=torch.bfloat16, generator=generator)
    k = torch.randn(256, SPEC.num_kv_heads, SPEC.head_dim, device="cuda", dtype=torch.bfloat16, generator=generator)
    v = torch.randn(256, SPEC.num_kv_heads, SPEC.head_dim, device="cuda", dtype=torch.bfloat16, generator=generator)

    with torch.no_grad():
        out = ref.attention(q, k, v, is_causal=False)
        target = _attention(q, k, v, is_causal=False)
    cos = torch.nn.functional.cosine_similarity(out.float().flatten(), target.float().flatten(), dim=0)
    rel_l2 = torch.linalg.vector_norm((out - target).float()) / torch.linalg.vector_norm(target.float())

    assert cos.item() > 0.9999
    assert rel_l2.item() < 0.01


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Thor native attention fallback check requires CUDA")
def test_cosmos3_edge_native_attention_matches_reference_at_small_shape():
    ref = EdgeTransformerFvkLinearReference(object(), device="cuda")
    if ref.ctx is None:
        pytest.skip("Thor attention_mha_bf16 is not available")
    generator = torch.Generator(device="cuda").manual_seed(5678)
    q = torch.randn(32, SPEC.num_heads, SPEC.head_dim, device="cuda", dtype=torch.bfloat16, generator=generator)
    k = torch.randn(48, SPEC.num_kv_heads, SPEC.head_dim, device="cuda", dtype=torch.bfloat16, generator=generator)
    v = torch.randn(48, SPEC.num_kv_heads, SPEC.head_dim, device="cuda", dtype=torch.bfloat16, generator=generator)

    with torch.no_grad():
        out = ref.attention(q, k, v, is_causal=False)
        target = _attention(q, k, v, is_causal=False)

    cos = torch.nn.functional.cosine_similarity(out.float().flatten(), target.float().flatten(), dim=0)
    rel_l2 = torch.linalg.vector_norm((out - target).float()) / torch.linalg.vector_norm(target.float())
    assert cos.item() > 0.9999
    assert rel_l2.item() < 0.01


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Thor FA4 fwd reference check requires CUDA")
def test_cosmos3_edge_fa4_fwd_attention_matches_reference(monkeypatch):
    monkeypatch.setenv("FLASHRT_COSMOS3_EDGE_FA4_FWD", "1")
    ref = EdgeTransformerFvkLinearReference(object(), device="cuda")
    if ref._get_fa4_fwd() is None:
        pytest.skip("Thor FA4 fwd path is not available")

    generator = torch.Generator(device="cuda").manual_seed(9012)
    q = torch.randn(64, SPEC.num_heads, SPEC.head_dim, device="cuda", dtype=torch.bfloat16, generator=generator)
    k = torch.randn(128, SPEC.num_kv_heads, SPEC.head_dim, device="cuda", dtype=torch.bfloat16, generator=generator)
    v = torch.randn(128, SPEC.num_kv_heads, SPEC.head_dim, device="cuda", dtype=torch.bfloat16, generator=generator)

    with torch.no_grad():
        out = ref.attention(q, k, v, is_causal=False)
        target = _attention(q, k, v, is_causal=False)

    cos = torch.nn.functional.cosine_similarity(out.float().flatten(), target.float().flatten(), dim=0)
    rel_l2 = torch.linalg.vector_norm((out - target).float()) / torch.linalg.vector_norm(target.float())
    assert cos.item() > 0.9999
    assert rel_l2.item() < 0.01

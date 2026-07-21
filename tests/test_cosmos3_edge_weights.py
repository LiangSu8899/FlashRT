import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from flash_rt.frontends.torch._cosmos3_edge_thor_spec import layer_key  # noqa: E402
from flash_rt.models.cosmos3_edge.weights import EdgeTransformerWeights  # noqa: E402


def _local_checkpoint() -> Path | None:
    raw = os.environ.get("COSMOS3_EDGE_CHECKPOINT")
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path if path.exists() else None


def test_cosmos3_edge_weight_loader_refs_expected_shard():
    checkpoint = _local_checkpoint()
    if checkpoint is None:
        pytest.skip("local Cosmos3-Edge checkpoint is not available")

    weights = EdgeTransformerWeights(checkpoint)
    ref = weights.ref(layer_key(0, "self_attn.to_q.weight"))

    assert ref.shape == (2048, 2048)
    assert ref.shard.name in {
        "diffusion_pytorch_model-00001-of-00002.safetensors",
        "diffusion_pytorch_model-00002-of-00002.safetensors",
    }


def test_cosmos3_edge_weight_loader_reads_selected_tensors():
    checkpoint = _local_checkpoint()
    if checkpoint is None:
        pytest.skip("local Cosmos3-Edge checkpoint is not available")

    weights = EdgeTransformerWeights(checkpoint)

    norm = weights.load_tensor("norm.weight")
    gen_q = weights.layer_tensor(0, "self_attn.to_q.weight", dtype=torch.bfloat16)
    und_k = weights.layer_tensor(0, "self_attn.add_k_proj.weight", dtype=torch.bfloat16)

    assert norm.shape == (2048,)
    assert gen_q.shape == (2048, 2048)
    assert und_k.shape == (1024, 2048)
    assert gen_q.dtype == torch.bfloat16

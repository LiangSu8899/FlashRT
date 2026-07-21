import os
from pathlib import Path

import pytest

from flash_rt.frontends.torch._cosmos3_edge_thor_spec import (
    GEN_ATTENTION_KEYS,
    SPEC,
    UND_ATTENTION_KEYS,
    iter_expected_shapes,
    layer_key,
    validate_transformer_index,
)


def test_cosmos3_edge_spec_counts_and_tower_names():
    shapes = dict(iter_expected_shapes())

    assert SPEC.num_layers == 28
    assert len(shapes) == 549
    assert shapes[layer_key(0, "self_attn.to_q.weight")] == (2048, 2048)
    assert shapes[layer_key(0, "self_attn.add_k_proj.weight")] == (1024, 2048)
    assert shapes[layer_key(27, "self_attn.k_norm_und_for_gen.weight")] == (128,)

    assert "self_attn.add_q_proj.weight" in GEN_ATTENTION_KEYS
    assert "self_attn.to_q.weight" in UND_ATTENTION_KEYS


def test_cosmos3_edge_local_checkpoint_index_matches_when_present():
    raw = os.environ.get("COSMOS3_EDGE_CHECKPOINT")
    checkpoint = Path(raw).expanduser() if raw else None
    if checkpoint is None or not checkpoint.exists():
        pytest.skip("local Cosmos3-Edge checkpoint is not available")
    validate_transformer_index(checkpoint)

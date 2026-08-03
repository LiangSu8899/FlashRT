"""Contract pins for the decode_loop family (CPU, no CUDA needed)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from flash_rt.structures.impls.decode_loop.whole_step import (
    _StaticHybridCache,
    _find_stack,
)


def test_static_hybrid_cache_serves_the_layer_surface():
    c = _StaticHybridCache(4, [1, 3], 2, 8, 32, "cpu")
    k = torch.randn(1, 2, 1, 8, dtype=torch.bfloat16)
    c._cp = torch.tensor([5])
    ko, vo = c.update(k, k, 1)
    assert ko.shape == (1, 2, 32, 8)
    assert torch.equal(ko[:, :, 5], k[:, :, 0])
    # untouched attention slots stay empty; gated-delta slots are plain
    assert c.key_cache[0] is None and c.conv_states[2] is None
    # decode is signalled by a filled conv slot, the host convention
    assert not c.has_previous_state
    c.conv_states[0] = torch.zeros(1)
    assert c.has_previous_state
    assert c.get_mask_sizes(1, 0) == (32, 0)
    assert c.get_seq_length() == 32


def test_stack_discovery_is_by_slots_not_names():
    class _LM(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList()
            self.embed_tokens = nn.Embedding(8, 4)
            self.norm = nn.LayerNorm(4)
            self.rotary_emb = nn.Identity()

    class _Host(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _LM()

    lm = _find_stack(_Host())
    assert hasattr(lm, "rotary_emb")

    class _Wrapped(nn.Module):
        def __init__(self):
            super().__init__()
            inner = nn.Module()
            inner.language_model = _LM()
            self.model = inner

    assert hasattr(_find_stack(_Wrapped()), "embed_tokens")

    with pytest.raises(ValueError, match="refused"):
        _find_stack(nn.Linear(4, 4))


def test_mtp_and_release_arms_are_scheme_decisions():
    from flash_rt.structures import schemes

    assert schemes.QuantScheme.mtp_projection_format is None
    assert schemes.QuantScheme.gdn_projection_format is None
    base = schemes.get("w4a4_decode")
    rel = schemes.get("w4a4_decode_release")
    assert not getattr(base, "gdn_release_host_weights", False)
    assert rel.gdn_release_host_weights is True
    assert rel.gdn_projection_format == "nvfp4_dynamic"
    # the release arm is its own registered name, never a mutation of
    # the default arm
    assert base is not rel


def test_decode_loop_door_is_exported():
    from flash_rt import structures

    assert callable(structures.decode_loop)
    assert "decode_loop" in structures.__all__

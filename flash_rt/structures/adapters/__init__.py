"""Host-family adapters — where a structure's seam is host-specific.

Importing this package registers the built-in adapters with autobuild.
Attention seams (attention_core) live here because where the attention
math runs differs by host family; a static module pattern cannot find
them, so each family gets a small adapter.
"""
from ..autobuild import (
    register_attention_adapter,
    register_qk_norm_rope_adapter,
)
from .diffusers_attention import DiffusersAttentionAdapter
from .factored_two_way_attention import FactoredTwoWayAttentionAdapter
from .gemma_attention import GemmaAttentionAdapter
from .qwen_per_head_qk_norm_rope import QwenPerHeadQkNormRopeAdapter

register_qk_norm_rope_adapter(QwenPerHeadQkNormRopeAdapter())
register_attention_adapter(GemmaAttentionAdapter())
register_attention_adapter(FactoredTwoWayAttentionAdapter())
register_attention_adapter(DiffusersAttentionAdapter())

__all__ = [
    "DiffusersAttentionAdapter",
    "GemmaAttentionAdapter",
    "FactoredTwoWayAttentionAdapter",
    "QwenPerHeadQkNormRopeAdapter",
]

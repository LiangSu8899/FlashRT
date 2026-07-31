import torch
from torch import nn

from flash_rt.structures.autobuild import _layer_of, _seam_key
from flash_rt.structures.discover import discover, seam_weights


class _GatedMlp(nn.Module):
    def __init__(self, *, bias: bool):
        super().__init__()
        self.gate_proj = nn.Linear(8, 16, bias=bias)
        self.up_proj = nn.Linear(8, 16, bias=bias)
        self.down_proj = nn.Linear(16, 8, bias=bias)
        self.act_fn = torch.nn.functional.silu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x))
                              * self.up_proj(x))


class _Host(nn.Module):
    def __init__(self, *, bias: bool):
        super().__init__()
        self.mlp = _GatedMlp(bias=bias)


class _DualPathAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(512, 512, bias=False)
        self.to_k = nn.Linear(512, 128, bias=False)
        self.to_v = nn.Linear(512, 128, bias=False)
        self.to_out = nn.Linear(512, 512, bias=False)
        self.add_q_proj = nn.Linear(512, 512, bias=False)
        self.add_k_proj = nn.Linear(512, 128, bias=False)
        self.add_v_proj = nn.Linear(512, 128, bias=False)
        self.to_add_out = nn.Linear(512, 512, bias=False)


class _ConditionalNorm(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.linear = nn.Linear(dim, 2 * dim)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)

    def forward(self, x, temb=None):
        scale, shift = self.linear(temb).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None]


class _DenseGelu(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        first = nn.Module()
        first.proj = nn.Linear(dim, 4 * dim)
        self.net = nn.ModuleList([first, nn.GELU(), nn.Linear(4 * dim, dim)])


class _DiffusionBlock(nn.Module):
    def __init__(self, *, positional=False, cross=False):
        super().__init__()
        self.norm1 = _ConditionalNorm()
        self.norm3 = nn.LayerNorm(512, elementwise_affine=False)
        self.attn1 = nn.Module()
        self.attn1.to_q = nn.Linear(512, 512)
        kv_dim = 768 if cross else 512
        self.attn1.to_k = nn.Linear(kv_dim, 512)
        self.attn1.to_v = nn.Linear(kv_dim, 512)
        self.attn1.to_out = nn.ModuleList([nn.Linear(512, 512)])
        self.ff = _DenseGelu()
        self.pos_embed = nn.Identity() if positional else None


def test_decoder_ffn_discovery_accepts_its_bias_free_weight_contract():
    seams = discover(_Host(bias=False), structures=("decoder_ffn",))

    assert [seam.path for seam in seams] == ["mlp"]


def test_decoder_ffn_discovery_refuses_unrepresented_bias_weights():
    seams = discover(_Host(bias=True), structures=("decoder_ffn",))

    assert seams == []


def test_dual_path_attention_discovers_both_independent_qkv_groups():
    seams = discover(
        nn.ModuleDict({"attention": _DualPathAttention()}),
        structures=("qkv_pack",),
    )

    assert [seam.pack_attrs for seam in seams] == [
        ("to_q", "to_k", "to_v"),
        ("add_q_proj", "add_k_proj", "add_v_proj"),
    ]
    assert [_seam_key(seam) for seam in seams] == [
        "attention.to_q",
        "attention.add_q_proj",
    ]


def test_dual_path_attention_discovers_profitable_projections_on_both_paths():
    seams = discover(
        nn.ModuleDict({"attention": _DualPathAttention()}),
        structures=("linear_proj",),
    )

    assert [seam.proj_attr for seam in seams] == [
        "to_q",
        "to_out",
        "add_q_proj",
        "to_add_out",
    ]


def test_modnorm_qkv_chain_discovers_by_direct_dataflow_slots():
    host = nn.ModuleDict({"block": _DiffusionBlock()})

    seams = discover(host, structures=("modnorm_qkv_chain",))

    assert [seam.path for seam in seams] == ["block"]
    assert seams[0].dims == {"D": 512, "C": 512}
    assert seams[0].variant["fanout"] == "qkv"


def test_modnorm_qkv_chain_refuses_an_intervening_positional_module():
    host = nn.ModuleDict({"block": _DiffusionBlock(positional=True)})

    assert discover(host, structures=("modnorm_qkv_chain",)) == []


def test_nested_diffusers_feedforward_is_a_vision_ffn_slice():
    host = nn.ModuleDict({"block": _DiffusionBlock()})

    seams = discover(host, structures=("vision_ffn",))

    assert [seam.path for seam in seams] == ["block.ff"]
    assert seams[0].fc_attrs == ("net.0.proj", "net.2")
    assert seams[0].norm_attr == "norm3"
    assert seams[0].variant["norm_affine"] == "identity"
    weights = seam_weights(host, seams[0])
    assert weights["w_norm"] is None
    assert weights["b_norm"] is None


def test_vision_ffn_refuses_rms_like_one_sided_affine_norm():
    host = nn.ModuleDict({"block": _DiffusionBlock()})
    host.block.norm3.weight = nn.Parameter(torch.ones(512))
    host.block.norm3.bias = None
    refused = []

    seams = discover(host, structures=("vision_ffn",), refused=refused)

    assert seams == []
    assert refused and "one-sided affine" in refused[0][1]


def test_cross_attention_chain_owns_only_the_query_wire():
    host = nn.ModuleDict({"block": _DiffusionBlock(cross=True)})

    chain = discover(host, structures=("modnorm_qkv_chain",))
    packs = discover(host, structures=("qkv_pack",))
    projections = discover(host, structures=("linear_proj",))

    assert chain[0].variant["fanout"] == "q_only"
    assert packs == []
    assert [seam.path for seam in projections] == [
        "block.attn1.to_q",
        "block.attn1.to_k",
        "block.attn1.to_v",
        "block.attn1.to_out.0",
    ]


def test_transformer_block_layer_key_works_at_root_and_nested_paths():
    assert _layer_of("transformer_blocks.1.attn1.to_q") == (
        "transformer_blocks.1")
    assert _layer_of("head.model.transformer_blocks.1.norm1") == (
        "head.model.transformer_blocks.1")

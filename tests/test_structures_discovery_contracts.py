import torch
from torch import nn

from flash_rt.structures.autobuild import _seam_key
from flash_rt.structures.discover import discover


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

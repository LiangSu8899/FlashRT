import torch
from torch import nn

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


def test_decoder_ffn_discovery_accepts_its_bias_free_weight_contract():
    seams = discover(_Host(bias=False), structures=("decoder_ffn",))

    assert [seam.path for seam in seams] == ["mlp"]


def test_decoder_ffn_discovery_refuses_unrepresented_bias_weights():
    seams = discover(_Host(bias=True), structures=("decoder_ffn",))

    assert seams == []

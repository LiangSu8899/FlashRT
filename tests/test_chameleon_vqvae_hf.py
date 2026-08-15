"""Tests for the Apache-2.0 Transformers Chameleon VQ-VAE adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
pytest.importorskip("transformers")

from safetensors.torch import save_file
from transformers import ChameleonConfig, ChameleonVQVAE
from transformers.models.chameleon.configuration_chameleon import (
    ChameleonVQVAEConfig,
)

from flash_rt.models.chameleon.vqvae_hf import (
    encode_vqvae_tokens,
    load_chameleon_vqvae,
    vqvae_checkpoint_digest,
)


def test_loads_only_prefixed_vqvae_weights_strictly(tmp_path):
    vq_config = ChameleonVQVAEConfig(
        embed_dim=32,
        num_embeddings=16,
        latent_channels=32,
        resolution=16,
        base_channels=32,
        channel_multiplier=[1],
        num_res_blocks=1,
        attn_resolutions=[],
    )
    config = ChameleonConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        vq_config=vq_config.to_dict(),
        vocabulary_map={},
    )
    config.save_pretrained(tmp_path)
    reference = ChameleonVQVAE(vq_config)
    state = {
        f"model.vqmodel.{key}": value
        for key, value in reference.state_dict().items()
    }
    save_file(state, tmp_path / "model.safetensors")

    loaded, mapping = load_chameleon_vqvae(
        tmp_path, device="cpu", dtype=torch.float32)

    assert type(mapping).__name__ == "ChameleonImageVocabularyMapping"
    for key, value in reference.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key])
    assert vqvae_checkpoint_digest(tmp_path) == vqvae_checkpoint_digest(tmp_path)


def test_no_research_license_vqgan_is_vendored():
    root = Path(__file__).resolve().parents[1]
    vendored = root / "flash_rt/models/chameleon/vqgan"
    assert not vendored.exists() or not any(vendored.rglob("*"))


def test_encode_tokens_supports_old_and_new_transformers_outputs():
    expected = torch.tensor([1, 2, 3])

    class OldModel:
        def encode(self, _):
            return None, None, expected

    class NewOutput:
        image_tokens = expected

    class NewModel:
        def encode(self, _):
            return NewOutput()

    assert encode_vqvae_tokens(OldModel(), None) is expected
    assert encode_vqvae_tokens(NewModel(), None) is expected

"""FlashVLA — Model and quantization configuration."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

CONFIGS_DIR = Path(__file__).parent.parent.parent / "configs"


@dataclass
class VisionConfig:
    patch_size: int = 14
    image_size: int = 224
    num_patches: int = 256
    hidden_dim: int = 1152
    num_layers: int = 27


@dataclass
class QuantConfig:
    enabled: bool = True
    dtype: str = "fp8"          # global default: "fp8", "fp4", "bf16"
    vision: str = "fp8"
    encoder: str = "fp8"
    decoder: str = "fp8"
    action_head: str = "bf16"   # precision-sensitive, no quant by default
    calibration: str = "runtime"  # "runtime" or "offline"

    def get_dtype(self, stage: str) -> str:
        """Get quantization dtype for a given pipeline stage."""
        if not self.enabled:
            return "bf16"
        return getattr(self, stage, self.dtype)


@dataclass
class ModelConfig:
    name: str = "pi05"
    vision_type: str = "siglip"
    vlm_hidden: int = 2048
    expert_hidden: int = 768
    vlm_layers: int = 18
    expert_layers: int = 18
    decoder_steps: int = 10
    action_dim: int = 32
    chunk_size: int = 10

    # Attention
    num_heads: int = 16
    num_kv_heads: int = 16
    head_dim: int = 128
    rope_type: str = "standard"  # "standard" or "mrope"

    # FFN
    activation: str = "gelu"     # "gelu" or "silu"
    intermediate_size: int = 8192

    # Sub-configs
    vision: VisionConfig = field(default_factory=VisionConfig)
    quant: QuantConfig = field(default_factory=QuantConfig)


def load_config(name_or_path: str) -> ModelConfig:
    """Load model config from name or YAML path.

    Args:
        name_or_path: Built-in name ("pi05", "pi0") or path to .yaml file.
    """
    path = Path(name_or_path)
    if not path.exists():
        path = CONFIGS_DIR / f"{name_or_path}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {name_or_path} (searched {path})")

    with open(path) as f:
        data = yaml.safe_load(f)

    cfg = ModelConfig()
    vision_data = data.pop("vision", {})
    quant_data = data.pop("quant", {})

    for k, v in data.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    for k, v in vision_data.items():
        if hasattr(cfg.vision, k):
            setattr(cfg.vision, k, v)

    for k, v in quant_data.items():
        if hasattr(cfg.quant, k):
            setattr(cfg.quant, k, v)

    return cfg

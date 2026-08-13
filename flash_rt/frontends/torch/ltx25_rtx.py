"""FlashRT -- LTX-2.5 22B distilled (audio+video) torch frontend for RTX SM120.

Wraps the official ``ltx-pipelines`` two-stage distilled pipeline behind
FlashRT's ``set_prompt`` / ``infer`` surface and installs FlashRT compute
swaps (attention backends now; fused NVFP4 epilogues and CUDA graph capture
in later stages).

Scope:
    * Official LTX-2.5 split-pack checkpoints (one safetensors per component).
    * NVFP4 prequantized transformer by default (static activation scales ship
      in the checkpoint -- no calibration pass).
    * RTX SM120 registration only. No CMake or pybind changes.

The LTX-2 monorepo is located through ``FLASH_RT_LTX2_ROOT`` (checkout root;
``packages/*/src`` are added to ``sys.path``) unless ``ltx_pipelines`` is
already importable in the environment.
"""

from __future__ import annotations

import logging
import os
import pathlib
import sys
import time
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)

_PACKAGES = ("ltx-core", "ltx-pipelines", "ltx-kernels")


class Ltx25TorchFrontendRtx:
    """LTX-2.5 distilled two-stage pipeline frontend for RTX SM120."""

    DEFAULT_WIDTH = 1536
    DEFAULT_HEIGHT = 1024
    DEFAULT_FRAMES = 121
    DEFAULT_FPS = 24.0

    def __init__(
        self,
        checkpoint_dir: str,
        num_views: int = 1,
        attention: Optional[str] = None,
        quantization: str = "nvfp4-prequant",
        dtype: torch.dtype = torch.bfloat16,
        **_: Any,
    ) -> None:
        self.checkpoint_dir = pathlib.Path(checkpoint_dir).expanduser()
        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(
                f"LTX-2.5 checkpoint pack not found: {self.checkpoint_dir}")
        self.num_views = num_views
        self.dtype = dtype
        self.quantization = quantization
        self.attention = attention or os.environ.get(
            "FLASH_RT_LTX25_ATTN", "auto")
        self.device = torch.device("cuda")
        self.prompt: Optional[str] = None
        self._pipe = None
        self._attn_label: Optional[str] = None
        self._load_seconds: Optional[float] = None
        self._last_stats: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Official package discovery
    # ------------------------------------------------------------------
    @staticmethod
    def _candidate_roots() -> list[pathlib.Path]:
        roots: list[pathlib.Path] = []
        for key in ("FLASH_RT_LTX2_ROOT", "LTX2_ROOT"):
            value = os.environ.get(key)
            if value:
                roots.append(pathlib.Path(value).expanduser())
        return roots

    @classmethod
    def _ensure_ltx_importable(cls) -> None:
        try:
            import ltx_pipelines  # noqa: F401
            return
        except ModuleNotFoundError as exc:
            if exc.name not in ("ltx_pipelines", "ltx_core"):
                raise

        for root in cls._candidate_roots():
            added = []
            for pkg in _PACKAGES:
                src = root / "packages" / pkg / "src"
                if src.is_dir() and str(src) not in sys.path:
                    sys.path.insert(0, str(src))
                    added.append(str(src))
            try:
                import ltx_pipelines  # noqa: F401
                return
            except ModuleNotFoundError as exc:
                if exc.name not in ("ltx_pipelines", "ltx_core"):
                    raise
                for p in added:
                    sys.path.remove(p)
                continue

        raise ModuleNotFoundError(
            "Cannot import the official LTX-2 packages. Install "
            "ltx-pipelines into the environment, or set FLASH_RT_LTX2_ROOT "
            "to an LTX-2 monorepo checkout (the directory containing "
            "packages/ltx-core and packages/ltx-pipelines).")

    # ------------------------------------------------------------------
    # Checkpoint pack resolution
    # ------------------------------------------------------------------
    def _find_one(self, subdir: str, patterns: list[str],
                  required: bool = True) -> Optional[str]:
        base = self.checkpoint_dir / subdir
        for pattern in patterns:
            hits = sorted(base.glob(pattern))
            if hits:
                return str(hits[0])
        if required:
            raise FileNotFoundError(
                f"No file matching {patterns} under {base}. The frontend "
                "expects the official LTX-2.5 split pack layout.")
        return None

    def _resolve_paths(self) -> dict[str, str]:
        transformer = self._find_one(
            "diffusion_models",
            ["*distilled-transformer-nvfp4.safetensors",
             "*distilled-transformer-bf16.safetensors"])
        text_encoder = self._find_one(
            "text_encoders", ["*with-proj*bf16.safetensors"])
        video_vae = self._find_one("vae", ["*video-vae-bf16.safetensors"])
        audio_vae = self._find_one("vae", ["*audio-vae-bf16.safetensors"])
        duration_head = self._find_one(
            "model_patches", ["*duration-head*.safetensors"], required=False)
        spatial_upsampler = self._find_one(
            "latent_upscale_models",
            ["*latent-spatial-upscaler-x2*.safetensors",
             "*spatial-upscaler*.safetensors"])
        return {
            "transformer": transformer,
            "text_encoder": text_encoder,
            "video_vae": video_vae,
            "audio_vae": audio_vae,
            "duration_head": duration_head,
            "spatial_upsampler": spatial_upsampler,
        }

    # ------------------------------------------------------------------
    # Pipeline assembly
    # ------------------------------------------------------------------
    def _load_pipe(self):
        if self._pipe is not None:
            return self._pipe

        self._ensure_ltx_importable()
        from ltx_pipelines.distilled import DistilledPipeline
        from ltx_pipelines.utils.model_paths import ModelPaths
        from ltx_pipelines.utils.quantization_factory import QuantizationKind

        paths = self._resolve_paths()
        quant = None
        if self.quantization and paths["transformer"].endswith(
                "nvfp4.safetensors"):
            quant = QuantizationKind(self.quantization).to_policy(
                paths["transformer"])

        t0 = time.perf_counter()
        pipe = DistilledPipeline(
            model_paths=ModelPaths.from_split(
                transformer_path=paths["transformer"],
                text_encoder_path=paths["text_encoder"],
                video_vae_path=paths["video_vae"],
                audio_vae_path=paths["audio_vae"],
                duration_head_path=paths["duration_head"],
            ),
            spatial_upsampler_path=paths["spatial_upsampler"],
            loras=[],
            quantization=quant,
        )

        from flash_rt.models.ltx25._attn_swap import make_ltx25_attention
        attn = make_ltx25_attention(self.attention)
        if attn is not None and getattr(attn, "label", "") != "sdpa":
            pipe.stage = pipe.stage.with_attention(attn)
        self._attn_label = getattr(attn, "label", str(self.attention))
        self._load_seconds = time.perf_counter() - t0
        logger.info("[ltx25] pipeline ready in %.1fs (attention=%s)",
                    self._load_seconds, self._attn_label)
        self._pipe = pipe
        return pipe

    # ------------------------------------------------------------------
    # FlashRT surface
    # ------------------------------------------------------------------
    def set_prompt(self, prompt: str, **_: Any) -> None:
        self.prompt = prompt
        self._load_pipe()

    @torch.inference_mode()
    def infer(
        self,
        prompt: Optional[str] = None,
        seed: int = 42,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        frame_rate: Optional[float] = None,
        output_path: Optional[str] = None,
        **_: Any,
    ) -> dict[str, Any]:
        prompt = prompt or self.prompt
        if not prompt:
            raise ValueError("No prompt: call set_prompt() or pass prompt=")
        pipe = self._load_pipe()

        from ltx_core.model.video_vae import AUTO_TILING, get_video_chunks_number

        height = height or self.DEFAULT_HEIGHT
        width = width or self.DEFAULT_WIDTH
        num_frames = num_frames or self.DEFAULT_FRAMES
        frame_rate = frame_rate or self.DEFAULT_FPS

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        video, audio, frames, tiling = pipe(
            prompt=prompt, seed=seed, height=height, width=width,
            num_frames=num_frames, frame_rate=frame_rate,
            images=[], tiling_config=AUTO_TILING,
        )
        torch.cuda.synchronize()
        denoise_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        if output_path:
            from ltx_pipelines.utils.media_io import encode_video
            encode_video(
                video=video, fps=frame_rate, audio=audio,
                output_path=output_path,
                video_chunks_number=get_video_chunks_number(frames, tiling))
        else:
            for _chunk in video:
                pass
        torch.cuda.synchronize()
        decode_s = time.perf_counter() - t1

        self._last_stats = {
            "attention": self._attn_label,
            "quantization": self.quantization,
            "resolution": f"{width}x{height}x{num_frames}",
            "denoise_and_prep_s": round(denoise_s, 3),
            "vae_decode_encode_s": round(decode_s, 3),
            "total_s": round(denoise_s + decode_s, 3),
            "peak_mem_gb": round(
                torch.cuda.max_memory_allocated() / 2 ** 30, 2),
            "output_path": output_path,
        }
        return dict(self._last_stats)

    def get_latency_stats(self) -> dict[str, Any]:
        return dict(self._last_stats)

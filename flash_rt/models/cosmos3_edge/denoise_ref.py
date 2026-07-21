"""Torch reference denoise loop for Cosmos3-Edge AV inverse dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from flash_rt.models.cosmos3_edge.boundary_dump import EdgeBoundaryDump
from flash_rt.models.cosmos3_edge.dump_replay import (
    EDGE_ACTION_MODEL_SHAPE,
    EDGE_NUM_TRAIN_TIMESTEPS,
    EDGE_SHIFT,
    EdgeDenoiseDump,
    EdgeLatentParts,
)
from flash_rt.models.cosmos3_edge.layer_ref import EdgeTransformerTorchReference
from flash_rt.models.cosmos3_edge.static_engine import EdgeStaticBufferEngine
from flash_rt.models.cosmos3_edge.static_unipc import EdgeStaticUniPCScheduler
from flash_rt.models.cosmos3_edge.weights import EdgeTransformerWeights
from flash_rt.models.cosmos3_video.fm_solvers_unipc import FlowUniPCMultistepScheduler


@dataclass(frozen=True)
class ReferenceDenoiseResult:
    final_flat: torch.Tensor
    final_parts: EdgeLatentParts
    final_action: torch.Tensor
    timesteps: tuple[int, ...]
    max_input_abs_diff: float
    max_velocity_abs_diff: float
    steps_run: int


class EdgeDenoiseTorchReference:
    """Full denoise loop that computes action velocity with the Torch reference.

    This is still a correctness scaffold, not the optimized FlashRT engine. For
    AV inverse dynamics the official dump predicts zero velocity for the vision
    segment, so only the action tail is produced by the transformer reference.
    """

    def __init__(
        self,
        denoise_dump: EdgeDenoiseDump,
        boundary_dump: EdgeBoundaryDump,
        weights: EdgeTransformerWeights,
        *,
        device: str | torch.device = "cuda",
        shift: float = EDGE_SHIFT,
        use_static_und_cache: bool = True,
        use_cuda_graphs: bool = False,
    ):
        self.denoise_dump = denoise_dump
        self.boundary_dump = boundary_dump
        self.device = torch.device(device)
        self.shift = float(shift)
        self.use_cuda_graphs = bool(use_cuda_graphs)
        self.transformer = EdgeTransformerTorchReference(weights, device=self.device)
        self.static_engine = None
        self.denoise_dump.validate_geometry()
        self.boundary_dump.validate_geometry()
        self.und_cache = None
        self.static_scheduler = None
        if use_static_und_cache:
            self.static_engine = EdgeStaticBufferEngine(weights, self.boundary_dump, device=self.device)
            self.und_cache = self.static_engine.und_cache

    def _scheduler(self) -> FlowUniPCMultistepScheduler:
        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=EDGE_NUM_TRAIN_TIMESTEPS,
            shift=1.0,
            use_dynamic_shifting=False,
        )
        scheduler.set_timesteps(self.denoise_dump.num_steps, device=self.device, shift=self.shift)
        expected = tuple(int(t.item()) for t in scheduler.timesteps.cpu())
        actual = self.denoise_dump.timesteps()
        if expected != actual:
            raise ValueError(f"dump timesteps {actual} do not match UniPC schedule {expected}")
        return scheduler

    @staticmethod
    def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
        return float((a.float() - b.float()).abs().max().item())

    def velocity_for_step(
        self,
        flat_noise: torch.Tensor,
        timestep: torch.Tensor,
        *,
        step_index: int | None = None,
    ) -> torch.Tensor:
        latent = flat_noise.to(device=self.device)
        if self.static_engine is not None:
            if self.use_cuda_graphs:
                return self.static_engine.replay_velocity_graph(latent, timestep)
            return self.static_engine.flat_velocity_for_step(latent, timestep, timestep_index=step_index)

        velocity = torch.zeros_like(latent)
        if self.und_cache is None:
            action_velocity = self.transformer.action_velocity_for_step(self.boundary_dump, latent, timestep)
        else:
            action_velocity = self.transformer.action_velocity_for_step_with_und_cache(
                self.boundary_dump,
                latent,
                timestep,
                self.und_cache,
            )
        action_flat = action_velocity.reshape(EDGE_ACTION_MODEL_SHAPE[0] * EDGE_ACTION_MODEL_SHAPE[1])
        velocity[-action_flat.numel() :] = action_flat.to(dtype=velocity.dtype)
        return velocity

    def run(self, *, max_steps: int | None = None, check_against_dump: bool = True) -> ReferenceDenoiseResult:
        scheduler = self._scheduler()
        steps_run = self.denoise_dump.num_steps if max_steps is None else min(max_steps, self.denoise_dump.num_steps)
        latent = self.denoise_dump.step_noise(0).to(device=self.device)
        input_diffs: list[float] = []
        velocity_diffs: list[float] = []
        static_scheduler = self.static_scheduler
        if static_scheduler is not None:
            static_scheduler.reset(latent)

        for step, timestep in enumerate(scheduler.timesteps[:steps_run]):
            if check_against_dump:
                expected_noise = self.denoise_dump.step_noise(step).to(device=self.device)
                input_diffs.append(self._max_abs_diff(latent, expected_noise))
            velocity = self.velocity_for_step(latent, timestep.reshape(1, 1), step_index=step)
            if check_against_dump:
                expected_velocity = self.denoise_dump.step_velocity(step).to(device=self.device)
                velocity_diffs.append(self._max_abs_diff(velocity, expected_velocity))
            if static_scheduler is not None:
                latent = static_scheduler.step(latent, velocity, step)
            else:
                latent = scheduler.step(
                    model_output=velocity,
                    timestep=timestep,
                    sample=latent.unsqueeze(0),
                    return_dict=False,
                )[0].squeeze(0)

        final_flat = latent.detach().cpu().contiguous()
        parts = self.denoise_dump.split_flat(final_flat)
        raw_action_dim = int(self.boundary_dump.tensors["s00/vfm_in/action/raw_action_dim/0"].item())
        final_action = parts.action_model[:, :raw_action_dim].float().contiguous()
        return ReferenceDenoiseResult(
            final_flat=final_flat,
            final_parts=parts,
            final_action=final_action,
            timesteps=tuple(int(t.item()) for t in scheduler.timesteps[:steps_run].cpu()),
            max_input_abs_diff=max(input_diffs) if input_diffs else 0.0,
            max_velocity_abs_diff=max(velocity_diffs) if velocity_diffs else 0.0,
            steps_run=steps_run,
        )


class EdgeDenoiseFlashRTQuant:
    """Quantized static-graph FlashRT denoise runner (Thor).

    Wraps ``CosmosEdgeThor``: pre-quantized gen-tower weights, static und K/V,
    FA4 attention, native UniPC, and (by default) the entire 30-step denoise
    captured in a single CUDA graph. The eager path is kept for warmup and
    per-step dump comparisons.
    """

    def __init__(
        self,
        denoise_dump: EdgeDenoiseDump,
        boundary_dump: EdgeBoundaryDump,
        weights: EdgeTransformerWeights,
        *,
        device: str | torch.device = "cuda",
        shift: float = EDGE_SHIFT,
        quant: str = "fp8",
        bf16_projs: tuple[str, ...] = (),
        ffn_fp4: bool = False,
        slim_last: bool = True,
        use_cuda_graphs: bool = True,
    ):
        from flash_rt.models.cosmos3_edge.pipeline_thor import CosmosEdgeThor

        if torch.device(device).type != "cuda":
            raise RuntimeError("EdgeDenoiseFlashRTQuant requires CUDA")
        self.denoise_dump = denoise_dump
        self.boundary_dump = boundary_dump
        self.device = torch.device(device)
        self.use_cuda_graphs = bool(use_cuda_graphs)
        self.denoise_dump.validate_geometry()
        self.boundary_dump.validate_geometry()
        self.engine = CosmosEdgeThor(
            weights,
            boundary_dump,
            self.denoise_dump.timesteps(),
            quant=quant,
            bf16_projs=bf16_projs,
            ffn_fp4=ffn_fp4,
            slim_last=slim_last,
            shift=shift,
        )
        self.static_engine = None
        self.static_scheduler = self.engine.unipc
        self.engine.calibrate(self.denoise_dump.step_noise(0))
        if self.use_cuda_graphs:
            self.engine.capture(warmup_noise=self.denoise_dump.step_noise(0))

    @property
    def native_attention_available(self) -> bool:
        return True

    @property
    def graph_attention_available(self) -> bool:
        return self.engine.graph is not None

    @property
    def native_scheduler_available(self) -> bool:
        return True

    def velocity_for_step(
        self,
        flat_noise: torch.Tensor,
        timestep: torch.Tensor,
        *,
        step_index: int | None = None,
    ) -> torch.Tensor:
        if step_index is None:
            raise ValueError("quant engine requires an explicit step index")
        del timestep  # baked into the per-step embed table
        if flat_noise.data_ptr() != self.engine.latent.data_ptr():
            self.engine.latent.copy_(flat_noise.to(device=self.device, dtype=torch.float32))
        return self.engine.forward_step(step_index, self.engine.latent)

    def _result(self, steps_run: int, input_diffs: list[float], velocity_diffs: list[float]) -> ReferenceDenoiseResult:
        final_flat = self.engine.latent.detach().cpu().contiguous()
        parts = self.denoise_dump.split_flat(final_flat)
        final_action = parts.action_model[:, : self.engine.raw_action_dim].float().contiguous()
        return ReferenceDenoiseResult(
            final_flat=final_flat,
            final_parts=parts,
            final_action=final_action,
            timesteps=self.denoise_dump.timesteps()[:steps_run],
            max_input_abs_diff=max(input_diffs) if input_diffs else 0.0,
            max_velocity_abs_diff=max(velocity_diffs) if velocity_diffs else 0.0,
            steps_run=steps_run,
        )

    def run(self, *, max_steps: int | None = None, check_against_dump: bool = True) -> ReferenceDenoiseResult:
        num_steps = self.denoise_dump.num_steps
        steps_run = num_steps if max_steps is None else min(max_steps, num_steps)
        noise0 = self.denoise_dump.step_noise(0)
        if self.engine.graph is not None and steps_run == num_steps:
            # Whole-denoise graph replay: per-step dump checks are not
            # observable inside the graph; the final-action gate is authoritative.
            self.engine.denoise(noise0)
            return self._result(steps_run, [], [])
        self.engine.latent.copy_(noise0.to(device=self.device, dtype=torch.float32))
        input_diffs: list[float] = []
        velocity_diffs: list[float] = []
        for step in range(steps_run):
            if check_against_dump:
                expected = self.denoise_dump.step_noise(step).to(device=self.device)
                input_diffs.append(float((self.engine.latent.float() - expected.float()).abs().max().item()))
            velocity = self.engine.forward_step(step, self.engine.latent)
            if check_against_dump:
                expected_v = self.denoise_dump.step_velocity(step).to(device=self.device)
                velocity_diffs.append(float((velocity.float() - expected_v.float()).abs().max().item()))
            self.engine.unipc.step(self.engine.latent, velocity, step)
        return self._result(steps_run, input_diffs, velocity_diffs)


class EdgeDenoiseFlashRT(EdgeDenoiseTorchReference):
    """Optimized eager FlashRT denoise engine for Cosmos3-Edge AV.

    The scheduler and correctness accounting intentionally stay shared with
    ``EdgeDenoiseTorchReference``. The model velocity path is the FlashRT static
    engine: static vision tokens, cached und K/V, FVK BF16 GEMMs/RMSNorm/RoPE,
    and native Thor attention when available.
    """

    def __init__(
        self,
        denoise_dump: EdgeDenoiseDump,
        boundary_dump: EdgeBoundaryDump,
        weights: EdgeTransformerWeights,
        *,
        device: str | torch.device = "cuda",
        shift: float = EDGE_SHIFT,
        use_cuda_graphs: bool = False,
    ):
        super().__init__(
            denoise_dump,
            boundary_dump,
            weights,
            device=device,
            shift=shift,
            use_static_und_cache=True,
            use_cuda_graphs=use_cuda_graphs,
        )
        if self.static_engine is None:
            raise RuntimeError("EdgeDenoiseFlashRT requires the static denoise engine")
        self.static_engine.precompute_timestep_embeds(self.denoise_dump.timesteps())
        self.static_scheduler = EdgeStaticUniPCScheduler(
            self.denoise_dump.num_steps,
            device=self.device,
            shift=self.shift,
        )

    @property
    def native_attention_available(self) -> bool:
        return bool(
            self.static_engine is not None
            and getattr(self.static_engine.reference, "native_attention_available", False)
        )

    @property
    def graph_attention_available(self) -> bool:
        return bool(
            self.static_engine is not None
            and getattr(self.static_engine.reference, "graph_attention_available", False)
        )

    @property
    def native_scheduler_available(self) -> bool:
        return bool(self.static_scheduler is not None and self.static_scheduler.native_available)

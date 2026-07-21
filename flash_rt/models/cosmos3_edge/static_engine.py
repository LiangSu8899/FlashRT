"""Static-shape Cosmos3-Edge denoise engine scaffold.

This module is the bridge between the math reference and the native Thor
kernel engine. It owns the fixed AV inverse-dynamics geometry, static vision
conditioning, action token slots, and cached und/text K/V. The individual math
calls still delegate to the Torch reference; later P1 steps replace those calls
with fvk/FA2/FP4 kernels behind this stable boundary.
"""

from __future__ import annotations

import torch

from flash_rt.frontends.torch._cosmos3_edge_thor_spec import SPEC
from flash_rt.models.cosmos3_edge.boundary_dump import EdgeBoundaryDump
from flash_rt.models.cosmos3_edge.dump_replay import EDGE_ACTION_MODEL_SHAPE, EDGE_FLAT_DIM
from flash_rt.models.cosmos3_edge.layer_ref import (
    EdgeTransformerFvkLinearReference,
    EdgeTransformerTorchReference,
    EdgeUndKVCache,
)
from flash_rt.models.cosmos3_edge.weights import EdgeTransformerWeights


class EdgeStaticBufferEngine:
    """Static-buffer, static-und-cache Edge AV denoise engine.

    The public methods expose the contract needed by the future optimized
    backend: feed a flat denoise latent plus timestep, receive a flat velocity.
    The vision velocity segment is zero for the current AV inverse-dynamics
    workload; only the action tail is computed.
    """

    def __init__(
        self,
        weights: EdgeTransformerWeights,
        boundary_dump: EdgeBoundaryDump,
        *,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.weights = weights
        self.boundary_dump = boundary_dump
        self.device = torch.device(device)
        self.dtype = dtype
        self.reference = EdgeTransformerFvkLinearReference(weights, device=self.device, dtype=self.dtype)
        if self.reference.gemm is None:
            self.reference = EdgeTransformerTorchReference(weights, device=self.device, dtype=self.dtype)
        self.gemm = None
        self.fill_flat_velocity = None
        self.add_bias_zero_action_tail = None
        self.scatter_rows = None
        self.gather_rows = None
        self.copy_action_tail = None
        self.add_action_bias_timestep = None
        if self.device.type == "cuda":
            try:
                import flash_rt.flash_rt_kernels as fvk

                self.gemm = fvk.GemmRunner()
                self.fill_flat_velocity = getattr(fvk, "cosmos3_edge_fill_flat_velocity_bf16", None)
                self.add_bias_zero_action_tail = getattr(
                    fvk,
                    "cosmos3_edge_add_bias_zero_action_tail_bf16",
                    None,
                )
                self.scatter_rows = getattr(fvk, "cosmos3_edge_scatter_rows_bf16", None)
                self.gather_rows = getattr(fvk, "cosmos3_edge_gather_rows_bf16", None)
                self.copy_action_tail = getattr(fvk, "cosmos3_edge_copy_action_tail_f32_to_bf16", None)
                self.add_action_bias_timestep = getattr(
                    fvk,
                    "cosmos3_edge_add_action_bias_timestep_bf16",
                    None,
                )
            except Exception:
                self.gemm = None
                self.fill_flat_velocity = None
                self.add_bias_zero_action_tail = None
                self.scatter_rows = None
                self.gather_rows = None
                self.copy_action_tail = None
                self.add_action_bias_timestep = None

        self.boundary_dump.validate_geometry()
        self.base = int(self.boundary_dump.layer0_input_causal.shape[0])
        self.vision_indexes = (
            self.boundary_dump.tensors["s00/vfm_in/vision/sequence_indexes"].to(device=self.device) - self.base
        ).contiguous()
        self.action_indexes = (
            self.boundary_dump.tensors["s00/vfm_in/action/sequence_indexes"].to(device=self.device) - self.base
        ).contiguous()
        self.domain_id = int(self.boundary_dump.tensors["s00/vfm_in/action/domain_id/0"].item())
        self.raw_action_dim = int(self.boundary_dump.tensors["s00/vfm_in/action/raw_action_dim/0"].item())

        self.full = torch.zeros_like(self.boundary_dump.layer0_input_full, device=self.device, dtype=self.dtype)
        self.velocity = torch.zeros(EDGE_FLAT_DIM, device=self.device, dtype=self.dtype)
        self.graph_flat_noise = torch.empty(EDGE_FLAT_DIM, device=self.device, dtype=torch.float32)
        self.graph_timestep = torch.empty(1, 1, device=self.device, dtype=torch.int64)
        self.graph = None
        self.timestep_embed_cache: torch.Tensor | None = None
        self.action_input = torch.empty(EDGE_ACTION_MODEL_SHAPE, device=self.device, dtype=self.dtype)
        self.action_encoded = torch.empty(EDGE_ACTION_MODEL_SHAPE[0], SPEC.hidden_size, device=self.device, dtype=self.dtype)
        self.action_hidden = torch.empty_like(self.action_encoded)
        self.action_output = torch.empty(EDGE_ACTION_MODEL_SHAPE, device=self.device, dtype=self.dtype)
        self.action_in_w = weights.load_tensor(
            "action_proj_in.fc.weight",
            device=self.device,
            dtype=self.dtype,
        )[self.domain_id].view(SPEC.action_dim, SPEC.hidden_size).contiguous()
        self.action_in_bias = weights.load_tensor(
            "action_proj_in.bias.weight",
            device=self.device,
            dtype=self.dtype,
        )[self.domain_id].contiguous()
        self.action_modality = weights.load_tensor(
            "action_modality_embed",
            device=self.device,
            dtype=self.dtype,
        ).contiguous()
        self.action_static_bias = (self.action_in_bias + self.action_modality).contiguous()
        self.action_out_w = weights.load_tensor(
            "action_proj_out.fc.weight",
            device=self.device,
            dtype=self.dtype,
        )[self.domain_id].view(SPEC.hidden_size, SPEC.action_dim).contiguous()
        self.action_out_bias = weights.load_tensor(
            "action_proj_out.bias.weight",
            device=self.device,
            dtype=self.dtype,
        )[self.domain_id].contiguous()
        self._install_static_vision_tokens()
        self.und_cache: EdgeUndKVCache = self.reference.precompute_und_kv_cache(self.boundary_dump)

    def precompute_timestep_embeds(self, timesteps: tuple[int, ...] | list[int]) -> None:
        if not timesteps:
            self.timestep_embed_cache = None
            return
        t = torch.tensor(tuple(int(v) for v in timesteps), device=self.device, dtype=torch.int64)
        self.timestep_embed_cache = self.reference.timestep_embed(t).contiguous()

    def _install_static_vision_tokens(self) -> None:
        vision_noisy = self.boundary_dump.tensors["s00/vfm_in/vision/noisy_frame_indexes/0"].to(device=self.device)
        vision_hidden = self.reference.encode_vision_tokens(
            self.boundary_dump.vision_tokens,
            torch.zeros((), device=self.device),
            vision_noisy,
        )
        self.full[self.vision_indexes] = vision_hidden

    def _stream(self) -> int:
        return torch.cuda.current_stream().cuda_stream

    def _encode_action_tokens(
        self,
        action: torch.Tensor,
        timestep: torch.Tensor,
        *,
        action_loaded: bool = False,
        timestep_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not action_loaded:
            self.action_input.copy_(action.to(device=self.device, dtype=self.dtype))
        if self.gemm is None:
            self.action_encoded.copy_(self.action_input @ self.action_in_w)
        else:
            self.gemm.bf16_nn(
                self.action_input.data_ptr(),
                self.action_in_w.data_ptr(),
                self.action_encoded.data_ptr(),
                EDGE_ACTION_MODEL_SHAPE[0],
                SPEC.hidden_size,
                SPEC.action_dim,
                self._stream(),
            )
        if self.add_action_bias_timestep is not None:
            if timestep_embed is None:
                timestep_embed = self.reference.timestep_embed(timestep.reshape(-1)[:1]).contiguous()
            self.add_action_bias_timestep(
                self.action_encoded.data_ptr(),
                self.action_static_bias.data_ptr(),
                timestep_embed.data_ptr(),
                EDGE_ACTION_MODEL_SHAPE[0],
                SPEC.hidden_size,
                self._stream(),
            )
        else:
            self.action_encoded.add_(self.action_static_bias)
            if timestep_embed is None:
                timestep_embed = self.reference.timestep_embed(
                    timestep.reshape(-1)[:1].expand(EDGE_ACTION_MODEL_SHAPE[0])
                )
            elif timestep_embed.shape[0] == 1:
                timestep_embed = timestep_embed.expand(EDGE_ACTION_MODEL_SHAPE[0], SPEC.hidden_size)
            self.action_encoded.add_(timestep_embed)
        return self.action_encoded

    def _decode_action_velocity(self, full_out: torch.Tensor) -> torch.Tensor:
        if self.gather_rows is not None:
            self.gather_rows(
                full_out.data_ptr(),
                self.action_hidden.data_ptr(),
                self.action_indexes.data_ptr(),
                EDGE_ACTION_MODEL_SHAPE[0],
                SPEC.hidden_size,
                self._stream(),
            )
        else:
            self.action_hidden.copy_(full_out[self.action_indexes])
        if self.gemm is None:
            self.action_output.copy_(self.action_hidden @ self.action_out_w)
        else:
            self.gemm.bf16_nn(
                self.action_hidden.data_ptr(),
                self.action_out_w.data_ptr(),
                self.action_output.data_ptr(),
                EDGE_ACTION_MODEL_SHAPE[0],
                SPEC.action_dim,
                SPEC.hidden_size,
                self._stream(),
            )
        if self.add_bias_zero_action_tail is not None:
            self.add_bias_zero_action_tail(
                self.action_output.data_ptr(),
                self.action_out_bias.data_ptr(),
                EDGE_ACTION_MODEL_SHAPE[0],
                EDGE_ACTION_MODEL_SHAPE[1],
                self.raw_action_dim,
                self._stream(),
            )
        else:
            self.action_output.add_(self.action_out_bias)
            self.action_output[:, self.raw_action_dim :] = 0
        return self.action_output

    def full_sequence_for_step(
        self,
        flat_noise: torch.Tensor,
        timestep: torch.Tensor,
        *,
        timestep_index: int | None = None,
    ) -> torch.Tensor:
        action_loaded = False
        if (
            self.copy_action_tail is not None
            and flat_noise.device == self.device
            and flat_noise.dtype == torch.float32
        ):
            self.copy_action_tail(
                flat_noise.data_ptr(),
                self.action_input.data_ptr(),
                flat_noise.numel(),
                EDGE_ACTION_MODEL_SHAPE[0] * EDGE_ACTION_MODEL_SHAPE[1],
                self._stream(),
            )
            action_loaded = True
            action = self.action_input
        else:
            action = flat_noise.to(device=self.device, dtype=self.dtype)[-60 * SPEC.action_dim :].reshape(
                EDGE_ACTION_MODEL_SHAPE
            )
        timestep_embed = None
        if self.timestep_embed_cache is not None and timestep_index is not None:
            timestep_embed = self.timestep_embed_cache[timestep_index : timestep_index + 1]
        action_encoded = self._encode_action_tokens(
            action,
            timestep,
            action_loaded=action_loaded,
            timestep_embed=timestep_embed,
        )
        if self.scatter_rows is not None:
            self.scatter_rows(
                action_encoded.data_ptr(),
                self.full.data_ptr(),
                self.action_indexes.data_ptr(),
                EDGE_ACTION_MODEL_SHAPE[0],
                SPEC.hidden_size,
                self._stream(),
            )
        else:
            self.full[self.action_indexes] = action_encoded
        return self.full

    def action_velocity_for_step(
        self,
        flat_noise: torch.Tensor,
        timestep: torch.Tensor,
        *,
        timestep_index: int | None = None,
    ) -> torch.Tensor:
        full = self.full_sequence_for_step(flat_noise, timestep, timestep_index=timestep_index)
        full_out = self.reference.forward_gen_with_und_cache(self.boundary_dump, full, self.und_cache)
        return self._decode_action_velocity(full_out)

    def flat_velocity_for_step(
        self,
        flat_noise: torch.Tensor,
        timestep: torch.Tensor,
        *,
        timestep_index: int | None = None,
    ) -> torch.Tensor:
        action_velocity = self.action_velocity_for_step(flat_noise, timestep, timestep_index=timestep_index)
        action_flat = action_velocity.reshape(EDGE_ACTION_MODEL_SHAPE[0] * EDGE_ACTION_MODEL_SHAPE[1])
        if self.fill_flat_velocity is not None:
            self.fill_flat_velocity(
                action_flat.data_ptr(),
                self.velocity.data_ptr(),
                self.velocity.numel(),
                action_flat.numel(),
                self._stream(),
            )
            return self.velocity
        self.velocity.zero_()
        self.velocity[-action_flat.numel() :] = action_flat.to(dtype=self.velocity.dtype)
        return self.velocity

    def capture_velocity_graph(self, flat_noise: torch.Tensor, timestep: torch.Tensor) -> None:
        if self.device.type != "cuda":
            raise RuntimeError("Cosmos3-Edge velocity graph capture requires CUDA")
        if not getattr(self.reference, "graph_attention_available", False):
            raise RuntimeError(
                "Cosmos3-Edge velocity graph capture requires graph-safe native "
                "attention. On Thor, set FLASHRT_COSMOS3_EDGE_FA4_FWD=1 to use "
                "the lower-level FA4 forward entry that supports the current "
                "opt-in graph replay path."
            )
        self.graph_flat_noise.copy_(flat_noise.to(device=self.device, dtype=torch.float32))
        self.graph_timestep.copy_(timestep.to(device=self.device, dtype=torch.int64).reshape(1, 1))

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                self.flat_velocity_for_step(self.graph_flat_noise, self.graph_timestep)
        torch.cuda.current_stream().wait_stream(stream)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.flat_velocity_for_step(self.graph_flat_noise, self.graph_timestep)

    def replay_velocity_graph(self, flat_noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        if self.graph is None:
            self.capture_velocity_graph(flat_noise, timestep)
        self.graph_flat_noise.copy_(flat_noise.to(device=self.device, dtype=torch.float32))
        self.graph_timestep.copy_(timestep.to(device=self.device, dtype=torch.int64).reshape(1, 1))
        self.graph.replay()
        return self.velocity

"""PyTorch reference pieces for Cosmos3-Edge layer bring-up."""

from __future__ import annotations

import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from flash_rt.frontends.torch._cosmos3_edge_thor_spec import SPEC
from flash_rt.models.cosmos3_edge.boundary_dump import EdgeBoundaryDump
from flash_rt.models.cosmos3_edge.weights import EdgeTransformerWeights


EdgeUndKVCache = tuple[tuple[torch.Tensor, torch.Tensor], ...]


class _CudaMatmulTf32:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.previous_allow_tf32: bool | None = None

    def __enter__(self) -> None:
        if not torch.cuda.is_available():
            return
        self.previous_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = self.enabled

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.previous_allow_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = self.previous_allow_tf32


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = SPEC.rms_eps) -> torch.Tensor:
    dtype = x.dtype
    y = x.float()
    y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (y * weight.float()).to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope_partial(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rot_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rot_dim], q[..., rot_dim:]
    k_rot, k_pass = k[..., :rot_dim], k[..., rot_dim:]
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    return torch.cat((q_embed, q_pass), dim=-1), torch.cat((k_embed, k_pass), dim=-1)


def _linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.linear(x, weight)


def _attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, is_causal: bool) -> torch.Tensor:
    groups = q.shape[1] // k.shape[1]
    if groups != 1:
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
    q_b = q.transpose(0, 1).unsqueeze(0)
    k_b = k.transpose(0, 1).unsqueeze(0)
    v_b = v.transpose(0, 1).unsqueeze(0)
    out = F.scaled_dot_product_attention(q_b, k_b, v_b, is_causal=is_causal, scale=SPEC.head_dim ** -0.5)
    return out.squeeze(0).transpose(0, 1).reshape(q.shape[0], SPEC.hidden_size)


def _apply_timestep_embeds_to_noisy_tokens(
    packed_tokens: torch.Tensor,
    packed_timestep_embeds: torch.Tensor,
    noisy_frame_indexes: torch.Tensor,
    token_shape: tuple[int, ...],
) -> torch.Tensor:
    if noisy_frame_indexes.numel() == 0:
        return packed_tokens
    spatial_numel = math.prod(token_shape[1:])
    spatial_indexes = torch.arange(spatial_numel, device=packed_tokens.device)
    token_indexes = (noisy_frame_indexes * spatial_numel).unsqueeze(-1).expand(-1, spatial_numel)
    token_indexes = (token_indexes + spatial_indexes).flatten().to(torch.long)
    token_indexes = token_indexes.unsqueeze(-1).expand(-1, packed_tokens.shape[1])
    return packed_tokens.scatter_add(dim=0, index=token_indexes, src=packed_timestep_embeds)


class EdgeLayer0TorchReference:
    """Torch math reference for Edge transformer layer 0.

    This is intentionally not the optimized path. It exists to validate the
    Edge tower mapping and formulas before replacing individual pieces with
    FlashRT kernels.
    """

    def __init__(
        self,
        weights: EdgeTransformerWeights,
        *,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.weights = weights
        self.device = torch.device(device)
        self.dtype = dtype
        self._cache: dict[str, torch.Tensor] = {}
        self._float_cache: dict[str, torch.Tensor] = {}
        self._dump_cache: dict[str, torch.Tensor] = {}

    def w(self, key: str) -> torch.Tensor:
        if key not in self._cache:
            self._cache[key] = self.weights.load_tensor(key, device=self.device, dtype=self.dtype)
        return self._cache[key]

    def wf(self, key: str) -> torch.Tensor:
        if key not in self._float_cache:
            self._float_cache[key] = self.weights.load_tensor(key, device=self.device, dtype=torch.float32)
        return self._float_cache[key]

    def dump_tensor(self, dump: EdgeBoundaryDump, key: str, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        target_dtype = dtype or self.dtype
        cache_key = f"{id(dump)}:{key}:{target_dtype}"
        if cache_key not in self._dump_cache:
            self._dump_cache[cache_key] = dump.tensors[key].to(device=self.device, dtype=target_dtype).contiguous()
        return self._dump_cache[cache_key]

    def layer_w(self, suffix: str) -> torch.Tensor:
        return self.w(f"layers.0.{suffix}")

    def linear(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return _linear(x, weight)

    def norm(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, weight)

    def relu2(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x).square()

    def add(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b

    def concat_gen_kv(
        self,
        k_und_for_gen_rope: torch.Tensor,
        k_gen_rope: torch.Tensor,
        v_und: torch.Tensor,
        v_gen: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.cat([k_und_for_gen_rope, k_gen_rope], dim=0), torch.cat([v_und, v_gen], dim=0)

    def apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return apply_rope_partial(q, k, cos, sin)

    def norm_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.apply_rope(
            self.norm(q, q_weight),
            self.norm(k, k_weight),
            cos,
            sin,
        )

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, is_causal: bool) -> torch.Tensor:
        return _attention(q, k, v, is_causal=is_causal)

    def clear_cache(self) -> None:
        self._cache.clear()
        self._float_cache.clear()
        self._dump_cache.clear()

    def _tower_attention(
        self,
        causal: torch.Tensor,
        full: torch.Tensor,
        dump: EdgeBoundaryDump,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_und = causal.shape[0]
        n_gen = full.shape[0]

        und_norm = rms_norm(causal, self.layer_w("input_layernorm.weight"))
        gen_norm = rms_norm(full, self.layer_w("input_layernorm_moe_gen.weight"))

        q_und = _linear(und_norm, self.layer_w("self_attn.to_q.weight")).view(n_und, SPEC.num_heads, SPEC.head_dim)
        k_und = _linear(und_norm, self.layer_w("self_attn.to_k.weight")).view(n_und, SPEC.num_kv_heads, SPEC.head_dim)
        v_und = _linear(und_norm, self.layer_w("self_attn.to_v.weight")).view(n_und, SPEC.num_kv_heads, SPEC.head_dim)

        q_gen = _linear(gen_norm, self.layer_w("self_attn.add_q_proj.weight")).view(
            n_gen, SPEC.num_heads, SPEC.head_dim
        )
        k_gen = _linear(gen_norm, self.layer_w("self_attn.add_k_proj.weight")).view(
            n_gen, SPEC.num_kv_heads, SPEC.head_dim
        )
        v_gen = _linear(gen_norm, self.layer_w("self_attn.add_v_proj.weight")).view(
            n_gen, SPEC.num_kv_heads, SPEC.head_dim
        )

        cos_c = self.dump_tensor(dump, "s00/layers/00/rope/cos/causal_seq")
        sin_c = self.dump_tensor(dump, "s00/layers/00/rope/sin/causal_seq")
        cos_f = self.dump_tensor(dump, "s00/layers/00/rope/cos/full_only_seq")
        sin_f = self.dump_tensor(dump, "s00/layers/00/rope/sin/full_only_seq")
        q_und_rope, k_und_rope = self.apply_rope(q_und, k_und, cos_c, sin_c)
        q_gen_rope, k_gen_rope = self.norm_rope(
            q_gen,
            k_gen,
            self.layer_w("self_attn.norm_added_q.weight"),
            self.layer_w("self_attn.norm_added_k.weight"),
            cos_f,
            sin_f,
        )

        k_und_for_gen = rms_norm(k_und, self.layer_w("self_attn.k_norm_und_for_gen.weight"))
        _, k_und_for_gen_rope = apply_rope_partial(q_und_rope, k_und_for_gen, cos_c, sin_c)

        und_attn = self.attention(q_und_rope, k_und_rope, v_und, is_causal=True)
        k_gen_attn, v_gen_attn = self.concat_gen_kv(k_und_for_gen_rope, k_gen_rope, v_und, v_gen)
        gen_attn = self.attention(q_gen_rope, k_gen_attn, v_gen_attn, is_causal=False)

        und_out = _linear(und_attn, self.layer_w("self_attn.to_out.weight"))
        gen_out = _linear(gen_attn, self.layer_w("self_attn.to_add_out.weight"))
        return und_out, gen_out

    def forward(self, dump: EdgeBoundaryDump) -> tuple[torch.Tensor, torch.Tensor]:
        causal = dump.layer0_input_causal.to(device=self.device, dtype=self.dtype)
        full = dump.layer0_input_full.to(device=self.device, dtype=self.dtype)

        und_attn, gen_attn = self._tower_attention(causal, full, dump)
        residual_und = causal + und_attn
        residual_gen = full + gen_attn

        und_mlp_in = rms_norm(residual_und, self.layer_w("post_attention_layernorm.weight"))
        gen_mlp_in = rms_norm(residual_gen, self.layer_w("post_attention_layernorm_moe_gen.weight"))

        und_mlp = _linear(
            self.relu2(_linear(und_mlp_in, self.layer_w("mlp.up_proj.weight"))),
            self.layer_w("mlp.down_proj.weight"),
        )
        gen_mlp = _linear(
            self.relu2(_linear(gen_mlp_in, self.layer_w("mlp_moe_gen.up_proj.weight"))),
            self.layer_w("mlp_moe_gen.down_proj.weight"),
        )

        return residual_und + und_mlp, residual_gen + gen_mlp


class EdgeTransformerTorchReference(EdgeLayer0TorchReference):
    """Full 28-layer Torch reference for step-0 velocity validation."""

    def layer_w_at(self, layer: int, suffix: str) -> torch.Tensor:
        return self.w(f"layers.{layer}.{suffix}")

    def _tower_attention_at(
        self,
        layer: int,
        causal: torch.Tensor,
        full: torch.Tensor,
        dump: EdgeBoundaryDump,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_und = causal.shape[0]
        n_gen = full.shape[0]
        lw = lambda suffix: self.layer_w_at(layer, suffix)

        und_norm = rms_norm(causal, lw("input_layernorm.weight"))
        gen_norm = rms_norm(full, lw("input_layernorm_moe_gen.weight"))

        q_und = self.linear(und_norm, lw("self_attn.to_q.weight")).view(n_und, SPEC.num_heads, SPEC.head_dim)
        k_und = self.linear(und_norm, lw("self_attn.to_k.weight")).view(n_und, SPEC.num_kv_heads, SPEC.head_dim)
        v_und = self.linear(und_norm, lw("self_attn.to_v.weight")).view(n_und, SPEC.num_kv_heads, SPEC.head_dim)

        q_gen = _linear(gen_norm, lw("self_attn.add_q_proj.weight")).view(n_gen, SPEC.num_heads, SPEC.head_dim)
        k_gen = _linear(gen_norm, lw("self_attn.add_k_proj.weight")).view(n_gen, SPEC.num_kv_heads, SPEC.head_dim)
        v_gen = _linear(gen_norm, lw("self_attn.add_v_proj.weight")).view(n_gen, SPEC.num_kv_heads, SPEC.head_dim)

        cos_c = dump.rope_cos_causal.to(device=self.device, dtype=self.dtype)
        sin_c = dump.rope_sin_causal.to(device=self.device, dtype=self.dtype)
        cos_f = dump.rope_cos_full.to(device=self.device, dtype=self.dtype)
        sin_f = dump.rope_sin_full.to(device=self.device, dtype=self.dtype)
        q_und_rope, k_und_rope = apply_rope_partial(q_und, k_und, cos_c, sin_c)
        q_gen_rope, k_gen_rope = self.norm_rope(
            q_gen,
            k_gen,
            lw("self_attn.norm_added_q.weight"),
            lw("self_attn.norm_added_k.weight"),
            cos_f,
            sin_f,
        )

        k_und_for_gen = rms_norm(k_und, lw("self_attn.k_norm_und_for_gen.weight"))
        _, k_und_for_gen_rope = apply_rope_partial(q_und_rope, k_und_for_gen, cos_c, sin_c)

        und_attn = _attention(q_und_rope, k_und_rope, v_und, is_causal=True)
        k_gen_attn, v_gen_attn = self.concat_gen_kv(k_und_for_gen_rope, k_gen_rope, v_und, v_gen)
        gen_attn = _attention(q_gen_rope, k_gen_attn, v_gen_attn, is_causal=False)

        return _linear(und_attn, lw("self_attn.to_out.weight")), _linear(
            gen_attn, lw("self_attn.to_add_out.weight")
        )

    def forward_layer(
        self,
        layer: int,
        causal: torch.Tensor,
        full: torch.Tensor,
        dump: EdgeBoundaryDump,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lw = lambda suffix: self.layer_w_at(layer, suffix)
        und_attn, gen_attn = self._tower_attention_at(layer, causal, full, dump)
        residual_und = causal + und_attn
        residual_gen = full + gen_attn

        und_mlp_in = rms_norm(residual_und, lw("post_attention_layernorm.weight"))
        gen_mlp_in = rms_norm(residual_gen, lw("post_attention_layernorm_moe_gen.weight"))

        und_mlp = _linear(self.relu2(_linear(und_mlp_in, lw("mlp.up_proj.weight"))), lw("mlp.down_proj.weight"))
        gen_mlp = _linear(
            self.relu2(_linear(gen_mlp_in, lw("mlp_moe_gen.up_proj.weight"))),
            lw("mlp_moe_gen.down_proj.weight"),
        )
        return residual_und + und_mlp, residual_gen + gen_mlp

    def forward_transformer(self, dump: EdgeBoundaryDump) -> tuple[torch.Tensor, torch.Tensor]:
        causal = dump.layer0_input_causal.to(device=self.device, dtype=self.dtype)
        full = dump.layer0_input_full.to(device=self.device, dtype=self.dtype)
        return self.forward_transformer_from_sequences(dump, causal, full)

    def forward_transformer_from_sequences(
        self,
        dump: EdgeBoundaryDump,
        causal: torch.Tensor,
        full: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for layer in range(SPEC.num_layers):
            causal, full = self.forward_layer(layer, causal, full, dump)
            self.clear_cache()
        causal = rms_norm(causal, self.w("norm.weight"))
        full = rms_norm(full, self.w("norm_moe_gen.weight"))
        return causal, full

    def _causal_layer_with_gen_kv(
        self,
        layer: int,
        causal: torch.Tensor,
        dump: EdgeBoundaryDump,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_und = causal.shape[0]
        lw = lambda suffix: self.layer_w_at(layer, suffix)
        und_norm = self.norm(causal, lw("input_layernorm.weight"))

        q_und = self.linear(und_norm, lw("self_attn.to_q.weight")).view(n_und, SPEC.num_heads, SPEC.head_dim)
        k_und = self.linear(und_norm, lw("self_attn.to_k.weight")).view(n_und, SPEC.num_kv_heads, SPEC.head_dim)
        v_und = self.linear(und_norm, lw("self_attn.to_v.weight")).view(n_und, SPEC.num_kv_heads, SPEC.head_dim)

        cos_c = self.dump_tensor(dump, "s00/layers/00/rope/cos/causal_seq")
        sin_c = self.dump_tensor(dump, "s00/layers/00/rope/sin/causal_seq")
        q_und_rope, k_und_rope = self.apply_rope(q_und, k_und, cos_c, sin_c)

        k_und_for_gen = self.norm(k_und, lw("self_attn.k_norm_und_for_gen.weight"))
        _, k_und_for_gen_rope = apply_rope_partial(k_und_for_gen, k_und_for_gen, cos_c, sin_c)

        und_attn = self.attention(q_und_rope, k_und_rope, v_und, is_causal=True)
        und_out = self.linear(und_attn, lw("self_attn.to_out.weight"))
        residual_und = causal + und_out
        und_mlp_in = self.norm(residual_und, lw("post_attention_layernorm.weight"))
        und_mlp = self.linear(
            self.relu2(self.linear(und_mlp_in, lw("mlp.up_proj.weight"))),
            lw("mlp.down_proj.weight"),
        )
        return residual_und + und_mlp, k_und_for_gen_rope.contiguous(), v_und.contiguous()

    def precompute_und_kv_cache(self, dump: EdgeBoundaryDump) -> EdgeUndKVCache:
        causal = dump.layer0_input_causal.to(device=self.device, dtype=self.dtype)
        cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in range(SPEC.num_layers):
            causal, k_und_for_gen, v_und = self._causal_layer_with_gen_kv(layer, causal, dump)
            cache.append((k_und_for_gen.clone(), v_und.clone()))
            self.clear_cache()
        return tuple(cache)

    def _gen_layer_with_und_cache(
        self,
        layer: int,
        full: torch.Tensor,
        dump: EdgeBoundaryDump,
        k_und_for_gen_rope: torch.Tensor,
        v_und: torch.Tensor,
    ) -> torch.Tensor:
        n_gen = full.shape[0]
        lw = lambda suffix: self.layer_w_at(layer, suffix)
        gen_norm = self.norm(full, lw("input_layernorm_moe_gen.weight"))

        q_gen = self.linear(gen_norm, lw("self_attn.add_q_proj.weight")).view(n_gen, SPEC.num_heads, SPEC.head_dim)
        k_gen = self.linear(gen_norm, lw("self_attn.add_k_proj.weight")).view(n_gen, SPEC.num_kv_heads, SPEC.head_dim)
        v_gen = self.linear(gen_norm, lw("self_attn.add_v_proj.weight")).view(n_gen, SPEC.num_kv_heads, SPEC.head_dim)

        cos_f = self.dump_tensor(dump, "s00/layers/00/rope/cos/full_only_seq")
        sin_f = self.dump_tensor(dump, "s00/layers/00/rope/sin/full_only_seq")
        q_gen_rope, k_gen_rope = self.norm_rope(
            q_gen,
            k_gen,
            lw("self_attn.norm_added_q.weight"),
            lw("self_attn.norm_added_k.weight"),
            cos_f,
            sin_f,
        )

        k_gen_attn, v_gen_attn = self.concat_gen_kv(k_und_for_gen_rope, k_gen_rope, v_und, v_gen)
        gen_attn = self.attention(q_gen_rope, k_gen_attn, v_gen_attn, is_causal=False)
        gen_out = self.linear(gen_attn, lw("self_attn.to_add_out.weight"))
        residual_gen = self.add(full, gen_out)
        gen_mlp_in = self.norm(residual_gen, lw("post_attention_layernorm_moe_gen.weight"))
        gen_mlp = self.linear(
            self.relu2(self.linear(gen_mlp_in, lw("mlp_moe_gen.up_proj.weight"))),
            lw("mlp_moe_gen.down_proj.weight"),
        )
        return self.add(residual_gen, gen_mlp)

    def forward_gen_with_und_cache(
        self,
        dump: EdgeBoundaryDump,
        full: torch.Tensor,
        und_cache: EdgeUndKVCache,
    ) -> torch.Tensor:
        if len(und_cache) != SPEC.num_layers:
            raise ValueError(f"expected {SPEC.num_layers} cached layers, got {len(und_cache)}")
        for layer, (k_und_for_gen, v_und) in enumerate(und_cache):
            full = self._gen_layer_with_und_cache(layer, full, dump, k_und_for_gen, v_und)
            self.clear_cache()
        return self.norm(full, self.w("norm_moe_gen.weight"))

    def timestep_embed(self, timestep: torch.Tensor) -> torch.Tensor:
        t = timestep.to(device=self.device, dtype=torch.float32).reshape(-1) * 0.001
        half = 128
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, dtype=torch.float32, device=self.device) / half)
        args = t[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        h = F.linear(
            emb,
            self.wf("time_embedder.linear_1.weight"),
            self.wf("time_embedder.linear_1.bias"),
        )
        h = F.silu(h)
        h = F.linear(
            h,
            self.wf("time_embedder.linear_2.weight"),
            self.wf("time_embedder.linear_2.bias"),
        )
        return h.to(self.dtype)

    def patchify_vision_tokens(self, vision: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
        latent = vision.to(device=self.device, dtype=self.dtype).squeeze(0)
        if latent.dim() != 4:
            raise ValueError(f"expected vision latent [1,C,T,H,W] or [C,T,H,W], got {tuple(vision.shape)}")
        channels, latent_t, latent_h, latent_w = latent.shape
        if channels != SPEC.latent_channels:
            raise ValueError(f"expected {SPEC.latent_channels} latent channels, got {channels}")

        patch = 2
        h_padded = ((latent_h + patch - 1) // patch) * patch
        w_padded = ((latent_w + patch - 1) // patch) * patch
        if h_padded != latent_h or w_padded != latent_w:
            padded = torch.zeros(
                (channels, latent_t, h_padded, w_padded),
                device=self.device,
                dtype=self.dtype,
            )
            padded[:, :, :latent_h, :latent_w] = latent
            latent = padded

        h_patches = h_padded // patch
        w_patches = w_padded // patch
        latent = latent.reshape(channels, latent_t, h_patches, patch, w_patches, patch)
        packed = torch.einsum("cthpwq->thwpqc", latent).reshape(-1, SPEC.patch_latent_dim)
        return packed, (latent_t, h_patches, w_patches)

    def encode_vision_tokens(
        self,
        vision: torch.Tensor,
        timestep: torch.Tensor,
        noisy_frame_indexes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        packed, token_shape = self.patchify_vision_tokens(vision)
        encoded = F.linear(
            packed,
            self.weights.load_tensor("proj_in.weight", device=self.device, dtype=self.dtype),
            self.weights.load_tensor("proj_in.bias", device=self.device, dtype=self.dtype),
        )
        if noisy_frame_indexes is None or noisy_frame_indexes.numel() == 0:
            return encoded
        noisy_frame_indexes = noisy_frame_indexes.to(device=self.device)
        noisy_patches = noisy_frame_indexes.numel() * math.prod(token_shape[1:])
        timestep_embeds = self.timestep_embed(timestep.reshape(-1)[:1].expand(noisy_patches))
        return _apply_timestep_embeds_to_noisy_tokens(encoded, timestep_embeds, noisy_frame_indexes, token_shape)

    def encode_action_tokens(self, action: torch.Tensor, timestep: torch.Tensor, domain_id: int) -> torch.Tensor:
        action = action.to(device=self.device, dtype=self.dtype)
        fc = self.weights.load_tensor("action_proj_in.fc.weight", device=self.device, dtype=self.dtype)[domain_id]
        bias = self.weights.load_tensor("action_proj_in.bias.weight", device=self.device, dtype=self.dtype)[domain_id]
        modality = self.weights.load_tensor("action_modality_embed", device=self.device, dtype=self.dtype)
        with _CudaMatmulTf32(False):
            encoded = action @ fc.view(SPEC.action_dim, SPEC.hidden_size)
            encoded = encoded + bias + modality
            encoded = encoded + self.timestep_embed(timestep.reshape(-1)[:1].expand(action.shape[0]))
        return encoded

    def full_sequence_for_step(self, dump: EdgeBoundaryDump, flat_noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        domain_id = int(dump.tensors["s00/vfm_in/action/domain_id/0"].item())
        action = flat_noise[-60 * SPEC.action_dim :].reshape(60, SPEC.action_dim)
        base = dump.layer0_input_causal.shape[0]
        full = torch.zeros_like(dump.layer0_input_full, device=self.device, dtype=self.dtype)

        vision_indexes = dump.tensors["s00/vfm_in/vision/sequence_indexes"].to(device=self.device) - base
        vision_noisy = dump.tensors["s00/vfm_in/vision/noisy_frame_indexes/0"].to(device=self.device)
        full[vision_indexes] = self.encode_vision_tokens(dump.vision_tokens, timestep, vision_noisy)

        action_indexes = dump.tensors["s00/vfm_in/action/sequence_indexes"].to(device=self.device) - base
        full[action_indexes] = self.encode_action_tokens(action, timestep, domain_id)
        return full

    def step0_action_velocity(self, dump: EdgeBoundaryDump) -> torch.Tensor:
        _, full = self.forward_transformer(dump)
        return self.action_velocity_from_final_full(dump, full)

    def action_velocity_from_final_full(self, dump: EdgeBoundaryDump, full: torch.Tensor) -> torch.Tensor:
        base = dump.layer0_input_causal.shape[0]
        action_indexes = dump.tensors["s00/vfm_in/action/sequence_indexes"].to(device=self.device) - base
        action_hidden = full[action_indexes]
        domain_id = int(dump.tensors["s00/vfm_in/action/domain_id/0"].item())
        raw_action_dim = int(dump.tensors["s00/vfm_in/action/raw_action_dim/0"].item())
        fc = self.weights.load_tensor("action_proj_out.fc.weight", device=self.device, dtype=self.dtype)[domain_id]
        bias = self.weights.load_tensor("action_proj_out.bias.weight", device=self.device, dtype=self.dtype)[domain_id]
        out = action_hidden @ fc.view(SPEC.hidden_size, SPEC.action_dim) + bias
        out[:, raw_action_dim:] = 0
        return out

    def action_velocity_for_step(
        self,
        dump: EdgeBoundaryDump,
        flat_noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        causal = dump.layer0_input_causal.to(device=self.device, dtype=self.dtype)
        full = self.full_sequence_for_step(dump, flat_noise, timestep)
        _, full_out = self.forward_transformer_from_sequences(dump, causal, full)
        return self.action_velocity_from_final_full(dump, full_out)

    def action_velocity_for_step_with_und_cache(
        self,
        dump: EdgeBoundaryDump,
        flat_noise: torch.Tensor,
        timestep: torch.Tensor,
        und_cache: EdgeUndKVCache,
    ) -> torch.Tensor:
        full = self.full_sequence_for_step(dump, flat_noise, timestep)
        full_out = self.forward_gen_with_und_cache(dump, full, und_cache)
        return self.action_velocity_from_final_full(dump, full_out)


class EdgeTransformerFvkLinearReference(EdgeTransformerTorchReference):
    """Torch reference with BF16 linear calls routed through ``GemmRunner``.

    This is an incremental P1 bridge: it validates FlashRT's BF16 GEMM calling
    convention for Edge's static cached path while leaving norm, RoPE, attention,
    and relu2 in the reference implementation.
    """

    def __init__(
        self,
        weights: EdgeTransformerWeights,
        *,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__(weights, device=device, dtype=dtype)
        self._linear_t_cache: dict[int, torch.Tensor] = {}
        self._fmha_buffers: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._mha_buffers: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self.gemm = None
        self.ctx = None
        self.fa2 = None
        self._fmha_loaded: bool | None = None
        self._fa4_tried = False
        self._fa4_func = None
        self._fa4_fwd_tried = False
        self._fa4_fwd = None
        self._fa4_fwd_buffers: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        if self.device.type == "cuda" and self.dtype == torch.bfloat16:
            try:
                import flash_rt.flash_rt_kernels as fvk

                self.gemm = fvk.GemmRunner()
                self.ctx = fvk.FvkContext()
            except Exception:
                self.gemm = None
                self.ctx = None
            try:
                from flash_rt import flash_rt_fa2 as fa2

                self.fa2 = fa2
            except ImportError:
                self.fa2 = None

    @property
    def native_attention_available(self) -> bool:
        return (
            self.fa2 is not None
            or self._get_fa4_func() is not None
            or (os.environ.get("FLASHRT_COSMOS3_EDGE_FMHA", "0") == "1" and self._ensure_fmha_strided())
            or (os.environ.get("FLASHRT_COSMOS3_EDGE_MHA", "0") == "1" and self.ctx is not None)
        )

    @property
    def graph_attention_available(self) -> bool:
        return self.fa2 is not None or (
            os.environ.get("FLASHRT_COSMOS3_EDGE_FA4_FWD", "0") == "1"
            and self._get_fa4_fwd() is not None
        )

    def _get_fa4_func(self):
        if self._fa4_tried:
            return self._fa4_func
        self._fa4_tried = True
        if self.gemm is None or self.device.type != "cuda":
            return None
        try:
            from flash_rt.hardware.thor import fa4_backend

            self._fa4_func = fa4_backend.fa4_func()
        except Exception:
            self._fa4_func = None
        return self._fa4_func

    def _get_fa4_fwd(self):
        if self._fa4_fwd_tried:
            return self._fa4_fwd
        self._fa4_fwd_tried = True
        if self.gemm is None or self.device.type != "cuda":
            return None
        try:
            from flash_rt.hardware.thor import fa4_backend

            self._fa4_fwd = fa4_backend.fa4_fwd()
        except Exception:
            self._fa4_fwd = None
        return self._fa4_fwd

    def _ensure_fmha_strided(self) -> bool:
        if self._fmha_loaded is not None:
            return self._fmha_loaded
        self._fmha_loaded = False
        if self.gemm is None or self.device.type != "cuda":
            return False
        try:
            import flash_rt.flash_rt_kernels as fvk

            so_path = Path(fvk.__file__).with_name("libfmha_fp16_strided.so")
            if not so_path.exists() or not hasattr(fvk, "fmha_strided_full"):
                return False
            fvk.load_fmha_strided_library(str(so_path))
            self._fmha_loaded = True
        except Exception:
            self._fmha_loaded = False
        return self._fmha_loaded

    def clear_cache(self) -> None:
        # The static P1 engine owns this subclass and reuses the same fixed
        # geometry across all denoise steps. Keep weights and transposes resident;
        # graph capture and latency measurements are meaningless if each layer
        # reloads safetensors from disk.
        return None

    def linear(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if self.gemm is None or x.dim() != 2 or weight.dim() != 2:
            return super().linear(x, weight)
        x_in = x.contiguous()
        cache_key = id(weight)
        if cache_key not in self._linear_t_cache:
            self._linear_t_cache[cache_key] = weight.t().contiguous()
        weight_t = self._linear_t_cache[cache_key]
        out = torch.empty(x_in.shape[0], weight.shape[0], device=self.device, dtype=self.dtype)
        self.gemm.bf16_nn(
            x_in.data_ptr(),
            weight_t.data_ptr(),
            out.data_ptr(),
            x_in.shape[0],
            weight.shape[0],
            x_in.shape[1],
            torch.cuda.current_stream().cuda_stream,
        )
        return out

    def relu2(self, x: torch.Tensor) -> torch.Tensor:
        if (
            self.gemm is None
            or self.device.type != "cuda"
            or self.dtype != torch.bfloat16
            or os.environ.get("FLASHRT_COSMOS3_EDGE_NATIVE_RELU2", "1") == "0"
        ):
            return super().relu2(x)
        import flash_rt.flash_rt_kernels as fvk

        if not hasattr(fvk, "relu2_inplace_bf16"):
            return super().relu2(x)
        out = x.contiguous()
        fvk.relu2_inplace_bf16(
            out.data_ptr(),
            out.numel(),
            torch.cuda.current_stream().cuda_stream,
        )
        return out

    def add(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if (
            self.gemm is None
            or self.device.type != "cuda"
            or self.dtype != torch.bfloat16
            or a.shape != b.shape
            or os.environ.get("FLASHRT_COSMOS3_EDGE_NATIVE_ADD", "0") != "1"
        ):
            return super().add(a, b)
        import flash_rt.flash_rt_kernels as fvk

        if not hasattr(fvk, "cosmos3_edge_add_bf16"):
            return super().add(a, b)
        a_in = a.contiguous()
        b_in = b.contiguous()
        out = torch.empty_like(a_in)
        fvk.cosmos3_edge_add_bf16(
            a_in.data_ptr(),
            b_in.data_ptr(),
            out.data_ptr(),
            out.numel(),
            torch.cuda.current_stream().cuda_stream,
        )
        return out.view_as(a)

    def norm(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if self.gemm is None or self.device.type != "cuda" or self.dtype != torch.bfloat16:
            return super().norm(x, weight)
        import flash_rt.flash_rt_kernels as fvk

        x_in = x.contiguous()
        out = torch.empty_like(x_in)
        dim = x_in.shape[-1]
        rows = x_in.numel() // dim
        fvk.rms_norm(
            x_in.data_ptr(),
            weight.contiguous().data_ptr(),
            out.data_ptr(),
            rows,
            dim,
            SPEC.rms_eps,
            torch.cuda.current_stream().cuda_stream,
        )
        return out.view_as(x)

    def apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.gemm is None or self.device.type != "cuda" or self.dtype != torch.bfloat16:
            return super().apply_rope(q, k, cos, sin)
        if q.dim() != 3 or k.dim() != 3 or q.shape[1:] != (SPEC.num_heads, SPEC.head_dim):
            return super().apply_rope(q, k, cos, sin)
        if k.shape[1:] != (SPEC.num_kv_heads, SPEC.head_dim):
            return super().apply_rope(q, k, cos, sin)
        import flash_rt.flash_rt_kernels as fvk

        q_in = q.contiguous()
        k_in = k.contiguous()
        cos_in = cos.contiguous()
        sin_in = sin.contiguous()
        q_out = torch.empty_like(q_in)
        k_out = torch.empty_like(k_in)
        fvk.qwen36_partial_rope_qk_bf16(
            q_in.data_ptr(),
            k_in.data_ptr(),
            cos_in.data_ptr(),
            sin_in.data_ptr(),
            q_out.data_ptr(),
            k_out.data_ptr(),
            q_in.shape[0],
            SPEC.num_heads,
            SPEC.num_kv_heads,
            SPEC.head_dim,
            SPEC.head_dim,
            torch.cuda.current_stream().cuda_stream,
        )
        return q_out, k_out

    def norm_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self.gemm is None
            or self.device.type != "cuda"
            or self.dtype != torch.bfloat16
            or os.environ.get("FLASHRT_COSMOS3_EDGE_FUSED_QK_NORM_ROPE", "1") == "0"
        ):
            return super().norm_rope(q, k, q_weight, k_weight, cos, sin)
        if q.dim() != 3 or k.dim() != 3 or q.shape[1:] != (SPEC.num_heads, SPEC.head_dim):
            return super().norm_rope(q, k, q_weight, k_weight, cos, sin)
        if k.shape[1:] != (SPEC.num_kv_heads, SPEC.head_dim) or q.shape[0] != k.shape[0]:
            return super().norm_rope(q, k, q_weight, k_weight, cos, sin)
        if cos.shape != (q.shape[0], SPEC.head_dim) or sin.shape != (q.shape[0], SPEC.head_dim):
            return super().norm_rope(q, k, q_weight, k_weight, cos, sin)
        import flash_rt.flash_rt_kernels as fvk

        if not hasattr(fvk, "cosmos3_edge_qk_norm_rope_bf16"):
            return super().norm_rope(q, k, q_weight, k_weight, cos, sin)
        q_in = q.contiguous()
        k_in = k.contiguous()
        q_out = torch.empty_like(q_in)
        k_out = torch.empty_like(k_in)
        fvk.cosmos3_edge_qk_norm_rope_bf16(
            q_in.data_ptr(),
            k_in.data_ptr(),
            q_weight.contiguous().data_ptr(),
            k_weight.contiguous().data_ptr(),
            cos.contiguous().data_ptr(),
            sin.contiguous().data_ptr(),
            q_out.data_ptr(),
            k_out.data_ptr(),
            q_in.shape[0],
            SPEC.num_heads,
            SPEC.num_kv_heads,
            SPEC.head_dim,
            SPEC.head_dim,
            SPEC.rms_eps,
            torch.cuda.current_stream().cuda_stream,
        )
        return q_out, k_out

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, is_causal: bool) -> torch.Tensor:
        if self.gemm is None or self.device.type != "cuda" or self.dtype != torch.bfloat16:
            return super().attention(q, k, v, is_causal=is_causal)
        if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
            return super().attention(q, k, v, is_causal=is_causal)
        if q.shape[1:] != (SPEC.num_heads, SPEC.head_dim):
            return super().attention(q, k, v, is_causal=is_causal)
        if k.shape[1:] != (SPEC.num_kv_heads, SPEC.head_dim) or v.shape[1:] != (SPEC.num_kv_heads, SPEC.head_dim):
            return super().attention(q, k, v, is_causal=is_causal)
        if not is_causal and os.environ.get("FLASHRT_COSMOS3_EDGE_FA4_FWD", "0") == "1":
            fa4_fwd = self._get_fa4_fwd()
            if fa4_fwd is not None:
                return self._attention_fa4_fwd(q, k, v, fa4_fwd)
        fa4 = self._get_fa4_func()
        if not is_causal and fa4 is not None:
            return self._attention_fa4(q, k, v, fa4)
        if (
            os.environ.get("FLASHRT_COSMOS3_EDGE_FMHA", "0") == "1"
            and not is_causal
            and q.shape[0] >= 128
            and k.shape[0] >= 128
            and self._ensure_fmha_strided()
        ):
            return self._attention_fmha_strided(q, k, v)
        if os.environ.get("FLASHRT_COSMOS3_EDGE_MHA", "0") == "1" and not is_causal and self.ctx is not None:
            return self._attention_mha_bf16(q, k, v)
        if self.fa2 is None:
            return super().attention(q, k, v, is_causal=is_causal)

        q_in = q.contiguous()
        k_in = k.contiguous()
        v_in = v.contiguous()
        out = torch.empty_like(q_in)
        lse = torch.empty(1, SPEC.num_heads, max(q_in.shape[0], k_in.shape[0]), dtype=torch.float32, device=self.device)
        fwd = self.fa2.fwd_bf16_causal if is_causal else self.fa2.fwd_bf16
        q_strides = (q_in.shape[0] * SPEC.num_heads * SPEC.head_dim, SPEC.num_heads * SPEC.head_dim, SPEC.head_dim)
        k_strides = (k_in.shape[0] * SPEC.num_kv_heads * SPEC.head_dim, SPEC.num_kv_heads * SPEC.head_dim, SPEC.head_dim)
        fwd(
            Q=q_in.data_ptr(),
            K=k_in.data_ptr(),
            V=v_in.data_ptr(),
            O=out.data_ptr(),
            softmax_lse=lse.data_ptr(),
            softmax_lse_accum=0,
            o_accum=0,
            batch=1,
            seqlen_q=q_in.shape[0],
            seqlen_k=k_in.shape[0],
            num_heads_q=SPEC.num_heads,
            num_heads_kv=SPEC.num_kv_heads,
            head_dim=SPEC.head_dim,
            q_strides=q_strides,
            k_strides=k_strides,
            v_strides=k_strides,
            o_strides=q_strides,
            softmax_scale=SPEC.head_dim ** -0.5,
            num_sms=torch.cuda.get_device_properties(0).multi_processor_count,
            stream=torch.cuda.current_stream().cuda_stream,
        )
        return out.reshape(q_in.shape[0], SPEC.hidden_size)

    def _attention_fa4(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, fa4) -> torch.Tensor:
        q_in = q.contiguous().unsqueeze(0)
        k_in = k.contiguous().unsqueeze(0)
        v_in = v.contiguous().unsqueeze(0)
        out = fa4(q_in, k_in, v_in, causal=False, pack_gqa=True)
        if isinstance(out, tuple):
            out = out[0]
        return out.reshape(q.shape[0], SPEC.hidden_size).to(dtype=self.dtype)

    def _attention_fa4_fwd(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, fa4_fwd) -> torch.Tensor:
        q_in = q.contiguous().unsqueeze(0)
        k_in = k.contiguous().unsqueeze(0)
        v_in = v.contiguous().unsqueeze(0)
        shape_key = (q_in.shape[1], k_in.shape[1])
        if shape_key not in self._fa4_fwd_buffers:
            sq, _ = shape_key
            self._fa4_fwd_buffers[shape_key] = (
                torch.empty_like(q_in),
                torch.empty(1, SPEC.num_heads, sq, dtype=torch.float32, device=self.device),
            )
        out, lse = self._fa4_fwd_buffers[shape_key]
        fa4_fwd(
            q_in,
            k_in,
            v_in,
            softmax_scale=SPEC.head_dim ** -0.5,
            causal=False,
            pack_gqa=True,
            out=out,
            lse=lse,
        )
        return out.reshape(q.shape[0], SPEC.hidden_size).to(dtype=self.dtype)

    def _attention_fmha_strided(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        import flash_rt.flash_rt_kernels as fvk

        q_in = q.contiguous()
        k_in = k.contiguous()
        v_in = v.contiguous()
        shape_key = (q_in.shape[0], k_in.shape[0])
        if shape_key not in self._fmha_buffers:
            sq, skv = shape_key
            self._fmha_buffers[shape_key] = (
                torch.empty(1, sq, SPEC.num_heads, SPEC.head_dim, dtype=torch.float16, device=self.device),
                torch.empty(1, skv, SPEC.num_kv_heads, SPEC.head_dim, dtype=torch.float16, device=self.device),
                torch.empty(1, skv, SPEC.num_kv_heads, SPEC.head_dim, dtype=torch.float16, device=self.device),
                torch.empty(1, sq, SPEC.num_heads, SPEC.head_dim, dtype=torch.float16, device=self.device),
                torch.empty(1, sq, SPEC.num_heads, SPEC.head_dim, dtype=self.dtype, device=self.device),
            )
        q_fp16, k_fp16, v_fp16, out_fp16, out_bf16 = self._fmha_buffers[shape_key]
        q_fp16.copy_(q_in.view_as(q_fp16))
        k_fp16.copy_(k_in.view_as(k_fp16))
        v_fp16.copy_(v_in.view_as(v_fp16))
        fvk.fmha_strided_full(
            q_fp16.data_ptr(),
            k_fp16.data_ptr(),
            v_fp16.data_ptr(),
            out_fp16.data_ptr(),
            1,
            q_in.shape[0],
            k_in.shape[0],
            SPEC.num_heads,
            SPEC.num_kv_heads,
            SPEC.head_dim,
            SPEC.num_heads * SPEC.head_dim,
            SPEC.num_kv_heads * SPEC.head_dim,
            torch.cuda.current_stream().cuda_stream,
        )
        out_bf16.copy_(out_fp16)
        return out_bf16.view(q_in.shape[0], SPEC.hidden_size)

    def _attention_mha_bf16(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        import flash_rt.flash_rt_kernels as fvk

        q_in = q.contiguous()
        k_source = k.contiguous()
        v_source = v.contiguous()
        shape_key = (q_in.shape[0], k_source.shape[0])
        if shape_key not in self._mha_buffers:
            sq, skv = shape_key
            skv_stride = ((skv + 7) // 8) * 8
            self._mha_buffers[shape_key] = (
                torch.empty(SPEC.num_heads * sq, skv_stride, dtype=self.dtype, device=self.device),
                torch.empty(sq, SPEC.num_heads, SPEC.head_dim, dtype=self.dtype, device=self.device),
                torch.empty(skv, SPEC.num_heads, SPEC.head_dim, dtype=self.dtype, device=self.device),
                torch.empty(skv, SPEC.num_heads, SPEC.head_dim, dtype=self.dtype, device=self.device),
            )
        logits, out, k_expanded, v_expanded = self._mha_buffers[shape_key]
        groups = SPEC.num_heads // SPEC.num_kv_heads
        k_expanded.view(k_source.shape[0], SPEC.num_kv_heads, groups, SPEC.head_dim).copy_(
            k_source.unsqueeze(2).expand(-1, -1, groups, -1)
        )
        v_expanded.view(v_source.shape[0], SPEC.num_kv_heads, groups, SPEC.head_dim).copy_(
            v_source.unsqueeze(2).expand(-1, -1, groups, -1)
        )
        fvk.gpu_fill_neginf_bf16(logits.data_ptr(), logits.numel(), torch.cuda.current_stream().cuda_stream)
        fvk.attention_mha_bf16(
            self.ctx,
            q_in.data_ptr(),
            k_expanded.data_ptr(),
            v_expanded.data_ptr(),
            logits.data_ptr(),
            out.data_ptr(),
            q_in.shape[0],
            k_source.shape[0],
            SPEC.num_heads,
            SPEC.head_dim,
            SPEC.head_dim ** -0.5,
            logits.shape[1],
            torch.cuda.current_stream().cuda_stream,
        )
        return out.view(q_in.shape[0], SPEC.hidden_size)

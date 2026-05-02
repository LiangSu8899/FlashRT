"""
BAGEL FP8 Engine — Complete FP8 diffusion pipeline for image generation.

Phase A + B from FP8_PIPELINE_PLAN.md:
  - BF16 text prefill → KV cache (understanding expert, run once)
  - FP8 fused diffusion steps with KV cache attention (generation expert)
  - CFG (3-way: cond + text_uncond + img_uncond)
  - VAE decode → PIL image
  - CUDA Graph for diffusion steps

Target: 3 views × 448×448, 24 steps + CFG < 4 seconds on Thor
"""
import os, sys, time, math, torch, torch.nn.functional as F, numpy as np
from safetensors import safe_open
from PIL import Image

# flash_wm/src/bagel_fp8_engine.py
# BAGEL_CODE: ../../code (bagel/code)  FLASH_VLA_ROOT: ../.. (repo root)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_FLASH_WM = os.path.dirname(_THIS_DIR)
BAGEL_CODE = os.path.join(os.path.dirname(os.path.dirname(_FLASH_WM)), "code")
FLASH_VLA_ROOT = os.path.dirname(_FLASH_WM)
sys.path.insert(0, BAGEL_CODE)
sys.path.insert(0, FLASH_VLA_ROOT)
sys.path.insert(0, _FLASH_WM)  # for flash_wm_kernels.so
sys.path.insert(0, os.path.join(_FLASH_WM, "csrc"))  # for ckernel_bagel.py

import flash_rt.flash_rt_kernels as fvk
from flash_rt.core.thor_frontend_utils import quant_fp8

# ── Model constants ──────────────────────────────────────────────────────────
D = 3584; H = 28; KV_H = 4; HD = 128; FFN = 18944; N_LAYERS = 28; K_DIM = KV_H * HD
VOCAB = 152064; FREQ_DIM = 256; MAX_LATENT_SIZE = 64
LATENT_DOWNSAMPLE = 16; LATENT_PATCH_SIZE = 2; LATENT_CH = 16
PATCH_DIM = LATENT_PATCH_SIZE ** 2 * LATENT_CH  # 64

bf16 = torch.bfloat16; fp16 = torch.float16; FP8 = torch.float8_e4m3fn
GEMM_NAMES = ['q', 'k', 'v', 'o', 'gate', 'up', 'down']


# ── Utilities ────────────────────────────────────────────────────────────────

def timestep_embedding(t, dim=FREQ_DIM, max_period=10000):
    """Sinusoidal timestep embedding. t: scalar or [N]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def get_vae_position_ids(H, W, downsample=LATENT_DOWNSAMPLE, max_patches=MAX_LATENT_SIZE):
    """Compute latent position IDs for VAE patches (extrapolate mode)."""
    h, w = H // downsample, W // downsample
    coords_h = torch.arange(0, h)
    coords_w = torch.arange(0, w)
    return (coords_h[:, None] * max_patches + coords_w).flatten()


def rms_norm_bf16(x, weight, eps=1e-6):
    """BF16 RMSNorm: x [*, D], weight [D] → [*, D]."""
    var = x.float().pow(2).mean(-1, keepdim=True)
    return ((x.float() * torch.rsqrt(var + eps)).to(bf16) * weight)


def build_rope(position_ids, dim=HD, base=10000.0):
    """Build RoPE cos/sin for given position IDs.
    position_ids: [N] int → returns (cos, sin) each [N, dim]
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32,
                                             device=position_ids.device) / dim))
    # [N, dim//2]
    freqs = position_ids[:, None].float() * inv_freq[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)  # [N, dim]
    return emb.cos().to(bf16), emb.sin().to(bf16)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply RoPE. q: [N, heads, HD], cos/sin: [N, HD]."""
    cos = cos.unsqueeze(1)  # [N, 1, HD]
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


# ── Engine ───────────────────────────────────────────────────────────────────

class BagelFP8Engine:
    """Complete FP8 diffusion engine for BAGEL image generation."""

    def __init__(self, weights_path, M=784, device='cuda'):
        self.M = M
        self.Sq = M + 2  # query seq len (start_img + M vae tokens + end_img)
        self.device = device
        self.graph = None
        self.calibrated = False

        print(f"[BagelFP8Engine] Loading weights (M={M})...")
        t0 = time.time()

        sf = safe_open(os.path.join(weights_path, "ema.safetensors"),
                       framework='pt', device='cpu')

        # ── Generation expert (FP8) — for diffusion steps ──
        self.gen_fp8_w = []; self.gen_w_scales = []
        self.gen_norms = []; self.gen_biases = []

        # ── Understanding expert ──
        # BF16 weights are kept for prefill (no-graph Python forward_step)
        # and as the calibration reference. FP8-quantized copies are also
        # loaded for the CKernel hot-loop: text path runs FP8 cutlass GEMMs
        # (M=2) when graph mode is active.
        self.und_w = []; self.und_norms = []; self.und_biases = []
        self.und_fp8_w = []; self.und_w_scales = []

        for i in range(N_LAYERS):
            p = f"language_model.model.layers.{i}"
            def _bf16(n): return sf.get_tensor(f"{p}.{n}").to(bf16).to(device).contiguous()
            def _fp16(n): return sf.get_tensor(f"{p}.{n}").to(fp16).to(device).contiguous()

            # Gen expert → FP8
            fp8_l = {}; ws_l = {}
            for gname, wname in [
                ('q', 'self_attn.q_proj_moe_gen.weight'),
                ('k', 'self_attn.k_proj_moe_gen.weight'),
                ('v', 'self_attn.v_proj_moe_gen.weight'),
                ('o', 'self_attn.o_proj_moe_gen.weight'),
                ('gate', 'mlp_moe_gen.gate_proj.weight'),
                ('up', 'mlp_moe_gen.up_proj.weight'),
                ('down', 'mlp_moe_gen.down_proj.weight'),
            ]:
                w_fp8, w_s = quant_fp8(_fp16(wname))
                fp8_l[gname] = w_fp8; ws_l[gname] = w_s
            self.gen_fp8_w.append(fp8_l)
            self.gen_w_scales.append(ws_l)

            self.gen_norms.append({
                'in': _bf16("input_layernorm_moe_gen.weight"),
                'pn': _bf16("post_attention_layernorm_moe_gen.weight"),
                'qn': _bf16("self_attn.q_norm_moe_gen.weight"),
                'kn': _bf16("self_attn.k_norm_moe_gen.weight"),
            })
            self.gen_biases.append({
                'q': _bf16("self_attn.q_proj_moe_gen.bias"),
                'k': _bf16("self_attn.k_proj_moe_gen.bias"),
                'v': _bf16("self_attn.v_proj_moe_gen.bias"),
            })

            # Und expert → BF16 (for prefill / Python forward_step)
            und_q    = _bf16("self_attn.q_proj.weight")
            und_k    = _bf16("self_attn.k_proj.weight")
            und_v    = _bf16("self_attn.v_proj.weight")
            und_o    = _bf16("self_attn.o_proj.weight")
            und_gate = _bf16("mlp.gate_proj.weight")
            und_up   = _bf16("mlp.up_proj.weight")
            und_down = _bf16("mlp.down_proj.weight")
            self.und_w.append({
                'q': und_q, 'k': und_k, 'v': und_v, 'o': und_o,
                'gate': und_gate, 'up': und_up, 'down': und_down,
            })
            # Und expert → FP8 (for CKernel text hot path)
            und_fp8 = {}; und_s = {}
            for gname, w_bf16 in [('q', und_q), ('k', und_k), ('v', und_v),
                                    ('o', und_o), ('gate', und_gate),
                                    ('up', und_up), ('down', und_down)]:
                fp8_w, s = quant_fp8(w_bf16.to(fp16))
                und_fp8[gname] = fp8_w
                und_s[gname] = s
            self.und_fp8_w.append(und_fp8)
            self.und_w_scales.append(und_s)
            self.und_norms.append({
                'in': _bf16("input_layernorm.weight"),
                'pn': _bf16("post_attention_layernorm.weight"),
                'qn': _bf16("self_attn.q_norm.weight"),
                'kn': _bf16("self_attn.k_norm.weight"),
            })
            self.und_biases.append({
                'q': _bf16("self_attn.q_proj.bias"),
                'k': _bf16("self_attn.k_proj.bias"),
                'v': _bf16("self_attn.v_proj.bias"),
            })

            if (i + 1) % 7 == 0:
                print(f"  layers: {i + 1}/{N_LAYERS}")

        # ── Shared weights ──
        # Final norms
        self.und_final_norm = sf.get_tensor("language_model.model.norm.weight").to(bf16).to(device)
        self.gen_final_norm = sf.get_tensor("language_model.model.norm_moe_gen.weight").to(bf16).to(device)

        # Embeddings
        self.embed_tokens = sf.get_tensor("language_model.model.embed_tokens.weight").to(bf16).to(device)

        # VAE projection
        self.vae2llm_w = sf.get_tensor("vae2llm.weight").to(bf16).to(device)
        self.vae2llm_b = sf.get_tensor("vae2llm.bias").to(bf16).to(device)
        self.llm2vae_w = sf.get_tensor("llm2vae.weight").to(bf16).to(device)
        self.llm2vae_b = sf.get_tensor("llm2vae.bias").to(bf16).to(device)

        # Timestep embedder: MLP(256→3584) → SiLU → MLP(3584→3584)
        self.time_mlp0_w = sf.get_tensor("time_embedder.mlp.0.weight").to(bf16).to(device)
        self.time_mlp0_b = sf.get_tensor("time_embedder.mlp.0.bias").to(bf16).to(device)
        self.time_mlp2_w = sf.get_tensor("time_embedder.mlp.2.weight").to(bf16).to(device)
        self.time_mlp2_b = sf.get_tensor("time_embedder.mlp.2.bias").to(bf16).to(device)

        # Latent position embed
        self.latent_pos_embed = sf.get_tensor("latent_pos_embed.pos_embed").to(bf16).to(device)

        del sf

        # ── Tokenizer + special tokens ──
        from modeling.qwen2 import Qwen2Tokenizer
        from data.data_utils import add_special_tokens
        self.tokenizer = Qwen2Tokenizer.from_pretrained(weights_path)
        self.tokenizer, self.special_tokens, _ = add_special_tokens(self.tokenizer)

        # ── VAE decoder ──
        from modeling.autoencoder import load_ae
        self.vae_model, self.vae_config = load_ae(
            local_path=os.path.join(weights_path, "ae.safetensors"))
        self.vae_model = self.vae_model.to(device).eval()

        # ── FP8 buffers and scales ──
        self._alloc_fp8_buffers(M, device)
        self.act_scales = [[torch.ones(1, dtype=torch.float32, device=device)
                            for _ in GEMM_NAMES] for _ in range(N_LAYERS)]
        self.alphas = [[1.0 for _ in GEMM_NAMES] for _ in range(N_LAYERS)]
        # Und (text) path FP8 activation scales. 4 per layer:
        #   [0] und_ln(text_x)        → Q/K/V
        #   [1] text rows of ao       → O
        #   [2] und_pn(text_x_post)   → gate/up
        #   [3] silu(gate)*up (text)  → down
        self.und_act_scales = [[torch.ones(1, dtype=torch.float32, device=device)
                                 for _ in range(4)] for _ in range(N_LAYERS)]
        # Per-weight und alphas (keyed by GEMM name): scale × w_scale.
        self.und_alphas = [{g: 1.0 for g in GEMM_NAMES} for _ in range(N_LAYERS)]

        # ── MoT token indices ──
        self._text_idx = torch.tensor([0, M + 1], device=device)  # start_img, end_img
        self._vae_idx = torch.arange(1, M + 1, device=device)     # VAE tokens

        # ── CKernel context (cuBLAS handle for CUDA Graph) ──
        import flash_wm_kernels as fwk
        self._wm_ctx = fwk.WmContext()

        # ── KV cache buffers (allocated when kv_len is known) ──
        self.kv_len = 0
        self.kv_cache = None  # list of (K, V) per layer, [kv_len, KV_H, HD] bf16

        # ── Merged attention buffers (pre-allocated for CUDA Graph) ──
        self._k_merged = None  # [N_LAYERS][1, H, kv_len+Sq, HD]
        self._v_merged = None

        t_load = time.time() - t0
        print(f"  Loaded in {t_load:.1f}s, GPU: {torch.cuda.memory_allocated() / 1e9:.1f}GB")

    # ── Buffer allocation ────────────────────────────────────────────────────

    def _alloc_fp8_buffers(self, M, dev):
        """Buffers used only by forward_step (Python path for calibration /
        no-graph fallback). Hot-loop graph path lives in CKernelBagel, which
        owns its own buffers."""
        Sq = M + 2
        self.act_fp8 = torch.empty(Sq * FFN, dtype=FP8, device=dev)
        self.buf_x = torch.empty(Sq, D, dtype=bf16, device=dev)
        self.buf_res = torch.empty(Sq, D, dtype=bf16, device=dev)
        self.buf_q = torch.empty(Sq, D, dtype=bf16, device=dev)
        self.buf_k = torch.empty(Sq, K_DIM, dtype=bf16, device=dev)
        self.buf_v = torch.empty(Sq, K_DIM, dtype=bf16, device=dev)
        self.buf_o_out = torch.empty(Sq, D, dtype=bf16, device=dev)

    def _alloc_kv_merged(self, kv_len):
        """Pre-allocate merged K,V buffers for attention with KV cache."""
        total = kv_len + self.Sq
        dev = self.device
        # Per-layer merged buffers: [total, KV_H, HD]
        self._k_merged = [torch.empty(total, KV_H, HD, dtype=bf16, device=dev)
                          for _ in range(N_LAYERS)]
        self._v_merged = [torch.empty(total, KV_H, HD, dtype=bf16, device=dev)
                          for _ in range(N_LAYERS)]

    def _fill_kv_cache(self, kv_cache):
        """Copy static KV cache into merged buffers (before graph capture)."""
        kv_len = kv_cache[0][0].shape[0]
        for i in range(N_LAYERS):
            self._k_merged[i][:kv_len].copy_(kv_cache[i][0])
            self._v_merged[i][:kv_len].copy_(kv_cache[i][1])

    # ── FP8 GEMM helper ─────────────────────────────────────────────────────

    def _fp8_gemm(self, layer_idx, gemm_idx, out_bf16, M, N, K):
        gname = GEMM_NAMES[gemm_idx]
        alpha = self.alphas[layer_idx][gemm_idx]
        w_fp8 = self.gen_fp8_w[layer_idx][gname]
        if N >= 8192:
            fvk.cutlass_fp8_t1_bf16out(
                self.act_fp8.data_ptr(), w_fp8.data_ptr(), out_bf16.data_ptr(),
                M, N, K, alpha, 0.0, 0)
        elif K >= 8192:
            fvk.cutlass_fp8_wide_bf16out(
                self.act_fp8.data_ptr(), w_fp8.data_ptr(), out_bf16.data_ptr(),
                M, N, K, alpha, 0.0, 0)
        else:
            fvk.cutlass_fp8_sq_bf16out(
                self.act_fp8.data_ptr(), w_fp8.data_ptr(), out_bf16.data_ptr(),
                M, N, K, alpha, 0.0, 0)

    # ── Timestep embedding ───────────────────────────────────────────────────

    def time_embed(self, t_scalar):
        """Compute timestep embedding for a single scalar t. Returns [1, D] bf16."""
        t = torch.tensor([t_scalar], device=self.device)
        freq = timestep_embedding(t, FREQ_DIM)  # [1, 256]
        h = F.linear(freq.to(bf16), self.time_mlp0_w, self.time_mlp0_b)
        h = F.silu(h)
        h = F.linear(h, self.time_mlp2_w, self.time_mlp2_b)
        return h  # [1, D]

    # ── Text Prefill (BF16, understanding expert) ────────────────────────────

    @torch.no_grad()
    def prefill_text(self, text):
        """Tokenize text → BF16 forward → KV cache.
        Returns: kv_cache (list of (K, V) per layer), kv_len, rope_pos (next position).
        """
        # Tokenize
        token_ids = self.tokenizer.encode(text)
        token_ids = [self.special_tokens['bos_token_id']] + token_ids + \
                    [self.special_tokens['eos_token_id']]
        T = len(token_ids)
        tokens = torch.tensor(token_ids, dtype=torch.long, device=self.device)

        # Embed
        x = self.embed_tokens[tokens]  # [T, D] bf16

        # Position IDs and RoPE
        pos_ids = torch.arange(T, device=self.device)
        cos, sin = build_rope(pos_ids)  # [T, HD]

        # Forward through 28 layers (understanding expert, BF16)
        kv_cache = []  # list of (K, V) per layer

        for i in range(N_LAYERS):
            uw = self.und_w[i]; nm = self.und_norms[i]; bi = self.und_biases[i]
            residual = x.clone()

            # Input RMSNorm
            x = rms_norm_bf16(x, nm['in'])

            # QKV
            q = F.linear(x, uw['q'], bi['q']).view(T, H, HD)
            k = F.linear(x, uw['k'], bi['k']).view(T, KV_H, HD)
            v = F.linear(x, uw['v'], bi['v']).view(T, KV_H, HD)

            # QK norm
            q = rms_norm_bf16(q, nm['qn'])
            k = rms_norm_bf16(k, nm['kn'])

            # RoPE
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            q = q.to(bf16); k = k.to(bf16); v = v.to(bf16)

            # Store KV for cache (after RoPE, before attention)
            kv_cache.append((k.clone(), v.clone()))

            # Self-attention (causal for text prefill)
            # Must use enable_gqa=True, NOT k.repeat — repeat+SDPA gives wrong results
            q_4d = q.unsqueeze(0).transpose(1, 2)  # [1, H, T, HD]
            k_4d = k.unsqueeze(0).transpose(1, 2)  # [1, KV_H, T, HD]
            v_4d = v.unsqueeze(0).transpose(1, 2)
            ao = F.scaled_dot_product_attention(q_4d, k_4d, v_4d, is_causal=True,
                                                enable_gqa=True)
            ao = ao.transpose(1, 2).contiguous().view(T, D)

            # O projection + residual
            x = residual + F.linear(ao, uw['o'])

            # Post-attention norm + FFN
            residual = x.clone()
            x = rms_norm_bf16(x, nm['pn'])
            g = F.linear(x, uw['gate'])
            u = F.linear(x, uw['up'])
            x = F.silu(g) * u
            x = residual + F.linear(x, uw['down'])

        return kv_cache, T, T  # cache, kv_len, next_rope_pos

    # ── Calibration ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def calibrate(self, x_input, kv_cache=None):
        """Calibrate FP8 activation scales using FP8-aware forward pass.
        Uses actual FP8 GEMMs for layer propagation so scales match real inference.
        x_input: [Sq, D] bf16 — packed input (start_img, vae tokens, end_img).
        kv_cache: list of (K, V) per layer, or None for self-attention only.
        """
        print("[Calibrate] Measuring activation scales (FP8-aware)...")
        Sq = self.Sq; dev = self.device
        kv_len = kv_cache[0][0].shape[0] if kv_cache is not None else 0

        # Setup merged KV if needed
        self._alloc_kv_merged(kv_len)
        self.kv_len = kv_len
        if kv_cache is not None:
            self._fill_kv_cache(kv_cache)

        # RoPE
        rope_pos = kv_len
        q_pos = torch.full((Sq,), rope_pos, dtype=torch.long, device=dev)
        cos, sin = build_rope(q_pos)
        self._rope_cos = cos; self._rope_sin = sin

        total_kv = kv_len + Sq
        self.buf_x.copy_(x_input)

        ti = self._text_idx  # text token indices [0, Sq-1]
        for i in range(N_LAYERS):
            nm = self.gen_norms[i]; bi = self.gen_biases[i]; ws = self.gen_w_scales[i]
            unm = self.und_norms[i]; uws = self.und_w_scales[i]

            self.buf_res.copy_(self.buf_x)

            # Measure amax of norm output for QKV scale
            xn = rms_norm_bf16(self.buf_x, nm['in'])
            amax_qkv = max(xn.abs().float().max().item(), 1e-12)
            s_qkv = max(amax_qkv / 448.0, 1e-12)
            for j in range(3):
                self.act_scales[i][j].fill_(s_qkv)
                self.alphas[i][j] = s_qkv * ws[GEMM_NAMES[j]]

            # Und path: amax of und_ln(text_x) for text Q/K/V FP8 scale
            xn_text_pre = rms_norm_bf16(self.buf_x[ti], unm['in'])
            s_und_qkv = max(xn_text_pre.abs().float().max().item() / 448.0, 1e-12)
            self.und_act_scales[i][0].fill_(s_und_qkv)
            for gname in ('q', 'k', 'v'):
                self.und_alphas[i][gname] = s_und_qkv * uws[gname]

            # FP8 norm → QKV GEMMs (use actual FP8 path)
            fvk.rms_norm_fp8(
                self.buf_x.data_ptr(), nm['in'].data_ptr(), self.act_fp8.data_ptr(),
                Sq, D, 1e-6, self.act_scales[i][0].data_ptr(), 0)
            self._fp8_gemm(i, 0, self.buf_q, Sq, D, D)
            self._fp8_gemm(i, 1, self.buf_k, Sq, K_DIM, D)
            self._fp8_gemm(i, 2, self.buf_v, Sq, K_DIM, D)

            # QK norm + RoPE + attention (BF16, same as forward_step)
            q = (self.buf_q + bi['q']).view(Sq, H, HD)
            k = (self.buf_k + bi['k']).view(Sq, KV_H, HD)
            v = self.buf_v + bi['v']
            q_var = q.float().pow(2).mean(-1, keepdim=True)
            q = (q.float() * torch.rsqrt(q_var + 1e-6)).to(bf16) * nm['qn']
            k_var = k.float().pow(2).mean(-1, keepdim=True)
            k = (k.float() * torch.rsqrt(k_var + 1e-6)).to(bf16) * nm['kn']
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            q = q.to(bf16); k = k.to(bf16); v = v.view(Sq, KV_H, HD).to(bf16)

            self._k_merged[i][kv_len:kv_len + Sq].copy_(k)
            self._v_merged[i][kv_len:kv_len + Sq].copy_(v)

            q_4d = q.unsqueeze(0).transpose(1, 2)
            k_4d = self._k_merged[i][:total_kv].unsqueeze(0).transpose(1, 2)
            v_4d = self._v_merged[i][:total_kv].unsqueeze(0).transpose(1, 2)
            ao = F.scaled_dot_product_attention(q_4d, k_4d, v_4d, is_causal=False,
                                                enable_gqa=True)
            ao = ao.transpose(1, 2).contiguous().view(Sq, D)

            # O scale + FP8 O GEMM
            amax_o = max(ao.abs().float().max().item(), 1e-12)
            s_o = max(amax_o / 448.0, 1e-12)
            self.act_scales[i][3].fill_(s_o)
            self.alphas[i][3] = s_o * ws['o']

            # Und O scale: amax of text rows of ao
            s_und_o = max(ao[ti].abs().float().max().item() / 448.0, 1e-12)
            self.und_act_scales[i][1].fill_(s_und_o)
            self.und_alphas[i]['o'] = s_und_o * uws['o']

            fvk.quantize_fp8_static(
                ao.data_ptr(), self.act_fp8.data_ptr(),
                self.act_scales[i][3].data_ptr(), Sq * D, 0)
            self._fp8_gemm(i, 3, self.buf_o_out, Sq, D, D)

            # Residual
            self.buf_x.copy_(self.buf_res + self.buf_o_out)
            self.buf_res.copy_(self.buf_x)

            # Post-attn norm for Gate+Up
            xn = rms_norm_bf16(self.buf_x, nm['pn'])
            amax_gu = max(xn.abs().float().max().item(), 1e-12)
            s_gu = max(amax_gu / 448.0, 1e-12)
            for j in [4, 5]:
                self.act_scales[i][j].fill_(s_gu)
                self.alphas[i][j] = s_gu * ws[GEMM_NAMES[j]]

            # Und gate/up scale: amax of und_pn(text_x_post_attn)
            xn_text = rms_norm_bf16(self.buf_x[ti], unm['pn'])
            s_und_gu = max(xn_text.abs().float().max().item() / 448.0, 1e-12)
            self.und_act_scales[i][2].fill_(s_und_gu)
            self.und_alphas[i]['gate'] = s_und_gu * uws['gate']
            self.und_alphas[i]['up']   = s_und_gu * uws['up']

            # FP8 Gate+Up GEMMs (calibrate uses temp buffers, runs once)
            fvk.rms_norm_fp8(
                self.buf_x.data_ptr(), nm['pn'].data_ptr(), self.act_fp8.data_ptr(),
                Sq, D, 1e-6, self.act_scales[i][4].data_ptr(), 0)
            _cal_gate = torch.empty(Sq, FFN, dtype=bf16, device=dev)
            _cal_up = torch.empty(Sq, FFN, dtype=bf16, device=dev)
            self._fp8_gemm(i, 4, _cal_gate, Sq, FFN, D)
            self._fp8_gemm(i, 5, _cal_up, Sq, FFN, D)

            # BF16 SiLU*mul for Down scale measurement
            silu_mul = F.silu(_cal_gate) * _cal_up
            amax_dn = max(silu_mul.abs().float().max().item(), 1e-12)
            s_dn = max(amax_dn / 448.0, 1e-12)
            self.act_scales[i][6].fill_(s_dn)
            self.alphas[i][6] = s_dn * ws['down']

            # Und down scale: simulate text FFN silu_mul (BF16, for scale-only)
            _t_gate = F.linear(xn_text, self.und_w[i]['gate'])
            _t_up   = F.linear(xn_text, self.und_w[i]['up'])
            _t_silu = F.silu(_t_gate) * _t_up
            s_und_down = max(_t_silu.abs().float().max().item() / 448.0, 1e-12)
            self.und_act_scales[i][3].fill_(s_und_down)
            self.und_alphas[i]['down'] = s_und_down * uws['down']

            # FP8 Down GEMM
            _cal_down = torch.empty(Sq, D, dtype=bf16, device=dev)
            fvk.quantize_fp8_static(
                silu_mul.data_ptr(), self.act_fp8.data_ptr(),
                self.act_scales[i][6].data_ptr(), Sq * FFN, 0)
            self._fp8_gemm(i, 6, _cal_down, Sq, D, FFN)

            # Residual → next layer input
            self.buf_x.copy_(self.buf_res + _cal_down)

        torch.cuda.synchronize()

        # ── Second pass: run the REAL MoT forward (forward_step, BF16 und path)
        # to observe text-row activation magnitudes along the correct trajectory.
        # The first pass above stays on the gen-only path so its text amax is
        # under-estimated for deeper layers — using those would clip text FP8
        # outputs catastrophically. This pass overwrites und_act_scales with
        # values measured on the true per-layer text path.
        self._recalibrate_und_scales(x_input)

        self.calibrated = True
        print(f"  Calibrated {N_LAYERS * 7} gen + {N_LAYERS * 4} und scales")

    @torch.no_grad()
    def _recalibrate_und_scales(self, x_input):
        """Observe per-layer text activation amaxes under the real MoT forward,
        then update self.und_act_scales + self.und_alphas."""
        Sq = self.Sq; dev = self.device
        ti = self._text_idx
        self.buf_x.copy_(x_input)

        for i in range(N_LAYERS):
            gnm = self.gen_norms[i]; gbi = self.gen_biases[i]; ws = self.gen_w_scales[i]
            unm = self.und_norms[i]; ubi = self.und_biases[i]
            uw = self.und_w[i]; uws = self.und_w_scales[i]

            # Measure text_qkv scale from und_ln(text_x)
            xn_text = rms_norm_bf16(self.buf_x[ti], unm['in'])
            s_qkv = max(xn_text.abs().float().max().item() / 448.0, 1e-12)
            self.und_act_scales[i][0].fill_(s_qkv)
            for g in ('q', 'k', 'v'):
                self.und_alphas[i][g] = s_qkv * uws[g]

            # Walk one full MoT layer forward (BF16 text + FP8 gen) so buf_x
            # evolves on the correct trajectory.
            self._fwd_step_one_layer(i, collect_text_amax=True)

    @torch.no_grad()
    def _fwd_step_one_layer(self, i, collect_text_amax=False):
        """One MoT layer in BF16/FP8. Optionally fills und_act_scales[i][1..3]
        with text-row amaxes for post-attn ao, post-attn norm, and silu_mul.
        Layer input read from self.buf_x; output written to self.buf_x."""
        Sq = self.Sq; M = self.M; dev = self.device
        kv_len = self.kv_len; total_kv = kv_len + Sq
        ti = self._text_idx; vi = self._vae_idx
        gnm = self.gen_norms[i]; gbi = self.gen_biases[i]
        unm = self.und_norms[i]; ubi = self.und_biases[i]
        uw = self.und_w[i]; uws = self.und_w_scales[i]

        self.buf_res.copy_(self.buf_x)

        # Full MoT attention path (condensed — matches forward_step)
        fvk.rms_norm_fp8(
            self.buf_x.data_ptr(), gnm['in'].data_ptr(), self.act_fp8.data_ptr(),
            Sq, D, 1e-6, self.act_scales[i][0].data_ptr(), 0)
        self._fp8_gemm(i, 0, self.buf_q, Sq, D, D)
        self._fp8_gemm(i, 1, self.buf_k, Sq, K_DIM, D)
        self._fp8_gemm(i, 2, self.buf_v, Sq, K_DIM, D)

        x_text = rms_norm_bf16(self.buf_res[ti], unm['in'])
        self.buf_q[ti] = F.linear(x_text, uw['q'], ubi['q'])
        self.buf_k[ti] = F.linear(x_text, uw['k'], ubi['k'])
        self.buf_v[ti] = F.linear(x_text, uw['v'], ubi['v'])

        q = self.buf_q.clone(); k = self.buf_k.clone(); v = self.buf_v.clone()
        q[vi] = q[vi] + gbi['q']; k[vi] = k[vi] + gbi['k']; v[vi] = v[vi] + gbi['v']
        q = q.view(Sq, H, HD); k = k.view(Sq, KV_H, HD); v = v.view(Sq, KV_H, HD)

        q_n = torch.empty_like(q); k_n = torch.empty_like(k)
        q_n[ti] = rms_norm_bf16(q[ti], unm['qn']); q_n[vi] = rms_norm_bf16(q[vi], gnm['qn'])
        k_n[ti] = rms_norm_bf16(k[ti], unm['kn']); k_n[vi] = rms_norm_bf16(k[vi], gnm['kn'])
        q_n, k_n = apply_rotary_pos_emb(q_n, k_n, self._rope_cos, self._rope_sin)
        q_n = q_n.to(bf16); k_n = k_n.to(bf16); v = v.to(bf16)

        self._k_merged[i][kv_len:kv_len + Sq].copy_(k_n)
        self._v_merged[i][kv_len:kv_len + Sq].copy_(v)

        q_4d = q_n.unsqueeze(0).transpose(1, 2)
        k_4d = self._k_merged[i][:total_kv].unsqueeze(0).transpose(1, 2)
        v_4d = self._v_merged[i][:total_kv].unsqueeze(0).transpose(1, 2)
        ao = F.scaled_dot_product_attention(q_4d, k_4d, v_4d, is_causal=False, enable_gqa=True)
        ao = ao.transpose(1, 2).contiguous().view(Sq, D)

        if collect_text_amax:
            s_o = max(ao[ti].abs().float().max().item() / 448.0, 1e-12)
            self.und_act_scales[i][1].fill_(s_o)
            self.und_alphas[i]['o'] = s_o * uws['o']

        # MoT O projection
        o_out = torch.empty_like(ao)
        o_out[ti] = F.linear(ao[ti], uw['o'])
        ao_vae = ao[vi].contiguous()
        o_vae = torch.empty(M, D, dtype=bf16, device=dev)
        fvk.quantize_fp8_static(ao_vae.data_ptr(), self.act_fp8.data_ptr(),
                                self.act_scales[i][3].data_ptr(), M * D, 0)
        fvk.cutlass_fp8_sq_bf16out(
            self.act_fp8.data_ptr(), self.gen_fp8_w[i]['o'].data_ptr(),
            o_vae.data_ptr(), M, D, D, self.alphas[i][3], 0.0, 0)
        o_out[vi] = o_vae
        self.buf_x.copy_(self.buf_res + o_out)

        # MoT FFN
        self.buf_res.copy_(self.buf_x)
        xn_text = rms_norm_bf16(self.buf_x[ti], unm['pn'])
        if collect_text_amax:
            s_gu = max(xn_text.abs().float().max().item() / 448.0, 1e-12)
            self.und_act_scales[i][2].fill_(s_gu)
            self.und_alphas[i]['gate'] = s_gu * uws['gate']
            self.und_alphas[i]['up']   = s_gu * uws['up']

        g_t = F.linear(xn_text, uw['gate']); u_t = F.linear(xn_text, uw['up'])
        silu_t = F.silu(g_t) * u_t
        if collect_text_amax:
            s_dn = max(silu_t.abs().float().max().item() / 448.0, 1e-12)
            self.und_act_scales[i][3].fill_(s_dn)
            self.und_alphas[i]['down'] = s_dn * uws['down']
        ffn_text = F.linear(silu_t, uw['down'])

        # VAE FFN via FP8 (same as forward_step VAE path)
        x_vae = self.buf_x[vi].contiguous()
        act_fp8_vae = torch.empty(M * FFN, dtype=FP8, device=dev)
        fvk.rms_norm_fp8(
            x_vae.data_ptr(), gnm['pn'].data_ptr(), act_fp8_vae.data_ptr(),
            M, D, 1e-6, self.act_scales[i][4].data_ptr(), 0)
        gate_vae = torch.empty(M, FFN, dtype=bf16, device=dev)
        up_vae = torch.empty(M, FFN, dtype=bf16, device=dev)
        fvk.cutlass_fp8_t1_bf16out(
            act_fp8_vae.data_ptr(), self.gen_fp8_w[i]['gate'].data_ptr(),
            gate_vae.data_ptr(), M, FFN, D, self.alphas[i][4], 0.0, 0)
        fvk.cutlass_fp8_t1_bf16out(
            act_fp8_vae.data_ptr(), self.gen_fp8_w[i]['up'].data_ptr(),
            up_vae.data_ptr(), M, FFN, D, self.alphas[i][5], 0.0, 0)
        silu_vae = F.silu(gate_vae) * up_vae
        fvk.quantize_fp8_static(silu_vae.data_ptr(), act_fp8_vae.data_ptr(),
                                self.act_scales[i][6].data_ptr(), M * FFN, 0)
        down_vae = torch.empty(M, D, dtype=bf16, device=dev)
        fvk.cutlass_fp8_wide_bf16out(
            act_fp8_vae.data_ptr(), self.gen_fp8_w[i]['down'].data_ptr(),
            down_vae.data_ptr(), M, D, FFN, self.alphas[i][6], 0.0, 0)
        ffn_out = torch.empty(Sq, D, dtype=bf16, device=dev)
        ffn_out[ti] = ffn_text; ffn_out[vi] = down_vae
        self.buf_x.copy_(self.buf_res + ffn_out)

    # ── FP8 Diffusion Step Forward ───────────────────────────────────────────

    def forward_step(self):
        """Full MoT 28-layer forward.
        Text tokens [0, Sq-1] → BF16 understanding expert.
        VAE tokens [1..M] → FP8 generation expert.
        Attention shared across all tokens.
        Verified: produces correct images (no-graph mode).
        """
        Sq = self.Sq; M = self.M
        kv_len = self.kv_len
        total_kv = kv_len + Sq
        ti = self._text_idx    # [0, Sq-1]
        vi = self._vae_idx     # [1..M]

        for i in range(N_LAYERS):
            gnm = self.gen_norms[i]; gbi = self.gen_biases[i]
            unm = self.und_norms[i]; ubi = self.und_biases[i]
            uw = self.und_w[i]

            self.buf_res.copy_(self.buf_x)

            # ── MoT LayerNorm + QKV ──
            # VAE: fused RMSNorm→FP8 (gen expert, full Sq — text rows get overwritten)
            fvk.rms_norm_fp8(
                self.buf_x.data_ptr(), gnm['in'].data_ptr(), self.act_fp8.data_ptr(),
                Sq, D, 1e-6, self.act_scales[i][0].data_ptr(), 0)
            self._fp8_gemm(i, 0, self.buf_q, Sq, D, D)
            self._fp8_gemm(i, 1, self.buf_k, Sq, K_DIM, D)
            self._fp8_gemm(i, 2, self.buf_v, Sq, K_DIM, D)

            # Text: BF16 und expert (overwrite text positions in buf_q/k/v)
            x_text = rms_norm_bf16(self.buf_res[ti], unm['in'])
            self.buf_q[ti] = F.linear(x_text, uw['q'], ubi['q'])
            self.buf_k[ti] = F.linear(x_text, uw['k'], ubi['k'])
            self.buf_v[ti] = F.linear(x_text, uw['v'], ubi['v'])

            # Bias + reshape (gen bias for VAE, text already has bias)
            q = self.buf_q.clone()
            k = self.buf_k.clone()
            v = self.buf_v.clone()
            q[vi] = q[vi] + gbi['q']
            k[vi] = k[vi] + gbi['k']
            v[vi] = v[vi] + gbi['v']
            q = q.view(Sq, H, HD); k = k.view(Sq, KV_H, HD); v = v.view(Sq, KV_H, HD)

            # ── MoT QK norm ──
            q_normed = torch.empty_like(q)
            k_normed = torch.empty_like(k)
            q_normed[ti] = rms_norm_bf16(q[ti], unm['qn'])
            q_normed[vi] = rms_norm_bf16(q[vi], gnm['qn'])
            k_normed[ti] = rms_norm_bf16(k[ti], unm['kn'])
            k_normed[vi] = rms_norm_bf16(k[vi], gnm['kn'])

            # RoPE
            q_normed, k_normed = apply_rotary_pos_emb(q_normed, k_normed,
                                                       self._rope_cos, self._rope_sin)
            q_normed = q_normed.to(bf16); k_normed = k_normed.to(bf16); v = v.to(bf16)

            # Write K,V to merged buffers
            self._k_merged[i][kv_len:kv_len + Sq].copy_(k_normed)
            self._v_merged[i][kv_len:kv_len + Sq].copy_(v)

            # ── Attention (shared, GQA) ──
            q_4d = q_normed.unsqueeze(0).transpose(1, 2)
            k_4d = self._k_merged[i][:total_kv].unsqueeze(0).transpose(1, 2)
            v_4d = self._v_merged[i][:total_kv].unsqueeze(0).transpose(1, 2)
            ao = F.scaled_dot_product_attention(q_4d, k_4d, v_4d, is_causal=False,
                                                enable_gqa=True)
            ao = ao.transpose(1, 2).contiguous().view(Sq, D)

            # ── MoT O projection ──
            o_out = torch.empty_like(ao)
            o_out[ti] = F.linear(ao[ti], uw['o'])
            ao_vae = ao[vi].contiguous()
            o_vae = torch.empty(M, D, dtype=bf16, device=self.device)
            fvk.quantize_fp8_static(
                ao_vae.data_ptr(), self.act_fp8.data_ptr(),
                self.act_scales[i][3].data_ptr(), M * D, 0)
            fvk.cutlass_fp8_sq_bf16out(
                self.act_fp8.data_ptr(), self.gen_fp8_w[i]['o'].data_ptr(),
                o_vae.data_ptr(), M, D, D, self.alphas[i][3], 0.0, 0)
            o_out[vi] = o_vae
            self.buf_x.copy_(self.buf_res + o_out)

            # ── MoT FFN ──
            self.buf_res.copy_(self.buf_x)
            # Text: BF16 und FFN
            xn_text = rms_norm_bf16(self.buf_x[ti], unm['pn'])
            g_t = F.linear(xn_text, uw['gate']); u_t = F.linear(xn_text, uw['up'])
            ffn_text = F.linear(F.silu(g_t) * u_t, uw['down'])
            # VAE: FP8 gen FFN
            x_vae = self.buf_x[vi].contiguous()
            act_fp8_vae = torch.empty(M * FFN, dtype=FP8, device=self.device)
            fvk.rms_norm_fp8(
                x_vae.data_ptr(), gnm['pn'].data_ptr(), act_fp8_vae.data_ptr(),
                M, D, 1e-6, self.act_scales[i][4].data_ptr(), 0)
            gate_vae = torch.empty(M, FFN, dtype=bf16, device=self.device)
            up_vae = torch.empty(M, FFN, dtype=bf16, device=self.device)
            fvk.cutlass_fp8_t1_bf16out(
                act_fp8_vae.data_ptr(), self.gen_fp8_w[i]['gate'].data_ptr(),
                gate_vae.data_ptr(), M, FFN, D, self.alphas[i][4], 0.0, 0)
            fvk.cutlass_fp8_t1_bf16out(
                act_fp8_vae.data_ptr(), self.gen_fp8_w[i]['up'].data_ptr(),
                up_vae.data_ptr(), M, FFN, D, self.alphas[i][5], 0.0, 0)
            silu_vae = F.silu(gate_vae) * up_vae
            fvk.quantize_fp8_static(
                silu_vae.data_ptr(), act_fp8_vae.data_ptr(),
                self.act_scales[i][6].data_ptr(), M * FFN, 0)
            down_vae = torch.empty(M, D, dtype=bf16, device=self.device)
            fvk.cutlass_fp8_wide_bf16out(
                act_fp8_vae.data_ptr(), self.gen_fp8_w[i]['down'].data_ptr(),
                down_vae.data_ptr(), M, D, FFN, self.alphas[i][6], 0.0, 0)
            # Combine
            ffn_out = torch.empty(Sq, D, dtype=bf16, device=self.device)
            ffn_out[ti] = ffn_text
            ffn_out[vi] = down_vae
            self.buf_x.copy_(self.buf_res + ffn_out)

        # ── MoT Final norm ──
        x_out = torch.empty_like(self.buf_x)
        var_t = self.buf_x[ti].float().pow(2).mean(-1, keepdim=True)
        x_out[ti] = (self.buf_x[ti].float() * torch.rsqrt(var_t + 1e-6)).to(bf16) * self.und_final_norm
        var_v = self.buf_x[vi].float().pow(2).mean(-1, keepdim=True)
        x_out[vi] = (self.buf_x[vi].float() * torch.rsqrt(var_v + 1e-6)).to(bf16) * self.gen_final_norm
        self.buf_x.copy_(x_out)

    # ── CKernel Graph helpers ──────────────────────────────────────────────
    #
    # The graph-safe hot-loop lives in CKernelBagel (csrc/ckernel_bagel.py).
    # BagelFP8Engine owns the Python-path `forward_step` (used for calibration
    # and no-graph fallback); CKernelBagel instances are built here per CFG
    # context and captured independently.

    def build_ckernel(self, kv_len, kv_cache=None, rope_pos=None):
        """Construct a CKernelBagel instance sharing this engine's weights.
        Copies calibrated act scales; optionally fills KV cache and sets RoPE.
        Returns the CKernelBagel (not yet graph-captured).
        """
        from ckernel_bagel import CKernelBagel
        ck = CKernelBagel.from_engine(self, Sq=self.Sq, kv_len=kv_len)
        ck.set_act_scales_from_engine(self)
        if rope_pos is not None:
            ck.set_rope_pos(rope_pos)
        if kv_cache is not None:
            ck.set_kv_cache(kv_cache)
        return ck

    def capture_ckernel_graph(self, ck):
        """Warmup + capture CUDA graph for ck.forward on a dedicated stream.
        Returns (graph, stream_handle) — caller keeps them alive.
        """
        assert self.calibrated, "Must calibrate before capturing graph"
        stream = torch.cuda.Stream()
        s_handle = stream.cuda_stream
        # Warmup (cuBLAS tactic selection, prevent capture-time allocations)
        with torch.cuda.stream(stream):
            for _ in range(3):
                ck.forward(s=s_handle)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream):
            g.capture_begin()
            ck.forward(s=s_handle)
            g.capture_end()
        torch.cuda.synchronize()
        return g, stream

    # ── Prepare diffusion input ──────────────────────────────────────────────
    #
    # Input preparation runs OUTSIDE the CUDA graph (before replay). PyTorch
    # ops are fine here — the 0-Python rule applies to the hot-loop forward.

    def _prepare_input(self, x_t, t_scalar, target=None):
        """Prepare packed input sequence for one diffusion step.
        x_t: [M, PATCH_DIM] — current noisy latent
        t_scalar: float — current timestep
        target: destination tensor [Sq, D] bf16. Defaults to self.buf_x.
        """
        if target is None:
            target = self.buf_x
        M = self.M

        t_emb = self.time_embed(t_scalar)  # [1, D]

        vae_pos_ids = self._vae_pos_ids
        pos_embed = self.latent_pos_embed[vae_pos_ids]  # [M, D]
        vae_hidden = F.linear(x_t.to(bf16), self.vae2llm_w, self.vae2llm_b)
        vae_hidden = vae_hidden + t_emb + pos_embed  # [M, D]

        start_embed = self.embed_tokens[self.special_tokens['start_of_image']]
        end_embed   = self.embed_tokens[self.special_tokens['end_of_image']]

        target[0].copy_(start_embed)
        target[1:M + 1].copy_(vae_hidden)
        target[M + 1].copy_(end_embed)

    def _prepare_rope(self, rope_pos):
        """Precompute RoPE cos/sin for the Python forward_step path (same pos for all query tokens)."""
        q_pos = torch.full((self.Sq,), rope_pos, dtype=torch.long, device=self.device)
        cos, sin = build_rope(q_pos)
        self._rope_cos = cos
        self._rope_sin = sin

    # ── Full Generation Pipeline ─────────────────────────────────────────────

    @torch.no_grad()
    def generate(self, text, image_h=448, image_w=448,
                 num_steps=24, timestep_shift=3.0,
                 cfg_text_scale=3.0, cfg_img_scale=1.5,
                 cfg_interval=(0.4, 1.0), cfg_renorm_min=0.0,
                 cfg_renorm_type="global",
                 use_graph=True, seed=42):
        """Generate a single image from text prompt.
        Returns: PIL Image.
        """
        M = self.M
        dev = self.device

        # ── 1. Precompute VAE position IDs ──
        self._vae_pos_ids = get_vae_position_ids(image_h, image_w).to(dev)

        # ── 2. Text Prefill → KV caches for CFG ──
        print("[Generate] Text prefill...")
        t0 = time.time()

        # Cond: full text context
        kv_cond, kv_len_cond, rope_cond = self.prefill_text(text)

        # Text-uncond: no text → empty KV cache (self-attention only)
        # We represent this as zero-length cache
        kv_len_text_uncond = 0
        rope_text_uncond = 0

        # Img-uncond: for text-only generation, same KV cache as cond
        kv_len_img_uncond = kv_len_cond
        rope_img_uncond = rope_cond

        print(f"  Prefill done: {time.time() - t0:.2f}s, kv_len={kv_len_cond}")

        # ── 3. Calibrate ──
        print("[Generate] Calibrating FP8 scales...")
        torch.manual_seed(seed)
        x_t = torch.randn(M, PATCH_DIM, dtype=bf16, device=dev)

        # Calibrate with cond context (largest kv_len, FP8-aware)
        self._prepare_input(x_t, 1.0)
        self.calibrate(self.buf_x.clone(), kv_cond)

        # ── 4. Build per-context state ──
        # Graph mode: each CFG context owns its own CKernelBagel instance
        #   (own KV-cache buffers, RoPE tables, cuBLAS handle, captured graph).
        # No-graph mode: engine shares a single set of KV-cache / RoPE buffers
        #   and runs forward_step (Python path) per context.

        contexts = {}

        # --- Cond context ---
        if use_graph:
            ck_cond = self.build_ckernel(kv_len=kv_len_cond,
                                         kv_cache=kv_cond, rope_pos=rope_cond)
            # Prime b_x with a dummy input before warmup (finite values only)
            self._prepare_input(x_t, 1.0, target=ck_cond.b_x)
            print("[Graph] Capturing cond graph...")
            g_cond, s_cond = self.capture_ckernel_graph(ck_cond)
            print("  Captured.")
            ctx_cond = {'mode': 'graph', 'ck': ck_cond,
                        'graph': g_cond, 'stream': s_cond,
                        'rope_pos': rope_cond}
        else:
            self._alloc_kv_merged(kv_len_cond)
            self.kv_len = kv_len_cond
            self._prepare_rope(rope_cond)
            self._fill_kv_cache(kv_cond)
            ctx_cond = {
                'mode': 'python',
                'kv_len': kv_len_cond, 'rope_pos': rope_cond,
                'k_merged': self._k_merged, 'v_merged': self._v_merged,
                'rope_cos': self._rope_cos, 'rope_sin': self._rope_sin,
            }

        contexts['cond'] = ctx_cond
        # img_uncond shares cond (same KV + rope for text-only generation)
        contexts['img_uncond'] = ctx_cond

        # --- Text-uncond context (kv_len=0) ---
        if cfg_text_scale > 1.0:
            if use_graph:
                ck_tunc = self.build_ckernel(kv_len=kv_len_text_uncond,
                                             kv_cache=None,
                                             rope_pos=rope_text_uncond)
                self._prepare_input(x_t, 1.0, target=ck_tunc.b_x)
                print("[Graph] Capturing text_uncond graph...")
                g_tunc, s_tunc = self.capture_ckernel_graph(ck_tunc)
                print("  Captured.")
                ctx_tunc = {'mode': 'graph', 'ck': ck_tunc,
                            'graph': g_tunc, 'stream': s_tunc,
                            'rope_pos': rope_text_uncond}
            else:
                self._alloc_kv_merged(kv_len_text_uncond)
                self.kv_len = kv_len_text_uncond
                self._prepare_rope(rope_text_uncond)
                ctx_tunc = {
                    'mode': 'python',
                    'kv_len': kv_len_text_uncond, 'rope_pos': rope_text_uncond,
                    'k_merged': self._k_merged, 'v_merged': self._v_merged,
                    'rope_cos': self._rope_cos, 'rope_sin': self._rope_sin,
                }
            contexts['text_uncond'] = ctx_tunc

        # ── 5. Diffusion Loop ──
        print(f"[Generate] Diffusion loop ({num_steps} steps)...")
        torch.manual_seed(seed)
        x_t = torch.randn(M, PATCH_DIM, dtype=bf16, device=dev)

        timesteps = torch.linspace(1, 0, num_steps, device=dev)
        timesteps = timestep_shift * timesteps / (1 + (timestep_shift - 1) * timesteps)
        dts = timesteps[:-1] - timesteps[1:]
        timesteps = timesteps[:-1]

        def _run_forward(ctx, x_t_cur, t_val):
            """Run one forward pass using the given context.
            Returns [M, PATCH_DIM] — llm2vae projection of VAE tokens."""
            if ctx['mode'] == 'graph':
                ck = ctx['ck']
                self._prepare_input(x_t_cur, t_val, target=ck.b_x)
                ctx['graph'].replay()
                # Read result BEFORE VAE tokens (rows 1..M+1 exclusive)
                return F.linear(ck.b_xn[1:M + 1].clone(),
                                self.llm2vae_w, self.llm2vae_b)
            else:
                # Python no-graph path: swap engine state + run forward_step
                self.kv_len = ctx['kv_len']
                self._k_merged = ctx['k_merged']
                self._v_merged = ctx['v_merged']
                self._rope_cos = ctx['rope_cos']
                self._rope_sin = ctx['rope_sin']
                self._prepare_input(x_t_cur, t_val)
                self.forward_step()
                return F.linear(self.buf_x[1:M + 1].clone(),
                                self.llm2vae_w, self.llm2vae_b)

        torch.cuda.synchronize()
        t_loop_start = time.time()
        n_fwd = 0

        for step_idx in range(len(timesteps)):
            t = timesteps[step_idx].item()
            dt = dts[step_idx].item()

            # Determine CFG scales for this step
            do_cfg = (t > cfg_interval[0] and t <= cfg_interval[1])
            cfg_text_s = cfg_text_scale if do_cfg else 1.0
            cfg_img_s = cfg_img_scale if do_cfg else 1.0

            # ── Cond forward ──
            v_cond = _run_forward(contexts['cond'], x_t, t)
            n_fwd += 1

            # ── CFG: text_uncond forward ──
            if cfg_text_s > 1.0:
                v_text_uncond = _run_forward(contexts['text_uncond'], x_t, t)
                n_fwd += 1

            # ── CFG: img_uncond forward ──
            # For text-only generation, img_uncond == cond (same context),
            # so skip the redundant forward and reuse v_cond.
            if cfg_img_s > 1.0 and contexts['img_uncond'] is not contexts['cond']:
                v_img_uncond = _run_forward(contexts['img_uncond'], x_t, t)
                n_fwd += 1
            else:
                v_img_uncond = v_cond  # skip: img_uncond == cond for text-only

            # ── CFG combination ──
            if cfg_text_s > 1.0:
                v_text_ = v_text_uncond + cfg_text_s * (v_cond - v_text_uncond)
                if cfg_img_s > 1.0:
                    v_t = v_img_uncond + cfg_img_s * (v_text_ - v_img_uncond)
                else:
                    v_t = v_text_

                # Renorm
                if cfg_renorm_type == "text_channel":
                    norm_orig = torch.norm(v_cond, dim=-1, keepdim=True)
                    norm_text = torch.norm(v_text_, dim=-1, keepdim=True)
                    scale_t = (norm_orig / (norm_text + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                    v_text_renorm = v_text_ * scale_t
                    if cfg_img_s > 1.0:
                        v_t = v_img_uncond + cfg_img_s * (v_text_renorm - v_img_uncond)
                    else:
                        v_t = v_text_renorm
                elif cfg_renorm_type == "global":
                    norm_orig = torch.norm(v_cond)
                    norm_cfg = torch.norm(v_t)
                    scale = (norm_orig / (norm_cfg + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                    v_t = v_t * scale
                elif cfg_renorm_type == "channel":
                    norm_orig = torch.norm(v_cond, dim=-1, keepdim=True)
                    norm_cfg = torch.norm(v_t, dim=-1, keepdim=True)
                    scale = (norm_orig / (norm_cfg + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                    v_t = v_t * scale
            else:
                v_t = v_cond

            # Euler step
            x_t = x_t - v_t * dt

        torch.cuda.synchronize()
        t_loop = time.time() - t_loop_start
        print(f"  Diffusion: {t_loop:.2f}s ({n_fwd} forwards, "
              f"{t_loop / len(timesteps) * 1000:.1f}ms/step avg)")

        # ── 7. VAE Decode ──
        print("[Generate] VAE decode...")
        image = self.decode_latent(x_t, image_h, image_w)
        print(f"  Done.")

        return image

    # ── VAE Decode ───────────────────────────────────────────────────────────

    def decode_latent(self, latent, H, W):
        """Decode latent [M, PATCH_DIM] → PIL Image.
        latent: [M, 64] where M = (H/16)*(W/16)
        """
        h, w = H // LATENT_DOWNSAMPLE, W // LATENT_DOWNSAMPLE
        p = LATENT_PATCH_SIZE

        # Reshape: [M, p*p*C] → [1, C, h*p, w*p]
        lat = latent.reshape(1, h, w, p, p, LATENT_CH)
        lat = torch.einsum("nhwpqc->nchpwq", lat)
        lat = lat.reshape(1, LATENT_CH, h * p, w * p)

        # VAE decode (FP32)
        with torch.autocast(device_type="cuda", enabled=False):
            img = self.vae_model.decode(lat.float().cuda())

        img = (img * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255
        return Image.fromarray(img.to(torch.uint8).cpu().numpy())

    # ── Multi-view Generation ────────────────────────────────────────────────

    def generate_multiview(self, text, n_views=3, image_h=448, image_w=448,
                           num_steps=24, **kwargs):
        """Generate n_views images sequentially.
        Note: Each call to generate() does its own prefill/calibrate/capture.
        For production, these should be cached across views (same prompt → same setup).
        Returns list of PIL Images.
        """
        images = []
        for v in range(n_views):
            print(f"\n{'='*60}")
            print(f"View {v + 1}/{n_views}")
            print(f"{'='*60}")
            img = self.generate(text, image_h, image_w, num_steps,
                                seed=42 + v, **kwargs)
            images.append(img)
        return images


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="<bagel_weights>")
    parser.add_argument("--prompt", default="A robot arm reaching for a red cup on a wooden table, front camera view, photorealistic")
    parser.add_argument("--size", type=int, default=448)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--views", type=int, default=1)
    parser.add_argument("--no-graph", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    H = W = args.size
    M = (H // LATENT_DOWNSAMPLE) * (W // LATENT_DOWNSAMPLE)

    print("=" * 60)
    print("BAGEL FP8 Engine — Complete Pipeline")
    print(f"  Image: {H}×{W}, M={M}, Steps={args.steps}, Views={args.views}")
    print("=" * 60)

    engine = BagelFP8Engine(args.weights, M=M)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(out_dir, exist_ok=True)

    if args.views == 1:
        torch.cuda.synchronize()
        t0 = time.time()
        img = engine.generate(
            args.prompt,
            image_h=H, image_w=W,
            num_steps=args.steps,
            use_graph=not args.no_graph,
            seed=args.seed,
        )
        torch.cuda.synchronize()
        t_total = time.time() - t0

        path = os.path.join(out_dir, f"fp8_engine_{H}_{args.steps}step.png")
        img.save(path)
        print(f"\nSaved: {path}")
        print(f"Total: {t_total:.2f}s")
    else:
        torch.cuda.synchronize()
        t0 = time.time()
        images = engine.generate_multiview(
            args.prompt,
            n_views=args.views,
            image_h=H, image_w=W,
            num_steps=args.steps,
            use_graph=not args.no_graph,
        )
        torch.cuda.synchronize()
        t_total = time.time() - t0

        for v, img in enumerate(images):
            path = os.path.join(out_dir, f"fp8_engine_view{v}_{H}_{args.steps}step.png")
            img.save(path)
            print(f"Saved: {path}")

        print(f"\nTotal {args.views} views: {t_total:.2f}s")
        print(f"Per view: {t_total / args.views:.2f}s")

    # Summary
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"  GPU memory: {torch.cuda.memory_allocated() / 1e9:.1f}GB")
    if args.views > 1:
        target = 4.0
        print(f"  {args.views} views: {t_total:.2f}s → "
              f"{'PASS' if t_total < target else 'NEED MORE'} (<{target}s target)")


if __name__ == "__main__":
    main()

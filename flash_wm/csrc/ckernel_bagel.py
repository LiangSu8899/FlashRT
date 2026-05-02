"""
CKernelBagel — Pure C-kernel 28-layer full-MoT forward for BAGEL.

All operations via flash_rt_kernels (BF16 native) and flash_wm_kernels (BF16 ports).
Zero PyTorch tensor creation / arithmetic in forward(). CUDA Graph safe.

Modeled after CKernelQwen3 (flash_rt/models/groot/pipeline_thor.py).

Architecture: Qwen2.5-7B MoT
  D=3584, H=28(Q)/4(KV), HD=128, FFN=18944, L=28
  GQA 28:4 → repeat KV 4→28
  Full MoT: text tokens [0, Sq-1] use understanding expert for
    input_layernorm, Q/K/V/O, QK norm, post_attn_layernorm, gate/up/down, final_norm.
    VAE tokens [1..M] use generation expert (FP8).
  Attention is shared (unified QKV buffer across all Sq tokens).
  KV cache: prefilled text context (total_kv = kv_len + Sq)
"""
import os, sys, math, torch
from safetensors import safe_open

_WM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VLA_DIR = os.path.dirname(_WM_DIR)
sys.path.insert(0, _VLA_DIR)
sys.path.insert(0, _WM_DIR)

import flash_rt.flash_rt_kernels as fvk
import flash_wm_kernels as fwk
from flash_rt.core.thor_frontend_utils import quant_fp8
from flash_attn import flash_attn_func  # native GQA, ~6x faster than cuBLAS

bf16 = torch.bfloat16
FP8 = torch.float8_e4m3fn

# ── Constants ──
D = 3584; NHQ = 28; NHKV = 4; HD = 128; FFN = 18944; L = 28
K_DIM = NHKV * HD  # 512


class CKernelBagel:
    """28-layer full-MoT forward — pure kernel calls, CUDA Graph safe."""

    # Gen expert weight keys used for scale lookup
    _GEN_GEMM_NAMES = ['q', 'k', 'v', 'o', 'gate', 'up', 'down']

    def __init__(self, Sq, kv_len=0, weights_path=None, engine=None):
        """
        Either weights_path (disk load) or engine (reuse preloaded tensors) must be given.

        Sq: query sequence length (M+2 = 786 for 448x448)
        kv_len: KV cache length from text prefill
        """
        assert (weights_path is None) ^ (engine is None), \
            "Provide exactly one of weights_path or engine"

        self.Sq = Sq; self.M = Sq - 2; self.kv_len = kv_len
        self.total_kv = kv_len + Sq
        self.S_kv_pad = ((self.total_kv + 7) // 8) * 8

        self.wm_ctx = fwk.WmContext()
        self.gemm = fvk.GemmRunner()

        # Per-layer act scales: 7 per layer (q, k, v, o, gate, up, down) for gen path.
        self.act_scales = torch.ones(L * 7, dtype=torch.float32, device='cuda')
        # Per-layer und (text) act scales: 4 per layer
        #   [0] und_ln(text_x)     → Q/K/V    [1] text(ao)      → O
        #   [2] und_pn(text_post)  → gate/up  [3] text silu_mul → down
        self.und_act_scales = torch.ones(L * 4, dtype=torch.float32, device='cuda')

        if engine is not None:
            self._load_from_engine(engine)
        else:
            self._load_weights_from_disk(weights_path)

        self._precompute_rope()
        self._alloc_buffers()

    # ── Public constructors ────────────────────────────────────────

    @classmethod
    def from_engine(cls, engine, Sq, kv_len):
        """Build CKernelBagel reusing BagelFP8Engine's preloaded weights.
        No extra weight allocation; all tensors are shared by reference.
        """
        return cls(Sq=Sq, kv_len=kv_len, engine=engine)

    # ── Scale / KV setters ─────────────────────────────────────────

    def set_act_scales_from_engine(self, engine):
        """Copy gen + und act scales from engine and precompute alphas.

        gen: act_scales[i][0..6]                    → self.act_scales   [L*7]
        und: und_act_scales[i][0..3]                → self.und_act_scales [L*4]
             → und_alphas[i][gname] = und_scale * und_w_scale (host floats)
        """
        flat = [float(engine.act_scales[i][j].item())
                for i in range(L) for j in range(7)]
        self.act_scales.copy_(torch.tensor(flat, dtype=torch.float32, device='cuda'))
        und_flat = [float(engine.und_act_scales[i][j].item())
                    for i in range(L) for j in range(4)]
        self.und_act_scales.copy_(torch.tensor(und_flat, dtype=torch.float32, device='cuda'))

        self._alphas = []
        self._und_alphas = []
        for i in range(L):
            layer_alphas = {}
            for j, gname in enumerate(self._GEN_GEMM_NAMES):
                act_s = flat[i * 7 + j]
                w_s = self.layers[i][f'{gname}_s'].item()
                layer_alphas[gname] = float(act_s * w_s)
            self._alphas.append(layer_alphas)
            # Und alphas: 4 scales map to 7 weights
            s_qkv   = und_flat[i * 4 + 0]
            s_o     = und_flat[i * 4 + 1]
            s_gu    = und_flat[i * 4 + 2]
            s_down  = und_flat[i * 4 + 3]
            und = {}
            und['q']    = s_qkv  * self.layers[i]['und_q_s'].item()
            und['k']    = s_qkv  * self.layers[i]['und_k_s'].item()
            und['v']    = s_qkv  * self.layers[i]['und_v_s'].item()
            und['o']    = s_o    * self.layers[i]['und_o_s'].item()
            und['gate'] = s_gu   * self.layers[i]['und_gate_s'].item()
            und['up']   = s_gu   * self.layers[i]['und_up_s'].item()
            und['down'] = s_down * self.layers[i]['und_down_s'].item()
            self._und_alphas.append(und)

    def set_act_scales(self, scales_list, und_scales_list=None):
        """Set gen scales (L*7) and optionally und text scales (L*4).
        Precomputes host alpha tables for both."""
        self.act_scales.copy_(torch.tensor(scales_list, dtype=torch.float32, device='cuda'))
        self._alphas = []
        for i in range(L):
            layer_alphas = {}
            for j, gname in enumerate(self._GEN_GEMM_NAMES):
                act_s = scales_list[i * 7 + j]
                w_s = self.layers[i][f'{gname}_s'].item()
                layer_alphas[gname] = float(act_s * w_s)
            self._alphas.append(layer_alphas)
        if und_scales_list is None:
            und_scales_list = [1.0] * (L * 4)
        self.und_act_scales.copy_(torch.tensor(und_scales_list,
                                                dtype=torch.float32, device='cuda'))
        self._und_alphas = []
        for i in range(L):
            s_qkv  = und_scales_list[i * 4 + 0]
            s_o    = und_scales_list[i * 4 + 1]
            s_gu   = und_scales_list[i * 4 + 2]
            s_down = und_scales_list[i * 4 + 3]
            und = {
                'q':    s_qkv  * self.layers[i]['und_q_s'].item(),
                'k':    s_qkv  * self.layers[i]['und_k_s'].item(),
                'v':    s_qkv  * self.layers[i]['und_v_s'].item(),
                'o':    s_o    * self.layers[i]['und_o_s'].item(),
                'gate': s_gu   * self.layers[i]['und_gate_s'].item(),
                'up':   s_gu   * self.layers[i]['und_up_s'].item(),
                'down': s_down * self.layers[i]['und_down_s'].item(),
            }
            self._und_alphas.append(und)

    def set_kv_cache(self, kv_cache):
        """Copy prefilled KV cache into merged buffers. kv_cache: list of (K, V) per layer."""
        kv_len = self.kv_len
        for i in range(L):
            k_flat = kv_cache[i][0].reshape(kv_len, K_DIM).contiguous()
            v_flat = kv_cache[i][1].reshape(kv_len, K_DIM).contiguous()
            fvk.gpu_copy(self.b_k_merged[i].data_ptr(),
                         k_flat.data_ptr(), kv_len * K_DIM * 2, 0)
            fvk.gpu_copy(self.b_v_merged[i].data_ptr(),
                         v_flat.data_ptr(), kv_len * K_DIM * 2, 0)
        torch.cuda.synchronize()

    # ── Weight loading: disk path (standalone) ──────────────────────

    def _load_weights_from_disk(self, wp):
        sf = safe_open(os.path.join(wp, "ema.safetensors"),
                       framework='pt', device='cpu')
        self.layers = []
        for i in range(L):
            p = f"language_model.model.layers.{i}"
            def _bf16(n): return sf.get_tensor(f"{p}.{n}").to(bf16).cuda().contiguous()
            def _fp16(n): return sf.get_tensor(f"{p}.{n}").to(torch.float16).cuda().contiguous()

            w = {}
            # Gen expert FP8 (Q/K/V/O/Gate/Up/Down)
            for gname, wname in [
                ('q', 'self_attn.q_proj_moe_gen.weight'),
                ('k', 'self_attn.k_proj_moe_gen.weight'),
                ('v', 'self_attn.v_proj_moe_gen.weight'),
                ('o', 'self_attn.o_proj_moe_gen.weight'),
                ('gate', 'mlp_moe_gen.gate_proj.weight'),
                ('up', 'mlp_moe_gen.up_proj.weight'),
                ('down', 'mlp_moe_gen.down_proj.weight'),
            ]:
                fp8_w, ws = quant_fp8(_fp16(wname))
                w[f'{gname}_fp8'] = fp8_w
                w[f'{gname}_s'] = torch.tensor([ws], dtype=torch.float32, device='cuda')

            # Gen BF16: norms + biases
            w['ln_w']  = _bf16("input_layernorm_moe_gen.weight")
            w['ln2_w'] = _bf16("post_attention_layernorm_moe_gen.weight")
            w['qn_w']  = _bf16("self_attn.q_norm_moe_gen.weight")
            w['kn_w']  = _bf16("self_attn.k_norm_moe_gen.weight")
            w['q_bias'] = _bf16("self_attn.q_proj_moe_gen.bias")
            w['k_bias'] = _bf16("self_attn.k_proj_moe_gen.bias")
            w['v_bias'] = _bf16("self_attn.v_proj_moe_gen.bias")

            # Und expert BF16 norms + biases
            w['und_ln_w']   = _bf16("input_layernorm.weight")
            w['und_pn_w']   = _bf16("post_attention_layernorm.weight")
            w['und_qn_w']   = _bf16("self_attn.q_norm.weight")
            w['und_kn_w']   = _bf16("self_attn.k_norm.weight")
            w['und_q_bias'] = _bf16("self_attn.q_proj.bias")
            w['und_k_bias'] = _bf16("self_attn.k_proj.bias")
            w['und_v_bias'] = _bf16("self_attn.v_proj.bias")

            # Und expert FP8 weights (text hot path)
            for gname, wname in [
                ('q', 'self_attn.q_proj.weight'),
                ('k', 'self_attn.k_proj.weight'),
                ('v', 'self_attn.v_proj.weight'),
                ('o', 'self_attn.o_proj.weight'),
                ('gate', 'mlp.gate_proj.weight'),
                ('up', 'mlp.up_proj.weight'),
                ('down', 'mlp.down_proj.weight'),
            ]:
                fp8_w, ws = quant_fp8(_fp16(wname))
                w[f'und_{gname}_fp8'] = fp8_w
                w[f'und_{gname}_s']   = torch.tensor([ws], dtype=torch.float32, device='cuda')

            self.layers.append(w)
            if (i + 1) % 7 == 0:
                print(f"  CKernelBagel: {i + 1}/{L}")

        self.gen_final_norm = sf.get_tensor(
            "language_model.model.norm_moe_gen.weight").to(bf16).cuda()
        self.und_final_norm = sf.get_tensor(
            "language_model.model.norm.weight").to(bf16).cuda()
        del sf

    # ── Weight loading: share engine's preloaded tensors ────────────

    def _load_from_engine(self, engine):
        """Zero-copy reference to engine.gen_fp8_w / gen_w_scales / gen_norms /
        gen_biases / und_w / und_norms / und_biases. Avoids loading a 2nd copy
        of the 6.7GB FP8 weight set.
        """
        self.layers = []
        for i in range(L):
            gfp8 = engine.gen_fp8_w[i]
            gws  = engine.gen_w_scales[i]
            gnm  = engine.gen_norms[i]
            gbi  = engine.gen_biases[i]
            uw   = engine.und_w[i]
            unm  = engine.und_norms[i]
            ubi  = engine.und_biases[i]

            w = {}
            # Gen FP8 weights + scales
            for gname in self._GEN_GEMM_NAMES:
                w[f'{gname}_fp8'] = gfp8[gname]
                ws_val = gws[gname]
                if torch.is_tensor(ws_val):
                    w[f'{gname}_s'] = ws_val.to(torch.float32).view(1).cuda().contiguous()
                else:
                    w[f'{gname}_s'] = torch.tensor([float(ws_val)],
                                                   dtype=torch.float32, device='cuda')

            # Gen BF16 norms/biases
            w['ln_w']  = gnm['in']
            w['ln2_w'] = gnm['pn']
            w['qn_w']  = gnm['qn']
            w['kn_w']  = gnm['kn']
            w['q_bias'] = gbi['q']
            w['k_bias'] = gbi['k']
            w['v_bias'] = gbi['v']

            # Und BF16 norms/biases
            w['und_ln_w']   = unm['in']
            w['und_pn_w']   = unm['pn']
            w['und_qn_w']   = unm['qn']
            w['und_kn_w']   = unm['kn']
            w['und_q_bias'] = ubi['q']
            w['und_k_bias'] = ubi['k']
            w['und_v_bias'] = ubi['v']

            # Und FP8 weights + scales (text hot path)
            ufp8 = engine.und_fp8_w[i]
            uws  = engine.und_w_scales[i]
            for gname in ('q', 'k', 'v', 'o', 'gate', 'up', 'down'):
                w[f'und_{gname}_fp8'] = ufp8[gname]
                w[f'und_{gname}_s']   = torch.tensor(
                    [float(uws[gname])], dtype=torch.float32, device='cuda')

            self.layers.append(w)

        self.gen_final_norm = engine.gen_final_norm
        self.und_final_norm = engine.und_final_norm

    # ── RoPE precompute ─────────────────────────────────────────────

    def _precompute_rope(self, rope_pos=0):
        """Precompute cos/sin tables for fixed RoPE position (all query tokens share position)."""
        theta = 10000.0
        freqs = 1.0 / (theta ** (torch.arange(0, HD, 2,
                        dtype=torch.float32, device='cuda') / HD))
        pos = torch.full((self.Sq,), rope_pos, dtype=torch.float32, device='cuda')
        angles = torch.outer(pos, freqs)  # [Sq, HD//2]
        # rope_rotate_half_bf16 expects [S, HD] with cos/sin in first HD//2
        self.cos_table = torch.zeros(self.Sq, HD, dtype=bf16, device='cuda')
        self.sin_table = torch.zeros(self.Sq, HD, dtype=bf16, device='cuda')
        self.cos_table[:, :HD // 2] = angles.cos().to(bf16)
        self.sin_table[:, :HD // 2] = angles.sin().to(bf16)

    def set_rope_pos(self, rope_pos):
        self._precompute_rope(rope_pos)

    # ── Buffer allocation ───────────────────────────────────────────

    def _alloc_buffers(self):
        Sq = self.Sq; M = self.M
        total_kv = self.total_kv
        S_kv_pad = self.S_kv_pad

        # Main state
        self.b_x  = torch.empty(Sq, D, dtype=bf16, device='cuda')
        self.b_xn = torch.empty(Sq, D, dtype=bf16, device='cuda')
        # Shared FP8 scratch (max(Sq*D, Sq*K_DIM, Sq*FFN) → Sq*FFN)
        self.b_fp8 = torch.empty(Sq * FFN, dtype=torch.uint8, device='cuda')
        # Dedicated text FP8 scratch (max(2*D, 2*FFN) = 2*FFN = ~37 KB)
        self.b_text_fp8 = torch.empty(2 * FFN, dtype=torch.uint8, device='cuda')
        # QKV (gen expert, full Sq — text rows overwritten in-place)
        self.b_q  = torch.empty(Sq, D,     dtype=bf16, device='cuda')   # NHQ*HD = D
        self.b_k  = torch.empty(Sq, K_DIM, dtype=bf16, device='cuda')
        self.b_v  = torch.empty(Sq, K_DIM, dtype=bf16, device='cuda')
        # Text scratch (2 rows contiguous) — und expert parallel path
        self.b_text_x    = torch.empty(2, D,     dtype=bf16, device='cuda')  # input-norm OR post-attn-norm
        self.b_text_q    = torch.empty(2, D,     dtype=bf16, device='cuda')
        self.b_text_k    = torch.empty(2, K_DIM, dtype=bf16, device='cuda')
        self.b_text_v    = torch.empty(2, K_DIM, dtype=bf16, device='cuda')
        self.b_text_ao   = torch.empty(2, D,     dtype=bf16, device='cuda')
        self.b_text_o    = torch.empty(2, D,     dtype=bf16, device='cuda')
        self.b_text_gate = torch.empty(2, FFN,   dtype=bf16, device='cuda')
        self.b_text_up   = torch.empty(2, FFN,   dtype=bf16, device='cuda')
        self.b_text_silu = torch.empty(2, FFN,   dtype=bf16, device='cuda')
        self.b_text_down = torch.empty(2, D,     dtype=bf16, device='cuda')
        # Flash-attn native GQA: no b_k_exp / b_v_exp / b_logits needed.
        # The attention output tensor is allocated by flash_attn_func; PyTorch's
        # caching allocator keeps its pointer stable across graph replays.
        self.b_o = torch.empty(Sq, D, dtype=bf16, device='cuda')
        # FFN (gen)
        self.b_gate = torch.empty(Sq, FFN, dtype=bf16, device='cuda')
        self.b_up   = torch.empty(Sq, FFN, dtype=bf16, device='cuda')
        self.b_down = torch.empty(Sq, D,   dtype=bf16, device='cuda')
        # Per-layer merged KV (kv_cache + current)
        self.b_k_merged = [torch.empty(total_kv, K_DIM, dtype=bf16, device='cuda')
                           for _ in range(L)]
        self.b_v_merged = [torch.empty(total_kv, K_DIM, dtype=bf16, device='cuda')
                           for _ in range(L)]

    # ── Forward (pure kernel calls, Graph safe) ─────────────────────

    def forward(self, s=0):
        """28-layer full-MoT forward. Input in self.b_x. Output in self.b_xn.
        All ops via fvk/fwk kernel calls. Zero PyTorch tensor arithmetic.

        F2: The gen input_layernorm (ln_w) for layer i>0 is fused into the
        previous layer's post-FFN residual. Likewise the gen post-attn norm
        (ln2_w) is fused into the post-attn residual. Each fusion collapses
        residual_add + rms_norm_fp8 into residual_add_rms_norm_fp8, saving
        one kernel launch and one b_x write-back per pair.
        """
        Sq = self.Sq
        kv_len = self.kv_len
        total_kv = self.total_kv
        S_kv_pad = self.S_kv_pad
        scale = 1.0 / math.sqrt(HD)
        # Byte offsets for text-row slicing
        LAST_D    = (Sq - 1) * D * 2
        LAST_KDIM = (Sq - 1) * K_DIM * 2
        act_scales_base = self.act_scales.data_ptr()

        # F2: initial input norm for layer 0 (subsequent layers' input norms
        # are fused into the previous layer's post-FFN residual).
        fvk.rms_norm_fp8(
            self.b_x.data_ptr(), self.layers[0]['ln_w'].data_ptr(),
            self.b_fp8.data_ptr(), Sq, D, 1e-6,
            act_scales_base + 0 * 4, s)

        for i in range(L):
            w = self.layers[i]
            a = self._alphas[i]
            def _asp(j): return act_scales_base + (i * 7 + j) * 4

            # Gen Q/K/V (FP8 with fused per-col bias epilogue — Class-1a).
            fwk.cutlass_fp8_sq_bias_bf16out(
                self.b_fp8.data_ptr(), w['q_fp8'].data_ptr(), w['q_bias'].data_ptr(),
                self.b_q.data_ptr(), Sq, D, D, a['q'], s)
            fwk.cutlass_fp8_sq_bias_bf16out(
                self.b_fp8.data_ptr(), w['k_fp8'].data_ptr(), w['k_bias'].data_ptr(),
                self.b_k.data_ptr(), Sq, K_DIM, D, a['k'], s)
            fwk.cutlass_fp8_sq_bias_bf16out(
                self.b_fp8.data_ptr(), w['v_fp8'].data_ptr(), w['v_bias'].data_ptr(),
                self.b_v.data_ptr(), Sq, K_DIM, D, a['v'], s)

            # Class D: fused per-head rms_norm + RoPE on gen Q/K.
            fwk.qk_rmsnorm_rope_fused_bf16(
                self.b_q.data_ptr(), w['qn_w'].data_ptr(),
                self.cos_table.data_ptr(), self.sin_table.data_ptr(),
                Sq, NHQ, HD, 1e-6, s)
            fwk.qk_rmsnorm_rope_fused_bf16(
                self.b_k.data_ptr(), w['kn_w'].data_ptr(),
                self.cos_table.data_ptr(), self.sin_table.data_ptr(),
                Sq, NHKV, HD, 1e-6, s)

            # ── Und expert text Q/K/V (FP8 GEMMs at M=2) ──
            # Skipped when self.skip_und: text rows inherit gen Q/K/V.
            skip_und = getattr(self, 'skip_und', False)
            if not skip_und:
                ua = self._und_alphas[i]
                und_asp = self.und_act_scales.data_ptr() + (i * 4) * 4  # [0] = QKV input
                fvk.gpu_copy(self.b_text_x.data_ptr(),
                             self.b_x.data_ptr(),              D * 2, s)
                fvk.gpu_copy(self.b_text_x.data_ptr() + D * 2,
                             self.b_x.data_ptr() + LAST_D,     D * 2, s)
                fvk.rms_norm(self.b_text_x.data_ptr(), w['und_ln_w'].data_ptr(),
                             self.b_text_x.data_ptr(), 2, D, 1e-6, s)
                # Quantize normed text [2, D] → b_text_fp8 [2*D]
                fvk.quantize_fp8_static(
                    self.b_text_x.data_ptr(), self.b_text_fp8.data_ptr(),
                    und_asp, 2 * D, s)
                fwk.cutlass_fp8_sq_bias_bf16out(
                    self.b_text_fp8.data_ptr(), w['und_q_fp8'].data_ptr(), w['und_q_bias'].data_ptr(),
                    self.b_text_q.data_ptr(), 2, D,     D, ua['q'], s)
                fwk.cutlass_fp8_sq_bias_bf16out(
                    self.b_text_fp8.data_ptr(), w['und_k_fp8'].data_ptr(), w['und_k_bias'].data_ptr(),
                    self.b_text_k.data_ptr(), 2, K_DIM, D, ua['k'], s)
                fwk.cutlass_fp8_sq_bias_bf16out(
                    self.b_text_fp8.data_ptr(), w['und_v_fp8'].data_ptr(), w['und_v_bias'].data_ptr(),
                    self.b_text_v.data_ptr(), 2, K_DIM, D, ua['v'], s)

                # Class D: fused rms_norm + RoPE on und text rows (M=2).
                fwk.qk_rmsnorm_rope_fused_bf16(
                    self.b_text_q.data_ptr(), w['und_qn_w'].data_ptr(),
                    self.cos_table.data_ptr(), self.sin_table.data_ptr(),
                    2, NHQ, HD, 1e-6, s)
                fwk.qk_rmsnorm_rope_fused_bf16(
                    self.b_text_k.data_ptr(), w['und_kn_w'].data_ptr(),
                    self.cos_table.data_ptr(), self.sin_table.data_ptr(),
                    2, NHKV, HD, 1e-6, s)

                # Overwrite text rows in gen-path QKV
                fvk.gpu_copy(self.b_q.data_ptr(),
                             self.b_text_q.data_ptr(),               D * 2, s)
                fvk.gpu_copy(self.b_q.data_ptr() + LAST_D,
                             self.b_text_q.data_ptr() + D * 2,       D * 2, s)
                fvk.gpu_copy(self.b_k.data_ptr(),
                             self.b_text_k.data_ptr(),               K_DIM * 2, s)
                fvk.gpu_copy(self.b_k.data_ptr() + LAST_KDIM,
                             self.b_text_k.data_ptr() + K_DIM * 2,   K_DIM * 2, s)
                fvk.gpu_copy(self.b_v.data_ptr(),
                             self.b_text_v.data_ptr(),               K_DIM * 2, s)
                fvk.gpu_copy(self.b_v.data_ptr() + LAST_KDIM,
                             self.b_text_v.data_ptr() + K_DIM * 2,   K_DIM * 2, s)

            # RoPE already applied inside qk_rmsnorm_rope_fused_bf16 (Class D);
            # und path's fused kernel pre-rotates text rows before scatter.

            # ── Append K/V to merged cache (4-head GQA layout, flash_attn reads as-is) ──
            fvk.gpu_copy(self.b_k_merged[i].data_ptr() + kv_len * K_DIM * 2,
                         self.b_k.data_ptr(), Sq * K_DIM * 2, s)
            fvk.gpu_copy(self.b_v_merged[i].data_ptr() + kv_len * K_DIM * 2,
                         self.b_v.data_ptr(), Sq * K_DIM * 2, s)

            # ── Flash Attention (native GQA 28:4, no expand, no materialized logits) ──
            # flash_attn_func: Q [1,Sq,NHQ,HD] × K,V [1,total_kv,NHKV,HD] → out [1,Sq,NHQ,HD]
            # Measured 6.2x faster than cuBLAS on Thor (Sq=786, total_kv=808, bf16).
            # The output tensor is allocated by flash_attn; PyTorch caching allocator
            # pins its pointer across graph replays (see flash_rt TorchFlashAttnBackend).
            q_view = self.b_q.view(1, Sq, NHQ, HD)
            k_view = self.b_k_merged[i][:total_kv].view(1, total_kv, NHKV, HD)
            v_view = self.b_v_merged[i][:total_kv].view(1, total_kv, NHKV, HD)
            out_ao = flash_attn_func(q_view, k_view, v_view,
                                     softmax_scale=scale, causal=False)
            ao_ptr = out_ao.data_ptr()  # [1, Sq, NHQ, HD] row-major = contiguous [Sq, D]

            # ── O projection: gen FP8 for all Sq, und BF16 overwrite for text ──
            fvk.quantize_fp8_static(
                ao_ptr, self.b_fp8.data_ptr(),
                _asp(3), Sq * D, s)
            fvk.cutlass_fp8_sq_bf16out(
                self.b_fp8.data_ptr(), w['o_fp8'].data_ptr(),
                self.b_o.data_ptr(), Sq, D, D, a['o'], 0.0, s)

            # Text O: und_o (FP8 GEMM, no bias) on 2 rows (skipped when skip_und)
            if not skip_und:
                fvk.gpu_copy(self.b_text_ao.data_ptr(),
                             ao_ptr,                               D * 2, s)
                fvk.gpu_copy(self.b_text_ao.data_ptr() + D * 2,
                             ao_ptr + LAST_D,                      D * 2, s)
                fvk.quantize_fp8_static(
                    self.b_text_ao.data_ptr(), self.b_text_fp8.data_ptr(),
                    self.und_act_scales.data_ptr() + (i * 4 + 1) * 4, 2 * D, s)
                fvk.cutlass_fp8_sq_bf16out(
                    self.b_text_fp8.data_ptr(), w['und_o_fp8'].data_ptr(),
                    self.b_text_o.data_ptr(), 2, D, D, ua['o'], 0.0, s)
                fvk.gpu_copy(self.b_o.data_ptr(),
                             self.b_text_o.data_ptr(),              D * 2, s)
                fvk.gpu_copy(self.b_o.data_ptr() + LAST_D,
                             self.b_text_o.data_ptr() + D * 2,      D * 2, s)

            # F2 fuse A: post-attn residual + FFN input norm → FP8
            fvk.residual_add_rms_norm_fp8(
                self.b_x.data_ptr(), self.b_o.data_ptr(),
                w['ln2_w'].data_ptr(), self.b_fp8.data_ptr(),
                Sq, D, 1e-6, _asp(4), s)

            # ── FFN gen path (all Sq — text rows overwritten later) ──
            fvk.cutlass_fp8_t1_bf16out(
                self.b_fp8.data_ptr(), w['gate_fp8'].data_ptr(),
                self.b_gate.data_ptr(), Sq, FFN, D, a['gate'], 0.0, s)
            fvk.cutlass_fp8_t1_bf16out(
                self.b_fp8.data_ptr(), w['up_fp8'].data_ptr(),
                self.b_up.data_ptr(), Sq, FFN, D, a['up'], 0.0, s)
            fwk.silu_mul_split_fp8_bf16(
                self.b_gate.data_ptr(), self.b_up.data_ptr(),
                self.b_fp8.data_ptr(), Sq * FFN, _asp(6), s)
            fvk.cutlass_fp8_wide_bf16out(
                self.b_fp8.data_ptr(), w['down_fp8'].data_ptr(),
                self.b_down.data_ptr(), Sq, D, FFN, a['down'], 0.0, s)

            # ── FFN text path (FP8 GEMMs at M=2) ── (skipped when skip_und)
            if not skip_und:
                fvk.gpu_copy(self.b_text_x.data_ptr(),
                             self.b_x.data_ptr(),             D * 2, s)
                fvk.gpu_copy(self.b_text_x.data_ptr() + D * 2,
                             self.b_x.data_ptr() + LAST_D,    D * 2, s)
                fvk.rms_norm(self.b_text_x.data_ptr(), w['und_pn_w'].data_ptr(),
                             self.b_text_x.data_ptr(), 2, D, 1e-6, s)

                # Gate/Up: shared FP8 input
                fvk.quantize_fp8_static(
                    self.b_text_x.data_ptr(), self.b_text_fp8.data_ptr(),
                    self.und_act_scales.data_ptr() + (i * 4 + 2) * 4, 2 * D, s)
                fvk.cutlass_fp8_t1_bf16out(
                    self.b_text_fp8.data_ptr(), w['und_gate_fp8'].data_ptr(),
                    self.b_text_gate.data_ptr(), 2, FFN, D, ua['gate'], 0.0, s)
                fvk.cutlass_fp8_t1_bf16out(
                    self.b_text_fp8.data_ptr(), w['und_up_fp8'].data_ptr(),
                    self.b_text_up.data_ptr(), 2, FFN, D, ua['up'], 0.0, s)

                # SiLU(gate)*up → FP8
                fwk.silu_mul_split_fp8_bf16(
                    self.b_text_gate.data_ptr(), self.b_text_up.data_ptr(),
                    self.b_text_fp8.data_ptr(), 2 * FFN,
                    self.und_act_scales.data_ptr() + (i * 4 + 3) * 4, s)
                fvk.cutlass_fp8_wide_bf16out(
                    self.b_text_fp8.data_ptr(), w['und_down_fp8'].data_ptr(),
                    self.b_text_down.data_ptr(), 2, D, FFN, ua['down'], 0.0, s)

                # Overwrite text rows in b_down
                fvk.gpu_copy(self.b_down.data_ptr(),
                             self.b_text_down.data_ptr(),            D * 2, s)
                fvk.gpu_copy(self.b_down.data_ptr() + LAST_D,
                             self.b_text_down.data_ptr() + D * 2,    D * 2, s)

            # F2 fuse B: post-FFN residual + NEXT layer's input norm → FP8.
            # Last layer has no next norm, so stays as plain residual_add.
            if i < L - 1:
                fvk.residual_add_rms_norm_fp8(
                    self.b_x.data_ptr(), self.b_down.data_ptr(),
                    self.layers[i + 1]['ln_w'].data_ptr(),
                    self.b_fp8.data_ptr(),
                    Sq, D, 1e-6,
                    act_scales_base + ((i + 1) * 7 + 0) * 4, s)
            else:
                fvk.residual_add(self.b_x.data_ptr(), self.b_down.data_ptr(), Sq * D, s)

        # ── Final norm: gen for all, und overwrite for text (skipped when skip_und) ──
        fvk.rms_norm(self.b_x.data_ptr(), self.gen_final_norm.data_ptr(),
                     self.b_xn.data_ptr(), Sq, D, 1e-6, s)
        if not getattr(self, 'skip_und', False):
            fvk.gpu_copy(self.b_text_x.data_ptr(),
                         self.b_x.data_ptr(),             D * 2, s)
            fvk.gpu_copy(self.b_text_x.data_ptr() + D * 2,
                         self.b_x.data_ptr() + LAST_D,    D * 2, s)
            fvk.rms_norm(self.b_text_x.data_ptr(), self.und_final_norm.data_ptr(),
                         self.b_text_x.data_ptr(), 2, D, 1e-6, s)
            fvk.gpu_copy(self.b_xn.data_ptr(),
                         self.b_text_x.data_ptr(),             D * 2, s)
            fvk.gpu_copy(self.b_xn.data_ptr() + LAST_D,
                         self.b_text_x.data_ptr() + D * 2,     D * 2, s)

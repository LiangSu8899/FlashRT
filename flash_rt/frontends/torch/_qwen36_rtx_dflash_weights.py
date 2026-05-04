"""FlashRT — Qwen3.6 DFlash drafter NVFP4 W4A16 loader.

Loads the z-lab/Qwen3.6-27B-DFlash drafter checkpoint
(``model.safetensors``, BF16 native, 3.3 GB) directly from safetensors
and quantizes every linear projection to NVFP4 swizzled at load time
via the G7 kernel ``bf16_weight_to_nvfp4_swizzled`` (~825 MB packed
NVFP4 + scales).

The drafter is the **draft component** of DFlash speculative decoding.
It does not own its embed_tokens or lm_head — both are shared with the
target Qwen3.6-27B (``tie_word_embeddings=false`` refers to the
drafter not having its OWN tied embed/lm_head pair; it inherits from
target). So this loader only handles the 58 keys actually present on
disk.

Drafter architecture (from ``config.json``):

  num_hidden_layers      : 5
  layer_types            : 4 × 'sliding_attention' + 1 × 'full_attention'
  num_attention_heads    : 32  (Q heads)
  num_key_value_heads    :  8  (KV heads, GQA 4:1)
  head_dim               : 128
  hidden_size            : 5120
  intermediate_size      : 17408
  vocab_size             : 248320
  sliding_window         : 2048
  block_size             : 16          (block-diffusion drafter block)
  mask_token_id          : 248070
  target_layer_ids       : [1, 16, 31, 46, 61]   # main model hidden taps
  num_target_layers      : 64
  rope_theta             : 10_000_000
  max_position_embeddings: 262144
  rms_norm_eps           : 1e-6

Tensor inventory (58 keys total, all BF16 on disk):

  Top-level (3):
    fc.weight                            (5120, 25600)  — concat 5 taps
    hidden_norm.weight                   (5120,)
    norm.weight                          (5120,)

  Per layer × 5 (11 each = 55):
    self_attn.q_proj.weight              (4096, 5120)
    self_attn.k_proj.weight              (1024, 5120)
    self_attn.v_proj.weight              (1024, 5120)
    self_attn.o_proj.weight              (5120, 4096)
    self_attn.q_norm.weight              (128,)
    self_attn.k_norm.weight              (128,)
    mlp.gate_proj.weight                 (17408, 5120)
    mlp.up_proj.weight                   (17408, 5120)
    mlp.down_proj.weight                 (5120, 17408)
    input_layernorm.weight               (5120,)
    post_attention_layernorm.weight      (5120,)

  → 36 NVFP4-quantized linears + 22 BF16 norms.

Norm precompute: drafter is plain Qwen3 (NOT Qwen3.5), so its RMSNorm
is ``x * w / sqrt(mean(x²) + eps)`` — there is **no ``1 + w`` quirk**.
Stored norm weights are raw BF16. Main-path keys use the suffix
``_eff_w`` (with the +1 precompute baked in); drafter keys use
``_norm_w`` to flag the difference and prevent accidental dispatch
through the main norm kernel call sites.

VRAM budget: ~825 MB drafter packed NVFP4 + scales + bf16 norms
≈ 1 GB. Main NVFP4 path peaks at 28.4 GB / 32 GB so headroom is fine.

Output: extends the existing ``WeightHandles`` produced by
``extract_weights_nvfp4`` with a new top-level key
``handles.ptrs['dflash']`` that holds all drafter pointers + config
constants. All anchors are appended to the same ``handles.anchors``
list, so the drafter weights stay alive for the lifetime of the
frontend.
"""
from __future__ import annotations

import json
import os

import torch

from flash_rt.frontends.torch._qwen36_rtx_weights import (
    WeightHandles,
    _ensure_anchored,
)
from flash_rt.frontends.torch._qwen36_rtx_nvfp4_weights import (
    _bf16_anchor,
    _quant_bf16_lin_proj,
)


# Expected geometry (from z-lab/Qwen3.6-27B-DFlash config.json).
# Loader asserts each tensor's shape against these to catch a mismatched
# checkpoint upfront rather than letting the GEMM crash later.
_EXPECTED_LAYERS         = 5
_EXPECTED_LAYER_TYPES    = (
    'sliding_attention', 'sliding_attention', 'sliding_attention',
    'sliding_attention', 'full_attention',
)
_EXPECTED_HIDDEN         = 5120
_EXPECTED_Q_DIM          = 4096   # 32 Q heads × 128
_EXPECTED_KV_DIM         = 1024   # 8 KV heads × 128
_EXPECTED_INTERMEDIATE   = 17408
_EXPECTED_HEAD_DIM       = 128
_EXPECTED_VOCAB          = 248320
_EXPECTED_FC_IN          = 25600  # 5 × hidden (concat of 5 hidden taps)


def _quant_bf16_drafter_proj(
    handles: WeightHandles,
    ld: dict,
    prefix: str,
    w_bf16_cpu: torch.Tensor,
    expected_shape: tuple[int, int],
    fvk,
    device: str,
) -> None:
    """Quantize one drafter linear (BF16 → NVFP4 swizzled) with shape check.

    Reuses ``_quant_bf16_lin_proj`` (the same G7 kernel used by the
    main NVFP4 path for in_proj_qkv/in_proj_z/out_proj/lm_head).
    """
    if tuple(w_bf16_cpu.shape) != expected_shape:
        raise RuntimeError(
            f'drafter: tensor for {prefix!r} shape '
            f'{tuple(w_bf16_cpu.shape)} != expected {expected_shape}'
        )
    _quant_bf16_lin_proj(handles, ld, prefix, w_bf16_cpu, fvk, device)


def _bf16_norm_anchor(
    handles: WeightHandles,
    w_cpu: torch.Tensor,
    expected_len: int,
    device: str,
) -> int:
    """Anchor a raw BF16 norm weight on GPU (no ``1+w`` precompute).

    Drafter is plain Qwen3, so RMSNorm = ``x * w / sqrt(...)``.
    """
    if w_cpu.dim() != 1 or w_cpu.shape[0] != expected_len:
        raise RuntimeError(
            f'drafter norm shape {tuple(w_cpu.shape)} != expected '
            f'({expected_len},)'
        )
    return _bf16_anchor(handles, w_cpu, device)


def extract_dflash_weights_nvfp4(
    handles: WeightHandles,
    ckpt_dir: str,
    fvk,
    device: str = 'cuda:0',
) -> dict:
    """Load DFlash drafter weights and attach to ``handles.ptrs['dflash']``.

    Args:
      handles: existing :class:`WeightHandles` from the main NVFP4
        load. Drafter anchors are appended to its ``anchors`` list.
      ckpt_dir: path to the z-lab/Qwen3.6-27B-DFlash dir containing
        ``model.safetensors`` + ``config.json``.
      fvk: ``flash_rt_kernels`` pybind module
        (provides ``bf16_weight_to_nvfp4_swizzled``).
      device: cuda device.

    Returns:
      The ``dflash`` dict (also stored at ``handles.ptrs['dflash']``).
    """
    from safetensors import safe_open

    cfg_path = os.path.join(ckpt_dir, 'config.json')
    st_path = os.path.join(ckpt_dir, 'model.safetensors')
    if not (os.path.isfile(cfg_path) and os.path.isfile(st_path)):
        raise RuntimeError(
            f'DFlash ckpt dir missing config.json or model.safetensors: '
            f'{ckpt_dir!r}'
        )

    cfg = json.load(open(cfg_path))

    # ── 1. Sanity-check config matches what the loader was written for ──
    arch = (cfg.get('architectures') or [''])[0]
    if arch != 'DFlashDraftModel':
        raise RuntimeError(
            f'DFlash ckpt config architectures={arch!r} '
            f"!= 'DFlashDraftModel'"
        )
    n_layers = int(cfg.get('num_hidden_layers', 0))
    layer_types = list(cfg.get('layer_types') or ())
    if n_layers != _EXPECTED_LAYERS:
        raise RuntimeError(
            f'DFlash drafter num_hidden_layers={n_layers} != '
            f'{_EXPECTED_LAYERS}; loader does not generalize'
        )
    if tuple(layer_types) != _EXPECTED_LAYER_TYPES:
        raise RuntimeError(
            f'DFlash drafter layer_types={layer_types} != '
            f'{list(_EXPECTED_LAYER_TYPES)}'
        )
    if int(cfg.get('hidden_size', 0)) != _EXPECTED_HIDDEN:
        raise RuntimeError('hidden_size mismatch')
    if int(cfg.get('intermediate_size', 0)) != _EXPECTED_INTERMEDIATE:
        raise RuntimeError('intermediate_size mismatch')
    if int(cfg.get('num_attention_heads', 0)) * int(
            cfg.get('head_dim', 0)) != _EXPECTED_Q_DIM:
        raise RuntimeError('Q_dim mismatch')
    if int(cfg.get('num_key_value_heads', 0)) * int(
            cfg.get('head_dim', 0)) != _EXPECTED_KV_DIM:
        raise RuntimeError('KV_dim mismatch')

    dflash_cfg = cfg.get('dflash_config') or {}
    target_layer_ids = list(dflash_cfg.get('target_layer_ids') or [])
    mask_token_id = int(dflash_cfg.get('mask_token_id', 0))
    if len(target_layer_ids) != _EXPECTED_FC_IN // _EXPECTED_HIDDEN:
        raise RuntimeError(
            f'target_layer_ids len {len(target_layer_ids)} != '
            f'{_EXPECTED_FC_IN // _EXPECTED_HIDDEN}'
        )

    debug = bool(int(os.environ.get('FLASHRT_DFLASH_LOAD_DEBUG', '0')
                     or '0'))

    def _vram_used():
        free, total = torch.cuda.mem_get_info()
        return (total - free) / 1e9

    out: dict = {
        # geometry (echo back so the forward code doesn't re-parse cfg)
        'num_layers':       n_layers,
        'layer_types':      list(layer_types),
        'hidden':           _EXPECTED_HIDDEN,
        'intermediate':     _EXPECTED_INTERMEDIATE,
        'q_dim':            _EXPECTED_Q_DIM,
        'kv_dim':           _EXPECTED_KV_DIM,
        'head_dim':         _EXPECTED_HEAD_DIM,
        'num_q_heads':      int(cfg['num_attention_heads']),
        'num_kv_heads':     int(cfg['num_key_value_heads']),
        'vocab_size':       int(cfg.get('vocab_size', _EXPECTED_VOCAB)),
        'fc_in':            _EXPECTED_FC_IN,
        # config constants used by the forward path
        'sliding_window':   int(cfg.get('sliding_window', 2048)),
        'block_size':       int(cfg.get('block_size', 16)),
        'rope_theta':       float(cfg.get('rope_theta', 10_000_000.0)),
        'rms_norm_eps':     float(cfg.get('rms_norm_eps', 1e-6)),
        'max_position_embeddings':
                            int(cfg.get('max_position_embeddings', 262144)),
        # DFlash-specific
        'mask_token_id':    mask_token_id,
        'target_layer_ids': list(target_layer_ids),
        # marker
        'quant_format':     'nvfp4',
    }

    with safe_open(st_path, framework='pt', device='cpu') as f:
        if debug:
            print(f'  [dflash-load] open, vram = {_vram_used():.2f} GB')

        # ── 2. Top-level (3 tensors) ───────────────────────────────────
        # fc.weight: (hidden=5120, fc_in=25600). Largest single drafter
        # weight; we NVFP4-quantize it for BW + memory savings.
        _quant_bf16_drafter_proj(
            handles, out, 'fc',
            f.get_tensor('fc.weight'),
            (_EXPECTED_HIDDEN, _EXPECTED_FC_IN),
            fvk, device,
        )
        # hidden_norm: applied to the FC output (concat of taps)
        out['hidden_norm_w'] = _bf16_norm_anchor(
            handles, f.get_tensor('hidden_norm.weight'),
            _EXPECTED_HIDDEN, device)
        # norm: final RMSNorm before sharing the lm_head with the target
        out['final_norm_w'] = _bf16_norm_anchor(
            handles, f.get_tensor('norm.weight'),
            _EXPECTED_HIDDEN, device)

        if debug:
            torch.cuda.synchronize()
            print(f'  [dflash-load] top-level done, '
                  f'vram = {_vram_used():.2f} GB')

        # ── 3. Per-layer (5 layers × 11 tensors each) ──────────────────
        per_layer: list[dict] = []
        for L in range(n_layers):
            base = f'layers.{L}.'
            t = layer_types[L]
            ld: dict = {'type': t, 'quant_format': 'nvfp4'}

            # NVFP4 linears (7 per layer)
            _quant_bf16_drafter_proj(
                handles, ld, 'q_proj',
                f.get_tensor(base + 'self_attn.q_proj.weight'),
                (_EXPECTED_Q_DIM, _EXPECTED_HIDDEN),
                fvk, device)
            _quant_bf16_drafter_proj(
                handles, ld, 'k_proj',
                f.get_tensor(base + 'self_attn.k_proj.weight'),
                (_EXPECTED_KV_DIM, _EXPECTED_HIDDEN),
                fvk, device)
            _quant_bf16_drafter_proj(
                handles, ld, 'v_proj',
                f.get_tensor(base + 'self_attn.v_proj.weight'),
                (_EXPECTED_KV_DIM, _EXPECTED_HIDDEN),
                fvk, device)
            _quant_bf16_drafter_proj(
                handles, ld, 'o_proj',
                f.get_tensor(base + 'self_attn.o_proj.weight'),
                (_EXPECTED_HIDDEN, _EXPECTED_Q_DIM),
                fvk, device)
            _quant_bf16_drafter_proj(
                handles, ld, 'mlp_gate',
                f.get_tensor(base + 'mlp.gate_proj.weight'),
                (_EXPECTED_INTERMEDIATE, _EXPECTED_HIDDEN),
                fvk, device)
            _quant_bf16_drafter_proj(
                handles, ld, 'mlp_up',
                f.get_tensor(base + 'mlp.up_proj.weight'),
                (_EXPECTED_INTERMEDIATE, _EXPECTED_HIDDEN),
                fvk, device)
            _quant_bf16_drafter_proj(
                handles, ld, 'mlp_down',
                f.get_tensor(base + 'mlp.down_proj.weight'),
                (_EXPECTED_HIDDEN, _EXPECTED_INTERMEDIATE),
                fvk, device)

            # BF16 norms (4 per layer; raw weights, no (1+w) for plain Qwen3)
            ld['input_norm_w'] = _bf16_norm_anchor(
                handles,
                f.get_tensor(base + 'input_layernorm.weight'),
                _EXPECTED_HIDDEN, device)
            ld['post_attn_norm_w'] = _bf16_norm_anchor(
                handles,
                f.get_tensor(base + 'post_attention_layernorm.weight'),
                _EXPECTED_HIDDEN, device)
            ld['q_norm_w'] = _bf16_norm_anchor(
                handles,
                f.get_tensor(base + 'self_attn.q_norm.weight'),
                _EXPECTED_HEAD_DIM, device)
            ld['k_norm_w'] = _bf16_norm_anchor(
                handles,
                f.get_tensor(base + 'self_attn.k_norm.weight'),
                _EXPECTED_HEAD_DIM, device)

            per_layer.append(ld)
            if debug:
                torch.cuda.synchronize()
                print(f'  [dflash-load] layer {L} ({t}) done, '
                      f'vram = {_vram_used():.2f} GB')

        out['layers'] = per_layer

    handles.ptrs['dflash'] = out
    if debug:
        torch.cuda.synchronize()
        print(f'  [dflash-load] DONE, vram = {_vram_used():.2f} GB')
    return out


# Per-layer linear keys produced by the loader (3 per quantized proj).
_LIN_KEYS = ('q_proj', 'k_proj', 'v_proj', 'o_proj',
             'mlp_gate', 'mlp_up', 'mlp_down')
_NORM_KEYS = ('input_norm_w', 'post_attn_norm_w', 'q_norm_w', 'k_norm_w')


def assert_dflash_extraction_invariants(handles: WeightHandles) -> None:
    """Verify loader output is well-formed. Run once after loading.

    Counts: 3 + 11 × 5 = 58 source tensors → 1 (fc) × 3 + 7 × 5 × 3
    = 108 NVFP4 sub-handles + 22 BF16 norm handles + 14 config keys.
    """
    assert 'dflash' in handles.ptrs, 'handles.ptrs[\"dflash\"] missing'
    d = handles.ptrs['dflash']
    assert d.get('quant_format') == 'nvfp4'
    assert d['num_layers'] == _EXPECTED_LAYERS
    assert tuple(d['layer_types']) == _EXPECTED_LAYER_TYPES
    assert d['hidden'] == _EXPECTED_HIDDEN
    assert d['mask_token_id'] != 0, 'mask_token_id missing'
    assert len(d['target_layer_ids']) == 5

    # top-level
    for k in ('fc_packed', 'fc_sf', 'fc_alpha',
              'hidden_norm_w', 'final_norm_w'):
        assert k in d, f'top-level dflash key missing: {k!r}'

    # per-layer
    layers = d['layers']
    assert len(layers) == _EXPECTED_LAYERS
    for L, ld in enumerate(layers):
        for short in _LIN_KEYS:
            for suf in ('_packed', '_sf', '_alpha'):
                k = short + suf
                assert k in ld, (
                    f'drafter layer {L} ({ld["type"]}) missing key {k!r}'
                )
        for k in _NORM_KEYS:
            assert k in ld, (
                f'drafter layer {L} ({ld["type"]}) missing norm {k!r}'
            )

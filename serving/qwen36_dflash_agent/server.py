#!/usr/bin/env python3
"""FlashRT — Qwen3.6-27B DFlash OpenAI-compatible serving host.

Serves /v1/chat/completions backed by the DFlash block-diffusion
speculative-decode path (`generate_own_speculative_DFlash_nvfp4`).
This is the policy layer above the FlashRT execution contract: it owns
request shaping and telemetry only, and adds no session or KV verbs.

Scope (v1):
  * Stateless per request — every call prefills the full prompt.
    For long-running agent sessions with prefix reuse, tool calling,
    and committed-token streaming, use ``serving/qwen36_agent``.
  * Batch size 1; concurrent requests are serialized on one GPU.
  * Greedy decode only — sampling parameters are accepted and ignored.
  * The DFlash loop generates the full ``max_tokens`` budget; the
    response is truncated at the first end token during detokenize.

Usage:
    pip install fastapi uvicorn

    export FLASHRT_QWEN36_MTP_CKPT_DIR=/models/Qwen3.6-27B-FP8
    export FLASHRT_QWEN36_DFLASH_CKPT_DIR=/models/Qwen3.6-27B-DFlash
    export FLASHRT_QWEN36_LONG_KV_CACHE=fp8

    python -m serving.qwen36_dflash_agent.server \\
        --checkpoint /models/Qwen3.6-27B-NVFP4 \\
        --max-seq 32768 --K 15 --port 8000
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
log = logging.getLogger('qwen36_dflash_server')


def _build_frontend(args):
    import torch

    cap = torch.cuda.get_device_capability()
    arch = args.arch
    if arch == 'auto':
        arch = 'thor' if cap == (11, 0) else 'rtx'
    if arch == 'thor':
        from flash_rt.frontends.torch.qwen36_thor import (
            Qwen36TorchFrontendThor as Frontend,
        )
    else:
        from flash_rt.frontends.torch.qwen36_rtx import (
            Qwen36TorchFrontendRtx as Frontend,
        )
    log.info('loading %s frontend (sm %s), checkpoint=%s',
             arch, cap, args.checkpoint)
    fe = Frontend(args.checkpoint, quant='nvfp4', max_seq=args.max_seq)
    fe.init_dflash_drafter(args.dflash_checkpoint or None)
    log.info('DFlash drafter ready (pertoken=%s window=%s)',
             getattr(fe, '_dflash_pertoken_window', False),
             getattr(fe, '_dflash_pertoken_win', None))
    return fe, arch


def _chat_ids(fe, messages: List[Dict[str, Any]], enable_thinking: bool):
    return fe._tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        return_tensors='pt',
    ).to(fe.device)


def create_app(args):
    from fastapi import FastAPI, HTTPException

    fe, arch = _build_frontend(args)
    tok = fe._tokenizer
    end_ids = {tid for tid in (
        tok.eos_token_id,
        tok.convert_tokens_to_ids('<|im_end|>'),
    ) if isinstance(tid, int) and tid >= 0}

    app = FastAPI(title='FlashRT Qwen3.6 DFlash server')
    gpu_lock = asyncio.Lock()
    state = {'requests': 0}

    @app.get('/health')
    async def health():
        return {
            'status': 'ok',
            'arch': arch,
            'path': 'dflash',
            'max_seq': args.max_seq,
            'K': args.K,
            'pertoken_window': bool(
                getattr(fe, '_dflash_pertoken_window', False)),
            'window': getattr(fe, '_dflash_pertoken_win', None),
            'requests_served': state['requests'],
        }

    @app.get('/v1/models')
    async def models():
        return {'object': 'list', 'data': [{
            'id': args.model_name, 'object': 'model',
            'owned_by': 'flashrt'}]}

    @app.post('/v1/chat/completions')
    async def chat(body: Dict[str, Any]):
        import torch

        messages = body.get('messages')
        if not messages:
            raise HTTPException(400, 'messages is required')
        max_tokens = int(body.get('max_tokens') or args.default_max_tokens)
        max_tokens = max(1, min(max_tokens, args.max_tokens_cap))
        enable_thinking = bool(body.get('enable_thinking', False))

        async with gpu_lock:
            t0 = time.perf_counter()
            ids = _chat_ids(fe, messages, enable_thinking)
            prompt_len = int(ids.shape[1])
            if prompt_len + max_tokens > args.max_seq:
                raise HTTPException(
                    400, f'prompt ({prompt_len}) + max_tokens '
                         f'({max_tokens}) exceeds max_seq ({args.max_seq})')
            out = await asyncio.to_thread(
                fe.generate_own_speculative_DFlash_nvfp4,
                ids, max_new_tokens=max_tokens, K=args.K)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0

        new_ids = out[0, prompt_len:].tolist()
        for i, t in enumerate(new_ids):
            if t in end_ids:
                new_ids = new_ids[:i]
                break
        text = tok.decode(new_ids, skip_special_tokens=True)
        attempts = int(getattr(fe, '_spec_attempts', 0))
        state['requests'] += 1
        return {
            'id': f'chatcmpl-{uuid.uuid4().hex[:24]}',
            'object': 'chat.completion',
            'created': int(time.time()),
            'model': args.model_name,
            'choices': [{
                'index': 0,
                'message': {'role': 'assistant', 'content': text},
                'finish_reason': (
                    'stop' if len(new_ids) < max_tokens else 'length'),
            }],
            'usage': {
                'prompt_tokens': prompt_len,
                'completion_tokens': len(new_ids),
                'total_tokens': prompt_len + len(new_ids),
            },
            'flashrt': {
                'path': 'dflash',
                'spec_cycles': attempts,
                'accept_length': (
                    round(len(new_ids) / attempts, 2) if attempts else None),
                'e2e_ms': round(dt * 1e3, 1),
            },
        }

    return app


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True,
                   help='Qwen3.6-27B NVFP4 checkpoint directory')
    p.add_argument('--dflash-checkpoint', default='',
                   help='DFlash drafter directory (default: '
                        'FLASHRT_QWEN36_DFLASH_CKPT_DIR)')
    p.add_argument('--model-name', default='qwen3.6-27b-dflash')
    p.add_argument('--arch', choices=['auto', 'thor', 'rtx'],
                   default='auto')
    p.add_argument('--max-seq', type=int, default=32768)
    p.add_argument('--K', type=int, default=15,
                   help='speculative tokens per cycle (block_size - 1)')
    p.add_argument('--default-max-tokens', type=int, default=256)
    p.add_argument('--max-tokens-cap', type=int, default=4096)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8000)
    args = p.parse_args()

    os.environ.setdefault('FLASHRT_QWEN36_LONG_KV_CACHE', 'fp8')

    import uvicorn
    uvicorn.run(create_app(args), host=args.host, port=args.port)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

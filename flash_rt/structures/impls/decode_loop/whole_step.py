"""decode_loop — the whole-loop serving form for cached LLM decode.

Seam swaps accelerate the pieces; this family owns the loop. The host's
``generate`` walks Python glue between every token, repoints its KV
cache with ``torch.cat``, and rebuilds masks per step — none of which
can replay inside a CUDA graph. The whole-step form drives the decoder
layers directly with static buffers:

- a duck-typed static hybrid cache (preallocated KV written by
  ``index_copy_`` at an in-graph position buffer; recurrent/conv slots
  as the layers expect them) — no host cache class is imported, only
  the surface the layers actually touch is implemented;
- the causal row itself masks the padded tail of the static KV, so the
  decode mask is one ``index_select`` on the position — no mask
  bookkeeping;
- argmax and the position increment run in-graph, and the step is
  optionally wrapped in ``torch.compile`` so the elementwise regions
  between custom kernels fuse before capture.

The loop is greedy and exact: its tokens are gated against the host's
own generation by the caller's probe. Structure swaps (fused layers,
precision bands) compose underneath — build the loop after
``auto_swaps`` and it captures whatever is attached.
"""

from __future__ import annotations

import torch


class _StaticHybridCache:
    """The surface a hybrid (attention + gated-delta) stack touches.

    ``update`` serves attention layers from preallocated buffers;
    ``conv_states``/``recurrent_states`` are plain per-layer slots the
    gated-delta layers read and write; ``has_previous_state`` mirrors
    the host convention (a filled conv slot means decode).
    """

    def __init__(self, n_layers, attn_layers, kv_heads, head_dim,
                 max_len, device, dtype=torch.bfloat16):
        self.conv_states = [None] * n_layers
        self.recurrent_states = [None] * n_layers
        self.key_cache = [None] * n_layers
        self.value_cache = [None] * n_layers
        self._max = int(max_len)
        self._cp = None
        for i in attn_layers:
            self.key_cache[i] = torch.zeros(
                1, kv_heads, self._max, head_dim, device=device,
                dtype=dtype)
            self.value_cache[i] = torch.zeros_like(self.key_cache[i])

    def update(self, k, v, layer_idx, cache_kwargs=None):
        self.key_cache[layer_idx].index_copy_(2, self._cp, k)
        self.value_cache[layer_idx].index_copy_(2, self._cp, v)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx=0):
        return self._max

    def get_mask_sizes(self, query_length, layer_idx):
        return self._max, 0

    @property
    def has_previous_state(self):
        return any(s is not None for s in self.conv_states)


def _find_stack(model):
    """Locate the decoder stack by the slots it must carry, not names."""
    for mod in (getattr(model, "model", model),):
        for cand in (getattr(mod, "language_model", None), mod):
            if cand is None:
                continue
            if (hasattr(cand, "layers") and hasattr(cand, "embed_tokens")
                    and hasattr(cand, "norm")
                    and hasattr(cand, "rotary_emb")):
                return cand
    raise ValueError(
        "refused: no decoder stack with (layers, embed_tokens, norm, "
        "rotary_emb) slots found on this host")


class WholeStepDecodeLoop:
    """Compiled, graph-captured greedy decode over the attached model."""

    def __init__(self, model, *, max_len, compile_step=True):
        lm = _find_stack(model)
        head = getattr(model, "lm_head", None)
        if head is None:
            raise ValueError("refused: host carries no lm_head")
        self._layers = lm.layers
        self._rotary = lm.rotary_emb
        self._norm = lm.norm
        self._embed = lm.embed_tokens
        self._head = head
        self._full = [i for i, lyr in enumerate(self._layers)
                      if hasattr(lyr, "self_attn")]
        cfg = getattr(model.config, "text_config", model.config)
        kvh = int(cfg.num_key_value_heads)
        hd = int(getattr(cfg, "head_dim",
                         cfg.hidden_size // cfg.num_attention_heads))
        dev = head.weight.device if hasattr(head, "weight") else "cuda"
        self._max = int(max_len)
        self.cache = _StaticHybridCache(
            len(self._layers), self._full, kvh, hd, self._max, dev)
        self._causal = torch.full(
            (self._max, self._max), float("-inf"), device=dev,
            dtype=torch.bfloat16).triu_(1)
        self._cur = torch.empty(1, 1, dtype=torch.long, device=dev)
        self._pos = torch.empty(1, dtype=torch.long, device=dev)
        if compile_step:
            # the layer loop specialises per layer index; the default
            # recompile budget is smaller than a deep stack
            torch._dynamo.config.cache_size_limit = max(
                torch._dynamo.config.cache_size_limit,
                4 * len(self._layers))
            self._step = torch.compile(self._fwd, dynamic=False)
        else:
            self._step = self._fwd
        self._graph = None

    def _fwd(self, tok_ids, pos_t):
        h = self._embed(tok_ids)
        pe = self._rotary(h, pos_t.view(1, -1))
        self.cache._cp = pos_t
        m4 = self._causal.index_select(0, pos_t).view(
            1, 1, -1, self._max)
        for i, lyr in enumerate(self._layers):
            h = lyr(h, position_embeddings=pe,
                    attention_mask=(m4 if i in self._full_set else None),
                    past_key_values=self.cache, use_cache=True,
                    cache_position=pos_t)
        return self._head(self._norm(h)[:, -1:])

    @property
    def _full_set(self):
        s = getattr(self, "_full_cache", None)
        if s is None:
            s = set(self._full)
            self._full_cache = s
        return s

    def _gstep(self):
        logits = self._step(self._cur, self._pos)
        # hosts sample from FP32 logits; argmax there too, or BF16
        # ties break differently and free runs diverge
        self._cur.copy_(logits.float().argmax(-1))
        self._pos.add_(1)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens):
        """Greedy generation; the first call warms, compiles, captures."""
        L = int(input_ids.shape[1])
        if L + max_new_tokens > self._max:
            raise ValueError(
                f"refused: {L}+{max_new_tokens} exceeds the static "
                f"window {self._max}")
        logits = self._step(input_ids,
                            torch.arange(L, device=input_ids.device))
        self._cur.copy_(logits.float().argmax(-1))
        self._pos.fill_(L)
        toks = [self._cur.clone()]
        warm = min(3, max_new_tokens - 1)
        if self._graph is None:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(warm):
                    self._gstep()
                    toks.append(self._cur.clone())
            torch.cuda.current_stream().wait_stream(side)
            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._graph):
                self._gstep()
            for _ in range(max_new_tokens - 1 - warm):
                self._graph.replay()
                toks.append(self._cur.clone())
        else:
            # steady calls run every decode step as a replay
            for _ in range(max_new_tokens - 1):
                self._graph.replay()
                toks.append(self._cur.clone())
        return torch.cat([input_ids] + toks, dim=1)


def build_decode_loop(model, *, max_len, compile_step=True):
    """Build the whole-loop form over whatever is attached to ``model``."""
    return WholeStepDecodeLoop(model, max_len=max_len,
                               compile_step=compile_step)

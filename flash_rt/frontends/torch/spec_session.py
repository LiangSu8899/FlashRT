"""FlashRT — speculative-decode session (shared orchestration).

One speculation loop for every (model family x drafter type x hardware)
combination. The session owns the cycle — draft, verify, accept,
rollback, commit — and delegates everything hardware- or model-shaped
to role objects:

  AcceptancePolicy   pure tensor accept/commit rules (CPU unit-testable)
  Drafter            proposes K tokens per cycle and maintains its own
                     context (window / chain KV)
  frontend hooks     the Verifier + StateCommitter seam: prefill,
                     S=K verify, state snapshot, partial-accept
                     rollback (``_dflash_prefill_nvfp4``,
                     ``_dflash_verify_forward_K`` via the verify graph,
                     ``_dflash_snap_state``, ``_dflash_partial_rollback``)

Contract invariants (violations are correctness bugs, not tuning):

  I1  The verify forward and the rollback state source must come from
      ONE kernel family; mixing families surfaces their occasional
      rounding disagreements as greedy divergence.
  I2  Token-exact parity is only defined against a reference that
      shares the verify's KV format.
  I3  The drafter's context commit reads feedback rows 0..N and must
      run BEFORE the end-of-cycle taps shuffle overwrites row 0.
  I4  Rollback checkpoint slot s holds the committed state after
      verify row s, and slot K-1 equals the post-verify state.

``step()`` is the interruption grid: each completed cycle is a
committed boundary where generation can pause, and ``boundary()``
enumerates the named buffers that make that boundary restorable.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional


def as_input_ids_tensor(input_ids, *, device=None):
    """Return the ``input_ids`` tensor from tokenizer output or a tensor.

    Hugging Face tokenizers can return either a tensor directly or a
    BatchEncoding/dict containing ``input_ids``. The speculative session
    consumes the tensor form; normalizing it here keeps hosts and Nexus
    examples from depending on tokenizer return-type details.
    """
    try:
        if 'input_ids' in input_ids:
            input_ids = input_ids['input_ids']
    except (KeyError, TypeError, RuntimeError):
        pass
    if device is not None:
        input_ids = input_ids.to(device)
    return input_ids


class StrictArgmax:
    """Exact greedy acceptance: a draft row matches iff it equals the
    verify argmax; committed tokens are the argmax rows."""

    relaxed = False

    def begin(self, input_ids) -> None:
        pass

    def matches(self, logits_K, drafts, all_argmax):
        K = int(drafts.shape[0])
        return (all_argmax[:K] == drafts).long()

    def commit_source(self, N, drafts, all_argmax):
        return lambda j: all_argmax[j:j + 1].view(1, 1)

    def observe(self, N, drafts, all_argmax) -> None:
        pass


class RelaxedThinking:
    """Thinking-phase relaxed acceptance (TensorRT-LLM MTP policy).

    Inside a <think> block a draft is accepted when it is in the verify
    logits' top-k AND within a logit margin of the argmax; the accepted
    token is the DRAFT (verify rows and per-step state already
    condition on the drafts). Rows from the first draft that closes the
    think block fall back to strict matching, keeping the visible
    answer exact-verified.
    """

    relaxed = True

    def __init__(self, topk: int, delta: float,
                 open_id: int, close_id: int):
        self.topk = max(1, int(topk))
        self.delta = float(delta)
        self.open_id = int(open_id)
        self.close_id = int(close_id)
        self.in_think = False

    @classmethod
    def from_env(cls, tokenizer) -> Optional['RelaxedThinking']:
        if os.environ.get(
                'FLASHRT_QWEN36_DFLASH_RELAXED_THINKING', '0',
        ).strip().lower() not in ('1', 'true', 'on'):
            return None
        think_open = tokenizer.convert_tokens_to_ids('<think>')
        think_close = tokenizer.convert_tokens_to_ids('</think>')
        if not (isinstance(think_open, int) and think_open >= 0):
            return None
        return cls(
            topk=int(os.environ.get(
                'FLASHRT_QWEN36_DFLASH_RELAXED_TOPK', '3')),
            delta=float(os.environ.get(
                'FLASHRT_QWEN36_DFLASH_RELAXED_DELTA', '1.0')),
            open_id=int(think_open),
            close_id=int(think_close),
        )

    @staticmethod
    def relaxed_matches(logits_K, drafts, all_argmax,
                        topk: int, delta: float, close_id: int):
        """Pure acceptance math; see class docstring. (K,) 0/1."""
        import torch

        K = int(drafts.shape[0])
        topv, topi = torch.topk(logits_K, topk, dim=-1)
        ok = (
            (topi == drafts.view(K, 1))
            & ((topv[:, :1] - topv) <= delta)
        ).any(-1).long()
        close_mask = drafts == close_id
        if bool(close_mask.any().item()):
            idx = int(close_mask.nonzero()[0].item())
            strict = (all_argmax[:K] == drafts).long()
            ok[idx:] = strict[idx:]
        return ok

    def begin(self, input_ids) -> None:
        # The chat template opens the think block at the end of the
        # generation prompt, so the phase can start active.
        self.in_think = self.open_id in input_ids[0, -8:].tolist()

    def matches(self, logits_K, drafts, all_argmax):
        if not self.in_think:
            K = int(drafts.shape[0])
            return (all_argmax[:K] == drafts).long()
        return self.relaxed_matches(
            logits_K, drafts, all_argmax,
            self.topk, self.delta, self.close_id)

    def commit_source(self, N, drafts, all_argmax):
        if not self.in_think:
            return lambda j: all_argmax[j:j + 1].view(1, 1)
        return lambda j: (
            drafts[j:j + 1].view(1, 1) if j < N
            else all_argmax[j:j + 1].view(1, 1))

    def observe(self, N, drafts, all_argmax) -> None:
        ids = (drafts[:N].tolist() if N else [])
        ids.append(int(all_argmax[N].item()))
        for t in ids:
            if t == self.open_id:
                self.in_think = True
            elif t == self.close_id:
                self.in_think = False


class DFlashBlockDrafter:
    """Block-diffusion drafter: one forward proposes the whole block.

    Owns the drafter-side context — the per-token feature window (or
    the legacy per-cycle shift window) — and its per-cycle update.
    """

    def __init__(self, fe, K: int):
        self.fe = fe
        self.K = int(K)
        self.eff_ctx = int(getattr(fe, '_dflash_eff_ctx', 16))
        self.pertoken = bool(
            getattr(fe, '_dflash_pertoken_window', False))

    def begin(self, input_ids) -> None:
        from flash_rt.frontends.torch._qwen36_rtx_dflash_forward import (
            alloc_drafter_capture_window,
            reset_drafter_capture_state,
        )

        fe = self.fe
        alloc_drafter_capture_window(fe, self.eff_ctx)
        reset_drafter_capture_state(fe)
        if self.pertoken:
            from flash_rt.frontends.torch._qwen36_rtx_dflash_forward import (  # noqa: E501
                alloc_pertoken_window,
                reset_pertoken_window,
            )
            alloc_pertoken_window(
                fe, int(getattr(fe, '_dflash_pertoken_win', 128)))
            reset_pertoken_window(fe)
        # Taps start at zero — the first drafter call gets no real
        # signal; AL on cycle 0 is below steady state.
        fe._dflash_taps_buf.zero_()

    def propose(self, tok):
        """One drafter forward; returns (K,) draft tokens.

        During ramp-up (window not yet full) the eager forward reads
        only the valid rows so attention never sees zero padding; once
        full, the captured graph replays.
        """
        fe = self.fe
        fe._dflash_buf['ids_static'][0:1].copy_(tok.view(1))
        if self.pertoken:
            valid = int(fe._dflash_buf['pt_valid'])
            if valid < int(fe._dflash_buf['pt_win']):
                from flash_rt.frontends.torch._qwen36_rtx_dflash_forward import (  # noqa: E501
                    dflash_drafter_forward_pertoken,
                )
                dflash_drafter_forward_pertoken(fe, max(1, valid))
            else:
                fe._ensure_drafter_graph_dflash_pertoken().replay()
        elif fe._spec_attempts < self.eff_ctx:
            from flash_rt.frontends.torch._qwen36_rtx_dflash_forward import (  # noqa: E501
                dflash_drafter_forward_capture_eager,
            )
            fe._dflash_buf['hidden_taps_static'].copy_(
                fe._dflash_taps_buf[:, 0])
            dflash_drafter_forward_capture_eager(
                fe, fe._spec_attempts + 1)
        else:
            fe._dflash_buf['hidden_taps_static'].copy_(
                fe._dflash_taps_buf[:, 0])
            fe._ensure_drafter_graph_dflash_nvfp4(self.eff_ctx).replay()
        # block layout: output[0] re-predicts prev_token; output[1..K]
        # predict the next K positions.
        return fe._dflash_buf['logits'].argmax(dim=-1)[1:self.K + 1]

    def commit(self, N: int) -> None:
        """Consume verify feedback for the committed rows (I3: must be
        called before the taps[:, 0] shuffle)."""
        if self.pertoken:
            self.fe._dflash_window_commit(N)


class SpecSession:
    """The speculation loop. One instance per generation request."""

    def __init__(self, fe, drafter, policy, *, max_new_tokens: int):
        self.fe = fe
        self.drafter = drafter
        self.policy = policy
        self.max_new_tokens = int(max_new_tokens)
        self.generated: List[Any] = []
        self.cur_pos = 0
        self.tok = None
        self._interrupt = False
        self._K = drafter.K

    # -------- Nexus-facing surface --------

    def request_interrupt(self) -> None:
        """Ask generate() to stop at the next committed boundary."""
        self._interrupt = True

    def boundary(self) -> Dict[str, Any]:
        """Named buffers of the committed boundary (restorable set).

        Enumerates the state a capsule needs to snapshot/restore this
        session between cycles. Tensors are live views, not copies.
        """
        fe = self.fe
        out: Dict[str, Any] = {
            'cur_pos': self.cur_pos,
            'tokens_generated': len(self.generated),
            'spec_attempts': int(getattr(fe, '_spec_attempts', 0)),
            'spec_accepts': int(getattr(fe, '_spec_accepts', 0)),
            'spec_full': int(getattr(fe, '_spec_full', 0)),
            'lin_state': fe._lin_state,
            'lin_conv_state': fe._lin_conv_state,
        }
        if hasattr(self.policy, 'in_think'):
            out['policy_in_think'] = bool(self.policy.in_think)
        if hasattr(fe, '_fp8_K_cache'):
            out['fp8_k_cache'] = fe._fp8_K_cache
            out['fp8_v_cache'] = fe._fp8_V_cache
        buf = getattr(fe, '_dflash_buf', None) or {}
        if buf.get('target_feat_window') is not None:
            out['drafter_shift_window'] = buf['target_feat_window']
        if buf.get('pt_window') is not None:
            out['drafter_window'] = buf['pt_window']
            out['drafter_window_valid'] = buf['pt_valid']
        out['taps_row0'] = fe._dflash_taps_buf[:, 0]
        return out

    # -------- generation --------

    def begin(self, input_ids) -> None:
        import torch

        fe = self.fe
        input_ids = as_input_ids_tensor(
            input_ids, device=getattr(fe, 'device', None))
        self.prompt_len = int(input_ids.shape[1])
        fe.reset_state()
        if not hasattr(fe, '_rope_cos_table'):
            fe._build_rope_table()
        self.drafter.begin(input_ids)
        self.policy.begin(input_ids)
        with torch.no_grad():
            self.tok = fe._dflash_prefill_nvfp4(input_ids)
        self.generated = [self.tok]
        self.cur_pos = self.prompt_len
        fe._spec_attempts = 0
        fe._spec_accepts = 0
        fe._spec_full = 0

    def step(self) -> int:
        """Run ONE speculation cycle; returns N (accepted drafts).

        Cycle order is contractual (I3): snap || draft -> verify ->
        accept -> commit tokens -> rollback (partial) -> policy phase
        update -> drafter context commit -> taps shuffle.
        """
        import torch

        fe = self.fe
        K = self._K
        Kv = K + 1
        d = fe._rope_dim

        with torch.no_grad():
            # snap runs on its own stream, overlapped with the drafter.
            snap_stream = fe._snap_stream
            snap_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(snap_stream):
                fe._dflash_snap_state(self.cur_pos, Kv)

            drafts = self.drafter.propose(self.tok)

            torch.cuda.current_stream().wait_stream(snap_stream)

            cos_KN = fe._rope_cos_table[
                self.cur_pos:self.cur_pos + Kv].view(1, Kv, d)
            sin_KN = fe._rope_sin_table[
                self.cur_pos:self.cur_pos + Kv].view(1, Kv, d)
            fe._verify_static_tokens[:, 0:1].copy_(self.tok)
            fe._verify_static_tokens[:, 1:Kv].copy_(drafts.view(1, K))
            fe._verify_static_cos[:, :Kv].copy_(cos_KN)
            fe._verify_static_sin[:, :Kv].copy_(sin_KN)
            vg = fe._ensure_verify_graph_dflash_nvfp4(self.cur_pos, Kv)
            gs = fe._graph_stream
            gs.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(gs):
                vg.replay()
            torch.cuda.current_stream().wait_stream(gs)
            logits_KN = fe._K_logits_buf[:Kv]

            all_argmax = logits_KN.argmax(dim=-1)
            matches = self.policy.matches(
                logits_KN[:K], drafts, all_argmax)
            matches_pad = torch.cat([
                matches,
                torch.zeros(1, device=matches.device,
                            dtype=matches.dtype),
            ])
            N = int(matches_pad.argmin().item())
            fe._spec_attempts += 1
            fe._spec_accepts += N

            argmax_at = (lambda j: all_argmax[j:j + 1].view(1, 1))
            commit_at = self.policy.commit_source(N, drafts, all_argmax)

            if N == K:
                fe._spec_full += 1
                for j in range(Kv):
                    if len(self.generated) < self.max_new_tokens:
                        self.generated.append(commit_at(j))
                self.tok = argmax_at(K)
                self.cur_pos += Kv
            else:
                for j in range(N + 1):
                    if len(self.generated) < self.max_new_tokens:
                        self.generated.append(commit_at(j))
                fe._dflash_partial_rollback(
                    self.cur_pos, N, Kv, self.tok, drafts,
                    cos_KN, sin_KN)
                self.tok = argmax_at(N)
                self.cur_pos += N + 1
            self.policy.observe(N, drafts, all_argmax)
            # I3: the drafter reads feedback rows 0..N; the shuffle
            # below overwrites row 0.
            self.drafter.commit(N)
            # Move taps[N] -> taps[0] as the next drafter input
            # (N == K on a full accept).
            fe._dflash_taps_buf[:, 0].copy_(
                fe._dflash_taps_buf[:, N])
        return N

    def done(self) -> bool:
        return len(self.generated) >= self.max_new_tokens

    def generate(self, input_ids):
        import torch

        self.begin(input_ids)
        while not self.done():
            if self._interrupt:
                break
            self.step()
        if len(self.generated) > self.max_new_tokens:
            self.generated = self.generated[:self.max_new_tokens]
        return torch.cat([input_ids] + self.generated, dim=1)

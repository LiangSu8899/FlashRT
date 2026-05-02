"""FlashRT new-model template — frontend (entry point).

Copy this file to `flash_rt/frontends/torch/<mymodel>_<hw>.py`. This
is the class that `flash_rt.load_model()` instantiates. It owns:

1. Weight loading (delegates to weights_spec.py)
2. CUDA buffer allocation (one-time, all forward buffers)
3. CUDA Graph capture (one-time, after calibration)
4. The public API: `set_prompt(text)` and `infer(observation)`
5. Calibration cache (per-prompt act scales, persisted to disk)

# WHAT YOU TRANSLATE
=====================

Your reference inference code:                 FlashRT frontend mapping:
------------------------------                 -------------------------
model = MyModel.from_pretrained(ckpt)          frontend = MyModelFrontend(ckpt)
                                                  └─→ STEP 1: _load_weights()
model.to("cuda").eval()                        frontend._allocate_buffers()
                                                  └─→ STEP 2: STEP 2 below
prompt_ids = tokenizer(prompt)                 frontend.set_prompt(prompt)
prompt_emb = model.embed_tokens(prompt_ids)       └─→ STEP 4
prompt_kv = model.encoder(prompt_emb,
   images, return_kv=True)                     # implicit, in graph
actions = model.decoder(noise, prompt_kv,
   steps=10)                                   actions = frontend.infer(obs)
                                                  └─→ STEP 6 (replays captured graph)

# LIFECYCLE OF A FRONTEND INSTANCE
==================================

t=0:    __init__() called
        STEP 1  → load weights (uses weights_spec.py, ~1-3s)
        STEP 2  → allocate all CUDA buffers (~10-100MB depending on model)
        STEP 3  → create attention backend (uses attention.py)
                  No CUDA Graph yet — we don't know prompt length S until
                  set_prompt() is called.

t=1s:   set_prompt("pick up the red cup") called for the first time
        STEP 4  → tokenize + embed prompt, fix prompt_len S
        STEP 5  → calibrate FP8 activation scales using a real-data sample
                  (or load from disk cache if same checkpoint+prompt seen before)
                  → capture the entire enc+dec forward into a CUDA Graph
                  → graph replay buffer pointers are now baked in
                  Total time: ~30-60s first time, ~0.1s on cache hit

t=2s:   infer({"image": ..., "wrist_image": ...}) called repeatedly
        STEP 6  → upload observation to fixed input buffers
                  → cudaGraphLaunch (~70ms on Thor for Pi0.5)
                  → download actions
                  Repeats forever at 10-30Hz.
"""

import logging
import pathlib

import numpy as np
import torch

import flash_rt.flash_rt_kernels as fvk
from flash_rt.core.cuda_buffer import CudaBuffer
from flash_rt.core.quant.calibrator import load_calibration, save_calibration

# Local imports — replace with your model's modules after copying.
# from flash_rt.frontends.torch._mymodel_thor_spec import load_weights, NUM_ENCODER_LAYERS, ...
# from flash_rt.models.mymodel.pipeline_thor import (
#     encoder_forward, encoder_forward_calibrate,
#     decoder_forward, decoder_forward_calibrate,
# )
# from flash_rt.hardware.thor.attn_backend import (
#     ThorFlashAttnBackend, make_mymodel_attention_spec,
# )

logger = logging.getLogger(__name__)


class TemplateTorchFrontendThor:
    """Replace `Template` with your model name (e.g. `Pi06`, `Custom`).

    Naming rule: `<Model><Framework>Frontend<Hardware>`. See
    docs/adding_new_model.md §0 rule 2.
    """

    # ──────────────────────────────────────────────────────────────
    # STEP 1: Construction (load weights, allocate buffers)
    # ──────────────────────────────────────────────────────────────

    def __init__(self, checkpoint_dir, num_views=2, autotune=3, **kwargs):
        # TODO: replace these with your model's actual hyperparameters.
        # For Pi0.5 these come from the config.json shipped with the checkpoint.
        self.D = 2048           # hidden dim
        self.NH = 8             # num query heads
        self.NUM_KV = 1         # num KV heads (GQA)
        self.HD = 256           # head dim
        self.H_ffn = 16384      # FFN intermediate size
        self.num_enc_layers = 18
        self.num_dec_layers = 18
        self.num_steps = 10     # diffusion steps
        self.action_horizon = 10
        self.action_dim = 7     # robot action vector dim

        self.checkpoint_dir = pathlib.Path(checkpoint_dir)
        self.num_views = num_views
        self.Se = None          # set in set_prompt — vision_token_count + prompt_len

        # STEP 1a: Load weights (calls into weights_spec.py)
        # Returns dict {"fp8": {...}, "plain": {...}, "scales": {...}}
        # TODO: from flash_rt.frontends.torch._mymodel_thor_spec import load_weights
        # self.weights = load_weights(str(self.checkpoint_dir / "model.safetensors"))
        self.weights = {}       # placeholder

        # STEP 1b: Upload everything to GPU once.
        # Use CudaBuffer (pinned-pool wrapper around torch.cuda allocator).
        # Each weight slot becomes a CudaBuffer; ptr is what kernels read.
        self._upload_weights_to_gpu()

        # STEP 1c: Allocate working buffers (input, intermediates, output)
        self._allocate_io_buffers()

        # STEP 1d: Create the attention backend.
        # The backend pre-allocates Q/K/V tensors per attention site
        # using the spec from attention.py.
        # TODO: from flash_rt.hardware.thor.attn_backend import (
        #     ThorFlashAttnBackend, make_mymodel_attention_spec,
        # )
        # self._attn_spec = make_mymodel_attention_spec()
        # self._attn = ThorFlashAttnBackend(self._attn_spec)

        # STEP 1e: Create the FvkContext (cuBLAS handle, autotune cache)
        self._ctx = fvk.FvkContext(autotune=autotune)

        # CUDA Graph state — populated lazily in set_prompt
        self._enc_ae_graph = None
        self._current_prompt = None

    def _upload_weights_to_gpu(self):
        """Move every weight from numpy to a CudaBuffer.

        Result: self._weights_gpu = {slot_key: CudaBuffer, ...}
                self._weights_ptr = {slot_key: int_ptr, ...}    # what kernels read
        """
        self._weights_gpu = {}
        self._weights_ptr = {}
        for category in ("fp8", "plain"):
            for slot_key, np_array in self.weights.get(category, {}).items():
                buf = CudaBuffer.from_numpy(np_array)
                self._weights_gpu[slot_key] = buf
                self._weights_ptr[slot_key] = buf.ptr

    def _allocate_io_buffers(self):
        """Allocate input/intermediate/output buffers ONCE.

        Critical: every buffer kernels write to during *_forward must be
        allocated here, NOT lazily inside the forward function. CUDA Graph
        capture bakes in pointer values — late allocation breaks replay.

        Sizes: use the maximum you'll ever see (e.g. max prompt len, max
        action horizon). The kernel S/D/H args determine how much of the
        buffer is actually written/read each call.
        """
        # TODO: enumerate every buffer pipeline.py reads from. For Pi0.5
        # this is ~15 buffers. Common ones:
        max_S = 512
        self._bufs = {
            "x":          CudaBuffer.zeros(max_S * self.D,           dtype=torch.float16),
            "x_norm":     CudaBuffer.zeros(max_S * self.D,           dtype=torch.float16),
            "qkv":        CudaBuffer.zeros(max_S * (self.NH + 2 * self.NUM_KV) * self.HD, dtype=torch.float16),
            "attn_out":   CudaBuffer.zeros(max_S * self.D,           dtype=torch.float16),
            "ffn_hid":    CudaBuffer.zeros(max_S * self.H_ffn,       dtype=torch.float16),
            # FP8 staging buffers (one per quantization point)
            "x_norm_fp8":     CudaBuffer.zeros(max_S * self.D,       dtype=torch.float8_e4m3fn),
            "attn_out_fp8":   CudaBuffer.zeros(max_S * self.D,       dtype=torch.float8_e4m3fn),
            # ... add the rest from pipeline.py's bufs[] reads
        }
        self._bufs_ptr = {k: v.ptr for k, v in self._bufs.items()}

    # ──────────────────────────────────────────────────────────────
    # STEP 4: set_prompt — first-call calibration + graph capture
    # ──────────────────────────────────────────────────────────────

    def set_prompt(self, prompt_text, state=None):
        """Tokenize + embed prompt, then capture/load the CUDA Graph.

        Same prompt twice → cache hit, returns immediately.
        New prompt → re-calibrate activation scales (~30s) and re-capture
        graph (~1s).
        """
        if prompt_text == self._current_prompt:
            return                           # no-op fast path

        # STEP 4a: Tokenize and embed the prompt.
        # TODO: use whatever tokenizer your model expects. Most VLAs use
        # PaliGemma's SentencePiece tokenizer; FlashRT ships a helper:
        # from flash_rt.core.thor_frontend_utils import embed_prompt_torch
        # prompt_emb_np = embed_prompt_torch(prompt_text, ...)
        prompt_emb_np = np.zeros((128, self.D), dtype=np.float16)  # placeholder
        prompt_len = prompt_emb_np.shape[0]

        # STEP 4b: Compute the full encoder input length.
        # vision_token_count depends on num_views + image patch grid.
        # E.g. SigLIP at 224px = 14*14 = 196 patches; 2 views = 392 tokens.
        vision_token_count = 196 * self.num_views
        self.Se = vision_token_count + prompt_len

        # Upload prompt embedding to a fixed input slot
        # (must be the SAME ptr the captured graph reads from!)
        self._bufs["prompt_emb"].copy_from_numpy(prompt_emb_np)

        # STEP 4c: Try the calibration cache first.
        cache_key = (self.checkpoint_dir, prompt_text, self.Se)
        cached_scales = load_calibration(cache_key)
        if cached_scales is not None:
            self._install_act_scales(cached_scales)
        else:
            # STEP 4d: Calibrate — run *_forward_calibrate with a real
            # observation to measure activation amax at every quantization
            # point. This populates self._weights_ptr["scales"][...].
            self._recalibrate_with_real_data()
            save_calibration(cache_key, self._extract_act_scales())

        # STEP 5: Capture the production CUDA Graph.
        # After capture, `self._enc_ae_graph` can be replayed with
        # ~zero CPU overhead from infer().
        self._capture_enc_ae_graph()

        self._current_prompt = prompt_text

    def _recalibrate_with_real_data(self):
        """Run *_forward_calibrate once with a representative observation.

        For multi-sample calibration (better accuracy when the deployment
        distribution is wider than one frame), see _calibrate_multi_frame
        in pi05_thor.py and docs/calibration.md §10.
        """
        # TODO: load a representative single observation (e.g. first frame
        # of a LIBERO rollout, or the user's own data sample).
        # Then call:
        #   encoder_forward_calibrate(self._ctx, fvk, self._bufs_ptr,
        #                              self._weights_ptr, self._dims,
        #                              act_scales=self._act_scales)
        #   decoder_forward_calibrate(...)
        # Both populate self._act_scales as a side effect.
        ...

    def _capture_enc_ae_graph(self):
        """Capture the entire encoder + decoder forward into a CUDA Graph.

        The graph captures POINTER values, not tensor contents — so
        subsequent replays read whatever is in the input buffers at
        replay time. This is how infer() can change images/noise without
        re-capturing.
        """
        # The standard pattern (mirror pi05_thor.py's _capture_enc_ae_graph):
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            # Warm-up the kernels first (not in graph) so cuBLAS picks
            # tactics. Otherwise the graph captures cuBLAS handshake too.
            from flash_rt.models.mymodel.pipeline_thor import encoder_forward, decoder_forward
            encoder_forward(self._ctx, fvk, self._bufs_ptr, self._weights_ptr,
                            self._dims, stream=s.cuda_stream, attn=self._attn)
            decoder_forward(self._ctx, fvk, self._bufs_ptr, self._weights_ptr,
                            self._dims, stream=s.cuda_stream, attn=self._attn)
            torch.cuda.current_stream().wait_stream(s)

            # Now capture
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=s):
                encoder_forward(self._ctx, fvk, self._bufs_ptr, self._weights_ptr,
                                self._dims, stream=s.cuda_stream, attn=self._attn)
                decoder_forward(self._ctx, fvk, self._bufs_ptr, self._weights_ptr,
                                self._dims, stream=s.cuda_stream, attn=self._attn)
            self._enc_ae_graph = graph

    # ──────────────────────────────────────────────────────────────
    # STEP 6: infer — the hot path (called every robot tick)
    # ──────────────────────────────────────────────────────────────

    def infer(self, observation):
        """Replay the captured graph with new image inputs.

        Args:
            observation: dict with at least "image" (numpy uint8 H×W×3).
                         For multi-view, also "wrist_image" or "image_2".
                         "state" is the robot state (used by some models).

        Returns:
            {"actions": np.ndarray of shape (action_horizon, action_dim)}
        """
        # STEP 6a: Upload images to the fixed input buffers.
        # MUST be the same buffer the graph captured against, else replay
        # reads stale data (or worse, garbage).
        for view_idx in range(self.num_views):
            key = "image" if view_idx == 0 else f"wrist_image"  # adjust to your conventions
            img = observation[key]
            self._bufs["images"].copy_from_numpy_view(img, offset=view_idx * IMG_SIZE)

        # STEP 6b: Sample / inject noise for the diffusion decoder.
        # For deterministic eval, use a fixed seed. For deployment, use
        # /dev/urandom or `torch.randn` — but allocate the buffer outside
        # the graph capture path (you ARE re-uploading, not capturing).
        noise = np.random.randn(self.action_horizon, self.action_dim).astype(np.float16)
        self._bufs["noise"].copy_from_numpy(noise)

        # STEP 6c: Replay! This is what makes inference fast.
        self._enc_ae_graph.replay()
        torch.cuda.synchronize()    # wait for graph to finish

        # STEP 6d: Download actions from the output buffer
        actions_np = self._bufs["actions_out"].to_numpy().reshape(self.action_horizon, self.action_dim)

        # STEP 6e (optional): unnormalize actions (LIBERO/your-dataset specific)
        # from flash_rt.core.utils.actions import unnormalize_actions
        # actions_np = unnormalize_actions(actions_np, ...)

        return {"actions": actions_np}


# ──────────────────────────────────────────────────────────────────
# DONE-CHECKLIST (verify before deleting the # TODO markers)
# ──────────────────────────────────────────────────────────────────
# - [ ] Class named <Model><Fw>Frontend<Hw> per docs/adding_new_model.md §0 rule 2
# - [ ] Buffer pointers used by graph capture and infer() are identical
#       (no fresh allocations in infer() that break the captured graph).
# - [ ] set_prompt() returns ~immediately on cache hit (verify with timing).
# - [ ] First infer() after set_prompt() completes without recapture.
# - [ ] Calibration cache file appears under ~/.flash_rt/calibration/
#       after the first successful set_prompt().
# - [ ] cosine vs your reference PyTorch FP32 output is >= 0.998 on a
#       representative observation (use /tmp/pytorch_reference.npz pattern
#       from existing tests/test_all_models_precision.py).

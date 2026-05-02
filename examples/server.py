#!/usr/bin/env python3
"""
FlashVLA — HTTP inference server.

Loads model once at startup, serves predictions via REST API.

Usage:
    pip install fastapi uvicorn

    # Torch:
    python examples/server.py \
        --checkpoint /path/to/pi05_checkpoint

    # JAX:
    python examples/server.py \
        --checkpoint /path/to/orbax_checkpoint \
        --framework jax

    # Test:
    curl -X POST http://localhost:8000/predict \
        -H "Content-Type: application/json" \
        -d '{"prompt": "pick up the red block", "image_shape": [224, 224, 3]}'
"""

import argparse
import asyncio
import base64
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="FlashVLA inference server")
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--framework', default='torch', choices=['torch', 'jax'])
    parser.add_argument('--num_views', type=int, default=2)
    parser.add_argument('--autotune', type=int, default=3)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--recalibrate', action='store_true')
    return parser.parse_args()


args = parse_args()

# ── Load model at startup ──
logger.info("Loading model: %s (%s)", args.checkpoint, args.framework)
t0 = time.time()

import flash_rt

model = flash_rt.load_model(
    checkpoint=args.checkpoint,
    framework=args.framework,
    num_views=args.num_views,
    autotune=args.autotune,
    recalibrate=args.recalibrate,
)
logger.info("Model ready in %.1fs", time.time() - t0)

# ── FastAPI app ──
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="FlashVLA", version=flash_rt.__version__)

# Single-GPU lock — only one inference at a time
_lock = asyncio.Lock()


class PredictRequest(BaseModel):
    images: Optional[List[str]] = None  # base64-encoded raw uint8 (224*224*3 bytes)
    prompt: Optional[str] = None
    image_shape: Optional[List[int]] = [224, 224, 3]  # for dummy images when images=None


class PredictResponse(BaseModel):
    actions: List[List[float]]
    latency_ms: float
    shape: List[int]


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "framework": model.framework,
        "version": flash_rt.__version__,
        "prompt": model.prompt,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    async with _lock:
        try:
            t0 = time.perf_counter()

            # Decode images
            if req.images is not None:
                imgs = []
                for img_b64 in req.images:
                    raw = base64.b64decode(img_b64)
                    arr = np.frombuffer(raw, dtype=np.uint8).reshape(req.image_shape)
                    imgs.append(arr)
            else:
                # Dummy images for testing
                h, w, c = req.image_shape
                imgs = [np.random.randint(0, 255, (h, w, c), dtype=np.uint8)
                        for _ in range(args.num_views)]

            actions = model.predict(images=imgs, prompt=req.prompt)
            latency = (time.perf_counter() - t0) * 1000

            return PredictResponse(
                actions=actions.tolist(),
                latency_ms=round(latency, 2),
                shape=list(actions.shape),
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:
            logger.exception("Prediction failed")
            raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    import uvicorn
    logger.info("Starting server on %s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

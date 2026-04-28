"""
OmniVoice Model Server — True Batching HTTP Service

Loads OmniVoice model on a specific GPU.
All requests (single /infer and batch /infer_batch) are routed through
a shared queue. batch_worker collects items within a time window and
processes them in a single model.generate() forward pass.

Usage:
    GPU_ID=0 WORKER_PORT=9000 python model_server.py

Scale with MPS (run multiple instances on same GPU):
    GPU_ID=0 WORKER_PORT=9000 python model_server.py &
    GPU_ID=0 WORKER_PORT=9001 python model_server.py &
    GPU_ID=0 WORKER_PORT=9002 python model_server.py &

    # Route with any load balancer (nginx, haproxy, your own gateway)
    # Enable NVIDIA MPS on host for true concurrent GPU execution:
    #   sudo nvidia-smi -i 0 -c EXCLUSIVE_PROCESS
    #   nvidia-cuda-mps-control -d
"""

import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import hashlib
import shutil
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from prometheus_client import Counter, Gauge, Histogram, Info
from pydantic import BaseModel, Field

# ─── Config ───────────────────────────────────────────────
GPU_ID = int(os.environ.get("GPU_ID", "0"))
WORKER_PORT = int(os.environ.get("WORKER_PORT", "9000"))
MODEL_ID = os.environ.get("MODEL_ID", "k2-fsa/OmniVoice")

BATCH_WINDOW_MS = int(os.environ.get("BATCH_WINDOW_MS", "50"))
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "64"))
NUM_STEP = int(os.environ.get("NUM_STEP", "16"))
USE_TORCH_COMPILE = os.environ.get("USE_TORCH_COMPILE", "true").lower() == "true"
VOICES_DIR = os.environ.get("VOICES_DIR", "./voices")

logging.basicConfig(
    level=logging.INFO,
    format=f"%(asctime)s [GPU-{GPU_ID}] [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ─── Request / Response Models ────────────────────────────
class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10_000)
    voice_id: Optional[str] = None       # ← registered voice clone ID
    instruct: Optional[str] = None       # ← voice design mode
    ref_audio_path: Optional[str] = None # ← one-off voice clone (no pre-register)
    ref_text: Optional[str] = None
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    language: str = Field(default="en")


class TTSBatchRequest(BaseModel):
    """Multiple texts to process in a single forward pass."""
    texts: list[str] = Field(..., min_length=1)
    voice_id: Optional[str] = None
    instruct: Optional[str] = None
    ref_audio_path: Optional[str] = None
    ref_text: Optional[str] = None
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    language: str = Field(default="en")


class VoiceRegisterResponse(BaseModel):
    voice_id: str
    name: str
    ref_text: str
    duration_s: float


class TTSResponse(BaseModel):
    audio_hex: str  # hex-encoded raw audio bytes
    sample_rate: int
    duration_s: float
    latency_ms: float


# ─── Batch Queue Item ─────────────────────────────────────
@dataclass
class QueueItem:
    request: TTSRequest
    future: asyncio.Future
    enqueue_time: float = field(default_factory=time.monotonic)


# ─── Model Server App ─────────────────────────────────────
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(title=f"OmniVoice Model Server (GPU {GPU_ID})")
Instrumentator().instrument(app).expose(app)  # auto /metrics endpoint

# ─── Custom Prometheus Metrics ────────────────────────────
LABELS = {"gpu_id": str(GPU_ID), "port": str(WORKER_PORT)}

ACTIVE_REQUESTS = Gauge(
    "tts_active_requests",
    "Number of requests currently being processed",
    ["gpu_id", "port"],
)
QUEUE_DEPTH = Gauge(
    "tts_queue_depth",
    "Number of items waiting in batch queue",
    ["gpu_id", "port"],
)
BATCH_SIZE = Histogram(
    "tts_batch_size",
    "Number of items per batch forward pass",
    ["gpu_id", "port"],
    buckets=[1, 2, 4, 8, 16, 32, 48, 64],
)
INFERENCE_LATENCY = Histogram(
    "tts_inference_latency_ms",
    "End-to-end inference latency per request (ms)",
    ["gpu_id", "port"],
    buckets=[10, 25, 50, 100, 250, 500, 1000, 2500],
)
GPU_FORWARD_LATENCY = Histogram(
    "tts_gpu_forward_latency_ms",
    "GPU forward pass latency per batch (ms)",
    ["gpu_id", "port"],
    buckets=[5, 10, 25, 50, 100, 250, 500],
)
ITEMS_PROCESSED = Counter(
    "tts_items_processed_total",
    "Total number of text items processed",
    ["gpu_id", "port"],
)
BATCHES_PROCESSED = Counter(
    "tts_batches_processed_total",
    "Total number of batch forward passes",
    ["gpu_id", "port"],
)
INFERENCE_ERRORS = Counter(
    "tts_inference_errors_total",
    "Total number of failed inference attempts",
    ["gpu_id", "port"],
)
GPU_VRAM_USED = Gauge(
    "tts_gpu_vram_used_bytes",
    "GPU VRAM currently used by this process",
    ["gpu_id", "port"],
)
SERVER_INFO = Info(
    "tts_server",
    "Model server metadata",
)

# Global state
model = None
batch_queue: asyncio.Queue = None
executor: ThreadPoolExecutor = None

# Voice clone cache: voice_id → VoiceClonePrompt (in-memory)
voice_cache: dict = {}
voice_meta: dict = {}   # voice_id → {name, ref_text, duration_s}


def load_model():
    """Load OmniVoice model onto assigned GPU with optional torch.compile."""
    global model
    from omnivoice.models.omnivoice import OmniVoice

    device = f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading model '{MODEL_ID}' on {device}...")

    model = OmniVoice.from_pretrained(
        MODEL_ID,
        device_map=device,
        dtype=torch.float16,
    )

    if USE_TORCH_COMPILE and device.startswith("cuda"):
        logger.info("Applying torch.compile(mode='max-autotune')...")
        model = torch.compile(model, mode="max-autotune")
        logger.info("torch.compile applied. First inference will be slow (compiling).")

    logger.info(f"Model loaded. VRAM used: {torch.cuda.memory_allocated(GPU_ID) / 1e9:.1f}GB")

    SERVER_INFO.info({
        "model_id": MODEL_ID,
        "gpu_id": str(GPU_ID),
        "port": str(WORKER_PORT),
        "torch_compile": str(USE_TORCH_COMPILE),
    })


def load_persisted_voices():
    """Load all .pt voice prompts saved to VOICES_DIR on startup."""
    os.makedirs(VOICES_DIR, exist_ok=True)
    count = 0
    for pt_file in os.listdir(VOICES_DIR):
        if not pt_file.endswith(".pt"):
            continue
        voice_id = pt_file[:-3]
        try:
            data = torch.load(os.path.join(VOICES_DIR, pt_file), map_location="cpu")
            from omnivoice.models.omnivoice import VoiceClonePrompt
            voice_cache[voice_id] = VoiceClonePrompt(
                ref_audio_tokens=data["ref_audio_tokens"],
                ref_text=data["ref_text"],
                ref_rms=data["ref_rms"],
            )
            voice_meta[voice_id] = data.get("meta", {"name": voice_id})
            count += 1
        except Exception as e:
            logger.warning(f"Failed to load voice {voice_id}: {e}")
    logger.info(f"Loaded {count} persisted voice(s) from {VOICES_DIR}")


def run_batch_inference(
    texts: list[str],
    languages: list[str],
    voice_prompts: list = None,   # list[VoiceClonePrompt | None]
) -> list[np.ndarray]:
    """Run true batched inference — single forward pass for N texts.

    OmniVoice model.generate() accepts:
      text: Union[str, list[str]]
      language: Union[str, list[str], None]
      voice_clone_prompt: VoiceClonePrompt (reusable, no audio reload)
    Returns list[np.ndarray] where each array is shape (T,) at model.sampling_rate.
    """
    # If all items share the same non-None prompt, pass it directly (common case)
    # Otherwise run individually to handle mixed prompts
    if voice_prompts and any(p is not None for p in voice_prompts):
        # Run each item separately when prompts differ (can't batch mixed prompts)
        unique_prompts = list(set(id(p) for p in voice_prompts if p is not None))
        if len(unique_prompts) == 1 and all(p is not None for p in voice_prompts):
            # All same prompt → single batched forward pass
            audios = model.generate(
                text=texts,
                language=languages,
                voice_clone_prompt=voice_prompts[0],
                num_step=NUM_STEP,
            )
        else:
            # Mixed prompts → one call per item
            audios = []
            for text, lang, prompt in zip(texts, languages, voice_prompts):
                result = model.generate(
                    text=text,
                    language=lang,
                    voice_clone_prompt=prompt,
                    num_step=NUM_STEP,
                )
                audios.append(result[0])
    else:
        audios = model.generate(
            text=texts,
            language=languages,
            num_step=NUM_STEP,
        )
    return audios


async def batch_worker():
    """
    Continuously collect requests from queue, form batches,
    and dispatch to GPU via ThreadPool.
    """
    logger.info(
        f"Batch worker started (window={BATCH_WINDOW_MS}ms, max_batch={MAX_BATCH_SIZE})"
    )

    while True:
        batch: list[QueueItem] = []

        # Wait for first request (blocking)
        try:
            item = await batch_queue.get()
            batch.append(item)
        except Exception:
            continue

        # Collect more requests within time window
        deadline = time.monotonic() + BATCH_WINDOW_MS / 1000.0
        while len(batch) < MAX_BATCH_SIZE:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                item = await asyncio.wait_for(
                    batch_queue.get(), timeout=remaining
                )
                batch.append(item)
            except asyncio.TimeoutError:
                break

        # Process batch
        batch_size = len(batch)
        batch_start = time.monotonic()
        logger.info(f"Processing batch of {batch_size} requests")

        # Record metrics
        ACTIVE_REQUESTS.labels(**LABELS).inc(batch_size)
        QUEUE_DEPTH.labels(**LABELS).set(batch_queue.qsize())
        BATCH_SIZE.labels(**LABELS).observe(batch_size)

        try:
            texts = [item.request.text for item in batch]
            langs = [item.request.language for item in batch]
            prompts = [
                voice_cache.get(item.request.voice_id)
                if item.request.voice_id else None
                for item in batch
            ]

            # Run inference in thread pool (blocking GPU call)
            loop = asyncio.get_event_loop()
            audios = await loop.run_in_executor(
                executor, run_batch_inference, texts, langs, prompts
            )

            batch_latency = (time.monotonic() - batch_start) * 1000
            GPU_FORWARD_LATENCY.labels(**LABELS).observe(batch_latency)
            BATCHES_PROCESSED.labels(**LABELS).inc()
            ITEMS_PROCESSED.labels(**LABELS).inc(batch_size)

            # Deliver results to each waiting request
            # model.generate() returns list[np.ndarray], not tensors
            for item, audio in zip(batch, audios):
                sample_rate = model.sampling_rate
                duration_s = len(audio) / sample_rate
                req_latency = (time.monotonic() - item.enqueue_time) * 1000

                INFERENCE_LATENCY.labels(**LABELS).observe(req_latency)

                item.future.set_result(TTSResponse(
                    audio_hex=audio.tobytes().hex(),
                    sample_rate=sample_rate,
                    duration_s=round(duration_s, 3),
                    latency_ms=round(req_latency, 1),
                ))

            logger.info(
                f"Batch {batch_size} done in {batch_latency:.0f}ms "
                f"({batch_latency/batch_size:.0f}ms/item)"
            )

        except Exception as e:
            logger.error(f"Batch inference failed: {e}")
            INFERENCE_ERRORS.labels(**LABELS).inc()
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(
                        HTTPException(status_code=500, detail=f"Inference failed: {e}")
                    )
        finally:
            ACTIVE_REQUESTS.labels(**LABELS).dec(batch_size)
            if torch.cuda.is_available():
                GPU_VRAM_USED.labels(**LABELS).set(
                    torch.cuda.memory_allocated(GPU_ID)
                )


@app.on_event("startup")
async def startup():
    global batch_queue, executor

    load_model()
    load_persisted_voices()

    batch_queue = asyncio.Queue(maxsize=200)
    executor = ThreadPoolExecutor(max_workers=1)  # 1 GPU = 1 thread

    asyncio.create_task(batch_worker())
    logger.info(f"Server ready on port {WORKER_PORT}")


# ─── Voice Clone Endpoints ────────────────────────────────
@app.post("/voices", response_model=VoiceRegisterResponse)
async def register_voice(
    name: str,
    file: UploadFile = File(...),
    ref_text: Optional[str] = None,
):
    """Upload a reference audio file to create a reusable voice clone.

    - Saves audio to VOICES_DIR/{voice_id}.wav
    - Creates VoiceClonePrompt (tokenizes audio → tensor)
    - Persists prompt to VOICES_DIR/{voice_id}.pt (survives restart)
    - Returns voice_id for use in /infer and /infer_batch
    """
    from omnivoice.models.omnivoice import VoiceClonePrompt  # noqa

    os.makedirs(VOICES_DIR, exist_ok=True)

    # Read uploaded file
    audio_bytes = await file.read()

    # Generate deterministic voice_id from content hash + name
    content_hash = hashlib.sha256(audio_bytes).hexdigest()[:12]
    voice_id = f"{name}_{content_hash}"

    audio_path = os.path.join(VOICES_DIR, f"{voice_id}.wav")
    pt_path = os.path.join(VOICES_DIR, f"{voice_id}.pt")

    # Save audio to disk
    with open(audio_path, "wb") as f:
        f.write(audio_bytes)

    # Create VoiceClonePrompt (tokenize audio → tensor, extract ref_text via Whisper if needed)
    loop = asyncio.get_event_loop()
    prompt = await loop.run_in_executor(
        executor,
        model.create_voice_clone_prompt,
        audio_path,
        ref_text,
    )

    # Cache in memory
    voice_cache[voice_id] = prompt
    meta = {
        "name": name,
        "ref_text": prompt.ref_text,
        "duration_s": round(
            prompt.ref_audio_tokens.shape[-1] / model.audio_tokenizer.config.frame_rate, 2
        ),
    }
    voice_meta[voice_id] = meta

    # Persist to disk (survives restart)
    torch.save({
        "ref_audio_tokens": prompt.ref_audio_tokens.cpu(),
        "ref_text": prompt.ref_text,
        "ref_rms": prompt.ref_rms,
        "meta": meta,
    }, pt_path)

    logger.info(f"Voice registered: {voice_id} (ref_text='{prompt.ref_text[:50]}...'")

    return VoiceRegisterResponse(
        voice_id=voice_id,
        name=name,
        ref_text=prompt.ref_text,
        duration_s=meta["duration_s"],
    )


@app.get("/voices")
async def list_voices():
    """List all registered voice clones."""
    return {
        "voices": [
            {"voice_id": vid, **meta}
            for vid, meta in voice_meta.items()
        ],
        "total": len(voice_meta),
    }


@app.delete("/voices/{voice_id}")
async def delete_voice(voice_id: str):
    """Remove a registered voice clone from cache and disk."""
    if voice_id not in voice_cache:
        raise HTTPException(status_code=404, detail=f"Voice '{voice_id}' not found")

    voice_cache.pop(voice_id)
    voice_meta.pop(voice_id, None)

    # Remove persisted files
    for ext in (".pt", ".wav"):
        path = os.path.join(VOICES_DIR, f"{voice_id}{ext}")
        if os.path.exists(path):
            os.remove(path)

    return {"deleted": voice_id}


@app.post("/infer", response_model=TTSResponse)
async def infer(request: TTSRequest):
    """Submit single TTS request to batching queue."""
    if batch_queue.full():
        raise HTTPException(status_code=429, detail="Server at capacity")

    future = asyncio.get_event_loop().create_future()
    await batch_queue.put(QueueItem(request=request, future=future))

    try:
        result = await asyncio.wait_for(future, timeout=120.0)
        return result
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Inference timeout")


@app.post("/infer_batch")
async def infer_batch(request: TTSBatchRequest):
    """Submit multiple texts — routed through shared batch queue.

    Each text is enqueued individually so batch_worker can merge them
    with other concurrent requests for optimal GPU utilization (cross-client batching).
    """
    texts = request.texts
    if len(texts) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Batch too large: {len(texts)} > {MAX_BATCH_SIZE}",
        )

    if batch_queue.qsize() + len(texts) > batch_queue.maxsize:
        raise HTTPException(status_code=429, detail="Server at capacity")

    start = time.monotonic()
    loop = asyncio.get_event_loop()

    futures = []
    for text in texts:
        future = loop.create_future()
        single_req = TTSRequest(
            text=text,
            voice=request.voice,
            instruct=request.instruct,
            ref_audio_path=request.ref_audio_path,
            ref_text=request.ref_text,
            speed=request.speed,
            language=request.language,
        )
        await batch_queue.put(QueueItem(request=single_req, future=future))
        futures.append(future)

    try:
        responses: list[TTSResponse] = await asyncio.wait_for(
            asyncio.gather(*futures), timeout=120.0
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Inference timeout")

    latency_ms = (time.monotonic() - start) * 1000

    return {
        "results": [
            {
                "audio_hex": r.audio_hex,
                "sample_rate": r.sample_rate,
                "duration_s": r.duration_s,
            }
            for r in responses
        ],
        "batch_size": len(texts),
        "latency_ms": round(latency_ms, 1),
    }


@app.get("/health")
async def health():
    """Health check with GPU metrics."""
    gpu_mem = torch.cuda.memory_allocated(GPU_ID) / 1e9 if torch.cuda.is_available() else 0
    return {
        "status": "ok",
        "gpu_id": GPU_ID,
        "gpu_memory_gb": round(gpu_mem, 2),
        "queue_depth": batch_queue.qsize() if batch_queue else 0,
        "model_loaded": model is not None,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=WORKER_PORT, workers=1, log_level="info")

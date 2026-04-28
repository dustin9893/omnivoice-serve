# omnivoice-serve

Production-grade batching inference server for [OmniVoice](https://github.com/k2-fsa/OmniVoice) TTS.

## Features

- **Dynamic batching** — cross-client request aggregation within a configurable time window
- **torch.compile** — `max-autotune` mode for 50-80% faster inference kernels
- **Prometheus metrics** — 9 custom metrics exposed on `/metrics`
- **MPS/MIG ready** — scale horizontally by running multiple instances per GPU
- **Zero-dependency serving** — single Python file, no Triton or TorchServe required

## Quick Start

```bash
pip install -r requirements.txt

# Start server on GPU 0, port 9000
GPU_ID=0 WORKER_PORT=9000 python model_server.py
```

## API

### `POST /infer` — Single text
```json
{
  "text": "Hello world",
  "language": "en",
  "speed": 1.0
}
```

### `POST /infer_batch` — Multiple texts (cross-client batching)
```json
{
  "texts": ["Hello", "Xin chào", "你好"],
  "language": "en"
}
```

### `GET /health` — Health + GPU info
```json
{
  "status": "ok",
  "gpu_id": 0,
  "gpu_memory_gb": 3.6,
  "queue_depth": 0,
  "model_loaded": true
}
```

### `GET /metrics` — Prometheus metrics

## Configuration

All config via environment variables:

| Variable | Default | Description |
|---|---|---|
| `GPU_ID` | `0` | GPU device index |
| `WORKER_PORT` | `9000` | HTTP listen port |
| `MODEL_ID` | `k2-fsa/OmniVoice` | HuggingFace model ID |
| `BATCH_WINDOW_MS` | `50` | Max wait time to fill a batch (ms) |
| `MAX_BATCH_SIZE` | `64` | Max items per forward pass |
| `NUM_STEP` | `16` | Diffusion steps (lower = faster, less quality) |
| `USE_TORCH_COMPILE` | `true` | Enable torch.compile max-autotune |

## Scaling with MPS (multi-instance on same GPU)

Run multiple instances on the same GPU for concurrent GPU execution:

```bash
# Enable NVIDIA MPS on host (one-time setup):
sudo nvidia-smi -i 0 -c EXCLUSIVE_PROCESS
nvidia-cuda-mps-control -d

# Start 3 instances on GPU 0:
GPU_ID=0 WORKER_PORT=9000 python model_server.py &
GPU_ID=0 WORKER_PORT=9001 python model_server.py &
GPU_ID=0 WORKER_PORT=9002 python model_server.py &

# Route with nginx, haproxy, or any load balancer
```

> **Why MPS?** Without MPS, multiple processes time-slice the GPU (sequential).
> With MPS, kernels from different processes execute concurrently on available SMs.
> Best for models that don't saturate the GPU individually (e.g., OmniVoice 0.6B on A100 80GB).

## Prometheus Metrics

| Metric | Type | Description |
|---|---|---|
| `tts_active_requests` | Gauge | Requests being processed |
| `tts_queue_depth` | Gauge | Items waiting in queue |
| `tts_batch_size` | Histogram | Items per forward pass distribution |
| `tts_inference_latency_ms` | Histogram | End-to-end latency per request |
| `tts_gpu_forward_latency_ms` | Histogram | GPU forward pass time per batch |
| `tts_items_processed_total` | Counter | Total items processed |
| `tts_batches_processed_total` | Counter | Total forward passes |
| `tts_inference_errors_total` | Counter | Total errors |
| `tts_gpu_vram_used_bytes` | Gauge | VRAM consumed by this process |

All metrics include `gpu_id` and `port` labels for multi-instance differentiation.

## Docker

```bash
docker build -t omnivoice-serve .
docker run --gpus '"device=0"' -p 9000:9000 \
  -e GPU_ID=0 \
  -e WORKER_PORT=9000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  omnivoice-serve
```

## Why not Triton?

This server is intentionally simple — a single Python file with no complex config.
Consider Triton if you need:
- Serving 4+ different models on the same server
- Pipeline ensembles (model A → model B)
- gRPC protocol
- Built-in TensorRT backend

For a single OmniVoice model, this server + MPS achieves equivalent throughput with far less complexity.

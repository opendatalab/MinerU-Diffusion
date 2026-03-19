#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${MODEL_PATH:-$SCRIPT_DIR/model}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-31002}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
LOG_PATH="${LOG_PATH:-$SCRIPT_DIR/sglang.log}"

nohup env PYTHONPATH=python CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
sglang serve \
  --model-path "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --tp-size 1 \
  --dllm-algorithm LowConfidence \
  --disable-cuda-graph \
  --attention-backend triton \
  --sampling-backend pytorch \
> "$LOG_PATH" 2>&1 &

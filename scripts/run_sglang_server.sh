#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-$REPO_DIR/model}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-31002}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
TP_SIZE="${TP_SIZE:-1}"
DLLM_ALGORITHM="${DLLM_ALGORITHM:-LowConfidence}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-triton}"
SAMPLING_BACKEND="${SAMPLING_BACKEND:-pytorch}"
LOG_PATH="${LOG_PATH:-$REPO_DIR/sglang.log}"

nohup env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
sglang serve \
  --model-path "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --tp-size "$TP_SIZE" \
  --dllm-algorithm "$DLLM_ALGORITHM" \
  --disable-cuda-graph \
  --attention-backend "$ATTENTION_BACKEND" \
  --sampling-backend "$SAMPLING_BACKEND" \
> "$LOG_PATH" 2>&1 &

echo "SGLang server started in background"
echo "model: $MODEL_PATH"
echo "endpoint: http://$HOST:$PORT/v1/chat/completions"
echo "log: $LOG_PATH"

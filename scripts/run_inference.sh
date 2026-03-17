#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-$REPO_DIR/model}"
IMAGE_PATH="${IMAGE_PATH:-$REPO_DIR/assets/image.png}"
PROMPT_TYPE="text"
PROMPT=""
DEVICE="cuda"
DTYPE="bfloat16"
MAX_LENGTH="4096"
GEN_LENGTH="1024"
BLOCK_SIZE="32"
TEMPERATURE="1.0"
REMASK_STRATEGY="low_confidence_dynamic"
DYNAMIC_THRESHOLD="0.95"

python "$REPO_DIR/scripts/run_inference.py" \
  --engine "${ENGINE:-hf}" \
  --model-path "$MODEL_PATH" \
  --image-path "$IMAGE_PATH" \
  --prompt-type "$PROMPT_TYPE" \
  --prompt "$PROMPT" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --max-length "$MAX_LENGTH" \
  --gen-length "$GEN_LENGTH" \
  --block-size "$BLOCK_SIZE" \
  --temperature "$TEMPERATURE" \
  --remask-strategy "$REMASK_STRATEGY" \
  --dynamic-threshold "$DYNAMIC_THRESHOLD"

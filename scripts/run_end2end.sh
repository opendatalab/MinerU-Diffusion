#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-$REPO_DIR/model}"
IMAGE_PATH="${IMAGE_PATH:-$REPO_DIR/assets/image.png}"
OUTPUT_PATH="${OUTPUT_PATH:-}"
BLOCKS_JSON_PATH="${BLOCKS_JSON_PATH:-}"
SAVE_LAYOUT_IMAGE="${SAVE_LAYOUT_IMAGE:-0}"
LAYOUT_IMAGE_PATH="${LAYOUT_IMAGE_PATH:-}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
LAYOUT_GEN_LENGTH="${LAYOUT_GEN_LENGTH:-2048}"
CONTENT_GEN_LENGTH="${CONTENT_GEN_LENGTH:-1024}"
TABLE_GEN_LENGTH="${TABLE_GEN_LENGTH:-2048}"
FORMULA_GEN_LENGTH="${FORMULA_GEN_LENGTH:-1024}"
BLOCK_SIZE="${BLOCK_SIZE:-32}"
TEMPERATURE="${TEMPERATURE:-1.0}"
REMASK_STRATEGY="${REMASK_STRATEGY:-low_confidence_dynamic}"
DYNAMIC_THRESHOLD="${DYNAMIC_THRESHOLD:-0.95}"
KEEP_PARATEXT="${KEEP_PARATEXT:-0}"
VERBOSE="${VERBOSE:-0}"

ARGS=(
  --model-path "$MODEL_PATH"
  --image-path "$IMAGE_PATH"
  --device "$DEVICE"
  --dtype "$DTYPE"
  --max-length "$MAX_LENGTH"
  --layout-gen-length "$LAYOUT_GEN_LENGTH"
  --content-gen-length "$CONTENT_GEN_LENGTH"
  --table-gen-length "$TABLE_GEN_LENGTH"
  --formula-gen-length "$FORMULA_GEN_LENGTH"
  --block-size "$BLOCK_SIZE"
  --temperature "$TEMPERATURE"
  --remask-strategy "$REMASK_STRATEGY"
  --dynamic-threshold "$DYNAMIC_THRESHOLD"
)

if [[ -n "$OUTPUT_PATH" ]]; then
  ARGS+=(--output-path "$OUTPUT_PATH")
fi

if [[ -n "$BLOCKS_JSON_PATH" ]]; then
  ARGS+=(--blocks-json-path "$BLOCKS_JSON_PATH")
fi

if [[ "$SAVE_LAYOUT_IMAGE" == "1" ]]; then
  ARGS+=(--save-layout-image)
fi

if [[ -n "$LAYOUT_IMAGE_PATH" ]]; then
  ARGS+=(--layout-image-path "$LAYOUT_IMAGE_PATH")
fi

if [[ "$KEEP_PARATEXT" == "1" ]]; then
  ARGS+=(--keep-paratext)
fi

if [[ "$VERBOSE" == "1" ]]; then
  ARGS+=(--verbose)
fi

python "$REPO_DIR/scripts/run_end2end.py" \
  "${ARGS[@]}"

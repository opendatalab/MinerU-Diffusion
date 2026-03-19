# MinerU-Diffusion SGLang Notes

This directory contains the original SGLang prototype and related notes for running MinerU-Diffusion with an OpenAI-compatible SGLang server.

For normal use, prefer the unified repository-level entrypoints:

- `scripts/run_sglang_server.sh`
- `scripts/run_inference.sh` with `ENGINE=sglang`

This document keeps the lower-level SGLang notes in one place for debugging, validation, and reproduction.

## Environment

When running inside an SGLang checkout, make sure the runtime environment is set correctly:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e "python[all]"
```

## Recommended Server Commands

Recommended launch command:

```bash
PYTHONPATH=python CUDA_VISIBLE_DEVICES=0 sglang serve \
  --model-path <MODEL_PATH> \
  --host 127.0.0.1 \
  --port 31000 \
  --tp-size 1 \
  --dllm-algorithm LowConfidence \
  --mem-fraction-static 0.72 \
  --cuda-graph-max-bs 160
```

If you want a more conservative setup, you can disable CUDA graph:

```bash
PYTHONPATH=python CUDA_VISIBLE_DEVICES=0 sglang serve \
  --model-path <MODEL_PATH> \
  --host 127.0.0.1 \
  --port 31000 \
  --tp-size 1 \
  --dllm-algorithm LowConfidence \
  --disable-cuda-graph \
  --attention-backend triton \
  --sampling-backend pytorch
```

## Request Examples

The examples below use:

- `<BASE_URL>` for the server address, for example `http://127.0.0.1:31000`
- `<MODEL_PATH>` for the model directory
- `<IMAGE_PATH>` for the input image

### Formula Recognition

```bash
python - <<'PY'
import base64, json, pathlib, urllib.request

base_url = "<BASE_URL>/v1/chat/completions"
model = "<MODEL_PATH>"
img_path = pathlib.Path("<IMAGE_PATH>")
img_b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")

payload = {
    "model": model,
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Formula Recognition:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            ],
        }
    ],
    "max_tokens": 128,
}

req = urllib.request.Request(
    base_url,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=120) as resp:
    print(resp.read().decode("utf-8"))
PY
```

### Table Recognition

```bash
python - <<'PY'
import base64, json, pathlib, urllib.request

base_url = "<BASE_URL>/v1/chat/completions"
model = "<MODEL_PATH>"
img_path = pathlib.Path("<IMAGE_PATH>")
img_b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")

payload = {
    "model": model,
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Table Recognition:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            ],
        }
    ],
    "max_tokens": 1024,
}

req = urllib.request.Request(
    base_url,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=300) as resp:
    print(resp.read().decode("utf-8"))
PY
```

If the tokenizer is configured correctly, table outputs should preserve structure tokens such as `<fcel>` and `<nl>`.

### Layout Analysis

```bash
python - <<'PY'
import base64, json, pathlib, urllib.request

base_url = "<BASE_URL>/v1/chat/completions"
model = "<MODEL_PATH>"
img_path = pathlib.Path("<IMAGE_PATH>")
img_b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")

payload = {
    "model": model,
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Layout Analysis:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            ],
        }
    ],
    "max_tokens": 1024,
}

req = urllib.request.Request(
    base_url,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=300) as resp:
    print(resp.read().decode("utf-8"))
PY
```

## HF Baseline Comparison

If you have an HF-based demo script, you can compare the outputs with the same image and the same prompt type.

Generic example:

```bash
python <HF_DEMO_SCRIPT> \
  --model-path <MODEL_PATH> \
  --image-path <IMAGE_PATH> \
  --prompt-type table
```

## Recommended Runtime Settings

For a single-GPU setup with correctness as the priority, the recommended settings are:

```bash
--tp-size 1
--dllm-algorithm LowConfidence
--mem-fraction-static 0.72
--cuda-graph-max-bs 160
```

## Prototype Files in This Directory

The following files are preserved from the original standalone prototype:

- `run_server.sh`: starts an SGLang server with hard-coded local paths
- `mineru_request.py`: sends a single test request to the OpenAI-compatible endpoint
- `run_infer.sh`: runs `mineru_request.py`

These files are mainly useful for reproducing the original experiment or debugging low-level SGLang behavior.

#!/usr/bin/env python3
import os
import base64
import json
import pathlib
import urllib.request

urllib.request.install_opener(
    urllib.request.build_opener(urllib.request.ProxyHandler({}))
)

script_dir = pathlib.Path(__file__).resolve().parent
base_url = os.environ.get("BASE_URL", "http://127.0.0.1:31002/v1/chat/completions")
model = os.environ.get("MODEL_PATH", str(script_dir / "model"))
img_path = pathlib.Path(os.environ.get("IMAGE_PATH", str(script_dir / "image.png")))
img_b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")

payload = {
    "model": model,
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Text Recognition:"},
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

resp = json.loads(urllib.request.urlopen(req, timeout=180).read().decode("utf-8"))
content = resp["choices"][0]["message"]["content"]
print(content)

import argparse
import base64
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from shutil import get_terminal_size

from termcolor import cprint


STOP_STRINGS = ("<|endoftext|>", "<|im_end|>")
SYSTEM_PROMPT = "You are a helpful assistant."
TASK_PROMPTS = {
    "text": "\nText Recognition:",
    "table": "\nTable Recognition:",
    "formula": "\nFormula Recognition:",
    "layout": "\nLayout Analysis:",
}
DEFAULT_MAX_LENGTH = 4096
DEFAULT_GEN_LENGTH = 1024
DEFAULT_SERVER_URL = "http://127.0.0.1:31002/v1/chat/completions"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path", required=True, help="Model path registered by the SGLang server.")
    parser.add_argument(
        "--image-path",
        default=str(Path(__file__).resolve().parents[2] / "assets" / "image.png"),
        help="Input image path.",
    )
    parser.add_argument("--prompt-type", choices=sorted(TASK_PROMPTS.keys()), default="text")
    parser.add_argument("--prompt", default=None, help="Optional custom prompt.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--gen-length", type=int, default=DEFAULT_GEN_LENGTH)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--remask-strategy", default="low_confidence_dynamic")
    parser.add_argument("--dynamic-threshold", type=float, default=0.95)
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL, help="OpenAI-compatible SGLang endpoint.")
    parser.add_argument("--request-timeout", type=int, default=180, help="HTTP request timeout in seconds.")
    parser.add_argument("--no-system-prompt", action="store_true", help="Do not prepend the default system prompt.")


def _print_summary(args: argparse.Namespace, model_path: Path) -> None:
    line = "=" * max(72, min(get_terminal_size((96, 20)).columns, 120))
    cprint(line, color="cyan", attrs=["bold"], flush=True)
    cprint("MinerU SGLang Inference", color="cyan", attrs=["bold"], flush=True)
    cprint("-" * len(line), color="cyan", flush=True)
    cprint(f"{'model':<12}: {model_path}", color="white", flush=True)
    cprint(f"{'image':<12}: {Path(args.image_path).resolve()}", color="white", flush=True)
    cprint(f"{'prompt':<12}: {'custom' if args.prompt else args.prompt_type}", color="white", flush=True)
    cprint(f"{'server_url':<12}: {args.server_url}", color="white", flush=True)
    cprint(f"{'gen_length':<12}: {args.gen_length}", color="white", flush=True)
    cprint(f"{'block_size':<12}: {args.block_size}", color="white", flush=True)
    cprint(f"{'temperature':<12}: {args.temperature}", color="white", flush=True)
    cprint(f"{'remask':<12}: {args.remask_strategy}", color="white", flush=True)


def _print_response(response: str, elapsed: float) -> None:
    line = "=" * max(72, min(get_terminal_size((96, 20)).columns, 120))
    cprint(line, color="green", attrs=["bold"], flush=True)
    cprint("MinerU Result", color="green", attrs=["bold"], flush=True)
    cprint(line, color="green", attrs=["bold"], flush=True)
    cprint(response or "(empty response)", color="white", flush=True)
    cprint(line, color="green", attrs=["bold"], flush=True)
    cprint(f"elapsed: {elapsed:.2f}s", color="cyan", flush=True)


def _build_messages(args: argparse.Namespace, prompt: str, image_url: str) -> list[dict]:
    messages: list[dict] = []
    if not args.no_system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    )
    return messages


def _build_payload(args: argparse.Namespace, prompt: str, image_path: Path) -> bytes:
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    payload = {
        "model": str(Path(args.model_path).resolve()),
        "messages": _build_messages(args, prompt, f"data:image/{image_path.suffix.lstrip('.').lower() or 'png'};base64,{image_b64}"),
        "max_tokens": args.gen_length,
        "temperature": args.temperature,
        "extra_body": {
            "denoising_steps": args.block_size,
            "block_length": args.block_size,
            "denoising_strategy": args.remask_strategy,
            "dynamic_threshold": args.dynamic_threshold,
        },
    }
    return json.dumps(payload).encode("utf-8")


def _send_request(args: argparse.Namespace, payload: bytes) -> dict:
    urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))
    request = urllib.request.Request(
        args.server_url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=args.request_timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"SGLang request failed with HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Unable to reach SGLang server at {args.server_url}: {exc}") from exc


def run(args: argparse.Namespace) -> None:
    prompt = args.prompt or TASK_PROMPTS[args.prompt_type]
    model_path = Path(args.model_path).resolve()
    image_path = Path(args.image_path).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    start_time = time.time()
    _print_summary(args, model_path)
    response = _send_request(args, _build_payload(args, prompt, image_path))
    content = response["choices"][0]["message"]["content"]
    if isinstance(content, list):
        content = "".join(item.get("text", "") for item in content if isinstance(item, dict))
    for stop in STOP_STRINGS:
        content = content.split(stop, 1)[0]
    _print_response(str(content).strip(), time.time() - start_time)


__all__ = ["add_arguments", "run"]

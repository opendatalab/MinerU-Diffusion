import argparse
import json
import sys
import time
from pathlib import Path
from shutil import get_terminal_size

import torch
from termcolor import cprint


NANO_DVLM_DIR = Path(__file__).resolve().parent
if str(NANO_DVLM_DIR) not in sys.path:
    sys.path.insert(0, str(NANO_DVLM_DIR))

from nanovllm import LLM, SamplingParams


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


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path", required=True, help="Converted HF model directory.")
    parser.add_argument(
        "--image-path",
        default=str(Path(__file__).resolve().parents[2] / "assets" / "demo.jpg"),
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
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--enforce-eager", action="store_true")


def _print_summary(args: argparse.Namespace, model_path: Path, mask_token_id: int) -> None:
    line = "=" * max(72, min(get_terminal_size((96, 20)).columns, 120))
    cprint(line, color="cyan", attrs=["bold"], flush=True)
    cprint("MinerU Nano-DVLM Inference", color="cyan", attrs=["bold"], flush=True)
    cprint("-" * len(line), color="cyan", flush=True)
    cprint(f"{'model':<12}: {model_path}", color="white", flush=True)
    cprint(f"{'image':<12}: {Path(args.image_path).resolve()}", color="white", flush=True)
    cprint(f"{'prompt':<12}: {'custom' if args.prompt else args.prompt_type}", color="white", flush=True)
    cprint(f"{'device':<12}: {args.device}", color="white", flush=True)
    cprint(f"{'dtype':<12}: {args.dtype}", color="white", flush=True)
    cprint(f"{'gen_length':<12}: {args.gen_length}", color="white", flush=True)
    cprint(f"{'block_size':<12}: {args.block_size}", color="white", flush=True)
    cprint(f"{'tp_size':<12}: {args.tensor_parallel_size}", color="white", flush=True)
    cprint(f"{'mask_token':<12}: {mask_token_id}", color="white", flush=True)


def _print_response(response: str, elapsed: float) -> None:
    line = "=" * max(72, min(get_terminal_size((96, 20)).columns, 120))
    cprint(line, color="green", attrs=["bold"], flush=True)
    cprint("MinerU Result", color="green", attrs=["bold"], flush=True)
    cprint(line, color="green", attrs=["bold"], flush=True)
    cprint(response or "(empty response)", color="white", flush=True)
    cprint(line, color="green", attrs=["bold"], flush=True)
    cprint(f"elapsed: {elapsed:.2f}s", color="cyan", flush=True)


def _load_mask_token_id(model_path: Path) -> int:
    with open(model_path / "config.json", "r", encoding="utf-8") as fh:
        config = json.load(fh)
    mask_token_id = config.get("mask_token_id")
    if mask_token_id is None:
        raise ValueError(f"mask_token_id is missing from {model_path / 'config.json'}")
    return mask_token_id


def _build_message(image_path: str, prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        },
    ]


def run(args: argparse.Namespace) -> None:
    if args.device != "cuda":
        raise ValueError("nano_dvlm currently supports only --device cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for nano_dvlm")

    prompt = args.prompt or TASK_PROMPTS[args.prompt_type]
    model_path = Path(args.model_path).resolve()
    mask_token_id = _load_mask_token_id(model_path)
    start_time = time.time()

    _print_summary(args, model_path, mask_token_id)

    llm = LLM(
        str(model_path),
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=args.tensor_parallel_size,
        mask_token_id=mask_token_id,
        block_size=args.block_size,
        max_model_len=args.max_length,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_new_tokens=args.gen_length,
        denoising_strategy=args.remask_strategy,
        dynamic_threshold=args.dynamic_threshold,
        stop_tokens=list(STOP_STRINGS),
    )

    results = llm.generate_messages(
        [_build_message(args.image_path, prompt)],
        sampling_params=sampling_params,
        use_tqdm=False,
    )
    response = results[0]["text"]
    for stop in STOP_STRINGS:
        response = response.split(stop, 1)[0]
    _print_response(response.strip(), time.time() - start_time)


__all__ = ["add_arguments", "run"]

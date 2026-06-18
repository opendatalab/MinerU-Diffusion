import argparse
import os
import time
from pathlib import Path
from shutil import get_terminal_size

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from termcolor import cprint
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from mineru_diffusion.utils.runtime import maybe_disable_flash_attention, resolve_torch_dtype


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


def _print_summary(args: argparse.Namespace, model_path: Path, device: str, dtype: torch.dtype) -> None:
    line = "=" * max(72, min(get_terminal_size((96, 20)).columns, 120))
    cprint(line, color="cyan", attrs=["bold"], flush=True)
    cprint("MinerU HF Inference", color="cyan", attrs=["bold"], flush=True)
    cprint("-" * len(line), color="cyan", flush=True)
    cprint(f"{'model':<12}: {model_path}", color="white", flush=True)
    cprint(f"{'image':<12}: {Path(args.image_path).resolve()}", color="white", flush=True)
    cprint(f"{'prompt':<12}: {'custom' if args.prompt else args.prompt_type}", color="white", flush=True)
    cprint(f"{'device':<12}: {device}", color="white", flush=True)
    cprint(f"{'dtype':<12}: {dtype}", color="white", flush=True)
    cprint(f"{'gen_length':<12}: {args.gen_length}", color="white", flush=True)
    cprint(f"{'block_size':<12}: {args.block_size}", color="white", flush=True)


def _print_response(response: str, elapsed: float) -> None:
    line = "=" * max(72, min(get_terminal_size((96, 20)).columns, 120))
    cprint(line, color="green", attrs=["bold"], flush=True)
    cprint("MinerU Result", color="green", attrs=["bold"], flush=True)
    cprint(line, color="green", attrs=["bold"], flush=True)
    cprint(response or "(empty response)", color="white", flush=True)
    cprint(line, color="green", attrs=["bold"], flush=True)
    cprint(f"elapsed: {elapsed:.2f}s", color="cyan", flush=True)


def run(args: argparse.Namespace) -> None:
    prompt = args.prompt or TASK_PROMPTS[args.prompt_type]
    model_path = Path(args.model_path).resolve()
    device = args.device
    flash_attn_disabled = maybe_disable_flash_attention(device)
    dtype, resolved_dtype_name = resolve_torch_dtype(device, args.dtype)

    if flash_attn_disabled:
        cprint(
            "FlashAttention disabled for this GPU; using PyTorch SDPA fallback.",
            color="yellow",
            flush=True,
        )
    if resolved_dtype_name != args.dtype:
        cprint(
            f"CUDA device does not support {args.dtype}; falling back to {resolved_dtype_name}.",
            color="yellow",
            flush=True,
        )

    _print_summary(args, model_path, device, dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model = model.eval().to(device)

    mask_token_id = tokenizer.convert_tokens_to_ids("<|MASK|>")
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "image", "image": args.image_path}, {"type": "text", "text": prompt}]},
    ]

    prompt_text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
    )
    inputs = processor(
        images=[args.image_path],
        text=prompt_text,
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"].to(torch.long).to(device)
    image_grid_thw = inputs.get("image_grid_thw")
    if image_grid_thw is not None:
        image_grid_thw = image_grid_thw.to(torch.long).to(device)
    pixel_values = inputs["pixel_values"].to(dtype).to(device)

    with torch.no_grad():
        start_time = time.time()
        response_ids, _, _ = model.generate(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            input_ids=input_ids,
            mask_token_id=mask_token_id,
            denoising_steps=args.block_size,
            gen_length=args.gen_length,
            block_length=args.block_size,
            temperature=args.temperature,
            remasking_strategy=args.remask_strategy,
            dynamic_threshold=args.dynamic_threshold,
            tokenizer=tokenizer,
            stopping_criteria=list(STOP_STRINGS),
        )
        elapsed = time.time() - start_time

    response = tokenizer.decode(response_ids[0], skip_special_tokens=False)
    for stop in STOP_STRINGS:
        response = response.split(stop, 1)[0]
    _print_response(response.strip(), elapsed)

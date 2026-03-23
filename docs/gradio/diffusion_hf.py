import argparse
import json
import os
import time
from pathlib import Path
from shutil import get_terminal_size

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from termcolor import cprint
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from speed_compare.config import DIFFUSION_OUTPUT_PATH, resolve_diffusion_model_path


STOP_STRINGS = ("<|endoftext|>", "<|im_end|>")
SYSTEM_PROMPT = "You are a helpful assistant."
TASK_PROMPTS = {
    "text": "\nText Recognition:",
    "table": "\nTable Recognition:",
    "formula": "\nFormula Recognition:",
    "layout": "\nLayout Analysis:",
}
DEFAULT_MAX_LENGTH = 4096
DEFAULT_GEN_LENGTH = 4096


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path", default=None, help="Converted HF model directory.")
    parser.add_argument("--image-path", required=True, help="Input image path.")
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


def _trim_generated_ids(tokenizer, response_ids: list[int]) -> list[int]:
    stop_token_ids = {
        token_id
        for token_id in (tokenizer.convert_tokens_to_ids(token) for token in STOP_STRINGS)
        if token_id is not None and token_id >= 0
    }
    trimmed_ids = []
    for token_id in response_ids:
        if token_id in stop_token_ids:
            break
        trimmed_ids.append(token_id)
    return trimmed_ids


def _normalize_step_time(step_time):
    if torch.is_tensor(step_time):
        return step_time.detach().cpu().tolist()
    if isinstance(step_time, tuple):
        return [_normalize_step_time(item) for item in step_time]
    if isinstance(step_time, list):
        return [_normalize_step_time(item) for item in step_time]
    return step_time


def _build_visible_text_pieces(tokenizer, visible_response_ids: list[int]) -> list[str]:
    pieces = []
    previous_text = ""
    for index in range(len(visible_response_ids)):
        current_text = tokenizer.decode(
            visible_response_ids[: index + 1],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if current_text.startswith(previous_text):
            pieces.append(current_text[len(previous_text) :])
        else:
            # Fallback for tokenizers whose decoded prefix is not strictly incremental.
            pieces.append(current_text)
        previous_text = current_text
    return pieces


def infer_diffusion(
    *,
    model_path: str | Path,
    image_path: str | Path,
    prompt_type: str = "text",
    prompt: str | None = None,
    device: str = "cuda",
    dtype: str = "bfloat16",
    max_length: int = DEFAULT_MAX_LENGTH,
    gen_length: int = DEFAULT_GEN_LENGTH,
    block_size: int = 32,
    temperature: float = 1.0,
    remask_strategy: str = "low_confidence_dynamic",
    dynamic_threshold: float = 0.95,
) -> dict:
    model_path = Path(model_path).resolve()
    prompt_text_value = prompt or TASK_PROMPTS[prompt_type]
    torch_dtype = getattr(torch, dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    model = model.eval().to(device)

    try:
        mask_token_id = tokenizer.convert_tokens_to_ids("<|MASK|>")
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "image", "image": str(image_path)}, {"type": "text", "text": prompt_text_value}]},
        ]

        prompt_text = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
        inputs = processor(
            images=[str(image_path)],
            text=prompt_text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].to(torch.long).to(device)
        image_grid_thw = inputs.get("image_grid_thw")
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(torch.long).to(device)
        pixel_values = inputs["pixel_values"].to(torch_dtype).to(device)

        with torch.no_grad():
            start_time = time.perf_counter()
            response_ids, _, step_time = model.generate(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                input_ids=input_ids,
                mask_token_id=mask_token_id,
                denoising_steps=block_size,
                gen_length=gen_length,
                block_length=block_size,
                temperature=temperature,
                remasking_strategy=remask_strategy,
                dynamic_threshold=dynamic_threshold,
                tokenizer=tokenizer,
                stopping_criteria=list(STOP_STRINGS),
            )
            elapsed = time.perf_counter() - start_time
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    step_time = _normalize_step_time(step_time)
    response_ids_list = response_ids[0].cpu().tolist()
    visible_response_ids = _trim_generated_ids(tokenizer, response_ids_list)
    tokens = tokenizer.convert_ids_to_tokens(response_ids_list)
    visible_tokens = tokenizer.convert_ids_to_tokens(visible_response_ids)
    visible_text_pieces = _build_visible_text_pieces(tokenizer, visible_response_ids)
    response_text = tokenizer.decode(response_ids[0], skip_special_tokens=False)
    for stop in STOP_STRINGS:
        response_text = response_text.split(stop, 1)[0]
    step_times = step_time[0] if isinstance(step_time, list) and step_time and isinstance(step_time[0], list) else step_time
    first_token_time = next((value for value, piece in zip(step_times, visible_text_pieces) if piece), 0.0)

    return {
        "model_name": "MinerU-Diffusion",
        "prompt_text": prompt_text_value,
        "response": response_text.strip(),
        "elapsed": elapsed,
        "first_token_time": first_token_time,
        "response_ids": response_ids_list,
        "visible_response_ids": visible_response_ids,
        "tokens": tokens,
        "visible_tokens": visible_tokens,
        "visible_text_pieces": visible_text_pieces,
        "step_time": step_time,
        "token_count": len(visible_response_ids),
    }


def run(args: argparse.Namespace) -> None:
    prompt = args.prompt or TASK_PROMPTS[args.prompt_type]
    model_path = Path(args.model_path).expanduser().resolve() if args.model_path else resolve_diffusion_model_path()
    device = args.device
    dtype = getattr(torch, args.dtype)

    _print_summary(args, model_path, device, dtype)

    result = infer_diffusion(
        model_path=model_path,
        image_path=args.image_path,
        prompt_type=args.prompt_type,
        prompt=prompt,
        device=args.device,
        dtype=args.dtype,
        max_length=args.max_length,
        gen_length=args.gen_length,
        block_size=args.block_size,
        temperature=args.temperature,
        remask_strategy=args.remask_strategy,
        dynamic_threshold=args.dynamic_threshold,
    )

    with DIFFUSION_OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "response_ids": result["response_ids"],
                "tokens": result["tokens"],
                "step_time": result["step_time"],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    _print_response(result["response"], result["elapsed"])

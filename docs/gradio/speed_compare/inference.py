import gc
import json
import tempfile
from pathlib import Path

from PIL import Image

from speed_compare.config import (
    DIFFUSION_OUTPUT_PATH,
    MINERU_OUTPUT_PATH,
    TASK_PROMPTS,
    resolve_diffusion_model_path,
    resolve_mineru_model_path,
)


def cleanup_torch() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def ensure_valid_prompt_type(prompt_type: str) -> None:
    if prompt_type not in TASK_PROMPTS:
        raise ValueError(f"Unsupported prompt type: {prompt_type}")


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _resolve_image_path(image_path: str | Path) -> Path:
    image_path = Path(image_path).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return image_path


def prepare_image_for_prompt(image_path: str | Path, prompt_type: str) -> Path:
    ensure_valid_prompt_type(prompt_type)
    resolved_image_path = _resolve_image_path(image_path)
    if prompt_type != "layout":
        return resolved_image_path

    target_size = (1036, 1036)
    temp_dir = Path(tempfile.gettempdir()) / "mineru_speed_compare"
    temp_dir.mkdir(parents=True, exist_ok=True)
    output_path = temp_dir / f"{resolved_image_path.stem}_layout_1036x1036.png"

    with Image.open(resolved_image_path) as image:
        resized = image.convert("RGB").resize(target_size, Image.Resampling.LANCZOS)
        resized.save(output_path, format="PNG")
    return output_path.resolve()


def run_mineru_model(image_path: str | Path, prompt_type: str, *, assume_prepared: bool = False) -> dict:
    ensure_valid_prompt_type(prompt_type)
    image_path = _resolve_image_path(image_path) if assume_prepared else prepare_image_for_prompt(image_path, prompt_type)
    from mineru_hf import infer_mineru

    mineru_result = infer_mineru(
        model_path=resolve_mineru_model_path(),
        image_path=str(image_path),
        prompt_type=prompt_type,
    )
    _write_json(MINERU_OUTPUT_PATH, mineru_result["records"])
    cleanup_torch()
    return mineru_result


def run_diffusion_model(image_path: str | Path, prompt_type: str, *, assume_prepared: bool = False) -> dict:
    ensure_valid_prompt_type(prompt_type)
    image_path = _resolve_image_path(image_path) if assume_prepared else prepare_image_for_prompt(image_path, prompt_type)
    from diffusion_hf import infer_diffusion

    diffusion_result = infer_diffusion(
        model_path=resolve_diffusion_model_path(),
        image_path=str(image_path),
        prompt_type=prompt_type,
    )
    _write_json(
        DIFFUSION_OUTPUT_PATH,
        {
            "response_ids": diffusion_result["response_ids"],
            "tokens": diffusion_result["tokens"],
            "step_time": diffusion_result["step_time"],
        },
    )
    cleanup_torch()
    return diffusion_result


def run_models(image_path: str | Path, prompt_type: str) -> tuple[dict, dict]:
    mineru_result = run_mineru_model(image_path, prompt_type)
    diffusion_result = run_diffusion_model(image_path, prompt_type)
    return mineru_result, diffusion_result

import json
import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
RUNTIME_CONFIG_PATH = ROOT_DIR / "runtime_paths.json"
OUTPUT_DIR = ROOT_DIR

TASK_PROMPTS = {
    "text": "\nText Recognition:",
    "table": "\nTable Recognition:",
    "formula": "\nFormula Recognition:",
    "layout": "\nLayout Analysis:",
}

PROMPT_LABELS = {
    "text": "Text",
    "table": "Table",
    "formula": "Formula",
    "layout": "Layout",
}

PROMPT_CHOICES = [(label, key) for key, label in PROMPT_LABELS.items()]

MINERU_OUTPUT_PATH = OUTPUT_DIR / "mineru.txt"
DIFFUSION_OUTPUT_PATH = OUTPUT_DIR / "diffusion.txt"

REPLAY_STEP_SECONDS = float(os.getenv("REPLAY_STEP_SECONDS", "0.05"))


def _load_runtime_config() -> dict:
    if not RUNTIME_CONFIG_PATH.exists():
        return {}

    with RUNTIME_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid runtime config format: {RUNTIME_CONFIG_PATH}")
    return payload


_RUNTIME_CONFIG = _load_runtime_config()


def _resolve_config_path(env_name: str, config_key: str, label: str) -> Path:
    raw_value = os.getenv(env_name) or _RUNTIME_CONFIG.get(config_key)
    if not raw_value:
        raise RuntimeError(
            f"Missing {label}. Set {env_name} or add {config_key} to {RUNTIME_CONFIG_PATH.name}."
        )

    path = Path(raw_value).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path.resolve()


def resolve_mineru_model_path() -> Path:
    return _resolve_config_path(
        "MINERU_MODEL_PATH",
        "mineru_model_path",
        "MinerU 2.5 model path",
    )


def resolve_diffusion_model_path() -> Path:
    return _resolve_config_path(
        "DIFFUSION_MODEL_PATH",
        "diffusion_model_path",
        "MinerU-Diffusion model path",
    )


def resolve_layout_font_path() -> Path | None:
    raw_value = os.getenv("LAYOUT_FONT_PATH") or _RUNTIME_CONFIG.get("layout_font_path")
    if not raw_value:
        return None
    path = Path(raw_value).expanduser()
    if not path.exists():
        return None
    return path.resolve()

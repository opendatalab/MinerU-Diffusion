import os

import torch


def _parse_device(device: str) -> torch.device | None:
    try:
        return torch.device(device)
    except (RuntimeError, TypeError, ValueError):
        return None


def should_disable_flash_attention(device: str) -> bool:
    if os.environ.get("MINERU_DISABLE_FLASH_ATTN") == "1":
        return True

    torch_device = _parse_device(device)
    if torch_device is None or torch_device.type != "cuda" or not torch.cuda.is_available():
        return False

    try:
        major, _ = torch.cuda.get_device_capability(torch_device)
    except Exception:
        return False
    return major < 8


def maybe_disable_flash_attention(device: str) -> bool:
    if not should_disable_flash_attention(device):
        return False

    os.environ["MINERU_DISABLE_FLASH_ATTN"] = "1"

    try:
        import flash_attn
    except ImportError:
        return True

    flash_attn.flash_attn_func = None
    return True


def resolve_torch_dtype(device: str, requested_dtype: str) -> tuple[torch.dtype, str]:
    torch_device = _parse_device(device)
    resolved_dtype = requested_dtype

    if (
        requested_dtype == "bfloat16"
        and torch_device is not None
        and torch_device.type == "cuda"
        and torch.cuda.is_available()
        and not torch.cuda.is_bf16_supported()
    ):
        resolved_dtype = "float16"

    return getattr(torch, resolved_dtype), resolved_dtype

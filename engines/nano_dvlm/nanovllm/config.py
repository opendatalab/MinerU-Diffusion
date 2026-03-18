import os, json, torch
from dataclasses import dataclass
from types import SimpleNamespace

def _to_namespace(d, _k=None):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _to_namespace(v, k) for k, v in d.items()})
    elif isinstance(d, list):
        return [_to_namespace(i) for i in d]
    elif _k == 'torch_dtype':
        return getattr(torch, d)
    else:
        return d


def _normalize_hf_config(hf_config: SimpleNamespace) -> SimpleNamespace:
    model_type = getattr(hf_config, "model_type", None)
    if model_type == "mineru_diffusion":
        if not hasattr(hf_config, "language_model_config") and hasattr(hf_config, "text_config"):
            hf_config.language_model_config = hf_config.text_config
        if not hasattr(hf_config, "vision_model_config") and hasattr(hf_config, "vision_config"):
            hf_config.vision_model_config = hf_config.vision_config
        if not hasattr(hf_config, "vision_abstractor_config"):
            hf_config.vision_abstractor_config = SimpleNamespace(
                projection_type=getattr(hf_config, "vision_projector_type", None)
            )
        if getattr(hf_config, "torch_dtype", None) is None and hasattr(hf_config, "language_model_config"):
            hf_config.torch_dtype = getattr(hf_config.language_model_config, "torch_dtype", None)
        if not hasattr(hf_config, "rm_vit_merger"):
            hf_config.rm_vit_merger = True
    return hf_config

@dataclass
class Config:
    model: str
    checkpoint: str | None = None
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: SimpleNamespace | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    mask_token_id: int = 151669
    block_size: int = 32

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        if self.checkpoint is not None:
            self.model = os.path.join(self.model, self.checkpoint)
        self.hf_config = _normalize_hf_config(
            _to_namespace(json.load(open(os.path.join(self.model, "config.json"), 'r')))
        )
        self.max_model_len = min(self.max_model_len, self.hf_config.language_model_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len

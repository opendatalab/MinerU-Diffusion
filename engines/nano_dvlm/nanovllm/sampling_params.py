from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_new_tokens: int = 64
    ignore_eos: bool = False
    denoising_strategy: str = "low_confidence_dynamic"
    dynamic_threshold: float = 0.9
    stop_tokens: list[str] = None
    stop_token_ids: list[int] = None

    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"

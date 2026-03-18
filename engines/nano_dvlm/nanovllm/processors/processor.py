from __future__ import annotations

import torch, json
from transformers import AutoImageProcessor, AutoProcessor
from transformers.image_utils import load_image
from pathlib import Path


class Processor:
    def __init__(
        self,
        model_path: str,
        image_token_id: int = 151655,
    ) -> None:
        processor_path = Path(model_path)
        if not (processor_path / "processor_config.json").exists() and (processor_path / "processor").is_dir():
            processor_path = processor_path / "processor"

        self.processor = AutoProcessor.from_pretrained(
            processor_path, trust_remote_code=True, use_fast=False
        )
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        self.image_processor = AutoImageProcessor.from_pretrained(
            processor_path, trust_remote_code=True, use_fast=False
        )
        self.image_token_id = image_token_id
        self.spatial_merge_size = self.image_processor.merge_size

    def apply_chat_template(
        self,
        messages: list[dict],
    ) -> tuple[str, list]:
        images = []
        for message in messages:
            content = message["content"]
            for item in content:
                if item["type"] == "image":
                    images.append(item["image"])
        prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
        if isinstance(prompt, tuple):
            prompt = prompt[0]
        if isinstance(prompt, list):
            if len(prompt) != 1:
                raise ValueError(f"Expected a single rendered prompt, got {len(prompt)}")
            prompt = prompt[0]
        return prompt, images

    def process(
        self,
        prompt: str | None = None,
        images: list | None = None,
    ) -> dict:
        if isinstance(prompt, str):
            input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        else:
            input_ids = list(prompt)

        pixel_values = None
        image_grid_thw = None
        if images:
            images = [load_image(img) for img in images]
            image_outputs = self.image_processor.preprocess(
                images=images, return_tensors="pt"
            )
            pixel_values = image_outputs["pixel_values"]
            image_grid_thw = image_outputs["image_grid_thw"]
            image_grid_thw = torch.as_tensor(image_grid_thw, dtype=torch.int32)
            token_counts = self._num_image_tokens(image_grid_thw)
            input_ids = self._expand_image_tokens(input_ids, token_counts)

        return {
            "token_ids": input_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }

    def _num_image_tokens(self, image_grid_thw: torch.Tensor) -> list[int]:
        grid_sizes = image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]
        denom = self.spatial_merge_size * self.spatial_merge_size
        return (grid_sizes // denom).tolist()

    def _expand_image_tokens(self, input_ids: list[int], counts: list[int]) -> list[int]:
        placeholder_count = sum(1 for token in input_ids if token == self.image_token_id)
        if placeholder_count != len(counts):
            raise ValueError(
                "image token count mismatch: "
                f"{placeholder_count} placeholders vs {len(counts)} images"
            )
        expanded: list[int] = []
        img_idx = 0
        for token in input_ids:
            if token == self.image_token_id:
                expanded.extend([self.image_token_id] * counts[img_idx])
                img_idx += 1
            else:
                expanded.append(token)
        return expanded

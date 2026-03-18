import re
from types import SimpleNamespace
import torch
from torch import nn
import torch.distributed as dist

from nanovllm.layers.linear import ReplicatedLinear
from nanovllm.models.qwen2_vit import Qwen2VisionTransformer
from nanovllm.models.sdar import SDARForCausalLM


class PatchMerger(nn.Module):
    def __init__(
        self,
        dim: int,
        context_dim: int,
        spatial_merge_size: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = nn.LayerNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            ReplicatedLinear(self.hidden_size, self.hidden_size, bias=True),
            nn.GELU(),
            ReplicatedLinear(self.hidden_size, dim, bias=True),
        )

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln_q(x).view(-1, self.hidden_size)
        return self.mlp(x)


def build_projection(
    projection_type: str,
    in_dim: int,
    out_dim: int,
) -> nn.Module:
    mlp_match = re.match(r"^mlp(\d+)x_gelu$", projection_type)
    if mlp_match:
        depth = int(mlp_match.group(1))
        modules: list[nn.Module] = [
            ReplicatedLinear(in_dim, out_dim, bias=True)
        ]
        for _ in range(1, depth):
            modules.append(nn.GELU())
            modules.append(
                ReplicatedLinear(out_dim, out_dim, bias=True)
            )
        return nn.Sequential(*modules)

    pm_match = re.match(r"(?:patch_merger|pm)(\d+)x$", projection_type)
    if pm_match:
        merge_size = int(pm_match.group(1))
        return PatchMerger(out_dim, in_dim, spatial_merge_size=merge_size)

    raise ValueError(f"Unknown projection_type: {projection_type}")


class PerceiverProjection(nn.Module):
    def __init__(
        self,
        projection_type: str,
        in_dim: int,
        out_dim: int,
    ) -> None:
        super().__init__()
        self.projection = build_projection(
            projection_type, in_dim, out_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class DMLLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()
        self.config = config
        self.image_token_id = getattr(config, "image_token_id", None)

        vision_cfg = getattr(config, "vision_model_config", None)
        if vision_cfg is None:
            vision_cfg = getattr(config, "vision_config", None)
        if vision_cfg is None:
            raise ValueError("vision_model_config or vision_config is required")
        if hasattr(vision_cfg, "vision_config"):
            vision_cfg = vision_cfg.vision_config
        self.vision_model = Qwen2VisionTransformer(vision_cfg)

        # MinerU-Diffusion removes the Qwen2-VL patch merger and projects from embed_dim.
        vision_out_dim = vision_cfg.hidden_size
        rm_vit_merger = getattr(config, "rm_vit_merger", False) or getattr(config, "model_type", None) == "mineru_diffusion"
        if rm_vit_merger and hasattr(self.vision_model, "merger"):
            self.vision_model.merger = nn.Identity()
            vision_out_dim = vision_cfg.embed_dim

        language_cfg = getattr(config, "language_model_config", None)
        if language_cfg is None:
            language_cfg = getattr(config, "text_config", None)
        if language_cfg is None:
            raise ValueError("language_model_config or text_config is required")
        self.language_model = SDARForCausalLM(language_cfg)

        vision_abstractor_cfg = getattr(config, "vision_abstractor_config", None)
        projection_type = getattr(vision_abstractor_cfg, "projection_type", None)
        if projection_type is None:
            projection_type = getattr(config, "vision_projector_type", None)
        if projection_type is None:
            raise ValueError("vision_abstractor_config.projection_type is required")
        self.vision_abstractor = PerceiverProjection(
            projection_type=projection_type,
            in_dim=vision_out_dim,
            out_dim=language_cfg.hidden_size,
        )

    def forward_vision(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        vision_embeds = self.vision_model(pixel_values, image_grid_thw)
        return self.vision_abstractor(vision_embeds)

    def _prepare_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        vision_embeds: torch.Tensor | None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        if vision_embeds is None:
            return inputs_embeds
        if self.image_token_id is None:
            raise ValueError("image_token_id must be set when using vision_embeds")

        vision_mask = input_ids == self.image_token_id
        vision_embeds = vision_embeds.reshape(-1, vision_embeds.size(-1))
        if vision_mask.sum().item() != vision_embeds.shape[0]:
            raise ValueError(
                "vision embeddings mismatch input embeddings: "
                f"vision_mask count={vision_mask.sum().item()}; "
                f"vision_embeds shape={vision_embeds.shape}"
            )
        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[vision_mask] = vision_embeds.to(inputs_embeds.dtype)
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        vision_embeds = None
        if pixel_values is not None:
            if image_grid_thw is None:
                raise ValueError("image_grid_thw is required when pixel_values is set")
            vision_embeds = self.forward_vision(pixel_values, image_grid_thw)

        inputs_embeds = self._prepare_inputs_embeds(input_ids, vision_embeds)
        hidden_states = self.language_model(
            input_ids=None,
            positions=positions,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.language_model.compute_logits(hidden_states)

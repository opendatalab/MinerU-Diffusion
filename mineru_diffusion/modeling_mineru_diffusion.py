import re, time
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers import PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel

from .configuration_mineru_diffusion import MinerUDiffusionConfig, SDARConfig

try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None


def _new_dynamic_cache(config: Optional[SDARConfig] = None) -> DynamicCache:
    try:
        if config is not None:
            return DynamicCache(config=config)
    except TypeError:
        pass
    return DynamicCache()


class PatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = nn.LayerNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln_q(x).view(-1, self.hidden_size)
        return self.mlp(x)


def build_projection(projection_type: str, in_dim: int, out_dim: int) -> nn.Module:
    pm_match = re.match(r"(?:patch_merger|pm)(\d+)x$", projection_type)
    if pm_match:
        merge_size = int(pm_match.group(1))
        return PatchMerger(out_dim, in_dim, spatial_merge_size=merge_size)

    raise ValueError(f"Only patch_merger-style projectors are supported, got: {projection_type}")


class PerceiverProjection(nn.Module):
    def __init__(self, projection_type: str, in_dim: int, out_dim: int):
        super().__init__()
        self.projection = build_projection(projection_type, in_dim, out_dim)

    def forward(self, input_embeds: torch.Tensor) -> torch.Tensor:
        return self.projection(input_embeds)


class SDARRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class SDARMLP(nn.Module):
    def __init__(self, config: SDARConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class SDARAttention(nn.Module):
    def __init__(self, config: SDARConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.scaling = self.head_dim**-0.5
        self.is_causal = False
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.q_norm = SDARRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = SDARRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def _project_hidden_states(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        return query_states, key_states, value_states

    def _use_initialized_cache(self, past_key_value: Cache) -> bool:
        cache_layers = getattr(past_key_value, "layers", None)
        return (
            cache_layers is not None
            and len(cache_layers) > self.layer_idx
            and cache_layers[self.layer_idx].is_initialized
        )

    def _update_past_key_values(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        past_key_value: Optional[Cache],
        store_kv: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if past_key_value is None:
            return key_states, value_states

        if store_kv:
            return past_key_value.update(key_states, value_states, self.layer_idx)

        if self._use_initialized_cache(past_key_value):
            cache_layer = past_key_value.layers[self.layer_idx]
            past_key_states = cache_layer.keys
            past_value_states = cache_layer.values
        elif len(past_key_value) > self.layer_idx:
            past_key_states, past_value_states = past_key_value[self.layer_idx]
        else:
            return key_states, value_states

        key_states = torch.cat([past_key_states, key_states], dim=-2)
        value_states = torch.cat([past_value_states, value_states], dim=-2)
        return key_states, value_states

    def _can_use_flash_attention(
        self,
        query_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> bool:
        return (
            flash_attn_func is not None
            and query_states.device.type == "cuda"
            and attention_mask is not None
            and torch.all(attention_mask)
            and query_states.dtype in (torch.float16, torch.bfloat16)
        )

    def _flash_attention_forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        input_shape: torch.Size,
    ) -> torch.Tensor:
        attn_output = flash_attn_func(
            query_states.transpose(1, 2),
            key_states.transpose(1, 2),
            value_states.transpose(1, 2),
            causal=self.is_causal,
            softmax_scale=self.scaling,
        )
        return attn_output.reshape(*input_shape, -1).contiguous()

    def _sdpa_attention_forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        input_shape: torch.Size,
    ) -> torch.Tensor:
        attn_output = F.scaled_dot_product_attention(
            query=query_states,
            key=key_states,
            value=value_states,
            attn_mask=attention_mask.bool() if attention_mask is not None else None,
            is_causal=self.is_causal,
            scale=self.scaling,
            enable_gqa=True,
        )
        return attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings,
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        store_kv: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        query_states, key_states, value_states = self._project_hidden_states(hidden_states)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        key_states, value_states = self._update_past_key_values(key_states, value_states, past_key_value, store_kv)

        if self._can_use_flash_attention(query_states, attention_mask):
            attn_output = self._flash_attention_forward(query_states, key_states, value_states, input_shape)
        else:
            attn_output = self._sdpa_attention_forward(query_states, key_states, value_states, attention_mask, input_shape)
        return self.o_proj(attn_output), None


class SDARDecoderLayer(nn.Module):
    def __init__(self, config: SDARConfig, layer_idx: int):
        super().__init__()
        self.self_attn = SDARAttention(config=config, layer_idx=layer_idx)
        self.mlp = SDARMLP(config)
        self.input_layernorm = SDARRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = SDARRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        store_kv: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings=None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            store_kv=store_kv,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        return outputs


class SDARRotaryEmbedding(nn.Module):
    def __init__(self, config: SDARConfig, device=None):
        super().__init__()
        self.config = config
        if config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        if self.rope_type == "default" or self.rope_type not in ROPE_INIT_FUNCTIONS:
            dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
            base = float(getattr(config, "rope_theta", 10000.0))
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
            self.attention_scaling = 1.0
        else:
            inv_freq, self.attention_scaling = ROPE_INIT_FUNCTIONS[self.rope_type](config, device)
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @dynamic_rope_update
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class SDARPreTrainedModel(PreTrainedModel):
    config_class = SDARConfig
    base_model_prefix = "model"
    _no_split_modules = ["SDARDecoderLayer"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, SDARRMSNorm):
            module.weight.data.fill_(1.0)


class SDARModel(SDARPreTrainedModel):
    def __init__(self, config: SDARConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([SDARDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = SDARRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = SDARRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        store_kv: bool = False,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = True,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = _new_dynamic_cache(self.config)
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                store_kv=store_kv,
                cache_position=cache_position,
                **kwargs,
            )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class SDARForCausalLM(SDARPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: SDARConfig):
        super().__init__(config)
        self.model = SDARModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @staticmethod
    def _build_block_attention_mask(num_blocks: int, block_length: int, device: torch.device) -> torch.Tensor:
        block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=device, dtype=torch.bool))
        return block_mask.repeat_interleave(block_length, dim=0).repeat_interleave(block_length, dim=1)

    def _build_full_attention_mask(
        self,
        prompt_length: int,
        gen_length: int,
        block_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        prompt_blocks = (prompt_length + block_length - 1) // block_length
        prompt_mask = self._build_block_attention_mask(prompt_blocks, block_length, device)
        prompt_mask = prompt_mask[-prompt_length:, -prompt_length:]

        gen_blocks = gen_length // block_length
        gen_mask = self._build_block_attention_mask(gen_blocks, block_length, device)

        full_attn_mask = torch.zeros(
            prompt_length + gen_length,
            prompt_length + gen_length,
            device=device,
            dtype=torch.bool,
        )
        full_attn_mask[:prompt_length, :prompt_length] = prompt_mask
        full_attn_mask[prompt_length:, :prompt_length] = True
        full_attn_mask[prompt_length:, prompt_length:] = gen_mask
        return full_attn_mask

    def _initialize_generation_buffers(
        self,
        inputs_embeds: torch.Tensor,
        gen_length: int,
        mask_token_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = inputs_embeds.size(0)
        prompt_length = inputs_embeds.size(1)
        device = inputs_embeds.device
        mask_token = torch.tensor([mask_token_id], device=device)
        mask_embeds = self.get_input_embeddings()(mask_token)
        masked_generation_embeds = mask_embeds.unsqueeze(0).expand(batch_size, gen_length, -1)
        x_embeds = torch.cat([inputs_embeds, masked_generation_embeds], dim=1)
        tokens = torch.full(
            (batch_size, prompt_length + gen_length),
            mask_token_id,
            dtype=torch.long,
            device=device,
        )
        step_map = torch.zeros_like(tokens, dtype=torch.int64)
        step_time = torch.zeros_like(tokens, dtype=torch.float)
        return x_embeds, tokens, step_map, step_time, mask_embeds

    @staticmethod
    def _prepare_stop_tokens(
        stopping_criteria,
        tokenizer,
        device: torch.device,
    ) -> list[torch.Tensor]:
        if stopping_criteria is None:
            return []
        if tokenizer is None:
            raise ValueError("tokenizer is required when stopping_criteria is not None")
        return [
            torch.tensor(tokenizer.encode(stop_str, add_special_tokens=False), device=device)
            for stop_str in stopping_criteria
        ]

    def _select_transfer_index(
        self,
        confidence: torch.Tensor,
        threshold: float,
        transfer_count: int,
    ) -> torch.Tensor:
        transfer_index = torch.zeros_like(confidence, dtype=torch.bool)
        for batch_idx in range(confidence.shape[0]):
            high_confidence = confidence[batch_idx] > threshold
            if high_confidence.sum() >= transfer_count:
                transfer_index[batch_idx] = high_confidence
                continue
            _, top_indices = torch.topk(confidence[batch_idx], transfer_count)
            transfer_index[batch_idx, top_indices] = True
        return transfer_index

    @staticmethod
    def _find_stop_position(
        generated_tokens: torch.Tensor,
        stop_tokens: list[torch.Tensor],
    ) -> Optional[int]:
        for stop_token in stop_tokens:
            stop_length = stop_token.numel()
            if stop_length == 0 or generated_tokens.numel() < stop_length:
                continue
            for end_idx in range(stop_length, generated_tokens.size(0) + 1):
                if torch.equal(generated_tokens[end_idx - stop_length : end_idx], stop_token):
                    return end_idx - stop_length
        return None

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=True,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        hidden_states = hidden_states[:, slice_indices, :].contiguous()
        logits = self.lm_head(hidden_states)
        if not return_dict:
            return (logits, outputs.past_key_values)
        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=None,
            attentions=None,
        )

    @staticmethod
    def top_k_logits(logits, k):
        if k <= 0:
            return logits
        values, _ = torch.topk(logits, k)
        min_values = values[..., -1, None]
        return torch.where(logits < min_values, torch.full_like(logits, float("-inf")), logits)

    @staticmethod
    def top_p_logits(logits, p):
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = cumulative_probs > p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        mask_indices = torch.scatter(torch.full_like(logits, False, dtype=torch.bool), -1, sorted_indices, sorted_mask)
        return logits.masked_fill(mask_indices, float("-inf"))

    def sample_with_temperature_topk_topp(self, logits, temperature=1.0, top_k=0, top_p=1.0):
        orig_shape = logits.shape[:-1]
        vocab_size = logits.shape[-1]
        logits = logits.reshape(-1, vocab_size)
        if temperature != 1.0:
            logits = logits / temperature
        if top_k > 0:
            logits = self.top_k_logits(logits, top_k)
        if top_p < 1.0:
            logits = self.top_p_logits(logits, top_p)
        probs = F.softmax(logits, dim=-1)
        token = torch.multinomial(probs, num_samples=1)
        token_prob = torch.gather(probs, -1, token)
        return token.view(*orig_shape), token_prob.view(*orig_shape)

    @staticmethod
    def get_num_transfer_tokens(block_length, steps):
        base = block_length // steps
        remainder = block_length % steps
        num_transfer_tokens = torch.zeros(steps, dtype=torch.int64) + base
        num_transfer_tokens[:remainder] += 1
        return num_transfer_tokens

    def generate_with_embeds(
        self,
        inputs_embeds,
        gen_length,
        block_length,
        mask_token_id,
        denoising_steps=8,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        remasking_strategy="low_confidence_dynamic",
        dynamic_threshold=0.85,
        stopping_criteria=None,
        tokenizer=None,
        **kwargs,
    ):
        if gen_length % block_length != 0:
            raise ValueError(f"gen_length({gen_length}) must be multiple of block_length({block_length})")
        if remasking_strategy != "low_confidence_dynamic":
            raise ValueError("Only remasking_strategy='low_confidence_dynamic' is supported.")

        prompt_length = inputs_embeds.size(1)
        past_key_values = _new_dynamic_cache(self.config)
        gen_blocks = gen_length // block_length
        full_attn_mask = self._build_full_attention_mask(prompt_length, gen_length, block_length, inputs_embeds.device)
        position_ids = torch.arange(0, prompt_length + gen_length, device=inputs_embeds.device).unsqueeze(0)
        x_embeds, x, step_map, step_time, mask_embeds = self._initialize_generation_buffers(inputs_embeds, gen_length, mask_token_id)

        if prompt_length > 0:
            prompt_attn_mask = full_attn_mask[:prompt_length, :prompt_length].unsqueeze(0).unsqueeze(0)
            self(
                inputs_embeds=x_embeds[:, :prompt_length, :],
                attention_mask=prompt_attn_mask,
                position_ids=position_ids[:, :prompt_length],
                past_key_values=past_key_values,
                use_cache=True,
                store_kv=True,
            )

        num_transfer_tokens = self.get_num_transfer_tokens(block_length, denoising_steps)
        stop_tokens = self._prepare_stop_tokens(stopping_criteria, tokenizer, inputs_embeds.device)
        global_step = 0
        found_stop_token = False
        stop_pos = -1
        stop_chunk_end = -1
        start_time = time.perf_counter()

        for num_blocks in range(gen_blocks):
            block_start = prompt_length + num_blocks * block_length
            block_end = prompt_length + (num_blocks + 1) * block_length
            cur_x = x[:, block_start:block_end]
            cur_x_embeds = x_embeds[:, block_start:block_end, :]
            cur_step_map = step_map[:, block_start:block_end]
            cur_step_time = step_time[:, block_start:block_end]
            cur_attn_mask = full_attn_mask[block_start:block_end, :block_end].unsqueeze(0).unsqueeze(0)
            cur_position_ids = position_ids[:, block_start:block_end]

            for step in range(denoising_steps + 1):
                mask_index = (cur_x_embeds == mask_embeds).all(dim=-1)
                if mask_index.sum() == 0:
                    self(
                        inputs_embeds=cur_x_embeds,
                        attention_mask=cur_attn_mask,
                        position_ids=cur_position_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                        store_kv=True,
                    )
                    break
                outputs = self(
                    inputs_embeds=cur_x_embeds,
                    attention_mask=cur_attn_mask,
                    position_ids=cur_position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    store_kv=False,
                )
                x0, x0_p = self.sample_with_temperature_topk_topp(
                    outputs.logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
                confidence = torch.where(mask_index, x0_p, -torch.inf)
                transfer_index = self._select_transfer_index(
                    confidence,
                    dynamic_threshold,
                    int(num_transfer_tokens[min(step, denoising_steps - 1)].item()),
                )

                global_step += 1
                x0_embeds = self.get_input_embeddings()(x0)
                cur_x_embeds[transfer_index] = x0_embeds[transfer_index]
                cur_x[transfer_index] = x0[transfer_index]
                cur_step_map[transfer_index] = global_step
                cur_step_time[transfer_index] = time.perf_counter() - start_time

            if stop_tokens:
                generated = x[0, prompt_length:block_end]
                stop_offset = self._find_stop_position(generated, stop_tokens)
                if stop_offset is not None:
                    found_stop_token = True
                    stop_pos = prompt_length + stop_offset
                    stop_chunk_end = block_end
            if found_stop_token:
                break

        if found_stop_token:
            x = x[:, :stop_pos]
            step_map = step_map[:, :stop_chunk_end]
            step_time = step_time[:, :stop_chunk_end]

        return x[:, prompt_length:], step_map[:, prompt_length:], step_time[:, prompt_length:]


class MinerUDiffusionForConditionalGeneration(PreTrainedModel, GenerationMixin):
    config_class = MinerUDiffusionConfig

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, config=None, **kwargs):
        official_kwargs = dict(kwargs)
        official_kwargs.pop("trust_remote_code", None)
        if "dtype" in official_kwargs and "torch_dtype" not in official_kwargs:
            official_kwargs["torch_dtype"] = official_kwargs.pop("dtype")
        device = official_kwargs.pop("device", None)
        if device is not None and "device_map" not in official_kwargs:
            official_kwargs["device_map"] = device
            official_kwargs.setdefault("low_cpu_mem_usage", True)

        return super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            **official_kwargs,
        )

    def _init_weights(self, module):
        return

    def __init__(self, config: MinerUDiffusionConfig):
        super().__init__(config)
        if config.vision_model_type != "qwen2_vl":
            raise ValueError(f"Only qwen2_vl vision towers are supported, got: {config.vision_model_type}")
        self.vision_model = Qwen2VisionTransformerPretrainedModel._from_config(config.vision_model_config)
        self.vision_model.merger = nn.Identity()
        self.vision_abstractor = PerceiverProjection(
            projection_type=config.vision_projector_type,
            in_dim=config.vision_model_config.embed_dim,
            out_dim=config.language_model_config.hidden_size,
        )
        self.language_model = SDARForCausalLM(config.language_model_config)
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def _prepare_inputs_embeds(
        self,
        input_ids: torch.LongTensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        image_features = None
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values, image_grid_thw)
        return self._merge_input_and_image_features(input_ids, image_features)

    def _extract_vision_hidden_states(self, vision_outputs):
        if hasattr(vision_outputs, "last_hidden_state"):
            return vision_outputs.last_hidden_state
        if isinstance(vision_outputs, (tuple, list)):
            return vision_outputs[0]
        return vision_outputs

    def get_image_features(self, pixel_values, image_grid_thw):
        vision_outputs = self.vision_model(pixel_values, image_grid_thw)
        vision_hidden_states = self._extract_vision_hidden_states(vision_outputs)
        return self.vision_abstractor(vision_hidden_states)

    def _merge_input_and_image_features(self, input_ids, image_features):
        inputs_embeds = self.get_input_embeddings()(input_ids)
        if image_features is None:
            return inputs_embeds
        vision_mask = input_ids == self.config.image_token_id
        num_image_tokens = torch.count_nonzero(vision_mask).item()
        num_image_features = image_features.shape[:-1].numel()
        if num_image_tokens != num_image_features:
            raise ValueError(
                f"vision token count mismatch: {num_image_tokens} vs {num_image_features}"
            )
        return torch.masked_scatter(
            inputs_embeds,
            vision_mask.unsqueeze(-1),
            image_features.to(inputs_embeds.dtype).view(-1, image_features.size(-1)),
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        return_dict: bool = True,
        **kwargs,
    ):
        inputs_embeds = self._prepare_inputs_embeds(input_ids, pixel_values, image_grid_thw)
        return self.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=return_dict,
            **kwargs,
        )

    @torch.no_grad()
    def generate(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ):
        inputs_embeds = self._prepare_inputs_embeds(input_ids, pixel_values, image_grid_thw)
        return self.language_model.generate_with_embeds(inputs_embeds=inputs_embeds, **generate_kwargs)


MinerUDiffusion = MinerUDiffusionForConditionalGeneration


__all__ = ["MinerUDiffusionForConditionalGeneration", "MinerUDiffusion", "MinerUDiffusionConfig", "SDARConfig"]

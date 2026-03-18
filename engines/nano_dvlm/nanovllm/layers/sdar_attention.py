import torch
from torch import nn

from flash_attn import flash_attn_with_kvcache
from nanovllm.utils.context import get_context
from nanovllm.kernels.triton.attention import sparse_attn_varlen
from nanovllm.layers.attention import store_kvcache
from einops import rearrange


def store_kvcache_maybe4d(k, v, k_cache, v_cache, slot_mapping):
    if k.dim() == 4:
        k = rearrange(k, 'b s h d -> (b s) h d')
        v = rearrange(v, 'b s h d -> (b s) h d')
        slot_mapping = rearrange(slot_mapping, 'b s -> (b s)')
    store_kvcache(k, v, k_cache, v_cache, slot_mapping)


class Attention(nn.Module):
    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache_maybe4d(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            # input: [N, H, D]
            o = sparse_attn_varlen(
                q, k, v,
                cu_seqlens_q=context.cu_seqlens_q,
                cu_seqlens_k=context.cu_seqlens_k,
                staircase_size=context.block_size,
            )
            o = o.flatten(1)  # [N, H*D]
        else:
            # input: [bs, seq_len, H, D]
            o = flash_attn_with_kvcache(
                q, k_cache, v_cache,
                cache_seqlens=context.context_lens,
                block_table=context.block_tables, 
                softmax_scale=self.scale, causal=False
            )
            o = o.flatten(2)  # [bs, seq_len, H*D]
        return o

from functools import partial
import torch
from torch import nn
import torch.distributed as dist
from einops import rearrange
from typing import Callable
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLVisionConfig
from flash_attn import flash_attn_varlen_func

from nanovllm.layers.linear import ColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import apply_rotary_emb


class QuickGELU(nn.Module):
    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class VisionRotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        max_position: int,
        base: float = 10000.0,
        partial_rotary_factor: float = 0.5,
    ) -> None:
        super().__init__()
        rotary_dim = int(head_size * partial_rotary_factor)
        assert rotary_dim % 2 == 0
        self.rotary_dim = rotary_dim
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim
            )
        )
        t = torch.arange(max_position, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def get_cos_sin(self, seqlen: int) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[:seqlen]
        return cos_sin.chunk(2, dim=-1)


class Qwen2VisionMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        act_layer: type[nn.Module] = QuickGELU,
    ) -> None:
        super().__init__()
        self.fc1 = ColumnParallelLinear(
            in_features,
            hidden_features,
            bias=True,
        )
        self.act = act_layer()
        self.fc2 = RowParallelLinear(
            hidden_features,
            in_features,
            bias=True,
        )

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Qwen2VisionAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        projection_size: int,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        assert num_heads % tp_size == 0
        self.num_heads = num_heads // tp_size
        self.head_dim = projection_size // num_heads
        assert self.head_dim % 2 == 0

        self.qkv = ColumnParallelLinear(
            input_size=embed_dim,
            output_size=3 * projection_size,
            bias=True,
        )
        self.proj = RowParallelLinear(
            input_size=projection_size,
            output_size=embed_dim,
            bias=True,
        )

    def _attn_flash(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        bsz, q_len = q.size()[:2]

        q, k, v = (rearrange(x, "b s ... -> (b s) ...") for x in (q, k, v))
        output = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=0.0,
            causal=False,
        )
        return rearrange(output, "(b s) h d -> b s h d", b=bsz)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        seq_len, batch, _ = q.shape
        new_shape = (seq_len, batch, self.num_heads, self.head_dim)
        q, k, v = (x.view(*new_shape) for x in (q, k, v))

        q, k, v = (rearrange(x, "s b ... -> b s ...") for x in (q, k, v))

        # import ipdb; ipdb.set_trace()
        q = apply_rotary_emb(q, rotary_pos_emb_cos, rotary_pos_emb_sin)
        k = apply_rotary_emb(k, rotary_pos_emb_cos, rotary_pos_emb_sin)

        context = self._attn_flash(q, k, v, cu_seqlens, max_seqlen)

        context = rearrange(context, "b s h d -> s b (h d)").contiguous()
        output = self.proj(context)
        return output


class Qwen2VisionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        act_layer: type[nn.Module] = QuickGELU,
        norm_layer: Callable[[int], nn.Module] | None = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.attn = Qwen2VisionAttention(
            embed_dim=dim,
            num_heads=num_heads,
            projection_size=dim,
        )
        self.mlp = Qwen2VisionMLP(
            dim,
            mlp_hidden_dim,
            act_layer=act_layer,
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb_cos=rotary_pos_emb_cos,
            rotary_pos_emb_sin=rotary_pos_emb_sin,
            max_seqlen=max_seqlen,
        )
        x = x + self.mlp(self.norm2(x))
        return x


class Qwen2VisionPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.embed_dim = embed_dim

        kernel_size = (temporal_patch_size, patch_size, patch_size)
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=False,
        )

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L, C = x.shape
        x = x.view(L, -1, self.temporal_patch_size, self.patch_size, self.patch_size)
        x = self.proj(x).view(L, self.embed_dim)
        return x


class Qwen2VisionPatchMerger(nn.Module):
    def __init__(
        self,
        d_model: int,
        context_dim: int,
        norm_layer: Callable[[int], nn.Module] | None = None,
        spatial_merge_size: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.ln_q = norm_layer(context_dim)
        self.mlp = nn.ModuleList(
            [
                ColumnParallelLinear(
                    self.hidden_size,
                    self.hidden_size,
                    bias=True,
                ),
                nn.GELU(),
                RowParallelLinear(
                    self.hidden_size,
                    d_model,
                    bias=True,
                ),
            ]
        )

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln_q(x)
        x = x.view(-1, self.hidden_size)
        mlp_fc1, mlp_act, mlp_fc2 = self.mlp
        x = mlp_fc1(x)
        x = mlp_act(x)
        x = mlp_fc2(x)
        return x


class Qwen2VisionTransformer(nn.Module):
    def __init__(
        self,
        vision_config: Qwen2VLVisionConfig,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()

        patch_size = vision_config.patch_size
        temporal_patch_size = vision_config.temporal_patch_size
        spatial_merge_size = vision_config.spatial_merge_size
        in_channels = vision_config.in_channels
        hidden_size = vision_config.hidden_size
        embed_dim = vision_config.embed_dim
        depth = vision_config.depth
        num_heads = vision_config.num_heads
        mlp_ratio = vision_config.mlp_ratio

        self.out_hidden_size = hidden_size
        self.spatial_merge_size = spatial_merge_size
        self.num_heads = num_heads
        self.embed_dim = embed_dim

        self.patch_embed = Qwen2VisionPatchEmbed(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        head_dim = embed_dim // num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(
            head_size=head_dim,
            max_position=8192,
            base=10000.0,
            partial_rotary_factor=0.5,
        )

        self.blocks = nn.ModuleList(
            [
                Qwen2VisionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    norm_layer=partial(nn.LayerNorm, eps=norm_eps),
                )
                for _ in range(depth)
            ]
        )
        self.merger = Qwen2VisionPatchMerger(
            d_model=hidden_size,
            context_dim=embed_dim,
            norm_layer=partial(nn.LayerNorm, eps=norm_eps),
            spatial_merge_size=spatial_merge_size,
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    def rot_pos_emb(
        self, grid_thw: list[list[int]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pos_ids = []
        max_grid_size = 0
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            hpos_ids = (
                hpos_ids.reshape(
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                )
                .permute(0, 2, 1, 3)
                .flatten()
            )
            wpos_ids = (
                wpos_ids.reshape(
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                )
                .permute(0, 2, 1, 3)
                .flatten()
            )
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
            max_grid_size = max(max_grid_size, h, w)
        pos_ids = torch.cat(pos_ids, dim=0)

        cos, sin = self.rotary_pos_emb.get_cos_sin(max_grid_size)
        cos_combined = cos[pos_ids].flatten(1)
        sin_combined = sin[pos_ids].flatten(1)
        return cos_combined, sin_combined

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor | list[list[int]],
    ) -> torch.Tensor:
        x = x.to(device=self.device, dtype=self.dtype)
        x = self.patch_embed(x)

        if isinstance(grid_thw, list):
            grid_thw_list = grid_thw
            grid_thw_tensor = torch.tensor(grid_thw, dtype=torch.int32)
        else:
            grid_thw_tensor = grid_thw.to(torch.int32).cpu()
            grid_thw_list = grid_thw_tensor.tolist()

        rotary_pos_emb_cos, rotary_pos_emb_sin = self.rot_pos_emb(grid_thw_list)
        rotary_pos_emb_cos = rotary_pos_emb_cos.unsqueeze(1).to(self.device)
        rotary_pos_emb_sin = rotary_pos_emb_sin.unsqueeze(1).to(self.device)

        grid_hw = grid_thw_tensor[:, 1] * grid_thw_tensor[:, 2]
        repeat = torch.repeat_interleave(grid_hw, grid_thw_tensor[:, 0])
        cu_seqlens = repeat.cumsum(0)
        cu_seqlens = torch.cat(
            [torch.zeros(1, dtype=torch.int32), cu_seqlens]
        )
        if cu_seqlens.numel() > 1:
            max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
        else:
            max_seqlen = 0
        cu_seqlens = cu_seqlens.to(self.device, dtype=torch.int32, non_blocking=True)

        x = x.unsqueeze(1)
        for blk in self.blocks:
            x = blk(
                x,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb_cos=rotary_pos_emb_cos,
                rotary_pos_emb_sin=rotary_pos_emb_sin,
                max_seqlen=max_seqlen,
            )

        x = self.merger(x)
        return x

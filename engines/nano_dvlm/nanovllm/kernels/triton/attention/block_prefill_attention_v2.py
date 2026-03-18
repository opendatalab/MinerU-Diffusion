import math
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64},  num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64},  num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 128}, num_stages=4, num_warps=4),
    ],
    key=["BLOCK_DMODEL", "LOG2_STAIRS"],
)
@triton.jit
def _staircase_attn_fwd_kernel_varlen_v2(
    Q, K, V, Out,
    cu_seqlens_q, cu_seqlens_k,
    stride_qt, stride_qh, stride_qk,
    stride_kt, stride_kh, stride_kk,
    stride_vt, stride_vh, stride_vk,
    stride_ot, stride_oh, stride_ok,
    n_heads, n_kv_heads,
    q_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    LOG2_STAIRS: tl.constexpr,
    NUM_HEADS_PER_KV_GROUP: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    kv_head_idx = head_idx // NUM_HEADS_PER_KV_GROUP

    q_seq_start = tl.load(cu_seqlens_q + seq_idx).to(tl.int32)
    q_seq_end = tl.load(cu_seqlens_q + seq_idx + 1).to(tl.int32)
    k_seq_start = tl.load(cu_seqlens_k + seq_idx).to(tl.int32)
    k_seq_end = tl.load(cu_seqlens_k + seq_idx + 1).to(tl.int32)
    q_seq_len = q_seq_end - q_seq_start
    k_seq_len = k_seq_end - k_seq_start

    if q_seq_len == 0:
        return

    STAIRS = 1 << LOG2_STAIRS  # S

    # 这里为了严格匹配你给的 torch 代码，默认你做的是 self-attn：q_seq_len == k_seq_len
    # 所以用 seq_len = k_seq_len（或 q_seq_len 都一样）
    seq_len = k_seq_len

    blocks = (seq_len + (STAIRS - 1)) >> LOG2_STAIRS   # ceil(seq_len / S)（S 是 2^LOG2）
    pad = (blocks << LOG2_STAIRS) - seq_len            # pad in [0, S-1]，等价 start_idx

    Q_head_ptr = Q + head_idx * stride_qh
    K_head_ptr = K + kv_head_idx * stride_kh
    V_head_ptr = V + kv_head_idx * stride_vh
    O_head_ptr = Out + head_idx * stride_oh

    # ---------- outer tiles over M (queries) ----------
    num_m_iters = (q_seq_len + BLOCK_M - 1) // BLOCK_M
    for it_m in range(num_m_iters):  # -> scf.for (supported)
        start_m = it_m * BLOCK_M

        # running softmax state for this M-tile
        m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

        q_block_abs_start = q_seq_start + start_m
        Q_block_ptr = tl.make_block_ptr(
            base=Q_head_ptr,
            shape=(q_seq_end, BLOCK_DMODEL),
            strides=(stride_qt, stride_qk),
            offsets=(q_block_abs_start, 0),
            block_shape=(BLOCK_M, BLOCK_DMODEL),
            order=(1, 0),
        )
        q = tl.load(Q_block_ptr, boundary_check=(0,))
        q = (q * q_scale).to(q.dtype)

        # staircase window end bound for this M-tile
        q_block_rel_end = tl.minimum(start_m + BLOCK_M - 1, q_seq_len - 1)

        # 在 padded 坐标里算这个 tile 覆盖到的最大 block
        q_block_rel_end_pad = q_block_rel_end + pad
        max_band = q_block_rel_end_pad >> LOG2_STAIRS

        # 对应的 key 允许范围：padded 的 (max_band+1)*S - 1
        # 转回未 pad 坐标后：exclusive end = (max_band+1)*S - pad
        end_n = ((max_band + 1) << LOG2_STAIRS) - pad
        end_n = tl.minimum(end_n, k_seq_len)

        offs_m = start_m + tl.arange(0, BLOCK_M)
        m_valid = offs_m < q_seq_len

        # 右对齐：块号在 padded 坐标上算
        row_band = (offs_m + pad) >> LOG2_STAIRS

        # padded 块的右端点是 (row_band+1)*S - 1
        # 转回未 pad 坐标：减 pad
        col_limit_abs = (((row_band + 1) << LOG2_STAIRS) - 1) - pad

        # ---------- inner tiles over N (keys/values) ----------
        num_n_iters = (end_n + BLOCK_N - 1) // BLOCK_N
        for it_n in range(num_n_iters):  # -> scf.for (supported)
            start_n = it_n * BLOCK_N
            # NOTE: We intentionally iterate up to ceil(end_n/BLOCK_N).
            # Masking below ensures correctness when start_n >= end_n or tail overflow.

            k_block_abs_start = k_seq_start + start_n
            K_iter_ptr = tl.make_block_ptr(
                base=K_head_ptr,
                shape=(k_seq_end, BLOCK_DMODEL),
                strides=(stride_kt, stride_kk),
                offsets=(k_block_abs_start, 0),
                block_shape=(BLOCK_N, BLOCK_DMODEL),
                order=(1, 0),
            )
            V_iter_ptr = tl.make_block_ptr(
                base=V_head_ptr,
                shape=(k_seq_end, BLOCK_DMODEL),
                strides=(stride_vt, stride_vk),
                offsets=(k_block_abs_start, 0),
                block_shape=(BLOCK_N, BLOCK_DMODEL),
                order=(1, 0),
            )

            k = tl.load(K_iter_ptr, boundary_check=(0,), cache_modifier=".cg")
            v = tl.load(V_iter_ptr, boundary_check=(0,), cache_modifier=".cg")

            qk = tl.dot(q, tl.trans(k))  # [M,N]

            offs_n_rel = tl.arange(0, BLOCK_N)
            offs_n_abs = start_n + offs_n_rel

            n_valid = offs_n_abs < k_seq_len
            valid_cols = offs_n_abs[None, :] <= col_limit_abs[:, None]
            mask = valid_cols & m_valid[:, None] & n_valid[None, :]

            qk = tl.where(mask, qk, -1.0e6)

            # stable running softmax
            m_i_new = tl.maximum(m_i, tl.max(qk, axis=1))
            alpha = tl.exp2(m_i - m_i_new)
            p = tl.exp2(qk - m_i_new[:, None])
            l_ij = tl.sum(p, axis=1)

            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]
            m_i = m_i_new
            acc = tl.dot(p.to(v.dtype), v, acc)

        # normalize and write out this M-tile
        l_i_safe = tl.where(l_i == 0, 1.0, l_i)
        acc = acc / l_i_safe[:, None]

        O_block_ptr = tl.make_block_ptr(
            base=O_head_ptr,
            shape=(q_seq_end, BLOCK_DMODEL),
            strides=(stride_ot, stride_ok),
            offsets=(q_block_abs_start, 0),
            block_shape=(BLOCK_M, BLOCK_DMODEL),
            order=(1, 0),
        )
        tl.store(O_block_ptr, acc.to(q.dtype), boundary_check=(0,))

class SparseAttentionVarlenFunctionV2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor, staircase_size: int) -> torch.Tensor:
        assert q.dim() == k.dim() == v.dim() == 3
        assert q.shape[0] == k.shape[0] == v.shape[0]
        assert k.shape[1] == v.shape[1] and k.shape[2] == v.shape[2]
        assert q.shape[2] == k.shape[2]
        assert q.shape[1] % k.shape[1] == 0
        assert staircase_size in [1,2,4,8,16,32,64,128]
        log2_stairs = int(math.log2(staircase_size)) if staircase_size > 0 else 0
        total_tokens, n_heads, head_dim = q.shape
        _, n_kv_heads, _ = k.shape
        batch_size = cu_seqlens_q.numel() - 1
        num_heads_per_kv_group = n_heads // n_kv_heads
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        BLOCK_DMODEL = head_dim
        o = torch.empty_like(q)
        grid = (batch_size, n_heads)
        _staircase_attn_fwd_kernel_varlen_v2[grid](
            q, k, v, o,
            cu_seqlens_q, cu_seqlens_k,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            n_heads, n_kv_heads,
            q_scale=1.0 / math.sqrt(head_dim) * 1.44269504,
            BLOCK_DMODEL=BLOCK_DMODEL,
            LOG2_STAIRS=log2_stairs,
            NUM_HEADS_PER_KV_GROUP=num_heads_per_kv_group,
        )
        return o

def sparse_attn_varlen_v2(q, k, v, cu_seqlens_q, cu_seqlens_k, staircase_size: int = 4):
    return SparseAttentionVarlenFunctionV2.apply(q, k, v, cu_seqlens_q, cu_seqlens_k, staircase_size)

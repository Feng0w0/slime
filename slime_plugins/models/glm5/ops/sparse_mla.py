"""Sparse MLA with a TileLang fast path and a portable PyTorch fallback."""

import os

import torch

try:
    import torch_npu
except ImportError:
    torch_npu = None

_SLIME_DSA_DTYPE_TRACE_EMITTED = False

try:
    from .tilelang_sparse_mla_bwd import sparse_mla_bwd
    from .tilelang_sparse_mla_fwd import sparse_mla_fwd_interface

    _TILELANG_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    sparse_mla_bwd = None
    sparse_mla_fwd_interface = None
    _TILELANG_AVAILABLE = False


def _npu_sparse_mla(q, kv, indices, scaling, cu_seqlens):
    """Run the same Ascend sparse-attention kernel used by SGLang."""
    if torch_npu is None:
        raise RuntimeError("SLIME_MEGATRON_DSA_NPU_KERNEL requires torch_npu")

    output_dtype = q.dtype
    q = q.to(torch.bfloat16)
    kv = kv.to(torch.bfloat16)
    q_nope, q_rope = q[..., :-64], q[..., -64:]
    kv_nope, kv_rope = kv[..., :-64], kv[..., -64:]
    actual_seq_lengths = cu_seqlens[1:].to(device=q.device, dtype=torch.int32)
    output, softmax_max, softmax_sum = torch_npu.npu_sparse_flash_attention(
        query=q_nope,
        key=kv_nope,
        value=kv_nope,
        query_rope=q_rope,
        key_rope=kv_rope,
        sparse_indices=indices.to(torch.int32),
        scale_value=scaling,
        actual_seq_lengths_query=actual_seq_lengths,
        actual_seq_lengths_kv=actual_seq_lengths,
        block_table=None,
        sparse_block_size=1,
        layout_query="TND",
        layout_kv="TND",
        sparse_mode=3,
        attention_mode=2,
        return_softmax_lse=True,
    )
    lse = torch.log(softmax_sum.float()) + softmax_max.float()
    return output.to(output_dtype), lse.squeeze(0)


def _pytorch_sparse_mla(q, kv, indices, scaling):
    """Differentiable reference DSA attention for short NPU smoke tests."""
    global _SLIME_DSA_DTYPE_TRACE_EMITTED
    output_dtype = q.dtype
    align_npu_bf16 = (
        os.getenv("SLIME_MEGATRON_DSA_BF16", "0") == "1"
        and q.device.type == "npu"
        and q.dtype == torch.float32
    )
    if align_npu_bf16:
        if not _SLIME_DSA_DTYPE_TRACE_EMITTED:
            print(
                "GLM52_MEGATRON_DSA_DTYPE="
                f"q={q.dtype},kv={kv.dtype},compute=torch.bfloat16,output={output_dtype}",
                flush=True,
            )
            _SLIME_DSA_DTYPE_TRACE_EMITTED = True
        q = q.to(torch.bfloat16)
        kv = kv.to(torch.bfloat16)

    seq_len, num_heads, total_dim = q.shape
    kv_groups = kv.shape[1]
    if num_heads % kv_groups != 0:
        raise ValueError(f"num_heads={num_heads} is not divisible by kv_groups={kv_groups}")

    # GLM-5.x stores [KV LoRA rank, RoPE tail] in the latent KV tensor.
    # The native model uses a 64-dimensional RoPE tail.
    value_dim = total_dim - 64
    heads_per_group = num_heads // kv_groups
    outputs = []
    lses = []
    for group in range(kv_groups):
        group_indices = indices[:, group, :]
        valid = group_indices >= 0
        safe_indices = group_indices.clamp(min=0).long()
        selected_kv = kv[:, group, :].index_select(0, safe_indices.reshape(-1))
        selected_kv = selected_kv.view(seq_len, safe_indices.shape[1], total_dim)

        q_group = q[:, group * heads_per_group : (group + 1) * heads_per_group, :]
        scores = torch.einsum(
            "thd,tkd->thk", q_group.float(), selected_kv.float()
        ) * scaling
        scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))
        probs = torch.softmax(scores, dim=-1).nan_to_num(0.0)
        out = torch.einsum(
            "thk,tkd->thd", probs, selected_kv[..., :value_dim].float()
        ).to(q.dtype)
        outputs.append(out)
        lses.append(torch.logsumexp(scores, dim=-1))

    # Preserve the BF16 output rounding boundary of Ascend sparse attention,
    # then restore FP32 for the following VC projection, as SGLang does.
    return torch.cat(outputs, dim=1).to(output_dtype), torch.cat(lses, dim=1)


class _TileSparseMLA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kv, indices, scaling):
        indices = indices.contiguous()
        q, kv = q.contiguous(), kv.contiguous()
        ctx.scaling = scaling
        tl_out, tl_lse = sparse_mla_fwd_interface(q, kv, indices, sm_scale=scaling)
        ctx.save_for_backward(q, kv, indices, tl_out, tl_lse)
        return tl_out, tl_lse

    @staticmethod
    def backward(ctx, grad_output, grad_lse):
        q, kv, indices, tl_out, tl_lse = ctx.saved_tensors
        tl_dq, tl_dkv = sparse_mla_bwd(
            q,
            kv,
            tl_out,
            grad_output.contiguous(),
            indices,
            tl_lse,
            sm_scale=ctx.scaling,
        )
        return tl_dq, tl_dkv, None, None


class SparseMLA:
    @staticmethod
    def apply(q, kv, indices, scaling, cu_seqlens=None):
        if _TILELANG_AVAILABLE and q.device.type == "cuda":
            return _TileSparseMLA.apply(q, kv, indices, scaling)
        if os.getenv("SLIME_MEGATRON_DSA_NPU_KERNEL", "0") == "1" and q.device.type == "npu":
            if cu_seqlens is None:
                raise ValueError("cu_seqlens is required by the NPU sparse-attention kernel")
            return _npu_sparse_mla(q, kv, indices, scaling, cu_seqlens)
        return _pytorch_sparse_mla(q, kv, indices, scaling)

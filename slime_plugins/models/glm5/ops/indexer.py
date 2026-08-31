"""DSA indexer with a TileLang fast path and a portable PyTorch fallback."""

import torch

try:
    from .tilelang_indexer_bwd import indexer_bwd_interface
    from .tilelang_indexer_fwd import indexer_fwd_interface

    _TILELANG_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    indexer_bwd_interface = None
    indexer_fwd_interface = None
    _TILELANG_AVAILABLE = False


def pytorch_extract_topk_scores(logits, topk_indices, dim=-1):
    valid_mask = topk_indices != -1
    safe_indices = topk_indices.clamp(min=0).to(torch.int64)
    scores = torch.gather(logits, dim=dim, index=safe_indices)
    scores = torch.where(valid_mask, scores, float("-inf"))
    return scores


def _pytorch_indexer_logits(index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke):
    """Reference implementation used on NPU and when TileLang is unavailable.

    It intentionally favors portability over speed and is suitable for short
    architecture smoke tests.  The formula mirrors the TileLang kernel:
    sum_h(relu(q_h @ k) * weight_h), followed by the packed causal mask.
    """
    scores_per_head = torch.einsum(
        "qhd,kd->qkh", index_q.float(), index_k.float()
    ).relu_()
    logits = torch.einsum("qkh,qh->qk", scores_per_head, weights.float())

    key_positions = torch.arange(index_k.shape[0], device=index_q.device).unsqueeze(0)
    valid = (key_positions >= cu_seqlen_ks.long().unsqueeze(1)) & (
        key_positions < cu_seqlen_ke.long().unsqueeze(1)
    )
    return logits.masked_fill(~valid, float("-inf"))


class IndexerFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        index_q,
        index_k,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        topk,
        topk_indices=None,
    ):
        _, head_num, _ = index_q.shape
        logits = indexer_fwd_interface(
            index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits=True
        )
        if topk_indices is None:
            effective_topk = min(topk, logits.shape[-1])
            index_score, topk_indices = torch.topk(logits, effective_topk, dim=-1)
            topk_indices = topk_indices.to(torch.int32)
            topk_indices = topk_indices.masked_fill(index_score == -torch.inf, -1)

        index_score = pytorch_extract_topk_scores(logits, topk_indices)
        ctx.save_for_backward(
            index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke, topk_indices
        )
        ctx.topk = topk
        ctx.head_num = head_num
        return index_score, topk_indices

    @staticmethod
    def backward(ctx, grad_scores, grad_indices):
        index_q, index_k, weights, _, _, topk_indices = ctx.saved_tensors
        grad_q, grad_w, grad_k = indexer_bwd_interface(
            index_q, weights, index_k, topk_indices, grad_scores
        )
        return grad_q, grad_k, grad_w, None, None, None, None


def lighting_indexer(
    index_q,
    index_k,
    weights,
    cu_seqlen_ks,
    cu_seqlen_ke,
    topk,
    topk_indices=None,
):
    weights = weights.squeeze(-1)
    if _TILELANG_AVAILABLE and index_q.device.type == "cuda":
        return IndexerFunction.apply(
            index_q,
            index_k,
            weights,
            cu_seqlen_ks,
            cu_seqlen_ke,
            topk,
            topk_indices,
        )

    logits = _pytorch_indexer_logits(
        index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke
    )
    if topk_indices is None:
        effective_topk = min(topk, logits.shape[-1])
        index_score, topk_indices = torch.topk(logits, effective_topk, dim=-1)
        topk_indices = topk_indices.to(torch.int32)
        topk_indices = topk_indices.masked_fill(index_score == -torch.inf, -1)
    else:
        index_score = pytorch_extract_topk_scores(logits, topk_indices)
    return index_score, topk_indices


def generate_varlen_mask_params(cu_seqlens):
    seq_len = cu_seqlens[-1].item()
    q_indices = torch.arange(0, seq_len, device=cu_seqlens.device)
    seq_indices = torch.searchsorted(cu_seqlens, q_indices, right=True) - 1
    starts = cu_seqlens[seq_indices]
    ends = q_indices + 1
    assert torch.all((ends - starts) > 0)
    return starts, ends

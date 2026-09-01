"""Cross-backend prefill hidden-state tracing for Megatron GLM-5.2."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


_MARKER = "GLM52_DFX_PREFILL_HIDDEN="
_HOOKED_MODELS: set[int] = set()
_EMITTED: dict[str, int] = {}


@dataclass
class _TraceContext:
    token_sha256: str
    total_length: int
    response_length: int
    prediction_position: int
    packed_position: int
    response_token: int


_CONTEXT: _TraceContext | None = None


def _enabled() -> bool:
    return os.environ.get("GLM52_DFX_PREFILL") == "1"


def _trace_layer_detail(layer_id: int) -> bool:
    if os.environ.get("GLM52_DFX_LAYER_DETAIL") != "1":
        return False
    selected = os.environ.get("GLM52_DFX_LAYER_DETAIL_LAYERS", "0").strip().lower()
    if selected in {"all", "*"}:
        return True
    for item in selected.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            if int(start) <= layer_id <= int(end):
                return True
        elif int(item) == layer_id:
            return True
    return False


def _trace_attention_detail(layer_id: int) -> bool:
    if not _trace_layer_detail(layer_id):
        return False
    if os.environ.get("GLM52_DFX_ATTN_DETAIL") != "1":
        return False
    selected = os.environ.get(
        "GLM52_DFX_ATTN_DETAIL_LAYERS",
        os.environ.get("GLM52_DFX_LAYER_DETAIL_LAYERS", "0"),
    ).strip().lower()
    if selected in {"all", "*"}:
        return True
    for item in selected.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            if int(start) <= layer_id <= int(end):
                return True
        elif int(item) == layer_id:
            return True
    return False


def set_megatron_prefill_context(
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
) -> None:
    """Select one packed sequence and expose its position to module hooks."""

    global _CONTEXT
    _CONTEXT = None
    if not _enabled():
        return

    expected_response_length = max(
        1, int(os.environ.get("GLM52_DFX_PREFILL_RESPONSE_LENGTH", "1"))
    )
    packed_offset = 0
    for tokens, total_length, response_length in zip(
        unconcat_tokens, total_lengths, response_lengths, strict=False
    ):
        total_length = int(total_length)
        response_length = int(response_length)
        if response_length == expected_response_length and total_length > response_length:
            sequence = tokens[:total_length].detach().to(device="cpu", dtype=torch.int64)
            sequence_bytes = np.asarray(sequence.numpy(), dtype=np.int64).tobytes()
            prediction_position = total_length - response_length - 1
            _CONTEXT = _TraceContext(
                token_sha256=hashlib.sha256(sequence_bytes).hexdigest(),
                total_length=total_length,
                response_length=response_length,
                prediction_position=prediction_position,
                packed_position=packed_offset + prediction_position,
                response_token=int(sequence[-response_length].item()),
            )
            return
        packed_offset += total_length


def clear_megatron_prefill_context() -> None:
    global _CONTEXT
    _CONTEXT = None


def _first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _boundary(module_name: str) -> str | None:
    if module_name == "embedding":
        return "embedding"
    if module_name == "decoder.final_layernorm":
        return "final_norm"
    match = re.fullmatch(r"decoder\.layers\.(\d+)", module_name)
    if match:
        return f"layer.{match.group(1)}"

    detail = re.fullmatch(
        r"decoder\.layers\.(\d+)\."
        r"(input_layernorm|pre_cross_attn_layernorm)",
        module_name,
    )
    if not detail:
        return None
    layer_id = int(detail.group(1))
    if not _trace_layer_detail(layer_id):
        return None
    suffix = {
        "input_layernorm": "input_norm",
        "pre_cross_attn_layernorm": "attention_residual",
    }[detail.group(2)]
    return f"layer.{layer_id}.{suffix}"


@torch.no_grad()
def _emit_hidden(
    boundary: str,
    output: Any,
    *,
    sequence_parallel: bool,
    trace_kind: str = "forward_output",
) -> None:
    context = _CONTEXT
    if not _enabled() or context is None:
        return
    max_calls = max(1, int(os.environ.get("GLM52_DFX_PREFILL_MAX_CALLS", "4")))
    if _EMITTED.get(boundary, 0) >= max_calls:
        return

    tensor = _first_tensor(output)
    if tensor is None or tensor.ndim < 2:
        return

    from megatron.core import parallel_state, tensor_parallel

    # Sequence parallel shards the token dimension over TP ranks.  Every rank
    # must enter the collective, then only TP rank zero serializes the vector.
    gathered = tensor.detach()
    if sequence_parallel:
        gathered = tensor_parallel.gather_from_sequence_parallel_region(
            gathered, tensor_parallel_output_grad=False
        )
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    if tp_rank != 0:
        return

    position = context.packed_position
    if gathered.ndim == 2:
        if position >= gathered.shape[0]:
            return
        vector = gathered[position]
    elif position < gathered.shape[0]:
        vector = gathered[position, 0]
    elif position < gathered.shape[1]:
        vector = gathered[0, position]
    else:
        return

    vector = vector.detach().reshape(-1).float().cpu().contiguous()
    vector_bytes = vector.numpy().tobytes()
    event = {
        "backend": "megatron",
        "boundary": boundary,
        "token_sha256": context.token_sha256,
        "total_length": context.total_length,
        "response_length": context.response_length,
        "prediction_position": context.prediction_position,
        "packed_position": context.packed_position,
        "response_token": context.response_token,
        "hidden_size": vector.numel(),
        "vector_dtype": "float32",
        "vector_sha256": hashlib.sha256(vector_bytes).hexdigest(),
        "vector_b64": base64.b64encode(vector_bytes).decode("ascii"),
        "finite": bool(torch.isfinite(vector).all().item()),
        "abs_max": float(torch.nan_to_num(vector).abs().max().item()),
        "rank": parallel_state.get_data_parallel_rank(with_context_parallel=True),
        "tp_rank": tp_rank,
        "tp_size": parallel_state.get_tensor_model_parallel_world_size(),
        "trace_kind": trace_kind,
    }
    _EMITTED[boundary] = _EMITTED.get(boundary, 0) + 1
    print(_MARKER + json.dumps(event, separators=(",", ":")), flush=True)


def _dense_te_post_attention_norm(layer: torch.nn.Module, hidden: torch.Tensor):
    """Reconstruct the RMSNorm fused into a dense TE MLP's linear_fc1.

    Dense Transformer-Engine layers expose ``pre_mlp_layernorm`` as IdentityOp;
    the real RMSNorm is fused into ``mlp.linear_fc1`` and has no hookable output.
    Reconstructing it from the actual residual and affine weight gives SGLang a
    semantically equivalent comparison point without changing either forward.
    """

    linear_fc1 = getattr(getattr(layer, "mlp", None), "linear_fc1", None)
    weight = getattr(linear_fc1, "layer_norm_weight", None)
    if weight is None:
        return None
    eps = float(getattr(layer.config, "layernorm_epsilon", 1.0e-5))
    scale = weight.detach().float()
    if bool(getattr(layer.config, "layernorm_zero_centered_gamma", False)):
        scale = scale + 1.0
    source = hidden.detach().float()
    variance = source.square().mean(dim=-1, keepdim=True)
    return (source * torch.rsqrt(variance + eps) * scale).to(hidden.dtype)


def install_megatron_prefill_hooks(model: torch.nn.Module) -> None:
    """Install boundary hooks without changing the model's numerical output."""

    if not _enabled() or id(model) in _HOOKED_MODELS:
        return
    _HOOKED_MODELS.add(id(model))
    sequence_parallel = bool(getattr(model.config, "sequence_parallel", False))
    detailed_layers: list[tuple[int, torch.nn.Module]] = []

    for module_name, module in model.named_modules():
        layer_match = re.fullmatch(r"decoder\.layers\.(\d+)", module_name)
        if layer_match and _trace_layer_detail(int(layer_match.group(1))):
            detailed_layers.append((int(layer_match.group(1)), module))
        boundary = _boundary(module_name)
        if boundary is None:
            continue

        def hook(_module, _inputs, output, *, trace_boundary=boundary):
            _emit_hidden(
                trace_boundary, output, sequence_parallel=sequence_parallel
            )

        module.register_forward_hook(hook)

    for layer_id, layer in detailed_layers:
        if _trace_attention_detail(layer_id):
            self_attention = getattr(layer, "self_attention", None)
            if self_attention is not None:

                def trace_attention(
                    suffix,
                    tensor,
                    *,
                    tensor_sequence_parallel=True,
                    trace_kind="forward_output",
                    trace_layer_id=layer_id,
                ):
                    _emit_hidden(
                        f"layer.{trace_layer_id}.attn.{suffix}",
                        tensor,
                        sequence_parallel=tensor_sequence_parallel,
                        trace_kind=trace_kind,
                    )

                self_attention._glm52_dfx_trace_attention = trace_attention

        pre_mlp_layernorm = getattr(layer, "pre_mlp_layernorm", None)
        if pre_mlp_layernorm is not None and not (
            pre_mlp_layernorm.__class__.__name__ == "IdentityOp"
        ):

            def norm_hook(_module, _inputs, output, *, trace_layer_id=layer_id):
                _emit_hidden(
                    f"layer.{trace_layer_id}.post_attention_norm",
                    output,
                    sequence_parallel=sequence_parallel,
                )

            pre_mlp_layernorm.register_forward_hook(norm_hook)
            continue

        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue

        def dense_mlp_pre_hook(
            _module,
            inputs,
            *,
            trace_layer_id=layer_id,
            trace_layer=layer,
        ):
            hidden = _first_tensor(inputs)
            if hidden is None:
                return
            normalized = _dense_te_post_attention_norm(trace_layer, hidden)
            if normalized is None:
                return
            _emit_hidden(
                f"layer.{trace_layer_id}.post_attention_norm",
                normalized,
                sequence_parallel=sequence_parallel,
                trace_kind="derived_dense_te_fused_rmsnorm",
            )

        mlp.register_forward_pre_hook(dense_mlp_pre_hook)

"""Opt-in forward-boundary diagnostics for GLM-5.2 train/infer consistency.

The diagnostics are intentionally environment-gated because full-tensor
finite checks synchronize the accelerator.  They are meant for one-token DFX
runs, not normal training.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from typing import Any

import torch
import torch.distributed as dist


_MARKER = "GLM52_DFX_PHASE2="
_EMITTED: defaultdict[str, int] = defaultdict(int)
_HOOKED_MODELS: set[int] = set()
_EVENT_INDEX = 0


def _enabled() -> bool:
    return os.environ.get("GLM52_DFX_PHASE2") == "1"


def _rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", "0"))


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


@torch.no_grad()
def _tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    detached = tensor.detach()
    flat = detached.reshape(-1)
    numel = flat.numel()
    chunk_elements = max(
        1, int(os.environ.get("GLM52_DFX_PHASE2_CHUNK_ELEMENTS", "4194304"))
    )
    nan_count = 0
    posinf_count = 0
    neginf_count = 0
    finite_abs_max = 0.0

    if detached.is_floating_point() or detached.is_complex():
        for start in range(0, numel, chunk_elements):
            part = flat[start : start + chunk_elements]
            nan_count += int(torch.isnan(part).sum().item())
            posinf_count += int(torch.isposinf(part).sum().item())
            neginf_count += int(torch.isneginf(part).sum().item())
            clean = torch.nan_to_num(
                part.float(), nan=0.0, posinf=0.0, neginf=0.0
            )
            if clean.numel():
                finite_abs_max = max(
                    finite_abs_max, float(clean.abs().max().item())
                )

    sample_count = min(8, numel)
    if sample_count:
        if sample_count == 1:
            positions = [0]
        else:
            positions = [
                round(index * (numel - 1) / (sample_count - 1))
                for index in range(sample_count)
            ]
        sample_values = flat[positions].float().cpu().tolist()
    else:
        positions = []
        sample_values = []

    return {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "numel": numel,
        "nan_count": nan_count,
        "posinf_count": posinf_count,
        "neginf_count": neginf_count,
        "non_finite_count": nan_count + posinf_count + neginf_count,
        "finite_abs_max": finite_abs_max,
        "sample_positions": positions,
        "sample_values": sample_values,
    }


def emit_phase2_tensor(
    name: str,
    value: Any,
    *,
    stage: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Emit one or more bounded tensor summaries for a forward boundary."""

    if not _enabled():
        return
    max_calls = max(1, int(os.environ.get("GLM52_DFX_PHASE2_MAX_CALLS", "1")))
    key = f"{stage}:{name}"
    if _EMITTED[key] >= max_calls:
        return
    tensor = _first_tensor(value)
    if tensor is None:
        return

    global _EVENT_INDEX
    _EMITTED[key] += 1
    _EVENT_INDEX += 1
    event = {
        "stage": stage,
        "name": name,
        "event_index": _EVENT_INDEX,
        "call_index": _EMITTED[key],
        "rank": _rank(),
        **_tensor_stats(tensor),
    }
    if metadata:
        event.update(metadata)
    print(_MARKER + json.dumps(event, separators=(",", ":")), flush=True)


def _trace_module(name: str) -> bool:
    if name in {"embedding", "decoder.final_layernorm", "output_layer"}:
        return True
    return re.fullmatch(
        r"decoder\.layers\.\d+(?:\.(?:self_attention|mlp))?", name
    ) is not None


def install_phase2_forward_hooks(model: torch.nn.Module) -> None:
    """Attach one-shot hooks to the useful Megatron forward boundaries."""

    if not _enabled() or id(model) in _HOOKED_MODELS:
        return
    _HOOKED_MODELS.add(id(model))

    selected: list[str] = []
    for name, module in model.named_modules():
        if not name or not _trace_module(name):
            continue
        selected.append(name)

        def hook(_module, _inputs, output, *, module_name=name):
            emit_phase2_tensor(
                module_name,
                output,
                stage="module_output",
                metadata={"module_type": type(_module).__name__},
            )

        module.register_forward_hook(hook)

    print(
        _MARKER
        + json.dumps(
            {
                "stage": "hook_manifest",
                "name": "megatron_model",
                "rank": _rank(),
                "modules": selected,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )

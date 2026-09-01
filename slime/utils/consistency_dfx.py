"""Opt-in tensor fingerprints for GLM train/inference consistency DFX."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from typing import Any

import torch

_EVENT_LOCK = threading.Lock()
_EVENT_COUNT = 0


def _enabled() -> bool:
    return os.environ.get("GLM52_DFX_ENABLE", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _selected(name: str) -> bool:
    pattern = os.environ.get("GLM52_DFX_WEIGHT_PATTERN", ".*")
    try:
        return re.search(pattern, name) is not None
    except re.error as exc:
        print(
            "GLM52_DFX_ERROR="
            + json.dumps(
                {"stage": "weight_filter", "pattern": pattern, "error": str(exc)},
                separators=(",", ":"),
            ),
            flush=True,
        )
        return False


def _reserve_event() -> int | None:
    global _EVENT_COUNT
    try:
        limit = max(0, int(os.environ.get("GLM52_DFX_MAX_WEIGHT_EVENTS", "4096")))
    except ValueError:
        limit = 4096
    with _EVENT_LOCK:
        if _EVENT_COUNT >= limit:
            return None
        event_index = _EVENT_COUNT
        _EVENT_COUNT += 1
        return event_index


def _tensor_fingerprint(tensor: torch.Tensor) -> dict[str, Any]:
    detached = tensor.detach()
    flat = detached.reshape(-1)
    numel = flat.numel()
    try:
        requested = max(2, int(os.environ.get("GLM52_DFX_SAMPLE_SIZE", "32")))
    except ValueError:
        requested = 32
    sample_count = min(requested, numel)

    if sample_count:
        if sample_count == 1:
            positions = [0]
        else:
            positions = [i * (numel - 1) // (sample_count - 1) for i in range(sample_count)]
        index = torch.tensor(positions, dtype=torch.long, device=flat.device)
        sample = flat.index_select(0, index).to(dtype=torch.float32, device="cpu")
        sample_values = sample.tolist()
        sample_bytes = sample.contiguous().numpy().tobytes()
        sample_sum = float(sample.sum().item())
        sample_abs_sum = float(sample.abs().sum().item())
        sample_abs_max = float(sample.abs().max().item())
        finite = bool(torch.isfinite(sample).all().item())
    else:
        positions = []
        sample_values = []
        sample_bytes = b""
        sample_sum = 0.0
        sample_abs_sum = 0.0
        sample_abs_max = 0.0
        finite = True

    return {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "numel": numel,
        "contiguous": detached.is_contiguous(),
        "sample_positions": positions,
        "sample_values": sample_values,
        "sample_sha256": hashlib.sha256(sample_bytes).hexdigest(),
        "sample_sum": sample_sum,
        "sample_abs_sum": sample_abs_sum,
        "sample_abs_max": sample_abs_max,
        "sample_finite": finite,
    }


def emit_weight_tensor(
    *, stage: str, name: str, tensor: torch.Tensor, **context: Any
) -> None:
    if not _enabled() or not _selected(name):
        return
    event_index = _reserve_event()
    if event_index is None:
        return
    try:
        payload = {
            "stage": stage,
            "name": name,
            "event_index": event_index,
            "pid": os.getpid(),
            **context,
            **_tensor_fingerprint(tensor),
        }
        print(
            "GLM52_DFX_WEIGHT=" + json.dumps(payload, separators=(",", ":"), default=str),
            flush=True,
        )
    except Exception as exc:  # DFX must never break training or synchronization.
        print(
            "GLM52_DFX_ERROR="
            + json.dumps(
                {"stage": stage, "name": name, "error": repr(exc)},
                separators=(",", ":"),
            ),
            flush=True,
        )

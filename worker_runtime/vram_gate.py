"""
vram_gate.py — Python-side capability gate, mirrors ghost-agent-go-repo's
gpu_capability.go determineSupportedTaskTypes() logic exactly, but reads
thresholds from model_dispatch.json (hot-reload) instead of Go constants,
so a threshold change in the JSON applies everywhere without a Go agent
rebuild+redeploy. The Go agent's own VRAM-gate stays in place for its
FAST, no-Python-startup capability declaration at registration time --
this module is the authoritative gate actually enforced at task-EXECUTION
time on any provider (Colab, local Python sidecar, RunPod, Vast.ai).
"""
from __future__ import annotations
import json
import os
import time
from typing import Dict, List, Optional

_DISPATCH_PATH = os.environ.get(
    "MODEL_DISPATCH_PATH",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "model_dispatch.json"),
)
_CACHE = {"mtime": 0.0, "data": None}


def _load_dispatch() -> dict:
    try:
        mtime = os.path.getmtime(_DISPATCH_PATH)
        if mtime != _CACHE["mtime"] or _CACHE["data"] is None:
            with open(_DISPATCH_PATH, "r", encoding="utf-8") as f:
                _CACHE["data"] = json.load(f)
            _CACHE["mtime"] = mtime
    except Exception as e:
        print(f"[worker_runtime][vram_gate] reload failed, using cached copy: {e}")
    return _CACHE["data"] or {"task_types": {}, "default_min_vram_gb": 4.0}


def resolve_model(task_type: str, model_name: Optional[str] = None) -> Dict:
    """Returns the model config dict {min_vram_gb, loader, repo, ...} for
    a given task_type/model_name, falling back to the task_type's
    default_model, then to a safe generic minimum -- never raises, never
    silently picks an arbitrary unrelated model."""
    dispatch = _load_dispatch()
    tt_cfg = dispatch.get("task_types", {}).get(task_type)
    if not tt_cfg:
        return {"min_vram_gb": dispatch.get("default_min_vram_gb", 4.0), "loader": None, "repo": None}
    models = tt_cfg.get("models", {})
    name = model_name or tt_cfg.get("default_model")
    entry = models.get(name)
    if entry is None:
        # Unknown model name for this task type -- fall back to the
        # task type's own default rather than an unrelated task type's
        # model, and fail toward a conservative VRAM assumption.
        default_name = tt_cfg.get("default_model")
        entry = models.get(default_name, {"min_vram_gb": dispatch.get("default_min_vram_gb", 4.0)})
        name = default_name
    return {"model_name": name, **entry}


def determine_supported_task_types(vram_total_gb: float) -> List[str]:
    """Same semantics as Go's determineSupportedTaskTypes(): evaluates
    EVERY task type, declares support only if vram_total_gb meets that
    task type's default model's min_vram_gb. Fails closed (empty list)
    for vram_total_gb <= 0."""
    if vram_total_gb <= 0:
        return []
    dispatch = _load_dispatch()
    supported = []
    for task_type, cfg in dispatch.get("task_types", {}).items():
        default_name = cfg.get("default_model")
        entry = cfg.get("models", {}).get(default_name, {})
        min_vram = entry.get("min_vram_gb", dispatch.get("default_min_vram_gb", 4.0))
        if vram_total_gb >= min_vram:
            supported.append(task_type)
    return supported

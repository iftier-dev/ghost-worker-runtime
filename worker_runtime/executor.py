"""
executor.py — Universal task executor, shared by EVERY provider (Colab,
local Python sidecar, RunPod, Vast.ai). One entrypoint, one place to fix
bugs or add task types -- no provider duplicates generation logic.

Contract (matches micro-vm's colab_tasks.py _dispatch_sync exactly):
  execute(task_type: str, payload: dict) -> dict
  Input payload has NO "code" key (that's the separate raw-code-exec path,
  handled elsewhere) -- this is always a structured task.
  Output shape: {"success": bool, "output_url": str, ...} or
                {"success": False, "error": str}
  matches what red-gpu's gpu_manager.py _verify_result_wellformed() and
  batch_coordinator.py finalize_batch() already expect (output_url for
  image/video/tts task types).
"""
from __future__ import annotations
import logging
import os
import time
import traceback
from typing import Any, Dict

from .vram_gate import resolve_model
from .model_cache import ensure_cache_dir, enforce_cache_cap, model_cache_path
from .pipeline_cache import get_loader_lock

logger = logging.getLogger("worker_runtime.executor")

# Lazy import registry -- each loader module is only imported (and its
# heavy dependencies like torch/diffusers only loaded) the first time a
# task of that type actually runs on THIS machine. A Colab node that only
# ever gets tts tasks never pays the torch/diffusers import cost.
_LOADER_MODULES = {
    "llm_ollama":        "worker_runtime.loaders.llm",
    "embedding_hf":       "worker_runtime.loaders.llm",
    "image_diffusers":    "worker_runtime.loaders.image",
    "image_upscale":      "worker_runtime.loaders.image",
    "video_diffusers":    "worker_runtime.loaders.video",
    "video_upscale":      "worker_runtime.loaders.video",
    "tts_coqui":          "worker_runtime.loaders.tts",
    "tts_coqui_clone":    "worker_runtime.loaders.tts",
    "tts_edge":           "worker_runtime.loaders.tts",
    "stt_whisper":        "worker_runtime.loaders.transcribe",
    "vision_qwen":        "worker_runtime.loaders.vision",
    "vision_llava":       "worker_runtime.loaders.vision",
    "vision_grounding":   "worker_runtime.loaders.vision",
}

# Which function inside each loader module handles which task_type --
# every loader module exports run(task_type, payload, model_cfg) as its
# single public entrypoint, dispatching internally by task_type.
_TASK_TYPE_TO_LOADER_KEY = {
    "llm_inference":    "llm_ollama",
    "embedding":         "embedding_hf",
    "image_generation":  "image_diffusers",
    "image_to_image":    "image_diffusers",
    "inpainting":        "image_diffusers",
    "image_upscale":     "image_upscale",
    "video_generation":  "video_diffusers",
    "image_to_video":    "video_diffusers",
    "video_upscale":     "video_upscale",
    "tts":               "tts_coqui",
    "voice_clone":       "tts_coqui_clone",
    "speech_to_text":    "stt_whisper",
    "vision_analyze":    "vision_qwen",
    "visual_click_grounding": "vision_grounding",
}

_loader_cache: Dict[str, Any] = {}


def _get_loader(loader_key: str):
    """Import-once, cache-forever per process -- avoids re-importing
    heavy ML libraries on every single task."""
    if loader_key in _loader_cache:
        return _loader_cache[loader_key]
    module_path = _LOADER_MODULES.get(loader_key)
    if not module_path:
        raise ValueError(f"No loader module registered for loader_key={loader_key}")
    import importlib
    mod = importlib.import_module(module_path)
    _loader_cache[loader_key] = mod
    return mod


def execute(task_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Synchronous entrypoint -- safe to call from a plain FastAPI
    endpoint (Colab worker, RunPod handler) without an event loop. Each
    loader manages its own async/sync internals as needed."""
    t0 = time.time()
    try:
        if task_type not in _TASK_TYPE_TO_LOADER_KEY:
            return {
                "success": False,
                "error": f"Unknown or unsupported task_type: {task_type!r}. "
                         f"Supported: {sorted(_TASK_TYPE_TO_LOADER_KEY.keys())}",
            }

        model_name = payload.get("model_name")
        model_cfg = resolve_model(task_type, model_name)
        loader_key = model_cfg.get("loader") or _TASK_TYPE_TO_LOADER_KEY[task_type]

        loader_mod = _get_loader(loader_key)

        # Best-effort cache housekeeping before every task -- cheap
        # (directory walk only), keeps the cache bounded continuously
        # rather than only at startup. vram_free_gb from payload if the
        # caller measured it (Go agent/Colab probe), else a conservative
        # default so the cap still applies somewhere sane.
        try:
            enforce_cache_cap(payload.get("vram_free_gb", 8.0))
        except Exception as e:
            logger.warning("[executor] cache cap enforcement failed (non-fatal): %s", e)

        # Exclusive per-loader-family lock -- a single GPU cannot safely
        # run two heavy diffusion/video jobs at once (VRAM OOM risk), so
        # concurrent requests to the SAME loader family on THIS process
        # serialize here rather than racing into the same VRAM pool.
        # Lightweight loaders (embedding_hf) skip this -- they truly can
        # run concurrently without VRAM contention.
        _EXCLUSIVE_LOADERS = {
            "image_diffusers", "image_upscale", "video_diffusers",
            "video_upscale", "tts_coqui", "tts_coqui_clone", "stt_whisper",
            "llm_ollama", "vision_qwen", "vision_llava", "vision_grounding",
        }
        if loader_key in _EXCLUSIVE_LOADERS:
            lock = get_loader_lock(loader_key)
            with lock:
                result = loader_mod.run(task_type=task_type, payload=payload, model_cfg=model_cfg)
        else:
            result = loader_mod.run(task_type=task_type, payload=payload, model_cfg=model_cfg)

        if not isinstance(result, dict):
            return {"success": False, "error": "loader returned a non-dict result"}
        result.setdefault("success", True)
        result["execution_time_seconds"] = round(time.time() - t0, 3)
        result["model_used"] = model_cfg.get("model_name")
        return result

    except Exception as e:
        logger.error("[executor] task_type=%s failed: %s\n%s", task_type, e, traceback.format_exc())
        return {
            "success": False,
            "error": f"{type(e).__name__}: {e}",
            "task_type": task_type,
            "execution_time_seconds": round(time.time() - t0, 3),
        }

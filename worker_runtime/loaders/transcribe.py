"""
transcribe.py — speech_to_text via faster-whisper, GPU-resident model
cache like every other loader.
"""
from __future__ import annotations
import os
import uuid
from typing import Any, Dict

from ..pipeline_cache import get_or_load
from ..model_cache import model_cache_path

OUTPUT_DIR = os.environ.get("WORKER_OUTPUT_DIR", "/tmp/ghost-worker-output")


def _load_whisper(repo: str):
    from faster_whisper import WhisperModel
    # faster-whisper uses short model-size names ("large-v3") rather than
    # full HF repo ids -- normalize by taking the last path segment if a
    # full repo id was configured (e.g. "openai/whisper-large-v3" -> "large-v3").
    size = repo.split("/")[-1].replace("whisper-", "") if repo else "large-v3"
    device = "cuda"
    try:
        import torch
        if not torch.cuda.is_available():
            device = "cpu"
    except Exception:
        device = "cpu"
    return WhisperModel(size, device=device, compute_type="float16" if device == "cuda" else "int8",
                         download_root=model_cache_path(repo or "whisper"))


def _run_speech_to_text(payload: Dict[str, Any], model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    audio_url = payload.get("audio_url")
    if not audio_url:
        return {"success": False, "error": "audio_url is required for speech_to_text"}
    repo = model_cfg.get("repo")
    model = get_or_load(
        cache_key=f"stt:{repo}",
        loader_fn=lambda: _load_whisper(repo),
        required_vram_gb=model_cfg.get("min_vram_gb", 6.0),
    )
    import httpx
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    audio_path = os.path.join(OUTPUT_DIR, f"stt_in_{uuid.uuid4().hex}.audio")
    with httpx.Client(timeout=120) as client:
        resp = client.get(audio_url)
        resp.raise_for_status()
    with open(audio_path, "wb") as f:
        f.write(resp.content)

    try:
        segments, info = model.transcribe(audio_path, language=payload.get("language"))
        full_text = " ".join(seg.text.strip() for seg in segments)
    finally:
        try:
            os.remove(audio_path)
        except Exception:
            pass

    return {"output_text": full_text, "detected_language": getattr(info, "language", None)}


def run(task_type: str, payload: Dict[str, Any], model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    if task_type == "speech_to_text":
        return _run_speech_to_text(payload, model_cfg)
    return {"success": False, "error": f"transcribe.py loader does not handle task_type={task_type!r}"}

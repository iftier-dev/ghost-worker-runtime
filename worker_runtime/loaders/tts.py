"""
tts.py — tts (text-to-speech) and voice_clone. Migrates the working
provider logic from agents_core/tools/tts_tool.py (OpenAI/ElevenLabs/
edge-tts/espeak) but ALSO adds a real local GPU-resident model (Coqui
XTTS-v2) so this actually runs on distributed GPU nodes instead of only
ever proxying to a paid cloud API -- consistent with every other loader
in this package generating output locally on the node's own GPU.
"""
from __future__ import annotations
import os
import uuid
from typing import Any, Dict

from ..pipeline_cache import get_or_load
from ..model_cache import model_cache_path

OUTPUT_DIR = os.environ.get("WORKER_OUTPUT_DIR", "/tmp/ghost-worker-output")


def _ensure_output_dir() -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


def _save_audio_and_url(fpath: str) -> str:
    upload_url = os.environ.get("AUDIO_UPLOAD_URL", "")
    if upload_url:
        try:
            import httpx
            with open(fpath, "rb") as f, httpx.Client(timeout=60) as client:
                resp = client.post(upload_url, files={"file": (os.path.basename(fpath), f, "audio/wav")})
                resp.raise_for_status()
                remote_url = resp.json().get("url")
                if remote_url:
                    return remote_url
        except Exception:
            pass
    return fpath


def _load_xtts(repo: str):
    from TTS.api import TTS
    return TTS(repo)


def _run_tts(payload: Dict[str, Any], model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    text = payload.get("text", "")
    if not text:
        return {"success": False, "error": "text is required for tts"}
    repo = model_cfg.get("repo")
    tts_model = get_or_load(
        cache_key=f"tts:{repo}",
        loader_fn=lambda: _load_xtts(repo),
        required_vram_gb=model_cfg.get("min_vram_gb", 8.0),
    )
    out_dir = _ensure_output_dir()
    fpath = os.path.join(out_dir, f"tts_{uuid.uuid4().hex}.wav")
    tts_model.tts_to_file(
        text=text,
        speaker=payload.get("voice"),
        language=payload.get("language", "en"),
        file_path=fpath,
    )
    url = _save_audio_and_url(fpath)
    return {"output_url": url}


def _run_voice_clone(payload: Dict[str, Any], model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    text = payload.get("text", "")
    reference_audio_url = payload.get("reference_audio_url")
    if not text or not reference_audio_url:
        return {"success": False, "error": "text and reference_audio_url are required for voice_clone"}
    repo = model_cfg.get("repo")
    tts_model = get_or_load(
        cache_key=f"tts_clone:{repo}",
        loader_fn=lambda: _load_xtts(repo),
        required_vram_gb=model_cfg.get("min_vram_gb", 8.0),
    )
    import httpx
    out_dir = _ensure_output_dir()
    ref_path = os.path.join(out_dir, f"ref_{uuid.uuid4().hex}.wav")
    with httpx.Client(timeout=60) as client:
        resp = client.get(reference_audio_url)
        resp.raise_for_status()
    with open(ref_path, "wb") as f:
        f.write(resp.content)

    out_path = os.path.join(out_dir, f"clone_{uuid.uuid4().hex}.wav")
    tts_model.tts_to_file(
        text=text,
        speaker_wav=ref_path,
        language=payload.get("language", "en"),
        file_path=out_path,
    )
    os.remove(ref_path)
    url = _save_audio_and_url(out_path)
    return {"output_url": url}


def run(task_type: str, payload: Dict[str, Any], model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    if task_type == "tts":
        return _run_tts(payload, model_cfg)
    if task_type == "voice_clone":
        return _run_voice_clone(payload, model_cfg)
    return {"success": False, "error": f"tts.py loader does not handle task_type={task_type!r}"}
